#include "cpu_pricer.h"
#include "black_scholes.h"
#ifdef RISK_ENGINE_HAS_CUDA
#include "gpu_pricer.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr int repetitions = 7;

struct Measurement {
    MonteCarloResult result;
    double totalMs;
#ifdef RISK_ENGINE_HAS_CUDA
    GpuTimings stages{};
#endif
};

Measurement measureCPU(const OptionParams& params, long long n) {
    const auto start = Clock::now();
    const auto result = priceEuropeanCallCPU(params, n, 42);
    return {result, std::chrono::duration<double, std::milli>(Clock::now() - start).count()};
}
#ifdef RISK_ENGINE_HAS_CUDA
Measurement measureGPU(const OptionParams& params, long long n) {
    GpuTimings stages{};
    const auto start = Clock::now();
    const auto result = priceEuropeanCallGPU(params, n, 42, &stages);
    const double elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return {result, elapsed, stages};
}
#endif

void writeRow(const char* backend, long long n, int repetition,
              const Measurement& measurement, double analytical) {
    const auto& result = measurement.result;
    if (!std::isfinite(result.price) || !std::isfinite(result.standardError) || result.standardError < 0) {
        throw std::runtime_error("Non-finite price or invalid standard error");
    }
    std::cout << backend << ',' << n << ',' << repetition << ",42,"
              << result.price << ',' << result.standardError << ',' << analytical << ','
              << std::abs(result.price - analytical) << ',' << measurement.totalMs;
#ifdef RISK_ENGINE_HAS_CUDA
    if (backend[0] == 'g') {
        std::cout << ',' << measurement.stages.kernelMs << ',' << measurement.stages.reductionMs
                  << ',' << measurement.stages.transferMs << ',' << measurement.stages.aggregationMs;
    } else
#endif
    {
        std::cout << ",,,,";
    }
    std::cout << '\n';
}

double summarize(const char* backend, std::vector<double> times) {
    std::sort(times.begin(), times.end());
    const double median = times[times.size() / 2];
    std::cerr << "  " << backend << " median " << median << " ms; range ["
              << times.front() << ", " << times.back() << "] ms\n";
    return median;
}
} // namespace

int main() {
    try {
        const OptionParams params{100, 100, .05, .2, 1};
        const double analytical = priceEuropeanCallBlackScholes(params);
        std::cout << std::setprecision(12)
                  << "backend,paths,repetition,seed,price,standard_error,black_scholes,absolute_error,total_ms,kernel_ms,reduction_ms,transfer_ms,merge_ms\n";
        std::cerr << std::fixed << std::setprecision(3)
                  << "7 measured repetitions per size; one full warm-up per backend per size.\n"
                  << "Fixed seed 42 measures timing variability, not independent pricing trials.\n";
        for (long long n : {10'000LL, 100'000LL, 1'000'000LL, 10'000'000LL}) {
            // Warm-up occurs in this process and is excluded from reported data.
            measureCPU(params, n);
#ifdef RISK_ENGINE_HAS_CUDA
            measureGPU(params, n);
            std::vector<double> gpuTimes;
#endif
            std::vector<double> cpuTimes;
            for (int repetition = 1; repetition <= repetitions; ++repetition) {
                Measurement cpu{};
#ifdef RISK_ENGINE_HAS_CUDA
                Measurement gpu{};
                // Alternate order to reduce systematic CPU-first/GPU-first bias.
                if (repetition % 2) {
                    cpu = measureCPU(params, n);
                    gpu = measureGPU(params, n);
                } else {
                    gpu = measureGPU(params, n);
                    cpu = measureCPU(params, n);
                }
#else
                cpu = measureCPU(params, n);
#endif
                // Console/file writes happen outside the measured calls.
                writeRow("cpu", n, repetition, cpu, analytical);
                cpuTimes.push_back(cpu.totalMs);
#ifdef RISK_ENGINE_HAS_CUDA
                writeRow("gpu", n, repetition, gpu, analytical);
                gpuTimes.push_back(gpu.totalMs);
#endif
            }
            std::cerr << "Paths: " << n << '\n';
            const double cpuMedian = summarize("CPU", cpuTimes);
#ifdef RISK_ENGINE_HAS_CUDA
            const double gpuMedian = summarize("GPU", gpuTimes);
            std::cerr << "  Speedup (CPU median / GPU median): " << cpuMedian / gpuMedian << "x\n";
#else
            (void)cpuMedian;
#endif
        }
        if (!std::cout) throw std::runtime_error("Failed to write benchmark CSV");
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
