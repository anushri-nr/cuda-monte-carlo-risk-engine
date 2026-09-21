#include "cpu_pricer.h"
#include "black_scholes.h"
#ifdef RISK_ENGINE_HAS_CUDA
#include "gpu_pricer.h"
#endif

#include <algorithm>
#include <array>
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
Measurement measureGPU(const OptionParams& params, long long n, GpuAggregation aggregation) {
    GpuTimings stages{};
    const auto start = Clock::now();
    const auto result = priceEuropeanCallGPU(params, n, 42, &stages, aggregation);
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
        std::cout << ',' << measurement.stages.kernelMs
                  << ',' << measurement.stages.transferMs << ',' << measurement.stages.aggregationMs;
    } else
#endif
    {
        std::cout << ",,,";
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
                  << "backend,paths,repetition,seed,price,standard_error,black_scholes,absolute_error,total_ms,kernel_ms,transfer_ms,aggregation_ms\n";
        std::cerr << std::fixed << std::setprecision(3)
                  << "7 measured repetitions per size; one full warm-up per backend per size.\n"
                  << "Fixed seed 42 measures timing variability, not independent pricing trials.\n";
        for (long long n : {10'000LL, 100'000LL, 1'000'000LL, 10'000'000LL}) {
#ifdef RISK_ENGINE_HAS_CUDA
            constexpr int backendCount = 3;
            const std::array<const char*, backendCount> names{
                "cpu", "gpu_no_reduction", "gpu_reduction"};
#else
            constexpr int backendCount = 1;
            const std::array<const char*, backendCount> names{"cpu"};
#endif
            const auto measure = [&](int backend) {
#ifdef RISK_ENGINE_HAS_CUDA
                if (backend != 0) return measureGPU(params, n,
                    backend == 1 ? GpuAggregation::CPU : GpuAggregation::GPU);
#else
                (void)backend;
#endif
                return measureCPU(params, n);
            };
            for (int backend = 0; backend < backendCount; ++backend) measure(backend);
            std::array<std::vector<double>, backendCount> times;
            for (int repetition = 1; repetition <= repetitions; ++repetition) {
                std::array<Measurement, backendCount> measurements{};
                // Rotate the first backend to distribute execution-order effects.
                for (int offset = 0; offset < backendCount; ++offset) {
                    const int backend = (repetition - 1 + offset) % backendCount;
                    measurements[backend] = measure(backend);
                }
                for (int backend = 0; backend < backendCount; ++backend) {
                    writeRow(names[backend], n, repetition, measurements[backend], analytical);
                    times[backend].push_back(measurements[backend].totalMs);
                }
            }
            std::cerr << "Paths: " << n << '\n';
            std::array<double, backendCount> medians{};
            for (int backend = 0; backend < backendCount; ++backend) {
                medians[backend] = summarize(names[backend], times[backend]);
            }
#ifdef RISK_ENGINE_HAS_CUDA
            std::cerr << "  CPU / GPU without reduction: " << medians[0] / medians[1] << "x\n"
                      << "  CPU / GPU with reduction: " << medians[0] / medians[2] << "x\n"
                      << "  GPU without / with reduction: " << medians[1] / medians[2] << "x\n";
#endif
        }
        if (!std::cout) throw std::runtime_error("Failed to write benchmark CSV");
    } catch (const std::exception& error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
