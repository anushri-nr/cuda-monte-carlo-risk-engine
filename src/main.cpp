#include "cpu_pricer.h"
#include "black_scholes.h"
#ifdef RISK_ENGINE_HAS_CUDA
#include "gpu_pricer.h"
#endif

#include <chrono>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>

int main() {
    const OptionParams params{100.0, 100.0, 0.05, 0.20, 1.0};
    const long long numSimulations = 1'000'000;
    const unsigned long long seed = 42;

    try {
        const auto start = std::chrono::steady_clock::now();
        const auto result = priceEuropeanCallCPU(params, numSimulations, seed);
        const auto end = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = end - start;
        const double analyticalPrice = priceEuropeanCallBlackScholes(params);
        const double margin = 1.96 * result.standardError;

        std::cout << "CPU Monte Carlo European call\n"
                  << "Simulations: " << numSimulations << '\n'
                  << "Seed: " << seed << '\n'
                  << std::fixed << std::setprecision(6)
                  << "Estimated price: " << result.price << '\n'
                  << "Standard error: " << result.standardError << '\n'
                  << "Approximate 95% confidence interval: ["
                  << result.price - margin << ", " << result.price + margin << "]\n"
                  << "Black-Scholes price: " << analyticalPrice << '\n'
                  << "Absolute error: " << std::abs(result.price - analyticalPrice) << '\n'
                  << std::setprecision(3)
                  << "Pricing time (ms): " << elapsed.count() << '\n';

#ifdef RISK_ENGINE_HAS_CUDA
        // Run the full workload once to initialize CUDA and load the kernel.
        // Reusing the seed keeps the reported sample unchanged.
        priceEuropeanCallGPU(params, numSimulations, seed);
        GpuTimings gpuTimings{};
        const auto gpuStart = std::chrono::steady_clock::now();
        const auto gpuResult = priceEuropeanCallGPU(params, numSimulations, seed, &gpuTimings);
        const auto gpuEnd = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> gpuElapsed = gpuEnd - gpuStart;
        const double gpuMargin = 1.96 * gpuResult.standardError;
        std::cout << "\nGPU Monte Carlo European call (fused simulation and reduction)\n"
                  << std::fixed << std::setprecision(6)
                  << "Estimated price: " << gpuResult.price << '\n'
                  << "Standard error: " << gpuResult.standardError << '\n'
                  << "Approximate 95% confidence interval: ["
                  << gpuResult.price - gpuMargin << ", " << gpuResult.price + gpuMargin << "]\n"
                  << "Black-Scholes price: " << analyticalPrice << '\n'
                  << "Absolute error: " << std::abs(gpuResult.price - analyticalPrice) << '\n'
                  << std::setprecision(3)
                  << "Fused kernel time (ms, CUDA events): " << gpuTimings.kernelMs << '\n'
                  << "Separate reduction time (ms, zero when fused): " << gpuTimings.reductionMs << '\n'
                  << "Device-to-host copy time (ms, wall clock): " << gpuTimings.transferMs << '\n'
                  << "CPU summary merge time (ms): " << gpuTimings.aggregationMs << '\n'
                  << "Warmed end-to-end time (ms): " << gpuElapsed.count() << '\n';
#endif
    } catch (const std::exception& error) {
        std::cerr << "Pricing failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
