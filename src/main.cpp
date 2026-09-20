#include "cpu_pricer.h"
#include "black_scholes.h"

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
    } catch (const std::exception& error) {
        std::cerr << "Pricing failed: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
