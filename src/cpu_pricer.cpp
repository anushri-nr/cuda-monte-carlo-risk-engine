#include "cpu_pricer.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

MonteCarloResult priceEuropeanCallCPU(
    const OptionParams& params,
    long long numSimulations,
    unsigned long long seed
) {
    if (numSimulations < 2) {
        throw std::invalid_argument("At least two simulations are required to estimate standard error");
    }
    if (!std::isfinite(params.spot) || params.spot <= 0.0 ||
        !std::isfinite(params.strike) || params.strike <= 0.0 ||
        !std::isfinite(params.rate) ||
        !std::isfinite(params.volatility) || params.volatility < 0.0 ||
        !std::isfinite(params.maturity) || params.maturity < 0.0) {
        throw std::invalid_argument("Invalid option parameters");
    }

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    const double drift =
        (params.rate - 0.5 * params.volatility * params.volatility)
        * params.maturity;
    const double diffusion = params.volatility * std::sqrt(params.maturity);

    double meanPayoff = 0.0;
    double squaredDeviationSum = 0.0;
    for (long long i = 0; i < numSimulations; ++i) {
        const double z = normal(rng);
        const double futurePrice = params.spot * std::exp(drift + diffusion * z);
        const double payoff = std::max(futurePrice - params.strike, 0.0);

        // Welford's algorithm tracks mean and variance without storing payoffs.
        const double delta = payoff - meanPayoff;
        meanPayoff += delta / static_cast<double>(i + 1);
        squaredDeviationSum += delta * (payoff - meanPayoff);
    }

    const double sampleVariance = squaredDeviationSum / static_cast<double>(numSimulations - 1);
    const double discount = std::exp(-params.rate * params.maturity);
    return {
        discount * meanPayoff,
        discount * std::sqrt(sampleVariance / static_cast<double>(numSimulations))
    };
}
