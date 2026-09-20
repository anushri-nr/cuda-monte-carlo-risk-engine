#include "black_scholes.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

double priceEuropeanCallBlackScholes(const OptionParams& params) {
    if (!std::isfinite(params.spot) || params.spot <= 0.0 ||
        !std::isfinite(params.strike) || params.strike <= 0.0 ||
        !std::isfinite(params.rate) ||
        !std::isfinite(params.volatility) || params.volatility < 0.0 ||
        !std::isfinite(params.maturity) || params.maturity < 0.0) {
        throw std::invalid_argument("Invalid option parameters");
    }

    if (params.maturity == 0.0) {
        return std::max(params.spot - params.strike, 0.0);
    }

    const double discountedStrike = params.strike * std::exp(-params.rate * params.maturity);
    if (params.volatility == 0.0) {
        return std::max(params.spot - discountedStrike, 0.0);
    }

    const double sigmaSqrtT = params.volatility * std::sqrt(params.maturity);
    const double d1 = (std::log(params.spot / params.strike)
        + (params.rate + 0.5 * params.volatility * params.volatility) * params.maturity)
        / sigmaSqrtT;
    const double d2 = d1 - sigmaSqrtT;
    const auto normalCDF = [](double x) {
        return 0.5 * std::erfc(-x / std::sqrt(2.0));
    };

    return params.spot * normalCDF(d1) - discountedStrike * normalCDF(d2);
}
