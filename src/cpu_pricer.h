#pragma once

#include "option_params.h"

struct MonteCarloResult {
    double price;
    double standardError;
};

// Requires at least two simulations to estimate sample variance.
MonteCarloResult priceEuropeanCallCPU(
    const OptionParams& params,
    long long numSimulations,
    unsigned long long seed
);
