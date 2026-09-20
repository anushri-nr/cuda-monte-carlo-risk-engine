#pragma once

#include "option_params.h"
#include "monte_carlo_result.h"

// One payoff per GPU thread, followed by CPU aggregation. Requires N >= 2.
MonteCarloResult priceEuropeanCallGPU(
    const OptionParams& params,
    long long numSimulations,
    unsigned long long seed
);
