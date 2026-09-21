#pragma once

#include "option_params.h"
#include "monte_carlo_result.h"

enum class GpuAggregation { CPU, GPU };

struct GpuTimings {
    double kernelMs;
    double transferMs;
    double aggregationMs;
};

// Both modes generate identical payoffs. CPU transfers all payoffs; GPU transfers
// block statistics. Each call owns its allocations. Requires at least two paths.
MonteCarloResult priceEuropeanCallGPU(
    const OptionParams& params,
    long long numSimulations,
    unsigned long long seed,
    GpuTimings* timings = nullptr,
    GpuAggregation aggregation = GpuAggregation::GPU
);
