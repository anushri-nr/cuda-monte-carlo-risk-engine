# cuda-monte-carlo-risk-engine
GPU-accelerated Monte Carlo option pricing and risk engine built with C++ and CUDA, featuring cuRAND simulation, parallel reductions, profiling, and CPU/GPU benchmarking.

## Status

CPU Monte Carlo baseline implemented with a timed example run and Black–Scholes
price comparison for a European call on a non-dividend-paying stock. CUDA kernels
and benchmarks will be added step by step.

## Structure

```text
cuda-monte-carlo-risk-engine/
├── CMakeLists.txt
├── README.md
├── include/
│   └── option_params.h
├── src/
│   ├── main.cpp
│   ├── black_scholes.cpp
│   ├── black_scholes.h
│   ├── cpu_pricer.cpp
│   └── cpu_pricer.h
├── benchmarks/
└── scripts/
```

`include/` holds shared inputs, `src/` holds pricing and application code,
`benchmarks/` will hold measurements, and `scripts/` will hold automation and plotting tools.

## Roadmap

1. CPU Monte Carlo baseline for a European call option
2. Black–Scholes validation
3. CUDA simulation kernel
4. cuRAND integration
5. GPU parallel reduction
6. Shared-memory and warp-level optimization
7. Nsight profiling
8. CPU vs GPU benchmarking
9. Optional Greeks / VaR extensions

## Build setup

The CPU baseline uses C++17 and CMake 3.18 or newer.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/risk_engine
```

The example prices a European call using one million simulations and seed 42,
with spot and strike of 100, a 5% risk-free rate, 20% volatility, and one year
to expiration. It reports the estimated price, Black–Scholes price, absolute error,
and elapsed pricing time, including
random number generation but excluding console output. A single timing is a
smoke check; repeated benchmarks will be added later. The analytical comparison
is outside the timed region. The CPU result includes standard error estimated
from sample payoff variance using Welford's algorithm and requires at least two
simulations. The displayed approximate 95% confidence interval is price ± 1.96
times standard error, using a large-sample normal approximation. Variance
estimation is included in pricing time. Monte Carlo sampling error is expected; a single
price comparison does not establish statistical convergence.

The GPU milestones will additionally require an NVIDIA GPU and CUDA toolkit.
