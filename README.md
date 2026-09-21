# cuda-monte-carlo-risk-engine
GPU-accelerated Monte Carlo option pricing and risk engine built with C++ and CUDA, featuring cuRAND simulation, parallel reductions, profiling, and CPU/GPU benchmarking.

## Status

CPU Monte Carlo baseline implemented with a timed example run and Black–Scholes
price comparison for a European call on a non-dividend-paying stock. An optional
CUDA baseline generates discounted payoffs on the GPU and aggregates them on the
CPU. GPU compilation and execution need verification on an NVIDIA machine.

## Structure

```text
cuda-monte-carlo-risk-engine/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── option_params.h
│   └── monte_carlo_result.h
├── src/
│   ├── main.cpp
│   ├── black_scholes.cpp
│   ├── black_scholes.h
│   ├── cpu_pricer.cpp
│   ├── cpu_pricer.h
│   ├── gpu_pricer.h
│   └── gpu_pricer.cu
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

## CUDA baseline on Colab (T4)

After pushing local changes to GitHub, run these commands in the Colab notebook:

```python
%cd /content/cuda-monte-carlo-risk-engine
!git pull --ff-only
!cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
!cmake --build build-gpu -j2
!./build-gpu/risk_engine
```

The architecture setting above targets the T4. CUDA is disabled by default so
the CPU build can still run on a Mac without a CUDA toolkit.

Each GPU thread initializes a cuRAND Philox subsequence using its path index,
generates one normal sample, and writes one discounted payoff. Threads beyond
the requested simulation count return without accessing the payoff array.
After copying the array to the CPU, Welford's algorithm estimates the mean and
standard error. CPU and GPU generators differ, so matching seeds do not imply
matching prices. Compare each estimate with Black–Scholes using its uncertainty;
an approximate 95% interval can miss the analytical value by chance.

The program runs one full GPU warm-up call before the measured call, using the
same seed. Warm-up samples are discarded. The measured call reports:

- Kernel time using CUDA events, including cuRAND initialization and sampling.
- Blocking device-to-host copy time using a CPU wall clock (including any host staging).
- CPU aggregation time using a CPU wall clock.
- Warmed end-to-end wall time, including allocations, all stages, timing
  instrumentation, and cleanup, but excluding the warm-up and console output.

The stage times do not sum to total time because setup and cleanup are additional
costs. Each program invocation warms up its own CUDA context. This is still one
measured run; repeated benchmarks and GPU reduction are later milestones.
