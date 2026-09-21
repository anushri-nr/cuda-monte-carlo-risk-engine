# CUDA Monte Carlo Risk Engine

European call option pricing in C++17 and CUDA, with analytical validation,
statistical error estimates, and reproducible CPU/GPU benchmarks.

## Implementations

The project compares three ways to price the same option:

| Implementation | Simulation | Aggregation | Data returned from GPU |
|---|---|---|---|
| `cpu` | Serial CPU | CPU | None |
| `gpu_no_reduction` | GPU, one path per thread | CPU | Every discounted payoff |
| `gpu_reduction` | GPU, one path per thread | GPU block statistics, then CPU merge | One statistical summary per block |

Both GPU implementations use the same payoff-generation function, cuRAND Philox
subsequences, seed, and arithmetic precision. Their comparison isolates the
choice of aggregation location and the resulting allocation and transfer costs.
The CPU uses a different random generator, so its comparison also reflects
backend and arithmetic differences; this is not a comparison with a vectorized
or multithreaded CPU implementation.

## Pricing model

For a European call on a non-dividend-paying stock, each path samples its
risk-neutral terminal price and discounted payoff:

```text
Z ~ N(0, 1)
S(T) = S(0) × exp((r − σ²/2)T + σ√T × Z)
X = exp(−rT) × max(S(T) − K, 0)
```

The option price is estimated by the mean of `X`. With sample standard deviation
`s` and `N` paths, the standard error is `s / √N`; the approximate 95% confidence
interval is `price ± 1.96 × standard_error`. Black–Scholes provides an analytical
reference for the same model. Sampling error is expected, and a single interval
can miss the reference by chance.

The example uses spot 100, strike 100, annual risk-free rate 5%, annual volatility
20%, maturity one year, one million paths, and seed 42. Inputs are currently
specified in `src/main.cpp` and `src/benchmark.cpp`. Both pricers require at least
two paths so sample variance is defined. The current scope is European call
pricing.

## Implementation details and rationale

### CPU

The CPU draws normal samples with `std::mt19937_64` and
`std::normal_distribution<double>`. It uses double precision throughout and
Welford's online algorithm to accumulate mean and squared deviations without
storing every payoff. Drift and diffusion coefficients are computed once per
call, since they are shared by all paths.

### GPU simulation

Each thread uses its path index as a Philox subsequence identifier. This makes
sampling repeatable within the same implementation and environment and gives
both GPU paths the same samples. CPU and GPU seeds do not imply matching samples.

Normal sampling and individual payoff arithmetic use `float`; each payoff is
promoted to `double` before statistical accumulation. This reduces expensive
double-precision arithmetic during simulation while retaining double precision
for variance and mean calculations. Coefficients are computed in double precision
on the host and converted for the kernel. GPU inputs outside representable
single-precision ranges and non-finite aggregate results are rejected.

Single-precision payoffs can introduce rounding error, particularly near the
strike or at extreme parameter scales. Same-sample accuracy tests distinguish
payoff rounding from Monte Carlo sampling error. Passing those tests does not
establish accuracy for every parameter combination.

### GPU without reduction

The kernel writes one promoted, double-precision discounted payoff per path.
The entire array is copied to the CPU, which uses sequential Welford accumulation.
This implementation demonstrates the cost of returning individual outcomes:
its GPU output buffer, host buffer, and return transfer each contain `8 × N`
bytes. At 10 million paths, the return transfer is 80 MB.

### GPU with reduction

Blocks of 256 threads combine payoffs in shared memory during the simulation
kernel. Each block outputs a count, mean, and sum of squared deviations (`M2`).
The CPU merges these summaries to produce the final price and standard error.
There is no intermediate global-memory payoff array.

For two nonempty groups A and B:

```text
n = nA + nB
delta = meanB − meanA
weight = nB / n
mean = meanA + delta × weight
M2 = M2A + M2B + delta² × nA × weight
```

These formulas combine variance statistics without subtracting a large squared
mean from a sum of squares. Equal-size groups use the exact weight `0.5`, avoiding
division in the common case. Unequal groups retain the general formula, and empty
groups contribute nothing. Threads beyond the requested path count contribute
empty statistics and still reach every synchronization barrier.

Each summary occupies 24 bytes on the tested CUDA platform. For 10 million paths,
39,063 summaries require 937,512 bytes, versus 80 MB for individual payoffs.
Final CPU work scales with the number of blocks rather than the number of paths.

### Memory ownership and timing

Every pricing call allocates, uses, and releases its own buffers and timing
events. Resource-owning wrappers clean up when exceptions occur. CUDA launch and
execution failures are checked before results are returned. Buffers are not
retained between benchmark calls: total timings represent independent calls
within an initialized process.

## Build and run

Requirements: CMake 3.18+ and a C++17 compiler. GPU builds additionally require a
CUDA toolkit and compatible NVIDIA GPU. The GPU implementation has been developed
on a Tesla T4 with CUDA 12.8; the current three-way comparison must be measured
on the target system.

CPU build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/risk_engine
```

GPU build for a Tesla T4:

```sh
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-gpu -j2
ctest --test-dir build-gpu --output-on-failure
./build-gpu/risk_engine
```

Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU. In Colab, select a GPU runtime,
use `%cd` to enter the repository, and prefix shell commands with `!`.
The GPU executable displays results for all three implementations.

## Correctness checks

CTest runs CPU checks and, when CUDA is enabled, GPU checks. Coverage includes:

- The analytical reference price and deterministic expiration/zero-volatility cases.
- Invalid path counts, repeatability, and statistical agreement with Black–Scholes.
- Equal, unequal, and empty statistical merges, and counts in partially filled blocks.
- Payoff rounding against a double-precision calculation using identical samples.
- Agreement between both GPU aggregation paths over multiple seeds, path counts,
  and option inputs, allowing for floating-point summation order.

Statistical checks use a loose six-standard-error bound; they are smoke checks,
not a proof of unbiasedness. The deterministic GPU price tolerance is `1e-4`.
For the example inputs, the rounding check requires absolute mean error below
`1e-5` and maximum per-path error below `1e-3` across 100,000 samples.

## Benchmarks and results

Run tests first, then save a benchmark with environment metadata (Python 3):

```sh
python3 scripts/run_benchmarks.py --build-dir build-gpu
```

The script creates a new timestamped directory under `benchmarks/` containing
`results.csv`, `summary.txt`, `environment.json`, and the CMake cache. It records
the commit, working-tree status, compiler versions, and hardware information.
An explicit `--output` directory must not already exist. Use `--build-dir build`
for CPU-only results. Download Colab results before its runtime expires.

For direct execution:

```sh
./build-gpu/risk_benchmark > benchmarks/results.csv 2> benchmarks/summary.txt
```

### Measurement protocol

- 10K, 100K, 1M, and 10M paths with the example option inputs and fixed seed 42.
- One full warm-up per implementation per size, then seven measured repetitions.
- Rotating implementation order to distribute order effects (seven repetitions
  cannot balance three implementations perfectly).
- Median and min/max total times for each implementation; speedup is the ratio
  of medians, with separate CPU/GPU and GPU/GPU comparisons.

Warm-up excludes one-time initialization from the measured calls. Every measured
call still includes allocation, synchronization, transfers, CPU aggregation,
timing instrumentation, and cleanup. CSV and console writes occur outside timing.
Fixed seeds make repetitions timing trials, not independent pricing trials.

CSV columns are `backend`, `paths`, `repetition`, `seed`, `price`, `standard_error`,
`black_scholes`, `absolute_error`, `total_ms`, `kernel_ms`, `transfer_ms`, and
`aggregation_ms`. GPU fields are blank for CPU rows. Kernel time uses CUDA events;
transfer, aggregation, and total time use a steady host clock. With GPU reduction,
kernel time includes simulation and block aggregation. Stage times omit setup and
cleanup and therefore need not sum to total time.

GPU clocks and cloud scheduling can affect results. Compare all three methods
within the same run, retain timing ranges, and report hardware and precision
alongside speedups. Profiler-instrumented times are not benchmark results.
No performance table is hard-coded here; generate the complete comparison from
the current executable rather than combining measurements from different builds.

## Project structure

```text
include/       Shared option parameters and result types
src/           CPU/GPU pricing, analytical pricing, example, and benchmark
tests/        CPU and CUDA correctness checks
scripts/       Benchmark collection and environment capture
benchmarks/    Saved measurements
```
