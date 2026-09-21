# CUDA Monte Carlo Risk Engine

GPU-accelerated European option pricing in C++17 and CUDA, with statistical
error estimates, Black–Scholes validation, and CPU/GPU benchmarks.

## Features

- European call pricing for a non-dividend-paying stock.
- Serial CPU simulation and GPU simulation with cuRAND Philox random numbers.
- Shared-memory reduction of payoff statistics and a final CPU merge.
- Standard error and approximate 95% confidence intervals.
- Analytical Black–Scholes pricing for validation.
- Repeated benchmarks with CSV output and per-stage GPU timings.

## How it works

Each path samples the stock price at expiration under geometric Brownian motion:

```text
S(T) = S(0) × exp((r − σ²/2)T + σ√T × Z),   Z ~ N(0, 1)
payoff = exp(−rT) × max(S(T) − K, 0)
```

The estimated option price is the mean discounted payoff. Sample variance gives
its standard error; the approximate 95% confidence interval is the price plus
or minus 1.96 standard errors.

On the GPU, each thread generates one payoff. Threads cooperate in blocks of
256 to calculate counts, means, and squared deviations in shared memory. Only
block summaries are written to global memory and copied to the CPU for the
final calculation. GPU normal samples and payoff arithmetic use single precision;
block statistics and the final merge use double precision. Model coefficients
are calculated in double precision on the CPU before conversion for the kernel.
CPU pricing uses double precision throughout. CPU and GPU random generators
differ, so the same seed does not imply identical estimates across backends.

## Requirements

| Build | Requirements |
|---|---|
| CPU | CMake 3.18+ and a C++17 compiler |
| GPU | CPU requirements, CUDA toolkit, and a compatible NVIDIA GPU |

The GPU implementation has been built and tested on a Tesla T4 with CUDA 12.8.

## Build and run

From the repository root, build the CPU version:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/risk_engine
```

To enable CUDA on a Tesla T4:

```sh
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-gpu -j2
./build-gpu/risk_engine
```

Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU. In a Colab GPU runtime, run
shell commands with a `!` prefix and use `%cd` to enter the repository directory.

The example prices one million paths with seed 42 using these inputs:

| Parameter | Value |
|---|---:|
| Stock price | 100 |
| Strike price | 100 |
| Annual risk-free rate | 5% |
| Annual volatility | 20% |
| Time to expiration | 1 year |

The program prints the estimated price, standard error, confidence interval,
Black–Scholes price, absolute error, and execution times. Both pricing functions
require at least two paths to estimate sample variance.

## Tests

After configuring the GPU build, compile and run the CUDA checks:

```sh
nvcc -std=c++17 -O2 -arch=sm_75 -Iinclude tests/gpu_reduction_test.cu -o build-gpu/gpu_reduction_test
./build-gpu/gpu_reduction_test
```

Tests cover partial-block counts, expiration, zero volatility, reproducibility,
invalid path counts, analytical pricing, and payoff rounding error against a
double-precision reference. CUDA tests require an NVIDIA GPU.

## Benchmarks

```sh
./build-gpu/risk_benchmark > benchmarks/results.csv 2> benchmarks/summary.txt
cat benchmarks/summary.txt
```

Use `./build/risk_benchmark` for CPU-only measurements.

The benchmark measures 10K, 100K, 1M, and 10M paths. Each backend receives one
warm-up per size, followed by seven measured repetitions with alternating
CPU/GPU execution order. Inputs and seed stay fixed to measure timing
variability. The summary reports median times, min/max ranges, and the ratio
of CPU median time to GPU median time.

CSV rows identify the `cpu` or `gpu` backend and contain path count, repetition,
seed, price, standard error, analytical price, absolute error, total time,
kernel time, transfer time, and CPU merge time. GPU stage fields are blank for
CPU rows.

Kernel time uses CUDA events. Transfer, CPU merge, and total time use wall-clock
measurements. GPU total time includes allocation, synchronization, timing
instrumentation, transfers, and cleanup; warm-up and console output are excluded.
Stage times omit setup and cleanup, so they do not sum to total time.

For reproducible results, save the commit, compiler versions, GPU/CPU details,
and build configuration alongside the measurements:

```sh
git rev-parse HEAD > benchmarks/environment.txt
nvidia-smi >> benchmarks/environment.txt
nvcc --version >> benchmarks/environment.txt
c++ --version >> benchmarks/environment.txt
lscpu >> benchmarks/environment.txt
cat build-gpu/CMakeCache.txt >> benchmarks/environment.txt
```

These environment commands target Linux, including Colab. GPU clocks and cloud
resource conditions can affect timings. Report medians and ranges, and save
Colab output files before the runtime expires.

## Project structure

```text
include/       Shared option parameters and result types
src/           CPU/GPU pricing, Black–Scholes, example, and benchmarks
tests/         CUDA correctness checks
benchmarks/    Benchmark output and environment records
```
