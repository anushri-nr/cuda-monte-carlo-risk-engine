# CUDA Monte Carlo Risk Engine

C++17 and CUDA European call pricing for a non-dividend-paying stock, with a
serial CPU baseline, Black–Scholes validation, and repeated CPU/GPU benchmarks.

## Implementation

The CPU simulates discounted payoffs and estimates price and standard error
using Welford's algorithm. The GPU uses one fused simulation-and-reduction
kernel: each thread draws a normal sample from its own cuRAND Philox subsequence
and contributes a payoff to a shared-memory reduction. Each 256-thread block
writes a count, mean, and sum of squared deviations. The CPU merges these block
summaries. Inactive threads contribute empty summaries and reach every barrier.

Only block summaries are allocated in GPU global memory; no intermediate payoff
array is needed. At 10 million paths this avoids an 80 MB payoff buffer.
CPU and GPU generators differ, so matching seeds do not imply matching prices.
The approximate 95% confidence interval is price ± 1.96 × standard error;
a single interval can miss the analytical price by chance.

## Build and run

CPU only (CMake 3.18+ and a C++17 compiler):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/risk_engine
```

CUDA on a Colab T4 (NVIDIA GPU and CUDA toolkit required):

```python
%cd /content/cuda-monte-carlo-risk-engine
!git pull --ff-only
!cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
!cmake --build build-gpu -j2
!./build-gpu/risk_engine
```

The example uses one million paths, seed 42, spot and strike 100, a 5% risk-free
rate, 20% volatility, and one year to expiration. Both pricers require at least
two paths to estimate sample variance.

GPU output reports fused kernel time with CUDA events, blocking return-copy time
and CPU summary merge time with a wall clock, and total wall time. The full GPU
call is warmed up once before measurement. Total time includes allocations,
synchronization, timing instrumentation, transfers, merging, and cleanup.
Warm-up and console output are excluded. Stage times omit setup and cleanup,
so they do not sum to total time.

## Tests

Run the standalone fused-kernel checks on Colab:

```python
!nvcc -std=c++17 -O2 -arch=sm_75 -Iinclude tests/gpu_reduction_test.cu -o build-gpu/gpu_reduction_test
!./build-gpu/gpu_reduction_test
```

Checks cover partial-block counts, expiration, zero volatility, repeatability,
invalid simulation counts, and a loose statistical check against Black–Scholes.
Local CPU builds cannot validate CUDA execution.

## Benchmarks

`risk_benchmark` measures 10K, 100K, 1M, and 10M paths with one warm-up per
backend per size, followed by seven measured repetitions. CPU/GPU order
alternates. Fixed seed 42 measures timing variability, not independent pricing
trials. CPU and GPU total medians are compared, including allocation and cleanup
costs. This is not a persistent-buffer or kernel-only speedup benchmark.

CSV goes to stdout; median times, min/max ranges, and CPU/GPU speedup go to stderr:

```python
!./build-gpu/risk_benchmark > benchmarks/results_fused_only.csv 2> benchmarks/summary_fused_only.txt
!cat benchmarks/summary_fused_only.txt
```

Rows use `cpu` and `gpu_fused` backend labels. Columns include price, standard
error, analytical price, absolute error, total time, fused kernel time, transfer
time, and merge time. The obsolete `reduction_ms` column has been removed.
CPU rows leave GPU stage columns blank. Older CSV files remain historical data.
A failed run can leave partial output; check successful completion before using
results. Running the same command again replaces its output files.

Record environment information alongside results:

```python
!git rev-parse HEAD > benchmarks/environment.txt
!nvidia-smi >> benchmarks/environment.txt
!nvcc --version >> benchmarks/environment.txt
!c++ --version >> benchmarks/environment.txt
!lscpu >> benchmarks/environment.txt
!cat build-gpu/CMakeCache.txt >> benchmarks/environment.txt
```

Download results and environment information before the Colab runtime expires.
GPU clocks and shared-cloud conditions can affect timing; report medians and
ranges rather than selecting the fastest samples.

## Layout and next steps

- `include/`: shared option inputs and result types.
- `src/`: CPU pricing, fused GPU pricing, analytical formula, example, benchmarks.
- `tests/`: standalone CUDA correctness checks.
- `benchmarks/`: measured data and environment records.
- `scripts/`: reserved for analysis and plotting helpers.

Next milestones: Nsight profiling, evidence-based kernel optimization, and
optional Greeks or VaR extensions.
