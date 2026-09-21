# CUDA Monte Carlo Risk Engine

I built this project to compare CPU and GPU Monte Carlo option pricing and
measure how much GPU-side reduction changes the overall runtime. It prices
European call options, checks the estimates against Black–Scholes, and reports
both pricing uncertainty and execution time.

The implementation uses C++17, CUDA, and cuRAND. My notes on the financial model,
statistics, and GPU design are in [CONCEPTS.md](CONCEPTS.md).

## Implementations

I kept three implementations to make the comparison clear:

| Implementation | Simulation | Aggregation | Data returned from GPU |
|---|---|---|---|
| `cpu` | Serial CPU | CPU | None |
| `gpu_no_reduction` | GPU, one path per thread | CPU | Every discounted payoff |
| `gpu_reduction` | GPU, one path per thread | GPU block statistics, then CPU merge | One statistical summary per block |

Both GPU versions use the same payoff-generation function, cuRAND Philox
subsequences, seed, and precision. I kept these consistent so their timing
difference reflects aggregation and data movement. The CPU reference is serial
and uses a different random generator and double-precision arithmetic.

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

## Design choices

### CPU

For the CPU reference, I used `std::mt19937_64` and
`std::normal_distribution<double>`. Welford's algorithm tracks the mean and
variance in double precision without storing all the payoffs. Drift and
diffusion are calculated once per call because they are the same for every path.

### GPU simulation

Each thread uses its path index as a Philox subsequence identifier. This makes
sampling repeatable within the same implementation and environment and gives
both GPU paths the same samples. CPU and GPU seeds do not imply matching samples.

I use `float` for GPU normal samples and payoff calculations, then promote each
payoff to `double` for statistical accumulation. This keeps the simulation's
double-precision arithmetic cost down while retaining double precision for
means and variances. Coefficients are computed in double precision
on the host and converted for the kernel. GPU inputs outside representable
single-precision ranges and non-finite aggregate results are rejected.

Single-precision payoffs can introduce rounding error, particularly near the
strike or at extreme parameter scales. Same-sample accuracy tests distinguish
payoff rounding from Monte Carlo sampling error. Passing those tests does not
establish accuracy for every parameter combination.

### GPU without reduction

The kernel writes one promoted, double-precision discounted payoff per path.
The entire array is copied to the CPU, which uses sequential Welford accumulation.
I kept this version to measure the cost of copying every outcome back and
aggregating on the CPU. Its GPU output buffer, host buffer, and return transfer
each contain `8 × N` bytes. At 10 million paths, the return transfer is 80 MB.

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

I use these merge formulas to avoid the numerical cancellation that can occur
when variance is calculated by subtracting a squared mean from a mean of squares. Equal-size groups use the exact weight `0.5`, avoiding
division in the common case. Unequal groups retain the general formula, and empty
groups contribute nothing. Threads beyond the requested path count contribute
empty statistics and still reach every synchronization barrier.

Each summary occupies 24 bytes on the tested CUDA platform. For 10 million paths,
39,063 summaries require 937,512 bytes, versus 80 MB for individual payoffs.
Final CPU work scales with the number of blocks rather than the number of paths.

### Memory ownership and timing

Every pricing call allocates, uses, and releases its own buffers and timing
events. Resource-owning wrappers clean up when exceptions occur. CUDA launch and
execution failures are checked before results are returned. I allocate buffers within each pricing call rather than retaining them between
calls. The benchmark therefore includes the cost of obtaining a result from an
independent call within an initialized process.

## Build and run

Requirements: CMake 3.18+ and a C++17 compiler. GPU builds additionally require a
CUDA toolkit and compatible NVIDIA GPU. I built and tested the GPU code
on a Colab Tesla T4 with CUDA 12.8.

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

Set `CMAKE_CUDA_ARCHITECTURES` to match the GPU. The GPU executable displays
results for all three implementations.

## Run on Google Colab

Select **Runtime → Change runtime type → T4 GPU**, then run:

```python
!nvidia-smi
!nvcc --version
!git clone https://github.com/anushri-nr/cuda-monte-carlo-risk-engine.git
%cd /content/cuda-monte-carlo-risk-engine
!cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
!cmake --build build-gpu -j2
!ctest --test-dir build-gpu --output-on-failure
```

Once both tests pass:

```python
!./build-gpu/risk_engine
!python3 scripts/run_benchmarks.py --build-dir build-gpu
```

Results are saved under `benchmarks/` in the folder printed by the script.
Download the files from Colab's Files panel before disconnecting.

If the repository is already cloned, use `!git pull --ff-only` from the
repository directory instead of cloning again, then rebuild and test.

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

### Measured results

I ran these benchmarks on Google Colab on **September 21, 2026**, at commit
`bf9c0d2`, using a Tesla T4 and an Intel Xeon CPU @ 2.00 GHz (serial CPU pricing).
The Release build used GCC 13.3.0, CUDA 12.8, and CUDA architecture 75.
CPU calculations use double precision; both GPU methods use single-precision
payoffs and double-precision statistics, as described above.

Times below are **median total milliseconds [minimum–maximum]** across seven
measured runs after one warm-up per implementation and size. They include
allocation, transfers, aggregation, and cleanup. All three implementations
were measured in the same benchmark run.

| Paths | CPU (ms) | GPU without reduction (ms) | GPU with reduction (ms) | CPU / GPU with reduction | GPU without / with reduction |
|---|---:|---:|---:|---:|---:|
| 10K | 0.439 [0.431–0.553] | 1.657 [1.649–1.759] | 1.584 [1.561–1.621] | 0.28× | 1.05× |
| 100K | 4.332 [4.320–4.588] | 2.553 [2.538–2.589] | 1.591 [1.576–1.606] | 2.72× | 1.60× |
| 1M | 43.884 [43.656–51.268] | 11.114 [10.744–13.182] | 1.981 [1.936–2.081] | 22.16× | 5.61× |
| 10M | 444.896 [438.863–565.078] | 134.454 [129.318–154.511] | 6.020 [5.729–6.225] | 73.91× | 22.34× |

At **10 million paths**, GPU block reduction delivered **73.91× speedup over
the serial CPU** and **22.34× over GPU without reduction**. Returning block
statistics instead of individual payoffs reduces both transfer volume and CPU
aggregation work. At 10K paths, CPU execution was faster than either GPU method.
The reported speedups apply to this workload and Colab environment.

Source records: [raw measurements](benchmarks/results.csv),
[benchmark summary](benchmarks/summary.txt),
[environment metadata](benchmarks/environment.json), and
[build configuration](benchmarks/CMakeCache.txt).

### Running the benchmark

After running the tests, collect timings and environment metadata with Python 3:

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

GPU clocks and cloud scheduling affect timing, so I report ranges alongside
medians and compare all three methods in the same run. Profiling is done
separately; timings collected under Nsight are not used in the benchmark table.

## Project structure

```text
include/       Shared option parameters and result types
src/           CPU/GPU pricing, analytical pricing, example, and benchmark
tests/         CPU and CUDA correctness checks
scripts/       Benchmark collection and environment capture
benchmarks/    Saved measurements
```
