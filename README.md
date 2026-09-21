# cuda-monte-carlo-risk-engine
GPU-accelerated Monte Carlo option pricing and risk engine built with C++ and CUDA, featuring cuRAND simulation, parallel reductions, profiling, and CPU/GPU benchmarking.

## Status

CPU Monte Carlo baseline implemented with a timed example run and Black–Scholes
price comparison for a European call on a non-dividend-paying stock. An optional
CUDA baseline generates discounted payoffs on the GPU and reduces them into block statistics before a final CPU merge. The initial CUDA
baseline was run on Colab T4; the new reduction needs verification there.

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
A second kernel uses shared memory to merge counts, means, and squared
deviations within each 256-thread block. Inactive lanes contribute empty
statistics and still participate in all barriers. Only block summaries are
copied to the CPU, which merges them to estimate price and standard error. CPU and GPU generators differ, so matching seeds do not imply
matching prices. Compare each estimate with Black–Scholes using its uncertainty;
an approximate 95% interval can miss the analytical value by chance.

The program runs one full GPU warm-up call before the measured call, using the
same seed. Warm-up samples are discarded. The measured call reports:

- Kernel time using CUDA events, including cuRAND initialization and sampling.
- GPU block reduction time using CUDA events.
- Blocking device-to-host copy time using a CPU wall clock (including any host staging).
- CPU summary merge time using a CPU wall clock.
- Warmed end-to-end wall time, including allocations, all stages, timing
  instrumentation, and cleanup, but excluding the warm-up and console output.

The stage times do not sum to total time because setup and cleanup are additional
costs. Each program invocation warms up its own CUDA context. This is still one
measured run; repeated benchmarks and further reduction optimizations are later milestones.


For one million paths, 3,907 summaries of 24 bytes replace the 8,000,000-byte
payoff transfer (93,768 bytes). Payoffs are still stored on the GPU in this
version; fusing simulation and reduction is a possible later optimization.

Run the reduction checks on Colab after building:

```python
!nvcc -std=c++17 -O2 -arch=sm_75 -Iinclude tests/gpu_reduction_test.cu -o build-gpu/gpu_reduction_test
!./build-gpu/gpu_reduction_test
```

These compare GPU summaries against a two-pass long-double CPU reference for
small and partially filled blocks, constant payoffs, and large-offset data.


## Repeated benchmarks

`risk_benchmark` measures 10,000, 100,000, 1,000,000, and 10,000,000 paths.
Each size gets one full warm-up per backend and seven measured runs in the
same process. CPU/GPU execution order alternates. Seed 42 and the example
option inputs stay fixed: repeated rows measure timing variability, not
independent Monte Carlo errors. This is a serial CPU baseline.

CSV rows go to stdout; medians, min/max ranges, and the ratio of CPU median to
GPU median go to stderr. Total GPU timing includes allocation, synchronization,
transfers, CPU summary merging, timing instrumentation, and cleanup. Warm-up
and CSV writes are excluded. This measures repeated full calls, not persistent
buffer reuse or kernel-only speedup. CPU-only builds also support the benchmark.

On Colab, after pulling and building:

```python
!./build-gpu/risk_benchmark > benchmarks/results.csv 2> benchmarks/summary.txt
!cat benchmarks/summary.txt
```

Check that the benchmark exits successfully before using the CSV; a failed run
may leave partial data. Each run replaces these output files. Save environment
information alongside results for reproducibility:

```python
!git rev-parse HEAD > benchmarks/environment.txt
!nvidia-smi >> benchmarks/environment.txt
!nvcc --version >> benchmarks/environment.txt
!c++ --version >> benchmarks/environment.txt
!lscpu >> benchmarks/environment.txt
!cat build-gpu/CMakeCache.txt >> benchmarks/environment.txt
```

Download the results, summary, and environment files before the Colab runtime
expires. GPU timings still require an NVIDIA environment; local CPU checks do
not validate the CUDA benchmark path.
