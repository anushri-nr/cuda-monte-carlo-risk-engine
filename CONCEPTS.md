# Finance and GPU Notes

These are my notes on the concepts behind the project: what the option price
represents, how Monte Carlo estimates it, and how the calculation maps to CUDA.
The [README](README.md) covers the implementation and results, including the
[complete Colab commands](README.md#run-on-google-colab).

## European call options

A **call option** gives its holder the right to buy an underlying asset at a
specified **strike price**. A **European** option can be exercised only at
expiration.

For a strike of 100, the payoff at expiration is:

| Stock price at expiration | Call payoff |
|---|---:|
| 80 | 0 |
| 100 | 0 |
| 120 | 20 |

The formula is `max(stock price − strike, 0)`. These are values per unit of the
underlying; the code does not apply a contract multiplier. Payoff is also not
profit: profit would account for the premium paid and any other costs.

The engine estimates what that uncertain future payoff is worth **today**.
Theoretical option valuation uses inputs such as stock price, strike, interest
rate, volatility, and time to expiration. A model price is not a guarantee of
an available market price. See the Options Industry Council's
[Black–Scholes overview](https://prd-web.optionseducation.org/advancedconcepts/black-scholes-formula).

### Inputs and units

| Code field | Symbol | Meaning | Example |
|---|---|---|---|
| `spot` | S₀ | Stock price today | 100 |
| `strike` | K | Exercise price | 100 |
| `rate` | r | Annual continuously compounded risk-free rate | 0.05 |
| `volatility` | σ | Annualized return volatility | 0.20 |
| `maturity` | T | Years until expiration | 1.0 |

Volatility measures uncertainty, not an expected 20% price increase. Interest
rates and volatility are decimals, and time is in years. Consistent units matter:
using `20` instead of `0.20` describes a completely different model.

## Stock-price model

I use geometric Brownian motion for the stock-price model. Under its pricing distribution,
the terminal stock price is:

```text
S(T) = S₀ × exp((r − σ²/2)T + σ√T × Z)
Z ~ N(0, 1)
```

`Z` is a standard normal random sample, with mean zero and variance one.
The exponential keeps the modeled stock price positive. The random term
`σ√T × Z` controls the spread of outcomes. The `−σ²/2` adjustment makes the
expected terminal price under this distribution equal to `S₀ × exp(rT)`.

### Risk-neutral pricing

This is **risk-neutral pricing**, not a forecast of the stock's actual return.
Under the model's no-arbitrage and replication assumptions, the option value can
be written as an expected discounted payoff under a risk-neutral probability
measure. For a non-dividend-paying stock, that pricing measure uses drift `r`.

The simulation does not claim that investors are indifferent to risk or that
stocks actually earn the risk-free rate. Substituting a guessed stock return
would change the calculation and would not produce the Black–Scholes price.

### Simulation paths

In this implementation, one path is one possible stock price at expiration.
I sample it directly because the terminal distribution is known and the payoff
only depends on that final price. Daily time steps would add work and numerical
error without helping this calculation. An option based on the average price
over a period would need a different simulation.

### Discounting

A payoff received in a year is not a payment received today. The engine multiplies
each future payoff by `exp(−rT)` to express its present value:

```text
X = exp(−rT) × max(S(T) − K, 0)
```

With `r = 0.05`, `T = 1`, and a future payoff of 20, the discounted payoff is
approximately 19.02. This is one simulated outcome, not the final option price.

## Monte Carlo estimation

Monte Carlo estimates an expectation by averaging samples:

```text
estimated price = (X₁ + X₂ + ... + Xₙ) / N
```

A single pricing call with one million paths produces **one price estimate**.
More paths generally improve statistical precision, but they do not guarantee
that every individual estimate gets closer to the analytical answer.

### Standard error and confidence intervals

Payoff standard deviation describes variation between individual simulated
outcomes. **Standard error** describes uncertainty in their estimated mean:

```text
sample variance = Σ(Xᵢ − mean)² / (N − 1)
standard error = sqrt(sample variance / N)
approximate 95% interval = mean ± 1.96 × standard error
```

For a price of 10.4591 and standard error 0.0147, the interval is approximately
10.4303 to 10.4880. Its repeated-sampling interpretation is that approximately
95% of intervals constructed this way would contain the model price when the
large-sample approximation is appropriate.

Standard error decreases approximately as `1 / √N`. Halving it therefore takes
about four times as many paths. GPU acceleration makes those larger sample
counts cheaper to compute; it does not change this convergence rate.

The interval describes **Monte Carlo sampling uncertainty**. It does not include
uncertainty about volatility inputs, model assumptions, or market prices.

### Random seeds and repeated benchmark runs

A seed initializes a pseudorandom generator. Repeating a call with the same
inputs and seed reproduces its sample within the same implementation and
environment. Different generators need not produce identical samples from the
same seed.

I keep the inputs and seed fixed across the seven benchmark repetitions. Each
call produces the same estimate; the repetitions measure timing variability.
I report the median runtime, which is the fourth value after sorting seven times.

## Black–Scholes reference

For this particular contract and model, there is a closed-form answer:

```text
C = S₀ Φ(d₁) − K exp(−rT) Φ(d₂)
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ − σ√T
```

`Φ` is the standard normal cumulative distribution function. For the example
inputs, the price is approximately **10.450584**.

I chose a European call because Black–Scholes gives me a known answer to check
against. For this contract alone, the formula is much cheaper than Monte Carlo.
The purpose of the simulation is to study parallel computation and validate the
numerical implementation against an analytical result.

Both methods share assumptions: no dividends, constant rate and volatility,
continuous stock-price dynamics, and the idealized trading conditions of the
pricing model. Agreement validates a calculation within those assumptions,
not the model's realism for every market situation.

At expiration, the price is intrinsic value `max(S₀ − K, 0)`. At zero volatility,
the payoff is deterministic. These cases help test correctness without relying
on random sampling.

## CUDA execution model

A **kernel** is a function launched to execute across many GPU threads. In this
project each thread generates one terminal stock price and one payoff.

| CUDA term | Meaning in this project |
|---|---|
| Host | CPU code allocating memory, launching kernels, and collecting results |
| Device | NVIDIA GPU executing the pricing kernels |
| Thread | Computes one simulated outcome |
| Block | Group of 256 threads that can cooperate through shared memory |
| Grid | All blocks launched for a pricing call |
| Warp | Group of 32 threads scheduled for execution together |
| SM | Streaming multiprocessor that executes resident blocks and warps |

The global path index is `blockIdx.x × blockDim.x + threadIdx.x`. A million
paths require 3,907 blocks of 256 threads, so the final block has unused lanes.
All blocks do not need to run simultaneously; the hardware schedules them as
resources become available. NVIDIA describes these relationships in its
[CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).

### GPU memory and data movement

Registers hold thread-local working values. Shared memory allows cooperation
within a block. Global memory holds device-wide arrays and block outputs.
CPU memory and GPU memory are distinct in this implementation: returning data
requires an explicit device-to-host copy.

A fast simulation kernel can still leave a slow application if it returns a
large array. I include that transfer in the total pricing time.

## The three implementations

```text
CPU:
  simulate on CPU → accumulate statistics on CPU

GPU without reduction:
  simulate on GPU → copy every payoff → accumulate statistics on CPU

GPU with reduction:
  simulate and combine within GPU blocks → copy summaries → merge on CPU
```

The two GPU methods share payoff generation and precision. This makes their
comparison useful for understanding aggregation and transfer costs. The CPU
reference is serial and uses double precision throughout; it is not an optimized
multicore competitor.

At 10 million paths, the no-reduction method transfers 80 MB of promoted
`double` payoffs. The reduction method transfers 39,063 block summaries, about
0.94 MB. It also gives the CPU far fewer items to combine.

## Statistical reduction

Reduction combines many values into a smaller result. A tree reduction can
combine 256 values in eight rounds:

```text
256 → 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
```

The price and its uncertainty estimate require each group to carry
three statistics: count, mean, and `M2`, the sum of squared deviations from its
mean. Combining those summaries preserves enough information to calculate
sample variance. Averaging block means without considering counts would be
wrong when the last block contains fewer paths.

The merge formula uses `weight = countB / (countA + countB)`. For equally sized
groups, that weight is exactly `0.5`. Most merges inside full blocks have equal
counts, so the code avoids division there while retaining the general formula
for unequal groups.

### Synchronization

After threads write shared memory, `__syncthreads()` ensures the block reaches
the barrier before consuming those values. The reduction uses a barrier between
rounds so one round's reads do not race with the previous round's writes.

Unused lanes in the final block contribute empty statistics; they do not return
before the barriers. These barriers coordinate one block, not the whole grid.
See NVIDIA's [SIMT kernel guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html).

## Precision and performance

`float` and `double` offer different numerical precision and hardware costs.
Here GPU samples and payoffs use `float`, while statistical accumulation uses
`double`. Using more paths reduces sampling error but does not eliminate
floating-point rounding error. I check payoff rounding
against a double-precision reference using identical samples.

**Occupancy** describes resident active warps relative to the hardware limit.
High occupancy can help hide latency, but it does not guarantee high throughput:
a particular arithmetic pipeline can still be saturated.

A **memory-bound** kernel is limited by data movement. A **compute-bound** kernel
is limited by arithmetic resources. **Launch or allocation overhead** can instead
dominate short workloads. These are different problems and require different
measurements; adding threads is not a universal fix.

Nsight Compute examines kernel instructions and resource utilization. Its replay
and instrumentation affect execution, so application timings collected under
the profiler should not be compared with normal benchmark timings. NVIDIA's
[profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) explains
these metrics and collection behavior.

## Benchmark interpretation

A warm-up runs the full calculation once before measuring, so one-time startup
is excluded. Every measured call still allocates buffers and timing events,
executes the computation, copies results, performs CPU aggregation, and cleans
up. There is no persistent allocation reuse between pricing calls.

The reported speedup is a ratio of median **total** times. Kernel-only timing
answers a narrower question and excludes costs required to obtain the price.
Small workloads may favor the CPU because GPU setup costs outweigh parallel
execution benefits. Larger workloads can amortize those costs.

I report the hardware, precision, repetition count, and timing ranges with the
results. A speedup measured on the Colab T4 applies to that setup; it should not
be assumed for another GPU, CPU implementation, or option workload.

## Pricing and risk

An option price is a present value. Risk calculations ask different questions:

- **Delta:** sensitivity of price to a change in the underlying stock price.
- **Gamma:** how delta changes as the stock price changes.
- **Vega:** sensitivity to volatility.
- **Value at Risk (VaR):** a loss quantile for a specified horizon and probability.
- **Expected Shortfall:** average loss in the tail beyond a specified quantile.

The current code calculates prices and sampling uncertainty, not these risk
measures. In particular, the price confidence interval is not VaR. Portfolio
loss forecasting would require a real-world scenario model in addition to the
risk-neutral pricing model used here.

## Code references

| Concept | File |
|---|---|
| Financial inputs and units | [option_params.h](include/option_params.h) |
| Serial Monte Carlo and Welford statistics | [cpu_pricer.cpp](src/cpu_pricer.cpp) |
| Analytical reference | [black_scholes.cpp](src/black_scholes.cpp) |
| GPU samples, reductions, transfers, and statistics | [gpu_pricer.cu](src/gpu_pricer.cu) |
| Repeated timing and CSV output | [benchmark.cpp](src/benchmark.cpp) |
| Accuracy and boundary checks | [tests](tests/) |

For additional background on sensitivities, see the Options Industry Council's
[volatility and Greeks overview](https://prd-web.optionseducation.org/advancedconcepts/volatility-the-greeks).
