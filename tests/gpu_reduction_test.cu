// Standalone CUDA checks for the pricing kernel and its public pricing interface.
#include "../src/gpu_pricer.cu"
#include <algorithm>
#include <iostream>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        for (long long n : {2LL, 255LL, 256LL, 257LL, 1025LL, 100000LL}) {
            // Expiration makes every payoff exactly 10. Inspect each summary
            // to verify inactive lanes are excluded from partial-block counts.
            const auto blocks = (n - 1) / reductionThreads + 1;
            DeviceBuffer<Moments> output(blocks * sizeof(Moments));
            simulateAndReduce<<<static_cast<unsigned int>(blocks), reductionThreads>>>(
                110, 100, 0, 0, 1, 42, n, output.data);
            checkCuda(cudaGetLastError(), "test kernel launch");
            checkCuda(cudaDeviceSynchronize(), "test kernel execution");
            std::vector<Moments> partials(blocks);
            checkCuda(cudaMemcpy(partials.data(), output.data, blocks * sizeof(Moments),
                                 cudaMemcpyDeviceToHost), "download summaries");
            for (long long b = 0; b < blocks; ++b) {
                require(partials[b].count == std::min(static_cast<long long>(reductionThreads),
                                                     n - b * reductionThreads), "Wrong block count");
                require(partials[b].mean == 10 && partials[b].m2 == 0, "Wrong constant summary");
            }
            for (const OptionParams params : {OptionParams{110,100,.05,.2,0},
                                             OptionParams{90,100,.05,.2,0},
                                             OptionParams{100,100,.05,0,1}}) {
                const double expected = std::max(params.spot - params.strike *
                                                 std::exp(-params.rate * params.maturity), 0.0);
                const auto result = priceEuropeanCallGPU(params, n, 42);
                require(std::abs(result.price - expected) < 1e-10 && result.standardError == 0,
                        "Deterministic pricing mismatch");
            }
        }
        const OptionParams params{100,100,.05,.2,1};
        for (unsigned long long seed : {42ULL, 123ULL}) {
            const auto result = priceEuropeanCallGPU(params, 1000000, seed);
            const auto repeat = priceEuropeanCallGPU(params, 1000000, seed);
            require(std::isfinite(result.price) && std::isfinite(result.standardError) &&
                    result.standardError > 0, "Invalid stochastic result");
            require(result.price == repeat.price && result.standardError == repeat.standardError,
                    "Seed reproducibility failed");
            // Loose statistical smoke check against the known analytical price;
            // deterministic checks above establish exact boundary behavior.
            require(std::abs(result.price - 10.450583572185565) < 6 * result.standardError,
                    "Price outside six-standard-error reference bound");
        }
        for (long long n : {0LL, 1LL}) {
            bool rejected = false;
            try { priceEuropeanCallGPU(params, n, 42); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Invalid simulation count accepted");
        }
        std::cout << "GPU checks passed (block counts, deterministic prices, reproducibility, analytical reference).\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
