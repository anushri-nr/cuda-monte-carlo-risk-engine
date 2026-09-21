// Standalone CUDA checks for the pricing kernel and its public pricing interface.
#include "../src/gpu_pricer.cu"
#include <algorithm>
#include <iostream>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

__global__ void checkMomentMerges(Moments* output) {
    const Moments a{2, 2.0, 2.0}; // Samples [1, 3].
    output[0] = mergeMoments(a, Moments{2, 6.0, 2.0}); // [5, 7].
    output[1] = mergeMoments(a, Moments{1, 8.0, 0.0}); // [8].
    output[2] = mergeMoments(Moments{0, 0.0, 0.0}, a);
    output[3] = mergeMoments(a, Moments{0, 0.0, 0.0});
}

// Use identical normal samples to isolate payoff rounding from RNG differences.
__global__ void payoffPrecisionCheck(double* errors, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t state;
    curand_init(42, i, 0, &state);
    const float z = curand_normal(&state);
    const float payoff = static_cast<float>(exp(-0.05)) *
        fmaxf(100.0f * expf(0.03f + 0.2f * z) - 100.0f, 0.0f);
    const double reference = exp(-0.05) *
        fmax(100.0 * exp(0.03 + 0.2 * static_cast<double>(z)) - 100.0, 0.0);
    errors[i] = static_cast<double>(payoff) - reference;
}

int main() {
    try {
        DeviceBuffer<Moments> mergedDevice(4 * sizeof(Moments));
        checkMomentMerges<<<1, 1>>>(mergedDevice.data);
        checkCuda(cudaGetLastError(), "moment checks launch");
        checkCuda(cudaDeviceSynchronize(), "moment checks execution");
        Moments merged[4];
        checkCuda(cudaMemcpy(merged, mergedDevice.data, sizeof(merged), cudaMemcpyDeviceToHost),
                  "download moment checks");
        const Moments expected[] = {{4, 4.0, 20.0}, {3, 4.0, 26.0},
                                    {2, 2.0, 2.0}, {2, 2.0, 2.0}};
        for (int i = 0; i < 4; ++i) {
            require(merged[i].count == expected[i].count &&
                    std::abs(merged[i].mean - expected[i].mean) < 1e-12 &&
                    std::abs(merged[i].m2 - expected[i].m2) < 1e-12,
                    "Moment merge mismatch");
        }
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
                require(std::abs(result.price - expected) < 1e-4 && result.standardError == 0,
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
        // Validate rounding error against double-precision payoff arithmetic.
        constexpr int sampleCount = 100000;
        DeviceBuffer<double> deviceErrors(sampleCount * sizeof(double));
        payoffPrecisionCheck<<<(sampleCount + 255) / 256, 256>>>(deviceErrors.data, sampleCount);
        checkCuda(cudaGetLastError(), "precision check launch");
        checkCuda(cudaDeviceSynchronize(), "precision check execution");
        std::vector<double> errors(sampleCount);
        checkCuda(cudaMemcpy(errors.data(), deviceErrors.data, sampleCount * sizeof(double),
                             cudaMemcpyDeviceToHost), "download precision errors");
        double sumError = 0;
        double maxError = 0;
        for (double error : errors) {
            require(std::isfinite(error), "Non-finite precision error");
            sumError += error;
            maxError = std::max(maxError, std::abs(error));
        }
        require(std::abs(sumError / sampleCount) < 1e-5 && maxError < 1e-3,
                "Payoff precision error exceeds tolerance");
        for (long long n : {0LL, 1LL}) {
            bool rejected = false;
            try { priceEuropeanCallGPU(params, n, 42); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Invalid simulation count accepted");
        }
        std::cout << "GPU checks passed (block counts, deterministic prices, reproducibility, analytical reference, payoff precision).\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
