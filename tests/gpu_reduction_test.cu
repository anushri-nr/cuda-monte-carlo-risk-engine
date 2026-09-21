// Compile as a standalone CUDA test; including the implementation exposes its
// internal reduction kernel for testing with known, non-random payoffs.
#include "../src/gpu_pricer.cu"
#include <algorithm>
#include <iostream>

int main() {
    try {
        for (long long n : {2LL, 255LL, 256LL, 257LL, 1025LL}) {
            for (bool constant : {false, true}) {
                std::vector<double> values(static_cast<std::size_t>(n));
                long double sum = 0;
                for (long long i = 0; i < n; ++i) {
                    values[i] = constant ? 7.0 : 1e6 + static_cast<double>(i % 17) / 8;
                    sum += values[i];
                }
                const long double mean = sum / n;
                long double m2 = 0;
                for (double x : values) m2 += (x - mean) * (x - mean);
                const auto blocks = (n - 1) / reductionThreads + 1;
                DevicePayoffs<double> input(values.size() * sizeof(double));
                DevicePayoffs<Moments> output(blocks * sizeof(Moments));
                checkCuda(cudaMemcpy(input.data, values.data(), values.size() * sizeof(double),
                                     cudaMemcpyHostToDevice), "upload test input");
                reducePayoffs<<<static_cast<unsigned int>(blocks), reductionThreads>>>(input.data, n, output.data);
                checkCuda(cudaGetLastError(), "test reduction launch");
                checkCuda(cudaDeviceSynchronize(), "test reduction execution");
                std::vector<Moments> partials(blocks);
                checkCuda(cudaMemcpy(partials.data(), output.data, blocks * sizeof(Moments),
                                     cudaMemcpyDeviceToHost), "download test summaries");
                Moments total{0, 0, 0};
                for (const auto& partial : partials) total = mergeMoments(total, partial);
                if (total.count != n || std::abs(total.mean - mean) > 1e-8L ||
                    std::abs(total.m2 - m2) > 1e-7L * std::max(1.0L, m2)) {
                    throw std::runtime_error("Reduction mismatch for N=" + std::to_string(n));
                }
            }
        }
        for (long long n : {2LL, 255LL, 256LL, 257LL, 1025LL, 100000LL}) {
            for (const OptionParams params : {
                    OptionParams{100,100,.05,.2,1},
                    OptionParams{110,100,.05,.2,0},
                    OptionParams{100,100,.05,0,1}}) {
                for (unsigned long long seed : {42ULL, 123ULL}) {
                    const auto separate = priceEuropeanCallGPU(params, n, seed, nullptr, GpuMethod::Separate);
                    const auto fused = priceEuropeanCallGPU(params, n, seed, nullptr, GpuMethod::Fused);
                    if (!std::isfinite(fused.price) || !std::isfinite(fused.standardError) ||
                        std::abs(fused.price - separate.price) > 1e-10 * std::max(1.0, std::abs(separate.price)) ||
                        std::abs(fused.standardError - separate.standardError) > 1e-10 * std::max(1.0, separate.standardError)) {
                        throw std::runtime_error("Fused/separate mismatch for N=" + std::to_string(n));
                    }
                }
            }
        }
        std::cout << "GPU reduction checks passed (partial blocks, constant payoffs, large-offset variance, fused/separate equivalence).\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
