#include "gpu_pricer.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Own device memory so it is also released if a CUDA operation throws.
struct DevicePayoffs {
    double* data = nullptr;
    explicit DevicePayoffs(std::size_t bytes) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "cudaMalloc");
    }
    ~DevicePayoffs() { cudaFree(data); }
    DevicePayoffs(const DevicePayoffs&) = delete;
    DevicePayoffs& operator=(const DevicePayoffs&) = delete;
};

struct CudaEvent {
    cudaEvent_t event{};
    CudaEvent() { checkCuda(cudaEventCreate(&event), "cudaEventCreate"); }
    ~CudaEvent() { cudaEventDestroy(event); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
};

__global__ void simulatePayoffs(
    double spot, double strike, double drift, double diffusion, double discount,
    unsigned long long seed, long long n, double* payoffs
) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Each path gets its own Philox subsequence.
    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(i), 0, &state);
    const double z = curand_normal_double(&state);
    const double futurePrice = spot * exp(drift + diffusion * z);
    payoffs[i] = discount * fmax(futurePrice - strike, 0.0);
}
} // namespace

MonteCarloResult priceEuropeanCallGPU(
    const OptionParams& params, long long numSimulations, unsigned long long seed,
    GpuTimings* timings
) {
    if (numSimulations < 2) {
        throw std::invalid_argument("At least two simulations are required to estimate standard error");
    }
    if (!std::isfinite(params.spot) || params.spot <= 0.0 ||
        !std::isfinite(params.strike) || params.strike <= 0.0 ||
        !std::isfinite(params.rate) ||
        !std::isfinite(params.volatility) || params.volatility < 0.0 ||
        !std::isfinite(params.maturity) || params.maturity < 0.0) {
        throw std::invalid_argument("Invalid option parameters");
    }

    constexpr int threads = 256;
    const long long blocks = (numSimulations - 1) / threads + 1;
    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "cudaGetDevice");
    checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (blocks > properties.maxGridSize[0] ||
        static_cast<unsigned long long>(numSimulations) >
        std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw std::invalid_argument("Simulation count exceeds device or allocation limits");
    }

    const auto n = static_cast<std::size_t>(numSimulations);
    const auto bytes = n * sizeof(double);
    std::vector<double> payoffs(n);
    DevicePayoffs devicePayoffs(bytes);
    const double drift = (params.rate - 0.5 * params.volatility * params.volatility) * params.maturity;
    const double diffusion = params.volatility * std::sqrt(params.maturity);
    const double discount = std::exp(-params.rate * params.maturity);

    CudaEvent kernelStart;
    CudaEvent kernelEnd;
    checkCuda(cudaEventRecord(kernelStart.event), "record kernel start");
    simulatePayoffs<<<static_cast<unsigned int>(blocks), threads>>>(
        params.spot, params.strike, drift, diffusion, discount,
        seed, numSimulations, devicePayoffs.data);
    checkCuda(cudaGetLastError(), "simulatePayoffs launch");
    checkCuda(cudaEventRecord(kernelEnd.event), "record kernel end");
    checkCuda(cudaEventSynchronize(kernelEnd.event), "simulatePayoffs execution");
    float kernelMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&kernelMs, kernelStart.event, kernelEnd.event),
              "measure kernel time");

    const auto transferStart = std::chrono::steady_clock::now();
    checkCuda(cudaMemcpy(payoffs.data(), devicePayoffs.data, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy payoffs to host");
    const auto transferEnd = std::chrono::steady_clock::now();

    const auto aggregationStart = std::chrono::steady_clock::now();
    double mean = 0.0;
    double squaredDeviationSum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double delta = payoffs[i] - mean;
        mean += delta / static_cast<double>(i + 1);
        squaredDeviationSum += delta * (payoffs[i] - mean);
    }
    const double variance = squaredDeviationSum / static_cast<double>(numSimulations - 1);
    const MonteCarloResult result{mean, std::sqrt(variance / static_cast<double>(numSimulations))};
    const auto aggregationEnd = std::chrono::steady_clock::now();
    if (timings) {
        *timings = {
            static_cast<double>(kernelMs),
            std::chrono::duration<double, std::milli>(transferEnd - transferStart).count(),
            std::chrono::duration<double, std::milli>(aggregationEnd - aggregationStart).count()
        };
    }
    return result;
}
