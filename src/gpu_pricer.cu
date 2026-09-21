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
template <typename T>
struct DevicePayoffs {
    T* data = nullptr;
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

constexpr int reductionThreads = 256;

struct Moments {
    long long count;
    double mean;
    double m2; // Sum of squared deviations from the mean.
};

__host__ __device__ Moments mergeMoments(Moments a, Moments b) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    const long long count = a.count + b.count;
    const double delta = b.mean - a.mean;
    const double weight = static_cast<double>(b.count) / static_cast<double>(count);
    return {count, a.mean + delta * weight,
            a.m2 + b.m2 + delta * delta * static_cast<double>(a.count) * weight};
}

__global__ void reducePayoffs(const double* payoffs, long long n, Moments* partials) {
    __shared__ Moments shared[reductionThreads];
    const int tid = threadIdx.x;
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + tid;
    // Inactive lanes contribute an empty summary but still reach every barrier.
    shared[tid] = i < n ? Moments{1, payoffs[i], 0.0} : Moments{0, 0.0, 0.0};
    __syncthreads();
    for (int stride = reductionThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = mergeMoments(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = shared[0];
}

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

    constexpr int threads = reductionThreads;
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
    const auto partialBytes = static_cast<std::size_t>(blocks) * sizeof(Moments);
    std::vector<Moments> partials(static_cast<std::size_t>(blocks));
    DevicePayoffs<double> devicePayoffs(bytes);
    DevicePayoffs<Moments> devicePartials(partialBytes);
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

    CudaEvent reductionStart;
    CudaEvent reductionEnd;
    checkCuda(cudaEventRecord(reductionStart.event), "record reduction start");
    reducePayoffs<<<static_cast<unsigned int>(blocks), threads>>>(
        devicePayoffs.data, numSimulations, devicePartials.data);
    checkCuda(cudaGetLastError(), "reducePayoffs launch");
    checkCuda(cudaEventRecord(reductionEnd.event), "record reduction end");
    checkCuda(cudaEventSynchronize(reductionEnd.event), "reducePayoffs execution");
    float reductionMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&reductionMs, reductionStart.event, reductionEnd.event),
              "measure reduction time");

    const auto transferStart = std::chrono::steady_clock::now();
    checkCuda(cudaMemcpy(partials.data(), devicePartials.data, partialBytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy block summaries to host");
    const auto transferEnd = std::chrono::steady_clock::now();

    const auto aggregationStart = std::chrono::steady_clock::now();
    Moments total{0, 0.0, 0.0};
    for (const auto& partial : partials) {
        total = mergeMoments(total, partial);
    }
    const double variance = total.m2 / static_cast<double>(numSimulations - 1);
    const MonteCarloResult result{total.mean, std::sqrt(variance / static_cast<double>(numSimulations))};
    const auto aggregationEnd = std::chrono::steady_clock::now();
    if (timings) {
        *timings = {
            static_cast<double>(kernelMs),
            static_cast<double>(reductionMs),
            std::chrono::duration<double, std::milli>(transferEnd - transferStart).count(),
            std::chrono::duration<double, std::milli>(aggregationEnd - aggregationStart).count()
        };
    }
    return result;
}
