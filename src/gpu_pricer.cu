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
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(std::size_t bytes) {
        if (bytes == 0) return;
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes), "cudaMalloc");
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
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
    // Equal-sized groups have an exact half weight.
    const double weight = a.count == b.count
        ? 0.5
        : static_cast<double>(b.count) / static_cast<double>(count);
    return {count, a.mean + delta * weight,
            a.m2 + b.m2 + delta * delta * static_cast<double>(a.count) * weight};
}

// Shared by both GPU implementations so only aggregation and transfers differ.
__device__ double generatePayoff(
    float spot, float strike, float drift, float diffusion, float discount,
    unsigned long long seed, unsigned long long path
) {
    curandStatePhilox4_32_10_t state;
    curand_init(seed, path, 0, &state);
    const float z = curand_normal(&state);
    const float futurePrice = spot * expf(drift + diffusion * z);
    const float payoff = discount * fmaxf(futurePrice - strike, 0.0f);
    return static_cast<double>(payoff);
}

__global__ void simulatePayoffs(
    float spot, float strike, float drift, float diffusion, float discount,
    unsigned long long seed, long long n, double* payoffs
) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) payoffs[i] = generatePayoff(spot, strike, drift, diffusion, discount, seed, i);
}

// No payoff array: samples enter the shared-memory reduction directly.
__global__ void simulateAndReduce(
    float spot, float strike, float drift, float diffusion, float discount,
    unsigned long long seed, long long n, Moments* partials
) {
    __shared__ Moments shared[reductionThreads];
    const int tid = threadIdx.x;
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + tid;
    Moments value{0, 0.0, 0.0};
    if (i < n) {
        value = {1, generatePayoff(spot, strike, drift, diffusion, discount, seed, i), 0.0};
    }
    shared[tid] = value;
    __syncthreads();
    for (int stride = reductionThreads / 2; stride > 0; stride /= 2) {
        if (tid < stride) shared[tid] = mergeMoments(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) partials[blockIdx.x] = shared[0];
}
} // namespace

MonteCarloResult priceEuropeanCallGPU(
    const OptionParams& params, long long numSimulations, unsigned long long seed,
    GpuTimings* timings, GpuAggregation aggregation
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
        static_cast<unsigned long long>(blocks) >
        std::numeric_limits<std::size_t>::max() / sizeof(Moments)) {
        throw std::invalid_argument("Simulation count exceeds device or allocation limits");
    }

    const bool reduceOnGPU = aggregation == GpuAggregation::GPU;
    if (!reduceOnGPU && static_cast<unsigned long long>(numSimulations) >
        std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw std::invalid_argument("Payoff allocation exceeds addressable memory");
    }
    const auto payoffCount = reduceOnGPU ? 0 : static_cast<std::size_t>(numSimulations);
    const auto payoffBytes = payoffCount * sizeof(double);
    const auto partialBytes = static_cast<std::size_t>(blocks) * sizeof(Moments);
    std::vector<Moments> partials(reduceOnGPU ? static_cast<std::size_t>(blocks) : 0);
    std::vector<double> payoffs(payoffCount);
    DeviceBuffer<double> devicePayoffs(payoffBytes);
    DeviceBuffer<Moments> devicePartials(reduceOnGPU ? partialBytes : 0);
    const double drift = (params.rate - 0.5 * params.volatility * params.volatility) * params.maturity;
    const double diffusion = params.volatility * std::sqrt(params.maturity);
    const double discount = std::exp(-params.rate * params.maturity);

    const float spot = static_cast<float>(params.spot);
    const float strike = static_cast<float>(params.strike);
    const float driftF = static_cast<float>(drift);
    const float diffusionF = static_cast<float>(diffusion);
    const float discountF = static_cast<float>(discount);
    if (!std::isfinite(spot) || spot <= 0 || !std::isfinite(strike) || strike <= 0 ||
        !std::isfinite(driftF) || !std::isfinite(diffusionF) ||
        !std::isfinite(discountF) || discountF <= 0) {
        throw std::invalid_argument("Option parameters exceed GPU single-precision range");
    }

    CudaEvent kernelStart;
    CudaEvent kernelEnd;
    checkCuda(cudaEventRecord(kernelStart.event), "record kernel start");
    if (reduceOnGPU) {
        simulateAndReduce<<<static_cast<unsigned int>(blocks), threads>>>(
            spot, strike, driftF, diffusionF, discountF,
            seed, numSimulations, devicePartials.data);
    } else {
        simulatePayoffs<<<static_cast<unsigned int>(blocks), threads>>>(
            spot, strike, driftF, diffusionF, discountF,
            seed, numSimulations, devicePayoffs.data);
    }
    checkCuda(cudaGetLastError(), "GPU pricing kernel launch");
    checkCuda(cudaEventRecord(kernelEnd.event), "record kernel end");
    checkCuda(cudaEventSynchronize(kernelEnd.event), "GPU pricing kernel execution");
    float kernelMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&kernelMs, kernelStart.event, kernelEnd.event),
              "measure kernel time");

    const auto transferStart = std::chrono::steady_clock::now();
    if (reduceOnGPU) {
        checkCuda(cudaMemcpy(partials.data(), devicePartials.data, partialBytes, cudaMemcpyDeviceToHost),
                  "copy block statistics to host");
    } else {
        checkCuda(cudaMemcpy(payoffs.data(), devicePayoffs.data, payoffBytes, cudaMemcpyDeviceToHost),
                  "copy payoffs to host");
    }
    const auto transferEnd = std::chrono::steady_clock::now();

    const auto aggregationStart = std::chrono::steady_clock::now();
    Moments total{0, 0.0, 0.0};
    if (reduceOnGPU) {
        for (const auto& partial : partials) total = mergeMoments(total, partial);
    } else {
        // Sequential Welford accumulation of the same discounted GPU samples.
        for (double payoff : payoffs) {
            ++total.count;
            const double delta = payoff - total.mean;
            total.mean += delta / static_cast<double>(total.count);
            total.m2 += delta * (payoff - total.mean);
        }
    }
    const double variance = total.m2 / static_cast<double>(numSimulations - 1);
    const MonteCarloResult result{total.mean, std::sqrt(variance / static_cast<double>(numSimulations))};
    if (!std::isfinite(result.price) || !std::isfinite(result.standardError)) {
        throw std::runtime_error("GPU payoff statistics overflowed; reduce parameter magnitudes");
    }
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
