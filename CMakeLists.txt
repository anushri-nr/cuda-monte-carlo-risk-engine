cmake_minimum_required(VERSION 3.18)
project(cuda_monte_carlo_risk_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(risk_engine src/main.cpp src/cpu_pricer.cpp src/black_scholes.cpp)
add_executable(risk_benchmark src/benchmark.cpp src/cpu_pricer.cpp src/black_scholes.cpp)
foreach(target risk_engine risk_benchmark)
    target_include_directories(${target} PRIVATE include)
endforeach()

option(ENABLE_CUDA "Build the CUDA Monte Carlo implementation" OFF)
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    foreach(target risk_engine risk_benchmark)
        target_sources(${target} PRIVATE src/gpu_pricer.cu)
        target_link_libraries(${target} PRIVATE CUDA::cudart CUDA::curand)
        target_compile_definitions(${target} PRIVATE RISK_ENGINE_HAS_CUDA)
        set_target_properties(${target} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
        )
    endforeach()
endif()
