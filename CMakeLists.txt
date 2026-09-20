cmake_minimum_required(VERSION 3.18)
project(cuda_monte_carlo_risk_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(risk_engine src/main.cpp src/cpu_pricer.cpp src/black_scholes.cpp)
target_include_directories(risk_engine PRIVATE include)

option(ENABLE_CUDA "Build the CUDA Monte Carlo implementation" OFF)
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(risk_engine PRIVATE src/gpu_pricer.cu)
    target_link_libraries(risk_engine PRIVATE CUDA::cudart CUDA::curand)
    target_compile_definitions(risk_engine PRIVATE RISK_ENGINE_HAS_CUDA)
    set_target_properties(risk_engine PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
endif()
