cmake_minimum_required(VERSION 3.18)
project(cuda_monte_carlo_risk_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(risk_engine src/main.cpp src/cpu_pricer.cpp src/black_scholes.cpp)
target_include_directories(risk_engine PRIVATE include)

# Enable CUDA when we reach the GPU implementation milestone.
