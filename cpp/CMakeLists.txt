# Standalone native build. It does not depend on Python, so kernels can be
# tested and benchmarked on their own before any bindings exist.
#
#   cmake -S cpp -B cpp/build
#   cmake --build cpp/build
#   ctest --test-dir cpp/build --output-on-failure
cmake_minimum_required(VERSION 3.20)
project(cotorch_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release: benchmarking an unoptimised build is meaningless.
# Release defines NDEBUG, which disables assert(), so tests must not rely on it.
# -march=native is deliberately absent: choose SIMD flags yourself when you get there.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Lets `ctest` run the executables registered in tests/.
enable_testing()
add_subdirectory(tests)

add_library(cotorch_matmul STATIC src/matmul.cpp)

target_include_directories(cotorch_matmul PUBLIC include)
