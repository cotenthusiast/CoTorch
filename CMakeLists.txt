cmake_minimum_required(VERSION 3.20)

project(cotorch LANGUAGES CXX)

add_subdirectory(cpp)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core cpp/bindings.cpp)
target_link_libraries(_core PRIVATE cotorch_matmul)

install(TARGETS _core LIBRARY DESTINATION cotorch)