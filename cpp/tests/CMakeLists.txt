# Each test is a plain executable that exits non-zero on failure and is
# registered with CTest. No test framework yet.
add_executable(toolchain_smoke toolchain_smoke.cpp)
add_test(NAME toolchain_smoke COMMAND toolchain_smoke)

add_executable(test_matmul test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE cotorch_matmul)
add_test(NAME test_matmul COMMAND test_matmul)
