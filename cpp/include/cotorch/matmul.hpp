#pragma once

#include <vector>

using Matrix = std::vector<std::vector<float>>;

Matrix matmul(const Matrix& a, const Matrix& b);
