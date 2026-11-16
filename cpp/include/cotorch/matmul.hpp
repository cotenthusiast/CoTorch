#pragma once

#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

using Matrix = std::vector<std::vector<float>>;

Matrix matmul(const Matrix &a, const Matrix &b);

std::tuple<Matrix, std::unordered_map<std::string, double>>
benchmark_matmul(const Matrix &a, const Matrix &b);
