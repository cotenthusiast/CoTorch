#include "cotorch/matmul.hpp"

#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

void validate_matmul_inputs(const Matrix &a, const Matrix &b) {
  // Check for empty matrices or rows
  if (a.empty() || b.empty() || a[0].empty() || b[0].empty()) {
    throw std::invalid_argument("Matrices cannot be empty");
  }
  // Check for irregular matrices
  for (const auto &row : a) {
    if (row.size() != a[0].size())
      throw std::invalid_argument("Matrix A is irregular");
  }
  for (const auto &row : b) {
    if (row.size() != b[0].size())
      throw std::invalid_argument("Matrix B is irregular");
  }
  // Check for matrix dimension mismatch
  if (a[0].size() != b.size()) {
    throw std::invalid_argument(
        "Matrix dimensions aren't suitable for multiplication");
  }
}

Matrix allocate(const Matrix &a, const Matrix &b) {
  Matrix x(a.size(), std::vector<float>(b[0].size(), 0.0f));
  return x;
}

void initialize(Matrix &x) {
  //
}

void multiply(const Matrix &a, const Matrix &b, Matrix &x) {
  for (std::size_t i = 0; i < a.size(); i++) {
    for (std::size_t j = 0; j < b[0].size(); j++) {
      for (std::size_t k = 0; k < a[i].size(); k++) {
        x[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

Matrix matmul(const Matrix &a, const Matrix &b) {
  validate_matmul_inputs(a, b);

  Matrix x = allocate(a, b);
  initialize(x);
  multiply(a, b, x);

  return x;
}

std::tuple<Matrix, std::unordered_map<std::string, double>>
benchmark_matmul(const Matrix &a, const Matrix &b) {
  validate_matmul_inputs(a, b);

  std::unordered_map<std::string, double> times;

  auto start = std::chrono::steady_clock::now();
  Matrix x = allocate(a, b);
  auto end = std::chrono::steady_clock::now();
  times["allocation"] =
      std::chrono::duration<double, std::milli>(end - start).count();

  start = std::chrono::steady_clock::now();
  initialize(x);
  end = std::chrono::steady_clock::now();
  times["initialization"] =
      std::chrono::duration<double, std::milli>(end - start).count();

  start = std::chrono::steady_clock::now();
  multiply(a, b, x);
  end = std::chrono::steady_clock::now();
  times["multiplication"] =
      std::chrono::duration<double, std::milli>(end - start).count();

  return std::tuple(x, times);
}
