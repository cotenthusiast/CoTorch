// Tests for cotorch::matmul. Plain executable, no framework: exit non-zero on failure.
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <stdexcept>

#include "cotorch/matmul.hpp"

// Floats are not exact, so compare with a small tolerance instead of ==.
static const float rTol = 1e-5f;
static const float aTol = 1e-6f;

// Returns true if both matrices have the same shape and every element matches within tolerance.
static bool matrices_equal(const Matrix& actual, const Matrix& expected)
{
  if (actual.size() != expected.size())
    return false;

  for (std::size_t i = 0; i < actual.size(); i++)
  {
    if (actual[i].size() != expected[i].size())
      return false;

    for (std::size_t j = 0; j < actual[i].size(); j++)
    {
      const float a = actual[i][j];
      const float e = expected[i][j];

      if (std::isnan(a) || std::isnan(e))
        return false;

      if (std::isinf(a) || std::isinf(e))
      {
        if (a != e)
          return false;
      }
      else if (std::fabs(a - e) > aTol + rTol * std::fabs(e))
      {
        return false;
      }
    }
  }

  return true;
}

// Returns true if matmul(a, b) throws std::invalid_argument.
static bool throws_invalid_argument(const Matrix& a, const Matrix& b)
{
  try
  {
    matmul(a, b);
  }
  catch (const std::invalid_argument&)
  {
    return true;
  }
  return false;
}

bool test_dimensions()
{
  // Check the output shape
  // (2x3) * (3x4) -> (2x4)
  Matrix a(2, std::vector<float>(3, 1.0f));
  Matrix b(3, std::vector<float>(4, 1.0f));
  Matrix c = matmul(a, b);
  if (c.size() != 2)
    return false;
  for (const auto& row : c)
  {
    if (row.size() != 4)
      return false;
  }

  // (1x3) * (3x1) -> (1x1)
  Matrix d = matmul(Matrix(1, std::vector<float>(3, 1.0f)),
                    Matrix(3, std::vector<float>(1, 1.0f)));
  if (d.size() != 1 || d[0].size() != 1)
    return false;

  // (3x1) * (1x2) -> (3x2)
  Matrix e = matmul(Matrix(3, std::vector<float>(1, 1.0f)),
                    Matrix(1, std::vector<float>(2, 1.0f)));
  if (e.size() != 3 || e[0].size() != 2)
    return false;

  return true;
}

bool test_correctness()
{
  // Compare with a manually calculated expected matrix

  // Ordinary: (2x3) * (3x2)
  // [1 2 3]   [ 7  8]   [1*7+2*9+3*11, 1*8+2*10+3*12]   [ 58  64]
  // [4 5 6] * [ 9 10] = [4*7+5*9+6*11, 4*8+5*10+6*12] = [139 154]
  //           [11 12]
  Matrix a = {{1, 2, 3}, {4, 5, 6}};
  Matrix b = {{7, 8}, {9, 10}, {11, 12}};
  Matrix expected = {{58, 64}, {139, 154}};
  if (!matrices_equal(matmul(a, b), expected))
    return false;

  // Identity: A * I = A and I * A = A
  Matrix m = {{1.5f, -2.0f}, {3.0f, 4.25f}};
  Matrix identity = {{1, 0}, {0, 1}};
  if (!matrices_equal(matmul(m, identity), m))
    return false;
  if (!matrices_equal(matmul(identity, m), m))
    return false;

  // Zero: A * 0 = 0
  Matrix zero = {{0, 0}, {0, 0}};
  if (!matrices_equal(matmul(m, zero), zero))
    return false;

  // Fractional values: [0.1 0.2] * [0.3; 0.4] = [0.1*0.3 + 0.2*0.4] = [0.11]
  Matrix f1 = {{0.1f, 0.2f}};
  Matrix f2 = {{0.3f}, {0.4f}};
  Matrix f_expected = {{0.11f}};
  if (!matrices_equal(matmul(f1, f2), f_expected))
    return false;

  return true;
}

bool test_invalid_dimensions()
{
  // Verify that invalid inputs throw
  Matrix a = {{1, 2, 3}, {4, 5, 6}};  // 2x3

  // Columns of A (3) do not match rows of B (2)
  Matrix b = {{1, 2}, {3, 4}};  // 2x2
  if (!throws_invalid_argument(a, b))
    return false;

  // (2x3) * (2x3) is also incompatible
  if (!throws_invalid_argument(a, a))
    return false;

  return true;
}

bool test_empty_matrices()
{
  // Verify that empty matrices throw
  Matrix valid = {{1, 2}, {3, 4}};
  Matrix no_rows;
  Matrix no_columns = {{}};

  if (!throws_invalid_argument(no_rows, valid))
    return false;
  if (!throws_invalid_argument(valid, no_rows))
    return false;
  if (!throws_invalid_argument(no_columns, valid))
    return false;
  if (!throws_invalid_argument(valid, no_columns))
    return false;
  if (!throws_invalid_argument(no_rows, no_rows))
    return false;

  return true;
}

bool test_irregular_matrices()
{
  // Verify that irregular matrices throw
  Matrix valid = {{1, 2}, {3, 4}};
  Matrix irregular = {{1, 2}, {3}};  // second row is shorter

  // Irregular first operand
  if (!throws_invalid_argument(irregular, valid))
    return false;

  // Irregular second operand
  if (!throws_invalid_argument(valid, irregular))
    return false;

  // Irregular in a later row, with a longer row instead of a shorter one
  Matrix longer = {{1, 2}, {3, 4}, {5, 6, 7}};
  if (!throws_invalid_argument(longer, valid))
    return false;
  if (!throws_invalid_argument(valid, longer))
    return false;

  return true;
}

int main() {
  // Run the individual tests
  struct Test
  {
    const char* name;
    bool (*run)();
  };
  const Test tests[] = {
    {"test_dimensions", test_dimensions},
    {"test_correctness", test_correctness},
    {"test_invalid_dimensions", test_invalid_dimensions},
    {"test_empty_matrices", test_empty_matrices},
    {"test_irregular_matrices", test_irregular_matrices},
  };

  int failures = 0;
  for (const Test& test : tests)
  {
    bool passed = false;
    try
    {
      passed = test.run();
    }
    catch (const std::exception& e)
    {
      std::fprintf(stderr, "%s threw unexpectedly: %s\n", test.name, e.what());
    }
    std::printf("%s: %s\n", test.name, passed ? "PASS" : "FAIL");
    if (!passed)
      failures++;
  }

  // Return 1 if any failed
  // Otherwise return 0
  return failures == 0 ? 0 : 1;
}
