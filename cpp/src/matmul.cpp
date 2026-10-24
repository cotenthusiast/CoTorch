#include "cotorch/matmul.hpp"

#include <cstddef>
#include <vector>
#include <stdexcept>


void validate_matmul_inputs(const Matrix& a, const Matrix& b)
{
    // Check for empty matrices or rows
    if (a.empty() || b.empty() || a[0].empty() || b[0].empty())
    {
        throw std::invalid_argument("Matrices cannot be empty");
    }
    // Check for irregular matrices
    for (const auto& row : a)
    {
        if (row.size() != a[0].size())
            throw std::invalid_argument("Matrix A is irregular");
    }
    for (const auto& row : b)
    {
        if (row.size() != b[0].size())
            throw std::invalid_argument("Matrix B is irregular");
    }
    // Check for matrix dimension mismatch
    if (a[0].size() != b.size())
    {
        throw std::invalid_argument("Matrix dimensions aren't suitable for multiplication");
    }
}

Matrix matmul(const Matrix& a, const Matrix& b)
{
    validate_matmul_inputs(a, b);

    Matrix x(a.size(), std::vector<float>(b[0].size(), 0.0f));

    for (std::size_t  i = 0; i < a.size(); i ++)
    {
        for (std::size_t j = 0; j < b[0].size(); j++)
        {
            for (std::size_t k = 0; k < a[i].size(); k ++)
            {
                x[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return x;
}