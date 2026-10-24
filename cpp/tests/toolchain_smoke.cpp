// Verifies the toolchain and the platform assumptions CoTorch relies on.
// Not a numerical test.
#include <cstdio>
#include <limits>

static_assert(__cplusplus >= 202002L, "CoTorch requires C++20");
static_assert(sizeof(float) == 4, "float32 dtype assumes a 4-byte float");
static_assert(sizeof(double) == 8, "float64 dtype assumes an 8-byte double");
static_assert(std::numeric_limits<float>::is_iec559, "float must be IEEE 754");
static_assert(std::numeric_limits<double>::is_iec559, "double must be IEEE 754");

int main() {
  std::printf("toolchain ok: C++ standard %ld\n", static_cast<long>(__cplusplus));
  return 0;
}
