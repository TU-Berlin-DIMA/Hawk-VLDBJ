#include "test_utils.hpp"
#include <climits>
#include <cmath>

namespace perseus {

double unique(double min, double max) {
  return std::uniform_real_distribution<double>(min, max)(global_rnd);
}

size_t aPowerOfTwo(const size_t min) { return aPowerOfTwo(min, LONG_MAX); }

size_t aPowerOfTwo(const size_t min, const size_t max) {
  auto maxExponent = std::log2(max) - std::log2(min);
  auto exponent = unique<unsigned char>(0, maxExponent);
  size_t powerOfTwo = static_cast<size_t>(std::pow(2, exponent));
  return min * powerOfTwo;
}
}
