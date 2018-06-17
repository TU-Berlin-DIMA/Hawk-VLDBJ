#ifndef PERSEUS_TEST_UTILS_HPP
#define PERSEUS_TEST_UTILS_HPP

#include <algorithm>
#include <random>

namespace perseus {

  extern std::mt19937 global_rnd;

  template <typename T>
  T unique(T min, T max) {
    return std::uniform_int_distribution<T>(min, max)(global_rnd);
  }

  template <typename T>
  T unique(T min, T max, std::vector<T> redo) {
    while (true) {
      T value = std::uniform_int_distribution<T>(min, max)(global_rnd);
      bool unique = std::find(redo.begin(), redo.end(), value) == redo.end();
      if (unique) {
        return value;
      }
    }
  }

  double unique(double min, double max);

  size_t aPowerOfTwo(const size_t min = 1);

  size_t aPowerOfTwo(const size_t min, const size_t max);

  // All OpenCL types are pointers. These need to be initialized when used
  // in a test. An option is to initialize with nullptr but this can make
  // test succeed when they should fail. Since OpenCL functions are mocked,
  // their actual values do not matter.
  template <typename T>
  T uniqueOpenCLObject() {
    return reinterpret_cast<T>(unique<unsigned int>(1, 1000));
  }
}

#endif  // PERSEUS_TEST_UTILS_HPP
