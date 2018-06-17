#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

#ifndef TID
#define TID uint32_t
#endif

void generatePKColumn(TID* data, size_t n) {
  for (size_t i = 0; i < n; i++) {
    data[i] = i;
  }
  std::random_shuffle(&data[0], &data[n]);
}

void generateFKColumn(TID* data, size_t n, size_t num_primary_keys,
                      double match_rate) {
  float r;

  vector<char> matchFlags(num_primary_keys);
  for (size_t i = 0; i < num_primary_keys; i++) {
    r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    if (r < match_rate) {
      matchFlags[i] = 1;
    } else {
      matchFlags[i] = 0;
    }
  }

  TID v;
  for (size_t i = 0; i < n; i++) {
    v = i % num_primary_keys;
    if (matchFlags[v] == 1) {
      // match
      data[i] = v;
    } else {
      // no match
      data[i] = v + num_primary_keys;
    }
  }
}
