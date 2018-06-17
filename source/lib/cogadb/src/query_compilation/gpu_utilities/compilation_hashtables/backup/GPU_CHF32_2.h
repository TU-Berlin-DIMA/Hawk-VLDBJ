#include <stdint.h>
#include "../util/mt19937ar.h"

typedef uint32_t HtUnsigned;

namespace GPU_CHF32_2 {

  const int kNumHashFunctions = 2;

  const uint32_t kEmpty = 0xffffffffu;
  const uint64_t kEntryEmpty = 0xffffffffu;
  const uint32_t kPrimeDivisor = 4294967291u;
  const int kNumStashEntries = 101;

  struct __align__(8) HashConstants {
    uint32_t x;
    uint32_t y;
  };

  void generateHashConstants(HashConstants& constants) {
    unsigned new_a = genrand_int32() % kPrimeDivisor;
    constants.x = (1 > new_a ? 1 : new_a);
    constants.y = genrand_int32() % kPrimeDivisor;
  }

  inline __device__ __host__ uint32_t
  evaluate_hash_function(const HashConstants& constants, const uint32_t key) {
    return ((constants.x ^ key) + constants.y) % kPrimeDivisor;
  }

  typedef struct __align__(4) Entry { HtUnsigned key; }
  Entry;

  __host__ __device__ Entry make_entry(HtUnsigned& K) {
    Entry e;
    e.key = K;
    return e;
  }

  typedef struct {
    // N hash-functions + 1 for stash
    HashConstants constants[kNumHashFunctions + 1];

    // hash table including stash at the end
    Entry* table;
    size_t tableSizeBytes;
    size_t stashSizeBytes;
    size_t numStashEntries;
    int maxInsertIterations;

    // kernel configuration
    unsigned int blockSize;
    unsigned int gridSize;

    int buildAttempts;

    // kernel output information
    // single values (properties)
    int* stashCount;
    int* failures;
    // arrays (statistics)
    int* numProbeIterations;
    int* numBuildIterations;

  } HashTable;

  void clearHashTable(HashTable& ht) {
    cudaMemset(ht.table, 0xff, ht.tableSizeBytes + ht.stashSizeBytes);
    *ht.stashCount = 0;
    *ht.failures = 0;

    for (int i = 0; i <= kNumHashFunctions; i++) {
      generateHashConstants(ht.constants[i]);
    }
  }

  int computeMaxIterations(size_t n, size_t tableSizeNumEntries) {
    float lg_input_size = (float)(log((double)n) / log(2.0));

    // Use an empirical formula for determining what the maximum number of
    // iterations should be.  Works OK in most situations.
    float load_factor = float(n) / tableSizeNumEntries;
    float ln_load_factor = (float)(log(load_factor) / log(2.71828183));

    int max_iterations =
        (int)(4.0 * ceil(-1.0 / (0.028255 + 1.1594772 * ln_load_factor) *
                         lg_input_size));

    return max_iterations;
  }

  HashTable createHashTable(size_t numEntries, double sizeFactor) {
    HashTable ht;
    cudaMallocManaged((void**)&ht, sizeof(HashTable));
    size_t numTableEntries = (double)numEntries * sizeFactor;
    ht.tableSizeBytes = numTableEntries * sizeof(Entry);
    ht.stashSizeBytes = kNumStashEntries * sizeof(Entry);

    ht.blockSize = 64;
    ht.gridSize = 16384;
    ht.buildAttempts = 0;
    ht.numProbeIterations = NULL;
    ht.numBuildIterations = NULL;
    cudaMallocManaged((void**)&(ht.table),
                      ht.tableSizeBytes + ht.stashSizeBytes);
    cudaMemset(ht.table, 0xff, ht.tableSizeBytes + ht.stashSizeBytes);

    cudaMallocManaged((void**)&(ht.stashCount), sizeof(int));
    cudaMallocManaged((void**)&(ht.failures), sizeof(int));
    *(ht.stashCount) = 0;
    *(ht.failures) = 0;

    ht.maxInsertIterations = computeMaxIterations(numEntries, numTableEntries);

    for (int i = 0; i <= kNumHashFunctions; i++) {
      generateHashConstants(ht.constants[i]);
    }
    return ht;
  }

  void freeHashTable(HashTable ht) {
    cudaFree(ht.table);
    cudaFree(ht.stashCount);
    cudaFree(ht.failures);
  }

  void printHashTable(HashTable ht, int numEntriesToPrint = 0) {
    std::cout << "Hash table content" << std::endl;
    for (size_t i = 0;
         i < ht.tableSizeBytes / sizeof(Entry) && i < numEntriesToPrint; i++) {
      Entry e = ht.table[i];
      if (e.key != kEmpty)
        std::cout << "key: " << e.key;
      else
        std::cout << "key: -";
      std::cout << std::endl;
    }
    std::cout << "-------- hash table status --------" << std::endl;
    std::cout << "Hash table size: " << ht.tableSizeBytes / (1000 * 1000)
              << "MB" << std::endl;
    std::cout << "Num stash entries: " << (*ht.stashCount) << std::endl;
    std::cout << "Num failures: " << (*ht.failures) << std::endl;
    std::cout << "-----------------------------------" << std::endl;
  }

  __device__ HtUnsigned determine_next_location(HashTable& ht, HtUnsigned key,
                                                HtUnsigned previous_location) {
    // Identify all possible locations for the entry.
    HtUnsigned locations[kNumHashFunctions];
#pragma unroll
    for (int i = 0; i < kNumHashFunctions; ++i) {
      locations[i] = evaluate_hash_function(ht.constants[i], key) %
                     (ht.tableSizeBytes / sizeof(Entry));
    }

    // Figure out where the item should be inserted next.
    HtUnsigned next_location = locations[0];
#pragma unroll
    for (int i = kNumHashFunctions - 2; i >= 0; --i) {
      next_location = (previous_location == locations[i] ? locations[i + 1]
                                                         : next_location);
    }
    return next_location;
  }

  //! Attempts to insert a single entry into the hash table.
  /*! This process stops after a certain number of iterations.  If the thread is
  still holding onto an item because of an eviction, it tries the stash.
  If it fails to enter the stash, it returns false.
  Otherwise, it succeeds and returns true.
  */
  __device__ void insertEntry(HashTable& ht, HtUnsigned key,
                              int* iterationCount = NULL) {
    if (*ht.failures) return;

    Entry entry = make_entry(key);

    // The key is always inserted into its first slot at the start.
    HtUnsigned location = evaluate_hash_function(ht.constants[0], key) %
                          (ht.tableSizeBytes / sizeof(Entry));

    // Keep inserting until an empty slot is found or the eviction chain grows
    // too large.
    HtUnsigned old_key;
    for (int its = 1; its <= ht.maxInsertIterations; its++) {
      old_key = entry.key;

      *(unsigned int*)&entry = atomicExch((unsigned int*)&ht.table[location],
                                          *(unsigned int*)&entry);

      // If no key was evicted, we're done.
      if (entry.key == kEmpty || entry.key == old_key) {
        if (iterationCount != NULL) *iterationCount = its;
        return;
      }

      // Otherwise, determine where the evicted key will go.
      location = determine_next_location(ht, entry.key, location);
    }

    if (entry.key != kEmpty) {
      // Shove it into the stash.
      HtUnsigned slot =
          evaluate_hash_function(ht.constants[kNumHashFunctions], entry.key) %
          (ht.stashSizeBytes / sizeof(Entry));
      Entry* stash = ht.table + (ht.tableSizeBytes / sizeof(Entry));

      Entry replaced_entry;
      *(unsigned int*)&replaced_entry = atomicCAS(
          (unsigned int*)(stash + slot), kEntryEmpty, *(unsigned int*)&entry);

      if (replaced_entry.key != kEmpty) {
        atomicAdd(ht.failures, 1);
      } else {
        atomicAdd(ht.stashCount, 1);
      }
    }
  }

  __device__ void keyLocations(HashTable& ht, HtUnsigned key,
                               HtUnsigned locations[kNumHashFunctions]) {
// Compute all possible locations for the key in the big table.
#pragma unroll
    for (int i = 0; i < kNumHashFunctions; ++i) {
      locations[i] = evaluate_hash_function(ht.constants[i], key) %
                     (ht.tableSizeBytes / sizeof(Entry));
    }
  }

  __device__ HtUnsigned probeKey(HashTable& ht, HtUnsigned queryKey,
                                 int* num_probes_required = NULL) {
    // Identify all of the locations that the key can be located in.
    HtUnsigned locations[kNumHashFunctions];
    keyLocations(ht, queryKey, locations);

    // Check each location until the key is found.
    int num_probes = 1;
    Entry entry = ht.table[locations[0]];
    HtUnsigned key = entry.key;

#pragma unroll
    for (unsigned i = 1; i < kNumHashFunctions; ++i) {
      if (key != queryKey && key != kEmpty) {
        num_probes++;
        entry = ht.table[locations[i]];
        key = entry.key;
      }
    }

    // Check the stash.
    if ((*ht.stashCount) && entry.key != queryKey) {
      num_probes++;
      const Entry* stash = ht.table + (ht.tableSizeBytes / sizeof(Entry));
      unsigned slot =
          evaluate_hash_function(ht.constants[kNumHashFunctions], key) %
          (ht.stashSizeBytes / sizeof(Entry));
      entry = stash[slot];
    }

    if (num_probes_required) {
      *num_probes_required = num_probes;
    }

    if (entry.key == queryKey) {
      return 1;
    } else {
      return kEmpty;
    }
  }
}
