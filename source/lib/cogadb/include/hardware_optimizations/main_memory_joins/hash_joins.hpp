

#pragma once

#include <core/base_column.hpp>

namespace CoGaDB {
  namespace CDK {
    namespace main_memory_joins {

      const PositionListPairPtr serial_hash_join(int* build_relation,
                                                 size_t br_size,
                                                 int* probe_relation,
                                                 size_t pr_size);

      // const PositionListPairPtr parallel_hash_join(int* build_relation,
      // size_t br_size, int* probe_relation, size_t pr_size);
      const PositionListPairPtr parallel_hash_join(TID* build_relation,
                                                   size_t br_size,
                                                   TID* probe_relation,
                                                   size_t pr_size);

    }  // end namespace main_memory_joins
  }    // end namespace CDK
}  // end namespace CoGaDB
