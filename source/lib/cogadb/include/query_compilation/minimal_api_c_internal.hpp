#ifndef MINIMAL_API_C_INTERNAL_HPP
#define MINIMAL_API_C_INTERNAL_HPP

#include <core/table.hpp>

struct C_Table;

namespace CoGaDB {
  TablePtr getTablePtrFromCTable(C_Table*);
  C_Table* getCTableFromTablePtr(TablePtr);
}

#endif  // MINIMAL_API_C_INTERNAL_HPP
