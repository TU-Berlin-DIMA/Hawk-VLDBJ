
#include <core/vector.hpp>

namespace CoGaDB {

Vector::Vector(const std::string& name, AttributeType db_type,
               ColumnType column_type, TID begin_index, size_t num_elements)
    : ColumnBase(name, db_type, column_type),
      begin_index_(begin_index),
      num_elements_(num_elements) {}

const VectorPtr Vector::getVector(ColumnPtr col, BlockIteratorPtr it) const {
  return VectorPtr();
}

Vector::~Vector() {}

}  // end namespace CoGaDB
