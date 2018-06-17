
/*
 * File:   column_grouping_keys.cpp
 * Author: sebastian
 *
 * Created on 21. Dezember 2014, 12:50
 */

#include <bitset>
#include <boost/lexical_cast.hpp>
#include <util/column_grouping_keys.hpp>

//#define THRUST_DEVICE_SYSTEM_CUDA    1
//#define THRUST_DEVICE_SYSTEM_OMP     2
//#define THRUST_DEVICE_SYSTEM_TBB     3
//#define THRUST_DEVICE_SYSTEM_CPP     4
#define THRUST_DEVICE_SYSTEM 3

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#include <thrust/sort.h>
#pragma GCC diagnostic pop

namespace CoGaDB {

size_t ColumnGroupingKeys::getUniqueID() {
  static size_t id = 0;
  return id++;
}

ColumnGroupingKeys::ColumnGroupingKeys(
    const hype::ProcessingDeviceMemoryID& mem_id)
    : keys(new GroupingKeys(std::string("GROUPING_KEYS") +
                                boost::lexical_cast<std::string>(getUniqueID()),
                            OID, mem_id)),
      required_number_of_bits(65) {}

ColumnGroupingKeys::ColumnGroupingKeys(const ColumnGroupingKeys& x)
    //        : keys(boost::dynamic_pointer_cast<GroupingKeys>(copy(x.keys))),
    : keys(boost::dynamic_pointer_cast<GroupingKeys>(
          CoGaDB::copy(x.keys, x.keys->getMemoryID()))),
      required_number_of_bits(x.required_number_of_bits) {
  assert(keys != NULL);
}

ColumnGroupingKeys& ColumnGroupingKeys::operator=(
    const ColumnGroupingKeys& other) {
  // protect against invalid self-assignment
  if (this != &other) {
    this->keys = boost::dynamic_pointer_cast<GroupingKeys>(
        CoGaDB::copy(other.keys, other.keys->getMemoryID()));
    assert(this->keys != NULL);
    this->required_number_of_bits = other.required_number_of_bits;
  }

  return *this;
}

const PositionListPtr ColumnGroupingKeys::sort(
    const ProcessorSpecification& proc_spec) {
  SortParam param(proc_spec, ASCENDING);
  PositionListPtr tids = keys->sort(param, true);
  return tids;
}

hype::ProcessingDeviceMemoryID ColumnGroupingKeys::getMemoryID() const {
  return keys->getMemoryID();
}
ColumnGroupingKeysPtr ColumnGroupingKeys::copy() const {
  return ColumnGroupingKeysPtr(new ColumnGroupingKeys(*this));
}
ColumnGroupingKeysPtr ColumnGroupingKeys::copy(
    const hype::ProcessingDeviceMemoryID& mem_id) const {
  ColumnGroupingKeysPtr col(new ColumnGroupingKeys(mem_id));
  col->keys = boost::dynamic_pointer_cast<GroupingKeys>(
      CoGaDB::copy(this->keys, mem_id));
  if (!col->keys) {
    return ColumnGroupingKeysPtr();
  }
  col->required_number_of_bits = this->required_number_of_bits;
  return col;
}

void ColumnGroupingKeys::print(std::ostream& out) const {
  if (keys->getMemoryID() != hype::PD_Memory_0) {
    ColumnGroupingKeysPtr col = this->copy(hype::PD_Memory_0);
    return col->print(out);
  }
  const GroupingKeysType* values = keys->data();
  for (size_t i = 0; i < keys->size(); ++i) {
    out << std::bitset<64>(values[i]) << std::endl;
  }
}

GroupingKeys::value_type ColumnGroupingKeys::getGreaterPowerOfTwo(
    GroupingKeys::value_type val) {
  GroupingKeys::value_type current_power_of_two = 0;
  while (pow(2, current_power_of_two) <= val) {
    current_power_of_two++;
  }
  return current_power_of_two;
}

}  // end namespace CogaDB
