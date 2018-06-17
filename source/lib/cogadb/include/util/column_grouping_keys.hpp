/*
 * File:   column_grouping_keys.hpp
 * Author: sebastian
 *
 * Created on 21. Dezember 2014, 12:41
 */

#pragma once

#ifndef COLUMN_GROUPING_KEYS_HPP
#define COLUMN_GROUPING_KEYS_HPP

#include <core/column.hpp>

namespace CoGaDB {

  typedef Column<TID> GroupingKeys;
  typedef boost::shared_ptr<GroupingKeys> GroupingKeysPtr;

  class ColumnGroupingKeys {
   public:
    typedef GroupingKeys::value_type GroupingKeysType;
    typedef boost::shared_ptr<ColumnGroupingKeys> ColumnGroupingKeysPtr;
    static size_t getUniqueID();

    explicit ColumnGroupingKeys(const hype::ProcessingDeviceMemoryID& mem_id);
    ColumnGroupingKeys(const ColumnGroupingKeys&);
    ColumnGroupingKeys& operator=(const ColumnGroupingKeys&);

    const PositionListPtr sort(const ProcessorSpecification& proc_spec);
    hype::ProcessingDeviceMemoryID getMemoryID() const;
    ColumnGroupingKeysPtr copy() const;
    ColumnGroupingKeysPtr copy(const hype::ProcessingDeviceMemoryID&) const;

    void print(std::ostream& out) const;

    static GroupingKeysType getGreaterPowerOfTwo(GroupingKeysType val);

    GroupingKeysPtr keys;
    size_t required_number_of_bits;
  };

  typedef ColumnGroupingKeys::ColumnGroupingKeysPtr ColumnGroupingKeysPtr;

}  // end namespace CogaDB

#endif /* COLUMN_GROUPING_KEYS_HPP */
