
#pragma once

#include <string>

#include <core/global_definitions.hpp>

namespace CoGaDB {

  namespace util {

    const std::string getName(AttributeType);
    const std::string getName(ColumnType);
    const std::string getName(ComputeDevice);
    const std::string getName(AggregationMethod);
    const std::string getName(AggregationAlgorithm x);
    const std::string getName(ValueComparator);
    const std::string getName(SortOrder);
    const std::string getName(Operation);
    const std::string getName(JoinAlgorithm);
    const std::string getName(JoinType);
    const std::string getName(MaterializationStatus);
    const std::string getName(ParallelizationMode);
    const std::string getName(ColumnAlgebraOperation);
    const std::string getName(PositionListOperation);
    const std::string getName(BitmapOperation x);
    const std::string getName(GPUBufferManagementStrategy x);
    const std::string getName(TableLoaderMode x);
    const std::string getName(ColumnType x);
    const std::string getName(LogicalOperation x);
    const std::string getName(SetOperation x);
  }

}  // end namespace CogaDB
