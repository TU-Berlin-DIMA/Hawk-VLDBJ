/*
 * File:   operator_parameter_types.hpp
 * Author: sebastian
 *
 * Created on 28. Dezember 2014, 22:54
 */

#pragma once

#ifndef OPERATOR_PARAMETER_TYPES_HPP
#define OPERATOR_PARAMETER_TYPES_HPP

#include <core/global_definitions.hpp>
//#include <core/base_column.hpp>
//#include <core/bitmap.hpp>
//#include <lookup_table/join_index.hpp>
#include <hype.hpp>

namespace CoGaDB {

  class ColumnBase;
  typedef boost::shared_ptr<ColumnBase> ColumnPtr;

  class ProcessorSpecification {
   public:
    explicit ProcessorSpecification(const hype::ProcessingDeviceID&);
    hype::ProcessingDeviceID proc_id;
  };

  hype::ProcessingDeviceMemoryID getMemoryID(const ProcessorSpecification&);

  typedef AggregationMethod AggregationFunction;

  class AggregationParam {
   public:
    AggregationParam(const ProcessorSpecification&, const AggregationFunction&,
                     const AggregationAlgorithm&, const std::string&,
                     const bool = false);
    ProcessorSpecification proc_spec;
    AggregationFunction agg_func;
    AggregationAlgorithm agg_alg;
    std::string new_column_name;
    /* Internal flag to optimize has group by
     in case of multiple aggregation functions. */
    bool write_group_tid_array;
  };

  typedef boost::shared_ptr<AggregationParam> AggregationParamPtr;

  typedef std::list<std::string> GroupingColumns;
  typedef std::list<std::pair<std::string, AggregationParam> >
      AggregationFunctions;

  class GroupbyParam {
   public:
    GroupbyParam(const ProcessorSpecification&, const GroupingColumns&,
                 const AggregationFunctions&);

    ProcessorSpecification proc_spec;
    GroupingColumns grouping_columns;
    AggregationFunctions aggregation_functions;
  };

  class AlgebraOperationParam {
   public:
    AlgebraOperationParam(const ProcessorSpecification&,
                          const ColumnAlgebraOperation&);
    ProcessorSpecification proc_spec;
    ColumnAlgebraOperation alg_op;
  };

  class BitmapOperationParam {
   public:
    BitmapOperationParam(const ProcessorSpecification&, const BitmapOperation&);
    ProcessorSpecification proc_spec;
    BitmapOperation bitmap_op;
  };

  struct BitShiftParam {
    BitShiftParam(const ProcessorSpecification&, BitShiftOperation, size_t);
    ProcessorSpecification proc_spec;
    BitShiftOperation op;
    size_t number_of_bits;
  };

  struct BitwiseCombinationParam {
    BitwiseCombinationParam(const ProcessorSpecification&,
                            BitwiseCombinationOperation);
    ProcessorSpecification proc_spec;
    BitwiseCombinationOperation op;
  };

  class FetchJoinParam {
   public:
    FetchJoinParam(const ProcessorSpecification&);
    ProcessorSpecification proc_spec;
  };

  class GatherParam {
   public:
    GatherParam(const ProcessorSpecification&);
    ProcessorSpecification proc_spec;
  };

  class JoinParam {
   public:
    JoinParam(const ProcessorSpecification&, const JoinAlgorithm&,
              const JoinType& _join_type = INNER_JOIN);
    ProcessorSpecification proc_spec;
    JoinAlgorithm join_alg;
    JoinType join_type;
  };

  class SelectionParam {
   public:
    SelectionParam(const ProcessorSpecification&, const PredicateType&,
                   const boost::any&, const ValueComparator&);
    SelectionParam(const ProcessorSpecification&, const PredicateType&,
                   ColumnPtr, const ValueComparator&);
    ProcessorSpecification proc_spec;
    PredicateType pred_type;
    boost::any value;
    ColumnPtr comparison_column;
    ValueComparator comp;
  };

  class SetOperationParam {
   public:
    SetOperationParam(const ProcessorSpecification&, const SetOperation&);
    ProcessorSpecification proc_spec;
    SetOperation set_op;
  };

  class SortParam {
   public:
    SortParam(const ProcessorSpecification&, const SortOrder&,
              const bool& stable_sort = false);
    ProcessorSpecification proc_spec;
    SortOrder order;
    bool stable;
  };

}  // end namespace CoGaDB

#endif /* OPERATOR_PARAMETER_TYPES_HPP */
