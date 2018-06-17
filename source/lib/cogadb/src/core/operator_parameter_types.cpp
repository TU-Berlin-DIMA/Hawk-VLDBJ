/*
 * File:   operator_parameter_types.cpp
 * Author: sebastian
 *
 * Created on 28. Dezember 2014, 22:56
 */

#include <core/operator_parameter_types.hpp>
#include <util/utility_functions.hpp>

namespace CoGaDB {

ProcessorSpecification::ProcessorSpecification(
    const hype::ProcessingDeviceID& _proc_id)
    : proc_id(_proc_id) {}

hype::ProcessingDeviceMemoryID getMemoryID(
    const ProcessorSpecification& proc_spec) {
  return hype::util::getMemoryID(proc_spec.proc_id);
}

// typedef AggregationMethod AggregationFunction;

AggregationParam::AggregationParam(const ProcessorSpecification& _proc_spec,
                                   const AggregationFunction& _agg_func,
                                   const AggregationAlgorithm& _agg_alg,
                                   const std::string& _new_column_name,
                                   const bool _write_group_tid_array)
    : proc_spec(_proc_spec),
      agg_func(_agg_func),
      agg_alg(_agg_alg),
      new_column_name(_new_column_name),
      write_group_tid_array(_write_group_tid_array) {}

GroupbyParam::GroupbyParam(const ProcessorSpecification& _proc_spec,
                           const GroupingColumns& _grouping_columns,
                           const AggregationFunctions& _aggregation_functions)
    : proc_spec(_proc_spec),
      grouping_columns(_grouping_columns),
      aggregation_functions(_aggregation_functions) {}

AlgebraOperationParam::AlgebraOperationParam(
    const ProcessorSpecification& _proc_spec,
    const ColumnAlgebraOperation& _alg_op)
    : proc_spec(_proc_spec), alg_op(_alg_op) {}

BitmapOperationParam::BitmapOperationParam(
    const ProcessorSpecification& _proc_spec, const BitmapOperation& _bitmap_op)
    : proc_spec(_proc_spec), bitmap_op(_bitmap_op) {}

BitShiftParam::BitShiftParam(const ProcessorSpecification& _proc_spec,
                             BitShiftOperation _op, size_t _number_of_bits)
    : proc_spec(_proc_spec), op(_op), number_of_bits(_number_of_bits) {}

BitwiseCombinationParam::BitwiseCombinationParam(
    const ProcessorSpecification& _proc_spec, BitwiseCombinationOperation _op)
    : proc_spec(_proc_spec), op(_op) {}

FetchJoinParam::FetchJoinParam(const ProcessorSpecification& _proc_spec)
    : proc_spec(_proc_spec) {}

GatherParam::GatherParam(const ProcessorSpecification& _proc_spec)
    : proc_spec(_proc_spec) {}

JoinParam::JoinParam(const ProcessorSpecification& _proc_spec,
                     const JoinAlgorithm& _join_alg, const JoinType& _join_type)
    : proc_spec(_proc_spec), join_alg(_join_alg), join_type(_join_type) {}

SelectionParam::SelectionParam(const ProcessorSpecification& _proc_spec,
                               const PredicateType& _pred_type,
                               const boost::any& _value,
                               const ValueComparator& _comp)
    : proc_spec(_proc_spec),
      pred_type(_pred_type),
      value(_value),
      comparison_column(),
      comp(_comp) {}

SelectionParam::SelectionParam(const ProcessorSpecification& _proc_spec,
                               const PredicateType& _pred_type, ColumnPtr col,
                               const ValueComparator& _comp)
    : proc_spec(_proc_spec),
      pred_type(_pred_type),
      value(),
      comparison_column(col),
      comp(_comp) {}

SetOperationParam::SetOperationParam(const ProcessorSpecification& _proc_spec,
                                     const SetOperation& _set_op)
    : proc_spec(_proc_spec), set_op(_set_op) {}

SortParam::SortParam(const ProcessorSpecification& _proc_spec,
                     const SortOrder& _order, const bool& _stable)
    : proc_spec(_proc_spec), order(_order), stable(_stable) {}

}  // end namespace CogaDB
