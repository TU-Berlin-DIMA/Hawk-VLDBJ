#pragma once

#include <query_processing/chain_join_operator.hpp>
#include <query_processing/complex_selection_operator.hpp>
#include <query_processing/cross_join_operator.hpp>
#include <query_processing/fetch_join_operator.hpp>
#include <query_processing/generate_constant_column_operator.hpp>
#include <query_processing/groupby_operator.hpp>
#include <query_processing/indexed_tuple_reconstruction_operator.hpp>
#include <query_processing/invisible_join_operator.hpp>
#include <query_processing/join_operator.hpp>
#include <query_processing/logical_query_plan.hpp>
#include <query_processing/pk_fk_join_operator.hpp>
#include <query_processing/projection_operator.hpp>
#include <query_processing/rename_operator.hpp>
#include <query_processing/scan_operator.hpp>
#include <query_processing/selection_operator.hpp>
#include <query_processing/sort_operator.hpp>
#include <query_processing/udf_operator.hpp>

#include <query_processing/column_computation_algebra_operator.hpp>
#include <query_processing/column_computation_constant_operator.hpp>

#include <query_processing/column_processing/column_comparator.hpp>
#include <query_processing/column_processing/cpu_algebra_operator.hpp>
#include <query_processing/column_processing/cpu_column_constant_filter_operator.hpp>
#include <query_processing/column_processing/cpu_columnscan_operator.hpp>
#include <query_processing/column_processing/positionlist_operator.hpp>
// newe column based operators
#include <query_processing/column_processing/bitmap_operator.hpp>
#include <query_processing/column_processing/column_bitmap_fetch_join_operator.hpp>
#include <query_processing/column_processing/column_bitmap_selection_operator.hpp>
#include <query_processing/column_processing/column_convert_bitmap_to_positionlist.hpp>
#include <query_processing/column_processing/column_convert_bitmap_to_positionlist.hpp>
#include <query_processing/column_processing/column_convert_positionlist_to_bitmap.hpp>
#include <query_processing/column_processing/column_fetch_join_operator.hpp>

#include <parser/client.hpp>
#include <statistics/statistics_manager.hpp>

namespace CoGaDB {
  namespace query_processing {

    CoGaDB::query_processing::PhysicalQueryPlanPtr optimize_and_execute(
        const std::string& query_name, LogicalQueryPlan& log_plan,
        ClientPtr client);

    query_processing::column_processing::cpu::LogicalQueryPlanPtr
    createColumnPlanforDisjunction(TablePtr table,
                                   const Disjunction& disjunction,
                                   hype::DeviceConstraint dev_constr);

    const query_processing::column_processing::cpu::LogicalQueryPlanPtr
    createColumnBasedQueryPlan(
        TablePtr table, const KNF_Selection_Expression& knf_expr,
        hype::DeviceConstraint dev_constr = hype::DeviceConstraint());

    TablePtr two_phase_physical_optimization_selection(
        TablePtr table, const KNF_Selection_Expression&,
        hype::DeviceConstraint dev_constr = hype::DeviceConstraint(),
        MaterializationStatus mat_stat = MATERIALIZE,
        ParallelizationMode comp_mode = SERIAL, std::ostream* out = &std::cout);

  }  // end namespace query_processing
}  // end namespace CogaDB
