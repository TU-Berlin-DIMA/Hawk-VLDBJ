/*
 * File:   query_processing.hpp
 * Author: sebastian
 *
 * Created on 2. Januar 2016, 21:06
 */

#ifndef QUERY_PROCESSING_HPP
#define QUERY_PROCESSING_HPP

#include <parser/client.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_processing/query_processor.hpp>
#include <util/variant_configurator.hpp>
#include <util/variant_measurement.hpp>

namespace CoGaDB {

  class BaseTable;
  typedef boost::shared_ptr<BaseTable> TablePtr;

  const std::pair<TablePtr, VariantMeasurement> executeQueryPlanWithCompiler(
      query_processing::LogicalQueryPlanPtr log_plan, ClientPtr client,
      CodeGeneratorType code_gen_type, const Variant* variant);

  const TablePtr executeQueryPlan(
      query_processing::LogicalQueryPlanPtr log_plan, ClientPtr client);

  bool printResult(TablePtr table, ClientPtr client,
                   double exec_time_in_milliseconds);

}  // end namespace CoGaDB

#endif /* QUERY_PROCESSING_HPP */
