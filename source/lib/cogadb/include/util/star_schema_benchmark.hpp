#pragma once
#include <core/table.hpp>
#include <query_processing/query_processor.hpp>
#include <string>
#include <util/time_measurement.hpp>

namespace CoGaDB {

  bool Unittest_Create_Star_Schema_Benchmark_Database(
      const std::string& path_to_files, ClientPtr client);
  bool Unittest_Create_Denormalized_Star_Schema_Benchmark_Database(
      ClientPtr client);  //(const std::string& path_to_files);
  bool optimize_execute_print(const std::string& query_name,
                              query_processing::LogicalQueryPlan& log_plan,
                              ClientPtr client);
  bool optimize_execute_print(const std::string& query_name,
                              query_processing::LogicalQueryPlan& log_plan);

  /********** QUERIES ***********/
  query_processing::LogicalQueryPlanPtr SSB_Q11_plan();
  bool SSB_Q11(ClientPtr client);
  query_processing::LogicalQueryPlanPtr SSB_Q12_plan();
  bool SSB_Q12(ClientPtr client);
  query_processing::LogicalQueryPlanPtr SSB_Q13_plan();
  bool SSB_Q13(ClientPtr client);

  query_processing::LogicalQueryPlanPtr SSB_Q21_plan();
  bool SSB_Q21(ClientPtr client);
  query_processing::LogicalQueryPlanPtr SSB_Q22_plan();
  bool SSB_Q22(ClientPtr client);
  query_processing::LogicalQueryPlanPtr SSB_Q23_plan();
  bool SSB_Q23(ClientPtr client);

  query_processing::LogicalQueryPlanPtr SSB_Q31_plan();
  bool SSB_Q31(ClientPtr client);
  query_processing::LogicalQueryPlanPtr SSB_Q32_plan();
  bool SSB_Q32(ClientPtr client);
  query_processing::LogicalQueryPlanPtr SSB_Q33_plan();
  bool SSB_Q33(ClientPtr client);
  query_processing::LogicalQueryPlanPtr SSB_Q34_plan();
  bool SSB_Q34(ClientPtr client);

  query_processing::LogicalQueryPlanPtr SSB_Q41_plan();
  bool SSB_Q41(ClientPtr client);
  query_processing::LogicalQueryPlanPtr SSB_Q42_plan();
  bool SSB_Q42(ClientPtr client);
  query_processing::LogicalQueryPlanPtr SSB_Q43_plan();
  bool SSB_Q43(ClientPtr client);
  bool SSB_Selection_Query(ClientPtr client);
  bool SSB_SemiJoin_Query(ClientPtr client);

  /********** Denormalized QUERIES ***********/
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q11_plan();
  bool Denormalized_SSB_Q11(ClientPtr client);
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q12_plan();
  bool Denormalized_SSB_Q12(ClientPtr client);
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q13_plan();
  bool Denormalized_SSB_Q13(ClientPtr client);

  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q21_plan();
  bool Denormalized_SSB_Q21(ClientPtr client);
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q22_plan();
  bool Denormalized_SSB_Q22(ClientPtr client);
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q23_plan();
  bool Denormalized_SSB_Q23(ClientPtr client);

  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q31_plan();
  bool Denormalized_SSB_Q31(ClientPtr client);
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q32_plan();
  bool Denormalized_SSB_Q32(ClientPtr client);
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q33_plan();
  bool Denormalized_SSB_Q33(ClientPtr client);
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q34_plan();
  bool Denormalized_SSB_Q34(ClientPtr client);

  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q41_plan();
  bool Denormalized_SSB_Q41(ClientPtr client);
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q42_plan();
  bool Denormalized_SSB_Q42(ClientPtr client);
  query_processing::LogicalQueryPlanPtr Denormalized_SSB_Q43_plan();
  bool Denormalized_SSB_Q43(ClientPtr client);
}  // end namespace CogaDB
