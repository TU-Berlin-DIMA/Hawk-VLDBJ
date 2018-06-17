#pragma once
#include <core/table.hpp>
#include <parser/client.hpp>
#include <string>
#include <util/time_measurement.hpp>

#include <query_processing/query_processor.hpp>
//
// namespace hype{
//
//    namespace query_processing{
//        template <typename T>
//        class LogicalQueryPlan;
//
//		//Table-based query processing (hardware oblivious)
////		typedef
/// hype::queryprocessing::OperatorMapper_Helper_Template<TablePtr>::Map_Init_Function
/// Map_Init_Function;
////		typedef
/// hype::queryprocessing::OperatorMapper_Helper_Template<TablePtr>::Physical_Operator_Map
/// Physical_Operator_Map;
////		typedef
/// hype::queryprocessing::OperatorMapper_Helper_Template<TablePtr>::Physical_Operator_Map_Ptr
/// Physical_Operator_Map_Ptr;
////		typedef
/// hype::queryprocessing::OperatorMapper_Helper_Template<TablePtr>::TypedOperatorPtr
/// TypedOperatorPtr;
////		typedef
/// hype::queryprocessing::OperatorMapper_Helper_Template<TablePtr>::TypedLogicalNode
/// TypedLogicalNode;
////                typedef
/// hype::queryprocessing::OperatorMapper_Helper_Template<TablePtr>::TypedNodePtr
/// TypedNodePtr;
////                typedef hype::queryprocessing::NodePtr NodePtr;
////		typedef
/// hype::queryprocessing::OperatorMapper_Helper_Template<TablePtr>::PhysicalQueryPlanPtr
/// PhysicalQueryPlanPtr;
//
////                typedef
/// hype::queryprocessing::OperatorMapper_Helper_Template<TablePtr>::LogicalQueryPlan
/// LogicalQueryPlan;
//		typedef
// hype::queryprocessing::LogicalQueryPlan<CoGaDB::TablePtr>
// LogicalQueryPlan;
//                typedef boost::shared_ptr<LogicalQueryPlan>
//                LogicalQueryPlanPtr;
//
////    typedef boost::shared_ptr<LogicalQueryPlan<TablePtr> >
/// LogicalQueryPlanPtr;
//    };
//};

namespace CoGaDB {

  //    class Table;
  //    typedef boost::shared_ptr<Table> TablePtr;

  bool Unittest_Create_TPCH_Database(const std::string& path_to_files,
                                     ClientPtr client);

  bool TPCH_Q1(ClientPtr client);
  bool TPCH_Q2(ClientPtr client);
  bool TPCH_Q3(ClientPtr client);
  bool TPCH_Q4(ClientPtr client);
  bool TPCH_Q5(ClientPtr client);
  bool TPCH_Q6(ClientPtr client);
  bool TPCH_Q7(ClientPtr client);
  bool TPCH_Q9(ClientPtr client);
  bool TPCH_Q10(ClientPtr client);
  bool TPCH_Q15(ClientPtr client);
  bool TPCH_Q17(ClientPtr client);
  bool TPCH_Q18(ClientPtr client);
  bool TPCH_Q20(ClientPtr client);
  bool TPCH_Q21(ClientPtr client);

  //    CoGaDB::query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q2(ClientPtr
  //    client);
  //    CoGaDB::query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q4(ClientPtr
  //    client);
  //    CoGaDB::query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q7(ClientPtr
  //    client);
  //    CoGaDB::query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q9(ClientPtr
  //    client);
  //    CoGaDB::query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q15(ClientPtr
  //    client);
  //    CoGaDB::query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q18(ClientPtr
  //    client);
  //    CoGaDB::query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q20(ClientPtr
  //    client);

  query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q2(ClientPtr client);
  query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q4(ClientPtr client);
  query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q7(ClientPtr client);
  query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q9(ClientPtr client);
  query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q15(ClientPtr client);
  query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q18(ClientPtr client);
  query_processing::LogicalQueryPlanPtr getPlan_TPCH_Q20(ClientPtr client);

  bool TPCH_Q1_hand_compiled_cpu_single_threaded(ClientPtr client);

}  // end namespace CogaDB
