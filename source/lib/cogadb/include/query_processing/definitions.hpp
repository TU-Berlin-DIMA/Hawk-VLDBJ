#pragma once

#include <algorithm>
#include <boost/bind.hpp>
#include <boost/date_time.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/function.hpp>
#include <cstdlib>
#include <functional>
#include <hype.hpp>
#include <query_processing/logical_query_plan.hpp>
#include <query_processing/processing_device.hpp>
#include <stack>
// CoGaDB
#include <core/base_table.hpp>
#include <core/runtime_configuration.hpp>

namespace CoGaDB {
  namespace query_processing {
    // Table-based query processing (hardware oblivious)
    typedef hype::queryprocessing::OperatorMapper_Helper_Template<
        TablePtr>::Map_Init_Function Map_Init_Function;
    typedef hype::queryprocessing::OperatorMapper_Helper_Template<
        TablePtr>::Physical_Operator_Map Physical_Operator_Map;
    typedef hype::queryprocessing::OperatorMapper_Helper_Template<
        TablePtr>::Physical_Operator_Map_Ptr Physical_Operator_Map_Ptr;
    typedef hype::queryprocessing::OperatorMapper_Helper_Template<
        TablePtr>::TypedOperatorPtr TypedOperatorPtr;
    typedef hype::queryprocessing::OperatorMapper_Helper_Template<
        TablePtr>::TypedLogicalNode TypedLogicalNode;
    typedef hype::queryprocessing::OperatorMapper_Helper_Template<
        TablePtr>::TypedNodePtr TypedNodePtr;
    typedef hype::queryprocessing::NodePtr NodePtr;
    typedef hype::queryprocessing::OperatorMapper_Helper_Template<
        TablePtr>::PhysicalQueryPlanPtr PhysicalQueryPlanPtr;
    typedef hype::queryprocessing::LogicalQueryPlan<TablePtr> LogicalQueryPlan;
    typedef boost::shared_ptr<LogicalQueryPlan> LogicalQueryPlanPtr;

  }  // end namespace query_processing
}  // end namespace CogaDB
