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

namespace CoGaDB {
  namespace query_processing {
    /*! \brief Column-based query processing	*/
    namespace column_processing {
      /*! \brief Column-based query processing on CPU (hardware aware)	*/
      namespace cpu {
        // Column-based query processing on CPU (hardware aware)
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            ColumnPtr>::Map_Init_Function Map_Init_Function;
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            ColumnPtr>::Physical_Operator_Map Physical_Operator_Map;
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            ColumnPtr>::Physical_Operator_Map_Ptr Physical_Operator_Map_Ptr;
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            ColumnPtr>::TypedOperatorPtr TypedOperatorPtr;
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            ColumnPtr>::TypedLogicalNode TypedLogicalNode;
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            ColumnPtr>::TypedNodePtr TypedNodePtr;
        typedef boost::shared_ptr<TypedLogicalNode> TypedLogicalNodePtr;
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            ColumnPtr>::PhysicalQueryPlanPtr PhysicalQueryPlanPtr;
        typedef hype::queryprocessing::LogicalQueryPlan<ColumnPtr>
            LogicalQueryPlan;
        typedef boost::shared_ptr<LogicalQueryPlan> LogicalQueryPlanPtr;

        namespace filter_operation {
          // typedef
          // hype::queryprocessing::OperatorMapper_Helper_Template<PositionListPtr>::TypedOperatorPtr
          // TypedOperatorPtr;
          typedef hype::queryprocessing::OperatorMapper_Helper_Template<
              PositionListPtr>::Map_Init_Function Map_Init_Function;
          typedef hype::queryprocessing::OperatorMapper_Helper_Template<
              PositionListPtr>::Physical_Operator_Map Physical_Operator_Map;
          typedef hype::queryprocessing::OperatorMapper_Helper_Template<
              PositionListPtr>::Physical_Operator_Map_Ptr
              Physical_Operator_Map_Ptr;
          typedef hype::queryprocessing::OperatorMapper_Helper_Template<
              PositionListPtr>::TypedOperatorPtr TypedOperatorPtr;
          typedef hype::queryprocessing::OperatorMapper_Helper_Template<
              PositionListPtr>::TypedLogicalNode TypedLogicalNode;
          typedef hype::queryprocessing::OperatorMapper_Helper_Template<
              PositionListPtr>::PhysicalQueryPlanPtr PhysicalQueryPlanPtr;
          typedef hype::queryprocessing::LogicalQueryPlan<PositionListPtr>
              LogicalQueryPlan;
        }
      }
      /*! \brief Column-based query processing on GPU (hardware aware)	*/
      //			namespace gpu{
      //				//Column-based query processing on
      // GPU(hardware aware)
      //				typedef
      // hype::queryprocessing::OperatorMapper_Helper_Template<CoGaDB::gpu::GPU_Base_ColumnPtr>::Map_Init_Function
      // Map_Init_Function;
      //				typedef
      // hype::queryprocessing::OperatorMapper_Helper_Template<CoGaDB::gpu::GPU_Base_ColumnPtr>::Physical_Operator_Map
      // Physical_Operator_Map;
      //				typedef
      // hype::queryprocessing::OperatorMapper_Helper_Template<CoGaDB::gpu::GPU_Base_ColumnPtr>::Physical_Operator_Map_Ptr
      // Physical_Operator_Map_Ptr;
      //				typedef
      // hype::queryprocessing::OperatorMapper_Helper_Template<CoGaDB::gpu::GPU_Base_ColumnPtr>::TypedOperatorPtr
      // TypedOperatorPtr;
      //				typedef
      // hype::queryprocessing::OperatorMapper_Helper_Template<CoGaDB::gpu::GPU_Base_ColumnPtr>::TypedLogicalNode
      // TypedLogicalNode;
      //				typedef
      // hype::queryprocessing::OperatorMapper_Helper_Template<CoGaDB::gpu::GPU_Base_ColumnPtr>::PhysicalQueryPlanPtr
      // PhysicalQueryPlanPtr;
      //				typedef
      // hype::queryprocessing::LogicalQueryPlan<CoGaDB::gpu::GPU_Base_ColumnPtr>
      // LogicalQueryPlan;
      //			}
    }  // end namespace column_processing
  }    // end namespace query_processing
}  // end namespace CogaDB
