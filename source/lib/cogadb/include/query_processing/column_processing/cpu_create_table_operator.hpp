#pragma once

#include <persistence/storage_manager.hpp>
#include <query_processing/column_processing/definitions.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class column_scan_operator
          : public hype::queryprocessing::NAryOperator<ColumnPtr, ColumnPtr> {
       public:
        typedef column_processing::cpu::TypedOperatorPtr TypedOperatorPtr;
        column_scan_operator(const hype::SchedulingDecision& sched_dec,
                             const std::string& table_name,
                             const std::string& column_name);
        virtual bool execute();
        virtual ~column_scan_operator();

       private:
        std::string table_name_;
        std::string column_name_;
      };

      column_processing::cpu::Physical_Operator_Map_Ptr
      map_init_function_column_scan_operator();
      column_processing::cpu::TypedOperatorPtr create_column_scan_operator(
          column_processing::cpu::TypedLogicalNode& logical_node,
          const hype::SchedulingDecision&,
          column_processing::cpu::TypedOperatorPtr left_child,
          column_processing::cpu::TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_column_scan_operator;

    // Map_Init_Function
    // init_function_column_scan_operator=physical_operator::map_init_function_column_scan_operator;
    // //boost::bind();

    // Map_Init_Function getMap_Init_Function_Scan_Operation();

    namespace logical_operator {

      class Logical_Create_Table
          : public hype::queryprocessing::TypedNode_Impl<
                ColumnPtr,
                physical_operator::
                    map_init_function_column_scan_operator>  // init_function_column_scan_operator>
      // //init_function_column_scan_operator>
      {
       public:
        Logical_Create_Table(const std::string& table_name,
                             const std::string& column_name);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        const std::string& getTableName() const;

        const std::string& getColumnName() const;

        void addChild(OperatorInputType child) { childs_.push_back(child); }

        virtual column_processing::cpu::TypedOperatorPtr getOptimalOperator(
            column_processing::cpu::TypedOperatorPtr left_child,
            column_processing::cpu::TypedOperatorPtr right_child,
            hype::DeviceTypeConstraint dev_constr);

       private:
        std::list<OperatorInputType> childs_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
