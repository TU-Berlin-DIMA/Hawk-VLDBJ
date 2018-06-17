#pragma once

#include <persistence/storage_manager.hpp>
#include <query_processing/column_processing/definitions.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class column_scan_operator
          : public hype::queryprocessing::UnaryOperator<ColumnPtr, ColumnPtr> {
       public:
        typedef column_processing::cpu::TypedOperatorPtr TypedOperatorPtr;
        column_scan_operator(const hype::SchedulingDecision& sched_dec,
                             const std::string& table_name,
                             const std::string& column_name);
        column_scan_operator(const hype::SchedulingDecision& sched_dec,
                             TablePtr table_ptr,
                             const std::string& column_name);
        virtual bool execute();
        virtual void releaseInputData();
        virtual ~column_scan_operator();

       private:
        std::string table_name_;
        std::string column_name_;
        TablePtr table_ptr_;
      };

      column_processing::cpu::Physical_Operator_Map_Ptr
      map_init_function_column_scan_operator();
      column_processing::cpu::TypedOperatorPtr create_column_scan_operator(
          column_processing::cpu::TypedLogicalNode& logical_node,
          const hype::SchedulingDecision&,
          column_processing::cpu::TypedOperatorPtr left_child,
          column_processing::cpu::TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_Column_Scan
          : public hype::queryprocessing::TypedNode_Impl<
                ColumnPtr,
                physical_operator::
                    map_init_function_column_scan_operator>  // init_function_column_scan_operator>
      // //init_function_column_scan_operator>
      {
       public:
        Logical_Column_Scan(const std::string& table_name,
                            const std::string& column_name);
        Logical_Column_Scan(TablePtr table, const std::string& column_name);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        const std::string& getTableName() const;

        const TablePtr getTablePtr();

        const std::string& getColumnName() const;

        std::string toString(bool verbose) const;

        // virtual column_processing::cpu::TypedOperatorPtr
        // getOptimalOperator(column_processing::cpu::TypedOperatorPtr
        // left_child, column_processing::cpu::TypedOperatorPtr right_child,
        // hype::DeviceTypeConstraint dev_constr);
        virtual const hype::Tuple getFeatureVector() const;
        virtual hype::query_optimization::QEP_Node* toQEP_Node();

       private:
        std::string table_name_;
        std::string column_name_;
        TablePtr table_ptr_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
