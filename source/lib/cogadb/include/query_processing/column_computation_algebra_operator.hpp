#pragma once

#include <query_processing/definitions.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_ColumnAlgebraOperator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_ColumnAlgebraOperator(const hype::SchedulingDecision& sched_dec,
                                  TypedOperatorPtr child,
                                  const std::string& column1_name,
                                  const std::string& column2_name,
                                  const std::string& result_col_name,
                                  ColumnAlgebraOperation operation)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              column1_name_(column1_name),
              column2_name_(column2_name),
              result_column_name_(result_col_name),
              operation_(operation) {}

        virtual bool execute() {
          hype::ProcessingDeviceID id =
              sched_dec_.getDeviceSpecification().getProcessingDeviceID();
          ProcessorSpecification proc_spec(id);
          AlgebraOperationParam algebra_param(proc_spec, operation_);

          this->result_ = BaseTable::ColumnAlgebraOperation(
              this->getInputData(), column1_name_, column2_name_,
              result_column_name_, algebra_param);

          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~CPU_ColumnAlgebraOperator() {}

       private:
        std::string column1_name_;
        std::string column2_name_;
        std::string result_column_name_;
        CoGaDB::ColumnAlgebraOperation operation_;
      };

      Physical_Operator_Map_Ptr map_init_function_column_algebra_operator();
      TypedOperatorPtr create_CPU_ColumnAlgebra_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_sort_operator;

    // Map_Init_Function
    // init_function_sort_operator=physical_operator::map_init_function_column_algebra_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_ColumnAlgebraOperator
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_column_algebra_operator>  // init_function_sort_operator>
      {
       public:
        Logical_ColumnAlgebraOperator(
            const std::string& column1_name, const std::string& column2_name,
            const std::string& result_col_name,
            ColumnAlgebraOperation operation,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint());
        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const std::string& getColumn1Name();

        const std::string& getColumn2Name();

        const std::string& getResultColumnName();

        CoGaDB::ColumnAlgebraOperation getColumnAlgebraOperation();

        const std::list<std::string> getNamesOfReferencedColumns() const;

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
        std::string column1_name_;
        std::string column2_name_;
        std::string result_column_name_;
        CoGaDB::ColumnAlgebraOperation operation_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
