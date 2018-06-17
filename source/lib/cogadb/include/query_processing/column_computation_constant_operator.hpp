#pragma once

#include <query_processing/definitions.hpp>
#include <sstream>
#include <util/getname.hpp>
#include <util/iostream.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_ColumnConstantOperator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_ColumnConstantOperator(const hype::SchedulingDecision& sched_dec,
                                   TypedOperatorPtr child,
                                   std::string column_name,
                                   const boost::any& value,
                                   const std::string& result_col_name,
                                   ColumnAlgebraOperation operation)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              column_name_(column_name),
              value_(value),
              result_column_name_(result_col_name),
              operation_(operation) {}

        virtual bool execute() {
          hype::ProcessingDeviceID id =
              sched_dec_.getDeviceSpecification().getProcessingDeviceID();
          ProcessorSpecification proc_spec(id);
          AlgebraOperationParam algebra_param(proc_spec, operation_);

          this->result_ = BaseTable::ColumnConstantOperation(
              this->getInputData(), column_name_, value_, result_column_name_,
              algebra_param);

          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~CPU_ColumnConstantOperator() {}

       private:
        std::string column_name_;
        boost::any value_;
        std::string result_column_name_;
        CoGaDB::ColumnAlgebraOperation operation_;
      };

      Physical_Operator_Map_Ptr map_init_function_column_constant_operator();
      TypedOperatorPtr create_CPU_ColumnConstant_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_sort_operator;

    // Map_Init_Function
    // init_function_sort_operator=physical_operator::map_init_function_column_constant_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_ColumnConstantOperator
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_column_constant_operator>  // init_function_sort_operator>
      {
       public:
        Logical_ColumnConstantOperator(
            std::string column_name, const boost::any& value,
            const std::string& result_col_name,
            ColumnAlgebraOperation operation,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint())
            : TypedNode_Impl<TablePtr,
                             physical_operator::
                                 map_init_function_column_constant_operator>(
                  false, dev_constr),
              column_name_(column_name),
              value_(value),
              result_column_name_(result_col_name),
              operation_(operation) {}

        virtual unsigned int getOutputResultSize() const { return 10; }

        virtual double getCalculatedSelectivity() const { return 0.1; }

        virtual std::string getOperationName() const {
          return "ColumnConstantOperator";
        }
        std::string toString(bool verbose) const {
          std::string result = "ColumnConstantOperator";
          if (verbose) {
            result += " (";
            result += result_column_name_;
            result += "=";
            result += util::getName(operation_);
            result += "(";
            result += column_name_;
            result += ",";
            std::stringstream ss;
            ss << value_;
            result += ss.str();
            result += ")";

            result += ")";
          }
          return result;
        }
        const std::string& getColumnName() { return column_name_; }

        const boost::any& getValue() { return value_; }

        const std::string& getResultColumnName() { return result_column_name_; }

        CoGaDB::ColumnAlgebraOperation getColumnAlgebraOperation() {
          return operation_;
        }

        const std::list<std::string> getNamesOfReferencedColumns() const {
          std::list<std::string> result;
          result.push_back(column_name_);
          std::stringstream ss;
          ss << value_;
          result.push_back(ss.str());
          //                                            if(!result_column_name_.empty())
          //                                                result.push_back(result_column_name_);
          return result;
        }

       private:
        std::string column_name_;
        boost::any value_;
        std::string result_column_name_;
        CoGaDB::ColumnAlgebraOperation operation_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
