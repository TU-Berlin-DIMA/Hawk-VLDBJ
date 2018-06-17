#pragma once

#include <query_processing/definitions.hpp>
#include <sstream>
#include <util/getname.hpp>
#include <util/iostream.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class AddConstantValueColumn_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        AddConstantValueColumn_Operator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            const std::string& col_name, AttributeType type,
            const boost::any& value)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              col_name_(col_name),
              type_(type),
              value_(value) {}

        virtual bool execute() {
          ProcessorSpecification proc_spec(
              sched_dec_.getDeviceSpecification().getProcessingDeviceID());
          this->result_ = BaseTable::AddConstantValueColumnOperation(
              this->getInputData(), col_name_, type_, value_, proc_spec);
          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~AddConstantValueColumn_Operator() {}

       private:
        std::string col_name_;
        AttributeType type_;
        boost::any value_;
      };

      Physical_Operator_Map_Ptr
      map_init_function_addconstantvaluecolumn_operator();
      TypedOperatorPtr create_AddConstantValueColumn_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_AddConstantValueColumn
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_addconstantvaluecolumn_operator>  // init_function_AddConstantValueColumn_operator>
      {
       public:
        Logical_AddConstantValueColumn(
            const std::string& col_name, AttributeType type,
            const boost::any& value,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint())
            : TypedNode_Impl<
                  TablePtr,
                  physical_operator::
                      map_init_function_addconstantvaluecolumn_operator>(
                  false, dev_constr),
              col_name_(col_name),
              type_(type),
              value_(value) {}

        virtual unsigned int getOutputResultSize() const {
          assert(this->left_ != NULL);
          return this->left_->getOutputResultSize();  // same #rows as child
        }

        virtual double getSelectivity() const { return 1; }

        virtual std::string getOperationName() const {
          return "AddConstantValueColumn";
        }

        std::string toString(bool verbose) const {
          std::string result = "AddConstantValueColumn";
          if (verbose) {
            result += " ";
            result += col_name_;
            result += " (Type: ";
            result += util::getName(type_);
            result += " Value: ";
            std::stringstream ss;
            ss << value_;
            result += ss.str();
            result += ")";
          }
          return result;
        }

        const std::string& getColumnName() { return col_name_; }

        const AttributeType& getAttributeType() const { return type_; }

        const boost::any& getConstantValue() const { return value_; }

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
        std::string col_name_;
        AttributeType type_;
        boost::any value_;
      };
    }  // end namespace logical_operator
  }    // end namespace query_processing
}  // end namespace CogaDB
