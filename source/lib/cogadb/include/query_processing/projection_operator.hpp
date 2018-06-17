#pragma once

#include <query_processing/definitions.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_Projection_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_Projection_Operator(const hype::SchedulingDecision& sched_dec,
                                TypedOperatorPtr child,
                                const std::list<std::string>& columns_to_select)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              columns_to_select_(columns_to_select) {}

        virtual bool execute() {
          if (!quiet && verbose && debug)
            std::cout << "Execute Projection CPU" << std::endl;
          // const TablePtr sort(TablePtr table, const std::string& column_name,
          // SortOrder order=ASCENDING, MaterializationStatus
          // mat_stat=MATERIALIZE, ComputeDevice comp_dev=CPU);
          // this->result_=BaseTable::sort(this->getInputData(),
          // column_name_,order_, mat_stat_,CPU);
          this->result_ = BaseTable::projection(
              this->getInputData(), columns_to_select_, LOOKUP, CPU);
          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~CPU_Projection_Operator() {}

       private:
        std::list<std::string> columns_to_select_;
      };

      class GPU_Projection_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        GPU_Projection_Operator(const hype::SchedulingDecision& sched_dec,
                                TypedOperatorPtr child,
                                const std::list<std::string>& columns_to_select)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              columns_to_select_(columns_to_select) {}

        virtual bool execute() {
          if (!quiet && verbose && debug)
            std::cout << "Execute Projection GPU" << std::endl;
          // const TablePtr sort(TablePtr table, const std::string& column_name,
          // SortOrder order=ASCENDING, MaterializationStatus
          // mat_stat=MATERIALIZE, ComputeDevice comp_dev=CPU);
          this->result_ = BaseTable::projection(
              this->getInputData(), columns_to_select_, LOOKUP, GPU);
          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~GPU_Projection_Operator() {}

       private:
        std::list<std::string> columns_to_select_;
      };

      Physical_Operator_Map_Ptr map_init_function_projection_operator();
      TypedOperatorPtr create_CPU_Projection_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);
      TypedOperatorPtr create_GPU_Projection_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_Projection
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_projection_operator>  // init_function_Projection_operator>
      {
       public:
        Logical_Projection(const std::list<std::string>& columns_to_select)
            : TypedNode_Impl<
                  TablePtr,
                  physical_operator::map_init_function_projection_operator>(),
              columns_to_select_(columns_to_select),
              attribute_references_() {
          std::list<std::string>::const_iterator cit;
          for (cit = columns_to_select_.begin();
               cit != columns_to_select_.end(); ++cit) {
            AttributeReferencePtr attr = getAttributeFromColumnIdentifier(*cit);
            if (!attr) {
              COGADB_FATAL_ERROR(
                  "Could not retrieve column '"
                      << *cit << "': Either not found or name ambiguous!",
                  "");
            }
            attribute_references_.push_back(attr);
          }
        }

        Logical_Projection(
            const std::vector<AttributeReferencePtr>& attribute_references)
            : TypedNode_Impl<
                  TablePtr,
                  physical_operator::map_init_function_projection_operator>(),
              columns_to_select_(),
              attribute_references_(attribute_references) {}

        virtual unsigned int getOutputResultSize() const {
          return this->left_->getOutputResultSize();
        }

        virtual double getCalculatedSelectivity() const { return 1; }

        virtual std::string getOperationName() const { return "PROJECTION"; }

        std::string toString(bool verbose) const {
          std::string result = "PROJECTION";
          if (verbose) {
            result += " (";
            std::list<std::string>::const_iterator cit;
            //                        for(cit=columns_to_select_.begin();cit!=columns_to_select_.end();++cit){
            //                            result+=*cit;
            //                            if(cit!=--columns_to_select_.end())
            //                                result+=",";
            //                        }
            for (size_t i = 0; i < attribute_references_.size(); ++i) {
              result += createFullyQualifiedColumnIdentifier(
                  attribute_references_[i]);
              if (attribute_references_[i]->getResultAttributeName() !=
                  createFullyQualifiedColumnIdentifier(
                      attribute_references_[i])) {
                result += " AS ";
                result += attribute_references_[i]->getResultAttributeName();
              }
              if (i + 1 < attribute_references_.size()) {
                result += ",";
              }
            }

            result += ")";
          }
          return result;
        }

        const std::list<std::string>& getColumnList() {
          return columns_to_select_;
        }

        const std::list<std::string> getNamesOfReferencedColumns() const {
          return columns_to_select_;
        }

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
        std::list<std::string> columns_to_select_;
        std::vector<AttributeReferencePtr> attribute_references_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
