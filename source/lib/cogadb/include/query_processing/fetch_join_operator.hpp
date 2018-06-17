#pragma once

#include <query_processing/definitions.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_Fetch_Join_Operator
          : public hype::queryprocessing::BinaryOperator<TablePtr, TablePtr,
                                                         TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_Fetch_Join_Operator(const hype::SchedulingDecision& sched_dec,
                                TypedOperatorPtr left_child,
                                TypedOperatorPtr right_child,
                                const std::string& join_column1_name,
                                const std::string& join_column2_name,
                                MaterializationStatus mat_stat = LOOKUP)
            : BinaryOperator<TablePtr, TablePtr, TablePtr>(
                  sched_dec, left_child, right_child),
              join_column1_name_(join_column1_name),
              join_column2_name_(join_column2_name) {}

        virtual bool execute() {
          if (!quiet && verbose && debug)
            std::cout << "Execute Fetch_Join CPU" << std::endl;
          // const TablePtr sort(TablePtr table, const std::string& column_name,
          // SortOrder order=ASCENDING, MaterializationStatus
          // mat_stat=MATERIALIZE, ComputeDevice comp_dev=CPU);
          // this->result_=BaseTable::sort(this->getInputData(),
          // column_name_,order_, mat_stat_,CPU);
          // Fetch_JoinAlgorithm{SORT_MERGE_FETCH_JOIN,NESTED_LOOP_FETCH_JOIN,HASH_FETCH_JOIN};

          hype::ProcessingDeviceID id =
              sched_dec_.getDeviceSpecification().getProcessingDeviceID();
          ProcessorSpecification proc_spec(id);
          FetchJoinParam param(proc_spec);

          this->result_ = BaseTable::fetch_join(
              this->getInputDataLeftChild(), join_column1_name_,
              this->getInputDataRightChild(), join_column2_name_, param);
          if (this->result_) {
            setResultSize((this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~CPU_Fetch_Join_Operator() {}

       private:
        std::string join_column1_name_;
        std::string join_column2_name_;
      };

      Physical_Operator_Map_Ptr map_init_function_fetch_join_operator();
      TypedOperatorPtr create_CPU_Fetch_Join_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_Fetch_Join_operator;

    // Map_Init_Function
    // init_function_Fetch_Join_operator=physical_operator::map_init_function_Fetch_Join_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_Fetch_Join
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_fetch_join_operator>  // init_function_Fetch_Join_operator>
      {
       public:
        Logical_Fetch_Join(
            const std::string& join_pk_column_name,
            const std::string& join_fk_column_name,
            MaterializationStatus mat_stat = LOOKUP,
            hype::DeviceConstraint dev_constr = hype::DeviceConstraint())
            : TypedNode_Impl<
                  TablePtr,
                  physical_operator::map_init_function_fetch_join_operator>(
                  false, dev_constr),
              join_pk_column_name_(join_pk_column_name),
              join_fk_column_name_(join_fk_column_name),
              mat_stat_(mat_stat) {}

        virtual unsigned int getOutputResultSize() const {
          // right child has to be a FK Table, which is exactly the
          // number of matching keys and hence, the result size
          return this->getRight()->getOutputResultSize();
        }

        virtual double getCalculatedSelectivity() const { return 1; }

        virtual std::string getOperationName() const { return "FETCH_JOIN"; }
        std::string toString(bool verbose) const {
          std::string result = "FETCH_JOIN";
          if (verbose) {
            result += " (";
            result += join_pk_column_name_;
            result += "=";
            result += join_fk_column_name_;
            result += ")";
          }
          return result;
        }

        virtual bool isInputDataCachedInGPU();

        const std::string& getPKColumnName() { return join_pk_column_name_; }

        const std::string& getFKColumnName() { return join_fk_column_name_; }

        const MaterializationStatus& getMaterializationStatus() const {
          return mat_stat_;
        }

       private:
        std::string join_pk_column_name_;
        std::string join_fk_column_name_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
