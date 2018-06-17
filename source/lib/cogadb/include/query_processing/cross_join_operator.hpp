#pragma once

#include <query_processing/definitions.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_CrossJoin_Operator
          : public hype::queryprocessing::BinaryOperator<TablePtr, TablePtr,
                                                         TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_CrossJoin_Operator(const hype::SchedulingDecision& sched_dec,
                               TypedOperatorPtr left_child,
                               TypedOperatorPtr right_child,
                               MaterializationStatus mat_stat = MATERIALIZE)
            : BinaryOperator<TablePtr, TablePtr, TablePtr>(
                  sched_dec, left_child, right_child),
              join_column1_name_(),
              join_column2_name_(),
              mat_stat_(mat_stat) {}

        virtual bool execute() {
          if (!quiet && verbose && debug)
            std::cout << "Execute Join CPU" << std::endl;
          // const TablePtr sort(TablePtr table, const std::string& column_name,
          // SortOrder order=ASCENDING, MaterializationStatus
          // mat_stat=MATERIALIZE, ComputeDevice comp_dev=CPU);
          // this->result_=BaseTable::sort(this->getInputData(),
          // column_name_,order_, mat_stat_,CPU);
          // JoinAlgorithm{SORT_MERGE_JOIN,NESTED_LOOP_JOIN,HASH_JOIN};
          this->result_ =
              BaseTable::crossjoin(this->getInputDataLeftChild(),
                                   this->getInputDataRightChild(), mat_stat_);
          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~CPU_CrossJoin_Operator() {}

       private:
        std::string join_column1_name_;
        std::string join_column2_name_;
        MaterializationStatus mat_stat_;
      };

      Physical_Operator_Map_Ptr map_init_function_crossjoin_operator();
      TypedOperatorPtr create_CPU_CrossJoin_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);
    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_Join_operator;

    // Map_Init_Function
    // init_function_Join_operator=physical_operator::map_init_function_Join_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_CrossJoin
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_crossjoin_operator>  // init_function_Join_operator>
      {
       public:
        Logical_CrossJoin();

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
