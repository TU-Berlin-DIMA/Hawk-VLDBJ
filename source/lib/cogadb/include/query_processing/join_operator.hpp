#pragma once

#include <query_processing/definitions.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class Join_Operator
          : public hype::queryprocessing::BinaryOperator<TablePtr, TablePtr,
                                                         TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        Join_Operator(const hype::SchedulingDecision& sched_dec,
                      TypedOperatorPtr left_child, TypedOperatorPtr right_child,
                      const std::string& join_column1_name,
                      const std::string& join_column2_name,
                      const JoinParam& join_param)
            : BinaryOperator<TablePtr, TablePtr, TablePtr>(
                  sched_dec, left_child, right_child),
              join_column1_name_(join_column1_name),
              join_column2_name_(join_column2_name),
              join_param_(join_param) {}

        bool is_semi_join(const JoinType& join_type) {
          if (join_type == LEFT_SEMI_JOIN || join_type == LEFT_ANTI_SEMI_JOIN ||
              join_type == RIGHT_SEMI_JOIN ||
              join_type == RIGHT_ANTI_SEMI_JOIN) {
            return true;
          } else {
            return false;
          }
        }

        virtual bool execute() {
          if (!quiet && verbose && debug)
            std::cout << "Execute Join CPU" << std::endl;

          ProcessorSpecification proc_spec(
              sched_dec_.getDeviceSpecification().getProcessingDeviceID());
          JoinParam join_param(proc_spec, join_param_.join_alg,
                               join_param_.join_type);

          if (!is_semi_join(this->join_param_.join_type)) {
            this->result_ = BaseTable::join(
                this->getInputDataLeftChild(), join_column1_name_,
                this->getInputDataRightChild(), join_column2_name_, join_param);
          } else {
            this->result_ = BaseTable::semi_join(
                this->getInputDataLeftChild(), join_column1_name_,
                this->getInputDataRightChild(), join_column2_name_, join_param);
          }
          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else {
            return false;
          }
        }

        virtual ~Join_Operator() {}

       private:
        std::string join_column1_name_;
        std::string join_column2_name_;
        JoinParam join_param_;
      };

      Physical_Operator_Map_Ptr map_init_function_join_operator();
      TypedOperatorPtr create_NestedLoopJoin_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);
      TypedOperatorPtr create_SortMergeJoin_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);
      TypedOperatorPtr create_HashJoin_Operator(TypedLogicalNode& logical_node,
                                                const hype::SchedulingDecision&,
                                                TypedOperatorPtr left_child,
                                                TypedOperatorPtr right_child);
      //			TypedOperatorPtr
      // create_Parallel_HashJoin_Operator(TypedLogicalNode& logical_node, const
      // hype::SchedulingDecision&, TypedOperatorPtr left_child,
      // TypedOperatorPtr
      // right_child);
      TypedOperatorPtr create_RadixJoin_Operator(
          TypedLogicalNode& logical_node,
          const hype::SchedulingDecision& sched_dec,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);
      TypedOperatorPtr create_IndexNestedLoop_Join_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_Join_operator;

    // Map_Init_Function
    // init_function_Join_operator=physical_operator::map_init_function_Join_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_Join
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_join_operator>  // init_function_Join_operator>
      {
       public:
        Logical_Join(
            const std::string& join_column1_name,
            const std::string& join_column2_name,
            const JoinType& join_type = INNER_JOIN,
            hype::DeviceConstraint dev_constr = hype::DeviceConstraint());

        virtual unsigned int getOutputResultSize() const;
        virtual double getCalculatedSelectivity() const;
        virtual const hype::Tuple getFeatureVector() const;

        bool isInputDataCachedInGPU();

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const std::string& getLeftJoinColumnName();

        const std::string& getRightJoinColumnName();

        const JoinType getJoinType() const;

       private:
        std::pair<bool, double> isInputDataCachedInGPU_internal() const;
        std::string join_column1_name_;
        std::string join_column2_name_;
        JoinType join_type_;
        bool consume_comes_from_left_sub_tree_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
