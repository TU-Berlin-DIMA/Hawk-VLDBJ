#pragma once

#include <query_processing/definitions.hpp>
#include <util/get_name.hpp>

//		typedef
// hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::Map_Init_Function
// ColumnWise_Map_Init_Function;
//		typedef
// hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::Physical_Operator_Map
// ColumnWise_Physical_Operator_Map;
//		typedef
// hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::Physical_Operator_Map_Ptr
// ColumnWise_Physical_Operator_Map_Ptr;
//		typedef
// hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::TypedOperatorPtr
// ColumnWise_TypedOperatorPtr;
//		typedef
// hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::TypedLogicalNode
// ColumnWise_TypedLogicalNode;
//		typedef
// hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::PhysicalQueryPlanPtr
// ColumnWise_PhysicalQueryPlanPtr;
//
//		typedef hype::queryprocessing::LogicalQueryPlan<ColumnPtr>
// ColumnWise_LogicalQueryPlan;

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_ColumnAlgebraOperation
          : public hype::queryprocessing::BinaryOperator<ColumnPtr, ColumnPtr,
                                                         ColumnPtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            ColumnPtr>::TypedOperatorPtr ColumnWise_TypedOperatorPtr;

        CPU_ColumnAlgebraOperation(const hype::SchedulingDecision& sched_dec,
                                   ColumnWise_TypedOperatorPtr left_child,
                                   ColumnWise_TypedOperatorPtr right_child,
                                   ColumnAlgebraOperation op,
                                   MaterializationStatus mat_stat = MATERIALIZE)
            : BinaryOperator<ColumnPtr, ColumnPtr, ColumnPtr>(
                  sched_dec, left_child, right_child),
              op_(op),
              mat_stat_(mat_stat) {}

        virtual bool execute() {
          std::cout << "Execute Column Operator CPU" << std::endl;

          if (op_ == ADD) {
            this->getInputDataLeftChild()->add(this->getInputDataRightChild());
          } else if (op_ == SUB) {
            this->getInputDataLeftChild()->minus(
                this->getInputDataRightChild());
          } else if (op_ == MUL) {
            this->getInputDataLeftChild()->multiply(
                this->getInputDataRightChild());
          } else if (op_ == DIV) {
            this->getInputDataLeftChild()->division(
                this->getInputDataRightChild());
          }

          this->result_ = this->getInputDataLeftChild();

          if (this->result_)
            return true;
          else
            return false;
        }

        virtual ~CPU_ColumnAlgebraOperation() {}

       private:
        ColumnAlgebraOperation op_;
        MaterializationStatus mat_stat_;
      };

      class GPU_ColumnAlgebraOperation
          : public hype::queryprocessing::BinaryOperator<
                gpu::GPU_Base_ColumnPtr, gpu::GPU_Base_ColumnPtr,
                gpu::GPU_Base_ColumnPtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            gpu::GPU_Base_ColumnPtr>::TypedOperatorPtr
            GPU_ColumnWise_TypedOperatorPtr;

        GPU_ColumnAlgebraOperation(const hype::SchedulingDecision& sched_dec,
                                   GPU_ColumnWise_TypedOperatorPtr left_child,
                                   GPU_ColumnWise_TypedOperatorPtr right_child,
                                   ColumnAlgebraOperation op,
                                   MaterializationStatus mat_stat = MATERIALIZE)
            : BinaryOperator<gpu::GPU_Base_ColumnPtr, gpu::GPU_Base_ColumnPtr,
                             gpu::GPU_Base_ColumnPtr>(sched_dec, left_child,
                                                      right_child),
              op_(op),
              mat_stat_(mat_stat) {}

        virtual bool execute() {
          std::cout << "Execute Column Operator GPU" << std::endl;
          // const TablePtr sort(TablePtr table, const std::string& column_name,
          // SortOrder order=ASCENDING, MaterializationStatus
          // mat_stat=MATERIALIZE, ComputeDevice comp_dev=CPU);
          // this->result_=BaseTable::sort(this->getInputData(),
          // column_name_,order_, mat_stat_,CPU);
          // JoinAlgorithm{SORT_MERGE_JOIN,NESTED_LOOP_JOIN,HASH_JOIN};
          // this->result_=BaseTable::join(this->getInputDataLeftChild(),
          // join_column1_name_, this->getInputDataRightChild(),
          // join_column2_name_, NESTED_LOOP_JOIN, mat_stat_, CPU);

          if (this->result_)
            return true;
          else
            return false;
        }

        virtual ~GPU_ColumnAlgebraOperation() {}

       private:
        ColumnAlgebraOperation op_;
        MaterializationStatus mat_stat_;
      };

      ColumnWise_Physical_Operator_Map_Ptr
      map_init_function_column_algebra_operator();  // erfordert dass object auf
                                                    // unterschiedliche
                                                    // Processing devices gleich
                                                    // behandelt werden kann,
                                                    // was aber in diesem Fall
                                                    // nicht gegeben ist, weil
                                                    // CPU und GPU columns nix
                                                    // miteinander zu tun haben!
                                                    // ->
      ColumnWise_TypedOperatorPtr create_CPU_ColumnAlgebraOperator(
          ColumnWise_TypedLogicalNode& logical_node,
          const hype::SchedulingDecision&,
          ColumnWise_TypedOperatorPtr left_child,
          ColumnWise_TypedOperatorPtr right_child);
      GPU_ColumnWise_TypedOperatorPtr create_GPU_ColumnAlgebraOperator(
          GPU_ColumnWise_TypedLogicalNode& logical_node,
          const hype::SchedulingDecision&,
          GPU_ColumnWise_TypedOperatorPtr left_child,
          GPU_ColumnWise_TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_Join_operator;

    // Map_Init_Function
    // init_function_Join_operator=physical_operator::map_init_function_Join_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_ColumnAlgebraOperation
          : public hype::queryprocessing::TypedNode_Impl<
                ColumnPtr,
                physical_operator::
                    map_init_function_column_algebra_operator>  // init_function_Join_operator>
      {
       public:
        Logical_ColumnAlgebraOperation(
            ColumnAlgebraOperation op,
            MaterializationStatus mat_stat = MATERIALIZE)
            : TypedNode_Impl<ColumnPtr,
                             physical_operator::
                                 map_init_function_column_algebra_operator>(),
              op_(op),
              mat_stat_(mat_stat) {}

        virtual unsigned int getOutputResultSize() const { return 0; }

        virtual double getCalculatedSelectivity() const { return 1; }

        virtual std::string getOperationName() const {
          return util::getName(op_);
        }

        virtual ColumnAlgebraOperation getColumnAlgebraOperation() const {
          return op_;
        }

        const MaterializationStatus& getMaterializationStatus() const {
          return mat_stat_;
        }

       private:
        ColumnAlgebraOperation op_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
