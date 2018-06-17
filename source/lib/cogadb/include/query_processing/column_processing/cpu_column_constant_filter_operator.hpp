#pragma once

#include <core/selection_expression.hpp>
#include <persistence/storage_manager.hpp>
#include <query_processing/column_processing/definitions.hpp>
#include <query_processing/operator_extensions.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_column_constant_filter_operator
          : public hype::queryprocessing::UnaryOperator<ColumnPtr, ColumnPtr>,
            public PositionListOperator {
       public:
        typedef column_processing::cpu::TypedOperatorPtr TypedOperatorPtr;
        CPU_column_constant_filter_operator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            const Predicate& pred);
        virtual bool execute();
        virtual ~CPU_column_constant_filter_operator();

       private:
        Predicate pred_;
      };

      //            class GPU_column_constant_filter_operator : public
      //            hype::queryprocessing::UnaryOperator<ColumnPtr, ColumnPtr>,
      //            public PositionListOperator {
      //            public:
      //                typedef column_processing::cpu::TypedOperatorPtr
      //                TypedOperatorPtr;
      //                GPU_column_constant_filter_operator(const
      //                hype::SchedulingDecision& sched_dec, TypedOperatorPtr
      //                child, const Predicate& pred);
      //                virtual bool execute();
      //                virtual bool isInputDataCachedInGPU();
      //                virtual ~GPU_column_constant_filter_operator();
      //            private:
      //                Predicate pred_;
      //            };

      column_processing::cpu::Physical_Operator_Map_Ptr
      map_init_function_column_constant_filter_operator();
      column_processing::cpu::TypedOperatorPtr
      create_CPU_column_constant_filter_operator(
          column_processing::cpu::TypedLogicalNode& logical_node,
          const hype::SchedulingDecision&,
          column_processing::cpu::TypedOperatorPtr left_child,
          column_processing::cpu::TypedOperatorPtr right_child);
      //            column_processing::cpu::TypedOperatorPtr
      //            create_GPU_column_constant_filter_operator(column_processing::cpu::TypedLogicalNode&
      //            logical_node, const hype::SchedulingDecision&,
      //            column_processing::cpu::TypedOperatorPtr left_child,
      //            column_processing::cpu::TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_Column_Constant_Filter
          : public hype::queryprocessing::TypedNode_Impl<
                ColumnPtr,
                physical_operator::
                    map_init_function_column_constant_filter_operator>  // init_function_column_constant_filter_operator>
      // //init_function_column_constant_filter_operator>
      {
       public:
        Logical_Column_Constant_Filter(
            const Predicate&,
            hype::DeviceConstraint dev_constr = hype::DeviceConstraint());

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        const Predicate& getPredicate() const;

        std::string toString(bool verbose) const;
        // virtual column_processing::cpu::TypedOperatorPtr
        // getOptimalOperator(column_processing::cpu::TypedOperatorPtr
        // left_child, column_processing::cpu::TypedOperatorPtr right_child,
        // hype::DeviceTypeConstraint dev_constr);

        virtual bool isInputDataCachedInGPU();

       private:
        Predicate pred_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
