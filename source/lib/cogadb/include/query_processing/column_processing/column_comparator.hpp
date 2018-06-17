#pragma once

#include <core/lookup_array.hpp>
#include <core/selection_expression.hpp>
#include <query_processing/column_processing/definitions.hpp>
#include <query_processing/definitions.hpp>
#include <query_processing/operator_extensions.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class ColumnComparatorOperation
          : public hype::queryprocessing::BinaryOperator<ColumnPtr, ColumnPtr,
                                                         ColumnPtr>,
            public PositionListOperator {
       public:
        // typedef
        // hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::TypedOperatorPtr
        // ColumnWise_TypedOperatorPtr;
        typedef column_processing::cpu::TypedOperatorPtr TypedOperatorPtr;

        ColumnComparatorOperation(const hype::SchedulingDecision& sched_dec,
                                  TypedOperatorPtr left_child,
                                  TypedOperatorPtr right_child, Predicate op,
                                  MaterializationStatus mat_stat = MATERIALIZE)
            : BinaryOperator<ColumnPtr, ColumnPtr, ColumnPtr>(
                  sched_dec, left_child, right_child),
              PositionListOperator(),
              op_(op) {}

        virtual bool execute() {
          if (!quiet && debug && verbose)
            std::cout << "Execute Column Operator CPU" << std::endl;
          // assure we have a Value Value Predicate, meaning we compare to
          // columns
          assert(op_.getPredicateType() == ValueValuePredicate);
          hype::ProcessingDeviceID id =
              sched_dec_.getDeviceSpecification().getProcessingDeviceID();
          ProcessorSpecification proc_spec(id);
          SelectionParam param(proc_spec, op_.getPredicateType(),
                               this->getInputDataRightChild(),
                               op_.getValueComparator());

          ColumnPtr col = this->getInputDataLeftChild();
          col = copy_if_required(col, proc_spec);
          if (!col) {
            this->has_aborted_ = true;
            return false;
          }

          // compute the result TIDs
          PositionListPtr tids = col->selection(param);
          // create a LookupArray, the improtant thing is that the tids are
          // returned
          // since we need a ColumnPtr to cosntruct a Lookup Array, we pass it
          // the pointer of one of the childs
          this->result_ = createLookupArrayForColumn(col, tids);
          this->tids_ = tids;
          this->result_size_ = tids->size();

          if (this->result_)
            return true;
          else
            return false;
        }

        virtual ~ColumnComparatorOperation() {}

       private:
        Predicate op_;
      };

      column_processing::cpu::Physical_Operator_Map_Ptr
      map_init_function_cpu_column_comparison_operator();
      column_processing::cpu::TypedOperatorPtr
      create_CPU_ColumnComparatorOperator(
          column_processing::cpu::TypedLogicalNode& logical_node,
          const hype::SchedulingDecision&,
          column_processing::cpu::TypedOperatorPtr left_child,
          column_processing::cpu::TypedOperatorPtr right_child);

      //			GPU_ColumnWise_TypedOperatorPtr
      // create_GPU_ColumnComparatorOperator(GPU_ColumnWise_TypedLogicalNode&
      // logical_node, const hype::SchedulingDecision&,
      // GPU_ColumnWise_TypedOperatorPtr left_child,
      // GPU_ColumnWise_TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_ColumnComparatorOperation
          : public hype::queryprocessing::TypedNode_Impl<
                ColumnPtr,
                physical_operator::
                    map_init_function_cpu_column_comparison_operator>  // init_function_Join_operator>
      {
       public:
        Logical_ColumnComparatorOperation(
            Predicate pred, hype::DeviceConstraint dev_constr =
                                hype::DeviceConstraint(hype::CPU_ONLY))
            : TypedNode_Impl<
                  ColumnPtr,
                  physical_operator::
                      map_init_function_cpu_column_comparison_operator>(
                  false, dev_constr),
              op_(pred),
              mat_stat_() {}

        virtual unsigned int getOutputResultSize() const { return 0; }

        virtual double getCalculatedSelectivity() const { return 1; }

        virtual std::string getOperationName() const {
          return "ColumnComparatorOperation";  // util::getName(op_);
        }

        virtual Predicate getPredicate() const { return op_; }

        const MaterializationStatus& getMaterializationStatus() const {
          return mat_stat_;
        }

        std::string toString(bool verbose) const {
          std::string result = "ColumnComparatorOperation";
          if (verbose) {
            result += " (";
            result += op_.toString();
            result += " )";
          }
          return result;
        }

       private:
        Predicate op_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
