#pragma once

#include <query_processing/definitions.hpp>
#include <sstream>
#include <util/getname.hpp>
#include <util/iostream.hpp>

#include <core/selection_expression.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_Selection_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_Selection_Operator(const hype::SchedulingDecision& sched_dec,
                               TypedOperatorPtr child, Predicate pred,
                               MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              pred_(pred) {}

        virtual bool execute() {
          // std::cout << "Execute Selection CPU" << std::endl;
          // const TablePtr sort(TablePtr table, const std::string& column_name,
          // SortOrder order=ASCENDING, MaterializationStatus
          // mat_stat=MATERIALIZE, ComputeDevice comp_dev=CPU);
          // this->result_=BaseTable::sort(this->getInputData(),
          // column_name_,order_, mat_stat_,CPU);
          // this->result_ = BaseTable::selection(this->getInputData(),
          // column_name_, value_for_comparison_, comp_, mat_stat_, SERIAL);
          assert(pred_.getPredicateType() == ValueConstantPredicate);
          ProcessorSpecification proc_spec(hype::PD0);
          SelectionParam param(proc_spec, pred_.getPredicateType(),
                               pred_.getConstant(), pred_.getValueComparator());
          this->result_ = BaseTable::selection(this->getInputData(),
                                               pred_.getColumn1Name(), param);

          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~CPU_Selection_Operator() {}

       private:
        Predicate pred_;
      };

      class CPU_ParallelSelection_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_ParallelSelection_Operator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            Predicate pred, MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              pred_(pred) {}

        virtual bool execute() {
          assert(pred_.getPredicateType() == ValueConstantPredicate);
          ProcessorSpecification proc_spec(hype::PD0);
          SelectionParam param(proc_spec, pred_.getPredicateType(),
                               pred_.getConstant(), pred_.getValueComparator());
          this->result_ = BaseTable::selection(this->getInputData(),
                                               pred_.getColumn1Name(), param);

          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~CPU_ParallelSelection_Operator() {}

       private:
        Predicate pred_;
      };

      class GPU_Selection_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        GPU_Selection_Operator(const hype::SchedulingDecision& sched_dec,
                               TypedOperatorPtr child, Predicate pred,
                               MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              pred_(pred) {}

        virtual bool execute() {
          // std::cout << "Execute Selection GPU" << std::endl;
          // const TablePtr sort(TablePtr table, const std::string& column_name,
          // SortOrder order=ASCENDING, MaterializationStatus
          // mat_stat=MATERIALIZE, ComputeDevice comp_dev=CPU);
          // this->result_ = BaseTable::selection(this->getInputData(),
          // column_name_, value_for_comparison_, comp_, mat_stat_, PARALLEL,
          // GPU);
          assert(pred_.getPredicateType() == ValueConstantPredicate);
          ProcessorSpecification proc_spec(hype::PD1);
          SelectionParam param(proc_spec, pred_.getPredicateType(),
                               pred_.getConstant(), pred_.getValueComparator());
          this->result_ = BaseTable::selection(this->getInputData(),
                                               pred_.getColumn1Name(), param);

          if (this->result_) {
            setResultSize(((TablePtr) this->result_)->getNumberofRows());
            return true;
          } else
            return false;
        }

        virtual ~GPU_Selection_Operator() {}

       private:
        Predicate pred_;
      };

      Physical_Operator_Map_Ptr map_init_function_selection_operator();
      TypedOperatorPtr create_CPU_Selection_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);
      TypedOperatorPtr create_CPU_ParallelSelection_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);
      TypedOperatorPtr create_GPU_Selection_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_Selection_operator;

    // Map_Init_Function
    // init_function_Selection_operator=physical_operator::map_init_function_Selection_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_Selection
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_selection_operator>  // init_function_Selection_operator>
      {
       public:
        Logical_Selection(
            std::string column_name, const boost::any& value_for_comparison,
            const ValueComparator& comp,
            MaterializationStatus mat_stat = LOOKUP,
            hype::DeviceConstraint dev_constr = hype::DeviceConstraint())
            : TypedNode_Impl<
                  TablePtr,
                  physical_operator::map_init_function_selection_operator>(
                  false, dev_constr),
              pred_(Predicate(column_name, value_for_comparison,
                              ValueConstantPredicate, comp)),
              mat_stat_(mat_stat) {}

        Logical_Selection(
            Predicate pred, MaterializationStatus mat_stat = LOOKUP,
            hype::DeviceConstraint dev_constr = hype::DeviceConstraint())
            : TypedNode_Impl<
                  TablePtr,
                  physical_operator::map_init_function_selection_operator>(
                  false, dev_constr),
              pred_(pred),
              mat_stat_(mat_stat) {}

        virtual unsigned int getOutputResultSize() const { return 10; }

        virtual double getCalculatedSelectivity() const { return 0.1; }

        virtual std::string getOperationName() const { return "SELECTION"; }
        std::string toString(bool verbose) const {
          std::string result = "SELECTION";
          if (verbose) {
            result += " ";
            result += pred_.toString();
          }
          return result;
        }
        const Predicate& getPredicate() { return pred_; }

        const MaterializationStatus getMaterializationStatus() const {
          return mat_stat_;
        }

        virtual bool isInputDataCachedInGPU();

       private:
        Predicate pred_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
