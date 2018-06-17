#pragma once

#include <core/lookup_array.hpp>
#include <query_processing/column_processing/definitions.hpp>
#include <query_processing/definitions.hpp>
#include <query_processing/operator_extensions.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_PositionList_Operator
          : public hype::queryprocessing::BinaryOperator<ColumnPtr, ColumnPtr,
                                                         ColumnPtr>,
            public PositionListOperator {
       public:
        // typedef
        // hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::TypedOperatorPtr
        // ColumnWise_TypedOperatorPtr;
        typedef column_processing::cpu::TypedOperatorPtr TypedOperatorPtr;

        CPU_PositionList_Operator(const hype::SchedulingDecision& sched_dec,
                                  TypedOperatorPtr left_child,
                                  TypedOperatorPtr right_child,
                                  PositionListOperation op,
                                  MaterializationStatus mat_stat = MATERIALIZE)
            : BinaryOperator<ColumnPtr, ColumnPtr, ColumnPtr>(
                  sched_dec, left_child, right_child),
              PositionListOperator(),
              op_(op) {}

        virtual bool execute();

        //                virtual bool execute() {
        //                    if (!quiet && debug && verbose) std::cout <<
        //                    "Execute Column Operator CPU" << std::endl;
        //
        //
        ////
        /// assert(!this->getInputDataLeftChild()->isMaterialized() &&
        ///!this->getInputDataLeftChild()->isCompressed() );
        ////                    //Workaround: We can only work on ColumnPtr and
        /// cannot mix PostionListPtr in the plan, so we need a workaround
        ////                    //create LookupArrays, which store tid lists in
        /// the filter outine and get TID lists in the Positionlistroutine
        ////                    PositionListPtr
        /// tids_left_child=getPositonListfromLookupArray(this->getInputDataLeftChild());
        ////                    PositionListPtr
        /// tids_right_child=getPositonListfromLookupArray(this->getInputDataRightChild());
        ////
        ////                    assert(tids_left_child!=NULL);
        ////                    assert(tids_right_child!=NULL);
        //
        //
        //                    PositionListOperator* pos_list_op_left =
        //                    dynamic_cast<PositionListOperator*>
        //                    (this->left_child_.get());
        //                    PositionListOperator* pos_list_op_right =
        //                    dynamic_cast<PositionListOperator*>
        //                    (this->right_child_.get());
        //
        //                    assert(pos_list_op_left != NULL);
        //                    assert(pos_list_op_right != NULL);
        //                    assert(pos_list_op_left->hasResultPositionList()
        //                    ||
        //                    pos_list_op_left->hasCachedResult_GPU_PositionList());
        //                    assert(pos_list_op_right->hasResultPositionList()
        //                    ||
        //                    pos_list_op_right->hasCachedResult_GPU_PositionList());
        //
        //                    PositionListPtr input_tids_left;
        //                    if (!pos_list_op_left->hasResultPositionList() &&
        //                    pos_list_op_left->hasCachedResult_GPU_PositionList())
        //                    {
        //                        input_tids_left =
        //                        gpu::copy_PositionList_device_to_host(pos_list_op_left->getResult_GPU_PositionList());
        //                    } else {
        //                        input_tids_left =
        //                        pos_list_op_left->getResultPositionList();
        //                    }
        //                    assert(input_tids_left != NULL);
        //
        //                    PositionListPtr input_tids_right;
        //                    if (!pos_list_op_right->hasResultPositionList() &&
        //                    pos_list_op_right->hasCachedResult_GPU_PositionList())
        //                    {
        //                        input_tids_right =
        //                        gpu::copy_PositionList_device_to_host(pos_list_op_right->getResult_GPU_PositionList());
        //                    } else {
        //                        input_tids_right =
        //                        pos_list_op_right->getResultPositionList();
        //                    }
        //                    assert(input_tids_right != NULL);
        //
        //                    PositionListPtr tids;
        //                    if (op_ == POSITIONLIST_INTERSECTION) {
        //                        tids =
        //                        computePositionListIntersection(input_tids_left,
        //                        input_tids_right);
        //                    } else if (op_ == POSITIONLIST_UNION) {
        //                        tids =
        //                        computePositionListUnion(input_tids_left,
        //                        input_tids_right);
        //                    }
        //
        //
        //                    //this->result_ =
        //                    createLookupArrayForColumn(this->getInputDataLeftChild(),tids);
        //                    this->cpu_tids_ = tids;
        //                    this->result_size_ = tids->size();
        //
        //                    if (this->result_)
        //                        return true;
        //                    else
        //                        return false;
        //                }

        virtual ~CPU_PositionList_Operator() {}

       private:
        PositionListOperation op_;
      };

      column_processing::cpu::Physical_Operator_Map_Ptr
      map_init_function_cpu_positionlist_operator();
      column_processing::cpu::TypedOperatorPtr create_CPU_PositionList_Operator(
          column_processing::cpu::TypedLogicalNode& logical_node,
          const hype::SchedulingDecision&,
          column_processing::cpu::TypedOperatorPtr left_child,
          column_processing::cpu::TypedOperatorPtr right_child);

      //			GPU_ColumnWise_TypedOperatorPtr
      // create_GPU_ColumnAlgebraOperator(GPU_ColumnWise_TypedLogicalNode&
      // logical_node, const hype::SchedulingDecision&,
      // GPU_ColumnWise_TypedOperatorPtr left_child,
      // GPU_ColumnWise_TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_PositionList_Operator
          : public hype::queryprocessing::TypedNode_Impl<
                ColumnPtr, physical_operator::
                               map_init_function_cpu_positionlist_operator> {
       public:
        Logical_PositionList_Operator(
            PositionListOperation op,
            // MaterializationStatus mat_stat = MATERIALIZE,
            hype::DeviceConstraint dev_constr = hype::DeviceConstraint(
                RuntimeConfiguration::instance().getGlobalDeviceConstraint()))
            : TypedNode_Impl<ColumnPtr,
                             physical_operator::
                                 map_init_function_cpu_positionlist_operator>(
                  false, dev_constr),
              op_(op),
              mat_stat_(MATERIALIZE) {  //(mat_stat) {
        }

        virtual unsigned int getOutputResultSize() const { return 0; }

        virtual double getCalculatedSelectivity() const { return 1; }

        virtual std::string getOperationName() const {
          return "PositionList_Operator";  // util::getName(op_);
        }

        virtual PositionListOperation getPositionListOperation() const {
          return op_;
        }

        const MaterializationStatus& getMaterializationStatus() const {
          return mat_stat_;
        }
        std::string toString(bool verbose) const {
          std::string result = "PositionList_Operator";
          if (verbose) {
            result += " (";
            result += util::getName(op_);
            result += ")";
          }
          return result;
        }

       private:
        PositionListOperation op_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
