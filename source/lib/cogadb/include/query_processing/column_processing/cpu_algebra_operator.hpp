//#pragma once
//
//#include <query_processing/definitions.hpp>
//#include <query_processing/column_processing/definitions.hpp>
//#include <util/getname.hpp>
//
// namespace CoGaDB {
//    namespace query_processing {
//        namespace physical_operator {
//
//            class CPU_ColumnAlgebraOperation : public
//            hype::queryprocessing::BinaryOperator<ColumnPtr, ColumnPtr,
//            ColumnPtr> {
//            public:
//                //typedef
//                hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::TypedOperatorPtr
//                ColumnWise_TypedOperatorPtr;
//                typedef column_processing::cpu::TypedOperatorPtr
//                TypedOperatorPtr;
//
//                CPU_ColumnAlgebraOperation(const hype::SchedulingDecision&
//                sched_dec,
//                        TypedOperatorPtr left_child,
//                        TypedOperatorPtr right_child,
//                        ColumnAlgebraOperation op,
//                        MaterializationStatus mat_stat = MATERIALIZE) :
//                        BinaryOperator<ColumnPtr, ColumnPtr,
//                        ColumnPtr>(sched_dec, left_child, right_child),
//                op_(op),
//                mat_stat_(mat_stat) {
//                }
//
//                virtual bool execute() {
//                    if (!quiet && debug && verbose) std::cout << "Execute
//                    Column Operator CPU" << std::endl;
//
//                    if (op_ == ADD) {
//                        this->getInputDataLeftChild()->add(this->getInputDataRightChild());
//                    } else if (op_ == SUB) {
//                        this->getInputDataLeftChild()->minus(this->getInputDataRightChild());
//                    } else if (op_ == MUL) {
//                        this->getInputDataLeftChild()->multiply(this->getInputDataRightChild());
//                    } else if (op_ == DIV) {
//                        this->getInputDataLeftChild()->division(this->getInputDataRightChild());
//                    }
//
//                    this->result_ = this->getInputDataLeftChild();
//
//                    if (this->result_)
//                        return true;
//                    else
//                        return false;
//                }
//
//                virtual ~CPU_ColumnAlgebraOperation() {
//                }
//            private:
//                ColumnAlgebraOperation op_;
//                MaterializationStatus mat_stat_;
//            };
//
//            column_processing::cpu::Physical_Operator_Map_Ptr
//            map_init_function_cpu_column_algebra_operator();
//            column_processing::cpu::TypedOperatorPtr
//            create_CPU_ColumnAlgebraOperator(column_processing::cpu::TypedLogicalNode&
//            logical_node, const hype::SchedulingDecision&,
//            column_processing::cpu::TypedOperatorPtr left_child,
//            column_processing::cpu::TypedOperatorPtr right_child);
//
//            //			GPU_ColumnWise_TypedOperatorPtr
//            create_GPU_ColumnAlgebraOperator(GPU_ColumnWise_TypedLogicalNode&
//            logical_node, const hype::SchedulingDecision&,
//            GPU_ColumnWise_TypedOperatorPtr left_child,
//            GPU_ColumnWise_TypedOperatorPtr right_child);
//
//        }//end namespace physical_operator
//
//        namespace logical_operator {
//
//            class Logical_CPU_ColumnAlgebraOperation : public
//            hype::queryprocessing::TypedNode_Impl<ColumnPtr,
//            physical_operator::map_init_function_cpu_column_algebra_operator>
//            //init_function_Join_operator>
//            {
//            public:
//
//                Logical_CPU_ColumnAlgebraOperation(ColumnAlgebraOperation op,
//                        MaterializationStatus mat_stat = MATERIALIZE,
//                        hype::DeviceConstraint dev_constr =
//                        hype::DeviceConstraint(hype::CPU_ONLY))
//                : TypedNode_Impl<ColumnPtr,
//                physical_operator::map_init_function_cpu_column_algebra_operator>(false,
//                dev_constr),
//                op_(op),
//                mat_stat_(mat_stat) {
//                }
//
//                virtual unsigned int getOutputResultSize() const {
//                    return 0;
//                }
//
//                virtual double getCalculatedSelectivity() const {
//                    return 1;
//                }
//
//                virtual std::string getOperationName() const {
//                    return "CPU_ColumnAlgebra_Operator"; //util::getName(op_);
//                }
//
//                virtual ColumnAlgebraOperation getColumnAlgebraOperation()
//                const {
//                    return op_;
//                }
//
//                const MaterializationStatus& getMaterializationStatus() const
//                {
//                    return mat_stat_;
//                }
//            private:
//                ColumnAlgebraOperation op_;
//                MaterializationStatus mat_stat_;
//            };
//
//        }//end namespace logical_operator
//
//    }//end namespace query_processing
//
//}  //end namespace CogaDB
