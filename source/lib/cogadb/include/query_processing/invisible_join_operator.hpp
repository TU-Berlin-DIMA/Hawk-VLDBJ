#pragma once

#include <core/runtime_configuration.hpp>
#include <core/selection_expression.hpp>
#include <hardware_optimizations/primitives.hpp>
#include <query_processing/definitions.hpp>
#include "query_processor.hpp"

//#define INVISIBLE_JOIN_USE_POSITIONLIST_ONLY_PLANS

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_InvisibleJoin_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_InvisibleJoin_Operator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            const InvisibleJoinSelectionList& dimensions,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint(),  // hype::DeviceConstraint(),
            MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              dimensions_(dimensions),
              dev_constr_(dev_constr) {}

        virtual bool execute();
        //                {
        //
        //
        //
        //#ifdef ENABLE_TWO_PHASE_PHYSICAL_OPTIMIZATION
        //                    this->result_ =
        //                    query_processing::two_phase_physical_optimization_selection(this->getInputData(),
        //                    dimensions_, mat_stat_);
        //#else
        //                    this->result_ =
        //                    BaseTable::selection(this->getInputData(),
        //                    dimensions_, mat_stat_, SERIAL);
        //#endif
        //                    if (this->result_)
        //                        return true;
        //                    else
        //                        return false;
        //                }

        virtual ~CPU_InvisibleJoin_Operator() {}

       private:
        InvisibleJoinSelectionList dimensions_;
        hype::DeviceConstraint dev_constr_;
      };

      Physical_Operator_Map_Ptr map_init_function_invisible_join_operator();
      TypedOperatorPtr create_CPU_InvisibleJoin_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_InvisibleJoin_operator;

    // Map_Init_Function
    // init_function_InvisibleJoin_operator=physical_operator::map_init_function_InvisibleJoin_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_InvisibleJoin
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_invisible_join_operator>  // init_function_InvisibleJoin_operator>
      {
       public:
        Logical_InvisibleJoin(
            const InvisibleJoinSelectionList& dimensions,
            MaterializationStatus mat_stat = LOOKUP,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint()  // = hype::DeviceConstraint()
            )
            : TypedNode_Impl<
                  TablePtr,
                  physical_operator::map_init_function_invisible_join_operator>(
                  false, dev_constr),
              dimensions_(dimensions),
              mat_stat_(mat_stat) {}

        virtual unsigned int getOutputResultSize() const { return 10; }

        virtual double getCalculatedSelectivity() const { return 0.1; }

        virtual std::string getOperationName() const {
          return "INVISIBLE_JOIN";
        }
        std::string toString(bool verbose) const {
          std::string result = "INVISIBLE_JOIN";
          if (verbose) {
            result += " (";
            InvisibleJoinSelectionList::const_iterator it;
            for (it = dimensions_.begin(); it != dimensions_.end(); ++it) {
              if (!it->knf_sel_expr.disjunctions.empty()) {
                result += "SELECTION";
                result += it->knf_sel_expr.toString();
                result += " ON ";
              }
              //                            }else if(it!=dimensions_.begin()){
              //                               result+=" ";
              //                            }
              result += it->table_name;
              if (it != (--dimensions_.end())) result += ", ";
            }
            result += ")";
          }
          return result;
        }

        virtual bool generatesSubQueryPlan() const {
          // this operator generates a subqueryplan and has to be
          // handle differently by the execution engine
          return true;
        }

        const InvisibleJoinSelectionList& getInvisibleJoinSelectionList() {
          return dimensions_;
        }

        const MaterializationStatus& getMaterializationStatus() const {
          return mat_stat_;
        }

        //                virtual TypedOperatorPtr
        //                getOptimalOperator(TypedOperatorPtr left_child,
        //                TypedOperatorPtr right_child,
        //                hype::DeviceTypeConstraint dev_constr){
        //                    return TypedOperatorPtr(new
        //                    CPU_InvisibleJoin_Operator(sched_dec,
        //                                                            left_child,
        //                                                            log_selection_ref.getInvisibleJoinSelectionList(),
        //                                                            log_selection_ref.getDeviceConstraint(),
        //                                                            log_selection_ref.getMaterializationStatus())
        //                                                            );
        //                }

       private:
        InvisibleJoinSelectionList dimensions_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
