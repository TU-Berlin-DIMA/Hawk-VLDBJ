#pragma once

#include <core/data_dictionary.hpp>
#include <core/processor_data_cache.hpp>
#include <core/runtime_configuration.hpp>
#include <core/selection_expression.hpp>
#include <query_processing/definitions.hpp>
#include <util/hardware_detector.hpp>

#include "query_processor.hpp"

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_ComplexSelection_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_ComplexSelection_Operator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            const KNF_Selection_Expression& knf_expr,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint(),  // hype::DeviceConstraint(),
            MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              knf_expr_(knf_expr),
              dev_constr_(dev_constr),
              mat_stat_(mat_stat) {}

        virtual bool execute();
        //                {
        //
        //
        //
        //#ifdef ENABLE_TWO_PHASE_PHYSICAL_OPTIMIZATION
        //                    this->result_ =
        //                    query_processing::two_phase_physical_optimization_selection(this->getInputData(),
        //                    knf_expr_, mat_stat_);
        //#else
        //                    this->result_ =
        //                    BaseTable::selection(this->getInputData(),
        //                    knf_expr_, mat_stat_, SERIAL);
        //#endif
        //                    if (this->result_)
        //                        return true;
        //                    else
        //                        return false;
        //                }

        virtual ~CPU_ComplexSelection_Operator() {}

       private:
        KNF_Selection_Expression knf_expr_;
        hype::DeviceConstraint dev_constr_;
        MaterializationStatus mat_stat_;
      };

      Physical_Operator_Map_Ptr map_init_function_complex_selection_operator();
      TypedOperatorPtr create_CPU_ComplexSelection_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_ComplexSelection_operator;

    // Map_Init_Function
    // init_function_ComplexSelection_operator=physical_operator::map_init_function_ComplexSelection_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_ComplexSelection
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_complex_selection_operator>  // init_function_ComplexSelection_operator>
      {
       public:
        Logical_ComplexSelection(
            const KNF_Selection_Expression& knf_expr,
            MaterializationStatus mat_stat = MATERIALIZE,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint()  // = hype::DeviceConstraint()
            )
            : TypedNode_Impl<TablePtr,
                             physical_operator::
                                 map_init_function_complex_selection_operator>(
                  false, dev_constr),
              knf_expr_(knf_expr),
              mat_stat_(mat_stat),
              could_not_be_pushed_down_further_(false) {}

        virtual bool generatesSubQueryPlan() const {
          // this operator generates a subqueryplan and has to be
          // handle differently by the execution engine
          // the engine generates subplans for nonprimitive selections,
          // i.e., selectiosn with more than one predicate
          if (this->knf_expr_.disjunctions.size() > 1) {
            return true;
          } else {
            // a disjunction may not be empty
            assert(!this->knf_expr_.disjunctions.empty());
            // in case of a single predicate, the engine will
            // not generate a subplan
            if (knf_expr_.disjunctions.front().size() > 1) {
              return true;
            } else {
              return false;
            }
          }
        }

        virtual unsigned int getOutputResultSize() const {
          if (this->getLeft()) {
            return this->getLeft()->getOutputResultSize() *
                   this->getCalculatedSelectivity();
          } else {
            return 10;
          }
        }

        virtual double getCalculatedSelectivity() const;
        //                {
        //                    return 0.1;
        //                }

        virtual std::string getOperationName() const {
          return "COMPLEX_SELECTION";
        }
        std::string toString(bool verbose) const {
          std::string result = "COMPLEX_SELECTION";
          if (verbose) {
            result += knf_expr_.toString();
          }
          return result;
        }
        const KNF_Selection_Expression& getKNF_Selection_Expression() {
          return knf_expr_;
        }

        const MaterializationStatus& getMaterializationStatus() const {
          return mat_stat_;
        }
        bool couldNotBePushedDownFurther() {
          return could_not_be_pushed_down_further_;
        }
        void couldNotBePushedDownFurther(bool val) {
          could_not_be_pushed_down_further_ = val;
        }

        bool isInputDataCachedInGPU();

        const std::list<std::string> getNamesOfReferencedColumns() const;

        //                const hype::Tuple getFeatureVector() const;

        //                virtual TypedOperatorPtr
        //                getOptimalOperator(TypedOperatorPtr left_child,
        //                TypedOperatorPtr right_child,
        //                hype::DeviceTypeConstraint dev_constr){
        //                    return TypedOperatorPtr(new
        //                    CPU_ComplexSelection_Operator(sched_dec,
        //                                                            left_child,
        //                                                            log_selection_ref.getKNF_Selection_Expression(),
        //                                                            log_selection_ref.getDeviceConstraint(),
        //                                                            log_selection_ref.getMaterializationStatus())
        //                                                            );
        //                }

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
        KNF_Selection_Expression knf_expr_;
        MaterializationStatus mat_stat_;
        /* \brief Bit that shows the optimizer that this selection could not be
         * pushed down more in the query plan*/
        bool could_not_be_pushed_down_further_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
