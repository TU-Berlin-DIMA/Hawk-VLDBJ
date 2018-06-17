#pragma once

#include <core/runtime_configuration.hpp>
#include <core/selection_expression.hpp>
#include <hardware_optimizations/primitives.hpp>
#include <query_processing/definitions.hpp>
//#include <query_processing/query_processor.hpp>
#include <optimizer/join_order_optimization.hpp>

//#define CHAIN_JOIN_USE_POSITIONLIST_ONLY_PLANS

namespace CoGaDB {
  namespace query_processing {

    // contains for each tale in the join path, which filter criteria has to be
    // applied before the join
    typedef std::map<std::string, KNF_Selection_Expression> SelectionMap;

    struct ChainJoinSpecification {
      ChainJoinSpecification(const optimizer::JoinPath& a_join_path,
                             const SelectionMap& a_sel_map)
          : join_path(a_join_path), sel_map(a_sel_map) {}

      optimizer::JoinPath join_path;
      SelectionMap sel_map;
    };

    namespace physical_operator {

      class Chain_Join_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        Chain_Join_Operator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            const ChainJoinSpecification& chain_join_specification,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint(),  // hype::DeviceConstraint(),
            MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              chain_join_specification_(chain_join_specification),
              dev_constr_(dev_constr) {}

        virtual bool execute();

        virtual ~Chain_Join_Operator() {}

       private:
        ChainJoinSpecification chain_join_specification_;
        hype::DeviceConstraint dev_constr_;
      };

      Physical_Operator_Map_Ptr map_init_function_chain_join_operator();
      TypedOperatorPtr create_Chain_Join_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_InvisibleJoin_operator;

    // Map_Init_Function
    // init_function_InvisibleJoin_operator=physical_operator::map_init_function_InvisibleJoin_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_ChainJoin
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_chain_join_operator>  // init_function_InvisibleJoin_operator>
      {
       public:
        Logical_ChainJoin(
            const ChainJoinSpecification& chain_join_specification,
            MaterializationStatus mat_stat = LOOKUP,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint()  // = hype::DeviceConstraint()
            )
            : TypedNode_Impl<
                  TablePtr,
                  physical_operator::map_init_function_chain_join_operator>(
                  false, dev_constr),
              chain_join_specification_(chain_join_specification),
              mat_stat_(mat_stat) {}

        virtual unsigned int getOutputResultSize() const { return 10; }

        virtual double getCalculatedSelectivity() const { return 0.1; }

        virtual std::string getOperationName() const { return "CHAIN_JOIN"; }
        std::string toString(bool verbose) const {
          std::string result = "CHAIN_JOIN";

          if (verbose) {
            std::list<optimizer::PartialJoinSpecification>::const_iterator it;
            SelectionMap::const_iterator sel_map_it;
            for (unsigned int i = 0;
                 i < chain_join_specification_.join_path.second.size(); ++i)
              result += " (";

            sel_map_it = this->chain_join_specification_.sel_map.find(
                chain_join_specification_.join_path.first);
            if (sel_map_it != chain_join_specification_.sel_map.end()) {
              if (!sel_map_it->second.disjunctions.empty()) {
                result += "SELECTION";
                result += it->second.toString();  //  knf_sel_expr.toString();
                result += " ON ";
              }
            }
            result += chain_join_specification_.join_path.first;

            for (it = chain_join_specification_.join_path.second.begin();
                 it != chain_join_specification_.join_path.second.end(); ++it) {
              sel_map_it =
                  this->chain_join_specification_.sel_map.find(it->first);
              result += " JOIN  ";
              if (sel_map_it != chain_join_specification_.sel_map.end()) {
                if (!sel_map_it->second.disjunctions.empty()) {
                  result += "( SELECTION ";
                  result += sel_map_it->second
                                .toString();  //  knf_sel_expr.toString();
                  result += " ON ";
                }
              }
              // if(it!=(--chain_join_specification_.join_path.second.end())){

              // result+=it->second.toString();
              // result+=")";
              //}
              result += it->first;  // table_name;
              if (sel_map_it != chain_join_specification_.sel_map.end()) {
                result += " )";
              }
              // if(it!=(--chain_join_specification_.join_path.second.end())){
              result += " ON ( ";
              result += it->second.toString();
              result += " ) )";
              //}
              // result+=", ";
            }
            result += " )";
          }

          //                    if(verbose){
          //                        result+=" (";
          //                        InvisibleJoinSelectionList::const_iterator
          //                        it;
          //                        for(it=dimensions_.begin();it!=dimensions_.end();++it){
          //                            if(!it->knf_sel_expr.disjunctions.empty()){
          //                               result+="SELECTION";
          //                               result+=it->knf_sel_expr.toString();
          //                               result+=" ON ";
          //                            }
          ////                            }else if(it!=dimensions_.begin()){
          ////                               result+=" ";
          ////                            }
          //                            result+=it->table_name;
          //                            if(it!=(--dimensions_.end()))
          //                                result+=", ";
          //                        }
          //                        result+=")";
          //                    }
          return result;
        }

        virtual bool generatesSubQueryPlan() const {
          //                    //this operator generates a subqueryplan and has
          //                    to be
          //                    //handle differently by the execution engine
          //                    return true;
          return false;
        }

        const ChainJoinSpecification& getChainJoinSpecification() {
          return chain_join_specification_;
        }

        const MaterializationStatus& getMaterializationStatus() const {
          return mat_stat_;
        }

       private:
        ChainJoinSpecification chain_join_specification_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
