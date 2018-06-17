/*
 * File:   indexed_tuple_reconstruction.hpp
 * Author: sebastian
 *
 * Created on 9. Januar 2015, 19:30
 */

#pragma once
#ifndef INDEXED_TUPLE_RECONSTRUCTION_HPP
#define INDEXED_TUPLE_RECONSTRUCTION_HPP

#include <core/runtime_configuration.hpp>
#include <core/selection_expression.hpp>
#include <query_processing/definitions.hpp>
#include <query_processing/invisible_join_operator.hpp>
//#include <hardware_optimizations/primitives.hpp>

//#define CHAIN_JOIN_USE_POSITIONLIST_ONLY_PLANS

namespace CoGaDB {

  struct IndexedTupleReconstructionParam {
    IndexedTupleReconstructionParam(PositionListPtr _fact_table_tids,
                                    InvisibleJoinSelectionList _dimensions)
        : fact_table_tids(_fact_table_tids), dimensions(_dimensions) {}
    PositionListPtr fact_table_tids;
    InvisibleJoinSelectionList dimensions;
  };

  //    TablePtr reconstruct_tuples_reverse_join_index(TablePtr fact_table,
  //    PositionListPtr fact_table_tids, IndexedTupleReconstructionParam
  //    dimensions, hype::DeviceConstraint dev_constr, std::ostream& out);
  TablePtr reconstruct_tuples_reverse_join_index(
      TablePtr fact_table, PositionListPtr fact_table_tids,
      InvisibleJoinSelectionList dimensions,
      const ProcessorSpecification& proc_spec, std::ostream& out);

  namespace query_processing {

    namespace physical_operator {

      class IndexedTupleReconstructionOperator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        IndexedTupleReconstructionOperator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            const IndexedTupleReconstructionParam& dimensions,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint(),  // hype::DeviceConstraint(),
            MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              param_(dimensions),
              dev_constr_(dev_constr) {}

        virtual bool execute();

        virtual ~IndexedTupleReconstructionOperator() {}

       private:
        IndexedTupleReconstructionParam param_;
        hype::DeviceConstraint dev_constr_;
      };

      Physical_Operator_Map_Ptr
      map_init_function_indexed_tuple_reconstruction_operator();
      TypedOperatorPtr create_Indexed_Tuple_Reconstruction_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_InvisibleJoin_operator;

    // Map_Init_Function
    // init_function_InvisibleJoin_operator=physical_operator::map_init_function_InvisibleJoin_operator;
    // //boost::bind();

    namespace logical_operator {

      class Logical_IndexedTupleReconstruction
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_indexed_tuple_reconstruction_operator>  // init_function_InvisibleJoin_operator>
      {
       public:
        Logical_IndexedTupleReconstruction(
            const IndexedTupleReconstructionParam& dimensions,
            MaterializationStatus mat_stat = LOOKUP,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint()  // = hype::DeviceConstraint()
            )
            : TypedNode_Impl<
                  TablePtr,
                  physical_operator::
                      map_init_function_indexed_tuple_reconstruction_operator>(
                  false, dev_constr),
              param_(dimensions),
              mat_stat_(mat_stat) {}

        virtual unsigned int getOutputResultSize() const { return 10; }

        virtual double getCalculatedSelectivity() const { return 0.1; }

        virtual std::string getOperationName() const {
          return "INDEXED_TUPLE_RECONSTRUCTION";
        }

        std::string toString(bool verbose) const {
          std::string result = "INDEXED_TUPLE_RECONSTRUCTION";

          //                    if(verbose){
          //                        std::list<optimizer::PartialJoinSpecification>::const_iterator
          //                        it;
          //                        SelectionMap::const_iterator sel_map_it;
          //                        for(unsigned int
          //                        i=0;i<chain_join_specification_.join_path.second.size();++i)
          //                            result+=" (";
          //
          //                        sel_map_it=this->chain_join_specification_.sel_map.find(chain_join_specification_.join_path.first);
          //                        if(sel_map_it!=chain_join_specification_.sel_map.end()){
          //                           if(!sel_map_it->second.disjunctions.empty()){
          //                                result+="SELECTION";
          //                                result+=it->second.toString(); //
          //                                knf_sel_expr.toString();
          //                                result+=" ON ";
          //                           }
          //                        }
          //                        result+=
          //                        chain_join_specification_.join_path.first;
          //
          //                        for(it=chain_join_specification_.join_path.second.begin();it!=chain_join_specification_.join_path.second.end();++it){
          //                            sel_map_it=this->chain_join_specification_.sel_map.find(it->first);
          //                            result+=" JOIN  ";
          //                            if(sel_map_it!=chain_join_specification_.sel_map.end()){
          //                               if(!sel_map_it->second.disjunctions.empty()){
          //                                    result+="( SELECTION ";
          //                                    result+=sel_map_it->second.toString();
          //                                    //  knf_sel_expr.toString();
          //                                    result+=" ON ";
          //                               }
          //
          //                            }
          //                            //if(it!=(--chain_join_specification_.join_path.second.end())){
          //
          //                                //result+=it->second.toString();
          //                                //result+=")";
          //                            //}
          //                            result+=it->first; // table_name;
          //                            if(sel_map_it!=chain_join_specification_.sel_map.end()){
          //                                result+=" )";
          //                            }
          //                            //if(it!=(--chain_join_specification_.join_path.second.end())){
          //                                result+=" ON ( ";
          //                                result+=it->second.toString();
          //                                result+=" ) )";
          //                            //}
          //                                //result+=", ";
          //                        }
          //                        result+=" )";
          //                    }
          //
          //                    if(verbose){
          //                        result+=" (";
          //                        IndexedTupleReconstructionParam::const_iterator
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

        const IndexedTupleReconstructionParam&
        getIndexedTupleReconstructionParam() {
          return param_;
        }

        const MaterializationStatus& getMaterializationStatus() const {
          return mat_stat_;
        }

       private:
        IndexedTupleReconstructionParam param_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB

#endif /* INDEXED_TUPLE_RECONSTRUCTION_HPP */
