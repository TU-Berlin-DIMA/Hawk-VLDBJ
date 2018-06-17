#include <hype.hpp>
#include <query_processing/chain_join_operator.hpp>
#include <query_processing/query_processor.hpp>
#include <util/hardware_detector.hpp>

#include <core/data_dictionary.hpp>
#include <lookup_table/join_index.hpp>
#include <lookup_table/lookup_table.hpp>

#include <boost/utility.hpp>

#include <algorithm>  // for copy
#include <iterator>   // for ostream_iterator
//#include <../4.6.4/bits/stl_vector.h> // for ostream_iterator

namespace CoGaDB {

namespace query_processing {
//#define CHAIN_JOIN_USE_POSITIONLIST_ONLY_PLANS
// Map_Init_Function
// init_function_ChainJoin_operator=physical_operator::map_init_function_ChainJoin_operator;

#ifndef CHAIN_JOIN_USE_POSITIONLIST_ONLY_PLANS
const query_processing::column_processing::cpu::LogicalQueryPlanPtr
createColumnBasedPlan(TablePtr fact_table,
                      const ChainJoinSpecification& chain_join_specification,
                      hype::DeviceConstraint dev_constr, std::ostream& out) {
  using namespace query_processing::column_processing::cpu;
  using namespace query_processing::column_processing;
  using namespace query_processing::logical_operator;
  using namespace query_processing;

  return query_processing::column_processing::cpu::LogicalQueryPlanPtr();
  //
  //        ChainJoinSelectionList::const_iterator it;
  //        const int NUM_DIM_TABLES=dimensions.size();
  //
  //        std::queue<query_processing::column_processing::cpu::TypedLogicalNodePtr>
  //        bitmaps_queue;
  //        for(it=dimensions.begin();it!=dimensions.end();++it){
  //            query_processing::column_processing::cpu::LogicalQueryPlanPtr
  //            complex_selection;
  //            //omit selection sub plan for dimension table were no selection
  //            is performed
  //            if(it->knf_sel_expr.disjunctions.empty()) continue;
  //
  //            complex_selection =
  //            query_processing::createColumnBasedQueryPlan(getTablebyName(it->table_name),it->knf_sel_expr,
  //            dev_constr);
  //
  //            //complex_selections->push_back(complex_selection);
  //            query_processing::column_processing::cpu::TypedLogicalNodePtr
  //            selection_sub_tree = complex_selection->getRoot();
  //            //avoid that our plan is cleaned up when the plan goes out of
  //            scope
  //            complex_selection->setNewRoot(TypedLogicalNodePtr());
  //            //if(RuntimeConfiguration::instance().)
  ////            PK_FK_Join_Predicate pk_fk_join_pred(it->table_name,
  /// it->join_pred.getColumn1Name(), fact_table->getName(),
  /// it->join_pred.getColumn2Name());
  ////
  /// boost::shared_ptr<query_processing::logical_operator::Logical_Column_Fetch_Join>
  /// fetch_join (new
  /// query_processing::logical_operator::Logical_Column_Fetch_Join(pk_fk_join_pred,
  /// dev_constr));
  ////            fetch_join->setLeft(selection_sub_tree);
  ////
  ////
  /// boost::shared_ptr<Logical_Column_Convert_PositionList_To_Bitmap>
  /// tids_to_bitmap(new
  /// Logical_Column_Convert_PositionList_To_Bitmap(fact_table->getNumberofRows(),
  /// dev_constr)); //hype::CPU_ONLY)); //dev_constr));
  ////            tids_to_bitmap->setLeft(fetch_join);
  ////            bitmaps_queue.push(tids_to_bitmap);
  //
  //            PK_FK_Join_Predicate pk_fk_join_pred(it->table_name,
  //            it->join_pred.getColumn1Name(), fact_table->getName(),
  //            it->join_pred.getColumn2Name());
  //            boost::shared_ptr<query_processing::logical_operator::Logical_Column_Bitmap_Fetch_Join>
  //            fetch_join (new
  //            query_processing::logical_operator::Logical_Column_Bitmap_Fetch_Join(pk_fk_join_pred,
  //            dev_constr));
  //            fetch_join->setLeft(selection_sub_tree);
  //            bitmaps_queue.push(fetch_join);
  //        }
  //
  //
  //        //create bitmap operations (bushy tree)
  //                //merge together all tids from this disjunction
  //                while(bitmaps_queue.size()>1){
  //                    TypedLogicalNodePtr node1 =bitmaps_queue.front();
  //                    bitmaps_queue.pop();
  //                    TypedLogicalNodePtr node2=bitmaps_queue.front();
  //                    bitmaps_queue.pop();
  //                    query_processing::column_processing::cpu::TypedLogicalNodePtr
  //                    bitwise_and(new
  //                    query_processing::logical_operator::Logical_Bitmap_Operator(BITMAP_AND,LOOKUP,
  //                    dev_constr)); //hype::CPU_ONLY)); //dev_constr));
  //                    //boost::shared_ptr<logical_operator::Logical_PositionList_Operator>
  //                    tid_union(new
  //                    logical_operator::Logical_PositionList_Operator(POSITIONLIST_UNION));
  //                    bitwise_and->setLeft(node1);
  //                    bitwise_and->setRight(node2);
  //                    bitmaps_queue.push(bitwise_and);
  //                    if(!quiet && verbose && debug)
  //                    std::cout << "ADD BITMAP_AND Operator for Childs " <<
  //                    node1->toString() << " and " << node2->toString() <<
  //                    std::endl;
  //                }
  //
  //                boost::shared_ptr<Logical_Column_Convert_Bitmap_To_PositionList>
  //                bitmap_to_tids(new
  //                Logical_Column_Convert_Bitmap_To_PositionList(dev_constr));
  //                //hype::CPU_ONLY)); //dev_constr));
  //                bitmap_to_tids->setLeft(bitmaps_queue.front());
  //
  //
  //                return
  //                query_processing::column_processing::cpu::LogicalQueryPlanPtr(new
  //                query_processing::column_processing::cpu::LogicalQueryPlan(bitmap_to_tids,
  //                out));
}

#else
const query_processing::column_processing::cpu::LogicalQueryPlanPtr
createColumnBasedPlan(TablePtr fact_table,
                      const ChainJoinSelectionList& dimensions,
                      hype::DeviceConstraint dev_constr) {
  using namespace query_processing::column_processing::cpu;
  using namespace query_processing::column_processing;
  using namespace query_processing::logical_operator;
  using namespace query_processing;

  ChainJoinSelectionList::const_iterator it;
  const int NUM_DIM_TABLES = dimensions.size();

  std::queue<query_processing::column_processing::cpu::TypedLogicalNodePtr>
      positionlist_queue;
  for (it = dimensions.begin(); it != dimensions.end(); ++it) {
    query_processing::column_processing::cpu::LogicalQueryPlanPtr
        complex_selection;
    // omit selection sub plan for dimension table were no selection is
    // performed
    if (it->knf_sel_expr.disjunctions.empty()) continue;

    complex_selection = query_processing::createColumnBasedQueryPlan(
        getTablebyName(it->table_name), it->knf_sel_expr, dev_constr);

    // complex_selections->push_back(complex_selection);
    query_processing::column_processing::cpu::TypedLogicalNodePtr
        selection_sub_tree = complex_selection->getRoot();
    // avoid that our plan is cleaned up when the plan goes out of scope
    complex_selection->setNewRoot(TypedLogicalNodePtr());

    PK_FK_Join_Predicate pk_fk_join_pred(
        it->table_name, it->join_pred.getColumn1Name(), fact_table->getName(),
        it->join_pred.getColumn2Name());
    boost::shared_ptr<
        query_processing::logical_operator::Logical_Column_Fetch_Join>
        fetch_join(
            new query_processing::logical_operator::Logical_Column_Fetch_Join(
                pk_fk_join_pred, dev_constr));
    fetch_join->setLeft(selection_sub_tree);

    positionlist_queue.push(fetch_join);
  }

  // create bitmap operations (bushy tree)
  // merge together all tids from this disjunction
  while (positionlist_queue.size() > 1) {
    TypedLogicalNodePtr node1 = positionlist_queue.front();
    positionlist_queue.pop();
    TypedLogicalNodePtr node2 = positionlist_queue.front();
    positionlist_queue.pop();
    // query_processing::column_processing::cpu::TypedLogicalNodePtr
    // bitwise_and(new
    // query_processing::logical_operator::Logical_Bitmap_Operator(BITMAP_AND,LOOKUP,
    // hype::CPU_ONLY)); //dev_constr));
    boost::shared_ptr<logical_operator::Logical_PositionList_Operator>
        tid_intersection(new logical_operator::Logical_PositionList_Operator(
            POSITIONLIST_INTERSECTION, dev_constr));
    tid_intersection->setLeft(node1);
    tid_intersection->setRight(node2);
    positionlist_queue.push(tid_intersection);
    if (!quiet && verbose && debug)
      std::cout << "ADD PositionList Intersection Operator for Childs "
                << node1->toString() << " and " << node2->toString()
                << std::endl;
  }

  return query_processing::column_processing::cpu::LogicalQueryPlanPtr(
      new query_processing::column_processing::cpu::LogicalQueryPlan(
          positionlist_queue.front()));
}
#endif

TablePtr two_phase_physical_optimization_chain_join(
    TablePtr fact_table, const optimizer::JoinPath& join_path,
    const SelectionMap& sel_map, hype::DeviceConstraint dev_constr,
    std::ostream& out) {
  hype::ProcessingDeviceID id = hype::PD0;
  ProcessorSpecification proc_spec(id);
  JoinParam param(proc_spec, HASH_JOIN);

  TablePtr result = fact_table;
  SelectionMap::const_iterator map_it = sel_map.find(fact_table->getName());
  if (map_it != sel_map.end()) {
    result = Table::selection(result, map_it->second, LOOKUP, SERIAL);
  }

  std::list<optimizer::PartialJoinSpecification>::const_iterator jp_it;
  for (jp_it = join_path.second.begin(); jp_it != join_path.second.end();
       ++jp_it) {
    Timestamp begin = getTimestamp();
    TablePtr dimension_table = getTablebyName(jp_it->first);

    TablePtr filtered_dimension_table = dimension_table;
    map_it = sel_map.find(jp_it->first);
    if (map_it != sel_map.end()) {
      filtered_dimension_table =
          Table::selection(dimension_table, map_it->second, LOOKUP, SERIAL);
    }
    result = Table::join(
        result, jp_it->second.getColumn2Name(), filtered_dimension_table,
        jp_it->second.getColumn1Name(), param);  // HASH_JOIN, LOOKUP);
    Timestamp end = getTimestamp();
    // if(!quiet)
    std::cout << "TOPPO JOIN with " << jp_it->first << ": "
              << double(end - begin) / (1000 * 1000) << "ms" << std::endl;
    if (!quiet)
      std::cout << "Dim " << jp_it->first
                << ": #Rows: " << result->getNumberofRows() << std::endl;
  }

  return result;
}

TablePtr two_phase_physical_optimization_chain_join_plan_based(
    TablePtr fact_table, const optimizer::JoinPath& join_path,
    const SelectionMap& sel_map, hype::DeviceConstraint dev_constr,
    std::ostream& out) {
  GatherParam gather_param{ProcessorSpecification{hype::PD0}};

  const bool DEBUG_CHAIN_JOIN = false;

  std::map<std::string, PositionListPtr>
      result_tids;  // maps table_name to current positionlist
  std::map<std::string, PositionListPtr>::iterator result_map_it;
  std::list<optimizer::PartialJoinSpecification>::const_iterator jp_it;

  // prefilter input tables according to join path
  for (jp_it = join_path.second.begin(); jp_it != join_path.second.end();
       ++jp_it) {
    SelectionMap::const_iterator map_it = sel_map.find(jp_it->first);
    if (map_it != sel_map.end()) {
      query_processing::column_processing::cpu::LogicalQueryPlanPtr
          complex_selection;
      // omit selection sub plan for dimension table were no selection is
      // performed
      if (map_it->second.disjunctions.empty()) continue;

      complex_selection = query_processing::createColumnBasedQueryPlan(
          getTablebyName(map_it->first), map_it->second, dev_constr);

      query_processing::column_processing::cpu::PhysicalQueryPlanPtr phy_plan =
          complex_selection->runChoppedPlan();
      phy_plan->run();

      assert(phy_plan->getRoot().get() != NULL);
      PositionListOperator* pos_list_op =
          dynamic_cast<PositionListOperator*>(phy_plan->getRoot().get());
      assert(pos_list_op != NULL);
      assert(pos_list_op->hasResultPositionList());

      assert(pos_list_op != NULL);
      assert(pos_list_op->hasResultPositionList());
      PositionListPtr tids =
          copy_if_required(pos_list_op->getResultPositionList(),
                           CoGaDB::getMemoryID(gather_param.proc_spec));
      //    			    if (!pos_list_op->hasResultPositionList() &&
      //    pos_list_op->hasCachedResult_GPU_PositionList()) {
      //				tids =
      // gpu::copy_PositionList_device_to_host(pos_list_op->getResult_GPU_PositionList());
      //    			    } else {
      //			    	tids =
      // pos_list_op->getResultPositionList();
      //    			    }
      assert(tids != NULL);
      std::cout << "Apply Selection " << map_it->second.toString()
                << " on Table " << map_it->first << std::endl;
      // ok, store result positionlist in the map
      result_tids.insert(std::make_pair(jp_it->first, tids));
      // result_tids[jp_it->first]=tids;
    }
  }

  // fact table may already be prefiltered, and if yes, it has to be a
  // LookupTable

  // assert(fact_table->getName()==join_path.first);
  // is Lookup Table?
  if (!fact_table->isMaterialized()) {
    // ok, cast to lookup table
    LookupTablePtr lookup_table_fact =
        boost::dynamic_pointer_cast<LookupTable>(fact_table);
    assert(lookup_table_fact != NULL);
    // we expect that there could only be a selection between this operator, and
    // the scan on the fact table
    // Therefore, we expect a lookup table consisting of a single lookup column
    const std::vector<LookupColumnPtr>& lookup_columns =
        lookup_table_fact->getLookupColumns();
    assert(lookup_columns.size() == 1);
    // get positionlist and insert it into the result_tids map
    LookupColumnPtr lookup_col = lookup_columns.front();
    // first check that input table is the start of the join path
    assert(join_path.first == lookup_col->getTable()->getName());
    // insert positionlist into result_tid map
    PositionListPtr tids = lookup_col->getPositionList();
    assert(tids != NULL);
    if (DEBUG_CHAIN_JOIN)
      std::cout << "Number of Rows in table: " << fact_table->getName() << ": "
                << tids->size() << "rows" << std::endl;
    result_tids.insert(make_pair(lookup_col->getTable()->getName(), tids));
  }

  // prefiltering step complete, now comes the join phase
  TablePtr left_side_table = fact_table;
  std::set<std::string> processed_tables;
  // mark first table as processed
  processed_tables.insert(join_path.first);
  for (jp_it = join_path.second.begin(); jp_it != join_path.second.end();
       ++jp_it) {
    TablePtr right_side_table = getTablebyName(jp_it->first);

    std::string name1 = jp_it->second.getColumn2Name();  // left hand side table
    std::string name2 = jp_it->second.getColumn1Name();  // right hand side
                                                         // table

    // get the Column!
    std::list<std::pair<ColumnPtr, TablePtr> > col_list =
        DataDictionary::instance().getColumnsforColumnName(name1);
    assert(col_list.size() == 1);

    if (DEBUG_CHAIN_JOIN) {
      std::cout << "[";
      std::set<std::string>::iterator proc_tab_it = processed_tables.begin();
      for (proc_tab_it = processed_tables.begin();
           proc_tab_it != processed_tables.end(); ++proc_tab_it) {
        std::cout << *proc_tab_it;
        // is last element?
        if (proc_tab_it != processed_tables.end() &&
            boost::next(proc_tab_it) != processed_tables.end()) {
          std::cout << " JOIN ";
        }
      }
      std::cout << "] JOIN " << jp_it->first << std::endl;
    }
    // get columns from the persistent tables
    ColumnPtr left_join_column =
        getTablebyName(col_list.front().second->getName())
            ->getColumnbyName(name1);  // col_list.front().first;
    ColumnPtr right_join_column = right_side_table->getColumnbyName(name2);
    // input tables may already be filtered, and the resulting positionlist
    // needs to be applied to the columns
    ColumnPtr filered_left_join_column = left_join_column;
    ColumnPtr filered_right_join_column = right_join_column;
    // fetch values from columns if positionlist is found
    // left hand side table
    result_map_it = result_tids.find(col_list.front().second->getName());
    if (result_map_it !=
        result_tids
            .end()) {  // result_tids[col_list.front().second->getName()])
      if (DEBUG_CHAIN_JOIN)
        std::cout << "Fetch Values for column " << left_join_column->getName()
                  << std::endl;
      filered_left_join_column =
          left_join_column->gather(result_map_it->second, gather_param);
      if (DEBUG_CHAIN_JOIN)
        std::cout << "Number of Rows for column " << left_join_column->getName()
                  << ": " << filered_left_join_column->size() << std::endl;
    }
    // right hand side table
    result_map_it = result_tids.find(jp_it->first);  // name2);
    if (result_map_it != result_tids.end()) {        // result_tids[name2])
      if (DEBUG_CHAIN_JOIN)
        std::cout << "Fetch Values for column " << right_join_column->getName()
                  << std::endl;
      filered_right_join_column =
          right_join_column->gather(result_map_it->second, gather_param);
      if (DEBUG_CHAIN_JOIN)
        std::cout << "Number of Rows for column "
                  << right_join_column->getName() << ": "
                  << filered_right_join_column->size() << std::endl;
    }
    // perform actual join, we use for now the hashjoin as the default
    PositionListPairPtr join_result;
    if (filered_left_join_column->size() <= filered_right_join_column->size()) {
      join_result =
          filered_left_join_column->hash_join(filered_right_join_column);
    } else {
      join_result =
          filered_right_join_column->hash_join(filered_left_join_column);
      swap(join_result->first, join_result->second);
    }
    assert(join_result->first->size() == join_result->second->size());
    if (DEBUG_CHAIN_JOIN)
      std::cout << "Number of Rows After Join: " << join_result->first->size()
                << std::endl;

    /*
                            std::cout << "TIDS Left:" << std::endl;
                            std::copy(join_result->first->begin(),
       join_result->first->end(), std::ostream_iterator<TID>(std::cout, " "));
                            std::cout << std::endl;
                            std::cout << "TIDS right: " << std::endl;
                            std::copy(join_result->second->begin(),
       join_result->second->end(), std::ostream_iterator<TID>(std::cout, " "));
                            std::cout << std::endl;
    */

    // update positionlists of tables in join path
    // traverse over all tids in join path (current left hand side tables) and
    // update them using gather
    for (result_map_it = result_tids.begin();
         result_map_it != result_tids.end(); ++result_map_it) {
      // skip tid lists which are not yet joined (i.e., not in the join path)
      if (processed_tables.find(result_map_it->first) == processed_tables.end())
        continue;
      if (DEBUG_CHAIN_JOIN)
        std::cout << "Update PositionList of left hand side Table "
                  << result_map_it->first << "...("
                  << result_map_it->second->size() << " rows)" << std::endl;
      result_map_it->second = CDK::util::gather(
          result_map_it->second,
          join_result->first);  // join_result->second); //join_result->first);
      if (DEBUG_CHAIN_JOIN)
        std::cout << "Table " << result_map_it->first << " has now "
                  << result_map_it->second->size() << " rows" << std::endl;
      assert(result_map_it->second != NULL);
    }
    processed_tables.insert(jp_it->first);
    // first iteration, add left hand side join partner's tids from join:
    if (processed_tables.size() == 2)
      result_tids.insert(
          make_pair(col_list.front().second->getName(), join_result->first));

    result_map_it = result_tids.find(jp_it->first);
    if (DEBUG_CHAIN_JOIN)
      std::cout << "right hand side table: " << jp_it->first << std::endl;
    // positionlist of right side table, by default we take the join tids
    PositionListPtr right_join_column_tids = join_result->second;
    // if we notice, that we already prefiltered this table,
    // we need to perform a gather, so we only fetch values of
    // the base table that are part of the result
    if (result_map_it != result_tids.end()) {
      if (DEBUG_CHAIN_JOIN)
        std::cout << "Update PositionList of right hand side Table "
                  << result_map_it->first << "...("
                  << result_map_it->second->size() << " rows)" << std::endl;
      result_map_it->second = CDK::util::gather(
          result_map_it->second, join_result->second);  // join_result->second);
      if (DEBUG_CHAIN_JOIN)
        std::cout << "Table " << result_map_it->first << " has now "
                  << result_map_it->second->size() << " rows" << std::endl;
    }
    assert(right_join_column_tids != NULL);
    // insert current positionlist of right hand side table to result
    // positionlists
    result_tids.insert(make_pair(jp_it->first, right_join_column_tids));
    if (DEBUG_CHAIN_JOIN)
      std::cout << "ADD poslist for right side table " << jp_it->first
                << std::endl;
  }

  std::vector<LookupColumnPtr> lookup_columns;

  // concatenate schemas
  TableSchema schema;
  // concatenate lookup arrays
  ColumnVectorPtr lookup_arrays(new ColumnVector());
  for (result_map_it = result_tids.begin(); result_map_it != result_tids.end();
       ++result_map_it) {
    if (DEBUG_CHAIN_JOIN)
      std::cout << "Adding positionlist of table " << result_map_it->first
                << " to result table (" << result_map_it->second->size()
                << " rows)" << std::endl;
    // result_map_it->second  = CDK::util::gather(result_map_it->second,
    // join_result->first);
    LookupColumnPtr lookup_col(new LookupColumn(
        getTablebyName(result_map_it->first), result_map_it->second));
    ColumnVectorPtr current_lookup_arrays = lookup_col->getLookupArrays();
    TableSchema current_schema = lookup_col->getTable()->getSchema();

    lookup_columns.push_back(lookup_col);
    lookup_arrays->insert(lookup_arrays->end(), current_lookup_arrays->begin(),
                          current_lookup_arrays->end());
    schema.insert(schema.end(), current_schema.begin(), current_schema.end());
  }

  // Create Lookup Table and return!
  TablePtr result_table = LookupTablePtr(new LookupTable(
      std::string("Chain Join"), schema, lookup_columns, *lookup_arrays));
  /*
          std::cout << "Result Table (column oriented chain join):" <<
     std::endl;
          std::cout << result_table->toString() << std::endl;
          TablePtr reference_table =
     two_phase_physical_optimization_chain_join(fact_table, join_path, sel_map,
     dev_constr, out);
          std::cout << "Result Table (reference chain join):" << std::endl;
          std::cout << reference_table->toString() << std::endl;
  */
  return result_table;
}

namespace physical_operator {

bool Chain_Join_Operator::execute() {
  hype::DeviceConstraint dev_constr =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
  this->result_ = two_phase_physical_optimization_chain_join(
      this->getInputData(), this->chain_join_specification_.join_path,
      this->chain_join_specification_.sel_map, dev_constr, *this->out);
  if (this->result_) {
    setResultSize((this->result_)->getNumberofRows());
    return true;
  } else
    return false;
}

TypedOperatorPtr create_Chain_Join_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_ChainJoin& log_selection_ref =
      static_cast<logical_operator::Logical_ChainJoin&>(logical_node);
  // std::cout << "create Chain_Join_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new Chain_Join_Operator(
      sched_dec, left_child, log_selection_ref.getChainJoinSpecification(),
      log_selection_ref.getDeviceConstraint(),
      log_selection_ref.getMaterializationStatus()));
}

Physical_Operator_Map_Ptr map_init_function_chain_join_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for INVISIBLE JOIN operator!"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SELECTION","CPU_ChainJoin_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("SELECTION","GPU_ChainJoin_Algorithm",hype::GPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_ChainJoin_Algorithm", "CHAIN_JOIN", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_ChainJoin_Algorithm", "CHAIN_JOIN", hype::Least_Squares_1D,
      hype::Periodic);
#endif

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu,
                                               dev_specs[i]);
      // hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu_parallel,dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
      // hype::Scheduler::instance().addAlgorithm(selection_alg_spec_gpu,dev_specs[i]);
    }
  }

  map["CPU_ChainJoin_Algorithm"] = create_Chain_Join_Operator;
  // map["CPU_ParallelChainJoin_Algorithm"]=create_CPU_ParallelChainJoin_Operator;
  // map["GPU_ChainJoin_Algorithm"]=create_GPU_ChainJoin_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
