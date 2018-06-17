
#include <query_processing/invisible_join_operator.hpp>
#include <query_processing/query_processor.hpp>
#include <util/hardware_detector.hpp>

#include <core/data_dictionary.hpp>
#include <lookup_table/join_index.hpp>
#include <lookup_table/lookup_table.hpp>
#include <optimizer/join_order_optimization.hpp>

#include <query_processing/chain_join_operator.hpp>

#include <core/variable_manager.hpp>

namespace CoGaDB {

namespace query_processing {
//#define INVISIBLE_JOIN_USE_POSITIONLIST_ONLY_PLANS
// Map_Init_Function
// init_function_InvisibleJoin_operator=physical_operator::map_init_function_InvisibleJoin_operator;

#ifndef INVISIBLE_JOIN_USE_POSITIONLIST_ONLY_PLANS
const query_processing::column_processing::cpu::LogicalQueryPlanPtr
createColumnBasedPlan(TablePtr fact_table,
                      const InvisibleJoinSelectionList& dimensions,
                      hype::DeviceConstraint dev_constr, std::ostream& out) {
  using namespace query_processing::column_processing::cpu;
  using namespace query_processing::column_processing;
  using namespace query_processing::logical_operator;
  using namespace query_processing;

  InvisibleJoinSelectionList::const_iterator it;

  std::queue<query_processing::column_processing::cpu::TypedLogicalNodePtr>
      bitmaps_queue;
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
    // if(RuntimeConfiguration::instance().)
    //            PK_FK_Join_Predicate pk_fk_join_pred(it->table_name,
    //            it->join_pred.getColumn1Name(), fact_table->getName(),
    //            it->join_pred.getColumn2Name());
    //            boost::shared_ptr<query_processing::logical_operator::Logical_Column_Fetch_Join>
    //            fetch_join (new
    //            query_processing::logical_operator::Logical_Column_Fetch_Join(pk_fk_join_pred,
    //            dev_constr));
    //            fetch_join->setLeft(selection_sub_tree);
    //
    //            boost::shared_ptr<Logical_Column_Convert_PositionList_To_Bitmap>
    //            tids_to_bitmap(new
    //            Logical_Column_Convert_PositionList_To_Bitmap(fact_table->getNumberofRows(),
    //            dev_constr)); //hype::CPU_ONLY)); //dev_constr));
    //            tids_to_bitmap->setLeft(fetch_join);
    //            bitmaps_queue.push(tids_to_bitmap);

    PK_FK_Join_Predicate pk_fk_join_pred(
        it->table_name, it->join_pred.getColumn1Name(), fact_table->getName(),
        it->join_pred.getColumn2Name());
    boost::shared_ptr<
        query_processing::logical_operator::Logical_Column_Bitmap_Fetch_Join>
        fetch_join(
            new query_processing::logical_operator::
                Logical_Column_Bitmap_Fetch_Join(pk_fk_join_pred, dev_constr));
    fetch_join->setLeft(selection_sub_tree);
    bitmaps_queue.push(fetch_join);
  }

  // create bitmap operations (bushy tree)
  // merge together all tids from this disjunction
  while (bitmaps_queue.size() > 1) {
    TypedLogicalNodePtr node1 = bitmaps_queue.front();
    bitmaps_queue.pop();
    TypedLogicalNodePtr node2 = bitmaps_queue.front();
    bitmaps_queue.pop();
    query_processing::column_processing::cpu::TypedLogicalNodePtr bitwise_and(
        new query_processing::logical_operator::Logical_Bitmap_Operator(
            BITMAP_AND, LOOKUP,
            dev_constr));  // hype::CPU_ONLY)); //dev_constr));
    // boost::shared_ptr<logical_operator::Logical_PositionList_Operator>
    // tid_union(new
    // logical_operator::Logical_PositionList_Operator(POSITIONLIST_UNION));
    bitwise_and->setLeft(node1);
    bitwise_and->setRight(node2);
    bitmaps_queue.push(bitwise_and);
    if (!quiet && verbose && debug)
      std::cout << "ADD BITMAP_AND Operator for Childs " << node1->toString()
                << " and " << node2->toString() << std::endl;
  }

  boost::shared_ptr<Logical_Column_Convert_Bitmap_To_PositionList>
      bitmap_to_tids(new Logical_Column_Convert_Bitmap_To_PositionList(
          dev_constr));  // hype::CPU_ONLY)); //dev_constr));
  bitmap_to_tids->setLeft(bitmaps_queue.front());

  return query_processing::column_processing::cpu::LogicalQueryPlanPtr(
      new query_processing::column_processing::cpu::LogicalQueryPlan(
          bitmap_to_tids, out));
}

#else
const query_processing::column_processing::cpu::LogicalQueryPlanPtr
createColumnBasedPlan(TablePtr fact_table,
                      const InvisibleJoinSelectionList& dimensions,
                      hype::DeviceConstraint dev_constr, std::ostream& out) {
  using namespace query_processing::column_processing::cpu;
  using namespace query_processing::column_processing;
  using namespace query_processing::logical_operator;
  using namespace query_processing;

  InvisibleJoinSelectionList::const_iterator it;
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
          positionlist_queue.front(), out));
}
#endif

//    void pk_fk_semi_join_thread(TablePtr  filtered_fact_tab,
//    InvisibleJoinSelection inv_join_sel, LookupTablePtr* result, unsigned int
//    thread_id){
//
//                TablePtr dim_table = getTablebyName(inv_join_sel.table_name);
//                TablePtr join_result = BaseTable::pk_fk_join(dim_table,
//                inv_join_sel.join_pred.getColumn1Name(), filtered_fact_tab,
//                inv_join_sel.join_pred.getColumn2Name(), HASH_JOIN, LOOKUP);
//
//                std::cout << "Dim " << inv_join_sel.table_name << ": #Rows: "
//                <<  join_result->getNumberofRows() << std::endl;
//                std::list<std::string> columns_dim_table;
//                TableSchema schema = dim_table->getSchema();
//                TableSchema::iterator sit;
//                for(sit=schema.begin();sit!=schema.end();++sit){
//                    columns_dim_table.push_back(sit->second);
//                }
//
//                TablePtr dimension_table_semi_join =
//                BaseTable::projection(join_result, columns_dim_table,
//                MATERIALIZE, CPU);
//                LookupTablePtr lookup_table =
//                boost::dynamic_pointer_cast<LookupTable>(dimension_table_semi_join);
//                //dimension_semi_joins.push_back(lookup_table);
//                *result=lookup_table;
//     }

void semi_join_from_join_index_thread(TablePtr fact_table,
                                      PositionListPtr fact_table_tids,
                                      InvisibleJoinSelection inv_join_sel,
                                      LookupTablePtr* result,
                                      unsigned int thread_id) {
  hype::ProcessingDeviceID id = hype::PD0;
  ProcessorSpecification proc_spec(id);
  TablePtr dim_table = getTablebyName(inv_join_sel.table_name);
  JoinIndexPtr join_index = JoinIndexes::instance().getJoinIndex(
      fact_table, inv_join_sel.join_pred.getColumn2Name(), dim_table,
      inv_join_sel.join_pred.getColumn1Name());
  PositionListPtr matching_dim_table_tids =
      CoGaDB::fetchMatchingTIDsFromJoinIndex(join_index, fact_table_tids);
  std::cout << "#Matching Rows in DimensionTable " << inv_join_sel.table_name
            << ": " << matching_dim_table_tids->size() << std::endl;
  LookupTablePtr lookup_table = createLookupTableforUnaryOperation(
      "", dim_table, matching_dim_table_tids, proc_spec);
  *result = lookup_table;
}

// called after invisible join
TablePtr reconstruct_tuples_chain_join(TablePtr filtered_fact_table,
                                       InvisibleJoinSelectionList dimensions,
                                       hype::DeviceConstraint dev_constr,
                                       std::ostream& out) {
  // TODO: we can just fetch values here if we know that the corresponding
  // primary key column is a dense value column starting from 0

  // generic case, just perform the join between the prefiltered fact table and
  // the dimension tables
  optimizer::JoinPath join_path;
  SelectionMap sel_map;
  InvisibleJoinSelectionList::iterator it;

  join_path.first = filtered_fact_table->getName();
  for (it = dimensions.begin(); it != dimensions.end(); ++it) {
    sel_map.insert(make_pair(it->table_name, it->knf_sel_expr));
    join_path.second.push_back(
        optimizer::PartialJoinSpecification(it->table_name, it->join_pred));
  }
  // scan the filtered fact table
  boost::shared_ptr<query_processing::logical_operator::Logical_Scan>
      scan_fact_table(new query_processing::logical_operator::Logical_Scan(
          filtered_fact_table));
  // and join it with the other tables
  boost::shared_ptr<query_processing::logical_operator::Logical_ChainJoin>
      chain_join(new query_processing::logical_operator::Logical_ChainJoin(
          query_processing::ChainJoinSpecification(join_path, sel_map), LOOKUP,
          CoGaDB::RuntimeConfiguration::instance()
              .getGlobalDeviceConstraint()));

  chain_join->setLeft(scan_fact_table);
  scan_fact_table->setParent(chain_join);

  query_processing::LogicalQueryPlanPtr log_plan(
      new query_processing::LogicalQueryPlan(chain_join, out));
  if (RuntimeConfiguration::instance().getProfileQueries() ||
      RuntimeConfiguration::instance().getProfileQueries()) {
    log_plan->print();
  }
  query_processing::PhysicalQueryPlanPtr phy_plan;

  if (RuntimeConfiguration::instance().isQueryChoppingEnabled()) {
    Timestamp begin = getTimestamp();
    phy_plan = log_plan->runChoppedPlan();
    Timestamp end = getTimestamp();
    double total_delay_time = phy_plan->getTotalSchedulingDelay();
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      phy_plan->printResults(true, true, true);  // plan->print();
    } else if (RuntimeConfiguration::instance().getProfileQueries()) {
      phy_plan->printResults(true, false, false);  // plan->print();
      out << "\\e[31m"
          << "QEP Runtime: " << double(end - begin) / (1000 * 1000) << "ms"
          << "\\e[39m" << std::endl;
      out << "\\e[31m"
          << "QEP Expected Runtime: "
          << phy_plan->getExpectedExecutionTime() / (1000 * 1000) << "ms"
          << "\\e[39m" << std::endl;
      out << "\\e[31m"
          << "Total Scheduling Delay Time: " << total_delay_time / (1000 * 1000)
          << "ms"
          << "\\e[39m" << std::endl;
    }
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_SECONDS",
        total_delay_time / (1000 * 1000 * 1000));
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_NS", total_delay_time);
  } else {
    hype::query_optimization::QEPPtr hype_qep = log_plan->convertToQEP();
    hype::query_optimization::optimizeQueryPlan(
        *hype_qep,
        RuntimeConfiguration::instance()
            .getQueryOptimizationHeuristic());  // hype::GREEDY_HEURISTIC);//hype::BACKTRACKING);
    phy_plan = log_plan->convertQEPToPhysicalQueryPlan(hype_qep);

    Timestamp begin = getTimestamp();
    phy_plan->run();
    Timestamp end = getTimestamp();
    double total_delay_time = phy_plan->getTotalSchedulingDelay();
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      phy_plan->printResults(true, true, true);  // plan->print();
    } else if (RuntimeConfiguration::instance().getProfileQueries()) {
      phy_plan->printResults(true, false, false);  // plan->print();
      out << "\\e[31m"
          << "QEP Runtime: " << double(end - begin) / (1000 * 1000) << "ms"
          << "\\e[39m" << std::endl;
      out << "\\e[31m"
          << "Total Scheduling Delay Time: " << total_delay_time / (1000 * 1000)
          << "ms"
          << "\\e[39m" << std::endl;
    }
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_SECONDS",
        total_delay_time / (1000 * 1000 * 1000));
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_NS", total_delay_time);
  }
  return phy_plan->getResult();
}

TablePtr reconstruct_tuples(TablePtr fact_table,
                            PositionListPtr fact_table_tids,
                            InvisibleJoinSelectionList dimensions,
                            std::ostream& out) {
  IndexedTupleReconstructionParam param(fact_table_tids, dimensions);

  // scan the filtered fact table
  boost::shared_ptr<query_processing::logical_operator::Logical_Scan>
      scan_fact_table(
          new query_processing::logical_operator::Logical_Scan(fact_table));
  // and join it with the other tables
  boost::shared_ptr<
      query_processing::logical_operator::Logical_IndexedTupleReconstruction>
      tuple_reconstr(new query_processing::logical_operator::
                         Logical_IndexedTupleReconstruction(param));

  tuple_reconstr->setLeft(scan_fact_table);
  scan_fact_table->setParent(tuple_reconstr);

  query_processing::LogicalQueryPlanPtr log_plan(
      new query_processing::LogicalQueryPlan(tuple_reconstr, out));
  if (RuntimeConfiguration::instance().getProfileQueries() ||
      RuntimeConfiguration::instance().getProfileQueries()) {
    log_plan->print();
  }
  query_processing::PhysicalQueryPlanPtr phy_plan;

  if (RuntimeConfiguration::instance().isQueryChoppingEnabled()) {
    Timestamp begin = getTimestamp();
    phy_plan = log_plan->runChoppedPlan();
    Timestamp end = getTimestamp();
    double total_delay_time = phy_plan->getTotalSchedulingDelay();
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      phy_plan->printResults(true, true, true);  // plan->print();
    } else if (RuntimeConfiguration::instance().getProfileQueries()) {
      phy_plan->printResults(true, false, false);  // plan->print();
      out << "\\e[31m"
          << "QEP Runtime: " << double(end - begin) / (1000 * 1000) << "ms"
          << "\\e[39m" << std::endl;
      out << "\\e[31m"
          << "QEP Expected Runtime: "
          << phy_plan->getExpectedExecutionTime() / (1000 * 1000) << "ms"
          << "\\e[39m" << std::endl;
      out << "\\e[31m"
          << "Total Scheduling Delay Time: " << total_delay_time / (1000 * 1000)
          << "ms"
          << "\\e[39m" << std::endl;
    }
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_SECONDS",
        total_delay_time / (1000 * 1000 * 1000));
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_NS", total_delay_time);
  } else {
    hype::query_optimization::QEPPtr hype_qep = log_plan->convertToQEP();
    hype::query_optimization::optimizeQueryPlan(
        *hype_qep,
        RuntimeConfiguration::instance()
            .getQueryOptimizationHeuristic());  // hype::GREEDY_HEURISTIC);//hype::BACKTRACKING);
    phy_plan = log_plan->convertQEPToPhysicalQueryPlan(hype_qep);

    Timestamp begin = getTimestamp();
    phy_plan->run();
    Timestamp end = getTimestamp();
    double total_delay_time = phy_plan->getTotalSchedulingDelay();
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      phy_plan->printResults(true, true, true);  // plan->print();
    } else if (RuntimeConfiguration::instance().getProfileQueries()) {
      phy_plan->printResults(true, false, false);  // plan->print();
      out << "\\e[31m"
          << "QEP Runtime: " << double(end - begin) / (1000 * 1000) << "ms"
          << "\\e[39m" << std::endl;
      out << "\\e[31m"
          << "Total Scheduling Delay Time: " << total_delay_time / (1000 * 1000)
          << "ms"
          << "\\e[39m" << std::endl;
    }
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_SECONDS",
        total_delay_time / (1000 * 1000 * 1000));
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_NS", total_delay_time);
  }
  return phy_plan->getResult();
}

TablePtr two_phase_physical_optimization_invisible_join(
    TablePtr fact_table, InvisibleJoinSelectionList dimensions,
    hype::DeviceConstraint dev_constr, std::ostream& out) {
  Timestamp begin_compute_matching_fact_table_rows = getTimestamp();
  // const query_processing::column_processing::cpu::LogicalQueryPlanPtr
  // log_plan = createColumnBasedPlan(fact_table, dimensions, dev_constr);
  const query_processing::column_processing::cpu::LogicalQueryPlanPtr log_plan =
      createColumnBasedPlan(fact_table, dimensions, dev_constr, out);
  if (RuntimeConfiguration::instance().getProfileQueries() ||
      RuntimeConfiguration::instance()
          .getPrintQueryPlan()) {  // getProfileQueries()){
    log_plan->print();
  }
  query_processing::column_processing::cpu::PhysicalQueryPlanPtr phy_plan;

  if (RuntimeConfiguration::instance().isQueryChoppingEnabled()) {
    Timestamp begin = getTimestamp();
    phy_plan = log_plan->runChoppedPlan();
    Timestamp end = getTimestamp();
    double total_delay_time = phy_plan->getTotalSchedulingDelay();
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      phy_plan->printResults(true, true, true);  // plan->print();
    } else if (RuntimeConfiguration::instance().getProfileQueries()) {
      phy_plan->printResults(true, false, false);  // plan->print();
      out << "\\e[31m"
          << "QEP Runtime: " << double(end - begin) / (1000 * 1000) << "ms"
          << "\\e[39m" << std::endl;
      out << "\\e[31m"
          << "QEP Expected Runtime: "
          << phy_plan->getExpectedExecutionTime() / (1000 * 1000) << "ms"
          << "\\e[39m" << std::endl;
      out << "\\e[31m"
          << "Total Scheduling Delay Time: " << total_delay_time / (1000 * 1000)
          << "ms"
          << "\\e[39m" << std::endl;
    }
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_SECONDS",
        total_delay_time / (1000 * 1000 * 1000));
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_NS", total_delay_time);
  } else {
    hype::query_optimization::QEPPtr hype_qep = log_plan->convertToQEP();
    hype::query_optimization::optimizeQueryPlan(
        *hype_qep,
        RuntimeConfiguration::instance()
            .getQueryOptimizationHeuristic());  // hype::GREEDY_HEURISTIC);//hype::BACKTRACKING);
    phy_plan = log_plan->convertQEPToPhysicalQueryPlan(hype_qep);

    Timestamp begin = getTimestamp();
    phy_plan->run();
    Timestamp end = getTimestamp();
    double total_delay_time = phy_plan->getTotalSchedulingDelay();
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      phy_plan->printResults(true, true, true);  // plan->print();
    } else if (RuntimeConfiguration::instance().getProfileQueries()) {
      phy_plan->printResults(true, false, false);  // plan->print();
      out << "\\e[31m"
          << "QEP Runtime: " << double(end - begin) / (1000 * 1000) << "ms"
          << "\\e[39m" << std::endl;
      out << "\\e[31m"
          << "Total Scheduling Delay Time: " << total_delay_time / (1000 * 1000)
          << "ms"
          << "\\e[39m" << std::endl;
    }
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_SECONDS",
        total_delay_time / (1000 * 1000 * 1000));
    StatisticsManager::instance().addToValue(
        "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_NS", total_delay_time);
  }

  //        hype::query_optimization::QEPPtr hype_qep =
  //        log_plan->convertToQEP();
  //        hype::query_optimization::optimizeQueryPlan(*hype_qep,RuntimeConfiguration::instance().getQueryOptimizationHeuristic());
  //        //hype::GREEDY_HEURISTIC);//hype::BACKTRACKING);
  //        phy_plan = log_plan->convertQEPToPhysicalQueryPlan(hype_qep);
  //
  //        phy_plan->print();
  //        //Timestamp end_optimize_query_plan = getTimestamp();
  //
  //        //std::cout << "Time for query plan generation and optimization: "
  //        <<
  //        double(end_optimize_query_plan-begin_compute_matching_fact_table_rows)/(1000*1000)
  //        << "ms" << std::endl;
  //
  //        Timestamp begin = getTimestamp();
  //        phy_plan->run();
  //        Timestamp end = getTimestamp();
  //        std::cout << "QEP Runtime: " << double(end-begin)/(1000*1000) <<
  //        "ms" << std::endl;

  if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
    phy_plan->printResults(true, true, false);
  }

  assert(phy_plan->getRoot().get() != NULL);
  PositionListOperator* pos_list_op =
      dynamic_cast<PositionListOperator*>(phy_plan->getRoot().get());
  assert(pos_list_op != NULL);
  assert(pos_list_op->hasResultPositionList());

  assert(pos_list_op != NULL);

  hype::ProcessingDeviceID id = phy_plan->getRoot()
                                    ->getSchedulingDecision()
                                    .getDeviceSpecification()
                                    .getProcessingDeviceID();
  ProcessorSpecification proc_spec(id);

  PositionListPtr fact_table_tids =
      copy_if_required(pos_list_op->getResultPositionList(), proc_spec);

  //        assert(pos_list_op->hasResultPositionList() ||
  //        pos_list_op->hasCachedResult_GPU_PositionList());
  //        PositionListPtr fact_table_tids;
  //        if (!pos_list_op->hasResultPositionList() &&
  //        pos_list_op->hasCachedResult_GPU_PositionList()) {
  //            fact_table_tids =
  //            gpu::copy_PositionList_device_to_host(pos_list_op->getResult_GPU_PositionList());
  //        } else {
  //            fact_table_tids = pos_list_op->getResultPositionList();
  //        }
  assert(fact_table_tids != NULL);

  //        PositionListPtr fact_table_tids =
  //        input_tids->getResultPositionList();

  // assert sorted TID list
  //        for(unsigned int i=0;i<fact_table_tids->size()-1;++i){
  //            assert((*fact_table_tids)[i]<(*fact_table_tids)[i+1]);
  //        }

  LookupTablePtr filtered_fact_tab = createLookupTableforUnaryOperation(
      "InvisibleJoin(Fact_Table)", fact_table, fact_table_tids, proc_spec);
  Timestamp end_compute_matching_fact_table_rows = getTimestamp();

  //        filtered_fact_tab->concatenate()

  if (!quiet)
    out << "Size of Filtered Fact Table (invisible Join): "
        << filtered_fact_tab->getNumberofRows() << "rows" << std::endl;

  Timestamp begin_construct_result = getTimestamp();
  TablePtr result = filtered_fact_tab;

#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY

  filtered_fact_tab->setName(fact_table->getName());
  if (VariableManager::instance().getVariableValueBoolean(
          "use_indexed_tuple_reconstruction")) {
    result = reconstruct_tuples(fact_table, fact_table_tids, dimensions, out);
  } else {
    result = reconstruct_tuples_chain_join(filtered_fact_tab, dimensions,
                                           dev_constr, out);
  }
#endif

  Timestamp end_construct_result = getTimestamp();

  // if(!quiet)
  out << "TOPPO: Time for Creating Matching Rows TID List: "
      << double(end_compute_matching_fact_table_rows -
                begin_compute_matching_fact_table_rows) /
             (1000 * 1000)
      << "ms" << std::endl;
  //                cout << "Time for Combining Bitmaps: " <<
  //                double(end_combine_bitmaps-begin_combine_bitmaps)/(1000*1000)
  //                << "ms" << std::endl;
  // if(!quiet)
  out << "TOPPO: Time for Result Construction: "
      << double(end_construct_result - begin_construct_result) / (1000 * 1000)
      << "ms" << std::endl;

  return result;
}

/*
//contains for each tale in the join path, which filter criteria has to be
applied before the join
typedef std::map<std::string,KNF_Selection_Expression> SelectionMap;



TablePtr two_phase_physical_optimization_chain_join(const optimizer::JoinPath&
join_path, const SelectionMap& sel_map, hype::DeviceConstraint dev_constr,
std::ostream& out){
               std::map<std::string,PositionListPtr> result_tids;
               std::map<std::string,PositionListPtr>::iterator result_map_it;
               std::list<optimizer::PartialJoinSpecification>::const_iterator
jp_it;
               for(jp_it=join_path.second.begin();jp_it!=join_path.second.end();++jp_it){
                    //jp_it->second.getColumn2Name()
                    //jp_it->second.getColumn1Name()

                    SelectionMap::const_iterator
map_it=sel_map.find(jp_it->first);
                    if(map_it!=sel_map.end()){
                            query_processing::column_processing::cpu::LogicalQueryPlanPtr
complex_selection;
                            //omit selection sub plan for dimension table were
no selection is performed
                            if(map_it->second.disjunctions.empty()) continue;

                            complex_selection =
query_processing::createColumnBasedQueryPlan(getTablebyName(map_it->first),map_it->second,
dev_constr);

                            query_processing::column_processing::cpu::PhysicalQueryPlanPtr
phy_plan = complex_selection->runChoppedPlan();
                            phy_plan->run();

                            assert(phy_plan->getRoot().get()!=NULL);
                            PositionListOperator* pos_list_op =
dynamic_cast<PositionListOperator*>(phy_plan->getRoot().get());
                            assert(pos_list_op!=NULL);
                            assert(pos_list_op->hasResultPositionList() ||
pos_list_op->hasCachedResult_GPU_PositionList());

                            assert(pos_list_op != NULL);
                            assert(pos_list_op->hasResultPositionList() ||
pos_list_op->hasCachedResult_GPU_PositionList());
                            PositionListPtr tids;
                            if (!pos_list_op->hasResultPositionList() &&
pos_list_op->hasCachedResult_GPU_PositionList()) {
                                tids =
gpu::copy_PositionList_device_to_host(pos_list_op->getResult_GPU_PositionList());
                            } else {
                                tids = pos_list_op->getResultPositionList();
                            }
                            assert(tids != NULL);

                            //ok, store result positionlist in the map
                            result_tids[jp_it->first]=tids;

                    }


               }


               std::set<std::string> processed_tables;
                //todo: insert first table, which is per definition in the join
path, and should be marked as processed
               for(jp_it=join_path.second.begin();jp_it!=join_path.second.end();++jp_it){

                        //TablePtr table_in_join_path =
getTablebyName(jp_it->first);i
                        TablePtr right_side_table =
getTablebyName(jp_it->first);

                        std::string name1 = jp_it->second.getColumn1Name();
                        std::string name2 = jp_it->second.getColumn2Name();

                        //get the Column!
                        std::list<std::pair<ColumnPtr,TablePtr> > col_list =
DataDictionary::instance().getColumnsforColumnName(name1);
                        assert(col_list.size()==1);

                        ColumnPtr left_join_column = col_list.front().first; //
=  //= dimension_table1->getColumnbyName(name1);
                        ColumnPtr right_join_column =
right_side_table->getColumnbyName(name2);

                        ColumnPtr filered_left_join_column;
                        ColumnPtr filered_right_join_column;

                        if(result_tids[col_list.front().second->getName()])
                                filered_left_join_column =
left_join_column->gather(result_tids[col_list.front().second->getName()]);
                        if(result_tids[name2])
                                filered_right_join_column =
right_join_column->gather(result_tids[name2]);


                        PositionListPairPtr join_result =
filered_left_join_column->hash_join(filered_right_join_column);

                        //update positionlists of tables in joinpath
                        //traverse over all tids in join path and update them
using gather
                        for(result_map_it=result_tids.begin();result_map_it!=result_tids.end();++result_map_it){
                            result_map_it->second  =
CDK::util::gather(result_map_it->second, join_result->first);

                        }
                        PositionListPtr right_join_column_tids =
CDK::util::gather(result_tids[name2], join_result->second);
                        result_tids.insert(make_pair(col_list.front().second->getName(),right_join_column_tids));


//            ColumnPtr filered_left_join_column =
left_join_column->gather(result_tids[name1]);
//            ColumnPtr filered_right_join_column =
right_join_column->gather(result_tids[name2]);




                }

        std::vector<LookupColumnPtr> lookup_columns; //(new LookupColumn());

        //concatenate schemas
        TableSchema schema;

        //concatenate lookup arrays
        ColumnVectorPtr lookup_arrays(new ColumnVector());


        //ColumnVectorPtr appended_dense_values_arrays(new ColumnVector());

        //		//Lookup Colums
        //		std::vector<ColumnPtr> lookup_arrays_to_real_columns_;






                for(result_map_it=result_tids.begin();result_map_it!=result_tids.end();++result_map_it){
                    //result_map_it->second  =
CDK::util::gather(result_map_it->second, join_result->first);
                    LookupColumnPtr lookup_col(new
LookupColumn(getTablebyName(result_map_it->first), result_map_it->second));
                    ColumnVectorPtr current_lookup_arrays =
lookup_col->getLookupArrays();
                    TableSchema current_schema =
lookup_col->getTable()->getSchema();



                    lookup_columns.push_back(lookup_col);
                    lookup_arrays->insert(lookup_arrays->end(),current_lookup_arrays->begin(),current_lookup_arrays->end());
                    schema.insert(schema.end(), current_schema.begin(),
current_schema.end());

                }

                //Create Lookup Table and return!

        return LookupTablePtr(new LookupTable(std::string("Chain Join"),
                schema,
                lookup_columns,
                *lookup_arrays));



        return TablePtr();
}
*/

/*
TablePtr two_phase_physical_optimization_chain_join(InvisibleJoinSelectionList
dimensions, hype::DeviceConstraint dev_constr, std::ostream& out){


    InvisibleJoinSelectionList::iterator it;
    const int NUM_DIM_TABLES=dimensions.size();

    Timestamp begin_construct_result = getTimestamp();
    //TablePtr result=fact_table;

//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
std::map<std::string,PositionListPtr> result_tids;

for(it=dimensions.begin();it!=dimensions.end();++it){
    Timestamp begin = getTimestamp();

query_processing::column_processing::cpu::LogicalQueryPlanPtr complex_selection;
//omit selection sub plan for dimension table were no selection is performed
if(it->knf_sel_expr.disjunctions.empty()) continue;

complex_selection =
query_processing::createColumnBasedQueryPlan(getTablebyName(it->table_name),it->knf_sel_expr,
dev_constr);

query_processing::column_processing::cpu::PhysicalQueryPlanPtr phy_plan =
complex_selection->runChoppedPlan();
phy_plan->run();

assert(phy_plan->getRoot().get()!=NULL);
PositionListOperator* pos_list_op =
dynamic_cast<PositionListOperator*>(phy_plan->getRoot().get());
assert(pos_list_op!=NULL);
assert(pos_list_op->hasResultPositionList() ||
pos_list_op->hasCachedResult_GPU_PositionList());

assert(pos_list_op != NULL);
assert(pos_list_op->hasResultPositionList() ||
pos_list_op->hasCachedResult_GPU_PositionList());
PositionListPtr tids;
if (!pos_list_op->hasResultPositionList() &&
pos_list_op->hasCachedResult_GPU_PositionList()) {
    tids =
gpu::copy_PositionList_device_to_host(pos_list_op->getResult_GPU_PositionList());
} else {
    tids = pos_list_op->getResultPositionList();
}
assert(tids != NULL);


result_tids[it->table_name]=tids;

}


std::queue<InvisibleJoinSelection> join_queue;
//        std::queue<InvisibleJoinSelection>::iterator qit;

InvisibleJoinSelection left_join_table = join_queue.front();
join_queue.pop();

while(join_queue.size()>1){
InvisibleJoinSelection right_join_table = join_queue.front();
join_queue.pop();

TablePtr dimension_table1 = getTablebyName(left_join_table.table_name);
TablePtr dimension_table2 = getTablebyName(right_join_table.table_name);


std::string name1 = right_join_table.join_pred.getColumn1Name();
std::string name2 = right_join_table.join_pred.getColumn2Name();

ColumnPtr left_join_column = dimension_table1->getColumnbyName(name1);
ColumnPtr right_join_column = dimension_table2->getColumnbyName(name2);

ColumnPtr filered_left_join_column =
left_join_column->gather(result_tids[name1]);
ColumnPtr filered_right_join_column =
right_join_column->gather(result_tids[name2]);

PositionListPairPtr join_result =
filered_left_join_column->hash_join(filered_right_join_column);

//update positionlists of tables in joinpath
//FIXME: implement gather for positionlists
//            ColumnPtr filered_left_join_column =
left_join_column->gather(result_tids[name1]);
//            ColumnPtr filered_right_join_column =
right_join_column->gather(result_tids[name2]);


}

//        for(qit=dimensions.begin();qit!=dimensions.end();++qit){
//
//            TablePtr dimension_table = getTablebyName(it->table_name);
//
//            std::string name1 = it->join_pred.getColumn1Name();
//            std::string name2 = it->join_pred.getColumn2Name();
//
//            ColumnPtr left_join_column =
dimension_table->getColumnbyName(name1);
//            ColumnPtr right_join_column =
dimension_table->getColumnbyName(name2);
//
//            ColumnPtr filered_left_join_column =
left_join_column->gather(map[name1]);
//            ColumnPtr filered_right_join_column =
right_join_column->gather(map[name2]);
//
//            PositionListPairPtr join_result =
filered_left_join_column->hash_join(filered_right_join_column);
//
//            //update stored tids with left join tid list
//
//
//
//        }







//
//            //TablePtr filtered_dimension_table =
getTablebyName(it->table_name);
//            TablePtr filtered_dimension_table =
Table::selection(dimension_table, it->knf_sel_expr, LOOKUP, SERIAL);
//


//                result = Table::join(result, it->join_pred.getColumn2Name(),
filtered_dimension_table, it->join_pred.getColumn1Name(), HASH_JOIN, LOOKUP);
//
//
//
//
//
//                ColumnPtr left =
result->getColumnbyName(it->join_pred.getColumn2Name());
//                ColumnPtr right =
result->getColumnbyName(it->join_pred.getColumn1Name());
//                PositionListPairPtr join_result = left->hash_join(right);
//
//                left->gather(join_result->first);
//                right->gather(join_result->second);
//
            Timestamp end = getTimestamp();
//                if(!quiet)
//                    std::cout << "TOPPO Tuple Reconstruct JOIN with "<<
it->table_name <<": " << double(end-begin)/(1000*1000) << "ms" << std::endl;
//
//                if(!quiet) std::cout << "Dim " << it->table_name << ": #Rows:
" <<  result->getNumberofRows() << std::endl;

    Timestamp end_construct_result = getTimestamp();
    //result.concat(fact_tab);



    return TablePtr();

}
*/

namespace physical_operator {

bool CPU_InvisibleJoin_Operator::execute() {
#ifdef ENABLE_TWO_PHASE_PHYSICAL_OPTIMIZATION
  //                    Timestamp begin_stable_invisible_join = getTimestamp();
  //                    TablePtr tab =
  //                    CDK::join::invisibleJoin(this->getInputData(),
  //                    dimensions_);
  //                    Timestamp end_stable_invisible_join = getTimestamp();

  hype::DeviceConstraint dev_constr =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
  this->result_ = two_phase_physical_optimization_invisible_join(
      this->getInputData(), dimensions_, dev_constr, *this->out);
#else
  this->result_ = CDK::join::invisibleJoin(this->getInputData(), dimensions_);
#endif
  if (this->result_) {
    setResultSize((this->result_)->getNumberofRows());
    return true;
  } else
    return false;
}

TypedOperatorPtr create_CPU_InvisibleJoin_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_InvisibleJoin& log_selection_ref =
      static_cast<logical_operator::Logical_InvisibleJoin&>(logical_node);
  // std::cout << "create CPU_InvisibleJoin_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new CPU_InvisibleJoin_Operator(
      sched_dec, left_child, log_selection_ref.getInvisibleJoinSelectionList(),
      log_selection_ref.getDeviceConstraint(),
      log_selection_ref.getMaterializationStatus()));
}

Physical_Operator_Map_Ptr map_init_function_invisible_join_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for INVISIBLE JOIN operator!"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SELECTION","CPU_InvisibleJoin_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("SELECTION","GPU_InvisibleJoin_Algorithm",hype::GPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_InvisibleJoin_Algorithm", "INVISIBLE_JOIN", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_InvisibleJoin_Algorithm", "INVISIBLE_JOIN", hype::Least_Squares_1D,
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

  map["CPU_InvisibleJoin_Algorithm"] = create_CPU_InvisibleJoin_Operator;
  // map["CPU_ParallelInvisibleJoin_Algorithm"]=create_CPU_ParallelInvisibleJoin_Operator;
  // map["GPU_InvisibleJoin_Algorithm"]=create_GPU_InvisibleJoin_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
