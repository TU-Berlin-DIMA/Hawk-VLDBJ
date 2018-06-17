

#include <config/configuration.hpp>
#include <core/lookup_array.hpp>
#include <core/runtime_configuration.hpp>
#include <query_optimization/qep.hpp>
#include <query_processing/query_processor.hpp>
#include <queue>

#include "query_compilation/code_generator.hpp"

namespace CoGaDB {
namespace query_processing {

using namespace std;

CoGaDB::query_processing::PhysicalQueryPlanPtr optimize_and_execute(
    const std::string& query_name, LogicalQueryPlan& log_plan,
    ClientPtr client) {
  assert(client != NULL);
  std::ostream& out = client->getOutputStream();
  log_plan.setOutputStream(out);
#ifdef PRINT_QUERY_STARTINGTIME
  cout << "Startingtime: " << double(getTimestamp()) << endl;
#endif

  if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
    log_plan.print();
  }

  CoGaDB::query_processing::PhysicalQueryPlanPtr plan;
  if (RuntimeConfiguration::instance().isQueryChoppingEnabled()) {
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      out << "Executing Query " << query_name << ":" << endl;
    }
    plan = log_plan.runChoppedPlan();
  } else {
    hype::query_optimization::QEPPtr hype_qep = log_plan.convertToQEP();
    hype::query_optimization::optimizeQueryPlan(
        *hype_qep,
        RuntimeConfiguration::instance()
            .getQueryOptimizationHeuristic());  // hype::GREEDY_HEURISTIC);//hype::BACKTRACKING);
    plan = log_plan.convertQEPToPhysicalQueryPlan(hype_qep);
    // plan = log_plan.convertToPhysicalQueryPlan();
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      out << "Executing Query " << query_name << ":" << endl;
      plan->print();
    }
    // execute the query
    plan->run();
  }

  if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
    plan->printResults(true, true, true);  // plan->print();
  } else if (RuntimeConfiguration::instance().getProfileQueries()) {
    plan->printResults(true, false, false);  // plan->print();
  }

  assert(plan->getResult() != NULL);
  return plan;
}

query_processing::column_processing::cpu::LogicalQueryPlanPtr
createColumnPlanforDisjunction(TablePtr table, const Disjunction& disjunction,
                               hype::DeviceConstraint dev_constr) {
  if (!table)
    return query_processing::column_processing::cpu::LogicalQueryPlanPtr();
  if (disjunction.empty())
    return query_processing::column_processing::cpu::LogicalQueryPlanPtr();
  typedef query_processing::column_processing::cpu::TypedLogicalNodePtr
      TypedLogicalNodePtr;

  std::queue<TypedLogicalNodePtr> disjunction_queue;
  for (unsigned int i = 0; i < disjunction.size(); i++) {
    // stores the tid list for each predicate
    // std::vector<PositionListPtr>
    // predicate_result_tid_lists(knf_expr.disjunctions[i].size());

    if (disjunction[i].getPredicateType() == ValueValuePredicate) {
      boost::shared_ptr<logical_operator::Logical_Column_Scan> scan_col1(
          new logical_operator::Logical_Column_Scan(
              table, disjunction[i].getColumn1Name()));
      boost::shared_ptr<logical_operator::Logical_Column_Scan> scan_col2(
          new logical_operator::Logical_Column_Scan(
              table, disjunction[i].getColumn2Name()));
      boost::shared_ptr<logical_operator::Logical_ColumnComparatorOperation>
          filter_col(new logical_operator::Logical_ColumnComparatorOperation(
              disjunction[i]));

      filter_col->setLeft(scan_col1);
      filter_col->setRight(scan_col2);
      disjunction_queue.push(filter_col);

      if (!quiet && verbose && debug)
        cout << "Process predicate: " << disjunction[i].toString()
             << " and add to Disjunction Queue " << endl;

    } else if (disjunction[i].getPredicateType() == ValueConstantPredicate) {
      boost::shared_ptr<logical_operator::Logical_Column_Scan> scan_col(
          new logical_operator::Logical_Column_Scan(
              table, disjunction[i].getColumn1Name()));
      boost::shared_ptr<logical_operator::Logical_Column_Constant_Filter>
          filter_col(new logical_operator::Logical_Column_Constant_Filter(
              disjunction[i], dev_constr));

      filter_col->setLeft(scan_col);
      disjunction_queue.push(filter_col);
    }
  }
  // merge together all tids from this disjunction
  while (disjunction_queue.size() > 1) {
    TypedLogicalNodePtr node1 = disjunction_queue.front();
    disjunction_queue.pop();
    TypedLogicalNodePtr node2 = disjunction_queue.front();
    disjunction_queue.pop();
    boost::shared_ptr<logical_operator::Logical_PositionList_Operator>
        tid_union(new logical_operator::Logical_PositionList_Operator(
            POSITIONLIST_UNION));
    tid_union->setLeft(node1);
    tid_union->setRight(node2);
    disjunction_queue.push(tid_union);
    if (!quiet && verbose && debug)
      cout << "ADD UNION Operator for Childs " << node1->toString() << " and "
           << node2->toString() << endl;
  }

  return query_processing::column_processing::cpu::LogicalQueryPlanPtr(
      new query_processing::column_processing::cpu::LogicalQueryPlan(
          disjunction_queue.front(), std::cout));
}

const query_processing::column_processing::cpu::LogicalQueryPlanPtr
createComplexBitmapSelectionQueryPlan(TablePtr table,
                                      const KNF_Selection_Expression& knf_expr,
                                      hype::DeviceConstraint dev_constr) {
  return query_processing::column_processing::cpu::LogicalQueryPlanPtr();
}

const query_processing::column_processing::cpu::LogicalQueryPlanPtr
createColumnBasedQueryPlan(TablePtr table,
                           const KNF_Selection_Expression& knf_expr,
                           hype::DeviceConstraint dev_constr) {
  if (!table)
    return query_processing::column_processing::cpu::LogicalQueryPlanPtr();
  if (knf_expr.disjunctions.empty())
    return query_processing::column_processing::cpu::LogicalQueryPlanPtr();
  // stores the result for each disjunction
  // std::vector<PositionListPtr>
  // disjunctions_result_tid_lists(knf_expr.disjunctions.size());
  typedef query_processing::column_processing::cpu::TypedLogicalNodePtr
      TypedLogicalNodePtr;
  std::queue<TypedLogicalNodePtr> conjunction_queue;  // current_tree_level;
  std::vector<std::queue<TypedLogicalNodePtr> > disjunction_queues(
      knf_expr.disjunctions.size());
  for (unsigned int i = 0; i < knf_expr.disjunctions.size(); i++) {
    // stores the tid list for each predicate
    // std::vector<PositionListPtr>
    // predicate_result_tid_lists(knf_expr.disjunctions[i].size());

    for (unsigned j = 0; j < knf_expr.disjunctions[i].size(); j++) {
      if (knf_expr.disjunctions[i][j].getPredicateType() ==
          ValueValuePredicate) {
        boost::shared_ptr<logical_operator::Logical_Column_Scan> scan_col1(
            new logical_operator::Logical_Column_Scan(
                table, knf_expr.disjunctions[i][j].getColumn1Name()));
        boost::shared_ptr<logical_operator::Logical_Column_Scan> scan_col2(
            new logical_operator::Logical_Column_Scan(
                table, knf_expr.disjunctions[i][j].getColumn2Name()));
        boost::shared_ptr<logical_operator::Logical_ColumnComparatorOperation>
            filter_col(new logical_operator::Logical_ColumnComparatorOperation(
                knf_expr.disjunctions[i][j]));

        filter_col->setLeft(scan_col1);
        filter_col->setLeft(scan_col2);
        disjunction_queues[i].push(filter_col);

        if (!quiet && verbose && debug)
          cout << "Process predicate: "
               << knf_expr.disjunctions[i][j].toString()
               << " and add to Disjunction Queue " << i << endl;

      } else if (knf_expr.disjunctions[i][j].getPredicateType() ==
                 ValueConstantPredicate) {
        boost::shared_ptr<logical_operator::Logical_Column_Scan> scan_col(
            new logical_operator::Logical_Column_Scan(
                table, knf_expr.disjunctions[i][j].getColumn1Name()));
        boost::shared_ptr<logical_operator::Logical_Column_Constant_Filter>
            filter_col(new logical_operator::Logical_Column_Constant_Filter(
                knf_expr.disjunctions[i][j], dev_constr));

        filter_col->setLeft(scan_col);
        disjunction_queues[i].push(filter_col);

        if (!quiet && verbose && debug)
          cout << "Process predicate: "
               << knf_expr.disjunctions[i][j].toString()
               << " and add to Disjunction Queue " << i << endl;

      } else {
        std::cerr
            << "FATAL ERROR! in BaseTable::selection(): Unknown Predicate Type!"
            << std::endl;
        std::cerr << "In File " << __FILE__ << ":" << __LINE__ << std::endl;
        exit(-1);
      }
    }
  }
  // merge sorted tid lists (compute the disjunction)
  //->build a subtree for each disjunction
  for (unsigned int i = 0; i < disjunction_queues.size(); i++) {
    while (disjunction_queues[i].size() > 1) {
      TypedLogicalNodePtr node1 = disjunction_queues[i].front();
      disjunction_queues[i].pop();
      TypedLogicalNodePtr node2 = disjunction_queues[i].front();
      disjunction_queues[i].pop();
      boost::shared_ptr<logical_operator::Logical_PositionList_Operator>
          tid_union(new logical_operator::Logical_PositionList_Operator(
              POSITIONLIST_UNION, dev_constr));
      tid_union->setLeft(node1);
      tid_union->setRight(node2);
      disjunction_queues[i].push(tid_union);
      if (!quiet && verbose && debug)
        cout << "ADD UNION Operator for Childs " << node1->toString() << " and "
             << node2->toString() << endl;
    }
    // subtree for disjunction completed, now it has to be combined with the
    // other disjunctions
    conjunction_queue.push(disjunction_queues[i].front());

    if (!quiet && verbose && debug)
      cout << "added DisjunctionSubtree to Conjunction Queue" << endl;
  }

  // now combine all sub trees for the disjunctions by adding TID Intersion
  // nodes
  while (conjunction_queue.size() > 1) {
    TypedLogicalNodePtr node1 = conjunction_queue.front();
    conjunction_queue.pop();
    TypedLogicalNodePtr node2 = conjunction_queue.front();
    conjunction_queue.pop();
    boost::shared_ptr<logical_operator::Logical_PositionList_Operator>
        tid_intersection(new logical_operator::Logical_PositionList_Operator(
            POSITIONLIST_INTERSECTION, dev_constr));
    tid_intersection->setLeft(node1);
    tid_intersection->setRight(node2);
    conjunction_queue.push(tid_intersection);
    if (!quiet && verbose && debug)
      cout << "ADD INTERSECT Operator for Childs " << node1->toString()
           << " and " << node2->toString() << endl;
  }

  return query_processing::column_processing::cpu::LogicalQueryPlanPtr(
      new query_processing::column_processing::cpu::LogicalQueryPlan(
          conjunction_queue.front()));
}

//           PositionListPtr
//           optimize_and_execute_column_based_plan(query_processing::column_processing::cpu::LogicalQueryPlanPtr){
//
//
//           }

/*! \brief puts highly selective equality predicates to the end of the
 * expression,
 * because these are evaluated first */
KNF_Selection_Expression optimizeKNFOrder(
    const KNF_Selection_Expression& knf_expr) {
  KNF_Selection_Expression optimized_knf_order;

  KNF_Selection_Expression simple_equality_predicates;

  for (unsigned i = 0; i < knf_expr.disjunctions.size(); i++) {
    if (knf_expr.disjunctions[i].size() == 1 &&
        knf_expr.disjunctions[i].at(0).getPredicateType() ==
            ValueConstantPredicate &&
        knf_expr.disjunctions[i].at(0).getValueComparator() == EQUAL) {
      simple_equality_predicates.disjunctions.push_back(
          knf_expr.disjunctions[i]);
    } else {
      optimized_knf_order.disjunctions.push_back(knf_expr.disjunctions[i]);
    }
  }

  optimized_knf_order.disjunctions.insert(
      optimized_knf_order.disjunctions.begin(),
      simple_equality_predicates.disjunctions.begin(),
      simple_equality_predicates.disjunctions.end());

  return optimized_knf_order;
}

// Note: would make sence to execute the whole query plan completely on GPU or
// on CPU
const query_processing::LogicalQueryPlanPtr createSerialColumnBasedQueryPlan(
    TablePtr table, const KNF_Selection_Expression& knf_expr,
    hype::DeviceConstraint dev_constr) {
  query_processing::TypedNodePtr subplan_root(
      new logical_operator::Logical_Scan(table));

  // boost::shared_ptr<logical_operator::Logical_Selection>  selection(new
  // logical_operator::Logical_Selection("Sales",boost::any(6),LESSER));

  KNF_Selection_Expression opt_knf_expr = optimizeKNFOrder(knf_expr);

  for (unsigned i = 0; i < opt_knf_expr.disjunctions.size(); i++) {
    // break down the conjunctions in multiple selections that are executed
    // sequentially (Rest is handled by the complex selection operator)
    KNF_Selection_Expression simple_knf_expr;
    simple_knf_expr.disjunctions.push_back(opt_knf_expr.disjunctions[i]);
    boost::shared_ptr<logical_operator::Logical_ComplexSelection> selection(
        new logical_operator::Logical_ComplexSelection(simple_knf_expr, LOOKUP,
                                                       dev_constr));

    query_processing::TypedNodePtr oldsubplan_root = subplan_root;
    subplan_root = selection;

    subplan_root->setLeft(oldsubplan_root);
    oldsubplan_root->setParent(subplan_root);
  }

  return query_processing::LogicalQueryPlanPtr(
      new query_processing::LogicalQueryPlan(subplan_root));
}

TablePtr two_phase_physical_optimization_selection(
    TablePtr table, const KNF_Selection_Expression& knf_expr,
    hype::DeviceConstraint dev_constr, MaterializationStatus mat_stat,
    ParallelizationMode comp_mode, std::ostream* out_ptr) {
  std::ostream& out = *out_ptr;

  if (!table) {
    cout << "Fatal Error: in two_phase_physical_optimization_selection(): "
            "input table pointer is NULL!"
         << endl;
    return TablePtr();
  }

  // cout << "TOPPO PARALLELIZATION MODE: " << CoGaDB::util::getName(comp_mode)
  // << endl;
  PositionListPtr tids;
  // in parallel mode, create column based bushy query plan and execute the tree
  // in parallel (beneficial for high selectivity of scans)
  if (comp_mode == CoGaDB::PARALLEL) {
    query_processing::column_processing::cpu::LogicalQueryPlanPtr log_plan =
        createColumnBasedQueryPlan(table, knf_expr, dev_constr);
    log_plan->setOutputStream(out);
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      log_plan->print();
    }
    query_processing::column_processing::cpu::PhysicalQueryPlanPtr phy_plan;
    if (RuntimeConfiguration::instance().isQueryChoppingEnabled()) {
      phy_plan = log_plan->runChoppedPlan();
    } else {
      // phy_plan = log_plan->convertToPhysicalQueryPlan();
      hype::query_optimization::QEPPtr hype_qep = log_plan->convertToQEP();
      hype::query_optimization::optimizeQueryPlan(
          *hype_qep,
          RuntimeConfiguration::instance()
              .getQueryOptimizationHeuristic());  // hype::GREEDY_HEURISTIC);//hype::BACKTRACKING);
      phy_plan = log_plan->convertQEPToPhysicalQueryPlan(hype_qep);
      if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
        phy_plan->print();
      }
      phy_plan->run();
    }

    hype::ProcessingDeviceID id = phy_plan->getRoot()
                                      ->getSchedulingDecision()
                                      .getDeviceSpecification()
                                      .getProcessingDeviceID();
    ProcessorSpecification proc_spec(id);
    // JoinParam param(proc_spec, HASH_JOIN);

    PositionListOperator* pos_list_op =
        dynamic_cast<PositionListOperator*>(phy_plan->getRoot().get());
    assert(pos_list_op != NULL);
    //                    assert(pos_list_op->hasResultPositionList() ||
    //                    pos_list_op->hasCachedResult_GPU_PositionList());
    PositionListPtr input_tids;
    input_tids =
        copy_if_required(pos_list_op->getResultPositionList(), proc_spec);

    //                    if (!pos_list_op->hasResultPositionList() &&
    //                    pos_list_op->hasCachedResult_GPU_PositionList()) {
    //                        input_tids =
    //                        gpu::copy_PositionList_device_to_host(pos_list_op->getResult_GPU_PositionList());
    //                    } else {
    //                        input_tids = pos_list_op->getResultPositionList();
    //                    }

    assert(input_tids != NULL);
    tids = input_tids;

    if (!tids) return TablePtr();
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      phy_plan->printResults(true, true, true);  // plan->print();
    } else if (RuntimeConfiguration::instance().getProfileQueries()) {
      phy_plan->printResults(true, false, false);  // plan->print();
    }
    return BaseTable::createResultTable(table, tids, mat_stat, "selection",
                                        proc_spec);
  } else if (comp_mode == CoGaDB::SERIAL) {
    //                    query_processing::LogicalQueryPlanPtr  log_plan  =
    //                    createSerialColumnBasedQueryPlan(table, knf_expr,
    //                    dev_constr);
    query_processing::LogicalQueryPlanPtr log_plan =
        createSerialColumnBasedQueryPlan(table, knf_expr,
                                         dev_constr);  // hype::CPU_ONLY);
    log_plan->setOutputStream(out);
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      log_plan->print();
    }
    query_processing::PhysicalQueryPlanPtr phy_plan;
    if (RuntimeConfiguration::instance().isQueryChoppingEnabled()) {
      phy_plan = log_plan->runChoppedPlan();
    } else {
      // phy_plan = log_plan->convertToPhysicalQueryPlan();
      hype::query_optimization::QEPPtr hype_qep = log_plan->convertToQEP();
      hype::query_optimization::optimizeQueryPlan(
          *hype_qep,
          RuntimeConfiguration::instance()
              .getQueryOptimizationHeuristic());  // hype::GREEDY_HEURISTIC);//hype::BACKTRACKING);
      phy_plan = log_plan->convertQEPToPhysicalQueryPlan(hype_qep);
      if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
        phy_plan->print();
      }
      phy_plan->run();
    }
    //                    if(RuntimeConfiguration::instance().getPrintQueryPlan())
    //                    {
    //                        cout << "Physical TOPPO Plan (after Execution): "
    //                        << endl;
    //                        phy_plan->print();
    //                    }
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      phy_plan->printResults(true, true, true);  // plan->print();
    } else if (RuntimeConfiguration::instance().getProfileQueries()) {
      phy_plan->printResults(true, false, false);  // plan->print();
    }
    // since this is a Table based plan, we can just return the result
    return phy_plan->getResult();
  } else {
    COGADB_FATAL_ERROR("Invalid Parallelization Mode!", "");
    return TablePtr();
  }
}

}  // end namespace query_processing
}  // end namespace CogaDB
