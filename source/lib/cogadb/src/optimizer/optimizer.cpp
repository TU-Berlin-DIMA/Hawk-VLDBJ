
#include <boost/algorithm/string.hpp>
#include <boost/function.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <core/data_dictionary.hpp>
#include <iostream>
#include <optimizer/optimizer.hpp>

#include <core/variable_manager.hpp>
#include <optimizer/join_order_optimization.hpp>
#include <query_processing/extension/artificial_pipeline_breaker.hpp>
#include <query_processing/generate_constant_column_operator.hpp>
#include <sql/server/sql_driver.hpp>
#include "query_processing/chain_join_operator.hpp"
#include "query_processing/pk_fk_join_operator.hpp"

using namespace std;

namespace CoGaDB {

namespace optimizer {

template <class UnaryFunction>
UnaryFunction traverse_inorder(UnaryFunction f,
                               query_processing::TypedNodePtr node) {
  if (!node) return f;

  UnaryFunction f1 = traverse_inorder(
      f, boost::dynamic_pointer_cast<
             typename query_processing::TypedNodePtr::element_type>(
             node->getLeft()));
  f(node);
  UnaryFunction f2 = traverse_inorder(
      f, boost::dynamic_pointer_cast<
             typename query_processing::TypedNodePtr::element_type>(
             node->getRight()));
  f.accumulate(f1, f2);
  return f;
}

template <class UnaryFunction>
UnaryFunction traverse_inorder(UnaryFunction f,
                               query_processing::NodePtr node) {
  if (!node) return f;

  UnaryFunction f1 = traverse_inorder(f, node->getLeft());
  f(node);
  UnaryFunction f2 = traverse_inorder(f, node->getRight());
  f.accumulate(f1, f2);
  return f;
}

void replace_operator(query_processing::LogicalQueryPlanPtr log_plan,
                      query_processing::NodePtr old_node,
                      query_processing::NodePtr new_node) {
  assert(new_node != NULL);
  assert(new_node->getLeft() == NULL);
  assert(new_node->getRight() == NULL);
  assert(new_node->getParent() == NULL);

  new_node->setLeft(old_node->getLeft());
  new_node->setRight(old_node->getRight());
  new_node->setParent(old_node->getParent());
  new_node->setLevel(old_node->getLevel());

  if (new_node->getLeft()) new_node->getLeft()->setParent(new_node);
  if (new_node->getRight()) new_node->getRight()->setParent(new_node);

  if (new_node->getParent()) {
    if (old_node->getParent()->getLeft() == old_node) {
      // node is left child of its parent
      old_node->getParent()->setLeft(new_node);
    } else {
      // node is right child of its parent
      old_node->getParent()->setRight(new_node);
    }
  } else {
    query_processing::TypedNodePtr new_root = boost::dynamic_pointer_cast<
        typename query_processing::TypedNodePtr::element_type>(new_node);
    log_plan->setNewRoot(new_root);
  }
}

Logical_Optimizer::OptimizerPipeline::OptimizerPipeline(
    std::string optimizer_name_, OptimizerRules optimizer_rules_)
    : optimizer_name(optimizer_name_), optimizer_rules(optimizer_rules_) {}

Logical_Optimizer::Logical_Optimizer() : optimizers_() {  // optimizer_rules_(){
  {
    OptimizerRules optimizer_rules;
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::decompose_complex_selections));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::push_down_selections));
    // optimizer_rules.push_back(
    // OptimizerRule(optimizer_rules::cross_product_to_join) );
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::join_order_optimization));
    // optimizer_rules.push_back(
    // OptimizerRule(optimizer_rules::compose_complex_selections) );
    // optimizer_rules.push_back(
    // OptimizerRule(optimizer_rules::rewrite_join_to_pk_fk_join) );
    //            optimizer_rules.push_back(
    //            OptimizerRule(optimizer_rules::rewrite_join_to_fetch_join) );
    optimizer_rules.push_back(OptimizerRule(
        optimizer_rules::set_device_constaints_for_unsupported_operations));
    OptimizerPipelinePtr opt_pipe(
        new OptimizerPipeline("default_optimizer", optimizer_rules));
    optimizers_.insert(make_pair(opt_pipe->optimizer_name, opt_pipe));
  }
  {
    OptimizerRules optimizer_rules;
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::decompose_complex_selections));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::push_down_selections));
    // optimizer_rules.push_back(
    // OptimizerRule(optimizer_rules::cross_product_to_join) );
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::join_order_optimization));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::compose_complex_selections));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::rewrite_join_tree_to_invisible_join));
    // optimizer_rules.push_back(
    // OptimizerRule(optimizer_rules::rewrite_join_to_pk_fk_join) );
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::rewrite_join_to_fetch_join));
    optimizer_rules.push_back(OptimizerRule(
        optimizer_rules::set_device_constaints_for_unsupported_operations));
    OptimizerPipelinePtr opt_pipe(
        new OptimizerPipeline("star_join_optimizer", optimizer_rules));
    optimizers_.insert(make_pair(opt_pipe->optimizer_name, opt_pipe));
  }
  {
    OptimizerRules optimizer_rules;
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::decompose_complex_selections));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::push_down_selections));
    //            //optimizer_rules.push_back(
    //            OptimizerRule(optimizer_rules::cross_product_to_join) );
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::join_order_optimization));
    //            optimizer_rules.push_back(
    //            OptimizerRule(optimizer_rules::compose_complex_selections) );
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::rewrite_join_tree_to_chain_join));
    //            optimizer_rules.push_back(
    //            OptimizerRule(optimizer_rules::rewrite_join_to_pk_fk_join) );
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::rewrite_join_to_fetch_join));
    optimizer_rules.push_back(OptimizerRule(
        optimizer_rules::set_device_constaints_for_unsupported_operations));
    OptimizerPipelinePtr opt_pipe(
        new OptimizerPipeline("chain_join_optimizer", optimizer_rules));
    optimizers_.insert(make_pair(opt_pipe->optimizer_name, opt_pipe));
  }
  {
    OptimizerRules optimizer_rules;
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::decompose_complex_selections));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::push_down_selections));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::remove_cross_joins_and_keep_join_order));
    optimizer_rules.push_back(OptimizerRule(
        optimizer_rules::move_fact_table_scan_to_right_side_of_join));
    // optimizer_rules.push_back(
    // OptimizerRule(optimizer_rules::cross_product_to_join) );
    //            optimizer_rules.push_back(
    //            OptimizerRule(optimizer_rules::join_order_optimization) );
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::compose_complex_selections));
    // optimizer_rules.push_back(
    // OptimizerRule(optimizer_rules::rewrite_join_to_pk_fk_join) );
    //            optimizer_rules.push_back(
    //            OptimizerRule(optimizer_rules::rewrite_join_to_fetch_join) );
    optimizer_rules.push_back(OptimizerRule(
        optimizer_rules::set_device_constaints_for_unsupported_operations));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::add_artificial_pipeline_breakers));
    OptimizerPipelinePtr opt_pipe(
        new OptimizerPipeline("no_join_order_optimizer", optimizer_rules));
    optimizers_.insert(make_pair(opt_pipe->optimizer_name, opt_pipe));
  }
  {
    OptimizerRules optimizer_rules;
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::decompose_complex_selections));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::push_down_selections));
    optimizer_rules.push_back(
        OptimizerRule(optimizer_rules::rewrite_join_to_gather_join));
    optimizer_rules.push_back(OptimizerRule(
        optimizer_rules::set_device_constaints_for_unsupported_operations));
    OptimizerPipelinePtr opt_pipe(
        new OptimizerPipeline("gather_join_optimizer", optimizer_rules));
    optimizers_.insert(make_pair(opt_pipe->optimizer_name, opt_pipe));
  }

  //                optimizer_rules_.push_back(
  //                OptimizerRule(optimizer_rules::decompose_complex_selections)
  //                );
  //                optimizer_rules_.push_back(
  //                OptimizerRule(optimizer_rules::push_down_selections) );
  //                optimizer_rules_.push_back(
  //                OptimizerRule(optimizer_rules::cross_product_to_join) );
  //                optimizer_rules_.push_back(
  //                OptimizerRule(optimizer_rules::compose_complex_selections)
  //                );
  //                optimizer_rules_.push_back(
  //                OptimizerRule(optimizer_rules::rewrite_join_to_pk_fk_join)
  //                );
  //                optimizer_rules_.push_back(
  //                OptimizerRule(optimizer_rules::set_device_constaints_for_unsupported_operations)
  //                );
}

Logical_Optimizer& Logical_Optimizer::instance() {
  static Logical_Optimizer optimizer;
  return optimizer;
}

bool Logical_Optimizer::optimize(
    query_processing::LogicalQueryPlanPtr log_plan) {
  OptimizerRules::iterator it;
  if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
    cout << "Input Plan:" << endl;
    log_plan->print();
  }

  if (!checkQueryPlan(log_plan->getRoot())) {
    COGADB_FATAL_ERROR("checkQueryPlan failed!", "");
  }

  std::string optimizer_name =
      RuntimeConfiguration::instance().getOptimizer();  //"star_join_optimizer";

  OptimizerPipelines::iterator it_opt;

  it_opt = optimizers_.find(optimizer_name);
  assert(it_opt != optimizers_.end());
  OptimizerRules optimizer_rules_ = it_opt->second->optimizer_rules;
  for (it = optimizer_rules_.begin(); it != optimizer_rules_.end(); ++it) {
    if (!it->empty()) {
      if (!(*it)(log_plan)) {
        cout << "Logical Optimization Failed!" << endl;
        return false;
      }
    }
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      cout << "Optimized Plan:" << endl;
      log_plan->print();
    }
  }
  return true;
}

//        void
//        Logical_Optimizer::checkQueryPlan(query_processing::LogicalQueryPlanPtr
//        log_plan){
//
//
//        }

bool is_simple_selection(query_processing::NodePtr node) {
  if (!node) return false;
  // if(node->getLeft()!=NULL) return false;
  std::string op_name = node->getOperationName();
  if (op_name == "SELECTION") {
    return true;
  }
  return false;
}
bool is_complex_selection(query_processing::NodePtr node) {
  if (!node) return false;
  if (!quiet && verbose && debug)
    cout << node->toString() << " is Complex Selection:"
         << bool(node->getOperationName() == "COMPLEX_SELECTION") << endl;
  // if(node->getLeft()!=NULL) return false;
  std::string op_name = node->getOperationName();
  if (op_name == "COMPLEX_SELECTION") {
    return true;
  }
  return false;
}

bool is_join(query_processing::NodePtr node) {
  if (!node) return false;
  std::string op_name = node->getOperationName();
  if (op_name == "JOIN") {
    return true;
  }
  return false;
}

bool is_cross_join(query_processing::NodePtr node) {
  if (!node) return false;
  std::string op_name = node->getOperationName();
  if (op_name == "CROSS_JOIN") {
    return true;
  }
  return false;
}

bool is_scan(query_processing::NodePtr node) {
  if (!node) return false;
  std::string op_name = node->getOperationName();
  if (op_name == "SCAN") {
    return true;
  }
  return false;
}

bool is_projection(query_processing::NodePtr node) {
  if (!node) return false;
  std::string op_name = node->getOperationName();
  if (op_name == "PROJECTION") {
    return true;
  }
  return false;
}

bool is_groupby(query_processing::NodePtr node) {
  if (!node) return false;
  std::string op_name = node->getOperationName();
  if (op_name == "GROUPBY") {
    return true;
  }
  return false;
}

bool is_addConstantValueColumnOp(query_processing::NodePtr node) {
  if (!node) return false;
  std::string op_name = node->getOperationName();
  if (op_name == "AddConstantValueColumn") {
    return true;
  }
  return false;
}

bool is_ColumnConstantAlgebraOp(query_processing::NodePtr node) {
  if (!node) return false;
  std::string op_name = node->getOperationName();
  if (op_name == "ColumnConstantOperator") {
    return true;
  }
  return false;
}

bool containsJoinPredicate(const KNF_Selection_Expression& knf) {
  for (unsigned int i = 0; i < knf.disjunctions.size(); ++i) {
    for (unsigned int j = 0; j < knf.disjunctions[i].size(); ++j) {
      if (knf.disjunctions[i][j].getPredicateType() == ValueValuePredicate) {
        return true;
      }
    }
  }
  return false;
}

const std::list<Predicate> getJoinPredicates(
    const KNF_Selection_Expression& knf) {
  std::list<Predicate> result;
  for (unsigned int i = 0; i < knf.disjunctions.size(); ++i) {
    for (unsigned int j = 0; j < knf.disjunctions[i].size(); ++j) {
      if (knf.disjunctions[i][j].getPredicateType() == ValueValuePredicate) {
        result.push_back(knf.disjunctions[i][j]);
      }
    }
  }
  return result;
}

void setLevelsOfTree(query_processing::NodePtr node) {
  assert(node != NULL);
  if (CoGaDB::optimizer::verbose_optimizer) {
    cout << node->toString(true) << ": Left: ";
    if (node->getLeft()) cout << node->getLeft()->toString(true);
    cout << " Right: ";
    if (node->getRight()) cout << node->getRight()->toString(true);
    cout << endl;
  }

  if (CoGaDB::optimizer::verbose_optimizer) {
    cout << "Set Level " << node->getLevel() + 1
         << " to childs of node: " << node->toString(true) << endl;
  }
  if (is_scan(node)) {
    if (node->getLeft() || node->getRight()) {
      cout << "Error! SCAN operation has to be a leaf, but has at least one "
              "child!"
           << endl;
    }
  }
  if (node->getLeft()) {
    node->getLeft()->setLevel(node->getLevel() + 1);
    setLevelsOfTree(node->getLeft());
  }

  if (node->getRight()) {
    //                cout << "Current Node: " << node->toString(true) << endl;
    //                cout << "Right Child: " <<
    //                node->getRight()->toString(true) << endl;
    if (is_complex_selection(node)) {
      cout << "Current Node: " << node->toString(true) << endl;
      if (node->getParent())
        cout << "Parent Node: " << node->getParent()->toString(true) << endl;
      if (node->getLeft())
        cout << "Left Child: " << node->getLeft()->toString(true) << endl;
      if (node->getRight())
        cout << "Right Child: " << node->getRight()->toString(true) << endl;
    }
    assert(!is_complex_selection(node));
    node->getRight()->setLevel(node->getLevel() + 1);
    setLevelsOfTree(node->getRight());
  }
}

void cleanupNodeTree(query_processing::NodePtr node) {
  if (node) {
    cleanupNodeTree(node->getLeft());
    cleanupNodeTree(node->getRight());
    node->setParent(query_processing::NodePtr());
    node->setLeft(query_processing::NodePtr());
    node->setRight(query_processing::NodePtr());
  }
}

// this function basicallys destroys a query plan, but cleans up only cross join
// operators
// and complex selections that are join conditions
// this is needed when we rewrite plans and do not want that the delete routine
// cleans up
// the operators we still need. This function is currently used by the join
// order optimizer only
void cleanupCrossJoinsandJoinComplexSelectionFromNodeTree(
    query_processing::NodePtr node) {
  if (node) {
    cleanupCrossJoinsandJoinComplexSelectionFromNodeTree(node->getLeft());
    cleanupCrossJoinsandJoinComplexSelectionFromNodeTree(node->getRight());
    if (node->getOperationName() == "CROSS_JOIN") {
      //                        std::cout << "[Optimizer:] cleanup node " <<
      //                        node->toString(true) << std::endl;
      node->setParent(query_processing::NodePtr());
      node->setLeft(query_processing::NodePtr());
      node->setRight(query_processing::NodePtr());
    } else if (is_complex_selection(node)) {
      boost::shared_ptr<
          query_processing::logical_operator::Logical_ComplexSelection>
          selection = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_ComplexSelection>(
              node);
      if (selection->getKNF_Selection_Expression().disjunctions.size() == 1 &&
          selection->getKNF_Selection_Expression()
                  .disjunctions.front()
                  .size() == 1 &&
          selection->getKNF_Selection_Expression()
                  .disjunctions.front()
                  .front()
                  .getPredicateType() == ValueValuePredicate) {
        //                            std::cout << "[Optimizer:] cleanup node "
        //                            << node->toString(true) << std::endl;
        node->setParent(query_processing::NodePtr());
        node->setLeft(query_processing::NodePtr());
        node->setRight(query_processing::NodePtr());
      }
    }
  }
}

std::list<Attribut> getListOfAvailableAttributesChildrenAndSelf(
    query_processing::NodePtr node) {
  std::list<Attribut> result;

  if (node) {
    // handle special cases first
    if (is_scan(node)) {
      boost::shared_ptr<query_processing::logical_operator::Logical_Scan> scan =
          boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Scan>(node);
      assert(scan != NULL);
      TablePtr table = CoGaDB::getTablebyName(scan->getTableName());
      assert(table != NULL);
      // return table->getSchema();
      TableSchema schema = table->getSchema();
      TableSchema::const_iterator cit;
      for (cit = schema.begin(); cit != schema.end(); ++cit) {
        std::string name = cit->second;
        if (!isFullyQualifiedColumnIdentifier(cit->second)) {
          if (!convertColumnNameToFullyQualifiedName(cit->second, name)) {
          }
        }
        result.push_back(Attribut(cit->first, name));
      }

    } else if (is_addConstantValueColumnOp(node)) {
      assert(node->getLeft() != NULL);
      std::list<Attribut> left_node_result =
          getListOfAvailableAttributesChildrenAndSelf(node->getLeft());
      boost::shared_ptr<
          query_processing::logical_operator::Logical_AddConstantValueColumn>
          add_const_column =
              boost::dynamic_pointer_cast<query_processing::logical_operator::
                                              Logical_AddConstantValueColumn>(
                  node);
      assert(add_const_column != NULL);

      left_node_result.push_back(Attribut(add_const_column->getAttributeType(),
                                          add_const_column->getColumnName()));
      return left_node_result;

    } else if (is_ColumnConstantAlgebraOp(node)) {
      assert(node->getLeft() != NULL);
      std::list<Attribut> left_node_result =
          getListOfAvailableAttributesChildrenAndSelf(node->getLeft());
      boost::shared_ptr<
          query_processing::logical_operator::Logical_ColumnConstantOperator>
          column_const_operator =
              boost::dynamic_pointer_cast<query_processing::logical_operator::
                                              Logical_ColumnConstantOperator>(
                  node);
      assert(column_const_operator != NULL);

      left_node_result.push_back(
          Attribut(DOUBLE, column_const_operator->getResultColumnName()));
      return left_node_result;

    } else if (is_projection(node)) {
      assert(node->getLeft() != NULL);
      std::list<Attribut> left_node_result =
          getListOfAvailableAttributesChildrenAndSelf(node->getLeft());
      boost::shared_ptr<query_processing::logical_operator::Logical_Projection>
          projection = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Projection>(node);
      assert(projection != NULL);

      const std::list<std::string>& list = projection->getColumnList();

      std::list<Attribut> result;

      std::list<Attribut>::iterator cit;
      std::list<std::string>::const_iterator fit;
      for (cit = left_node_result.begin(); cit != left_node_result.end();
           ++cit) {
        fit = std::find(list.begin(), list.end(), cit->second);
        if (fit != list.end()) {
          result.push_back(*cit);
        }
      }
      // override prior observed attribute hull
      return result;
    } else if (is_groupby(node)) {
      // a groupby overrides the current hull, because it also performs a
      // prejection
      // and renaming
      assert(node->getLeft() != NULL);
      std::list<Attribut> left_node_result =
          getListOfAvailableAttributesChildrenAndSelf(node->getLeft());
      boost::shared_ptr<query_processing::logical_operator::Logical_Groupby>
          groupby = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Groupby>(node);
      assert(groupby != NULL);

      // grouping columns
      std::list<std::string> groupng_columns = groupby->getGroupingColumns();

      // handle default case, traverse left and right subtree and compute convex
      // hull
      if (node->getLeft()) {
        std::list<std::string>::const_iterator cit1;
        std::list<Attribut>::const_iterator cit2;

        std::list<Attribut> allowed_grouping_columns =
            getListOfAvailableAttributesChildrenAndSelf(node->getLeft());
        for (cit1 = groupng_columns.begin(); cit1 != groupng_columns.end();
             ++cit1) {
          for (cit2 = allowed_grouping_columns.begin();
               cit2 != allowed_grouping_columns.end(); ++cit2) {
            if (*cit1 == cit2->second) {
              result.push_back(*cit2);
            }
          }
        }
      }

      const std::list<ColumnAggregation>& aggregation_functions =
          groupby->getColumnAggregationFunctions();

      // renamed column names
      std::list<ColumnAggregation>::const_iterator cit;
      for (cit = aggregation_functions.begin();
           cit != aggregation_functions.end(); ++cit) {
        result.push_back(Attribut(DOUBLE, cit->second.second));
      }
      // override prior observed attribute hull
      return result;
    }

    // handle default case, traverse left and right subtree and compute convex
    // hull
    if (node->getLeft()) {
      std::list<Attribut> left_node_result =
          getListOfAvailableAttributesChildrenAndSelf(node->getLeft());
      result.insert(result.end(), left_node_result.begin(),
                    left_node_result.end());
    }
    if (node->getRight()) {
      std::list<Attribut> right_node_result =
          getListOfAvailableAttributesChildrenAndSelf(node->getRight());
      result.insert(result.end(), right_node_result.begin(),
                    right_node_result.end());
    }

  } else {
    cout << "FATAL Error! In optimizer::getListOfAvailableAttributes(): "
            "Invalid NodePtr!"
         << endl;
    cout << "File: " << __FILE__ << " at Line: " << __LINE__ << endl;
    exit(-1);
  }

  if (verbose_optimizer) {
    cout << "Current Transitive Hull in Node " << node->toString() << ": "
         << endl;
    for (std::list<Attribut>::iterator cit = result.begin();
         cit != result.end(); ++cit) {
      cout << cit->second << "," << endl;
    }
  }
  return result;
}

std::list<Attribut> getListOfAvailableAttributesChildrenOnly(
    query_processing::NodePtr node) {
  std::list<Attribut> result;
  if (node) {
    std::list<Attribut> left;
    if (node->getLeft())
      left = getListOfAvailableAttributesChildrenAndSelf(node->getLeft());
    std::list<Attribut> right;
    if (node->getRight())
      right = getListOfAvailableAttributesChildrenAndSelf(node->getRight());
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());
  }
  return result;
}

namespace optimizer_rules {
// swaps two Nodes in an operator tree, where one node is the child of the other
//            void swap(query_processing::NodePtr node1,
//            query_processing::NodePtr node2){
//                //is node 1 child of node 2?
//                if(node2->getLeft()==node1 || node2->getRight()==node1){
//                    swap(node2,node1);
//                }
//                //is node 2 child of node 1?
//                assert(node1->getLeft()==node2 || node1->getRight()==node2);
//
//
//                query_processing::NodePtr tmp_left = node1->getLeft();
//                query_processing::NodePtr tmp_right = node1->getRight();
//                unsigned int tmp_level = node1->getLevel();
//                query_processing::NodePtr tmp_parent = node1->getParent();
//
//                node1->setLeft(node2->getLeft());
//                node1->setRight(node2->getLeft());
//                node1->setLeft(node2->getLeft());
//
//                node2->setLeft(tmp_left);
//                node2->setRight(tmp_right);
//                node2->setLevel(tmp_level);
//
//                //is node 2 the left child of node 1?
//                if(node1->getLeft()==node2){
//
//
//                    node1->getParent()->setLeft(node2);
//                    node1->setParent(node2);
//
//                    node2->getParent()->setLeft();
//                    node2->getParent()->setRight();
//
//                }else if(node1->getLeft()==node2){
//
//                }
//
//                query_processing::NodePtr tmp_left = node1->getLeft();
//                query_processing::NodePtr tmp_right = node1->getRight();
//                unsigned int tmp_level = node1->getLevel();
//                query_processing::NodePtr tmp_parent = node1->getParent();
//
//                node1->setLeft(node2->getLeft());
//                node1->setRight(node2->getLeft());
//                node1->setLeft(node2->getLeft());
//
//                node2->setLeft(tmp_left);
//                node2->setRight(tmp_right);
//                node2->setLevel(tmp_level);
//
//            }

bool isColumnInTransitiveHull(
    const std::string& column_name,
    std::list<Attribut>& transitive_hull_of_attributes) {
  std::list<Attribut>::iterator it;
  for (it = transitive_hull_of_attributes.begin();
       it != transitive_hull_of_attributes.end(); ++it) {
    if (column_name == it->second) {
      return true;
    }
  }
  return false;
}

bool isDisjunctionInTransitiveHull(
    CoGaDB::Disjunction& d,
    std::list<Attribut>& transitive_hull_of_attributes) {
  for (unsigned int i = 0; i < d.size(); ++i) {
    if (d[i].getPredicateType() == CoGaDB::ValueConstantPredicate) {
      // Is the column needed by the predicate in the transitive hull?
      if (!isColumnInTransitiveHull(d[i].getColumn1Name(),
                                    transitive_hull_of_attributes)) {
        return false;
      }
    } else if (d[i].getPredicateType() == CoGaDB::ValueValuePredicate) {
      // ValueValuePredicates can only be pushed down if both predicates are in
      // the transitive hull
      // otherwise, the Selection reached a possible place of a cross join!
      if (!isColumnInTransitiveHull(d[i].getColumn1Name(),
                                    transitive_hull_of_attributes) ||
          !isColumnInTransitiveHull(d[i].getColumn2Name(),
                                    transitive_hull_of_attributes)) {
        return false;
      }
    }
  }
  return true;
}

bool isKNFInTransitiveHull(CoGaDB::KNF_Selection_Expression sel_expr,
                           std::list<Attribut>& transitive_hull_of_attributes) {
  for (unsigned int i = 0; i < sel_expr.disjunctions.size(); ++i) {
    if (!isDisjunctionInTransitiveHull(sel_expr.disjunctions[i],
                                       transitive_hull_of_attributes)) {
      return false;
    }
  }
  return true;
}

void push_down_selection_in_left_subtree_of_child(
    query_processing::NodePtr selection,
    query_processing::LogicalQueryPlanPtr log_plan) {
  assert(is_complex_selection(selection));
  assert(selection->getRight() == NULL);

  /*Swap the nodes by changing the Pointers*/
  query_processing::NodePtr node = selection->getLeft();
  query_processing::NodePtr my_parent = selection->getParent();
  // adjust levels
  unsigned int tmp_level = selection->getLevel();
  selection->setLevel(node->getLevel());
  node->setLevel(tmp_level);
  assert(selection->getRight() == NULL);
  // have parent pointer?
  if (my_parent) {
    // set the parent pointer of the selection to the child of the selection
    if (my_parent->getLeft() == selection)
      // selection is left child of parent
      my_parent->setLeft(node);
    else
      // selection is right child of parent
      my_parent->setRight(node);
  }
  assert(selection->getRight() == NULL);
  /*set the parent pointers*/
  // set the parent pointer of theselections child to selectiosn current parent
  node->setParent(my_parent);
  // selection will have former child as its new parent
  selection->setParent(node);

  /*No change pointers to push the selection in the LEFT SUBTREE*/
  query_processing::NodePtr tmp_left = node->getLeft();
  // push the selection down in left subtree!
  node->setLeft(selection);
  // update last parent pointer
  tmp_left->setParent(selection);
  // connect selection
  selection->setLeft(tmp_left);
  if (verbose_optimizer) {
    cout << "Selection Right Child: " << selection->getRight();
    if (selection->getRight()) {
      cout << selection->getRight()->toString(true);
      cout << endl;
    }
  }
  if (!my_parent)
    log_plan->setNewRoot(
        boost::dynamic_pointer_cast<
            typename query_processing::TypedNodePtr::element_type>(node));
  assert(selection->getRight() == NULL);
  // when level of root node of a tree changes, the level of all nodes have to
  // be changed
  setLevelsOfTree(node);
  assert(selection->getRight() == NULL);
}

void push_down_selection_in_right_subtree_of_child(
    query_processing::NodePtr selection,
    query_processing::LogicalQueryPlanPtr log_plan) {
  assert(is_complex_selection(selection));

  /*Swap the nodes by changing the Pointers*/
  query_processing::NodePtr node = selection->getLeft();
  query_processing::NodePtr my_parent = selection->getParent();
  // adjust levels
  unsigned int tmp_level = selection->getLevel();
  selection->setLevel(node->getLevel());
  node->setLevel(tmp_level);
  // have parent pointer?
  if (my_parent) {
    // set the parent pointer of the selection to the child of the selection
    if (my_parent->getLeft() == selection)
      // selection is left child of parent
      my_parent->setLeft(node);
    else
      // selection is right child of parent
      my_parent->setRight(node);
  }

  /*set the parent pointers*/
  // set the parent pointer of theselections child to selectiosn current parent
  node->setParent(my_parent);
  // selection will have former child as its new parent
  selection->setParent(node);

  /*Now change pointers to push the selection in the RIGHT SUBTREE*/
  query_processing::NodePtr tmp_right = node->getRight();
  // push the selection down in right subtree!
  node->setRight(selection);
  // update last parent pointer
  tmp_right->setParent(selection);
  // connect selection
  selection->setLeft(tmp_right);
  if (!my_parent)
    log_plan->setNewRoot(
        boost::dynamic_pointer_cast<
            typename query_processing::TypedNodePtr::element_type>(node));
  // when level of root node of a tree changes, the level of all nodes have to
  // be changed
  setLevelsOfTree(node);
}

struct Check_Tree_Consistency_Functor {
  bool operator()(query_processing::NodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Check_Tree_Consistency_Functor: visiting " << node->toString()
           << endl;
    }

    query_processing::NodePtr my_parent = node->getParent();
    if (my_parent) {
      if (my_parent->getLeft() == node || my_parent->getRight() == node) {
        if (verbose_optimizer)
          cout << "Relationship of current node to Parent ok" << endl;
      } else {
        COGADB_FATAL_ERROR("Check_Tree_Consistency_Functor::operator()",
                           "Inconsistent Queryplan! Broken Relationship of "
                           "current node to Parent!");
      }
    }
    if (node->getLeft()) {
      if (node->getLeft()->getParent() == node) {
        if (verbose_optimizer)
          cout << "Relationship of current node to Left Child ok" << endl;
      } else {
        COGADB_FATAL_ERROR(
            "Check_Tree_Consistency_Functor::operator(): "
                << std::string(
                       "Inconsistent Queryplan! Broken Relationship of current "
                       "node to Left Child! Node '")
                << node->getLeft()->toString(true)
                << std::string("' has wrong parent pointer! (")
                << node->getLeft()->getParent()->toString(true) + ")",
            "");
      }
      if (node->getLeft() == node) {
        COGADB_FATAL_ERROR(
            "Check_Tree_Consistency_Functor::operator()",
            std::string("Inconsistent Queryplan! Node's '") +
                node->toString(true) +
                std::string("' left pointer refers to itself! (") +
                node->getLeft()->toString(true) + ")");
      }
    }

    if (node->getRight()) {
      if (node->getRight()->getParent() == node) {
        if (verbose_optimizer)
          cout << "Relationship of current node to Right Child ok" << endl;
      } else {
        COGADB_FATAL_ERROR(
            "Check_Tree_Consistency_Functor::operator()",
            std::string("Inconsistent Queryplan! Broken Relationship of "
                        "current node to Right Child! Node '") +
                node->getRight()->toString(true) +
                std::string("' has wrong parent pointer! (") +
                node->getRight()->getParent()->toString(true) + ")");
      }
      if (node->getRight() == node) {
        COGADB_FATAL_ERROR(
            "Check_Tree_Consistency_Functor::operator()",
            std::string("Inconsistent Queryplan! Node's '") +
                node->toString(true) +
                std::string("' right pointer refers to itself! (") +
                node->getRight()->toString(true) + ")");
      }
      if (is_complex_selection(node)) {
        COGADB_FATAL_ERROR(
            "Check_Tree_Consistency_Functor::operator()",
            std::string("Inconsistent Queryplan! Node's '") +
                node->toString(true) +
                std::string("' is a selection (unary operator) but has "
                            "right child!  (") +
                node->getRight()->toString(true) + ")");
      }
    }
    if (node->getParent()) {
      if (node->getParent() == node) {
        COGADB_FATAL_ERROR(
            "Check_Tree_Consistency_Functor::operator()",
            std::string("Inconsistent Queryplan! Node's '") +
                node->toString(true) +
                std::string("' parent pointer refers to itself! (") +
                node->getParent()->toString(true) + ")");
      }
    }

    return false;
  }
};

struct Push_Down_Selection_Functor {
  Push_Down_Selection_Functor(query_processing::LogicalQueryPlanPtr _log_plan)
      : matched_at_least_once(false), log_plan(_log_plan) {}

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Push_Down_Selection_Functor: visiting " << node->toString(true)
           << endl;
    }

    if (node->getLeft() == NULL) return false;

    if (is_simple_selection(node)) {
      //                            boost::shared_ptr<query_processing::logical_operator::Logical_Selection>
      //                            selection=
      //                            boost::dynamic_pointer_cast<query_processing::logical_operator::Logical_Selection>(node);
      //                            assert(selection!=NULL);
      //                            std::list<Attribut>
      //                            transitive_hull_of_attributes =
      //                            getListOfAvailableAttributes(node);
      //
      //                            std::string val =
      //                            selection->getPredicate().getColumn1Name();
      //                            //->getColumnName();
      //
      //                            std::list<Attribut>::iterator it;
      //                            for(it=transitive_hull_of_attributes.begin();it!=transitive_hull_of_attributes.end();++it){
      //                                if(val==it->second){
      //                                    //CoGaDB::optimizer::optimizer_rules::swap(node,node->getLeft());
      //                                    matched_at_least_once=true;
      //                                    return true;
      //                                }
      //                            }

    } else if (is_complex_selection(node)) {
      if (!quiet && verbose && debug)
        cout << "Detected Complex Selection!" << endl;

      boost::shared_ptr<
          query_processing::logical_operator::Logical_ComplexSelection>
          selection = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_ComplexSelection>(
              node);
      assert(selection != NULL);
      // Unary operator
      assert(selection->getLeft() != NULL);
      assert(selection->getRight() == NULL);
      // we cannot push the selection down beyond a scan (or a groupby or
      // projection in case of nested queries), so stop there
      if (is_scan(selection->getLeft()) || is_groupby(selection->getLeft()) ||
          is_projection(selection->getLeft())) {
        selection->couldNotBePushedDownFurther(true);
        return false;
      }
      // if we already psuhed down the selection as much as possible once, we do
      // not consider it anymore
      // it is still possible that other selections where pushed down under the
      // current selection,
      // but multiple selections on the same table are are handled by another
      // optimizer rule
      if (selection->couldNotBePushedDownFurther()) return false;

      std::list<Attribut> transitive_hull_of_attributes_left;
      std::list<Attribut> transitive_hull_of_attributes_right;

      assert(selection->getLeft() != NULL);
      // is the left child of node not a leaf?
      if (node->getLeft()->getLeft() != NULL)
        transitive_hull_of_attributes_left =
            getListOfAvailableAttributesChildrenAndSelf(
                node->getLeft()->getLeft());
      // is binary operator?
      if (node->getLeft()->getRight() != NULL)
        transitive_hull_of_attributes_right =
            getListOfAvailableAttributesChildrenAndSelf(
                node->getLeft()->getRight());

      if (isKNFInTransitiveHull(selection->getKNF_Selection_Expression(),
                                transitive_hull_of_attributes_left)) {
        if (verbose_optimizer)
          cout << "Push Down Selection " << selection->toString(true)
               << " in Left Subtree of " << selection->getLeft()->toString(true)
               << endl;
        // change the pointers to push down the selection
        push_down_selection_in_left_subtree_of_child(selection, log_plan);

        // keep book, whether rule matched once, and repeat until rule had no
        // further matches
        matched_at_least_once = true;
        // push down one selection as far as possible
        return this->operator()(selection);
      } else {
        if (verbose_optimizer)
          cout << "Cannot Push Down Selection " << selection->toString(true)
               << " in Left Subtree of " << selection->getLeft()->toString(true)
               << endl;
      }

      if (isKNFInTransitiveHull(selection->getKNF_Selection_Expression(),
                                transitive_hull_of_attributes_right)) {
        if (verbose_optimizer)
          cout << "Push Down Selection " << selection->toString(true)
               << " in Right Subtree of "
               << selection->getLeft()->toString(true) << endl;
        // change the pointers to push down the selection
        push_down_selection_in_right_subtree_of_child(selection, log_plan);
        // keep book, whether rule matched once, and repeat until rule had no
        // further matches
        matched_at_least_once = true;
        // push down one selection as far as possible
        return this->operator()(selection);
      } else {
        if (verbose_optimizer)
          cout << "Cannot Push Down Selection " << selection->toString(true)
               << " in Right Subtree of "
               << selection->getLeft()->toString(true) << endl;
      }

      // mark selection, so that we do not end up in an infinite loop
      if (verbose_optimizer)
        cout << "Cannot push Down Selection " << selection->toString(true)
             << " more, set the finish bit" << endl;
      selection->couldNotBePushedDownFurther(true);
      //                            log_plan->print();
      log_plan->reassignTreeLevels();
      return false;
    }
    return false;
  }
  bool matched_at_least_once;
  query_processing::LogicalQueryPlanPtr log_plan;
};

struct Decompose_Complex_Selection_Functor {
  Decompose_Complex_Selection_Functor(
      query_processing::LogicalQueryPlan* _log_plan)
      : matched_at_least_once(false), log_plan(_log_plan) {}

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Decompose_Complex_Selection_Functor: visiting "
           << node->toString() << endl;
    }

    if (is_complex_selection(node)) {
      if (!quiet && verbose && debug)
        cout << "Detected Complex Selection!" << endl;

      boost::shared_ptr<
          query_processing::logical_operator::Logical_ComplexSelection>
          selection = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_ComplexSelection>(
              node);
      assert(selection != NULL);
      // Unary operator
      assert(selection->getLeft() != NULL);
      assert(selection->getRight() == NULL);

      KNF_Selection_Expression knf_expr =
          selection->getKNF_Selection_Expression();

      if (knf_expr.disjunctions.size() <= 1) {
        if (verbose_optimizer)
          cout << "Cannot Split Selection which only consist of one "
                  "disjunction: "
               << selection->toString(true) << endl;
        return false;
      }
      query_processing::TypedNodePtr root_of_subplan;
      for (unsigned int i = 0; i < knf_expr.disjunctions.size(); ++i) {
        KNF_Selection_Expression new_knf;
        // add one disjunction!
        new_knf.disjunctions.push_back(knf_expr.disjunctions[i]);
        // create selection operator for this one disjunction
        boost::shared_ptr<
            query_processing::logical_operator::Logical_ComplexSelection>
            tmp(new query_processing::logical_operator::Logical_ComplexSelection(
                new_knf, selection->getMaterializationStatus(),
                selection
                    ->getDeviceConstraint()));  // hype::DeviceConstraint(hype::CPU_ONLY)));
        // integrate operator in current query plan
        query_processing::TypedNodePtr old_root =
            root_of_subplan;  //->getLeft();

        if (old_root) {
          tmp->setLeft(old_root);
          old_root->setParent(tmp);
        }
        root_of_subplan = tmp;
        Check_Tree_Consistency_Functor f;
        f(root_of_subplan);
      }
      // compute leaf of subplan
      query_processing::NodePtr leaf_of_subplan = root_of_subplan;
      while (leaf_of_subplan->getLeft()) {
        leaf_of_subplan = leaf_of_subplan->getLeft();
      }
      // std::cout << "LEAF Node of sub plan: " << leaf_of_subplan->toString()
      // << std::endl;

      /*Integrate Generated Selection Sequence by changing the Pointers*/
      query_processing::NodePtr tail = selection->getLeft();
      query_processing::NodePtr my_parent = selection->getParent();

      assert(root_of_subplan != my_parent);
      // connect root node of subplan with upper part of original query plan
      //                           cout << "Assign " <<
      //                           root_of_subplan->toString(true) << " " <<
      //                           root_of_subplan.get() << " new parent: " <<
      //                           my_parent->toString(true)  << " " <<
      //                           my_parent.get() << endl;
      //                           cout << "Leaf of Subplan: " <<
      //                           leaf_of_subplan->toString(true) << endl;

      // cout << "Parent of root from subtree: " <<
      // root_of_subplan->toString(true) << " has Parent " <<
      // root_of_subplan->getParent()->toString(true) << endl;
      // set the parent pointer of the selection to the root of the generated
      // subplan
      if (my_parent) {
        if (my_parent->getLeft() == selection)
          // selection is left child of parent
          my_parent->setLeft(root_of_subplan);
        else if (my_parent->getRight() == selection)
          // selection is right child of parent
          my_parent->setRight(root_of_subplan);
        else {
          COGADB_FATAL_ERROR("Decompose_Complex_Selection_Functor::operator()",
                             "Inconsistent Queryplan!");
        }

        root_of_subplan->setParent(my_parent);  // my_parent);

        if (verbose_optimizer) {
          cout << "Parent of root from subtree: "
               << root_of_subplan->toString(true) << " has Parent "
               << root_of_subplan->getParent()->toString(true) << endl;
          cout << "Childs of my Parent Pointer (" << my_parent->toString(true)
               << ") Left: " << my_parent->getLeft()->toString(true);
          if (my_parent->getRight())
            cout << " Right: " << my_parent->getRight()->toString(true);
          cout << endl;
        }
      } else {
        // my_parent=
        // parent plan is empty
        // query_processing::NodePtr tmp;
        // tmp=log_plan->getRoot();
        // if(tmp){
        //     cleanupNodeTree(tmp);
        //}
        log_plan->setNewRoot(
            boost::dynamic_pointer_cast<
                typename query_processing::TypedNodePtr::element_type>(
                root_of_subplan));
        // log_plan->reassignParentPointers();
        // log_plan->reassignTreeLevels();
        // explain select * from dates, customer, supplier, part, lineorder
        // where lo_custkey = c_custkey and lo_suppkey = s_suppkey and
        // lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region =
        // 'AMERICA' and s_region = 'AMERICA' and (d_year = 1997 or d_year =
        // 1998) and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2');
      }

      //                           Check_Tree_Consistency_Functor f;
      //                           f(selection->getParent());
      // connect leaf node of subplan with lower part of original query plan
      leaf_of_subplan->setLeft(tail);
      tail->setParent(leaf_of_subplan);

      log_plan->reassignParentPointers();
      log_plan->reassignTreeLevels();

      // delete original selection
      selection->setLeft(query_processing::NodePtr());
      selection->setRight(query_processing::NodePtr());
      selection->setParent(query_processing::NodePtr());
      selection.reset();
      //                            std::cout << "Decompose Selection: Pointer
      //                            use count: " << selection.use_count() <<
      //                            std::endl;

      // update all tree levels under my_parent
      Check_Tree_Consistency_Functor f;
      // f(selection->getParent());
      f(log_plan->getRoot());
      // setLevelsOfTree(my_parent);
      return true;
    }

    return false;
  }
  bool matched_at_least_once;
  query_processing::LogicalQueryPlan* log_plan;
};

struct Compose_Complex_Selections_Functor {
  Compose_Complex_Selections_Functor(
      query_processing::LogicalQueryPlanPtr _log_plan)
      : matched_at_least_once(false), log_plan(_log_plan) {}

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Compose_Complex_Selections_Functor: visiting "
           << node->toString() << endl;
    }

    if (is_complex_selection(node)) {
      if (!quiet && verbose && debug)
        cout << "Detected Complex Selection!" << endl;

      boost::shared_ptr<
          query_processing::logical_operator::Logical_ComplexSelection>
          selection = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_ComplexSelection>(
              node);
      assert(selection != NULL);
      // Unary operator
      assert(selection->getLeft() != NULL);
      assert(selection->getRight() == NULL);

      if (is_complex_selection(selection->getLeft())) {
        boost::shared_ptr<
            query_processing::logical_operator::Logical_ComplexSelection>
            lower_selection = boost::dynamic_pointer_cast<
                query_processing::logical_operator::Logical_ComplexSelection>(
                selection->getLeft());
        assert(lower_selection != NULL);
        // Unary operator
        assert(lower_selection->getLeft() != NULL);
        assert(lower_selection->getRight() == NULL);

        if (verbose_optimizer)
          cout << "Compose Complex Selections: " << selection->toString(true)
               << " and " << lower_selection->toString(true) << endl;
        KNF_Selection_Expression new_knf;

        KNF_Selection_Expression upper_selection_knf =
            selection->getKNF_Selection_Expression();
        KNF_Selection_Expression lower_selection_knf =
            lower_selection->getKNF_Selection_Expression();

        // add disjunctions of current selection
        for (unsigned int i = 0; i < upper_selection_knf.disjunctions.size();
             ++i) {
          new_knf.disjunctions.push_back(upper_selection_knf.disjunctions[i]);
        }
        // add disjunctions of lower selection
        for (unsigned int i = 0; i < lower_selection_knf.disjunctions.size();
             ++i) {
          new_knf.disjunctions.push_back(lower_selection_knf.disjunctions[i]);
        }

        // create selection operator for this one disjunction
        boost::shared_ptr<
            query_processing::logical_operator::Logical_ComplexSelection>
            composed_selection(
                new query_processing::logical_operator::Logical_ComplexSelection(
                    new_knf, selection->getMaterializationStatus(),
                    selection
                        ->getDeviceConstraint()));  // hype::DeviceConstraint(hype::CPU_ONLY)));
        if (verbose_optimizer)
          cout << "new composed Complex Selection: "
               << composed_selection->toString(true) << endl;
        /*Integrate Generated Selection by changing the Pointers*/
        query_processing::NodePtr tail = lower_selection->getLeft();
        query_processing::NodePtr my_parent = selection->getParent();

        if (my_parent) {
          // set the parent pointer of the selection to the generated selection
          if (my_parent->getLeft() == selection)
            // selection is left child of parent
            my_parent->setLeft(composed_selection);
          else if (my_parent->getRight() == selection)
            // selection is right child of parent
            my_parent->setRight(composed_selection);
          else {
            COGADB_FATAL_ERROR(
                "Decompose_Complex_Selection_Functor::operator()",
                "Inconsistent Queryplan!");
          }
          // set the parent pointer of the generated selection
          composed_selection->setParent(my_parent);
        } else {
          log_plan->setNewRoot(
              boost::dynamic_pointer_cast<
                  typename query_processing::TypedNodePtr::element_type>(
                  composed_selection));
          my_parent = composed_selection;
        }
        // update tail
        composed_selection->setLeft(tail);
        tail->setParent(composed_selection);

        // cleanup old selections
        lower_selection->setLeft(query_processing::NodePtr());
        lower_selection->setRight(query_processing::NodePtr());
        lower_selection->setParent(query_processing::NodePtr());
        selection->setLeft(query_processing::NodePtr());
        selection->setRight(query_processing::NodePtr());
        selection->setParent(query_processing::NodePtr());

        // set the correct tree levels
        setLevelsOfTree(my_parent);

        matched_at_least_once = true;
        return true;
      }
      return false;
    }
    return false;
  }
  bool matched_at_least_once;
  query_processing::LogicalQueryPlanPtr log_plan;
};

struct Eliminate_Cross_Join_Functor {
  Eliminate_Cross_Join_Functor() : matched_at_least_once(false) {}

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Eliminate_Cross_Join_Functor: visiting " << node->toString()
           << endl;
    }

    if (is_complex_selection(node)) {
      if (!quiet && verbose && debug)
        cout << "Detected Complex Selection!" << endl;

      boost::shared_ptr<
          query_processing::logical_operator::Logical_ComplexSelection>
          selection = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_ComplexSelection>(
              node);
      assert(selection != NULL);
      // Unary operator
      assert(selection->getLeft() != NULL);
      assert(selection->getRight() == NULL);

      KNF_Selection_Expression knf = selection->getKNF_Selection_Expression();

      if (is_cross_join(node->getLeft())) {
        boost::shared_ptr<query_processing::logical_operator::Logical_CrossJoin>
            cross_join = boost::dynamic_pointer_cast<
                query_processing::logical_operator::Logical_CrossJoin>(
                node->getLeft());
        assert(cross_join != NULL);
        assert(cross_join->getLeft() != NULL);
        assert(cross_join->getRight() != NULL);

        if (containsJoinPredicate(selection->getKNF_Selection_Expression())) {
          list<Predicate> predicates = getJoinPredicates(knf);
          assert(predicates.size() == 1);

          Predicate p = predicates.front();

          /*Check whether the order of the Join PRedicate matches to the
           * underlying tables (left column name-> left subtree etc.)*/
          std::list<Attribut> transitive_hull_of_attributes_left;
          std::list<Attribut> transitive_hull_of_attributes_right;

          transitive_hull_of_attributes_left =
              getListOfAvailableAttributesChildrenAndSelf(
                  cross_join->getLeft());
          transitive_hull_of_attributes_right =
              getListOfAvailableAttributesChildrenAndSelf(
                  cross_join->getRight());

          if (!isColumnInTransitiveHull(p.getColumn1Name(),
                                        transitive_hull_of_attributes_left)) {
            // swap order of Columns in Predicate, so The Left Column (1) refers
            // to an Attribute in the left child
            // and the Right Right Column (2) refers to an Attribute of the
            // right child)
            p.invertOrder();
            // p=Predicate(p.getColumn2Name(),p.getColumn1Name(),p.getPredicateType(),p.getValueComparator());
          }
          assert(isColumnInTransitiveHull(p.getColumn1Name(),
                                          transitive_hull_of_attributes_left));
          assert(isColumnInTransitiveHull(p.getColumn2Name(),
                                          transitive_hull_of_attributes_right));

          // create join node
          boost::shared_ptr<query_processing::logical_operator::Logical_Join>
              join(new query_processing::logical_operator::Logical_Join(
                  p.getColumn1Name(), p.getColumn2Name(), INNER_JOIN,
                  selection->getDeviceConstraint()));
          /*Integrate Generated Selection by changing the Pointers*/
          // query_processing::NodePtr tail = lower_selection->getLeft();
          query_processing::NodePtr my_parent = selection->getParent();

          if (my_parent) {
            // set the parent pointer of the selection to the generated
            // selection
            if (my_parent->getLeft() == selection)
              // selection is left child of parent
              my_parent->setLeft(join);
            else if (my_parent->getRight() == selection)
              // selection is right child of parent
              my_parent->setRight(join);
            else {
              COGADB_FATAL_ERROR(
                  "Decompose_Complex_Selection_Functor::operator()",
                  "Inconsistent Queryplan!");
            }
            join->setParent(my_parent);
          }

          // set the parent, left and right child pointers of the generated join
          // join->setParent(my_parent);
          join->setLeft(cross_join->getLeft());
          join->setRight(cross_join->getRight());

          // if pointer my_parent is NUll, set the join as current parent
          if (!my_parent) my_parent = join;

          // set the parent pointers of the childs from the cross join
          cross_join->getLeft()->setParent(join);
          cross_join->getRight()->setParent(join);

          // set the correct tree levels
          setLevelsOfTree(my_parent);

          matched_at_least_once = true;
          return true;
        }
      }
    }
    return false;
  }
  bool matched_at_least_once;
};

struct Set_Device_Constraints_Functor {
  Set_Device_Constraints_Functor() : matched_at_least_once(false) {}

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Set_Device_Constraints_Functor: visiting "
           << node->toString(true) << endl;
    }
    if (node->getOperationName() == "SORT") {
      boost::shared_ptr<query_processing::logical_operator::Logical_Sort> sort =
          boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Sort>(node);
      assert(sort != NULL);
      const SortAttributeList& col_names = sort->getSortAttributes();
      SortAttributeList::const_iterator cit;
      for (cit = col_names.begin(); cit != col_names.end(); ++cit) {
        AttributeReferencePtr attr =
            getAttributeFromColumnIdentifier(cit->first);
        if (!attr) {
          COGADB_WARNING(
              "Optimzer SetDeviceConstraints: Cannot get Type of Column '"
                  << (cit->first) << "': Column not found in database!",
              "");
          // we cannot determine the column type, so we
          // assume it is not GPU compatible
          node->setDeviceConstraint(hype::DeviceConstraint(hype::CPU_ONLY));
          return false;
        }
        AttributeType type = attr->getAttributeType();
        // we do not support sorting of string columns on the GPU
        if (type == VARCHAR || cit->second == DESCENDING) {
          node->setDeviceConstraint(hype::DeviceConstraint(hype::CPU_ONLY));
          if (!quiet && verbose)
            std::cout << "Set Device Constraint CPU only on Node "
                      << node->toString() << std::endl;
        }
      }

      //;scol_names
    } else if (node->getOperationName() == "GROUPBY") {
      boost::shared_ptr<query_processing::logical_operator::Logical_Groupby>
          groupby = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Groupby>(node);
      assert(groupby != NULL);
      const std::list<ColumnAggregation>& aggregation_functions =
          groupby->getColumnAggregationFunctions();
      std::list<ColumnAggregation>::const_iterator cit;
      for (cit = aggregation_functions.begin();
           cit != aggregation_functions.end(); ++cit) {
        // if we support it do nothing, else set device constraint to CPU_ONLY
        if (cit->second.first == COUNT || cit->second.first == SUM ||
            cit->second.first == MIN || cit->second.first == MAX ||
            cit->second.first == AVERAGE) {
        } else {
          node->setDeviceConstraint(hype::DeviceConstraint(hype::CPU_ONLY));
        }
      }
    } else if (node->getOperationName() == "COMPLEX_SELECTION") {
      boost::shared_ptr<
          query_processing::logical_operator::Logical_ComplexSelection>
          cl = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_ComplexSelection>(
              node);
      assert(cl != NULL);
      KNF_Selection_Expression knf = cl->getKNF_Selection_Expression();

      for (unsigned int i = 0; i < knf.disjunctions.size(); ++i) {
        for (unsigned int j = 0; j < knf.disjunctions[i].size(); ++j) {
          if (knf.disjunctions[i][j].getPredicateType() ==
                  ValueConstantPredicate &&
              knf.disjunctions[i][j].getValueComparator() != EQUAL) {
            std::list<std::pair<ColumnPtr, TablePtr> > l =
                DataDictionary::instance().getColumnsforColumnName(
                    knf.disjunctions[i][j].getColumn1Name());
            assert(l.size() <= 1);
            if (l.size() == 1) {
              if (l.front().first->getType() == VARCHAR) {
                cl->setDeviceConstraint(hype::CPU_ONLY);
              }
            }
          }
        }
      }
    }

    return false;
  }
  bool matched_at_least_once;
};

struct hasJoinInSubtree_functor {
  hasJoinInSubtree_functor() : result_value(false) {}
  void operator()(query_processing::NodePtr node) {
    if (node->getOperationName() == "JOIN" ||
        node->getOperationName() == "PK_FK_JOIN" ||
        node->getOperationName() == "CROSS_JOIN") {
      // found an operator invalidating PK FK constraint
      result_value = true;
    }
  }
  void accumulate(const hasJoinInSubtree_functor& f1,
                  const hasJoinInSubtree_functor& f2) {
    if (f1.result_value || f2.result_value) result_value = true;
  }
  bool result_value;
};

struct hasGatherJoinOnlyInSubtree_functor {
  hasGatherJoinOnlyInSubtree_functor() : result_value(true) {}
  void operator()(query_processing::NodePtr node) {
    boost::shared_ptr<query_processing::logical_operator::Logical_Join> join =
        boost::dynamic_pointer_cast<
            query_processing::logical_operator::Logical_Join>(node);
    if (!join) {
      if (!is_scan(node)) {
        this->result_value = false;
      }
      return;
    }

    if (join->getJoinType() != GATHER_JOIN) {
      this->result_value = false;
    }
  }
  void accumulate(const hasGatherJoinOnlyInSubtree_functor& f1,
                  const hasGatherJoinOnlyInSubtree_functor& f2) {
    if (f1.result_value && f2.result_value) result_value = true;
  }
  bool result_value;
};

struct hasSelectionInSubtree_functor {
  hasSelectionInSubtree_functor() : result_value(false) {}
  void operator()(query_processing::NodePtr node) {
    if (node->getOperationName() == "SELECTION" ||
        node->getOperationName() == "COMPLEX_SELECTION") {
      // found an operator invalidating PK FK constraint
      result_value = true;
    }
  }
  void accumulate(const hasSelectionInSubtree_functor& f1,
                  const hasSelectionInSubtree_functor& f2) {
    if (f1.result_value || f2.result_value) result_value = true;
  }
  bool result_value;
};

bool hasJoinInSubtree(query_processing::NodePtr node) {
  hasJoinInSubtree_functor f;
  hasJoinInSubtree_functor result_functor =
      optimizer::traverse_inorder(f, node);
  return result_functor.result_value;
}

bool hasSelectionInSubtree(query_processing::NodePtr node) {
  hasSelectionInSubtree_functor f;
  hasSelectionInSubtree_functor result_functor =
      optimizer::traverse_inorder(f, node);
  return result_functor.result_value;
}

bool hasGatherJoinOnlyInSubtree(query_processing::NodePtr node) {
  hasGatherJoinOnlyInSubtree_functor f;
  hasGatherJoinOnlyInSubtree_functor result_functor =
      optimizer::traverse_inorder(f, node);
  return result_functor.result_value;
}

struct Replace_Join_with_PK_FK_Join_Functor {
  Replace_Join_with_PK_FK_Join_Functor(
      query_processing::LogicalQueryPlanPtr log_plan_arg)
      : log_plan(log_plan_arg), matched_at_least_once(false) {}

  boost::shared_ptr<query_processing::logical_operator::Logical_PK_FK_Join>
  convertJoinToPK_FK_Join(
      boost::shared_ptr<query_processing::logical_operator::Logical_Join>
          join) {
    return boost::shared_ptr<
        query_processing::logical_operator::Logical_PK_FK_Join>(
        new query_processing::logical_operator::Logical_PK_FK_Join(
            join->getLeftJoinColumnName(), join->getRightJoinColumnName(),
            LOOKUP, join->getDeviceConstraint()));
  }

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Replace_Join_with_PK_FK_Join_Functor: visiting "
           << node->toString(true) << endl;
    }
    if (node->getOperationName() == "JOIN") {
      boost::shared_ptr<query_processing::logical_operator::Logical_Join> join =
          boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Join>(node);
      assert(join != NULL);

      if (!hasJoinInSubtree(node->getLeft()) &&
          !hasJoinInSubtree(node->getRight())) {
        if (DataDictionary::instance().hasPrimaryKeyConstraint(
                join->getLeftJoinColumnName()) &&
            DataDictionary::instance().hasForeignKeyConstraint(
                join->getRightJoinColumnName())) {
          // filtering the table with the primary key column may result in a
          // broken Foreign Key Reference
          if (!hasSelectionInSubtree(node->getLeft())) {
            boost::shared_ptr<
                query_processing::logical_operator::Logical_PK_FK_Join>
                pk_fk_join = convertJoinToPK_FK_Join(join);
            replace_operator(log_plan, join, pk_fk_join);
            matched_at_least_once = true;
          }

        } else if (DataDictionary::instance().hasForeignKeyConstraint(
                       join->getLeftJoinColumnName()) &&
                   DataDictionary::instance().hasPrimaryKeyConstraint(
                       join->getRightJoinColumnName())) {
          // filtering the table with the primary key column may result in a
          // broken Foreign Key Reference
          if (!hasSelectionInSubtree(node->getRight())) {
            // swap join Tables
            query_processing::NodePtr left_child = join->getLeft();
            query_processing::NodePtr right_child = join->getRight();
            join->setLeft(right_child);
            join->setRight(left_child);

            boost::shared_ptr<
                query_processing::logical_operator::Logical_PK_FK_Join>
                pk_fk_join(
                    new query_processing::logical_operator::Logical_PK_FK_Join(
                        join->getRightJoinColumnName(),
                        join->getLeftJoinColumnName(), LOOKUP,
                        join->getDeviceConstraint()));
            // boost::shared_ptr<query_processing::logical_operator::Logical_PK_FK_Join>
            // pk_fk_join = convertJoinToPK_FK_Join(join);
            replace_operator(log_plan, join, pk_fk_join);
            matched_at_least_once = true;
          }
        }
      }

      //;scol_names
    }

    return false;
  }
  query_processing::LogicalQueryPlanPtr log_plan;
  bool matched_at_least_once;
};

ColumnType getColumnType(const std::string& column_name) {
  std::list<std::pair<ColumnPtr, TablePtr> > cols =
      DataDictionary::instance().getColumnsforColumnName(column_name);
  if (cols.size() == 1) {
    return cols.front().first->getColumnType();
  } else if (cols.size() > 1) {
    COGADB_FATAL_ERROR("Ambiguous column name: '" << column_name << "'", "");
  } else {
    COGADB_FATAL_ERROR("Column not found: '" << column_name << "'", "");
  }
}

struct Replace_Join_with_Gather_Join_Functor {
  Replace_Join_with_Gather_Join_Functor(
      query_processing::LogicalQueryPlanPtr log_plan_arg)
      : log_plan(log_plan_arg), matched_at_least_once(false) {}

  //                    boost::shared_ptr<query_processing::logical_operator::Logical_PK_FK_Join>
  //                    convertJoinToPK_FK_Join(boost::shared_ptr<query_processing::logical_operator::Logical_Join>
  //                    join){
  //                        return
  //                        boost::shared_ptr<query_processing::logical_operator::Logical_PK_FK_Join>(new
  //                        query_processing::logical_operator::Logical_PK_FK_Join(join->getLeftJoinColumnName(),
  //                        join->getRightJoinColumnName(), LOOKUP,
  //                        join->getDeviceConstraint()));
  //                    }

  boost::shared_ptr<query_processing::logical_operator::Logical_Join>
  convertJoinToGatherJoin(
      boost::shared_ptr<query_processing::logical_operator::Logical_Join>
          join) {
    return boost::make_shared<query_processing::logical_operator::Logical_Join>(
        join->getLeftJoinColumnName(), join->getRightJoinColumnName(),
        GATHER_JOIN, join->getDeviceConstraint());
  }

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Replace_Join_with_PK_FK_Join_Functor: visiting "
           << node->toString(true) << endl;
    }
    boost::shared_ptr<query_processing::logical_operator::Logical_Join> join =
        boost::dynamic_pointer_cast<
            query_processing::logical_operator::Logical_Join>(node);
    if (join && join->getJoinType() == INNER_JOIN) {
      //                            boost::shared_ptr<query_processing::logical_operator::Logical_Join>
      //                            join =
      //                            boost::dynamic_pointer_cast<query_processing::logical_operator::Logical_Join>(node);
      //                            assert(join!=NULL);

      std::cout << "Has GatherJoin only in Left Subplan: "
                << hasGatherJoinOnlyInSubtree(node->getLeft()) << std::endl;
      std::cout << "Has GatherJoin only in Right Subplan: "
                << hasGatherJoinOnlyInSubtree(node->getRight()) << std::endl;

      if (hasGatherJoinOnlyInSubtree(node->getLeft()) &&
          hasGatherJoinOnlyInSubtree(node->getRight())) {
        if (DataDictionary::instance().hasPrimaryKeyConstraint(
                join->getLeftJoinColumnName()) &&
            DataDictionary::instance().hasForeignKeyConstraint(
                join->getRightJoinColumnName())) {
          std::cout << "Try to apply gather Join, column type of PK column is "
                    << util::getName(
                           getColumnType(join->getLeftJoinColumnName()))
                    << std::endl;

          if (getColumnType(join->getLeftJoinColumnName()) ==
              VOID_COMPRESSED_NUMBER) {
            // filtering the table with the primary key column may result in a
            // broken Foreign Key Reference
            if (!hasSelectionInSubtree(node->getLeft())) {
              boost::shared_ptr<
                  query_processing::logical_operator::Logical_Join>
                  gather_join = convertJoinToGatherJoin(join);

              replace_operator(log_plan, join, gather_join);
              matched_at_least_once = true;
            }
          }

        } else if (DataDictionary::instance().hasForeignKeyConstraint(
                       join->getLeftJoinColumnName()) &&
                   DataDictionary::instance().hasPrimaryKeyConstraint(
                       join->getRightJoinColumnName())) {
          std::cout << "Try to apply gather Join, column type of PK column is "
                    << util::getName(
                           getColumnType(join->getRightJoinColumnName()))
                    << std::endl;

          if (getColumnType(join->getRightJoinColumnName()) ==
              VOID_COMPRESSED_NUMBER) {
            // filtering the table with the primary key column may result in a
            // broken Foreign Key Reference
            if (!hasSelectionInSubtree(node->getRight())) {
              //                                            //swap join Tables
              //                                            query_processing::NodePtr
              //                                            left_child =
              //                                            join->getLeft();
              //                                            query_processing::NodePtr
              //                                            right_child =
              //                                            join->getRight();
              //                                            join->setLeft(right_child);
              //                                            join->setRight(left_child);

              boost::shared_ptr<
                  query_processing::logical_operator::Logical_Join>
                  gather_join = convertJoinToGatherJoin(join);
              replace_operator(log_plan, join, gather_join);

              //                                        boost::shared_ptr<query_processing::logical_operator::Logical_PK_FK_Join>
              //                                        pk_fk_join(new
              //                                        query_processing::logical_operator::Logical_PK_FK_Join(join->getRightJoinColumnName(),
              //                                        join->getLeftJoinColumnName(),
              //                                        LOOKUP,
              //                                        join->getDeviceConstraint()));
              //                                        //boost::shared_ptr<query_processing::logical_operator::Logical_PK_FK_Join>
              //                                        pk_fk_join =
              //                                        convertJoinToPK_FK_Join(join);
              //                                        replace_operator(log_plan,
              //                                        join,pk_fk_join);
              matched_at_least_once = true;
            }
          }
        }
      }

      //;scol_names
    }

    return false;
  }
  query_processing::LogicalQueryPlanPtr log_plan;
  bool matched_at_least_once;
};

struct Replace_Join_with_Fetch_Join_Functor {
  Replace_Join_with_Fetch_Join_Functor(
      query_processing::LogicalQueryPlanPtr log_plan_arg)
      : log_plan(log_plan_arg), matched_at_least_once(false) {}

  boost::shared_ptr<query_processing::logical_operator::Logical_Fetch_Join>
  convertJoinToFetchJoin(
      boost::shared_ptr<query_processing::logical_operator::Logical_Join>
          join) {
    return boost::shared_ptr<
        query_processing::logical_operator::Logical_Fetch_Join>(
        new query_processing::logical_operator::Logical_Fetch_Join(
            join->getLeftJoinColumnName(), join->getRightJoinColumnName(),
            LOOKUP, join->getDeviceConstraint()));
  }

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Replace_Join_with_Fetch_Join_Functor: visiting "
           << node->toString(true) << endl;
    }
    if (node->getOperationName() == "JOIN") {
      boost::shared_ptr<query_processing::logical_operator::Logical_Join> join =
          boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Join>(node);
      assert(join != NULL);

      if (!hasJoinInSubtree(node->getLeft()) &&
          !hasJoinInSubtree(node->getRight())) {
        if (DataDictionary::instance().hasPrimaryKeyConstraint(
                join->getLeftJoinColumnName()) &&
            DataDictionary::instance().hasForeignKeyConstraint(
                join->getRightJoinColumnName())) {
          // filtering the table with the primary key column may result in a
          // broken Foreign Key Reference
          if (!hasSelectionInSubtree(node->getRight())) {
            boost::shared_ptr<
                query_processing::logical_operator::Logical_Fetch_Join>
                fetch_join = convertJoinToFetchJoin(join);
            replace_operator(log_plan, join, fetch_join);
            matched_at_least_once = true;
          }

        } else if (DataDictionary::instance().hasForeignKeyConstraint(
                       join->getLeftJoinColumnName()) &&
                   DataDictionary::instance().hasPrimaryKeyConstraint(
                       join->getRightJoinColumnName())) {
          // filtering the table with the primary key column may result in a
          // broken Foreign Key Reference
          if (!hasSelectionInSubtree(node->getLeft())) {
            // swap join Tables
            query_processing::NodePtr left_child = join->getLeft();
            query_processing::NodePtr right_child = join->getRight();
            join->setLeft(right_child);
            join->setRight(left_child);

            boost::shared_ptr<
                query_processing::logical_operator::Logical_Fetch_Join>
                fetch_join(
                    new query_processing::logical_operator::Logical_Fetch_Join(
                        join->getRightJoinColumnName(),
                        join->getLeftJoinColumnName(), LOOKUP,
                        join->getDeviceConstraint()));
            // boost::shared_ptr<query_processing::logical_operator::Logical_PK_FK_Join>
            // pk_fk_join = convertJoinToPK_FK_Join(join);
            replace_operator(log_plan, join, fetch_join);
            matched_at_least_once = true;
          }
        }
      }

      //;scol_names
    }

    return false;
  }
  query_processing::LogicalQueryPlanPtr log_plan;
  bool matched_at_least_once;
};

struct Check_Existence_Of_Referenced_Columns {
  Check_Existence_Of_Referenced_Columns()
      : all_referenced_columns_exist(true), matched_at_least_once(false) {}

  /* \brief simplify complex column names (e.g., "(a+b)" -> "a", "b")*/
  const std::list<std::string> decomposeComplexColumnNames(
      const std::list<std::string>& column_names) {
    std::list<std::string> simplefied_column_names;
    std::list<std::string>::const_iterator cit;

    for (cit = column_names.begin(); cit != column_names.end(); ++cit) {
      std::vector<std::string> strs;
      boost::split(strs, *cit, boost::is_any_of("()+*-/"));
      for (size_t i = 0; i < strs.size(); ++i) {
        if (!strs[i].empty()) {
          simplefied_column_names.push_back(strs[i]);
        }
        //                                std:cout << "Token: " << strs[i]
        //                                <<std::endl;
      }
    }
    return simplefied_column_names;
  }

  bool operator()(query_processing::NodePtr node) {
    if (!node) return false;

    std::list<std::string> names_of_referenced_columns =
        decomposeComplexColumnNames(node->getNamesOfReferencedColumns());

    std::list<Attribut> available_attributes =
        getListOfAvailableAttributesChildrenOnly(node);
    std::set<std::string> available_attributes_set;

    {
      std::list<Attribut>::const_iterator cit;
      for (cit = available_attributes.begin();
           cit != available_attributes.end(); ++cit) {
        available_attributes_set.insert(cit->second);
      }
    }

    std::list<std::string>::const_iterator cit;

    for (cit = names_of_referenced_columns.begin();
         cit != names_of_referenced_columns.end(); ++cit) {
      if (available_attributes_set.find(*cit) ==
          available_attributes_set.end()) {
        // is number column (e.g., 5)? This hapens in SQL expressions such as
        // 'select A+5 ...'
        // we check whether the string is a double, and if yes,
        // everything is fine
        bool is_numeric_constant = true;
        try {
          boost::lexical_cast<double>(*cit);
        } catch (boost::bad_lexical_cast& e) {
          is_numeric_constant = false;
        }
        if (!is_numeric_constant) {
          std::stringstream ss;
          ss << "Column '" << *cit << "' referenced by Operator '"
             << node->toString(true) << "' not found!";
          //                                COGADB_FATAL_ERROR(ss.str(),"");
          this->all_referenced_columns_exist = false;
          //                                std::cout << ss.str() << std::endl;
          throw SQL::Driver::ParseError(ss.str());
        }
      } else {
        //                                std::cout << "Found Referenced Column:
        //                                " << *cit << std::endl;
      }
    }
    return false;
  }
  void accumulate(const Check_Existence_Of_Referenced_Columns& f1,
                  const Check_Existence_Of_Referenced_Columns& f2) {
    this->all_referenced_columns_exist =
        (f1.all_referenced_columns_exist && f2.all_referenced_columns_exist);
  }
  bool all_referenced_columns_exist;
  bool matched_at_least_once;
};

struct Collect_Joins_Functor {
  Collect_Joins_Functor() : joins(), matched_at_least_once(false) {}

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Collect_Joins_Functor: visiting " << node->toString(true)
           << endl;
    }
    if (node->getOperationName() == "JOIN") {
      boost::shared_ptr<query_processing::logical_operator::Logical_Join> join =
          boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Join>(node);
      assert(join != NULL);
      joins.push_back(join);

      // col_names
    }

    return false;
  }
  void accumulate(const Collect_Joins_Functor& f1,
                  const Collect_Joins_Functor& f2) {
    joins.insert(joins.end(), f1.joins.begin(), f1.joins.end());
    joins.insert(joins.end(), f2.joins.begin(), f2.joins.end());

    // if(f1.result_value || f2.result_value) result_value=true;
  }
  std::list<
      boost::shared_ptr<query_processing::logical_operator::Logical_Join> >
      joins;
  bool matched_at_least_once;
};

struct Collect_Scans_Functor {
  Collect_Scans_Functor() : scans(), matched_at_least_once(false) {}

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Collect_Scans_Functor: visiting " << node->toString(true)
           << endl;
    }
    if (node->getOperationName() == "SCAN") {
      boost::shared_ptr<query_processing::logical_operator::Logical_Scan> scan =
          boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Scan>(node);
      assert(scan != NULL);
      if (!quiet)
        std::cout << "Found SCAN on " << scan->getTableName() << std::endl;
      scans.push_back(scan);
      matched_at_least_once = true;
      // col_names
    }

    return true;
  }
  void accumulate(const Collect_Scans_Functor& f1,
                  const Collect_Scans_Functor& f2) {
    //                       std::list<boost::shared_ptr<query_processing::logical_operator::Logical_Scan>
    //                       >::const_iterator it;
    //                       std::cout << "Detected Scans Left Subtree: ";
    //                       for(it=f1.scans.begin();it!=f1.scans.end();++it){
    //                           TablePtr current_table = (*it)->getTablePtr();
    //                           std::cout << current_table->getName() << ",";
    //                       }
    //                       std::cout << std::endl;
    //                       std::cout << "Detected Scans Right Subtree: ";
    //                       for(it=f2.scans.begin();it!=f2.scans.end();++it){
    //                           TablePtr current_table = (*it)->getTablePtr();
    //                           std::cout << current_table->getName() << ",";
    //                       }
    //                       std::cout << std::endl;

    scans.insert(scans.end(), f1.scans.begin(), f1.scans.end());
    scans.insert(scans.end(), f2.scans.begin(), f2.scans.end());
    scans.unique();

    //                       std::cout << "Combined Results: ";
    //                       for(it=scans.begin();it!=scans.end();++it){
    //                           TablePtr current_table = (*it)->getTablePtr();
    //                           std::cout << current_table->getName() << ",";
    //                       }
    //                       std::cout << std::endl;

    // if(f1.result_value || f2.result_value) result_value=true;
  }
  std::list<
      boost::shared_ptr<query_processing::logical_operator::Logical_Scan> >
      scans;
  bool matched_at_least_once;
};

struct Collect_Joins_And_Cross_Joins_Functor {
  Collect_Joins_And_Cross_Joins_Functor()
      : joins(), cross_joins(), matched_at_least_once(false) {}

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Collect_Joins_And_Cross_Joins_Functor: visiting "
           << node->toString(true) << endl;
    }
    if (node->getOperationName() == "JOIN") {
      boost::shared_ptr<query_processing::logical_operator::Logical_Join> join =
          boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_Join>(node);
      assert(join != NULL);
      joins.push_back(join);

      // col_names
    } else if (node->getOperationName() == "CROSS_JOIN") {
      boost::shared_ptr<query_processing::logical_operator::Logical_CrossJoin>
          cross_join = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_CrossJoin>(node);
      assert(cross_join != NULL);
      cross_joins.push_back(cross_join);

      // col_names
    }

    return false;
  }
  void accumulate(const Collect_Joins_And_Cross_Joins_Functor& f1,
                  const Collect_Joins_And_Cross_Joins_Functor& f2) {
    joins.insert(joins.end(), f1.joins.begin(), f1.joins.end());
    joins.insert(joins.end(), f2.joins.begin(), f2.joins.end());

    cross_joins.insert(cross_joins.end(), f1.cross_joins.begin(),
                       f1.cross_joins.end());
    cross_joins.insert(cross_joins.end(), f2.cross_joins.begin(),
                       f2.cross_joins.end());

    // if(f1.result_value || f2.result_value) result_value=true;
  }
  std::list<
      boost::shared_ptr<query_processing::logical_operator::Logical_Join> >
      joins;
  std::list<
      boost::shared_ptr<query_processing::logical_operator::Logical_CrossJoin> >
      cross_joins;
  bool matched_at_least_once;
};

struct Collect_Complex_Selection_Functor {
  Collect_Complex_Selection_Functor()
      : selections(), matched_at_least_once(false) {}

  bool operator()(query_processing::TypedNodePtr node) {
    if (!node) return false;

    if (verbose_optimizer) {
      cout << "Collect_Complex_Selection_Functor: visiting "
           << node->toString(true) << endl;
    }
    if (is_complex_selection(node)) {
      boost::shared_ptr<
          query_processing::logical_operator::Logical_ComplexSelection>
          join = boost::dynamic_pointer_cast<
              query_processing::logical_operator::Logical_ComplexSelection>(
              node);
      assert(join != NULL);
      selections.push_back(join);

      // col_names
    }

    return false;
  }
  void accumulate(const Collect_Complex_Selection_Functor& f1,
                  const Collect_Complex_Selection_Functor& f2) {
    selections.insert(selections.end(), f1.selections.begin(),
                      f1.selections.end());
    selections.insert(selections.end(), f2.selections.begin(),
                      f2.selections.end());

    // if(f1.result_value || f2.result_value) result_value=true;
  }
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_ComplexSelection> >
      selections;
  bool matched_at_least_once;
};

bool push_down_selections(query_processing::LogicalQueryPlanPtr log_plan) {
  bool matched_at_least_once = true;

  Check_Tree_Consistency_Functor cf;
  log_plan->traverse(cf);

  while (matched_at_least_once) {
    Push_Down_Selection_Functor f(log_plan);
    log_plan->traverse(f);
    // log_plan->traverse_preorder(f);
    matched_at_least_once = f.matched_at_least_once;
  }

  return true;
}

bool decompose_complex_selections(
    query_processing::LogicalQueryPlanPtr log_plan) {
  bool matched_at_least_once = true;

  while (matched_at_least_once) {
    Decompose_Complex_Selection_Functor f(log_plan.get());
    log_plan->traverse(f);
    matched_at_least_once = f.matched_at_least_once;
  }

  return true;
}

bool compose_complex_selections(
    query_processing::LogicalQueryPlanPtr log_plan) {
  bool matched_at_least_once = true;

  while (matched_at_least_once) {
    Compose_Complex_Selections_Functor f(log_plan);
    // log_plan->traverse(f);
    //                        log_plan->traverse_preorder(f);
    log_plan->traverse_inorder(f);
    matched_at_least_once = f.matched_at_least_once;
  }

  return true;
}

bool cross_product_to_join(query_processing::LogicalQueryPlanPtr log_plan) {
  bool matched_at_least_once = true;

  while (matched_at_least_once) {
    Eliminate_Cross_Join_Functor f;
    // log_plan->traverse(f);
    //                        log_plan->traverse_preorder(f);
    log_plan->traverse_inorder(f);
    matched_at_least_once = f.matched_at_least_once;
  }

  return true;
}

void traverse_plan_insert_pipeline_breakers(query_processing::NodePtr node) {
  if (node) {
    if (node->getLeft()) {
      std::string operator_name = node->getLeft()->getOperationName();
      if (node->getOperationName() != "ARTIFICIAL_PIPELINE_BREAKER" &&
          node->getOperationName() != "JOIN" &&
          node->getOperationName() != "SORT BY" && operator_name != "SCAN" &&
          operator_name != "SORT BY" && operator_name != "GROUPBY" &&
          operator_name != "COMPLEX_SELECTION") {
        query_processing::NodePtr pipeline_breaker(
            new query_processing::logical_operator::
                Logical_Artificial_Pipeline_Breaker());
        pipeline_breaker->setLeft(node->getLeft());
        pipeline_breaker->setParent(node);
        node->getLeft()->setParent(pipeline_breaker);
        node->setLeft(pipeline_breaker);
      }
    }

    if (node->getRight()) {
      std::string operator_name = node->getRight()->getOperationName();
      if (node->getOperationName() != "ARTIFICIAL_PIPELINE_BREAKER"
          //         && node->getOperationName()!="JOIN"
          && node->getOperationName() != "SORT BY" && operator_name != "SCAN" &&
          operator_name != "SORT BY" && operator_name != "GROUPBY" &&
          operator_name != "COMPLEX_SELECTION") {
        query_processing::NodePtr pipeline_breaker(
            new query_processing::logical_operator::
                Logical_Artificial_Pipeline_Breaker());
        pipeline_breaker->setLeft(node->getRight());
        pipeline_breaker->setParent(node);
        node->getRight()->setParent(pipeline_breaker);
        node->setRight(pipeline_breaker);
      }
    }

    traverse_plan_insert_pipeline_breakers(node->getLeft());
    traverse_plan_insert_pipeline_breakers(node->getRight());
  }
}

bool add_artificial_pipeline_breakers(
    query_processing::LogicalQueryPlanPtr log_plan) {
  if (!VariableManager::instance().getVariableValueBoolean(
          "code_gen.insert_artificial_pipeline_breakers"))
    return true;

  if (log_plan) {
    query_processing::NodePtr node = log_plan->getRoot();
    traverse_plan_insert_pipeline_breakers(node);
    log_plan->reassignTreeLevels();
  }
  return true;
}

query_processing::NodePtr getNodeWithMinimalLevel(
    query_processing::NodePtr last_join,
    query_processing::NodePtr last_cross_join,
    query_processing::NodePtr last_selection) {
  std::map<size_t, query_processing::NodePtr> my_map;
  if (last_join)
    my_map.insert(std::make_pair(last_join->getLevel(), last_join));
  if (last_cross_join)
    my_map.insert(std::make_pair(last_cross_join->getLevel(), last_cross_join));
  if (last_selection)
    my_map.insert(std::make_pair(last_selection->getLevel(), last_selection));
  return my_map.begin()->second;
}

bool move_fact_table_scan_to_right_side_of_join(
    query_processing::LogicalQueryPlanPtr log_plan) {
  Collect_Joins_And_Cross_Joins_Functor join_func;
  Collect_Joins_And_Cross_Joins_Functor join_func_result =
      optimizer::traverse_inorder(join_func, log_plan->getRoot());
  boost::shared_ptr<query_processing::logical_operator::Logical_Join> last_join;
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_Join> >::iterator jit;
  uint32_t minimal_level = 0;
  for (jit = join_func_result.joins.begin();
       jit != join_func_result.joins.end(); ++jit) {
    if (minimal_level < (*jit)->getLevel()) {
      last_join = *jit;
      minimal_level = (*jit)->getLevel();
    }
  }
  /* continue if there is no join in the plan*/
  if (!last_join) return true;

  if (last_join->getLeft()->getOutputResultSize() >=
      last_join->getRight()->getOutputResultSize()) {
    boost::shared_ptr<query_processing::logical_operator::Logical_Join> join(
        new query_processing::logical_operator::Logical_Join(
            last_join->getRightJoinColumnName(),
            last_join->getLeftJoinColumnName(), INNER_JOIN,
            RuntimeConfiguration::instance().getGlobalDeviceConstraint()));

    join->setParent(last_join->getParent());
    /* swap child nodes */
    join->setLeft(last_join->getRight());
    join->setRight(last_join->getLeft());

    if (join->getParent()) {
      if (join->getParent()->getLeft() == last_join) {
        join->getParent()->setLeft(join);
      } else if (join->getParent()->getRight() == last_join) {
        join->getParent()->setRight(join);
      } else {
        COGADB_FATAL_ERROR("", "");
      }
    }

    last_join->setParent(query_processing::NodePtr());
    last_join->setLeft(query_processing::NodePtr());
    last_join->setRight(query_processing::NodePtr());

    log_plan->reassignTreeLevels();
    log_plan->reassignParentPointers();
  }

  return true;
}

/* This optimizer will only change the query plan if it contains cross joins.
 * This ensures that users can specify the join order manually. However,
 * when the users do not specifiy a join order (plan contains cross joins),
 * we try to resolve the cross join and pick a join order. */
bool remove_cross_joins_and_keep_join_order(
    query_processing::LogicalQueryPlanPtr log_plan) {
  Collect_Joins_And_Cross_Joins_Functor join_func;
  Collect_Joins_And_Cross_Joins_Functor join_func_result =
      optimizer::traverse_inorder(join_func, log_plan->getRoot());
  if (!join_func_result.cross_joins.empty()) {
    return join_order_optimization(log_plan);
  }
  return true;
}

bool join_order_optimization(query_processing::LogicalQueryPlanPtr log_plan) {
  bool matched_at_least_once = true;

  // remove subplan starting from first join and replace it later with new join
  // chain
  // find join node closest to root, this is the node we will replace
  Collect_Joins_And_Cross_Joins_Functor join_func;
  Collect_Joins_And_Cross_Joins_Functor join_func_result =
      optimizer::traverse_inorder(join_func, log_plan->getRoot());

  Collect_Complex_Selection_Functor selection_func;
  Collect_Complex_Selection_Functor selection_func_result =
      optimizer::traverse_inorder(selection_func, log_plan->getRoot());

  // identify fact table
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_Join> >::iterator jit;
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_CrossJoin> >::iterator cjit;
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_ComplexSelection> >::iterator
      csit;

  boost::shared_ptr<query_processing::logical_operator::Logical_Join> last_join;
  boost::shared_ptr<query_processing::logical_operator::Logical_CrossJoin>
      last_cross_join;
  boost::shared_ptr<
      query_processing::logical_operator::Logical_ComplexSelection>
      last_selection;

  uint32_t minimal_level = std::numeric_limits<uint32_t>::max();

  JoinConditions conditions;

  for (jit = join_func_result.joins.begin();
       jit != join_func_result.joins.end(); ++jit) {
    if (minimal_level > (*jit)->getLevel()) {
      last_join = *jit;
      minimal_level = (*jit)->getLevel();
    }
    conditions.insert(Predicate((*jit)->getLeftJoinColumnName(),
                                (*jit)->getRightJoinColumnName(),
                                ValueValuePredicate, EQUAL));
  }
  if (!quiet) printJoinConditions(conditions);

  for (cjit = join_func_result.cross_joins.begin();
       cjit != join_func_result.cross_joins.end(); ++cjit) {
    if (minimal_level > (*cjit)->getLevel()) {
      last_cross_join = *cjit;
      minimal_level = (*cjit)->getLevel();
    }
  }

  for (csit = selection_func_result.selections.begin();
       csit != selection_func_result.selections.end(); ++csit) {
    //                   if(minimal_level>(*csit)->getLevel()){
    //                       last_selection = *csit;
    //                       minimal_level=(*csit)->getLevel();
    //                   }
    const KNF_Selection_Expression& knf =
        (*csit)->getKNF_Selection_Expression();
    for (unsigned int i = 0; i < knf.disjunctions.size(); ++i) {
      for (unsigned int j = 0; j < knf.disjunctions[i].size(); ++j) {
        if (knf.disjunctions[i].at(j).getPredicateType() ==
            ValueValuePredicate) {
          conditions.insert(knf.disjunctions[i].at(j));
          // if selection is join condition, then this node is a candidate
          // for replacement
          if (minimal_level > (*csit)->getLevel()) {
            last_selection = *csit;
            minimal_level = (*csit)->getLevel();
          }
        }
      }
    }

    // conditions.push_back(Predicate((*jit)->getLeftJoinColumnName(),(*jit)->getRightJoinColumnName(),
    // ValueValuePredicate, EQUAL));
  }

  // very simple queries have no basis for join order optimization
  // in this case, we just do nothing
  if (!last_join && !last_cross_join && !last_selection) {
    // COGADB_ERROR("Could not find join nodes or join predicates in query
    // plan!","");
    return true;
  }
  if (!quiet && verbose && debug) {
    if (last_join)
      cout << "Last Join: " << last_join->toString(true)
           << " (Level: " << last_join->getLevel() << ")" << endl;
    if (last_cross_join)
      cout << "Last Join: " << last_cross_join->toString(true)
           << " (Level: " << last_cross_join->getLevel() << ")" << endl;
    if (last_selection)
      cout << "Last Selection: " << last_selection->toString(true)
           << " (Level: " << last_selection->getLevel() << ")" << endl;
  }

  // find out which node is the root of the subplan we are going to repalce with
  // a join path
  query_processing::NodePtr root_of_substitution_plan =
      getNodeWithMinimalLevel(last_join, last_cross_join, last_selection);
  if (!quiet && verbose && debug) {
    if (root_of_substitution_plan)
      cout << "Root of Substitution Plan: "
           << root_of_substitution_plan->toString(true) << endl;
  }
  if (!quiet) printJoinConditions(conditions);

  Collect_Scans_Functor f;
  Collect_Scans_Functor result =
      optimizer::traverse_inorder(f, log_plan->getRoot());
  // identify fact table
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_Scan> >::iterator it;
  std::vector<TablePtr> scans;
  JoinTables join_tables;
  std::map<std::string, query_processing::NodePtr> sub_trees;

  for (it = result.scans.begin(); it != result.scans.end(); ++it) {
    TablePtr current_table = (*it)->getTablePtr();
    scans.push_back(current_table);
    join_tables.insert(
        JoinTable(current_table->getNumberofRows(), current_table->getName()));

    query_processing::NodePtr current_root_of_subplan = *it;
    while (current_root_of_subplan->getParent() &&
           !is_cross_join(current_root_of_subplan->getParent()) &&
           !is_join(current_root_of_subplan->getParent())) {
      current_root_of_subplan = current_root_of_subplan->getParent();
    }
    sub_trees[current_table->getName()] = current_root_of_subplan;
  }
  // not sufficient tables involved to perform join order optimization
  if (scans.size() == 1) return true;
  // check whether we can make a sound join order optimization
  if (conditions.size() + 1 != scans.size()) {
    COGADB_ERROR(
        "Number of join conditions does not fit together with number of "
        "involved tables!"
            << std::endl
            << "Number of Conditions: " << conditions.size() << std::endl
            << "Number of Involved Tables: " << scans.size() << std::endl
            << "Join Order Optimization Omitted!",
        "");
    return true;
  }

  std::pair<std::string, std::list<PartialJoinSpecification> > join_path =
      getJoinPath(join_tables, conditions);

  {
    query_processing::NodePtr current_root = sub_trees[join_path.first];
    assert(current_root != NULL);

    std::list<PartialJoinSpecification>::iterator jp_it;
    for (jp_it = join_path.second.begin(); jp_it != join_path.second.end();
         ++jp_it) {
      //                    boost::shared_ptr<query_processing::logical_operator::Logical_Join>
      //                    join(new
      //                    query_processing::logical_operator::Logical_Join(jp_it->second.getColumn1Name(),
      //                    jp_it->second.getColumn2Name(), LOOKUP,
      //                    hype::CPU_ONLY));
      boost::shared_ptr<query_processing::logical_operator::Logical_Join> join(
          new query_processing::logical_operator::Logical_Join(
              jp_it->second.getColumn2Name(), jp_it->second.getColumn1Name(),
              INNER_JOIN,
              RuntimeConfiguration::instance().getGlobalDeviceConstraint()));
      query_processing::NodePtr left_child = sub_trees[jp_it->first];
      join->setLeft(left_child);
      join->setRight(current_root);
      left_child->setParent(join);
      current_root->setParent(join);
      current_root = join;
    }

    // add new join path to plan, and delete the old one
    query_processing::NodePtr connector_node =
        root_of_substitution_plan->getParent();
    // current_root->toString(true);
    if (!quiet)
      std::cout << "Root of Substitution Plan: "
                << root_of_substitution_plan->toString(true) << std::endl;
    if (connector_node)
      if (!quiet) {
        std::cout << "Connector Node of Initial Plan: "
                  << connector_node->toString(true) << std::endl;
        std::cout << "Root of New Plan: " << current_root->toString(true)
                  << std::endl;
      }
    if (connector_node) {
      query_processing::NodePtr old_plan = connector_node->getLeft();
      connector_node->setLeft(current_root);
      current_root->setParent(connector_node);
      // we could solve this much more elegant and maintainable
      // if we hade a virtual copy constructor allowing us to just copy
      // Node objects when neccessary
      cleanupCrossJoinsandJoinComplexSelectionFromNodeTree(old_plan);
    } else {
      cleanupCrossJoinsandJoinComplexSelectionFromNodeTree(log_plan->getRoot());
      log_plan->setNewRoot(
          boost::dynamic_pointer_cast<
              typename query_processing::TypedNodePtr::element_type>(
              current_root));
    }
    // log_plan->print();
    log_plan->reassignTreeLevels();
    log_plan->reassignParentPointers();
    if (!quiet) log_plan->print();

    //                   query_processing::LogicalQueryPlan
    //                   log_plan(boost::dynamic_pointer_cast<typename
    //                   query_processing::TypedNodePtr::element_type
    //                   >(current_root)); //->toString() << std::endl;
    //                   log_plan.print();
  }

  //               {
  //               std::map<std::string,query_processing::NodePtr>::iterator it;
  //               for(it=sub_trees.begin();it!=sub_trees.end();++it){
  //                   std::cout << it->first << ": " << std::endl;
  //                   //query_processing::NodePtr tmp (new
  //                   hype::queryprocessing::Node(*it->second));
  //                   query_processing::LogicalQueryPlan
  //                   log_plan(boost::dynamic_pointer_cast<typename
  //                   query_processing::TypedNodePtr::element_type
  //                   >(it->second)); //->toString() << std::endl;
  //                   log_plan.print();
  //                   std::cout << std::endl;
  //
  //               }
  //
  //               }

  return true;

  //               query_processing::NodePtr parent_of_subtree =
  //               last_join->getParent();
  //               if(parent_of_subtree->getLeft()==last_join){
  //
  //                  parent_of_subtree->setLeft(inv_join);
  //               }else if(parent_of_subtree->getRight()==last_join){
  //                  parent_of_subtree->setRight(inv_join);
  //               }else{
  //                   COGADB_FATAL_ERROR("Detected broken parent child
  //                   relationship in query plan!","");
  //               }
  //               //replace_operator(log_plan, last_join, inv_join);
  //               inv_join->setParent(parent_of_subtree);
  //               inv_join->setLevel(last_join->getLevel());
  //               scan_fact_table->setLevel(last_join->getLevel()+1);

  // cleanup sub plan
  cleanupNodeTree(last_join);

  //                while(matched_at_least_once){
  //
  //                        //Eliminate_Cross_Join_Functor f;
  //                        //log_plan->traverse_inorder(f);
  //                        //matched_at_least_once=f.matched_at_least_once;
  //                }

  return true;
}

bool rewrite_join_to_pk_fk_join(
    query_processing::LogicalQueryPlanPtr log_plan) {
  bool matched_at_least_once = true;

  while (matched_at_least_once) {
    Replace_Join_with_PK_FK_Join_Functor f(log_plan);
    // log_plan->traverse(f);
    //                        log_plan->traverse_preorder(f);
    log_plan->traverse_inorder(f);
    matched_at_least_once = f.matched_at_least_once;
  }

  return true;
}

bool rewrite_join_to_gather_join(
    query_processing::LogicalQueryPlanPtr log_plan) {
  bool matched_at_least_once = true;

  while (matched_at_least_once) {
    Replace_Join_with_Gather_Join_Functor f(log_plan);
    log_plan->traverse_inorder(f);
    matched_at_least_once = f.matched_at_least_once;
  }

  return true;
}

bool rewrite_join_to_fetch_join(
    query_processing::LogicalQueryPlanPtr log_plan) {
  bool matched_at_least_once = true;

  while (matched_at_least_once) {
    Replace_Join_with_Fetch_Join_Functor f(log_plan);
    // log_plan->traverse(f);
    //                        log_plan->traverse_preorder(f);
    log_plan->traverse_inorder(f);
    matched_at_least_once = f.matched_at_least_once;
  }

  return true;
}

bool rewrite_join_tree_to_invisible_join(
    query_processing::LogicalQueryPlanPtr log_plan) {
  // collect joins
  // identify fact table
  // create invisible join node
  // remove subplan starting from first join and replace it with invisible join
  // node

  // collect joins
  Collect_Scans_Functor f;
  Collect_Scans_Functor result =
      optimizer::traverse_inorder(f, log_plan->getRoot());
  // identify fact table
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_Scan> >::iterator it;
  std::vector<TablePtr> fact_table_candidates;
  if (!quiet) std::cout << "Detected Scans: ";
  for (it = result.scans.begin(); it != result.scans.end(); ++it) {
    TablePtr current_table = (*it)->getTablePtr();
    if (!quiet) std::cout << current_table->getName() << ",";
    std::vector<const ForeignKeyConstraint*> fk_constraints =
        current_table->getForeignKeyConstraints();
    if (!fk_constraints.empty()) {
      // make sure fact table candidates are not filtered
      // ATTENTION: we should put this selection after the invsible join node!
      if (!is_complex_selection((*it)->getParent())) {
        fact_table_candidates.push_back(current_table);
      } else {
        if ((*it)->getParent()) {
          COGADB_WARNING("Invisible_Join: Fact table candidate '"
                             << current_table->getName()
                             << "' cannot be used for invisible join, because "
                                "there is a filter condition '"
                             << (*it)->getParent()->toString(true)
                             << "' on the fact table!",
                         "");
        } else {
          COGADB_WARNING("Invisible_Join: Fact table candidate '"
                             << current_table->getName()
                             << "': There is only a single scan in the query "
                                "plan, omitting invisible join...",
                         "");
        }
      }
    }
  }
  if (!quiet) std::cout << std::endl;
  // this is not a star join query, because no fact table is involved
  if (fact_table_candidates.empty()) {
    COGADB_WARNING(
        "Invisible_Join: No fact table found in query, no changes to query "
        "plan!",
        "");
    return true;
  }
  // we currently support the invisible join only with one involved
  // fact table and a star schema
  // since multiple tables with foreign key constraints indicate we
  // have either a snowflake schema or multiple fact tables,
  // we simply abort here
  assert(fact_table_candidates.size());

  // verify that we choose an actual fact table of a star schema
  //->each table scanned in the query has to have a PK-FK
  // relationship with the fact table
  TablePtr fact_table = fact_table_candidates.front();
  InvisibleJoinSelectionList dimensions;
  std::vector<const ForeignKeyConstraint*> fk_constraints =
      fact_table->getForeignKeyConstraints();
  for (it = result.scans.begin(); it != result.scans.end(); ++it) {
    TablePtr current_table = (*it)->getTablePtr();
    bool found = false;
    for (unsigned int i = 0; i < fk_constraints.size(); ++i) {
      if (fk_constraints[i]->getNameOfPrimaryKeyTable() ==
          current_table->getName()) {
        found = true;

        // ATTENTION: we assume that the push down selection optimizer
        // was executed before this optimizer, so that selections
        // on dimension tables are direct parents of the scan operators
        boost::shared_ptr<
            query_processing::logical_operator::Logical_ComplexSelection>
            selection = boost::dynamic_pointer_cast<
                query_processing::logical_operator::Logical_ComplexSelection>(
                (*it)->getParent());
        KNF_Selection_Expression knf_expr;
        if (selection) {
          knf_expr = selection->getKNF_Selection_Expression();
        }
        dimensions.push_back(InvisibleJoinSelection(
            fk_constraints[i]->getNameOfPrimaryKeyTable(),
            Predicate(fk_constraints[i]->getNameOfPrimaryKeyColumn(),
                      fk_constraints[i]->getNameOfForeignKeyColumn(),
                      ValueValuePredicate, EQUAL),
            knf_expr));
        break;
      }
    }
    if (!found && fact_table->getName() != current_table->getName()) {
      COGADB_WARNING(
          "Invisible_Join: Could not find a primary key constraint between "
          "fact table '"
              << fact_table->getName() << "' and table '"
              << current_table->getName() << "'! No changes in query plan...",
          "");
      return true;
    }
    //                   if(found){
    //                       dimension_tables.push_back(current_table);
    //                   }
  }
  if (dimensions.empty()) {
    COGADB_WARNING(
        "Invisible_Join: Found no Dimension Tables! No changes in query "
        "plan...",
        "");
    return true;
  }
  if (dimensions.size() == 1) {
    COGADB_WARNING("Invisible_Join: Found only one Dimension Table! '"
                       << dimensions.front().table_name
                       << "'! No changes in query plan...",
                   "");
    return true;
  }
  // create invisible join node
  boost::shared_ptr<query_processing::logical_operator::Logical_Scan>
      scan_fact_table(new query_processing::logical_operator::Logical_Scan(
          fact_table->getName()));
  boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
      inv_join(new query_processing::logical_operator::Logical_InvisibleJoin(
          dimensions, LOOKUP, CoGaDB::RuntimeConfiguration::instance()
                                  .getGlobalDeviceConstraint()));
  inv_join->setLeft(scan_fact_table);
  scan_fact_table->setParent(inv_join);

  // remove subplan starting from first join and replace it with invisible join
  // node
  // find join node closest to root, this is the node we will replace with the
  // invisible join
  Collect_Joins_Functor join_func;
  Collect_Joins_Functor join_func_result =
      optimizer::traverse_inorder(join_func, log_plan->getRoot());

  // identify fact table
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_Join> >::iterator jit;
  boost::shared_ptr<query_processing::logical_operator::Logical_Join> last_join;
  uint32_t minimal_level = std::numeric_limits<uint32_t>::max();
  for (jit = join_func_result.joins.begin();
       jit != join_func_result.joins.end(); ++jit) {
    if (minimal_level > (*jit)->getLevel()) {
      last_join = *jit;
      minimal_level = (*jit)->getLevel();
    }
  }
  if (!last_join) {
    COGADB_FATAL_ERROR("Could not find join node in query plan!", "");
  }
  query_processing::NodePtr parent_of_subtree = last_join->getParent();
  if (parent_of_subtree) {
    if (parent_of_subtree->getLeft() == last_join) {
      parent_of_subtree->setLeft(inv_join);
    } else if (parent_of_subtree->getRight() == last_join) {
      parent_of_subtree->setRight(inv_join);
    } else {
      COGADB_FATAL_ERROR(
          "Detected broken parent child relationship in query plan!", "");
    }
  } else {
    // last_join is root of plan, so we need to replace the whole plan
    log_plan->setNewRoot(
        boost::dynamic_pointer_cast<
            typename query_processing::TypedNodePtr::element_type>(inv_join));
  }
  // replace_operator(log_plan, last_join, inv_join);
  inv_join->setParent(parent_of_subtree);
  inv_join->setLevel(last_join->getLevel());
  scan_fact_table->setLevel(last_join->getLevel() + 1);

  // cleanup sub plan
  cleanupNodeTree(last_join);
  return true;
}

bool rewrite_join_tree_to_chain_join(
    query_processing::LogicalQueryPlanPtr log_plan) {
  // collect joins
  // identify fact table
  // create invisible join node
  // remove subplan starting from first join and replace it with invisible join
  // node

  // collect joins
  Collect_Scans_Functor f;
  Collect_Scans_Functor result =
      optimizer::traverse_inorder(f, log_plan->getRoot());
  // identify fact table
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_Scan> >::iterator it;
  std::vector<TablePtr> fact_table_candidates;
  // keep track of selections on the fact table!
  boost::shared_ptr<
      query_processing::logical_operator::Logical_ComplexSelection>
      fact_table_selection;

  if (!quiet) std::cout << "Detected Scans: ";
  for (it = result.scans.begin(); it != result.scans.end(); ++it) {
    TablePtr current_table = (*it)->getTablePtr();
    if (!quiet) std::cout << current_table->getName() << ",";
    std::vector<const ForeignKeyConstraint*> fk_constraints =
        current_table->getForeignKeyConstraints();
    if (!fk_constraints.empty()) {
      fact_table_candidates.push_back(current_table);
      if ((*it)->getParent() && is_complex_selection((*it)->getParent())) {
        boost::shared_ptr<
            query_processing::logical_operator::Logical_ComplexSelection>
            tmp_sel = boost::dynamic_pointer_cast<
                query_processing::logical_operator::Logical_ComplexSelection>(
                (*it)->getParent());
        assert(tmp_sel != NULL);
        // copy complex selection, so we can use it later to insert it into the
        // new query plan
        fact_table_selection = boost::shared_ptr<
            query_processing::logical_operator::Logical_ComplexSelection>(
            new query_processing::logical_operator::Logical_ComplexSelection(
                tmp_sel->getKNF_Selection_Expression(),
                tmp_sel->getMaterializationStatus(),
                tmp_sel
                    ->getDeviceConstraint()  // CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint()
                ));
        fact_table_selection->couldNotBePushedDownFurther(
            tmp_sel->couldNotBePushedDownFurther());
      }
    }
  }
  if (!quiet) std::cout << std::endl;
  // this is not a star join query, because no fact table is involved
  if (fact_table_candidates.empty()) {
    COGADB_WARNING(
        "Invisible_Join: No fact table found in query, no changes to query "
        "plan!",
        "");
    return true;
  }
  // we currently support the invisible join only with one involved
  // fact table and a star schema
  // since multiple tables with foreign key constraints indicate we
  // have either a snowflake schema or multiple fact tables,
  // we simply abort here
  assert(fact_table_candidates.size() == 1);

  // verify that we choose an actual fact table of a star schema
  //->each table scanned in the query has to have a PK-FK
  // relationship with the fact table
  TablePtr fact_table = fact_table_candidates.front();

  query_processing::SelectionMap sel_map;
  JoinPath join_path;
  join_path.first = fact_table->getName();
  // ChainJoinSpecification chain_join_specification;

  // InvisibleJoinSelectionList dimensions;
  std::vector<const ForeignKeyConstraint*> fk_constraints =
      fact_table->getForeignKeyConstraints();
  for (it = result.scans.begin(); it != result.scans.end(); ++it) {
    TablePtr current_table = (*it)->getTablePtr();
    bool found = false;
    for (unsigned int i = 0; i < fk_constraints.size(); ++i) {
      if (fk_constraints[i]->getNameOfPrimaryKeyTable() ==
          current_table->getName()) {
        found = true;

        // ATTENTION: we assume that the push down selection optimizer
        // was executed before this optimizer, so that selections
        // on dimension tables are direct parents of the scan operators
        boost::shared_ptr<
            query_processing::logical_operator::Logical_ComplexSelection>
            selection = boost::dynamic_pointer_cast<
                query_processing::logical_operator::Logical_ComplexSelection>(
                (*it)->getParent());
        KNF_Selection_Expression knf_expr;
        if (selection) {
          knf_expr = selection->getKNF_Selection_Expression();
          sel_map.insert(make_pair(current_table->getName(), knf_expr));
        }
        join_path.second.push_back(PartialJoinSpecification(
            fk_constraints[i]->getNameOfPrimaryKeyTable(),
            Predicate(fk_constraints[i]->getNameOfPrimaryKeyColumn(),
                      fk_constraints[i]->getNameOfForeignKeyColumn(),
                      ValueValuePredicate, EQUAL)));
        // dimensions.push_back(InvisibleJoinSelection(fk_constraints[i]->getNameOfPrimaryKeyTable(),
        // Predicate(fk_constraints[i]->getNameOfPrimaryKeyColumn(),
        // fk_constraints[i]->getNameOfForeignKeyColumn(),ValueValuePredicate,
        // EQUAL), knf_expr));
        break;
      }
    }
    if (!found && fact_table->getName() != current_table->getName()) {
      COGADB_WARNING(
          "Chain_Join: Could not find a primary key constraint between fact "
          "table '"
              << fact_table->getName() << "' and table '"
              << current_table->getName() << "'! No changes in query plan...",
          "");
      return true;
    }
    //                   if(found){
    //                       dimension_tables.push_back(current_table);
    //                   }
  }
  if (join_path.second.empty()) {
    COGADB_WARNING(
        "Chain_Join: Found no Dimension Tables! No changes in query plan...",
        "");
    return true;
  }
  if (join_path.second.size() == 1) {
    COGADB_WARNING("Chain_Join: Found only one Dimension Table! '"
                       << join_path.second.front().first
                       << "'! No changes in query plan...",
                   "");
    return true;
  }
  // create invisible join node
  boost::shared_ptr<query_processing::logical_operator::Logical_Scan>
      scan_fact_table(new query_processing::logical_operator::Logical_Scan(
          fact_table->getName()));
  boost::shared_ptr<query_processing::logical_operator::Logical_ChainJoin>
      inv_join(new query_processing::logical_operator::Logical_ChainJoin(
          query_processing::ChainJoinSpecification(join_path, sel_map), LOOKUP,
          CoGaDB::RuntimeConfiguration::instance()
              .getGlobalDeviceConstraint()));

  // inv_join->setLeft(scan_fact_table);
  // scan_fact_table->setParent(inv_join);
  // if we have found a selection on the fact table earlier, we insert it into
  // the plan
  if (fact_table_selection) {
    inv_join->setLeft(fact_table_selection);
    fact_table_selection->setLeft(scan_fact_table);
    fact_table_selection->setParent(inv_join);
    scan_fact_table->setParent(fact_table_selection);
  } else {
    // otherwise, we omit the selection
    inv_join->setLeft(scan_fact_table);
    scan_fact_table->setParent(inv_join);
  }

  // remove subplan starting from first join and replace it with invisible join
  // node
  // find join node closest to root, this is the node we will replace with the
  // invisible join
  Collect_Joins_Functor join_func;
  Collect_Joins_Functor join_func_result =
      optimizer::traverse_inorder(join_func, log_plan->getRoot());

  // identify fact table
  std::list<boost::shared_ptr<
      query_processing::logical_operator::Logical_Join> >::iterator jit;
  boost::shared_ptr<query_processing::logical_operator::Logical_Join> last_join;
  uint32_t minimal_level = std::numeric_limits<uint32_t>::max();
  for (jit = join_func_result.joins.begin();
       jit != join_func_result.joins.end(); ++jit) {
    if (minimal_level > (*jit)->getLevel()) {
      last_join = *jit;
      minimal_level = (*jit)->getLevel();
    }
  }
  if (!last_join) {
    COGADB_FATAL_ERROR("Could not find join node in query plan!", "");
  }
  query_processing::NodePtr parent_of_subtree = last_join->getParent();
  if (parent_of_subtree) {
    if (parent_of_subtree->getLeft() == last_join) {
      parent_of_subtree->setLeft(inv_join);
    } else if (parent_of_subtree->getRight() == last_join) {
      parent_of_subtree->setRight(inv_join);
    } else {
      COGADB_FATAL_ERROR(
          "Detected broken parent child relationship in query plan!", "");
    }
  } else {
    log_plan->setNewRoot(
        boost::dynamic_pointer_cast<
            typename query_processing::TypedNodePtr::element_type>(inv_join));
  }

  inv_join->setParent(parent_of_subtree);
  // if we have found a selection on the fact table earlier, we insert it into
  // the plan
  if (fact_table_selection) {
    inv_join->setLevel(last_join->getLevel());
    fact_table_selection->setLevel(last_join->getLevel() + 1);
    scan_fact_table->setLevel(last_join->getLevel() + 2);
  } else {
    // otherwise, we omit the selection
    inv_join->setLevel(last_join->getLevel());
    scan_fact_table->setLevel(last_join->getLevel() + 1);
  }

  // cleanup sub plan
  cleanupNodeTree(last_join);
  return true;
}

bool set_device_constaints_for_unsupported_operations(
    query_processing::LogicalQueryPlanPtr log_plan) {
  bool matched_at_least_once = true;

  Check_Tree_Consistency_Functor cf;
  log_plan->traverse(cf);

  while (matched_at_least_once) {
    Set_Device_Constraints_Functor f;
    log_plan->traverse(f);
    // log_plan->traverse_preorder(f);
    matched_at_least_once = f.matched_at_least_once;
  }

  return true;
}

} /* namespace optimizer_rules */

bool checkQueryPlan(query_processing::NodePtr node) {
  //            optimizer_rules::Check_Existence_Of_Referenced_Columns func;
  //            optimizer_rules::Check_Existence_Of_Referenced_Columns
  //            func_result = optimizer::traverse_inorder(func,node);
  //            return func_result.all_referenced_columns_exist;
  return true;
}

} /* namespace optimizer */
} /* namespace CoGaDB */
