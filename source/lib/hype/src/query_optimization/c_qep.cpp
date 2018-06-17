
#include <cstdio>
#include <cstdlib>

#include <hype.h>
#include <query_optimization/qep.h>
#include <boost/any.hpp>
#include <hype.hpp>
#include <query_optimization/qep.hpp>

using namespace hype;
using namespace hype::query_optimization;

C_QEP_Node* hype_createQEPNode(C_OperatorSpecification* op_spec,
                               C_DeviceConstraint* dev_constr) {
  C_QEP_Node* ret = (C_QEP_Node*)malloc(sizeof(C_QEP_Node));

  if (op_spec == NULL || dev_constr == NULL) return NULL;

  if (op_spec->ptr == NULL || dev_constr->ptr == NULL) {
    return NULL;
  }

  OperatorSpecification* op_ptr =
      static_cast<OperatorSpecification*>(op_spec->ptr);
  DeviceConstraint* dev_ptr = static_cast<DeviceConstraint*>(dev_constr->ptr);
  //    assert(op_ptr!=NULL);
  //    assert(dev_ptr!=NULL);
  ret->ptr = new QEP_Node(*op_ptr, *dev_ptr, false);
  if (hype_QEPNodeSetPayload(ret, NULL)) {
    HYPE_WARNING("Failed to set Payload!", std::cerr);
  }
  // ret->user_payload=NULL;
  return ret;
}

char* hype_QEPNodeToString(C_QEP_Node* node) {
  QEP_Node* qep_node = (QEP_Node*)node->ptr;
  std::string val = qep_node->toString();

  size_t buffer_size = sizeof(char) * (val.size() + 1);
  char* val_str = (char*)malloc(buffer_size);
  std::strncpy(val_str, val.c_str(), buffer_size);
  return val_str;
}

C_QEP_Node* hype_convertQEPNode(QEP_Node* node) {
  C_QEP_Node* ret = (C_QEP_Node*)malloc(sizeof(C_QEP_Node));
  ret->ptr = node;
  return ret;
}

void hype_setParents(C_QEP_Node* node, C_QEP_Node** parents,
                     size_t num_parents) {
  assert(num_parents <= 1);
  if (num_parents == 1) {
    QEP_Node* qep_node = (QEP_Node*)node->ptr;
    QEP_Node* parent = (QEP_Node*)parents[0]->ptr;
    qep_node->setParent(parent);
  }
}

C_QEP_Node** hype_getParents(C_QEP_Node* node, size_t* num_parents) {
  QEP_Node* qep_node = (QEP_Node*)node->ptr;

  if (!qep_node) {
    *num_parents = 0;
    return NULL;
  }
  if (!qep_node->parent) {
    *num_parents = 0;
    return NULL;
  }
  // Note: C interface more general than implementation, currently, only 1
  // parent per node is supported
  C_QEP_Node** qep_parents = (C_QEP_Node**)malloc(sizeof(C_QEP_Node**) * 1);
  qep_parents[0] = hype_convertQEPNode(qep_node->parent);

  *num_parents = 1;  // qep_node->number_childs;
  return qep_parents;
}

void hype_setChildren(C_QEP_Node* node, C_QEP_Node** childs,
                      size_t num_childs) {
  QEP_Node* qep_node = (QEP_Node*)node->ptr;

  QEP_Node* qep_childs[num_childs];
  for (unsigned int i = 0; i < num_childs; ++i) {
    qep_childs[i] = (QEP_Node*)childs[i]->ptr;
  }
  qep_node->setChilds(qep_childs, num_childs);
}

void hype_addChild(C_QEP_Node* node, C_QEP_Node* child) {
  QEP_Node* qep_node = (QEP_Node*)node->ptr;
  QEP_Node* qep_child = (QEP_Node*)child->ptr;
  if (!qep_node || !qep_child) return;
  if (!qep_node->addChild(qep_child)) {
    HYPE_FATAL_ERROR("Maximal Number of Children Exceeded! For QEP Node: "
                         << qep_node->toString(),
                     std::cerr);
  }
}

int hype_isChildOf(C_QEP_Node* child, C_QEP_Node* node) {
  QEP_Node* qep_node = (QEP_Node*)node->ptr;
  QEP_Node* qep_child = (QEP_Node*)child->ptr;
  if (!qep_node || !qep_child) return 0;
  if (qep_child->isChildOf(qep_node))
    return 1;
  else
    return 0;
}

C_QEP_Node** hype_getChildren(C_QEP_Node* node, size_t* num_children) {
  QEP_Node* qep_node = (QEP_Node*)node->ptr;

  C_QEP_Node** qep_children =
      (C_QEP_Node**)malloc(sizeof(C_QEP_Node**) * qep_node->number_childs);
  for (unsigned int i = 0; i < qep_node->number_childs; ++i) {
    qep_children[i] = hype_convertQEPNode(qep_node->childs[i]);
  }
  *num_children = qep_node->number_childs;
  return qep_children;
}

int hype_optimizeQEP(C_QEP_Node* root, QueryOptimizationHeuristic opt_heu) {
  QEP_Node* qep_root = static_cast<QEP_Node*>(root->ptr);
  if (!root) return -1;
  if (!qep_root) return -1;
  // std::cout << qep_node->toString() << std::endl;
  //    QEP plan(new QEP_Node(*qep_root));
  //    std::cout << plan.toString() << std::endl;
  //    std::cout << "Optimizing Query Plan: " << std::endl;

  QEP query_plan_to_optimize(qep_root);
  optimizeQueryPlan(
      query_plan_to_optimize,
      opt_heu);  // BACKTRACKING); //GREEDY_HEURISTIC); //BACKTRACKING);
  //    QEP_Node* qep_root_new = query_plan_to_optimize.removeRoot();
  //    if(qep_root!=qep_root_new){
  //
  //        delete qep_root;
  //        root->ptr=qep_root_new;
  //        {
  //        std::cout << "hype_optimizeQEP:" << std::endl;
  //        QEP plan(new QEP_Node(*qep_root_new));
  //        std::cout << plan.toString() << std::endl;
  //        }
  //    }

  QEP_Node* qep_root_new = new QEP_Node(*query_plan_to_optimize.getRoot());
  {
    std::cout << "hype_optimizeQEP:" << std::endl;
    QEP plan(new QEP_Node(*qep_root_new));
    std::cout << plan.toString() << std::endl;
  }
  root->ptr = qep_root_new;

  return 0;
}

C_SchedulingDecision* hype_getSchedulingDecision(C_QEP_Node* node) {
  QEP_Node* qep_node = (QEP_Node*)node->ptr;
  if (!qep_node->sched_dec_ptr) return NULL;
  C_SchedulingDecision* c_sched_dec =
      (C_SchedulingDecision*)malloc(sizeof(C_SchedulingDecision));
  if (!c_sched_dec) return NULL;
  // create a copy of hte SchedulingDecision, because it will be deleted when
  // the query plan is destructed
  c_sched_dec->ptr =
      new hype::core::SchedulingDecision(*qep_node->sched_dec_ptr);
  return c_sched_dec;
}

void hype_printQEPNode(C_QEP_Node* node) {
  if (node) {
    QEP_Node* qep_node = (QEP_Node*)node->ptr;
    // std::cout << qep_node->toString() << std::endl;
    QEP plan(new QEP_Node(*qep_node));
    std::cout << plan.toString() << std::endl;
  }
}

static int hype_traverseQEP_(C_QEP_Node* root, int level,
                             hype_qep_node_function_ptr func_ptr) {
  int i = 0;
  int error_code = 0;

  // retrieve error code
  error_code = (*func_ptr)(root, level);
  // abort on error
  if (error_code) return error_code;

  size_t num_children = 0;
  C_QEP_Node** children = hype_getChildren(root, &num_children);

  for (i = 0; i < num_children; ++i) {
    if (children[i])
      if (hype_traverseQEP_(children[i], level + 1, func_ptr)) {
        // abort on error
        error_code = 1;
        break;
      }
  }

  for (i = 0; i < num_children; ++i) {
    if (children[i]) hype_freeQEPNode(children[i]);
  }
  if (children) free(children);
  return error_code;
}

int hype_traverseQEP(C_QEP_Node* root, hype_qep_node_function_ptr func_ptr) {
  // return 1 (error) in case function pointer is NULL
  if (!root) return 1;
  if (!func_ptr) return 1;
  return hype_traverseQEP_(root, 0, func_ptr);
}

int hype_QEPNodeSetPayload(C_QEP_Node* node, void* payload) {
  if (node == NULL) return -1;
  QEP_Node* qep_node = (QEP_Node*)node->ptr;
  if (qep_node == NULL) return -1;
  // qep_node->user_payload=boost::any((int*)payload);
  // std::cout << "Assign Payload " << payload << " to Node: " <<
  // qep_node->toString() << std::endl;
  qep_node->c_user_payload = payload;
  return 0;
}

void* hype_QEPNodegetPayload(C_QEP_Node* node) {
  void* payload = NULL;
  if (node == NULL) return NULL;
  QEP_Node* qep_node = (QEP_Node*)node->ptr;
  if (qep_node == NULL) return NULL;
  //   if(!qep_node->user_payload.empty()){
  //       payload=(void*)boost::any_cast<int*>(qep_node->user_payload);
  //   }
  payload = qep_node->c_user_payload;
  // std::cout << "Fetch Payload " << payload << " from Node: " <<
  // qep_node->toString() << std::endl;
  return payload;
}

void hype_freeQEPNode(C_QEP_Node* node) {
  if (node) free(node);
}

void hype_recursivefreeQEPNode(C_QEP_Node* node) {
  if (node) {
    QEP_Node* qep_node = (QEP_Node*)node->ptr;
    delete qep_node;
    free(node);
  }
}
