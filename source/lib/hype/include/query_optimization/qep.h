/*
 * File:   qep.h
 * Author: bress
 *
 * Created on December 18, 2013, 12:31 AM
 */

#ifndef QEP_H
#define QEP_H

#include <hype.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct C_QEP_Node {
  /* reserved by system*/
  void* ptr;
  //        /* freely usable by the user*/
  //        void* user_payload;
} C_QEP_Node;

typedef int (*hype_qep_node_function_ptr)(C_QEP_Node* root, int level);

C_QEP_Node* hype_createQEPNode(C_OperatorSpecification* op_spec,
                               C_DeviceConstraint* dev_constr);

char* hype_QEPNodeToString(C_QEP_Node* node);

void hype_setParents(C_QEP_Node* node, C_QEP_Node** parents,
                     size_t num_parents);

C_QEP_Node** hype_getParents(C_QEP_Node* node, size_t* num_parents);

void hype_setChildren(C_QEP_Node* node, C_QEP_Node** childs,
                      size_t num_children);

void hype_addChild(C_QEP_Node* node, C_QEP_Node* child);

int hype_isChildOf(C_QEP_Node* child, C_QEP_Node* node);

C_QEP_Node** hype_getChildren(C_QEP_Node* node, size_t* num_children);

int hype_optimizeQEP(C_QEP_Node* node, QueryOptimizationHeuristic opt_heu);

C_SchedulingDecision* hype_getSchedulingDecision(C_QEP_Node* node);

void hype_printQEPNode(C_QEP_Node* node);

int hype_traverseQEP(C_QEP_Node* root, hype_qep_node_function_ptr func_ptr);

int hype_QEPNodeSetPayload(C_QEP_Node* node, void* payload);

void* hype_QEPNodegetPayload(C_QEP_Node* node);

void hype_freeQEPNode(C_QEP_Node* node);

void hype_recursivefreeQEPNode(C_QEP_Node* node);

#ifdef __cplusplus
}
#endif

#endif /* QEP_H */
