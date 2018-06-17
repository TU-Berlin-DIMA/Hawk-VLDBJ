#pragma once

//#include <core/scheduling_decision.hpp>

//#include <boost/shared_ptr.hpp>
#include <boost/algorithm/string/replace.hpp>

#include <query_optimization/qep.hpp>
#include <query_processing/node.hpp>
#include <query_processing/physical_query_plan.hpp>
#include <stack>
#include <util/begin_ptr.hpp>
#include <util/get_name.hpp>

#include <boost/bind.hpp>
#include <boost/thread.hpp>

#include "processing_device.hpp"
namespace hype {
  namespace queryprocessing {

    struct print_NodePtr_functor {
      print_NodePtr_functor(std::ostream* output_stream) : out(output_stream) {}
      void operator()(NodePtr ptr) const {
        for (unsigned int i = 0; i < ptr->getLevel(); i++) (*out) << "\t";
        // std::cout << "Operation: " << ptr->getOperationName();
        (*out) << ptr->toString(true);
        if (ptr->getDeviceConstraint().getDeviceTypeConstraint() !=
            hype::ANY_DEVICE) {
          (*out) << "\t["
                 << hype::util::getName(
                        ptr->getDeviceConstraint().getDeviceTypeConstraint())
                 << "]";
        }
        (*out) << std::endl;
      }
      std::ostream* out;
    };

    struct init_output_stream_NodePtr_functor {
      init_output_stream_NodePtr_functor(std::ostream* output_stream)
          : out(output_stream) {}
      void operator()(NodePtr ptr) const {
        if (ptr) {
          ptr->setOutputStream(*out);
        }
      }
      std::ostream* out;
    };

    struct print_graph_NodePtr_functor {
      std::ostream& stream;
      print_graph_NodePtr_functor(std::ostream& _stream) : stream(_stream) {}

      void operator()(NodePtr ptr) const {
        std::string params(ptr->toString(true));

        stream << "\tnode_" << ptr << "[label = \"{" << ptr->getOperationName();

        /*
         * FIXME: the following simply parses the verbose description
         * and escapes some significant characters
         * Perhaps we should have special methods for serializing
         * the node parameters into DOT-friendly code.
         */
        size_t pos = params.find(" ");
        if (pos != std::string::npos) {
          params.erase(0, pos + 1);
          boost::algorithm::replace_all(params, "<", "\\<");
          boost::algorithm::replace_all(params, ">", "\\>");

          stream << "|" << params;
        }

        stream << "}\"];" << std::endl;

        if (ptr->getLeft())
          stream << "\tnode_" << ptr << " -> node_" << ptr->getLeft()
                 << ":n [label=\" L\"];" << std::endl;
        if (ptr->getRight())
          stream << "\tnode_" << ptr << " -> node_" << ptr->getRight()
                 << ":n [label=\" R\"];" << std::endl;
      }
    };

    struct assign_treelevel_NodePtr_functor {
      void operator()(NodePtr ptr) {
        if (ptr.get() != NULL) {
          NodePtr left = ptr->getLeft();
          NodePtr right = ptr->getRight();
          if (left.get() != NULL) {
            left->setLevel(ptr->getLevel() + 1);
          }
          if (right.get() != NULL) {
            right->setLevel(ptr->getLevel() + 1);
          }
        }
      }
    };

    struct init_parent_NodePtr_functor {
      void operator()(NodePtr ptr) {
        if (ptr) {
          NodePtr left = ptr->getLeft();
          NodePtr right = ptr->getRight();
          if (left) {
            left->setParent(ptr);
          }
          if (right) {
            right->setParent(ptr);
          }
        }
      }
    };

    struct Cleanup_NodePtr_functor {
      void operator()(NodePtr ptr) {
        if (ptr != NULL) {
          // set parent to NULL
          ptr->setParent(NodePtr());
          ptr->setLeft(NodePtr());
          ptr->setRight(NodePtr());
        }
      }
    };

    template <typename Type>
    struct Construct_Physical_Query_Plan_functor {
      typedef typename OperatorMapper_Helper_Template<Type>::TypedOperatorPtr
          TypedOperatorPtr;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedNodePtr
          TypedNodePtr;

      typedef std::map<TypedNodePtr, TypedOperatorPtr> Map;

      Construct_Physical_Query_Plan_functor(std::ostream& output_stream)
          : root_(), map_log_op_to_phy_op_(), out(output_stream) {
        pthread_mutex_init(&map_log_op_to_phy_op_lock, NULL);
      }

      TypedOperatorPtr getRootOperator() { return root_; }

      void operator()(TypedNodePtr ptr, bool chopped = false) {
        if (!hype::core::quiet && hype::core::verbose && hype::core::debug)
          std::cout << typeid(ptr).name();

        if (ptr) {
          // std::cout << "Allocate Processor for Node: " <<
          // ptr->getOperationName() << std::endl;

          TypedOperatorPtr left_op;
          TypedOperatorPtr right_op;

          TypedNodePtr left =
              boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                  ptr->getLeft());
          TypedNodePtr right =
              boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                  ptr->getRight());
          if (left != NULL) {
            typename Map::iterator it = map_log_op_to_phy_op_.find(left);
            if (it != map_log_op_to_phy_op_.end()) {
              left_op = it->second;
            } else {
              HYPE_FATAL_ERROR("Operator " << left->getOperationName()
                                           << " not found!",
                               std::cout);
            }

            if (chopped && left_op == 0) {
              std::cout << "Child operation of "
                        << ptr.get()->getOperationName()
                        << " not ready: " << left.get()->getOperationName()
                        << std::endl;
              // exit(-1);
            }
          }
          if (right != NULL) {
            assert(left != 0);  // check whether convention is obeyed
            typename Map::iterator it = map_log_op_to_phy_op_.find(right);
            if (it != map_log_op_to_phy_op_.end()) {
              right_op = it->second;
            } else {
              HYPE_FATAL_ERROR("Operator " << right->getOperationName()
                                           << " not found!",
                               std::cout);
            }
          }

          TypedOperatorPtr phy_op_ptr = ptr->getOptimalOperator(
              left_op, right_op, ptr->getDeviceConstraint());
          phy_op_ptr->setOutputStream(out);
          phy_op_ptr->setLogicalOperator(ptr);
          pthread_mutex_lock(&map_log_op_to_phy_op_lock);
          //                    assert(map_log_op_to_phy_op_.find(ptr)==map_log_op_to_phy_op_.end());
          map_log_op_to_phy_op_[ptr] = phy_op_ptr;
          pthread_mutex_unlock(&map_log_op_to_phy_op_lock);
          root_ = phy_op_ptr;  // update root (only important thing that the
                               // correct root node is stored in here at the
                               // end, when get function is called)
          if (chopped) {
            phy_op_ptr->run();
          }
        }
      }

      TypedOperatorPtr root_;
      // maps logical to physical operator
      Map map_log_op_to_phy_op_;
      pthread_mutex_t map_log_op_to_phy_op_lock;
      std::ostream& out;
    };

    inline void recursiveAssign(hype::query_optimization::QEP_Node* qep_node,
                                NodePtr node);

    template <typename Type>
    class LogicalQueryPlan {
     public:
      typedef Type NodeElementType;
      typedef
          typename OperatorMapper_Helper_Template<Type>::Physical_Operator_Map
              Physical_Operator_Map;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedOperatorPtr
          TypedOperatorPtr;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedNodePtr
          TypedNodePtr;
      typedef
          typename OperatorMapper_Helper_Template<Type>::PhysicalQueryPlanPtr
              PhysicalQueryPlanPtr;

      LogicalQueryPlan(TypedNodePtr root,
                       std::ostream& output_stream = std::cout)
          : root_(root), out(&output_stream) {
        root_->setLevel(0);
        // set the tree level of each node appropriately
        traverse(assign_treelevel_NodePtr_functor());
        // set the parent pointer of each node appropriately
        traverse(init_parent_NodePtr_functor());
        // init the output stream for each node
        traverse(init_output_stream_NodePtr_functor(out));
      }

      ~LogicalQueryPlan() { this->reverse_traverse(Cleanup_NodePtr_functor()); }

      const PhysicalQueryPlanPtr convertToPhysicalQueryPlan() {
        Construct_Physical_Query_Plan_functor<Type> functor =
            reverse_traverse(Construct_Physical_Query_Plan_functor<Type>(*out));
        TypedOperatorPtr root = functor.getRootOperator();
        PhysicalQueryPlanPtr ptr(new PhysicalQueryPlan<Type>(root, *out));
        return ptr;
      }

      //                                static const TypedOperatorPtr
      //                                converQEPNodeToPhysicalOperator(hype::query_optimization::QEP_Node*
      //                                qep_node){
      //
      //                                }
      //
      //                                static bool
      //                                recursiveAssign(hype::query_optimization::QEP_Node*
      //                                qep_node, TypedNodePtr node){
      //
      //                                    //hype::query_optimization::QEP_Node*
      //                                    qep_ = node->toQEP_Node();
      //                                    std::vector<hype::query_optimization::QEP_Node*>
      //                                    children;
      //                                    if(node->getLeft()){
      //                                        if(node->getLeft()->getOperationName()!="SCAN"){
      //                                            hype::query_optimization::QEP_Node*
      //                                            qep_left =
      //                                            node->getLeft()->toQEP_Node();
      //                                            if(qep_left){
      //                                                recursiveAssign(qep_left,
      //                                                node->getLeft());
      //                                                children.push_back(qep_left);
      //                                            }
      //                                        }
      //                                    }
      //                                    if(node->getRight()){
      //                                        if(node->getRight()->getOperationName()!="SCAN"){
      //                                            hype::query_optimization::QEP_Node*
      //                                            qep_right =
      //                                            node->getRight()->toQEP_Node();
      //                                            if(qep_right){
      //                                                recursiveAssign(qep_right,
      //                                                node->getRight());
      //                                                children.push_back(qep_right);
      //                                            }
      //                                        }
      //                                    }
      //
      //                                    if(!children.empty()){
      //                                        qep_node->setChilds(hype::util::begin_ptr(children),children.size());
      //                                    }
      //
      //                                }

      const PhysicalQueryPlanPtr convertQEPToPhysicalQueryPlan(
          hype::query_optimization::QEPPtr qep) {
        TypedOperatorPtr
            root;  // = converQEPNodeToPhysicalOperator(ptr->getRoot());
        root = recursiveConvert(qep->getRoot(), this->root_);
        PhysicalQueryPlanPtr ptr(new PhysicalQueryPlan<Type>(root, *out));
        return ptr;
      }

      TypedOperatorPtr recursiveConvert(
          hype::query_optimization::QEP_Node* qep_node,
          TypedNodePtr logical_node) {
        assert(qep_node != NULL);
        assert(logical_node != NULL);
        assert(qep_node->number_childs <= 2);
        assert(qep_node->sched_dec_ptr != NULL);

        // handle special case, where the physical plan contains a node that was
        // not in the logical plan
        // currently, this is only for inserted copy operations
        if (qep_node->op_spec.getOperatorName() !=
            logical_node->getOperationName()) {
          // is copy operation?
          if (!hype::core::quiet && hype::core::verbose && hype::core::debug)
            std::cout << "QEP Operationname: "
                      << qep_node->op_spec.getOperatorName()
                      << "\tLogical Operator Name: "
                      << logical_node->getOperationName() << std::endl;
          assert(qep_node->op_spec.getOperatorName() == "COPY_CP_CPU" ||
                 qep_node->op_spec.getOperatorName() == "COPY_CPU_CP");

          qep_node = qep_node->childs[0];
          assert(qep_node != NULL);
          assert(qep_node->number_childs <= 2);
          assert(qep_node->sched_dec_ptr != NULL);
        }

        TypedOperatorPtr result_physical_operator;

        if (qep_node->number_childs == 0) {
          result_physical_operator = logical_node->createPhysicalOperator(
              TypedOperatorPtr(), TypedOperatorPtr(), *qep_node->sched_dec_ptr);
          result_physical_operator->setLogicalOperator(logical_node);
          // unary operator?
        } else if (qep_node->number_childs == 1) {
          TypedNodePtr left_child_logical_node =
              boost::dynamic_pointer_cast<typename TypedNodePtr::element_type>(
                  logical_node->getLeft());
          assert(qep_node->childs[0] != NULL);
          TypedOperatorPtr child =
              recursiveConvert(qep_node->childs[0], left_child_logical_node);

          result_physical_operator = logical_node->createPhysicalOperator(
              child, TypedOperatorPtr(), *qep_node->sched_dec_ptr);
          result_physical_operator->setLogicalOperator(logical_node);

          // binary operator?
        } else if (qep_node->number_childs == 2) {
          TypedNodePtr left_child_logical_node =
              boost::dynamic_pointer_cast<typename TypedNodePtr::element_type>(
                  logical_node->getLeft());
          TypedNodePtr right_child_logical_node =
              boost::dynamic_pointer_cast<typename TypedNodePtr::element_type>(
                  logical_node->getRight());

          assert(qep_node->childs[0] != NULL);
          assert(qep_node->childs[1] != NULL);
          TypedOperatorPtr left_child =
              recursiveConvert(qep_node->childs[0], left_child_logical_node);
          TypedOperatorPtr right_child =
              recursiveConvert(qep_node->childs[1], right_child_logical_node);

          result_physical_operator = logical_node->createPhysicalOperator(
              left_child, right_child, *qep_node->sched_dec_ptr);
          result_physical_operator->setLogicalOperator(logical_node);
        }
        if (result_physical_operator) {
          result_physical_operator->setOutputStream(
              *out);  // logical_node->getOutputStream());
        }
        assert(result_physical_operator->getLogicalOperator() != NULL);
        return result_physical_operator;
      }

      hype::query_optimization::QEPPtr convertToQEP() {
        hype::query_optimization::QEP_Node* qep_root = root_->toQEP_Node();
        recursiveAssign(qep_root, root_);
        return hype::query_optimization::QEPPtr(
            new hype::query_optimization::QEP(qep_root));
      }

      const PhysicalQueryPlanPtr runChoppedPlan() {
        double begin = double(hype::core::getTimestamp());
        // Construct_Physical_Query_Plan_functor<Type> functor =
        // query_chopping_traversal(Construct_Physical_Query_Plan_functor<Type>());

        Node::chopOffAndExecute(this->getRoot(), out);
        this->getRoot()->waitForSelf();

        OperatorPtr root_node = this->getRoot()->getPhysicalOperator();
        TypedOperatorPtr root =
            boost::dynamic_pointer_cast<TypedOperator<Type> >(root_node);
        PhysicalQueryPlanPtr ptr(new PhysicalQueryPlan<Type>(root, *out));
        ptr->setTimeNeeded(double(hype::core::getTimestamp()) - begin);
        return ptr;
      }

      template <class UnaryFunction>
      UnaryFunction traverse(UnaryFunction f) {
        std::list<TypedNodePtr> queue;

        if (root_.get() == NULL) return f;

        queue.push_back(root_);

        while (!queue.empty()) {
          TypedNodePtr currentNode = queue.front();
          queue.pop_front();  // delete first element

          if (currentNode.get() != NULL) {
            TypedNodePtr left =
                boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                    currentNode->getLeft());
            TypedNodePtr right =
                boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                    currentNode->getRight());

            f(currentNode);

            if (left.get() != NULL) {
              queue.push_back(left);
            }
            if (right.get() != NULL) {
              queue.push_back(right);
            }
          }
        }

        return f;
      }

      template <class UnaryFunction>
      UnaryFunction query_chopping_traversal(UnaryFunction f) {
        chop(&f, root_);
        return f;
      }

      template <class UnaryFunction>
      static void chop(UnaryFunction* f, TypedNodePtr nodePtr) {
        TypedNodePtr left =
            boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                nodePtr->getLeft());
        TypedNodePtr right =
            boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                nodePtr->getRight());
        if (left != NULL && right != NULL) {
          boost::thread_group g;
          g.add_thread(new boost::thread(boost::bind(
              &LogicalQueryPlan<Type>::chop<UnaryFunction>, f, left)));
          g.add_thread(new boost::thread(boost::bind(
              &LogicalQueryPlan<Type>::chop<UnaryFunction>, f, right)));
          g.join_all();
        } else if (left != NULL) {
          chop(f, left);
        }
        (*f)(nodePtr, true);
      }

      template <class UnaryFunction>
      UnaryFunction reverse_traverse(UnaryFunction f) {
        std::list<TypedNodePtr> queue = this->getLevelOrder();
        queue.reverse();
        // process in reverse level order
        typename std::list<TypedNodePtr>::iterator it;
        for (it = queue.begin(); it != queue.end(); ++it) {
          f(*it);
        }

        return f;
      }

      template <class UnaryFunction>
      UnaryFunction traverse_inorder(UnaryFunction f, TypedNodePtr node) {
        if (!node) return f;

        traverse_inorder(
            f, boost::dynamic_pointer_cast<typename TypedNodePtr::element_type>(
                   node->getLeft()));
        f(node);
        traverse_inorder(
            f, boost::dynamic_pointer_cast<typename TypedNodePtr::element_type>(
                   node->getRight()));
        return f;
      }

      template <class UnaryFunction>
      UnaryFunction traverse_inorder(UnaryFunction f) {
        return traverse_inorder(f, root_);
      }

      std::list<TypedNodePtr> getLevelOrder() {
        std::list<TypedNodePtr> queue;
        std::list<TypedNodePtr> result_queue;

        if (root_.get() == NULL) return result_queue;

        queue.push_back(root_);
        result_queue.push_back(root_);

        while (!queue.empty()) {
          TypedNodePtr currentNode = queue.front();
          queue.pop_front();  // delete first element

          if (currentNode.get() != NULL) {
            TypedNodePtr left =
                boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                    currentNode->getLeft());
            TypedNodePtr right =
                boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                    currentNode->getRight());
            // f(currentNode);

            if (left.get() != NULL) {
              queue.push_back(left);
              result_queue.push_back(left);
            }
            if (right.get() != NULL) {
              queue.push_back(right);
              result_queue.push_back(right);
            }
          }
        }
        return result_queue;
      }

      template <class UnaryFunction>
      UnaryFunction traverse_preorder(UnaryFunction f) {
        std::stack<TypedNodePtr> nodeStack;
        TypedNodePtr curr = root_;
        while (true) {
          if (curr.get() != NULL) {
            f(curr);

            nodeStack.push(curr);
            curr =
                boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                    curr->getLeft());
            continue;
          }
          if (!nodeStack.size()) {
            return f;
          }
          curr = nodeStack.top();
          nodeStack.pop();
          curr =
              boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                  curr->getRight());
        }
        return f;
      }

      void print() { traverse_preorder(print_NodePtr_functor(out)); }

      void print_graph(std::ostream& stream = std::cout) {
        stream << "digraph \"Logical Query Plan\" {" << std::endl
               << "\tsplines = false;" << std::endl
               << "\tnode [shape = record, width = 2];" << std::endl;
        traverse_preorder(print_graph_NodePtr_functor(stream));
        stream << "}" << std::endl;
      }

      const TypedNodePtr getRoot() { return root_; }

      void setNewRoot(TypedNodePtr new_root) {
        root_ = new_root;
        // set the tree level of each node appropriately
        traverse(assign_treelevel_NodePtr_functor());
        // set the parent pointer of each node appropriately
        traverse(init_parent_NodePtr_functor());
        // init the output stream for each node
        traverse(init_output_stream_NodePtr_functor(out));
      }

      void reassignTreeLevels() {
        // set the tree level of each node appropriately
        traverse(assign_treelevel_NodePtr_functor());
      }

      void reassignParentPointers() {
        // set the parent pointer of each node appropriately
        traverse(init_parent_NodePtr_functor());
      }

      void setOutputStream(std::ostream& output_stream) {
        this->out = &output_stream;
      }
      std::ostream& getOutputStream() { return *out; }

     private:
      TypedNodePtr root_;
      std::ostream* out;
    };

    inline void recursiveAssign(hype::query_optimization::QEP_Node* qep_node,
                                NodePtr node) {
      // hype::query_optimization::QEP_Node* qep_ = node->toQEP_Node();
      std::vector<hype::query_optimization::QEP_Node*> children;
      if (node->getLeft()) {
        // if(node->getLeft()->getOperationName()!="SCAN")
        {
          hype::query_optimization::QEP_Node* qep_left =
              node->getLeft()->toQEP_Node();
          if (qep_left) {
            recursiveAssign(qep_left, node->getLeft());
            children.push_back(qep_left);
          }
        }
      }
      if (node->getRight()) {
        // if(node->getRight()->getOperationName()!="SCAN")
        {
          hype::query_optimization::QEP_Node* qep_right =
              node->getRight()->toQEP_Node();
          if (qep_right) {
            recursiveAssign(qep_right, node->getRight());
            children.push_back(qep_right);
          }
        }
      }
      if (!children.empty()) {
        qep_node->setChilds(hype::util::begin_ptr(children), children.size());
      }
    }

  }  // end namespace queryprocessing
}  // end namespace hype
