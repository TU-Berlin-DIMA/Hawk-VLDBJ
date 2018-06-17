
#pragma once

#include <boost/any.hpp>
#include <core/scheduler.hpp>
#include <core/specification.hpp>
#include <list>
#include <queue>
#include <util/print.hpp>
#include <util/utility_functions.hpp>

namespace hype {
  namespace query_optimization {

    enum { MAX_CHILDS = 30 };

    struct QEP_Node {
      core::OperatorSpecification op_spec;
      core::DeviceConstraint dev_constr;
      QEP_Node* parent;
      QEP_Node* childs[MAX_CHILDS];
      size_t number_childs;
      core::Algorithm* alg_ptr;
      double cost;
      core::SchedulingDecision* sched_dec_ptr;
      boost::any user_payload;
      void* c_user_payload;
      bool input_data_cached;

      QEP_Node(const core::OperatorSpecification& op_spec_arg,
               const core::DeviceConstraint& dev_constr_arg,
               bool input_data_cached = false);

      QEP_Node(const QEP_Node& other);

      QEP_Node& operator=(const QEP_Node& other);

      void assignAlgorithm(core::Algorithm* alg_ptr);
      // overload for cases where we already know the estimated time
      void assignAlgorithm(core::Algorithm* alg_ptr,
                           const core::EstimatedTime est_exec_time);

      bool addChild(QEP_Node* child);

      bool isChildOf(QEP_Node* node);

      void setChilds(QEP_Node** childs, size_t number_childs);

      void setParent(QEP_Node* parent);

      bool isLeave();

      bool isRoot();

      unsigned int getLevel();

      size_t getTotalNumberofChilds();

      // compute response time
      double computeCost();

      // compute total time
      double computeTotalCost();

      bool isInputDataCached() const;

      std::list<QEP_Node*> preorder_traversal();

      std::list<QEP_Node*> levelorder_traversal();

      std::list<QEP_Node*> reverselevelorder_traversal();

      std::list<QEP_Node*> postorder_traversal();

      template <typename UnaryFunction>
      void traverse_preorder(UnaryFunction& functor, unsigned int level = 0);

      ~QEP_Node();

      std::string toString();

      void cleanupChilds();
    };

    template <typename UnaryFunction>
    void QEP_Node::traverse_preorder(UnaryFunction& functor,
                                     unsigned int level) {
      functor(this, level);
      // std::cout << ">>>>>>>Visiting node: " << this->toString() << std::endl;
      for (unsigned int i = 0; i < this->number_childs; ++i) {
        if (this->childs[i]) {
          this->childs[i]->traverse_preorder(functor, level + 1);
        }
      }
    }

    class QEP;

    void insertCopyOperatorLeaveNode(hype::query_optimization::QEP_Node* leave);
    void insertCopyOperatorRootNode(hype::query_optimization::QEP& qep,
                                    hype::query_optimization::QEP_Node* root);
    void insertCopyOperatorInnerNode(
        hype::query_optimization::QEP_Node* inner_node);

    class QEP {
     public:
      QEP();

      explicit QEP(QEP_Node* root);

      QEP(const QEP& other);

      QEP& operator=(const QEP& other);

      // compute response time
      double computeCost();

      // compute total time
      double computeTotalCost();

      std::string toString(unsigned int indent = 0, bool enable_colors = true,
                           bool print_node_numbers = false);

      std::list<QEP_Node*> preorder_traversal();

      std::list<QEP_Node*> levelorder_traversal();

      std::list<QEP_Node*> reverselevelorder_traversal();

      std::vector<hype::query_optimization::QEP_Node*> getLeafNodes();

      size_t getNumberOfOperators();

      QEP_Node* getRoot();
      /* \brief remove root node and childs nodes from the
       * management of a QEP object, useful for interfacing with C*/
      QEP_Node* removeRoot();

      void setRoot(QEP_Node* new_root);

      ~QEP();

     private:
      QEP_Node* root_;
    };

    typedef boost::shared_ptr<QEP> QEPPtr;

    void optimizeQueryPlan(QEP& plan,
                           const QueryOptimizationHeuristic& heuristic);

  }  // end namespace query_optimization
}  // end namespace hype
