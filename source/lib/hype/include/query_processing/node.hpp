
#pragma once

#include <core/scheduling_decision.hpp>
#include <map>
#include <ostream>

#include <boost/shared_ptr.hpp>
#include <core/specification.hpp>
#include <query_optimization/qep.hpp>
#include <query_processing/typed_operator.hpp>
#include <util/get_name.hpp>

#define STRICT_DATA_PLACEMENT_DRIVEN_OPTIMIZATION true
#define HYPE_PULL_BASED_QUERY_CHOPPING

#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
namespace CoGaDB {
  class CodeGenerator;
  typedef boost::shared_ptr<CodeGenerator> CodeGeneratorPtr;
  class QueryContext;
  typedef boost::shared_ptr<QueryContext> QueryContextPtr;
}  // end namespace CoGaDB
#endif

namespace hype {
  namespace queryprocessing {

    // forward declaration
    class Node;
    typedef boost::shared_ptr<Node> NodePtr;
    // forward declaration
    template <typename Type>
    class TypedNode;

    template <typename Type>
    class PhysicalQueryPlan;

    // make Operator Mapper a singleton!
    // when OperatorMapper is instanciated, add second tempel argument where
    // user has to specify a function that returns the Physical_Operator_Map;

    // OperatorMapper<TablePtr,initFunction> mapper;
    // boost::function<std::map<std::string,boost::function<boost::shared_ptr<TypedOperator<Type>
    // > (const stemod::SchedulingDecision&)> > function> ()>
    // each node has to get one

    template <typename Type>
    struct OperatorMapper_Helper_Template {
      typedef Type type;
      typedef TypedNode<Type> TypedLogicalNode;
      typedef boost::shared_ptr<TypedLogicalNode> TypedNodePtr;
      // typedef TypedOperator<Type> TypedOperator;
      typedef boost::shared_ptr<TypedOperator<Type> > TypedOperatorPtr;
      typedef boost::function<TypedOperatorPtr(
          TypedLogicalNode&, const hype::core::SchedulingDecision&,
          TypedOperatorPtr, TypedOperatorPtr)>
          Create_Typed_Operator_Function;
      typedef std::map<std::string, Create_Typed_Operator_Function>
          Physical_Operator_Map;
      typedef boost::shared_ptr<Physical_Operator_Map>
          Physical_Operator_Map_Ptr;
      // typedef boost::function<Physical_Operator_Map_Ptr ()>
      // Map_Init_Function;
      typedef Physical_Operator_Map_Ptr(Map_Init_Function)();
      typedef boost::shared_ptr<PhysicalQueryPlan<Type> > PhysicalQueryPlanPtr;
    };

    template <typename Type, typename OperatorMapper_Helper_Template<
                                 Type>::Map_Init_Function& function>
    class OperatorMapper {
     public:
      // typedef boost::shared_ptr<boost::shared_ptr<TypedOperator<Type> > >
      // (*Create_Typed_Operator_Function)(const stemod::SchedulingDecision&
      // sched_dec);
      // typedef boost::function<boost::shared_ptr<TypedOperator<Type> > (const
      // stemod::SchedulingDecision&)> Create_Typed_Operator_Function_t;
      // typedef std::map<std::string,Create_Typed_Operator_Function_t>
      // Physical_Operator_Map;

      typedef typename OperatorMapper_Helper_Template<
          Type>::Create_Typed_Operator_Function Create_Typed_Operator_Function;
      typedef
          typename OperatorMapper_Helper_Template<Type>::Physical_Operator_Map
              Physical_Operator_Map;
      typedef typename OperatorMapper_Helper_Template<
          Type>::Physical_Operator_Map_Ptr Physical_Operator_Map_Ptr;
      typedef typename OperatorMapper_Helper_Template<Type>::Map_Init_Function
          Map_Init_Function;
      // typedef typename OperatorMapper_Helper_Template<Type>::TypedOperator
      // TypedOperator;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedOperatorPtr
          TypedOperatorPtr;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedLogicalNode
          TypedLogicalNode;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedNodePtr
          TypedNodePtr;

      static const Physical_Operator_Map_Ptr
          static_algorithm_name_to_physical_operator_map_ptr;  //=function();

      OperatorMapper() {}  //: algorithm_name_to_physical_operator_map_(){}

      TypedOperatorPtr getPhysicalOperator(
          TypedLogicalNode& logical_node,
          const hype::core::Tuple& features_of_input_dataset,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child,
          DeviceTypeConstraint dev_constr) const {
        const std::string& operation_name = logical_node.getOperationName();

        core::OperatorSpecification op_spec(
            operation_name, features_of_input_dataset,
            // parameters are the same, because in the query processing engine,
            // we model copy operations explicitely, so the copy cost have to be
            // zero
            hype::PD_Memory_0,   // input data is in CPU RAM
            hype::PD_Memory_0);  // output data has to be stored in CPU RAM

        // DeviceConstraint dev_constr;

        hype::core::SchedulingDecision sched_dec =
            hype::core::Scheduler::instance().getOptimalAlgorithm(op_spec,
                                                                  dev_constr);
        // find operation name in map
        typename Physical_Operator_Map::iterator it =
            static_algorithm_name_to_physical_operator_map_ptr->find(
                sched_dec.getNameofChoosenAlgorithm());
        if (it == static_algorithm_name_to_physical_operator_map_ptr->end()) {
          std::cout << "[HyPE library] FATAL Error! "
                    << typeid(OperatorMapper<Type, function>).name()
                    << ": Missing entry in PhysicalOperatorMap for Algorithm '"
                    << sched_dec.getNameofChoosenAlgorithm() << "'"
                    << std::endl;
          exit(-1);
        }
        TypedOperatorPtr physical_operator;
        // call create function
        if (it->second) {
          physical_operator =
              it->second(logical_node, sched_dec, left_child, right_child);
        } else {
          std::cout << "[HyPE library] FATAL Error! Invalid Function Pointer "
                       "in OperationMapper::getPhysicalOperator()"
                    << std::endl;
          exit(-1);
        }
        // return physical operator
        return physical_operator;
      }

      TypedOperatorPtr createPhysicalOperator(
          TypedLogicalNode& logical_node,
          const hype::core::SchedulingDecision& sched_dec,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child) const {
        // find operation name in map
        typename Physical_Operator_Map::iterator it =
            static_algorithm_name_to_physical_operator_map_ptr->find(
                sched_dec.getNameofChoosenAlgorithm());
        if (it == static_algorithm_name_to_physical_operator_map_ptr->end()) {
          std::cout << "[HyPE library] FATAL Error! "
                    << typeid(OperatorMapper<Type, function>).name()
                    << ": Missing entry in PhysicalOperatorMap for Algorithm '"
                    << sched_dec.getNameofChoosenAlgorithm() << "'"
                    << std::endl;
          exit(-1);
        }
        TypedOperatorPtr physical_operator;
        // call create function
        if (it->second) {
          physical_operator =
              it->second(logical_node, sched_dec, left_child, right_child);
        } else {
          std::cout << "[HyPE library] FATAL Error! Invalid Function Pointer "
                       "in OperationMapper::getPhysicalOperator()"
                    << std::endl;
          exit(-1);
        }
        // return physical operator
        return physical_operator;
      }

      // insert map definition here
      // Physical_Operator_Map algorithm_name_to_physical_operator_map_;
    };

    template <typename Type, typename OperatorMapper_Helper_Template<
                                 Type>::Map_Init_Function& function>
    const typename OperatorMapper<Type, function>::Physical_Operator_Map_Ptr
        OperatorMapper<Type, function>::
            static_algorithm_name_to_physical_operator_map_ptr = function();

    bool scheduleAndExecute(NodePtr logical_operator, std::ostream* out);
    // for logical plan, create derived class, which gets as template argument
    // the Types of the corresponding physical Operators

    class Node {
     public:
      Node(core::DeviceConstraint dev_constr = core::DeviceConstraint());

      virtual ~Node();

      bool isRoot() const;

      bool isLeaf() const;

      const NodePtr getLeft() const;

      const NodePtr getRight() const;

      const NodePtr getParent() const;

      unsigned int getLevel();

      void setLevel(unsigned int level);

      virtual hype::query_optimization::QEP_Node* toQEP_Node() = 0;

      virtual unsigned int getOutputResultSize() const = 0;

      virtual double getSelectivity() const = 0;

      virtual std::string getOperationName() const = 0;

#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
      void produce(CoGaDB::CodeGeneratorPtr code_gen,
                   CoGaDB::QueryContextPtr context);
      void consume(CoGaDB::CodeGeneratorPtr code_gen,
                   CoGaDB::QueryContextPtr context);
#endif

      virtual const std::list<std::string> getNamesOfReferencedColumns() const;

      // this function is needed to determine, whether an operator uses
      // wo phase physical optimization feature, where one operator may
      // generate a query plan itself (e.g., for invisible join and complex
      // selection)
      // in case query chopping is enable, the system can run into a deadlock,
      // because
      // if the operator generating and executing the subplan runs on the same
      // device as one operator in the subplan, a deadlock occures because the
      // generator operator waits for the processing operator, whereas the
      // processing
      // operator waits to be scheduled, but is blocked by the generator
      // operator
      //(operators are executed serially on one processing device)
      virtual bool generatesSubQueryPlan() const;

      virtual std::string toString(bool verbose = false) const;

      void setLeft(NodePtr left);

      void setRight(NodePtr right);

      void setParent(NodePtr parent);

      const core::DeviceConstraint& getDeviceConstraint() const;

      void setDeviceConstraint(const core::DeviceConstraint& dev_constr);

      virtual OperatorPtr getPhysicalOperator() = 0;
      virtual OperatorPtr getPhysicalOperator(
          const core::SchedulingDecision& sched_dec) = 0;

      void waitForChildren();

      void waitForSelf();

      static bool hasChildNodeAborted(NodePtr logical_operator);

      static void chopOffAndExecute(NodePtr logical_operator,
                                    std::ostream* out);

      void notify();

      void notify(NodePtr logical_operator);

      virtual const core::Tuple getFeatureVector() const;

      virtual bool useSelectivityEstimation() const = 0;

      virtual bool isInputDataCachedInGPU();

      void setOutputStream(std::ostream& output_stream);

      std::ostream& getOutputStream();

     private:
#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
      virtual void produce_impl(CoGaDB::CodeGeneratorPtr code_gen,
                                CoGaDB::QueryContextPtr context);
      virtual void consume_impl(CoGaDB::CodeGeneratorPtr code_gen,
                                CoGaDB::QueryContextPtr context);
#endif

     protected:
      NodePtr parent_;
      NodePtr left_;
      NodePtr right_;
      unsigned int level_;
      core::DeviceConstraint dev_constr_;

      boost::mutex mutex_;
      boost::condition_variable condition_variable_;
      volatile bool left_child_ready_;
      volatile bool right_child_ready_;
      volatile bool is_ready_;
      OperatorPtr physical_operator_;
      std::ostream* out;

      // std::string operation_name_;
    };

    //        inline void Node::produce(CoGaDB::CodeGeneratorPtr code_gen,
    //                CoGaDB::QueryContextPtr context){
    //            produce_impl(code_gen, context);
    //        }
    //        inline void Node::consume(CoGaDB::CodeGeneratorPtr code_gen,
    //                CoGaDB::QueryContextPtr context){
    //            consume_impl(code_gen, context);
    //        }

    //        inline bool scheduleAndExecute(NodePtr logical_operator,
    //        std::ostream* out){
    //                    Tuple t = logical_operator->getFeatureVector();
    //
    //                    OperatorSpecification
    //                    op_spec(logical_operator->getOperationName(),
    //                            t,
    //                            //parameters are the same, because in the
    //                            query processing engine, we model copy
    //                            operations explicitely, so the copy cost have
    //                            to be zero
    //                            hype::PD_Memory_0, //input data is in CPU RAM
    //                            hype::PD_Memory_0); //output data has to be
    //                            stored in CPU RAM
    //
    //                    SchedulingDecision sched_dec =
    //                    hype::Scheduler::instance().getOptimalAlgorithm(op_spec,
    //                    logical_operator->getDeviceConstraint());
    //                    OperatorPtr op =
    //                    logical_operator->getPhysicalOperator(sched_dec);
    //                    op->setLogicalOperator(logical_operator);
    //                    op->setOutputStream(*out);
    //                    return op->operator ()();
    //        }

    /*
    Automatic Processing Device Selector
    AProDeS

    Automatic Processing Device Selector for Coprocessing
    AProDeSCo*/

    template <typename Type>
    class TypedNode : public Node {
     public:
      typedef Type NodeElementType;
      typedef
          typename OperatorMapper_Helper_Template<Type>::Physical_Operator_Map
              Physical_Operator_Map;
      typedef typename OperatorMapper_Helper_Template<
          Type>::Physical_Operator_Map_Ptr Physical_Operator_Map_Ptr;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedOperatorPtr
          TypedOperatorPtr;

      TypedNode(core::DeviceConstraint dev_constr) : Node(dev_constr) {}

      virtual hype::query_optimization::QEP_Node* toQEP_Node() = 0;
      virtual TypedOperatorPtr getOptimalOperator(
          TypedOperatorPtr left_child = NULL,
          TypedOperatorPtr right_child = NULL,
          DeviceTypeConstraint dev_constr = ANY_DEVICE) = 0;
      virtual TypedOperatorPtr createPhysicalOperator(
          TypedOperatorPtr left_child, TypedOperatorPtr right_child,
          const hype::core::SchedulingDecision& sched_dec) = 0;
      virtual Physical_Operator_Map_Ptr getPhysical_Operator_Map() = 0;

      virtual bool useSelectivityEstimation() const = 0;

      // virtual const Tuple getFeatureVector() const = 0;

      virtual ~TypedNode() {}
    };

    template <typename Type, typename OperatorMapper_Helper_Template<
                                 Type>::Map_Init_Function& function>
    class TypedNode_Impl : public TypedNode<Type> {
     public:
      typedef
          typename TypedNode<Type>::Physical_Operator_Map Physical_Operator_Map;
      typedef typename TypedNode<Type>::Physical_Operator_Map_Ptr
          Physical_Operator_Map_Ptr;
      typedef typename TypedNode<Type>::TypedOperatorPtr TypedOperatorPtr;
      // typedef OperatorMapper<Type,function> OperatorMapper;

      TypedNode_Impl(
          bool use_selectivity_estimation = false,
          core::DeviceConstraint dev_constr = core::DeviceConstraint())
          : TypedNode<Type>(dev_constr),
            operator_mapper_(),
            use_selectivity_estimation_(use_selectivity_estimation) {
        customSelectivity = -1;
      }

      virtual ~TypedNode_Impl() {}

      virtual hype::query_optimization::QEP_Node* toQEP_Node() {
        hype::core::Tuple t = this->getFeatureVector();
        //                if (this->left_) { //if left child is valid (has to be
        //                by convention!), add input data size
        //                    t.push_back(this->left_->getOutputResultSize());
        //                    if (this->right_) { //if right child is valid (not
        //                    null), add input data size for it as well
        //                        t.push_back(this->right_->getOutputResultSize());
        //                    }
        //                }
        //                if (use_selectivity_estimation_)
        //                    t.push_back(this->getSelectivity()); //add
        //                    selectivity of this operation, when
        //                    use_selectivity_estimation_ is true

        core::OperatorSpecification op_spec(
            this->getOperationName(), t,
            // parameters are the same, because in the query processing engine,
            // we model copy oeprations explicitely, so the copy cost have to be
            // zero
            hype::PD_Memory_0,   // input data is in CPU RAM
            hype::PD_Memory_0);  // output data has to be stored in CPU RAM

        hype::query_optimization::QEP_Node* node =
            new hype::query_optimization::QEP_Node(
                op_spec, this->dev_constr_, this->isInputDataCachedInGPU());
        return node;  // hype::query_optimization::QEPPtr(new
                      // hype::query_optimization::QEP(node));
      }

      virtual bool useSelectivityEstimation() const {
        return this->use_selectivity_estimation_;
      }

      //            virtual const Tuple getFeatureVector() const{
      //                hype::Tuple t;
      //                if (this->left_) { //if left child is valid (has to be
      //                by convention!), add input data size
      //                    //if we already know the correct input data size,
      //                    because the child node was already executed
      //                    //during query chopping, we use the real
      //                    cardinality, other wise we call the estimator
      //                    if(this->left_->physical_operator_){
      //                        t.push_back(this->left_->physical_operator_->getResultSize());
      //                        // ->result_size_;
      //                    }else{
      //                        t.push_back(this->left_->getOutputResultSize());
      //                    }
      //                    if (this->right_) { //if right child is valid (not
      //                    null), add input data size for it as well
      //                        t.push_back(this->right_->getOutputResultSize());
      //                    }
      //                }
      //                if (use_selectivity_estimation_)
      //                    t.push_back(this->getSelectivity()); //add
      //                    selectivity of this operation, when
      //                    use_selectivity_estimation_ is true
      //                return t;
      //            }

      virtual TypedOperatorPtr getOptimalOperator(
          TypedOperatorPtr left_child, TypedOperatorPtr right_child,
          DeviceTypeConstraint dev_constr) {
        hype::core::Tuple t = this->getFeatureVector();
        //                if (this->left_) { //if left child is valid (has to be
        //                by convention!), add input data size
        //                    t.push_back(this->left_->getOutputResultSize());
        //                    if (this->right_) { //if right child is valid (not
        //                    null), add input data size for it as well
        //                        t.push_back(this->right_->getOutputResultSize());
        //                    }
        //                }
        //                if (use_selectivity_estimation_)
        //                    t.push_back(this->getSelectivity()); //add
        //                    selectivity of this operation, when
        //                    use_selectivity_estimation_ is true

        return operator_mapper_.getPhysicalOperator(
            *this, t, left_child, right_child,
            dev_constr);  // this->getOperationName(), t, left_child,
                          // right_child);
      }

      // this method allows an optimizer to create a physical operator based on
      // its final physical plan
      virtual TypedOperatorPtr createPhysicalOperator(
          TypedOperatorPtr left_child, TypedOperatorPtr right_child,
          const hype::core::SchedulingDecision& sched_dec) {
        return operator_mapper_.createPhysicalOperator(*this, sched_dec,
                                                       left_child, right_child);
      }

      virtual Physical_Operator_Map_Ptr getPhysical_Operator_Map() {
        return OperatorMapper<
            Type, function>::static_algorithm_name_to_physical_operator_map_ptr;
      }

      double getSelectivity() const {
        if (customSelectivity != -1) {
          return customSelectivity;
        } else {
          return getCalculatedSelectivity();
        }
      }

      void setSelectivity(double selectivity) {
        customSelectivity = selectivity;
      }
      // boost::shared_ptr<> toPhysicalNode();

      virtual OperatorPtr getPhysicalOperator() {
        return this->physical_operator_;
      }

      virtual OperatorPtr getPhysicalOperator(
          const core::SchedulingDecision& sched_dec) {
        if (this->physical_operator_) {
          return this->physical_operator_;
        } else {
          OperatorPtr left;
          OperatorPtr right;

          if (this->left_) left = this->left_->getPhysicalOperator();
          if (this->right_) right = this->right_->getPhysicalOperator();

          TypedOperatorPtr typed_left_child =
              boost::dynamic_pointer_cast<TypedOperator<Type> >(left);
          TypedOperatorPtr typed_right_child =
              boost::dynamic_pointer_cast<TypedOperator<Type> >(right);
          TypedOperatorPtr result = this->createPhysicalOperator(
              typed_left_child, typed_right_child, sched_dec);
          this->physical_operator_ = result;
          return result;
        }
      }

     protected:
      OperatorMapper<Type, function> operator_mapper_;
      bool use_selectivity_estimation_;
      double customSelectivity;

      virtual double getCalculatedSelectivity() const { return 0.1; }
    };

  }  // end namespace queryprocessing
}  // end namespace hype
