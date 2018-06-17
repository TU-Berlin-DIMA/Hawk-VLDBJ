
#include <query_processing/node.hpp>

//#define HYPE_PROFILE_PRODUCE_CONSUME

namespace hype {
namespace queryprocessing {

Node::Node(core::DeviceConstraint dev_constr)
    : parent_(),
      left_(),
      right_(),
      level_(0),
      dev_constr_(dev_constr),
      mutex_(),
      condition_variable_(),
      left_child_ready_(false),
      right_child_ready_(false),
      is_ready_(false),
      physical_operator_(),
      out(&std::cout) {}

/*
Node(NodePtr parent) : parent_(parent), left_(), right_(), level_(0){

}

Node(NodePtr parent, NodePtr left, NodePtr right) : parent_(parent),
left_(left), right_(right), level_(0){

}*/

Node::~Node() {
  // physical_operator_.reset();
  // set pointer to NULL
  physical_operator_ = OperatorPtr();
}

bool Node::isRoot() const {
  if (parent_.get() == NULL) return true;
  return false;
}

bool Node::isLeaf() const {
  if (left_.get() == NULL && right_.get() == NULL) return true;
  return false;
}

const NodePtr Node::getLeft() const { return left_; }

const NodePtr Node::getRight() const { return right_; }

const NodePtr Node::getParent() const { return parent_; }

unsigned int Node::getLevel() { return level_; }

void Node::setLevel(unsigned int level) { level_ = level; }

const std::list<std::string> Node::getNamesOfReferencedColumns() const {
  return std::list<std::string>();
}

// this function is needed to determine, whether an operator uses
// wo phase physical optimization feature, where one operator may
// generate a query plan itself (e.g., for invisible join and complex
// selection)
// in case query chopping is enable, the system can run into a deadlock, because
// if the operator generating and executing the subplan runs on the same
// device as one operator in the subplan, a deadlock occures because the
// generator operator waits for the processing operator, whereas the processing
// operator waits to be scheduled, but is blocked by the generator operator
//(operators are executed serially on one processing device)

bool Node::generatesSubQueryPlan() const { return false; }

std::string Node::toString(bool verbose) const {
  if (verbose) {
    return this->getOperationName();
    // return
    // this->getOperationName()+std::string("\t")+util::getName(this->dev_constr_);
  } else {
    return this->getOperationName();
  }
}

// const std::string& getOperationName() const;

void Node::setLeft(NodePtr left) {
  left_ = left;
  // left->setParent(left_);
}

void Node::setRight(NodePtr right) {
  right_ = right;
  // right->setParent(right_);
}

void Node::setParent(NodePtr parent) { parent_ = parent; }

const core::DeviceConstraint& Node::getDeviceConstraint() const {
  return this->dev_constr_;
}

void Node::setDeviceConstraint(const core::DeviceConstraint& dev_constr) {
  this->dev_constr_ = dev_constr;
}

void Node::waitForChildren() {
  {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    while (!this->left_child_ready_ || !this->right_child_ready_) {
      this->condition_variable_.wait(lock);
    }

    //                    boost::unique_lock<boost::mutex> lock(this->mutex_);
    //                    if(this->right_){
    //                        //binary operator
    //                        assert(this->left_!=NULL);
    //                        while(!this->left_->is_ready_ ||
    //                        !this->right_->is_ready_)
    //                        {
    //                            this->condition_variable_.wait(lock);
    //                        }
    //                    }else{
    //                        //unary operator
    //                        while(!this->left_->is_ready_)
    //                        {
    //                            this->condition_variable_.wait(lock);
    //                        }
    //                    }
  }
}

void Node::waitForSelf() {
  {
    boost::unique_lock<boost::mutex> lock(this->mutex_);
    while (!this->is_ready_) {
      this->condition_variable_.wait(lock);
    }
  }
}

bool Node::hasChildNodeAborted(NodePtr logical_operator) {
  assert(logical_operator != NULL);
  bool left_aborted = false;
  bool right_aborted = false;
  if (logical_operator->getLeft() &&
      logical_operator->getLeft()->getPhysicalOperator()) {
    left_aborted =
        logical_operator->getLeft()->getPhysicalOperator()->hasAborted();
    left_aborted |= hasChildNodeAborted(logical_operator->getLeft());
  }
  if (logical_operator->getRight() &&
      logical_operator->getRight()->getPhysicalOperator()) {
    right_aborted =
        logical_operator->getRight()->getPhysicalOperator()->hasAborted();
    right_aborted |= hasChildNodeAborted(logical_operator->getRight());
  }
  if (left_aborted || right_aborted) {
    return true;
  } else {
    return false;
  }
}

void Node::chopOffAndExecute(NodePtr logical_operator, std::ostream* out) {
  if (!logical_operator) return;
  assert(out != NULL);
  logical_operator->setOutputStream(*out);
  if (logical_operator->isLeaf()) {
    if (hype::core::Runtime_Configuration::instance()
            .isPullBasedQueryChoppingEnabled()) {
      // queue operator in global operator stream
      core::Scheduler::instance().addIntoGlobalOperatorStream(logical_operator);
    } else {
      scheduleAndExecute(logical_operator, out);
    }
    return;
    // logical_operator->waitForSelf();

  } else {
#ifdef HYPE_ENABLE_PARALLEL_QUERY_PLAN_EVALUATION
    boost::thread_group threads;
    if (logical_operator->right_) {
      // chopOffAndExecute(logical_operator->right_);
      threads.add_thread(new boost::thread(boost::bind(
          &Node::chopOffAndExecute, logical_operator->right_, out)));
    } else {
      logical_operator->right_child_ready_ = true;
    }
    if (logical_operator->left_) {
      // chopOffAndExecute(logical_operator->left_);
      threads.add_thread(new boost::thread(
          boost::bind(&Node::chopOffAndExecute, logical_operator->left_, out)));
    } else {
      logical_operator->left_child_ready_ = true;
    }
    threads.join_all();
#else
    if (logical_operator->right_) {
      Node::chopOffAndExecute(logical_operator->right_, out);
    } else {
      logical_operator->right_child_ready_ = true;
    }
    if (logical_operator->left_) {
      Node::chopOffAndExecute(logical_operator->left_, out);
    } else {
      logical_operator->left_child_ready_ = true;
    }
#endif

    logical_operator->waitForChildren();

    // operators generating query plans are executed immediately
    //(otherwise, the system may run into a deadlock, see also
    // the comments in method generatesSubQueryPlan()), all other
    // operators are executed by the execution engine
    // therefore, the plan generator should avoid doing data processing and
    // instead put the computational intensive tasks in the generated plans,
    // which are then executed by the execution engine
    if (logical_operator->generatesSubQueryPlan()) {
      scheduleAndExecute(logical_operator, out);
      //                        Tuple t = logical_operator->getFeatureVector();
      //
      //                        OperatorSpecification
      //                        op_spec(logical_operator->getOperationName(),
      //                                t,
      //                                //parameters are the same, because in
      //                                the query processing engine, we model
      //                                copy operations explicitely, so the copy
      //                                cost have to be zero
      //                                hype::PD_Memory_0, //input data is in
      //                                CPU RAM
      //                                hype::PD_Memory_0); //output data has to
      //                                be stored in CPU RAM
      //
      //                        SchedulingDecision sched_dec =
      //                        hype::Scheduler::instance().getOptimalAlgorithm(op_spec,
      //                        logical_operator->getDeviceConstraint());
      //                        OperatorPtr op =
      //                        logical_operator->getPhysicalOperator(sched_dec);
      //                        op->setLogicalOperator(logical_operator);
      //                        op->setOutputStream(*out);
      //                        op->operator ()();
    } else {
      if (core::Runtime_Configuration::instance()
              .getDataPlacementDrivenOptimization()) {
        // consider current data placement in query otpimization
        // if we find that a fetch joins join indexes is cached on the GPU, we
        // set a GPU_ONLY cosntraint
        // otherwise, we set a CPU_ONLY constrained
        // note that this requires to have a data placement routine that
        // regularily puts the most recently used
        // access structures on the GPU!
        if (logical_operator->isInputDataCachedInGPU() &&
            logical_operator->dev_constr_ != hype::CPU_ONLY) {
          if (STRICT_DATA_PLACEMENT_DRIVEN_OPTIMIZATION ||
              logical_operator->getOperationName() == "COLUMN_FETCH_JOIN" ||
              logical_operator->getOperationName() == "FETCH_JOIN") {
            // std::cout << "Annote COLUMN_FETCH_JOIN as GPU_ONLY!" <<
            // std::endl;
            logical_operator->dev_constr_ = hype::GPU_ONLY;
          }
          // logical_operator->dev_constr_=hype::GPU_ONLY;
        } else if (!logical_operator->isInputDataCachedInGPU() &&
                   (STRICT_DATA_PLACEMENT_DRIVEN_OPTIMIZATION ||
                    logical_operator->getOperationName() ==
                        "COLUMN_FETCH_JOIN" ||
                    logical_operator->getOperationName() == "FETCH_JOIN")) {
          logical_operator->dev_constr_ = hype::CPU_ONLY;
        }
      }
#define ENABLE_CHAINING_FOR_QUERY_CHOPPING
#ifdef ENABLE_CHAINING_FOR_QUERY_CHOPPING
      // if any operator in the subtree aborted, then we
      // enforce a CPU only execution
      if (hasChildNodeAborted(logical_operator)) {
        logical_operator->setDeviceConstraint(hype::CPU_ONLY);
      }
      // chain operators on same processor to avoid copy operations
      if (logical_operator->getLeft() && logical_operator->getRight()) {
        // binary operators
        if (logical_operator->getLeft()->getPhysicalOperator() &&
            logical_operator->getRight()->getPhysicalOperator()) {
          ProcessingDeviceType left_pt = logical_operator->getLeft()
                                             ->getPhysicalOperator()
                                             ->getSchedulingDecision()
                                             .getDeviceSpecification()
                                             .getDeviceType();
          ProcessingDeviceType right_pt = logical_operator->getRight()
                                              ->getPhysicalOperator()
                                              ->getSchedulingDecision()
                                              .getDeviceSpecification()
                                              .getDeviceType();

          DeviceTypeConstraint left_dev_constr =
              util::getDeviceConstraintForProcessingDeviceType(left_pt);
          DeviceTypeConstraint right_dev_constr =
              util::getDeviceConstraintForProcessingDeviceType(right_pt);

          // we omit chain breakers in our placement heuristic,
          // and handle them differently than normal operators
          if (!util::isChainBreaker(logical_operator->getOperationName())) {
            // if at least one of the childs is a
            // chain breaker, we let HyPE's operator
            // placement decide on which processor the
            // chain is continued
            if (util::isChainBreaker(
                    logical_operator->getLeft()->getOperationName()) ||
                util::isChainBreaker(
                    logical_operator->getRight()->getOperationName())
                // it is commonly aceptable to change the
                // processor for a binary operator when at
                // least one child nodes is a selection
                ||
                logical_operator->getLeft()->getOperationName() ==
                    "COMPLEX_SELECTION" ||
                logical_operator->getRight()->getOperationName() ==
                    "COMPLEX_SELECTION") {
              // logical_operator->setDeviceConstraint(hype::ANY_DEVICE);
            } else if (left_pt == right_pt &&
                       logical_operator->dev_constr_ == ANY_DEVICE) {
              // continue chain (but continue chain on GPU only in case no
              // operator aborted!)
              if ((left_pt == hype::GPU &&
                   !logical_operator->getLeft()
                        ->getPhysicalOperator()
                        ->hasAborted() &&
                   !logical_operator->getRight()
                        ->getPhysicalOperator()
                        ->hasAborted())) {
                logical_operator->setDeviceConstraint(
                    hype::GPU_ONLY);  // left_dev_constr);
                // std::cout << "[QC]: Operator aborted, will not continue
                // chain!" << std::endl;
              }
              if (left_pt == hype::CPU ||
                  logical_operator->getLeft()
                      ->getPhysicalOperator()
                      ->hasAborted() ||
                  logical_operator->getRight()
                      ->getPhysicalOperator()
                      ->hasAborted()) {
                logical_operator->setDeviceConstraint(hype::CPU_ONLY);
              }
            }
            //                                   }else
            //                                   if(left_dev_constr==right_dev_constr
            //                                   &&
            //                                   logical_operator->dev_constr_==ANY_DEVICE){
            //                                       //continue chain (but
            //                                       continue chain on GPU only
            //                                       in case no operator
            //                                       aborted!)
            //                                       if(!
            //                                       (left_dev_constr==hype::GPU_ONLY
            //                                       && (
            //                                       logical_operator->getLeft()->getPhysicalOperator()->hasAborted()
            //                                         ||
            //                                         logical_operator->getRight()->getPhysicalOperator()->hasAborted()
            //                                          ) ) )
            //                                       {
            //                                           logical_operator->setDeviceConstraint(left_dev_constr);
            //                                           std::cout << "[QC]:
            //                                           Operator aborted, will
            //                                           not continue chain!" <<
            //                                           std::endl;
            //                                       }
            //                                   }
          }
        }
      } else {
        // unary operators
        if (logical_operator->getLeft()->getPhysicalOperator()) {
          ProcessingDeviceType left_pt = logical_operator->getLeft()
                                             ->getPhysicalOperator()
                                             ->getSchedulingDecision()
                                             .getDeviceSpecification()
                                             .getDeviceType();

          DeviceTypeConstraint left_dev_constr =
              util::getDeviceConstraintForProcessingDeviceType(left_pt);
          // we omit chain breakers in our placement heuristic,
          // and handle them differently than normal operators
          if (!util::isChainBreaker(logical_operator->getOperationName()) &&
              logical_operator->dev_constr_ == ANY_DEVICE) {
            // if the child is a "chain breaker" (e.g., a
            // management operator pinned to the CPU such
            // as Scans or rename operations),
            // we let HyPE decide where to begin the chain,
            // either on a CPU or a co-processor
            if (util::isChainBreaker(
                    logical_operator->getLeft()->getOperationName())) {
              // logical_operator->setDeviceConstraint(hype::ANY_DEVICE);
            } else {
              // else, continue the chain
              //                                       logical_operator->setDeviceConstraint(left_dev_constr);
              // else, continue chain (but continue chain on GPU only in case no
              // operator aborted!)

              // continue chain (but continue chain on GPU only in case no
              // operator aborted!)
              if ((left_pt == hype::GPU &&
                   !logical_operator->getLeft()
                        ->getPhysicalOperator()
                        ->hasAborted())) {
                logical_operator->setDeviceConstraint(
                    hype::GPU_ONLY);  // left_dev_constr);
                // std::cout << "[QC]: Operator aborted, will not continue
                // chain!" << std::endl;
              }
              if (left_pt == hype::CPU ||
                  logical_operator->getLeft()
                      ->getPhysicalOperator()
                      ->hasAborted()) {
                logical_operator->setDeviceConstraint(hype::CPU_ONLY);
              }
            }

            //                                       if(!
            //                                       (left_dev_constr==hype::GPU_ONLY
            //                                       &&
            //                                       logical_operator->getLeft()->getPhysicalOperator()->hasAborted()
            //                                       ) ){
            //                                           logical_operator->setDeviceConstraint(left_dev_constr);
            //                                           std::cout << "[QC]:
            //                                           Operator aborted, will
            //                                           not continue chain!" <<
            //                                           std::endl;
            //                                       }
          }
        }
      }

#endif

      if (hype::core::Runtime_Configuration::instance()
              .isPullBasedQueryChoppingEnabled()) {
        // queue operator in global operator stream
        core::Scheduler::instance().addIntoGlobalOperatorStream(
            logical_operator);
      } else {
        scheduleAndExecute(logical_operator, out);
      }
    }
  }
}

void Node::notify() {
  {
    boost::lock_guard<boost::mutex> lock(mutex_);
    this->is_ready_ = true;
  }
  condition_variable_.notify_one();
}

void Node::notify(NodePtr logical_operator) {
  {
    boost::lock_guard<boost::mutex> lock(mutex_);
    if (logical_operator == left_) {
      left_child_ready_ = true;
    } else if (logical_operator == right_) {
      right_child_ready_ = true;
    } else {
      HYPE_FATAL_ERROR(
          "Broken child/Parent relationship: Notification of a node that is "
          "not the parent of the notifying node!",
          std::cout);
    }
  }
  condition_variable_.notify_one();
}

const hype::core::Tuple Node::getFeatureVector() const {
  hype::core::Tuple t;
  if (this->left_) {  // if left child is valid (has to be by convention!), add
                      // input data size
    // if we already know the correct input data size, because the child node
    // was already executed
    // during query chopping, we use the real cardinality, other wise we call
    // the estimator
    if (this->left_->physical_operator_) {
      t.push_back(
          this->left_->physical_operator_->getResultSize());  // ->result_size_;
    } else {
      t.push_back(this->left_->getOutputResultSize());
    }
    if (this->right_) {  // if right child is valid (not null), add input data
                         // size for it as well
      if (this->right_->physical_operator_) {
        t.push_back(this->right_->physical_operator_
                        ->getResultSize());  // ->result_size_;
      } else {
        t.push_back(this->right_->getOutputResultSize());
      }
    }
  } else {
    HYPE_FATAL_ERROR("Invalid Left Child!", std::cout);
  }
  if (useSelectivityEstimation())
    t.push_back(this->getSelectivity());  // add selectivity of this operation,
                                          // when use_selectivity_estimation_ is
                                          // true
#ifdef HYPE_INCLUDE_INPUT_DATA_LOCALITY_IN_FEATURE_VECTOR
  // dirty workaround! We should make isInputDataCachedInGPU() a const member
  // function!
  t.push_back(const_cast<Node*>(this)->isInputDataCachedInGPU());
#endif
  return t;
}

bool Node::isInputDataCachedInGPU() {
  // bool ret=true;
  if (!left_ && !right_) return false;
  if (left_) {
    if (left_->physical_operator_ &&
        (left_->physical_operator_->getSchedulingDecision()
                 .getDeviceSpecification()
                 .getDeviceType() != hype::GPU ||
         left_->physical_operator_->hasAborted())) {
      return false;
    } else if (!right_ && left_->physical_operator_ &&
               left_->physical_operator_->getSchedulingDecision()
                       .getDeviceSpecification()
                       .getDeviceType() == hype::GPU &&
               !left_->physical_operator_->hasAborted()) {
      return true;
    }
    if (!right_) {
      if (left_->getDeviceConstraint().getDeviceTypeConstraint() ==
          hype::GPU_ONLY) {
        return true;
      }
    }
  }
  if (right_) {
    if (right_->physical_operator_ &&
        (right_->physical_operator_->getSchedulingDecision()
                 .getDeviceSpecification()
                 .getDeviceType() != hype::GPU ||
         right_->physical_operator_->hasAborted())) {
      return false;
    } else if (right_->physical_operator_ &&
               right_->physical_operator_->getSchedulingDecision()
                       .getDeviceSpecification()
                       .getDeviceType() == hype::GPU &&
               !right_->physical_operator_->hasAborted()) {
      if (left_ && left_->physical_operator_) {
        if (left_->physical_operator_->getSchedulingDecision()
                    .getDeviceSpecification()
                    .getDeviceType() == hype::GPU &&
            !left_->physical_operator_->hasAborted()) {
          return true;
        } else {
          return false;
        }
      } else {
        return true;
      }
    }
    if (left_->getDeviceConstraint().getDeviceTypeConstraint() ==
            hype::GPU_ONLY &&
        right_->getDeviceConstraint().getDeviceTypeConstraint() ==
            hype::GPU_ONLY) {
      return true;
    }
  }

  return false;
}

void Node::setOutputStream(std::ostream& output_stream) {
  this->out = &output_stream;
}

std::ostream& Node::getOutputStream() { return *out; }

#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB

void Node::produce_impl(CoGaDB::CodeGeneratorPtr code_gen,
                        CoGaDB::QueryContextPtr context) {
  HYPE_FATAL_ERROR(
      "Called unimplemented produce function in "
      "operator '"
          << this->getOperationName() << "'",
      std::cerr);
}

void Node::consume_impl(CoGaDB::CodeGeneratorPtr code_gen,
                        CoGaDB::QueryContextPtr context) {
  HYPE_FATAL_ERROR(
      "Called unimplemented consume function in "
      "operator '"
          << this->getOperationName() << "'",
      std::cerr);
}
#endif

#ifdef HYPE_PROFILE_PRODUCE_CONSUME
void Node::produce(CoGaDB::CodeGeneratorPtr code_gen,
                   CoGaDB::QueryContextPtr context) {
  uint64_t begin = core::getTimestamp();
  produce_impl(code_gen, context);
  uint64_t end = core::getTimestamp();
  (*out) << "Time Produce " << this->getOperationName() << ": "
         << double(end - begin) / (1000 * 1000 * 1000) << "s" << std::endl;
}

void Node::consume(CoGaDB::CodeGeneratorPtr code_gen,
                   CoGaDB::QueryContextPtr context) {
  uint64_t begin = core::getTimestamp();
  consume_impl(code_gen, context);
  uint64_t end = core::getTimestamp();
  (*out) << "Time Consume " << this->getOperationName() << ": "
         << double(end - begin) / (1000 * 1000 * 1000) << "s" << std::endl;
}
#else
void Node::produce(CoGaDB::CodeGeneratorPtr code_gen,
                   CoGaDB::QueryContextPtr context) {
  //  std::cout << "[PRODUCE]: " << toString(true) << std::endl;
  produce_impl(code_gen, context);
}

void Node::consume(CoGaDB::CodeGeneratorPtr code_gen,
                   CoGaDB::QueryContextPtr context) {
  //  std::cout << "[CONSUME]: " << toString(true) << std::endl;
  consume_impl(code_gen, context);
}
#endif

bool scheduleAndExecute(NodePtr logical_operator, std::ostream* out) {
  core::Tuple t = logical_operator->getFeatureVector();

  core::OperatorSpecification op_spec(
      logical_operator->getOperationName(), t,
      // parameters are the same, because in the query processing engine, we
      // model copy operations explicitely, so the copy cost have to be zero
      hype::PD_Memory_0,   // input data is in CPU RAM
      hype::PD_Memory_0);  // output data has to be stored in CPU RAM

  core::SchedulingDecision sched_dec =
      hype::core::Scheduler::instance().getOptimalAlgorithm(
          op_spec, logical_operator->getDeviceConstraint());
  OperatorPtr op = logical_operator->getPhysicalOperator(sched_dec);
  op->setLogicalOperator(logical_operator);
  op->setOutputStream(*out);
  return op->operator()();
}

}  // end namespace queryprocessing
}  // end namespace hype
