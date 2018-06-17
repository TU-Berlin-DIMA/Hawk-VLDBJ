
#include <core/scheduler.hpp>
#include <core/specification.hpp>
#include <list>
#include <query_optimization/qep.hpp>
#include <queue>
#include <sstream>
#include <util/algorithm_name_conversion.hpp>
#include <util/get_name.hpp>
#include <util/print.hpp>
#include <util/utility_functions.hpp>

namespace hype {
using namespace hype::core;
namespace query_optimization {

QEP_Node::QEP_Node(const OperatorSpecification& op_spec_arg,
                   const DeviceConstraint& dev_constr_arg,
                   bool input_data_cached_arg)
    : op_spec(op_spec_arg),
      dev_constr(dev_constr_arg),
      parent(NULL),
      childs(),
      number_childs(0),
      alg_ptr(NULL),
      cost(0),
      sched_dec_ptr(NULL),
      user_payload(),
      c_user_payload(NULL),
      input_data_cached(input_data_cached_arg) {}

QEP_Node::QEP_Node(const QEP_Node& other)
    : op_spec(other.op_spec),
      dev_constr(other.dev_constr),
      parent(NULL),
      childs(),
      number_childs(other.number_childs),
      alg_ptr(other.alg_ptr),  // we NEVER copy algorithm objects, just pointers
      cost(other.cost),
      sched_dec_ptr(NULL),
      user_payload(other.user_payload),
      c_user_payload(other.c_user_payload),
      input_data_cached(other.input_data_cached) {
  if (other.sched_dec_ptr) {
    this->sched_dec_ptr = new core::SchedulingDecision(*other.sched_dec_ptr);
  }

  for (unsigned int i = 0; i < other.number_childs; ++i) {
    // call copy constructor
    if (other.childs[i]) {
      childs[i] = new QEP_Node(*other.childs[i]);
      childs[i]->parent = this;
    }
  }
}

QEP_Node& QEP_Node::operator=(const QEP_Node& other) {
  // check for self assignment
  if (this != &other) {
    // cleanup old data
    cleanupChilds();
    if (this->sched_dec_ptr) delete this->sched_dec_ptr;
    // assign new data
    this->alg_ptr = other.alg_ptr;
    this->cost = other.cost;
    this->dev_constr = other.dev_constr;
    this->op_spec = other.op_spec;
    this->number_childs = other.number_childs;
    this->parent = other.parent;
    this->sched_dec_ptr = other.sched_dec_ptr;
    this->user_payload = other.user_payload;
    this->c_user_payload = other.c_user_payload;
    this->input_data_cached = other.input_data_cached;

    for (unsigned int i = 0; i < other.number_childs; ++i) {
      // call copy constructor
      childs[i] = new QEP_Node(*other.childs[i]);
      childs[i]->parent = this;
    }
  }
  return *this;
}

boost::mutex global_mutex_estimated_execution_time_of_algorithm;

void QEP_Node::assignAlgorithm(core::Algorithm* alg_ptr) {
  boost::lock_guard<boost::mutex> lock(
      global_mutex_estimated_execution_time_of_algorithm);
  this->alg_ptr = alg_ptr;
  // std::cout << "AssignAlg: " << alg_ptr->toString() << std::endl;
  this->cost = alg_ptr->getEstimatedExecutionTime(op_spec.getFeatureVector())
                   .getTimeinNanoseconds();
}

void QEP_Node::assignAlgorithm(core::Algorithm* alg_ptr,
                               const EstimatedTime est_exec_time) {
  this->alg_ptr = alg_ptr;
  // std::cout << "AssignAlg: " << alg_ptr->toString() << std::endl;
  this->cost = est_exec_time.getTimeinNanoseconds();
}

//                        void QEP_Node::assignAlgorithm(const
//                        OperatorSpecification& op_spec, const
//                        SchedulingDecision& sched_dec){
//
//                        }

bool QEP_Node::addChild(QEP_Node* child) {
  if (child) {
    if (this->number_childs < MAX_CHILDS) {
      this->childs[this->number_childs++] = child;
      return true;
    } else {
      HYPE_ERROR("Number of Maximal children exceeded!", std::cerr);
    }
  }
  return false;
}

bool QEP_Node::isChildOf(QEP_Node* node) {
  if (!node) return false;
  for (unsigned int i = 0; i < this->number_childs; ++i) {
    if (this == node->childs[i]) {
      return true;
    }
  }
  return false;
}

void QEP_Node::setChilds(QEP_Node** childs, size_t number_childs) {
  assert(number_childs <= MAX_CHILDS);
  cleanupChilds();
  for (unsigned int i = 0; i < number_childs; ++i) {
    this->childs[i] = childs[i];
    this->childs[i]->parent = this;
  }
  this->number_childs = number_childs;
}

void QEP_Node::setParent(QEP_Node* parent) { this->parent = parent; }

bool QEP_Node::isLeave() {
  if (number_childs == 0) {
    return true;
  } else {
    return false;
  }
}

bool QEP_Node::isRoot() {
  if (this->parent == NULL) {
    return true;
  } else {
    return false;
  }
}

unsigned int QEP_Node::getLevel() {
  unsigned int level = 0;
  hype::query_optimization::QEP_Node* current_ancestor = this;
  while (!current_ancestor->isRoot()) {
    ++level;
    current_ancestor = current_ancestor->parent;
  }
  return level;
}

size_t QEP_Node::getTotalNumberofChilds() {
  if (number_childs == 0) {
    return 0;
  } else {
    size_t num_childs = number_childs;
    for (unsigned int i = 0; i < number_childs; ++i) {
      num_childs += childs[i]->getTotalNumberofChilds();
    }
    return num_childs;
  }
}

double QEP_Node::computeCost() {
  if (number_childs == 0) {
    return this->cost;
  } else {
    double total_cost = this->cost;
    double max_cost_of_childs = std::numeric_limits<double>::min();
    for (unsigned int i = 0; i < number_childs; ++i) {
      max_cost_of_childs =
          std::max(max_cost_of_childs, childs[i]->computeCost());
    }
    return max_cost_of_childs + total_cost;
  }
}

double QEP_Node::computeTotalCost() {
  if (number_childs == 0) {
    return this->cost;
  } else {
    double total_cost = this->cost;
    for (unsigned int i = 0; i < number_childs; ++i) {
      total_cost += childs[i]->computeCost();
    }
    return total_cost;
  }
}

bool QEP_Node::isInputDataCached() const {
  if (this->input_data_cached) return true;
  // if the cached flag is not set for fetch joins, the index is missing,
  // so we the largest input data is missing, even if the perior filter
  // operators
  // were executed on GPU
  if (!this->input_data_cached &&
      (this->op_spec.getOperatorName() == "COLUMN_FETCH_JOIN" ||
       this->op_spec.getOperatorName() == "COLUMN_BITMAP_FETCH_JOIN" ||
       this->op_spec.getOperatorName() == "FETCH_JOIN")) {
    return false;
  }
  bool ret = true;

  for (unsigned int i = 0; i < number_childs; ++i) {
    // if(childs[i]->dev_constr==hype::GPU_ONLY) continue;
    if (childs[i]->sched_dec_ptr &&
        childs[i]->sched_dec_ptr->getDeviceSpecification().getDeviceType() !=
            hype::GPU) {
      return false;
    }
  }
  return ret;
}

std::list<QEP_Node*> QEP_Node::preorder_traversal() {
  std::list<QEP_Node*> ret;
  ret.push_back(this);
  for (unsigned int i = 0; i < this->number_childs; ++i) {
    if (this->childs[i]) {
      std::list<QEP_Node*> child_order = this->childs[i]->preorder_traversal();
      ret.insert(ret.end(), child_order.begin(), child_order.end());
    }
  }
  // ret.push_back(this);
  return ret;
}

std::list<QEP_Node*> QEP_Node::levelorder_traversal() {
  std::list<QEP_Node*> result;

  std::queue<QEP_Node*> queue;

  queue.push(this);
  // result.push_back(this);
  while (!queue.empty()) {
    QEP_Node* node = queue.front();
    result.push_back(node);
    for (unsigned int i = 0; i < node->number_childs; ++i) {
      if (node->childs[i]) queue.push(node->childs[i]);
    }
    // remove processed node from queue
    queue.pop();
  }
  return result;
}

std::list<QEP_Node*> QEP_Node::reverselevelorder_traversal() {
  std::list<QEP_Node*> result = levelorder_traversal();
  result.reverse();
  return result;
}

std::list<QEP_Node*> QEP_Node::postorder_traversal() {
  HYPE_FATAL_ERROR("Called unimplemented Function!", std::cerr);
  return std::list<QEP_Node*>();
}

QEP_Node::~QEP_Node() {
  // node that we do not delete algorithm pointers on purpose!
  // algorithms are explicitely managed by their operations, which are in turn
  // managed by the central scheduler instance!

  if (sched_dec_ptr) {
    core::Scheduler::instance().getProcessingDevices().removeSchedulingDecision(
        *sched_dec_ptr);
    delete sched_dec_ptr;
  }
  // However, we clean up our childs recursively:
  for (unsigned int i = 0; i < number_childs; ++i) {
    if (childs[i]) delete childs[i];
  }
}

std::string QEP_Node::toString() {
  std::stringstream ss;
  if (this->sched_dec_ptr) {
    ss << alg_ptr->getName() << "(" << op_spec.getFeatureVector()
       << "), Cost: " << this->cost / (1000 * 1000) << "ms";
    ss << " at device: "
       << (int)sched_dec_ptr->getDeviceSpecification().getProcessingDeviceID()
       << " ("
       << util::getName(sched_dec_ptr->getDeviceSpecification().getDeviceType())
       << ", "
       << "MemoryID: "
       << (int)sched_dec_ptr->getDeviceSpecification().getMemoryID() << ")";
    if (this->c_user_payload) {
      ss << " User Payload (C Interface): "
         << this->c_user_payload;  // << std::endl;
    }
  } else if (alg_ptr) {
    // ss << alg_ptr->getName() << ": " << this->cost;
    ss << alg_ptr->getName() << "(" << op_spec.getFeatureVector()
       << "): " << this->cost;
  } else {
    // ss << op_spec.getOperatorName();
    ss << op_spec.getOperatorName() << "(" << op_spec.getFeatureVector() << ")"
       << util::getName(this->dev_constr);
  }
  return ss.str();
}

void QEP_Node::cleanupChilds() {
  for (unsigned int i = 0; i < number_childs; ++i) {
    if (childs[i]) delete childs[i];
  }
}

/* ============================================================================
 */

QEP::QEP() : root_(NULL) {}

QEP::QEP(QEP_Node* root) : root_(root) {}

QEP::QEP(const QEP& other) : root_(new QEP_Node(*other.root_)) {}

QEP& QEP::operator=(const QEP& other) {
  // check for self assignment
  if (this != &other) {
    // cleanup root node, which will recursively cleanup childs
    if (root_) {
      delete root_;
    }
    this->root_ = new QEP_Node(*other.root_);
  }
  return *this;
}

double QEP::computeCost() {
  if (root_)
    return root_->computeCost();
  else
    return false;
}

double QEP::computeTotalCost() {
  if (root_)
    return root_->computeTotalCost();
  else
    return false;
}

// enum TextColor{WHITE,RED,GREEN,YELLOW,BLUE,MAGENTA,TURQUOISE,BLACK};

struct Print_QEP_Functor {
  Print_QEP_Functor(std::string& str, bool enable_colors_a = true,
                    bool print_node_numbers_a = false)
      : result_string(str),
        enable_colors(enable_colors_a),
        print_node_numbers(print_node_numbers_a),
        node_counter(0) {}

  Print_QEP_Functor(Print_QEP_Functor& other)
      : result_string(other.result_string),
        enable_colors(other.enable_colors),
        print_node_numbers(other.print_node_numbers),
        node_counter(other.node_counter) {}

  Print_QEP_Functor& operator=(Print_QEP_Functor& other) {
    if (this != &other) {
      result_string = other.result_string;
      enable_colors = other.enable_colors;
      print_node_numbers = other.print_node_numbers;
      node_counter = other.node_counter;
    }
    return *this;
  }

  void operator()(QEP_Node* node, unsigned int level) {
    std::stringstream ss;
    if (!result_string.empty()) ss << std::endl;
    if (print_node_numbers &&
        node->op_spec.getOperatorName() != "COPY_CPU_CP" &&
        node->op_spec.getOperatorName() != "COPY_CP_CPU") {
      if (node_counter < 10) ss << "0";
      ss << node_counter << ". ";
      node_counter++;
    }
    for (unsigned int i = 0; i < level; ++i) {
      ss << "\t";
    }
    if (enable_colors) {
      if (node->sched_dec_ptr) {
        if (node->sched_dec_ptr->getDeviceSpecification().getDeviceType() ==
            hype::CPU) {
          ss << "\033[44m";  // highlight following text in blue
        } else if (node->sched_dec_ptr->getDeviceSpecification()
                       .getDeviceType() == hype::GPU) {
          ss << "\033[42m";  // highlight following text in green
        }
      }
    }
    ss << node->toString();  //<< std::endl;
    if (enable_colors) {
      ss << "\033[0m";  // reset text attributes
    }
    // std::cout << ss.str();
    result_string += ss.str();

    // std::cout << "result_string: " <<  result_string << std::endl;
    // std::cout << "Current Line: " <<  ss.str() << std::endl;
  }

  std::string& result_string;
  bool enable_colors;
  bool print_node_numbers;
  unsigned int node_counter;
};

std::string QEP::toString(unsigned int indent, bool enable_colors,
                          bool print_node_numbers) {
  if (root_ == NULL) {
    return std::string();
  }
  std::string result;
  Print_QEP_Functor functor(result, enable_colors, print_node_numbers);
  root_->traverse_preorder(functor, 0);

  std::stringstream ss;
  ss << std::endl
     << "Estimated Cost: " << this->computeCost();  // << std::endl;

  result += ss.str();

  return result;  // functor.result_string;

  std::list<QEP_Node*> nodes = root_->preorder_traversal();

  //                        std::stringstream ss;
  //
  //
  //                        std::list<QEP_Node*> nodes =
  //                        root_->preorder_traversal();
  //
  //
  //
  //                        for(unsigned int i=0;i<indent;++i){
  //                            ss << "\t";
  //                        }
  //                        ss << root_->toString() << std::endl;
  //
  //                        QEP_Node** childs=root_->childs;
  //                        size_t number_childs=root_->number_childs;
  //
  //                        for(unsigned int i=0;i<number_childs;++i){
  ////                              if(childs[i]){
  ////                                  QEP* tmp = new QEP(childs[i]);
  ////                                  tmp->toString(indent+1);
  ////                                  delete tmp;
  ////                              }
  //                            if(childs[i]){
  //                               ss << QEP(new
  //                               QEP_Node(*childs[i])).toString(indent+1);
  //                            }
  //                        }
  //
  //
  //                        return ss.str();
}
std::list<QEP_Node*> QEP::preorder_traversal() {
  if (root_ == NULL) {
    return std::list<QEP_Node*>();
  }
  return root_->preorder_traversal();
}

std::list<QEP_Node*> QEP::levelorder_traversal() {
  if (root_ == NULL) {
    return std::list<QEP_Node*>();
  }
  return root_->levelorder_traversal();
}

std::list<QEP_Node*> QEP::reverselevelorder_traversal() {
  if (root_ == NULL) {
    return std::list<QEP_Node*>();
  }
  return root_->reverselevelorder_traversal();
}

std::vector<hype::query_optimization::QEP_Node*> QEP::getLeafNodes() {
  std::list<hype::query_optimization::QEP_Node*> order =
      this->reverselevelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;
  std::vector<hype::query_optimization::QEP_Node*> leaves;
  for (it = order.begin(); it != order.end(); ++it) {
    if ((*it)->isLeave()) {
      leaves.push_back(*it);
    }
  }
  return leaves;
}

size_t QEP::getNumberOfOperators() {
  if (root_ == NULL) {
    return 0;
  } else {
    return root_->getTotalNumberofChilds();
  }
}

QEP_Node* QEP::getRoot() { return this->root_; }

QEP_Node* QEP::removeRoot() {
  QEP_Node* ret_val = this->root_;
  this->root_ = NULL;
  return ret_val;
}

void QEP::setRoot(QEP_Node* new_root) { this->root_ = new_root; }

QEP::~QEP() {
  if (root_) delete root_;
}

/* ============================================================================
 */

void advanceAlgorithmIDs(std::vector<int>& current_algorithm_ids,
                         const std::vector<int>& number_of_algorithms,
                         int begin_index = 0) {
  assert(current_algorithm_ids.size() == number_of_algorithms.size());
  if (begin_index == current_algorithm_ids.size()) return;
  if (begin_index < current_algorithm_ids.size()) {
    if (current_algorithm_ids[begin_index] <
        number_of_algorithms[begin_index] - 1) {
      current_algorithm_ids[begin_index]++;
    } else {
      current_algorithm_ids[begin_index] = 0;
      advanceAlgorithmIDs(current_algorithm_ids, number_of_algorithms,
                          begin_index + 1);
    }
  }
}

hype::core::OperatorSequence createOperatorSequence(QEP& plan) {
  hype::core::OperatorSequence op_seq;
  std::list<hype::query_optimization::QEP_Node*> order =
      plan.levelorder_traversal();  // plan.levelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;
  for (it = order.begin(); it != order.end(); ++it) {
    std::cout << (*it)->op_spec.getOperatorName() << std::endl;
    op_seq.push_back(std::make_pair((*it)->op_spec, (*it)->dev_constr));
  }
  return op_seq;
}

std::vector<std::vector<hype::core::AlgorithmPtr> >
getAvailableAlgorithmsPerOperator(hype::core::OperatorSequence& op_seq) {
  std::vector<std::vector<hype::core::AlgorithmPtr> >
      available_algorithms_per_operator(op_seq.size());
  for (unsigned int i = 0; i < op_seq.size(); ++i) {
    // core::Scheduler::instance().get
    const core::Scheduler::MapNameToOperation& map_operationname_to_operation_ =
        hype::core::Scheduler::instance().getOperatorMap();
    core::Scheduler::MapNameToOperation::const_iterator it =
        map_operationname_to_operation_.find(op_seq[i].first.getOperatorName());
    if (it == map_operationname_to_operation_.end()) {
      // error, return NUll Pointer
      HYPE_FATAL_ERROR("INVALID OPERATIONNAME!", std::cout);
      return std::vector<std::vector<hype::core::AlgorithmPtr> >();
    }
    // fill the number of algorithms array
    const std::vector<hype::core::AlgorithmPtr> alg_ptrs =
        it->second->getAlgorithms();
    for (unsigned int j = 0; j < alg_ptrs.size(); ++j) {
      if (op_seq[i].second.getDeviceTypeConstraint() == hype::ANY_DEVICE ||
          (alg_ptrs[j]->getDeviceSpecification().getDeviceType() ==
           hype::CPU)  // always add the CPU algorithms to avoid errors for CPU
                       // Only operations
          || (op_seq[i].second.getDeviceTypeConstraint() == hype::GPU_ONLY &&
              alg_ptrs[j]->getDeviceSpecification().getDeviceType() ==
                  hype::GPU)) {
        // number_of_algorithms[i]++;
        available_algorithms_per_operator[i].push_back(alg_ptrs[j]);
      }
    }
  }
  return available_algorithms_per_operator;
}

void createPhysicalQueryPlan(
    QEP& current_plan,
    const std::vector<std::vector<hype::core::AlgorithmPtr> >&
        available_algorithms_per_operator,
    const std::vector<int>& current_algorithm_ids) {
  //                                QEP current_plan(plan);
  std::list<hype::query_optimization::QEP_Node*> order =
      current_plan
          .levelorder_traversal();  // current_plan.reverselevelorder_traversal();
                                    // //plan.levelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;
  unsigned int operator_index = 0;
  for (it = order.begin(); it != order.end(); ++it, ++operator_index) {
    if (!hype::quiet && hype::verbose && hype::debug)
      std::cout << (*it)->op_spec.getOperatorName() << " ("
                << (*it)->op_spec.getFeatureVector() << ")" << std::endl;
    // fetch algorithm
    hype::core::AlgorithmPtr alg_ptr = available_algorithms_per_operator
        [operator_index][current_algorithm_ids[operator_index]];
    (*it)->assignAlgorithm(alg_ptr.get());
    // at leave level
    if ((*it)->isLeave()) {
      if (!(*it)->isRoot()) {
        insertCopyOperatorInnerNode(*it);
      }
      insertCopyOperatorLeaveNode(*it);
      // at root
    } else if ((*it)->isRoot()) {
      insertCopyOperatorRootNode(current_plan, *it);
      // inside plan
    } else {
      insertCopyOperatorInnerNode(*it);
      if ((*it)->parent->op_spec.getMemoryLocation() !=
          (*it)->op_spec.getMemoryLocation()) {
      }
    }
  }
  // return current_plan;
}

void createSchedulingDecisionsforQEP(QEP& plan) {
  // construct scheduling decisions
  std::list<hype::query_optimization::QEP_Node*> order =
      plan.levelorder_traversal();  // current_plan.reverselevelorder_traversal();
                                    // //plan.levelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;
  unsigned int operator_index = 0;
  for (it = order.begin(); it != order.end(); ++it, ++operator_index) {
    (*it)->sched_dec_ptr = new core::SchedulingDecision(
        *((*it)->alg_ptr), core::EstimatedTime((*it)->cost),
        (*it)->op_spec.getFeatureVector());
  }
}

//                void optimizeQueryPlanGreedy(QEP& plan){
//                            QEP current_plan(plan);
//                            if(!hype::quiet && hype::verbose && hype::debug)
//                                std::cout << "Create Plan: " << std::endl;
//                            //create plan
//                            //for(unsigned int j=0;j<op_seq.size();++j){
//
//                                std::list<hype::query_optimization::QEP_Node*>
//                                order = current_plan.levelorder_traversal();
//                                //current_plan.reverselevelorder_traversal();
//                                //plan.levelorder_traversal();
//                                std::list<hype::query_optimization::QEP_Node*>::iterator
//                                it;
//                                unsigned int operator_index=0;
//                                for(it=order.begin();it!=order.end();++it,++operator_index){
//                                    if(!hype::quiet && hype::verbose &&
//                                    hype::debug)
//                                        std::cout <<
//                                        (*it)->op_spec.getOperatorName() <<  "
//                                        (" <<
//                                        (*it)->op_spec.getFeatureVector() <<
//                                        ")" << std::endl;
//                                    //fetch algorithm
//
//                                    SchedulingDecision sched_dec =
//                                    core::Scheduler::instance().getOptimalAlgorithm((*it)->op_spec,
//                                    (*it)->dev_constr);
//                                    (*it)->sched_dec_ptr=new
//                                    core::SchedulingDecision(sched_dec);
//                                    core::AlgorithmPtr alg_ptr =
//                                    core::Scheduler::instance().getAlgorithm(hype::util::toInternalAlgName(sched_dec.getNameofChoosenAlgorithm(),
//                                    sched_dec.getDeviceSpecification()));
//                                    assert(alg_ptr!=NULL);
//                                    //decision is load aware, if WTAR is used.
//                                    //However, the single cost of this
//                                    algorithm
//                                    //does not display the load, only this
//                                    algorithm's
//                                    //execution time
//                                    (*it)->assignAlgorithm(alg_ptr.get());
//                                    //hype::core::AlgorithmPtr alg_ptr =
//                                    available_algorithms_per_operator[operator_index][current_algorithm_ids[operator_index]];
//                                    //(*it)->assignAlgorithm(alg_ptr.get());
//                                    //at leave level
//                                    if((*it)->isLeave()){
//                                        if(!(*it)->isRoot()){
//                                            insertCopyOperatorInnerNode(*it);
//                                        }
//                                        insertCopyOperatorLeaveNode(*it);
//                                    //at root
//                                    }else if((*it)->isRoot()){
//                                        insertCopyOperatorRootNode(current_plan,*it);
//                                    //inside plan
//                                    }else{
//                                        insertCopyOperatorInnerNode(*it);
//                                        if((*it)->parent->op_spec.getMemoryLocation()!=(*it)->op_spec.getMemoryLocation()){
//
//                                        }
//                                    }
//                                 }
//
//                        //construct scheduling decisions for copy operations
//                       order = current_plan.levelorder_traversal();
//                       //current_plan.reverselevelorder_traversal();
//                       //plan.levelorder_traversal();
//                        for(it=order.begin();it!=order.end();++it){
//                            if((*it)->sched_dec_ptr==NULL){
//                                if((*it)->alg_ptr){
//                                    (*it)->sched_dec_ptr=new
//                                    core::SchedulingDecision(*((*it)->alg_ptr),
//                                    core::EstimatedTime((*it)->cost),
//                                    (*it)->op_spec.getFeatureVector());
//                                }else{
//                                    HYPE_FATAL_ERROR("In Greedy Strategy:
//                                    Found Node without assigned
//                                    algorithm!",std::cout);
//                                }
//                            }
//
//                        }
//
//
//                        //assign result plan to input reference
//                        plan=current_plan;
//                }

void optimizeQueryPlanGreedy(QEP& plan) {
  QEP current_plan(plan);
  if (!hype::quiet && hype::verbose && hype::debug)
    std::cout << "Create Plan: " << std::endl;
  // create plan
  // for(unsigned int j=0;j<op_seq.size();++j){

  std::list<hype::query_optimization::QEP_Node*> order =
      current_plan
          .reverselevelorder_traversal();  // plan.levelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;

  for (it = order.begin(); it != order.end(); ++it) {
    if (!hype::quiet && hype::verbose && hype::debug)
      std::cout << (*it)->op_spec.getOperatorName() << " ("
                << (*it)->op_spec.getFeatureVector() << ")" << std::endl;
    // fetch algorithm

    SchedulingDecision sched_dec =
        core::Scheduler::instance().getOptimalAlgorithm((*it)->op_spec,
                                                        (*it)->dev_constr);
    (*it)->sched_dec_ptr = new core::SchedulingDecision(sched_dec);
    core::AlgorithmPtr alg_ptr = core::Scheduler::instance().getAlgorithm(
        hype::util::toInternalAlgName(sched_dec.getNameofChoosenAlgorithm(),
                                      sched_dec.getDeviceSpecification()));
    assert(alg_ptr != NULL);
    // decision is load aware, if WTAR is used.
    // However, the single cost of this algorithm
    // does not display the load, only this algorithm's
    // execution time
    (*it)->assignAlgorithm(alg_ptr.get(),
                           sched_dec.getEstimatedExecutionTimeforAlgorithm());
  }

  for (it = order.begin(); it != order.end(); ++it) {
    // hype::core::AlgorithmPtr alg_ptr =
    // available_algorithms_per_operator[operator_index][current_algorithm_ids[operator_index]];
    //(*it)->assignAlgorithm(alg_ptr.get());
    // at leave level
    if ((*it)->isLeave()) {
      if (!(*it)->isRoot()) {
        insertCopyOperatorInnerNode(*it);
      }
      insertCopyOperatorLeaveNode(*it);
      // at root
    } else if ((*it)->isRoot()) {
      insertCopyOperatorRootNode(current_plan, *it);
      // inside plan
    } else {
      insertCopyOperatorInnerNode(*it);
      if ((*it)->parent->op_spec.getMemoryLocation() !=
          (*it)->op_spec.getMemoryLocation()) {
      }
    }
  }

  // construct scheduling decisions for copy operations
  order =
      current_plan
          .levelorder_traversal();  // current_plan.reverselevelorder_traversal();
                                    // //plan.levelorder_traversal();
  for (it = order.begin(); it != order.end(); ++it) {
    if ((*it)->sched_dec_ptr == NULL) {
      if ((*it)->alg_ptr) {
        (*it)->sched_dec_ptr = new core::SchedulingDecision(
            *((*it)->alg_ptr), core::EstimatedTime((*it)->cost),
            (*it)->op_spec.getFeatureVector());
      } else {
        HYPE_FATAL_ERROR(
            "In Greedy Strategy: Found Node without assigned algorithm!",
            std::cout);
      }
    }
  }

  // assign result plan to input reference
  plan = current_plan;
}

size_t countNumberOfConstraints(
    const std::vector<DeviceTypeConstraint>& dev_type_constraints,
    DeviceTypeConstraint constr) {
  size_t number_of_occurences = 0;
  for (unsigned int i = 0; i < dev_type_constraints.size(); ++i) {
    if (dev_type_constraints[i] == constr) number_of_occurences++;
  }
  return number_of_occurences;
}

void optimizeQueryPlanGreedyChainer(QEP& plan) {
  QEP current_plan(plan);
  if (!hype::quiet && hype::verbose && hype::debug)
    std::cout << "Create Plan: " << std::endl;
  // create plan
  // for(unsigned int j=0;j<op_seq.size();++j){

  std::list<hype::query_optimization::QEP_Node*> order =
      current_plan
          .reverselevelorder_traversal();  // plan.levelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;

  for (it = order.begin(); it != order.end(); ++it) {
    if (!hype::quiet && hype::verbose && hype::debug)
      std::cout << (*it)->op_spec.getOperatorName() << " ("
                << (*it)->op_spec.getFeatureVector() << ")" << std::endl;
    // fetch algorithm

    // DeviceTypeConstraint
    // dev_constr_of_op=(*it)->dev_constr.getDeviceTypeConstraint();
    // DeviceTypeConstraint  (*it)->dev_constr.getDeviceTypeConstraint()
    if (!util::isChainBreaker((*it)->op_spec.getOperatorName())) {
      std::vector<DeviceTypeConstraint> dev_type_constraints(
          (*it)->number_childs, ANY_DEVICE);
      bool at_least_one_child_is_chain_breaker = false;
      for (unsigned int i = 0; i < (*it)->number_childs; ++i) {
        if ((*it)->childs[i]) {
          if (!util::isChainBreaker(
                  (*it)->childs[i]->op_spec.getOperatorName())) {
            assert((*it)->childs[i]->sched_dec_ptr != NULL);
            dev_type_constraints[i] =
                util::getDeviceConstraintForProcessingDeviceType(
                    (*it)
                        ->childs[i]
                        ->sched_dec_ptr->getDeviceSpecification()
                        .getDeviceType());
          } else {
            dev_type_constraints[i] = CPU_ONLY;
            at_least_one_child_is_chain_breaker = true;
          }
        } else {
          dev_type_constraints[i] = ANY_DEVICE;
        }
      }
      size_t num_cpu_only =
          countNumberOfConstraints(dev_type_constraints, CPU_ONLY);
      size_t num_gpu_only =
          countNumberOfConstraints(dev_type_constraints, GPU_ONLY);
      if (at_least_one_child_is_chain_breaker) {
        //(*it)->dev_constr=hype::ANY_DEVICE;
      } else {
        // this optimizer is used by other optimizers, by providing hints
        // if we find a non cahin breaking operator, and the device constraint
        // is
        // not ANY_DEVICE, we found a hint and use it, otherwise, we set a
        // device constraint ourselves
        if ((*it)->dev_constr == hype::ANY_DEVICE) {
          // use GPU in case more than halve of child operators were executed on
          // GPU
          if (num_cpu_only > num_gpu_only) {
            (*it)->dev_constr = hype::CPU_ONLY;
          } else {
            (*it)->dev_constr = hype::GPU_ONLY;
          }

          // Workaround: for higher levels (closer to the leafs, prefer GPU)
          // the critical decision starts at level 3, whether to proceed the
          // comupation of the Bitmap Operations on the GPU or not
          // on lower levels, e.g. for TID List intersection after a selection.
          // we prefer GPUs
          //                                                if((*it)->getLevel()>3){
          //                                                    if(num_cpu_only>num_gpu_only){
          //                                                        (*it)->dev_constr=hype::CPU_ONLY;
          //                                                    }else{
          //                                                        (*it)->dev_constr=hype::GPU_ONLY;
          //                                                    }
          //                                                }
          if (num_cpu_only > num_gpu_only) {
            (*it)->dev_constr = hype::CPU_ONLY;
          } else if (num_cpu_only < num_gpu_only) {
            (*it)->dev_constr = hype::GPU_ONLY;
          }
        }
      }
    }

    SchedulingDecision sched_dec =
        core::Scheduler::instance().getOptimalAlgorithm((*it)->op_spec,
                                                        (*it)->dev_constr);
    (*it)->sched_dec_ptr = new core::SchedulingDecision(sched_dec);
    core::AlgorithmPtr alg_ptr = core::Scheduler::instance().getAlgorithm(
        hype::util::toInternalAlgName(sched_dec.getNameofChoosenAlgorithm(),
                                      sched_dec.getDeviceSpecification()));
    assert(alg_ptr != NULL);
    // decision is load aware, if WTAR is used.
    // However, the single cost of this algorithm
    // does not display the load, only this algorithm's
    // execution time
    (*it)->assignAlgorithm(alg_ptr.get(),
                           sched_dec.getEstimatedExecutionTimeforAlgorithm());
  }

  for (it = order.begin(); it != order.end(); ++it) {
    // hype::core::AlgorithmPtr alg_ptr =
    // available_algorithms_per_operator[operator_index][current_algorithm_ids[operator_index]];
    //(*it)->assignAlgorithm(alg_ptr.get());
    // at leave level
    if ((*it)->isLeave()) {
      if (!(*it)->isRoot()) {
        insertCopyOperatorInnerNode(*it);
      }
      insertCopyOperatorLeaveNode(*it);
      // at root
    } else if ((*it)->isRoot()) {
      insertCopyOperatorRootNode(current_plan, *it);
      // inside plan
    } else {
      insertCopyOperatorInnerNode(*it);
      if ((*it)->parent->op_spec.getMemoryLocation() !=
          (*it)->op_spec.getMemoryLocation()) {
      }
    }
  }

  // construct scheduling decisions for copy operations
  order =
      current_plan
          .levelorder_traversal();  // current_plan.reverselevelorder_traversal();
                                    // //plan.levelorder_traversal();
  for (it = order.begin(); it != order.end(); ++it) {
    if ((*it)->sched_dec_ptr == NULL) {
      if ((*it)->alg_ptr) {
        (*it)->sched_dec_ptr = new core::SchedulingDecision(
            *((*it)->alg_ptr), core::EstimatedTime((*it)->cost),
            (*it)->op_spec.getFeatureVector());
      } else {
        HYPE_FATAL_ERROR(
            "In Greedy Strategy: Found Node without assigned algorithm!",
            std::cout);
      }
    }
  }

  // assign result plan to input reference
  plan = current_plan;
}

void optimizeQueryPlanAccelerateCriticalPath(
    QEP& plan, unsigned int& leaf_number_of_new_path) {
  //                                std::list<hype::query_optimization::QEP_Node*>
  //                                order = plan.reverselevelorder_traversal();
  //                                //plan.levelorder_traversal();
  //                                std::list<hype::query_optimization::QEP_Node*>::iterator
  //                                it;

  std::vector<hype::query_optimization::QEP_Node*> leafs = plan.getLeafNodes();

  // unsigned int leaf_number_of_optimal_path=0;
  std::vector<QEPPtr> qep_candidates(leafs.size());
  if (!hype::quiet && hype::verbose && hype::debug)
    std::cout << "Number of Leafs: " << leafs.size() << std::endl;
  // for each leave node, create a plan where a path
  // is processed on a co-processor
  for (unsigned int i = 0; i < qep_candidates.size(); ++i) {
    // copy input plan
    qep_candidates[i] = QEPPtr(new QEP(plan));
    std::vector<hype::query_optimization::QEP_Node*> leaves =
        qep_candidates[i]->getLeafNodes();
    if (!hype::quiet && hype::verbose && hype::debug) {
      std::cout << "Input Plan for Leaf " << leaves[i]->toString() << "    "
                << (void*)leaves[i] << "(i=" << i << "):" << std::endl;
      std::cout << qep_candidates[i]->toString() << std::endl;
    }
    if (!leaves[i]->isRoot()) {  // &&
      // !util::isChainBreaker(leaves[i]->parent->op_spec.getOperatorName())){
      for (unsigned int j = 0; j < qep_candidates.size(); ++j) {
        // only set CPU_ONLY, if we found no other hint,
        //(other device constraint than ANY_DEVICE)
        if (i != j)
          if (leaves[j]->parent->dev_constr == hype::ANY_DEVICE)
            leaves[j]->parent->dev_constr = hype::CPU_ONLY;
      }
      // find first ancestor that allows us to
      // start an operator chain on the GPU
      hype::query_optimization::QEP_Node* current_ancestor = leaves[i]->parent;
      while (current_ancestor != NULL) {
        if (current_ancestor->dev_constr == hype::ANY_DEVICE &&
            !util::isChainBreaker(
                current_ancestor->op_spec.getOperatorName())) {
          current_ancestor->dev_constr = hype::GPU_ONLY;
          break;
        }
        current_ancestor = current_ancestor->parent;
      }
      // leaves[i]->parent->dev_constr=hype::GPU_ONLY;
    }
    if (!hype::quiet && hype::verbose && hype::debug) {
      std::cout << "Annotated Plan for Leaf " << leaves[i]->toString() << "    "
                << (void*)leaves[i] << "(i=" << i << "):" << std::endl;
      std::cout << qep_candidates[i]->toString() << std::endl;
      std::cout << "Plan for Leaf " << leaves[i]->toString() << "    "
                << (void*)leaves[i] << "(i=" << i << "):" << std::endl;
    }
    optimizeQueryPlanGreedyChainer(*qep_candidates[i]);
    if (!hype::quiet && hype::verbose && hype::debug) {
      std::cout << qep_candidates[i]->toString() << std::endl;
      std::cout << "========================================" << std::endl;
    }
  }
  QEPPtr lowest_cost_plan;
  double lowest_cost = std::numeric_limits<double>::max();
  for (unsigned int i = 0; i < qep_candidates.size(); ++i) {
    if (qep_candidates[i])
      if (lowest_cost > qep_candidates[i]->computeTotalCost()) {
        lowest_cost = qep_candidates[i]->computeTotalCost();
        lowest_cost_plan = qep_candidates[i];
        leaf_number_of_new_path = i;
      }
  }
  assert(lowest_cost_plan != NULL);
  plan = *lowest_cost_plan;
}

hype::query_optimization::QEP_Node* getParentFetchJoin(
    hype::query_optimization::QEP_Node* current_node) {
  hype::query_optimization::QEP_Node* ancestor = current_node->parent;
  while (ancestor != NULL &&
         ancestor->op_spec.getOperatorName() != "COLUMN_FETCH_JOIN" &&
         ancestor->op_spec.getOperatorName() != "FETCH_JOIN") {
    ancestor = ancestor->parent;
  }
  return ancestor;
}

void optimizeQueryPlanDataPlacementAwareCriticalPath(QEP& plan) {
  //                                QEP current_plan(plan);
  //                                std::vector<hype::query_optimization::QEP_Node*>
  //                                leafs = current_plan.getLeafNodes();
  //                                for(unsigned int i=0;i<leafs.size();++i){
  //                                    hype::query_optimization::QEP_Node*
  //                                    fetch_join =
  //                                    getParentFetchJoin(leafs[i]);
  //                                    if(fetch_join){
  //                                        if(fetch_join->input_data_cached){
  //                                            if(fetch_join->dev_constr==hype::ANY_DEVICE){
  //                                                fetch_join->dev_constr=hype::GPU_ONLY;
  //                                            }
  //                                        }else{
  //                                                fetch_join->dev_constr=hype::CPU_ONLY;
  //                                        }
  //                                    }
  //                                }

  QEP current_plan(plan);
  std::list<hype::query_optimization::QEP_Node*> list_nodes =
      current_plan.reverselevelorder_traversal();
  std::vector<hype::query_optimization::QEP_Node*> nodes(list_nodes.begin(),
                                                         list_nodes.end());

  for (unsigned int i = 0; i < nodes.size(); ++i) {
    // hype::query_optimization::QEP_Node* fetch_join =
    // getParentFetchJoin(nodes[i]);
    if (nodes[i]) {
      if (nodes[i]->isInputDataCached()) {
        if (nodes[i]->dev_constr == hype::ANY_DEVICE) {
          nodes[i]->dev_constr = hype::GPU_ONLY;
        }
      } else {
        nodes[i]->dev_constr = hype::CPU_ONLY;
      }
    }
  }

  optimizeQueryPlanGreedyChainer(current_plan);
  plan = current_plan;

  //                                std::vector<QEPPtr>
  //                                qep_candidates(leafs.size());
  //                                if(!hype::quiet && hype::verbose &&
  //                                hype::debug) std::cout << "Number of Leafs:
  //                                " << leafs.size() << std::endl;
  //                                //for each leave node, create a plan where a
  //                                path
  //                                //is processed on a co-processor
  //                                for(unsigned int
  //                                i=0;i<qep_candidates.size();++i){
  //                                    //copy input plan
  //                                    qep_candidates[i]=QEPPtr(new QEP(plan));
  //                                    std::vector<hype::query_optimization::QEP_Node*>
  //                                    leaves  =
  //                                    qep_candidates[i]->getLeafNodes();
  //                                    //if(!hype::quiet && hype::verbose &&
  //                                    hype::debug)
  //                                    {
  //                                        std::cout << "Input Plan for Leaf "
  //                                        << leaves[i]->toString() << "    "
  //                                        << (void*)leaves[i] << "(i="<< i <<
  //                                        "):" << std::endl;
  //                                        std::cout <<
  //                                        qep_candidates[i]->toString() <<
  //                                        std::endl;
  //                                    }
  //                                    if(!leaves[i]->isRoot()){ // &&
  //                                    !util::isChainBreaker(leaves[i]->parent->op_spec.getOperatorName())){
  //                                        for(unsigned int
  //                                        j=0;j<qep_candidates.size();++j){
  //
  //                                            //only set CPU_ONLY, if we found
  //                                            no other hint,
  //                                            //(other device constraint than
  //                                            ANY_DEVICE)
  //                                            if(i!=j)
  //                                            if(leaves[j]->parent->dev_constr==hype::ANY_DEVICE)
  //                                                 leaves[j]->parent->dev_constr=hype::CPU_ONLY;
  //                                        }
  //                                        //find first ancestor that allows us
  //                                        to
  //                                        //start an operator chain on the GPU
  //                                        hype::query_optimization::QEP_Node*
  //                                        current_ancestor=leaves[i]->parent;
  //                                        while(current_ancestor!=NULL){
  //                                            if(current_ancestor->dev_constr==hype::ANY_DEVICE){
  //                                                current_ancestor->dev_constr=hype::GPU_ONLY;
  //                                                break;
  //                                            }
  //                                            current_ancestor=current_ancestor->parent;
  //                                        }
  //                                        //leaves[i]->parent->dev_constr=hype::GPU_ONLY;
  //                                    }
  //                                    //if(!hype::quiet && hype::verbose &&
  //                                    hype::debug)
  //                                    {
  //                                        std::cout << "Annotated Plan for
  //                                        Leaf " << leaves[i]->toString() << "
  //                                        " << (void*)leaves[i] << "(i="<< i
  //                                        <<  "):" << std::endl;
  //                                        std::cout <<
  //                                        qep_candidates[i]->toString() <<
  //                                        std::endl;
  //                                        std::cout << "Plan for Leaf " <<
  //                                        leaves[i]->toString() << "    " <<
  //                                        (void*)leaves[i] << "(i="<< i <<
  //                                        "):" << std::endl;
  //                                    }
  //                                    optimizeQueryPlanGreedyChainer(*qep_candidates[i]);
  //                                    //if(!hype::quiet && hype::verbose &&
  //                                    hype::debug)
  //                                    {
  //                                        std::cout <<
  //                                        qep_candidates[i]->toString() <<
  //                                        std::endl;
  //                                        std::cout <<
  //                                        "========================================"
  //                                        << std::endl;
  //                                    }
  //                                }
  //                                QEPPtr lowest_cost_plan;
  //                                double
  //                                lowest_cost=std::numeric_limits<double>::max();
  //                                for(unsigned int
  //                                i=0;i<qep_candidates.size();++i){
  //                                    if(qep_candidates[i])
  //                                    if(lowest_cost>qep_candidates[i]->computeCost()){
  //                                        lowest_cost=qep_candidates[i]->computeCost();
  //                                        lowest_cost_plan=qep_candidates[i];
  //                                        leaf_number_of_new_path=i;
  //                                    }
  //                                }
  //                                assert(lowest_cost_plan!=NULL);
  //                                plan=*lowest_cost_plan;
}

void optimizeQueryPlanDataPlacementAware2(QEP& plan) {
  QEP current_plan(plan);
  if (!hype::quiet && hype::verbose && hype::debug)
    std::cout << "Create Plan: " << std::endl;
  // create plan
  // for(unsigned int j=0;j<op_seq.size();++j){

  std::list<hype::query_optimization::QEP_Node*> order =
      current_plan
          .reverselevelorder_traversal();  // plan.levelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;

  for (it = order.begin(); it != order.end(); ++it) {
    if (!hype::quiet && hype::verbose && hype::debug)
      std::cout << (*it)->op_spec.getOperatorName() << " ("
                << (*it)->op_spec.getFeatureVector() << ")" << std::endl;
    // fetch algorithm
    if ((*it)->isInputDataCached() && (*it)->dev_constr != hype::CPU_ONLY)
      (*it)->dev_constr = hype::GPU_ONLY;
    else
      (*it)->dev_constr = hype::CPU_ONLY;

    SchedulingDecision sched_dec =
        core::Scheduler::instance().getOptimalAlgorithm((*it)->op_spec,
                                                        (*it)->dev_constr);
    (*it)->sched_dec_ptr = new core::SchedulingDecision(sched_dec);
    core::AlgorithmPtr alg_ptr = core::Scheduler::instance().getAlgorithm(
        hype::util::toInternalAlgName(sched_dec.getNameofChoosenAlgorithm(),
                                      sched_dec.getDeviceSpecification()));
    assert(alg_ptr != NULL);
    // decision is load aware, if WTAR is used.
    // However, the single cost of this algorithm
    // does not display the load, only this algorithm's
    // execution time
    (*it)->assignAlgorithm(alg_ptr.get(),
                           sched_dec.getEstimatedExecutionTimeforAlgorithm());
  }

  for (it = order.begin(); it != order.end(); ++it) {
    // hype::core::AlgorithmPtr alg_ptr =
    // available_algorithms_per_operator[operator_index][current_algorithm_ids[operator_index]];
    //(*it)->assignAlgorithm(alg_ptr.get());
    // at leave level
    if ((*it)->isLeave()) {
      if (!(*it)->isRoot()) {
        insertCopyOperatorInnerNode(*it);
      }
      insertCopyOperatorLeaveNode(*it);
      // at root
    } else if ((*it)->isRoot()) {
      insertCopyOperatorRootNode(current_plan, *it);
      // inside plan
    } else {
      insertCopyOperatorInnerNode(*it);
      if ((*it)->parent->op_spec.getMemoryLocation() !=
          (*it)->op_spec.getMemoryLocation()) {
      }
    }
  }

  // construct scheduling decisions for copy operations
  order =
      current_plan
          .levelorder_traversal();  // current_plan.reverselevelorder_traversal();
                                    // //plan.levelorder_traversal();
  for (it = order.begin(); it != order.end(); ++it) {
    if ((*it)->sched_dec_ptr == NULL) {
      if ((*it)->alg_ptr) {
        (*it)->sched_dec_ptr = new core::SchedulingDecision(
            *((*it)->alg_ptr), core::EstimatedTime((*it)->cost),
            (*it)->op_spec.getFeatureVector());
      } else {
        HYPE_FATAL_ERROR(
            "In Greedy Strategy: Found Node without assigned algorithm!",
            std::cout);
      }
    }
  }

  // assign result plan to input reference
  plan = current_plan;
}

//                void
//                optimizeQueryPlanAccelerateCriticalPathDataPlacementAware(QEP&
//                plan, unsigned int& leaf_number_of_new_path){
//
////
/// std::list<hype::query_optimization::QEP_Node*> order =
/// plan.reverselevelorder_traversal(); //plan.levelorder_traversal();
//// std::list<hype::query_optimization::QEP_Node*>::iterator it;
//
//                                std::vector<hype::query_optimization::QEP_Node*>
//                                leafs = plan.getLeafNodes();
//
//                                //unsigned int leaf_number_of_optimal_path=0;
//                                std::vector<QEPPtr>
//                                qep_candidates(leafs.size());
//                                if(!hype::quiet && hype::verbose &&
//                                hype::debug) std::cout << "Number of Leafs: "
//                                << leafs.size() << std::endl;
//                                //for each leave node, create a plan where a
//                                path
//                                //is processed on a co-processor
//                                for(unsigned int
//                                i=0;i<qep_candidates.size();++i){
//                                    //copy input plan
//                                    qep_candidates[i]=QEPPtr(new QEP(plan));
//                                    std::vector<hype::query_optimization::QEP_Node*>
//                                    leaves  =
//                                    qep_candidates[i]->getLeafNodes();
//                                    //if(!hype::quiet && hype::verbose &&
//                                    hype::debug)
//                                    {
//                                        std::cout << "Input Plan for Leaf " <<
//                                        leaves[i]->toString() << "    " <<
//                                        (void*)leaves[i] << "(i="<< i <<  "):"
//                                        << std::endl;
//                                        std::cout <<
//                                        qep_candidates[i]->toString() <<
//                                        std::endl;
//                                    }
//                                    if(!leaves[i]->isRoot()){ // &&
//                                    !util::isChainBreaker(leaves[i]->parent->op_spec.getOperatorName())){
//                                        for(unsigned int
//                                        j=0;j<qep_candidates.size();++j){
//
//                                            //only set CPU_ONLY, if we found
//                                            no other hint,
//                                            //(other device constraint than
//                                            ANY_DEVICE)
//                                            if(i!=j)
//                                            if(leaves[j]->parent->dev_constr==hype::ANY_DEVICE)
//                                                 leaves[j]->parent->dev_constr=hype::CPU_ONLY;
//                                        }
//                                        //find first ancestor that allows us
//                                        to
//                                        //start an operator chain on the GPU
//                                        hype::query_optimization::QEP_Node*
//                                        current_ancestor=leaves[i]->parent;
//                                        while(current_ancestor!=NULL){
//                                            if(current_ancestor->dev_constr==hype::ANY_DEVICE){
//                                                current_ancestor->dev_constr=hype::GPU_ONLY;
//                                                break;
//                                            }
//
//                                            current_ancestor=current_ancestor->parent;
//                                        }
//                                        //leaves[i]->parent->dev_constr=hype::GPU_ONLY;
//                                    }
//                                    //if(!hype::quiet && hype::verbose &&
//                                    hype::debug)
//                                    {
//                                        std::cout << "Annotated Plan for Leaf
//                                        " << leaves[i]->toString() << "    "
//                                        << (void*)leaves[i] << "(i="<< i <<
//                                        "):" << std::endl;
//                                        std::cout <<
//                                        qep_candidates[i]->toString() <<
//                                        std::endl;
//                                        std::cout << "Plan for Leaf " <<
//                                        leaves[i]->toString() << "    " <<
//                                        (void*)leaves[i] << "(i="<< i <<  "):"
//                                        << std::endl;
//                                    }
//                                    optimizeQueryPlanGreedyChainer(*qep_candidates[i]);
//                                    //if(!hype::quiet && hype::verbose &&
//                                    hype::debug)
//                                    {
//                                        std::cout <<
//                                        qep_candidates[i]->toString() <<
//                                        std::endl;
//                                        std::cout <<
//                                        "========================================"
//                                        << std::endl;
//                                    }
//                                }
//                                QEPPtr lowest_cost_plan;
//                                double
//                                lowest_cost=std::numeric_limits<double>::max();
//                                for(unsigned int
//                                i=0;i<qep_candidates.size();++i){
//                                    if(qep_candidates[i])
//                                    if(lowest_cost>qep_candidates[i]->computeCost()){
//                                        lowest_cost=qep_candidates[i]->computeCost();
//                                        lowest_cost_plan=qep_candidates[i];
//                                        leaf_number_of_new_path=i;
//                                    }
//                                }
//                                assert(lowest_cost_plan!=NULL);
//                                plan=*lowest_cost_plan;
//                }

void optimizeQueryPlanGPUBestEffort(QEP& plan) {
  QEP current_plan(plan);
  std::list<hype::query_optimization::QEP_Node*> list_nodes =
      current_plan.reverselevelorder_traversal();
  std::vector<hype::query_optimization::QEP_Node*> nodes(list_nodes.begin(),
                                                         list_nodes.end());

  for (unsigned int i = 0; i < nodes.size(); ++i) {
    // hype::query_optimization::QEP_Node* fetch_join =
    // getParentFetchJoin(nodes[i]);
    if (nodes[i]) {
      if (nodes[i]->dev_constr == hype::ANY_DEVICE) {
        nodes[i]->dev_constr = hype::GPU_ONLY;
      } else {
        // nodes[i]->dev_constr=hype::CPU_ONLY;
      }
    }
  }

  // optimizeQueryPlanGreedyChainer(current_plan);
  optimizeQueryPlanGreedy(current_plan);
  plan = current_plan;
}

void optimizeQueryPlanAccelerateCriticalPathRecursive(QEP& plan) {
  size_t max_number_of_gpu_accelerated_paths = 5;

  size_t num_iterations =
      std::min(plan.getLeafNodes().size(), max_number_of_gpu_accelerated_paths);
  QEPPtr best_qep_candidate;
  double best_qep_cost = std::numeric_limits<double>::max();
  std::vector<unsigned int> gpu_path_leaves;

  std::cout << "Recursive Critical Path Optimizer: " << std::endl;
  // create CPU ONLY plan
  QEPPtr cpu_only_qep(new QEP(plan));
  {
    std::list<hype::query_optimization::QEP_Node*> order =
        cpu_only_qep
            ->reverselevelorder_traversal();  // plan.levelorder_traversal();
    std::list<hype::query_optimization::QEP_Node*>::iterator it;

    for (it = order.begin(); it != order.end(); ++it) {
      if ((*it)->dev_constr == hype::ANY_DEVICE)
        (*it)->dev_constr = hype::CPU_ONLY;
    }
    optimizeQueryPlanGreedy(*cpu_only_qep);
  }

  //                        std::cout << cpu_only_qep->toString() << std::endl;

  // create hybrid plans
  for (unsigned int i = 0; i < num_iterations; ++i) {
    QEPPtr current_qep(new QEP(plan));
    // set device constraints of beneficial paths
    std::vector<hype::query_optimization::QEP_Node*> leafs =
        current_qep->getLeafNodes();
    //                        for(unsigned int
    //                        j=0;j<gpu_path_leaves.size();++j){
    //                            leafs[j]->dev_constr=hype::CPU_ONLY;
    //                        }
    for (unsigned int j = 0; j < gpu_path_leaves.size(); ++j) {
      //                            //only override constrained no other hint is
      //                            present
      //                            if(leafs[gpu_path_leaves[j]]->parent->dev_constr==hype::ANY_DEVICE)
      //                                leafs[gpu_path_leaves[j]]->parent->dev_constr=hype::GPU_ONLY;
      // only override constrains if no other hint is present, otherwise,
      // traverse the path to the root and look for a possible
      // starting point of an operator chain
      hype::query_optimization::QEP_Node* current_ancestor =
          leafs[gpu_path_leaves[j]]->parent;
      while (current_ancestor != NULL) {
        if (current_ancestor->dev_constr == hype::ANY_DEVICE) {
          current_ancestor->dev_constr = hype::GPU_ONLY;
          break;
        }

        current_ancestor = current_ancestor->parent;
      }
    }
    unsigned int leaf_number_of_new_path = 0;
    optimizeQueryPlanAccelerateCriticalPath(*current_qep,
                                            leaf_number_of_new_path);

    double cost =
        current_qep->computeTotalCost();  // current_qep->computeCost();
    if (!hype::quiet && hype::verbose && hype::debug) {
      std::cout << "Iteration: " << i << std::endl;
      std::cout << current_qep->toString() << std::endl;
      std::cout << "========================================" << std::endl;
    }
    if (best_qep_cost > cost) {
      best_qep_cost = cost;
      best_qep_candidate = current_qep;
      gpu_path_leaves.push_back(leaf_number_of_new_path);
      //                        }else{
      ////                            if(!hype::quiet && hype::verbose &&
      /// hype::debug)
      //                                std::cout << "Found no better plan in
      //                                iteration " << i << "! Returning
      //                                plan..." << std::endl;
      //                            break;
    }
  }
  assert(best_qep_candidate != NULL);
  if (cpu_only_qep->computeTotalCost() >=
      best_qep_candidate->computeTotalCost()) {
    plan = *best_qep_candidate;
  } else {
    plan = *cpu_only_qep;
  }
}

void optimizeQueryPlanAccelerateCriticalPath_old(QEP& plan) {
  QEP current_plan(plan);
  if (!hype::quiet && hype::verbose && hype::debug)
    std::cout << "Create Plan: " << std::endl;
  // create plan
  // for(unsigned int j=0;j<op_seq.size();++j){

  std::list<hype::query_optimization::QEP_Node*> order =
      current_plan
          .reverselevelorder_traversal();  // plan.levelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;

  // first, create a pure CPU plan
  for (it = order.begin(); it != order.end(); ++it) {
    if (!hype::quiet && hype::verbose && hype::debug)
      std::cout << (*it)->op_spec.getOperatorName() << " ("
                << (*it)->op_spec.getFeatureVector() << ")" << std::endl;
    // fetch algorithm

    SchedulingDecision sched_dec =
        core::Scheduler::instance().getOptimalAlgorithm(
            (*it)->op_spec, hype::CPU_ONLY);  //(*it)->dev_constr);
    (*it)->sched_dec_ptr = new core::SchedulingDecision(sched_dec);
    core::AlgorithmPtr alg_ptr = core::Scheduler::instance().getAlgorithm(
        hype::util::toInternalAlgName(sched_dec.getNameofChoosenAlgorithm(),
                                      sched_dec.getDeviceSpecification()));
    assert(alg_ptr != NULL);
    // decision is load aware, if WTAR is used.
    // However, the single cost of this algorithm
    // does not display the load, only this algorithm's
    // execution time
    (*it)->assignAlgorithm(alg_ptr.get(),
                           sched_dec.getEstimatedExecutionTimeforAlgorithm());
  }

  // TODO: traverse plan, identify critical path, and assign the critical path
  // GPU operators

  // insert copy operations
  for (it = order.begin(); it != order.end(); ++it) {
    // hype::core::AlgorithmPtr alg_ptr =
    // available_algorithms_per_operator[operator_index][current_algorithm_ids[operator_index]];
    //(*it)->assignAlgorithm(alg_ptr.get());
    // at leave level
    if ((*it)->isLeave()) {
      if (!(*it)->isRoot()) {
        insertCopyOperatorInnerNode(*it);
      }
      insertCopyOperatorLeaveNode(*it);
      // at root
    } else if ((*it)->isRoot()) {
      insertCopyOperatorRootNode(current_plan, *it);
      // inside plan
    } else {
      insertCopyOperatorInnerNode(*it);
      if ((*it)->parent->op_spec.getMemoryLocation() !=
          (*it)->op_spec.getMemoryLocation()) {
      }
    }
  }

  // construct scheduling decisions for copy operations
  order =
      current_plan
          .levelorder_traversal();  // current_plan.reverselevelorder_traversal();
                                    // //plan.levelorder_traversal();
  for (it = order.begin(); it != order.end(); ++it) {
    if ((*it)->sched_dec_ptr == NULL) {
      if ((*it)->alg_ptr) {
        (*it)->sched_dec_ptr = new core::SchedulingDecision(
            *((*it)->alg_ptr), core::EstimatedTime((*it)->cost),
            (*it)->op_spec.getFeatureVector());
      } else {
        HYPE_FATAL_ERROR(
            "In Greedy Strategy: Found Node without assigned algorithm!",
            std::cout);
      }
    }
  }

  // assign result plan to input reference
  plan = current_plan;
}

void optimizeQueryPlanInteractive_old(QEP& plan) {
  QEP current_plan(plan);

  optimizeQueryPlanGreedy(current_plan);

  while (true) {
    std::cout << current_plan.toString(0, true, true) << std::endl;

    std::list<hype::query_optimization::QEP_Node*> order =
        current_plan.preorder_traversal();

    std::vector<hype::query_optimization::QEP_Node*> indexed_order(
        order.begin(), order.end());

    std::string user_input;
    unsigned int node_number = 0;
    std::cout << "Which Node do you which to update?" << std::endl;
    std::cin >> user_input;
    if (user_input == "quit") {
      break;
    }
    node_number = boost::lexical_cast<unsigned int>(user_input);

    assert(node_number < indexed_order.size());
    hype::query_optimization::QEP_Node* node = indexed_order[node_number];

    const core::Scheduler::MapNameToOperation& map_operationname_to_operation_ =
        hype::core::Scheduler::instance().getOperatorMap();
    core::Scheduler::MapNameToOperation::const_iterator it =
        map_operationname_to_operation_.find(node->op_spec.getOperatorName());
    if (it == map_operationname_to_operation_.end()) {
      // error, return NUll Pointer
      HYPE_FATAL_ERROR("INVALID OPERATIONNAME!", std::cout);
      return;
    }
    std::cout << "Available Operators: " << std::endl;
    const std::vector<hype::core::AlgorithmPtr> alg_ptrs =
        it->second->getAlgorithms();
    for (unsigned int i = 0; i < alg_ptrs.size(); ++i) {
      std::cout << i << ". "
                << util::getName(
                       alg_ptrs[i]->getDeviceSpecification().getDeviceType())
                << alg_ptrs[i]->getName() << std::endl;
    }

    std::cout << "Choose Algorithm for Operator " << node->toString() << ":"
              << std::endl;
    unsigned int algorithm_id = 0;
    std::cin >> user_input;
    if (user_input == "quit") {
      break;
    }
    algorithm_id = boost::lexical_cast<unsigned int>(user_input);
    assert(algorithm_id < alg_ptrs.size());

    node->assignAlgorithm(alg_ptrs[algorithm_id].get());

    // TODO: FIXME: Scheduling Decision has to be removed from book keeping
    // mechanism
    if (node->sched_dec_ptr) delete node->sched_dec_ptr;
    node->sched_dec_ptr = new core::SchedulingDecision(
        *(node->alg_ptr), core::EstimatedTime(node->cost),
        node->op_spec.getFeatureVector());

    // at leave level
    if (node->isLeave()) {
      if (!node->isRoot()) {
        insertCopyOperatorInnerNode(node);
      }
      insertCopyOperatorLeaveNode(node);
      // at root
    } else if (node->isRoot()) {
      insertCopyOperatorRootNode(current_plan, node);
      // inside plan
    } else {
      insertCopyOperatorInnerNode(node);
      if (node->parent->op_spec.getMemoryLocation() !=
          node->op_spec.getMemoryLocation()) {
      }
    }
  }
}

void optimizeQueryPlanInteractive(QEP& plan) {
  // size_t num_op = plan.getNumberOfOperators();

  QEP optimal_plan;
  double total_cost_optimal_plan = std::numeric_limits<double>::max();

  // QEP current_plan(plan);

  hype::core::OperatorSequence op_seq = createOperatorSequence(plan);

  if (!hype::quiet && hype::verbose && hype::debug) {
    std::cout << "Serialized Plan: " << std::endl;
    util::print(op_seq, std::cout);
  }
  // stores for each operators the number of possible algorithms to choose from
  std::vector<std::vector<hype::core::AlgorithmPtr> >
      available_algorithms_per_operator =
          getAvailableAlgorithmsPerOperator(op_seq);

  // init array for number of algorithms per operator
  std::vector<int> number_of_algorithms(op_seq.size());
  for (unsigned int i = 0; i < op_seq.size(); ++i) {
    number_of_algorithms[i] = available_algorithms_per_operator[i]
                                  .size();  // number_of_algorithms[i];
  }

  std::list<hype::query_optimization::QEP_Node*> level_order =
      plan.levelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*> pre_order =
      plan.preorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;
  std::list<hype::query_optimization::QEP_Node*>::iterator inner_it;
  std::vector<unsigned int> operator_id_mapping_array(op_seq.size());
  unsigned int operator_index = 0;
  for (it = level_order.begin(); it != level_order.end();
       ++it, ++operator_index) {
    unsigned int inner_operator_index = 0;
    for (inner_it = pre_order.begin(); inner_it != pre_order.end();
         ++inner_it, ++inner_operator_index) {
      if (*it == *inner_it) {
        operator_id_mapping_array[inner_operator_index] = operator_index;
        //                                    operator_id_mapping_array[operator_index]=inner_operator_index;
        std::cout << "LevelOrder: " << (*it)->toString() << " (index "
                  << operator_index << ")" << std::endl
                  << "PreOrder: " << (*inner_it)->toString() << " (index "
                  << inner_operator_index << ")" << std::endl
                  << "Map Array: " << inner_operator_index << "->"
                  << operator_index << std::endl;
      }
    }
  }

  //                        std::cout << "Level Order: " << std::endl;
  //                        for(it=level_order.begin();it!=level_order.end();++it){
  //                            std::cout << (*it)->toString() << std::endl;
  //                        }
  //                        std::cout << "Pre Order: " << std::endl;
  //                        for(it=pre_order.begin();it!=pre_order.end();++it){
  //                            std::cout << (*it)->toString() << std::endl;
  //                        }

  std::vector<int> current_algorithm_ids(op_seq.size());

  // std::cout << "Backtracking exploring " << number_of_plans << " plans" <<
  // std::endl;
  while (true) {
    QEP current_plan(plan);
    if (!hype::quiet && hype::verbose && hype::debug)
      std::cout << "Create Plan: " << std::endl;
    // create plan
    // for(unsigned int j=0;j<op_seq.size();++j){
    createPhysicalQueryPlan(current_plan, available_algorithms_per_operator,
                            current_algorithm_ids);

    createSchedulingDecisionsforQEP(current_plan);

    std::cout << current_plan.toString(0, true, true) << std::endl;

    optimal_plan = current_plan;

    std::string user_input;
    unsigned int node_number = 0;
    std::cout << "Which Node do you which to update?" << std::endl;
    std::cin >> user_input;
    if (user_input == "quit") {
      break;
    }
    node_number = boost::lexical_cast<unsigned int>(user_input);

    std::cout << "transformed id: " << operator_id_mapping_array[node_number]
              << std::endl;
    node_number = operator_id_mapping_array[node_number];
    assert(node_number < op_seq.size());
    //                    hype::query_optimization::QEP_Node* node =
    //                    indexed_order[node_number];
    //
    //
    //                    const core::Scheduler::MapNameToOperation&
    //                    map_operationname_to_operation_=hype::core::Scheduler::instance().getOperatorMap();
    //                    core::Scheduler::MapNameToOperation::const_iterator it
    //                    =
    //                    map_operationname_to_operation_.find(op_seq[node_number].first.getOperatorName());
    //                    if(it==map_operationname_to_operation_.end()){
    //                        //error, return NUll Pointer
    //                        HYPE_FATAL_ERROR("INVALID
    //                        OPERATIONNAME!",std::cout);
    //                        return;
    //                    }
    //                    std::cout << "Available Operators: " << std::endl;

    const std::vector<hype::core::AlgorithmPtr> alg_ptrs =
        available_algorithms_per_operator
            [node_number];  // it->second->getAlgorithms();
    for (unsigned int i = 0; i < alg_ptrs.size(); ++i) {
      std::cout << i << ". "
                << util::getName(
                       alg_ptrs[i]->getDeviceSpecification().getDeviceType())
                << alg_ptrs[i]->getName() << std::endl;
    }

    std::cout << "Choose Algorithm for Operator "
              << op_seq[node_number].first.getOperatorName() << ":"
              << std::endl;
    unsigned int algorithm_id = 0;
    std::cin >> user_input;
    if (user_input == "quit") {
      break;
    }
    algorithm_id = boost::lexical_cast<unsigned int>(user_input);
    assert(algorithm_id < alg_ptrs.size());

    current_algorithm_ids[node_number] = algorithm_id;

    //                            double cost_current_plan =
    //                            current_plan.computeCost();
    //                            if(!hype::quiet && hype::verbose &&
    //                            hype::debug){
    //                                std::cout << current_plan.toString() <<
    //                                std::endl;
    //                                std::cout << "Total Cost: " <<
    //                                cost_current_plan/(1100*1000) << "ms" <<
    //                                std::endl;
    //                            }
    //
    //                            if(cost_current_plan<total_cost_optimal_plan){
    //                                total_cost_optimal_plan=cost_current_plan;
    //                                optimal_plan=current_plan;
    //                                if(!hype::quiet && hype::verbose &&
    //                                hype::debug)
    //                                    std::cout << "================ New
    //                                    Optimal Plan ===============" <<
    //                                    std::endl;
    //                            }

    // advance
    // advanceAlgorithmIDs(current_algorithm_ids, number_of_algorithms);
    // optimal_plan=current_plan;
  }

  // construct scheduling decisions
  // createSchedulingDecisionsforQEP(optimal_plan);

  // assign result plan to input reference
  plan = optimal_plan;
}

void optimizeQueryPlanBacktracking(QEP& plan) {
  // size_t num_op = plan.getNumberOfOperators();

  QEP optimal_plan;
  double total_cost_optimal_plan = std::numeric_limits<double>::max();

  // QEP current_plan(plan);

  hype::core::OperatorSequence op_seq = createOperatorSequence(plan);

  if (!hype::quiet && hype::verbose && hype::debug) {
    std::cout << "Serialized Plan: " << std::endl;
    util::print(op_seq, std::cout);
  }
  // stores for each operators the number of possible algorithms to choose from
  std::vector<std::vector<hype::core::AlgorithmPtr> >
      available_algorithms_per_operator =
          getAvailableAlgorithmsPerOperator(op_seq);

  // init array for number of algorithms per operator
  std::vector<int> number_of_algorithms(op_seq.size());
  for (unsigned int i = 0; i < op_seq.size(); ++i) {
    number_of_algorithms[i] = available_algorithms_per_operator[i]
                                  .size();  // number_of_algorithms[i];
  }

  // compute number of possible plans
  unsigned int number_of_plans = 1;
  for (unsigned int i = 0; i < op_seq.size(); ++i) {
    number_of_plans *= available_algorithms_per_operator[i]
                           .size();  // number_of_algorithms[i];
  }

  std::vector<int> current_algorithm_ids(op_seq.size());

  std::cout << "Backtracking exploring " << number_of_plans << " plans"
            << std::endl;
  for (unsigned int i = 0; i < number_of_plans; ++i) {
    QEP current_plan(plan);
    if (!hype::quiet && hype::verbose && hype::debug)
      std::cout << "Create Plan: " << std::endl;
    // create plan
    // for(unsigned int j=0;j<op_seq.size();++j){
    createPhysicalQueryPlan(current_plan, available_algorithms_per_operator,
                            current_algorithm_ids);

    double cost_current_plan = current_plan.computeCost();
    if (!hype::quiet && hype::verbose && hype::debug) {
      std::cout << current_plan.toString() << std::endl;
      std::cout << "Total Cost: " << cost_current_plan / (1100 * 1000) << "ms"
                << std::endl;
    }

    if (cost_current_plan < total_cost_optimal_plan) {
      total_cost_optimal_plan = cost_current_plan;
      optimal_plan = current_plan;
      if (!hype::quiet && hype::verbose && hype::debug)
        std::cout << "================ New Optimal Plan ==============="
                  << std::endl;
    }

    // advance
    advanceAlgorithmIDs(current_algorithm_ids, number_of_algorithms);
  }

  // construct scheduling decisions
  createSchedulingDecisionsforQEP(optimal_plan);

  // assign result plan to input reference
  plan = optimal_plan;
}

bool verifyPlan(QEP& plan) {
  // check the plan whether all QEP_Nodes have a valid Scheduling Decision and
  // that no Device Constraint is violated
  std::list<hype::query_optimization::QEP_Node*> order =
      plan.levelorder_traversal();  // current_plan.reverselevelorder_traversal();
                                    // //plan.levelorder_traversal();
  std::list<hype::query_optimization::QEP_Node*>::iterator it;
  unsigned int operator_index = 0;
  for (it = order.begin(); it != order.end(); ++it, ++operator_index) {
    // valid scheduling decisions?
    if ((*it)->sched_dec_ptr == NULL) {
      std::cerr << "Error! Node '" << (*it)->toString()
                << "' has no valid Scheduling Decision!" << std::endl;
      return false;
    }
    // device constraints considered?
    if (!util::satisfiesDeviceConstraint(
            (*it)->sched_dec_ptr->getDeviceSpecification(),
            (*it)->dev_constr)) {
      HYPE_WARNING(
          "Node '"
              << (*it)->toString() << "' does not satisfy its DeviceConstraint!"
              << "This can happen in case a co-processor has not enough memory",
          std::cerr);
      // return false;
    }
  }
  return true;
}

void optimizeQueryPlan(QEP& plan, const QueryOptimizationHeuristic& heuristic) {
  uint64_t begin = getTimestamp();

  //                    optimizeQueryPlanInteractive(plan);
  //                    if(!verifyPlan(plan)){
  //                        HYPE_FATAL_ERROR("QEP contains errors!",std::cout);
  //                    }
  //                    return;
  if (core::Runtime_Configuration::instance()
          .getDataPlacementDrivenOptimization()) {
    // optimizeQueryPlanDataPlacementAwareCriticalPath(plan);
    optimizeQueryPlanDataPlacementAware2(plan);
  } else {
    if (heuristic == BACKTRACKING) {
      optimizeQueryPlanBacktracking(plan);
    } else if (heuristic == GREEDY_HEURISTIC) {
      optimizeQueryPlanGreedy(plan);
    } else if (heuristic == GREEDY_CHAINER_HEURISTIC) {
      optimizeQueryPlanGreedyChainer(plan);
    } else if (heuristic == CRITICAL_PATH_HEURISTIC) {
      //                        unsigned int i;
      //                        optimizeQueryPlanAccelerateCriticalPath(plan,i);
      optimizeQueryPlanAccelerateCriticalPathRecursive(plan);
      // optimizeQueryPlanDataPlacementAwareCriticalPath(plan);
    } else if (heuristic == INTERACTIVE_USER_OPTIMIZATION) {
      optimizeQueryPlanInteractive(plan);
    } else if (heuristic == BEST_EFFORT_GPU_HEURISTIC) {
      optimizeQueryPlanGPUBestEffort(plan);
    }
  }
  // check whether for each operator a Scheduling Decision was created
  if (!verifyPlan(plan)) {
    HYPE_FATAL_ERROR("QEP contains errors!", std::cout);
  }
  uint64_t end = getTimestamp();
  std::string heuristic_name = util::getName(heuristic);
  if (core::Runtime_Configuration::instance()
          .getDataPlacementDrivenOptimization()) {
    heuristic_name = "DATA_PLACEMENT_AWARE_CRITICAL_PATH_HEURISTIC";
  }
  // std::cout << "Result Plan (" << heuristic_name << "): " << std::endl <<
  // plan.toString() << std::endl;
  // std::cout << "Time for optimization using heuristic '" << heuristic_name <<
  // "': " <<  double(end-begin)/(1000*1000) << "ms" << std::endl;
}

void insertCopyOperatorLeaveNode(hype::query_optimization::QEP_Node* leave) {
  assert(leave != NULL);
  assert(leave->alg_ptr != NULL);
  assert(leave->isLeave());
  if (leave->alg_ptr->getDeviceSpecification().getMemoryID() !=
      hype::PD_Memory_0) {
    std::string copy_type = util::getCopyOperationType(
        hype::PD0,
        leave->alg_ptr->getDeviceSpecification().getProcessingDeviceID());
    assert(!copy_type.empty());
    DeviceSpecification dev_spec =
        hype::core::Scheduler::instance().getDeviceSpecificationforCopyType(
            copy_type);
    core::AlgorithmPtr copy_alg_ptr =
        hype::core::Scheduler::instance().getAlgorithm(
            util::toInternalAlgName(copy_type, dev_spec));
    assert(copy_alg_ptr != NULL);
    // cout << "COPY Operator " << copy_alg_ptr.get() << endl;
    Tuple t;
    t.push_back(leave->op_spec.getFeatureVector().front());
    // DeviceSpecification dev_spec =
    // getDeviceSpecificationforCopyType(copy_type);
    // coyp data from CPU to co-processor:
    OperatorSpecification copy_op_spec(
        copy_type, t,
        hype::PD_Memory_0,  // memory id of input data
        leave->alg_ptr->getDeviceSpecification()
            .getMemoryID());  // memory id of result data

    hype::query_optimization::QEP_Node* copy_node =
        new hype::query_optimization::QEP_Node(copy_op_spec,
                                               hype::core::DeviceConstraint());
    copy_node->assignAlgorithm(copy_alg_ptr.get());
    // SchedulingDecision
    // copy_operator(*copy_alg_ptr,copy_alg_ptr->getEstimatedExecutionTime(t),t);
    // result_plan->push_back(copy_operator);
    //                                hype::query_optimization::QEP_Node*
    //                                old_parent = leave->parent;
    //                                hype::query_optimization::QEP_Node*
    //                                old_child = leave;
    leave->setChilds(&copy_node, 1);
  }
}
void insertCopyOperatorRootNode(hype::query_optimization::QEP& qep,
                                hype::query_optimization::QEP_Node* root) {
  if (root->alg_ptr->getDeviceSpecification().getMemoryID() !=
      hype::PD_Memory_0) {
    // cout << "Insert COPY Operation to transfer result to CPU" << endl;
    std::string copy_type = util::getCopyOperationType(
        root->alg_ptr->getDeviceSpecification().getProcessingDeviceID(),
        hype::PD0);  // assume that PD0 is CPU!
    assert(!copy_type.empty());
    DeviceSpecification dev_spec =
        hype::core::Scheduler::instance().getDeviceSpecificationforCopyType(
            copy_type);
    core::AlgorithmPtr copy_alg_ptr =
        hype::core::Scheduler::instance().getAlgorithm(
            util::toInternalAlgName(copy_type, dev_spec));
    assert(copy_alg_ptr != NULL);
    // cout << "COPY Operator " << copy_alg_ptr.get() << endl;
    Tuple t;
    t.push_back(root->op_spec.getFeatureVector().front());

    // copy data from CPU to co-processor:
    OperatorSpecification copy_op_spec(
        copy_type, t,
        root->alg_ptr->getDeviceSpecification()
            .getMemoryID(),  // memory id of input data
        hype::PD_Memory_0);  // memory id of result data

    hype::query_optimization::QEP_Node* copy_node =
        new hype::query_optimization::QEP_Node(copy_op_spec,
                                               hype::core::DeviceConstraint());
    copy_node->assignAlgorithm(copy_alg_ptr.get());

    copy_node->setChilds(&root, 1);
    qep.setRoot(copy_node);
    // InternalPhysicalOperator copy_operator(copy_alg_ptr, t,
    // copy_alg_ptr->getEstimatedExecutionTime(t).getTimeinNanoseconds());
    // SchedulingDecision
    // copy_operator(*copy_alg_ptr,copy_alg_ptr->getEstimatedExecutionTime(t),t);
  }

  //                            if(last_operator.alg_ptr->getDeviceSpecification().getMemoryID()!=hype::PD_Memory_0){
  //                                //cout << "Insert COPY Operation to transfer
  //                                result to CPU" << endl;
  //                                std::string copy_type =
  //                                util::getCopyOperationType(last_operator.alg_ptr->getDeviceSpecification().getProcessingDeviceID(),hype::PD0);
  //                                //assume that PD0 is CPU!
  //                                assert(!copy_type.empty());
  //                                DeviceSpecification dev_spec =
  //                                getDeviceSpecificationforCopyType(copy_type);
  //                                AlgorithmPtr copy_alg_ptr =
  //                                this->getAlgorithm(util::toInternalAlgName(copy_type,dev_spec
  //                                ));
  //                                assert(copy_alg_ptr!=NULL);
  //                                //cout << "COPY Operator " <<
  //                                copy_alg_ptr.get() << endl;
  //                                Tuple t;
  //                                t.push_back(last_operator.feature_vector.front());
  //                                InternalPhysicalOperator
  //                                copy_operator(copy_alg_ptr, t,
  //                                copy_alg_ptr->getEstimatedExecutionTime(t).getTimeinNanoseconds());
  //                                //SchedulingDecision
  //                                copy_operator(*copy_alg_ptr,copy_alg_ptr->getEstimatedExecutionTime(t),t);
  //                                result_plan.push_back(copy_operator);
  //                            }
}

void insertCopyOperatorInnerNode(
    hype::query_optimization::QEP_Node* inner_node) {
  assert(inner_node != NULL);
  assert(!inner_node->isRoot());
  // assert(!inner_node->isLeave());
  if (!util::isCopyOperation(inner_node->alg_ptr->getName())) {
    ProcessingDeviceMemoryID mem_id_current_node =
        inner_node->alg_ptr->getDeviceSpecification().getMemoryID();
    ProcessingDeviceMemoryID mem_id_parent_node =
        inner_node->parent->alg_ptr->getDeviceSpecification().getMemoryID();
    if (mem_id_current_node != mem_id_parent_node) {
      // if input data is ached we do not need
      // to insert he copy operation in the plan
      // because we do not want to include copy overhead
      // in this case
      if (inner_node->parent->input_data_cached) return;

      std::string copy_type = util::getCopyOperationType(
          inner_node->alg_ptr->getDeviceSpecification().getProcessingDeviceID(),
          inner_node->parent->alg_ptr->getDeviceSpecification()
              .getProcessingDeviceID());
      assert(!copy_type.empty());
      DeviceSpecification dev_spec =
          hype::core::Scheduler::instance().getDeviceSpecificationforCopyType(
              copy_type);
      core::AlgorithmPtr copy_alg_ptr =
          hype::core::Scheduler::instance().getAlgorithm(
              util::toInternalAlgName(copy_type, dev_spec));
      assert(copy_alg_ptr != NULL);
      Tuple t;

      t.push_back(inner_node->op_spec.getFeatureVector().front());

      // copy data from CPU to co-processor:
      OperatorSpecification copy_op_spec(
          copy_type, t,
          mem_id_current_node,  // memory id of input data
          mem_id_parent_node);  // memory id of result data

      hype::query_optimization::QEP_Node* copy_node =
          new hype::query_optimization::QEP_Node(
              copy_op_spec, hype::core::DeviceConstraint());
      copy_node->assignAlgorithm(copy_alg_ptr.get());
      if (!hype::quiet && hype::verbose && hype::debug)
        std::cout << "create Copy Operation: " << copy_node->toString()
                  << std::endl;

      hype::query_optimization::QEP_Node* old_parent = inner_node->parent;

      copy_node->setChilds(&inner_node, 1);
      copy_node->parent = old_parent;

      for (unsigned int i = 0; i < old_parent->number_childs; ++i) {
        if (old_parent->childs[i] == inner_node) {
          old_parent->childs[i] = copy_node;
        }
      }

      // InternalPhysicalOperator copy_operator(copy_alg_ptr, t,
      // copy_alg_ptr->getEstimatedExecutionTime(t).getTimeinNanoseconds());
      // SchedulingDecision copy_operator( (*(copy_alg_ptr.get())),
      // copy_alg_ptr->getEstimatedExecutionTime(t),t);
      // cout << "insert " << copy_operator.getNameofChoosenAlgorithm() << "
      // between " << result_plan->back().getNameofChoosenAlgorithm() << " and "
      // << current_operator.getNameofChoosenAlgorithm()   << endl;
      // result_plan.push_back(copy_operator);
    }
  }
}

}  // end namespace query_optimization
}  // end namespace hype
