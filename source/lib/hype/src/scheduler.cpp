

#include <stdlib.h>
#include <core/scheduler.hpp>
#include <plugins/pluginloader.hpp>

#include <boost/thread.hpp>
#include <core/device_memory.hpp>
#include <hype.hpp>
#include <query_processing/device_operator_queue.hpp>
#include <query_processing/node.hpp>
#include <queue>
#include <util/algorithm_name_conversion.hpp>
#include <util/print.hpp>
#include <util/utility_functions.hpp>
//#include <query_processing/node.hpp>

using namespace std;

namespace hype {
namespace core {

using namespace queryprocessing;

boost::mutex global_mutex;

Scheduler::Scheduler()
    : map_operationname_to_operation_(),
      map_statisticalmethodname_to_statisticalmethod_(),
      proc_devs_(),
      dma_cpu_cp_(hype::PD_DMA0, hype::DMA, hype::PD_NO_Memory),
      dma_cp_cpu_(hype::PD_DMA1, hype::DMA, hype::PD_NO_Memory),
      global_operator_stream_(),
      operator_stream_mutex_(),
      scheduling_thread_mutex_(),
      scheduling_thread_cond_var_(),
      scheduling_thread_(
          new boost::thread(boost::bind(&Scheduler::scheduling_thread, this))),
      terminate_threads_(false) {
  // is called once, because Scheduler is a Singelton
  PluginLoader::loadPlugins();

  // COPY from a CPU to a co-processor
  {
    AlgorithmSpecification copy_operator("COPY_CPU_CP", "COPY_CPU_CP",
                                         hype::Least_Squares_1D, hype::Periodic,
                                         hype::ResponseTime);
    // add explicit COPY operator to HyPE
    this->addAlgorithm(copy_operator, dma_cpu_cp_);
  }
  {
    // COPY from a co-processor to a CPU
    AlgorithmSpecification copy_operator("COPY_CP_CPU", "COPY_CP_CPU",
                                         hype::Least_Squares_1D, hype::Periodic,
                                         hype::ResponseTime);
    // add explicit COPY operator to HyPE
    this->addAlgorithm(copy_operator, dma_cp_cpu_);
  }
  // COPY from a co-processor to a co-processor
  {
    AlgorithmSpecification copy_operator("COPY_CP_CP", "COPY_CP_CP",
                                         hype::Least_Squares_1D, hype::Periodic,
                                         hype::ResponseTime);
    // We currentlyassume that the DMA controller for copying
    // data from CP to CPU also always performs copies CP to CP
    /* \todo Verify!*/
    this->addAlgorithm(copy_operator, dma_cp_cpu_);
  }
  //                        //DUMMY Operator doing nothing, e.g., the root node
  //                        of a MonetDB MAL plan
  //                        {
  //                        AlgorithmSpecification
  //                        dummy_operator("DUMMY_ALGORITHM",
  //                                                             "DUMMY_OPERATOR",
  //                                                             hype::Least_Squares_1D,
  //                                                             hype::No_Recomputation,
  //                                                             //we do not
  //                                                             want any
  //                                                             overhead for
  //                                                             this
  //                                                             hype::ResponseTime);
  //                        //We currently assume that the DMA controller for
  //                        copying
  //                        //data from CP to CPU also always performs copies CP
  //                        to CP
  //                        /* \todo Verify!*/
  //                        /*FIXME: DUMMY OPERATOR has ZERO execution cost, we
  //                         * just assign it to a DMA Controlelr for now (which
  //                         is
  //                         * obviously nonsense, but we do not know about
  //                         * registered cpus yet)*/
  //                        this->addAlgorithm(dummy_operator,dma_cp_cpu_);
  //                        }
}

Scheduler::~Scheduler() {
  {
    boost::lock_guard<boost::mutex> lock2(this->operator_stream_mutex_);
    terminate_threads_ = true;
  }
  // notify scheduling thread that it can terminate
  this->scheduling_thread_cond_var_.notify_one();
  //                    this->scheduling_thread_->interrupt();
  this->scheduling_thread_->join();
}

bool Scheduler::addAlgorithm(const AlgorithmSpecification& alg_spec,
                             const DeviceSpecification& dev_spec) {
  boost::lock_guard<boost::mutex> lock(global_mutex);

  std::string internal_algorithm_name =
      hype::util::toInternalAlgName(alg_spec.getAlgorithmName(), dev_spec);

  if (!this->proc_devs_.exists(dev_spec)) {
    if (!this->proc_devs_.addDevice(dev_spec)) {
      std::cout << "FATAL ERROR! Failed to add Processing Device while "
                   "creating Algorithm '"
                << alg_spec.getAlgorithmName() << "'" << std::endl;
    }
  }
  if (!DeviceMemories::instance().existDeviceMemory(dev_spec.getMemoryID())) {
    if (!DeviceMemories::instance().addDeviceMemory(
            dev_spec.getMemoryID(), dev_spec.getTotalMemoryCapacity())) {
      HYPE_WARNING("Failed to add DeviceMemory while creating Algorithm '"
                       << alg_spec.getAlgorithmName() << "'",
                   std::cerr);
    }
  }

  std::string name_of_operation = alg_spec.getOperationName();

  std::map<std::string, boost::shared_ptr<Operation> >::iterator it;

  it = map_operationname_to_operation_.find(name_of_operation);

  boost::shared_ptr<Operation> op;

  if (it == map_operationname_to_operation_.end()) {  // operation does not
                                                      // exist
    op = boost::shared_ptr<Operation>(new Operation(name_of_operation));
    op->setNewOptimizationCriterion(alg_spec.getOptimizationCriterionName());
    map_operationname_to_operation_[name_of_operation] = op;
  } else {
    op = it->second;  // boost::shared_ptr<Operation> (it->second);
  }
  return op->addAlgorithm(internal_algorithm_name, dev_spec,
                          alg_spec.getStatisticalMethodName(),
                          alg_spec.getRecomputationHeuristicName());
}

bool Scheduler::hasOperation(const std::string& operation_name) const {
  std::map<std::string, boost::shared_ptr<Operation> >::const_iterator cit;
  cit = map_operationname_to_operation_.find(operation_name);
  return (cit != map_operationname_to_operation_.end());
}

Scheduler& Scheduler::instance() {
  static Scheduler scheduler;
  return scheduler;
}
/*
        bool Scheduler::addAlgorithm(const std::string& name_of_operation,
                                                                        const
   std::string& name_of_algorithm,
                                                                        DeviceSpecification
   comp_dev,
                                                                        const
   std::string& name_of_statistical_method,
                                                                        const
   std::string& name_of_recomputation_strategy){

                boost::lock_guard<boost::mutex> lock(global_mutex);

                std::map<std::string,boost::shared_ptr<Operation> >::iterator
   it;

                it=map_operationname_to_operation_.find(name_of_operation);

                boost::shared_ptr<Operation> op;

                if(it==map_operationname_to_operation_.end()){ //operation does
   not exist
                        op = boost::shared_ptr<Operation>(new
   Operation(name_of_operation));
                        map_operationname_to_operation_[name_of_operation]=op;
                }else{
                        op=it->second; //boost::shared_ptr<Operation>
   (it->second);
                }

                return op->addAlgorithm(name_of_algorithm,
                                                                                comp_dev,
                                                                           name_of_statistical_method,
                                                                           name_of_recomputation_strategy);


         //return true;
        }*/

bool Scheduler::setOptimizationCriterion(
    const std::string& name_of_operation,
    const std::string& name_of_optimization_criterion) {
  boost::lock_guard<boost::mutex> lock(global_mutex);

  //			map_operationname_to_operation_.find
  //			StatisticalMethodMap::iterator it =
  // map_statisticalmethodname_to_statisticalmethod_.find(name_of_operation);

  boost::shared_ptr<OptimizationCriterion_Internal> opt_crit =
      getNewOptimizationCriterionbyName(name_of_optimization_criterion);
  if (opt_crit) {
    // StatisticalMethodMap::iterator it =
    // map_statisticalmethodname_to_statisticalmethod_.find(name_of_operation);

    MapNameToOperation::iterator it =
        map_operationname_to_operation_.find(name_of_operation);
    if (it == map_operationname_to_operation_.end()) {
      cout << "Operation not found! " << name_of_operation << endl;
      return false;
    }
    return (*it).second->setNewOptimizationCriterion(
        name_of_optimization_criterion);
    // map_operationname_to_operation_[name_of_operation]=opt_crit;
    // return true;
  }
  return false;
}

/* not part of interface!!! -> no thread safety implemented!!!*/
const AlgorithmPtr Scheduler::getAlgorithm(
    const std::string& name_of_algorithm) {
  // boost::lock_guard<boost::mutex> lock(global_mutex);

  MapNameToOperation::iterator it;
  for (it = map_operationname_to_operation_.begin();
       it != map_operationname_to_operation_.end(); it++) {
    if (it->second->hasAlgorithm(name_of_algorithm))
      return it->second->getAlgorithm(name_of_algorithm);
  }
  return AlgorithmPtr();  // NULL Pointer if algorithm is not found
}

bool Scheduler::setStatisticalMethod(
    const std::string& name_of_algorithm,
    const std::string& name_of_statistical_method) {
  boost::lock_guard<boost::mutex> lock(global_mutex);

  AlgorithmPtr alg_ptr = this->getAlgorithm(name_of_algorithm);
  if (!alg_ptr) {
    cout << "Error in Scheduler::setStatisticalMethod(): Algorithm "
         << name_of_algorithm << " not found!" << endl;
    return false;
  }
  cout << "Found Algorithm: " << alg_ptr->getName() << endl;
  StatisticalMethodPtr stat_meth_ptr =
      getNewStatisticalMethodbyName(name_of_statistical_method);
  if (!stat_meth_ptr) return false;
  cout << "created statistical method: " << name_of_statistical_method << endl;
  return alg_ptr->setStatisticalMethod(stat_meth_ptr);
}

bool Scheduler::setRecomputationHeuristic(
    const std::string& name_of_algorithm,
    const std::string& name_of_recomputation_strategy) {
  boost::lock_guard<boost::mutex> lock(global_mutex);
  // const std::string
  // name_of_algorithm(hype::util::toInternalAlgName(sched_dec.getNameofChoosenAlgorithm(),sched_dec.getDeviceSpecification()));
  AlgorithmPtr alg_ptr = this->getAlgorithm(name_of_algorithm);
  if (!alg_ptr) {
    cout << "Error in Scheduler::setRecomputationHeuristic(): Algorithm "
         << name_of_algorithm << " not found!" << endl;
    return false;
  }
  boost::shared_ptr<RecomputationHeuristic_Internal> recomp_heuristic =
      getNewRecomputationHeuristicbyName(name_of_recomputation_strategy);
  if (!recomp_heuristic) return false;
  return alg_ptr->setRecomputationHeuristic(recomp_heuristic);
}

const SchedulingDecision Scheduler::getOptimalAlgorithm(
    const OperatorSpecification& op_spec, const DeviceConstraint& dev_constr) {
  boost::lock_guard<boost::mutex> lock(global_mutex);

  std::map<std::string, boost::shared_ptr<Operation> >::iterator it;

  std::string name_of_operation = op_spec.getOperatorName();

  it = map_operationname_to_operation_.find(name_of_operation);

  boost::shared_ptr<Operation> op;

  if (it == map_operationname_to_operation_.end()) {  // operation does not
                                                      // exist

    std::cout << "[HyPE:] FATAL ERROR: In "
                 "hype::core::Scheduler::getOptimalAlgorithm(): Operation "
              << name_of_operation << " does not exist!!!" << std::endl;
    cout << "File: " << __FILE__ << " at Line: " << __LINE__ << endl;
    exit(-1);
  }  // else{

  op = it->second;  // boost::shared_ptr<Operation> (it->second);
  if (!op) {
    std::cout << "FATAL ERROR: Operation " << name_of_operation
              << ": NULL Pointer!" << std::endl;
    exit(-1);
  }
  return op->getOptimalAlgorithm(op_spec.getFeatureVector(), dev_constr);
}

void advanceAlgorithmIDs(vector<int>& current_algorithm_ids,
                         const vector<int>& number_of_algorithms,
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

// add scheduling decisions to internal LoadTracking mechanism
void addSchedulingDecisionVectorToLoadTracker(
    SchedulingDecisionVectorPtr phy_plan) {
  for (unsigned int i = 0; i < phy_plan->size(); ++i) {
    core::Scheduler::instance().getProcessingDevices().addSchedulingDecision(
        (*phy_plan)[i]);
  }
}

const SchedulingDecisionVectorPtr Scheduler::getOptimalAlgorithm(
    const OperatorSequence& op_seq,
    const QueryOptimizationHeuristic& heuristic) {
  SchedulingDecisionVectorPtr result_plan(
      new SchedulingDecisionVector());  // (new
  // SchedulingDecisionVector(op_seq.size()));

  if (heuristic == GREEDY_HEURISTIC) {
    for (unsigned int i = 0; i < op_seq.size(); ++i) {
      SchedulingDecision sched_dec =
          this->getOptimalAlgorithm(op_seq[i].first, op_seq[i].second);
      // util::print(sched_dec,cout);
      // is first operator?
      if (i == 0) {
        insertCopyOperationAtBegin(result_plan, sched_dec);
        result_plan->push_back(sched_dec);
        // is last operator?
      } else if (i == op_seq.size() - 1) {
        insertCopyOperationInsidePlan(result_plan, sched_dec);
        result_plan->push_back(sched_dec);
        insertCopyOperationAtEnd(result_plan, sched_dec);
      } else {
        insertCopyOperationInsidePlan(result_plan, sched_dec);
        result_plan->push_back(sched_dec);
      }
      // util::print(result_plan,cout);
    }
    addSchedulingDecisionVectorToLoadTracker(result_plan);
    return result_plan;
  } else if (heuristic == BACKTRACKING) {
    // stores for each operators the number of possible algorithms to choose
    // from
    vector<int> number_of_algorithms(op_seq.size());
    vector<vector<AlgorithmPtr> > available_algorithms_per_operator(
        op_seq.size());
    for (unsigned int i = 0; i < op_seq.size(); ++i) {
      MapNameToOperation::iterator it =
          this->map_operationname_to_operation_.find(
              op_seq[i].first.getOperatorName());
      if (it == this->map_operationname_to_operation_.end()) {
        // error, return NUll Pointer
        return SchedulingDecisionVectorPtr();
      }
      // fill the number of algorithms array
      const std::vector<AlgorithmPtr> alg_ptrs = it->second->getAlgorithms();
      for (unsigned int j = 0; j < alg_ptrs.size(); ++j) {
        if (util::satisfiesDeviceConstraint(
                alg_ptrs[j]->getDeviceSpecification(),
                op_seq[i].second.getDeviceTypeConstraint())) {
          number_of_algorithms[i]++;
          available_algorithms_per_operator[i].push_back(alg_ptrs[j]);
        }
        //                                if(op_seq[i].second.getDeviceTypeConstraint()==hype::ANY_DEVICE
        //                                  ||
        //                                  (op_seq[i].second.getDeviceTypeConstraint()==hype::CPU_ONLY
        //                                  &&
        //                                  alg_ptrs[j]->getDeviceSpecification().getDeviceType()==hype::CPU)
        //                                  ||
        //                                  (op_seq[i].second.getDeviceTypeConstraint()==hype::GPU_ONLY
        //                                  &&
        //                                  alg_ptrs[j]->getDeviceSpecification().getDeviceType()==hype::GPU)){
        //                                        number_of_algorithms[i]++;
        //                                        available_algorithms_per_operator[i].push_back(alg_ptrs[j]);
        //                                }
      }
    }

    // compute number of possible plans
    unsigned int number_of_plans = 1;
    for (unsigned int i = 0; i < op_seq.size(); ++i) {
      number_of_plans *= number_of_algorithms[i];
    }

    vector<int> current_algorithm_ids(number_of_algorithms.size());

    InternalPhysicalPlan optimal_plan;
    double total_cost_optimal_plan = std::numeric_limits<double>::max();

    cout << "Backtracking exploring " << number_of_plans << " plans" << endl;
    for (unsigned int i = 0; i < number_of_plans; ++i) {
      InternalPhysicalPlan current_plan;

      cout << "Create Plan: " << endl;
      // create plan
      for (unsigned int j = 0; j < op_seq.size(); ++j) {
        // cout << "Current Algorithm ID: " << current_algorithm_ids[j] << endl;
        // fetch algorithm
        AlgorithmPtr alg_ptr =
            available_algorithms_per_operator[j][current_algorithm_ids[j]];
        // compute cost
        double estimated_time =
            alg_ptr
                ->getEstimatedExecutionTime(op_seq[j].first.getFeatureVector())
                .getTimeinNanoseconds();
        // current_plan.push_back(make_pair(alg_ptr,estimated_time));
        InternalPhysicalOperator phy_op(
            alg_ptr, op_seq[j].first.getFeatureVector(), estimated_time);

        // is first operator?
        if (j == 0) {
          insertCopyOperationAtBegin(current_plan, phy_op);
          current_plan.push_back(phy_op);
          // is last operator?
        } else if (j == op_seq.size() - 1) {
          insertCopyOperationInsidePlan(current_plan, phy_op);
          current_plan.push_back(phy_op);
          insertCopyOperationAtEnd(current_plan, phy_op);
        } else {
          insertCopyOperationInsidePlan(current_plan, phy_op);
          current_plan.push_back(phy_op);
        }

        // current_plan.push_back(phy_op);
        // cost_current_plan+=estimated_time;

        // cout << current_plan.back().first->getName() << ": " <<
        // estimated_time/(1000*1000) << "ms" << endl;
      }
      // compute cost
      double cost_current_plan = 0;
      for (unsigned int j = 0; j < current_plan.size(); ++j) {
        cost_current_plan += current_plan[j].cost;
      }

      // cout << "Cost: " << cost_current_plan/(1000*1000) << "ms" << endl;
      util::print(current_plan, cout);

      if (cost_current_plan < total_cost_optimal_plan) {
        total_cost_optimal_plan = cost_current_plan;
        optimal_plan = current_plan;
      }

      // advance
      advanceAlgorithmIDs(current_algorithm_ids, number_of_algorithms);

      //                            for(unsigned int j=0;j<op_seq.size();++j){
      //                                if(current_algorithm_ids[j] <
      //                                number_of_algorithms[j]){
      //                                   current_algorithm_ids[j]++;
      //                                }else{
      //                                    current_algorithm_ids[j]=0
      //                                    f(j<op_seq.size()-1){
      //                                        current_algorithm_ids[j+1]++;
      //                                    }
      //                                }
      //                            }
    }
    // ensure result is a valid plan
    assert(!optimal_plan.empty());

    // create SchedulingDecisions
    //                        for(unsigned int i=0;i<optimal_plan.size();++i){
    //                            //create SchedulingDecision for this algorithm
    //                            cout << "feature Vector Size: " <<
    //                            op_seq[i].first.getFeatureVector().size() <<
    //                            endl;
    //                            result_plan->push_back(SchedulingDecision(*(optimal_plan[i].alg_ptr),
    //                            hype::core::EstimatedTime(optimal_plan[i].cost),
    //                            op_seq[i].first.getFeatureVector()));
    ////                            SchedulingDecision
    /// s(*(optimal_plan[i].first),
    /// hype::core::EstimatedTime(optimal_plan[i].second),
    /// op_seq[i].first.getFeatureVector());
    ////                            result_plan->push_back(s);
    //                        }

    for (unsigned int i = 0; i < optimal_plan.size(); ++i) {
      // create SchedulingDecision for this algorithm
      //                            cout << "feature Vector Size: " <<
      //                            op_seq[i].first.getFeatureVector().size() <<
      //                            endl;
      result_plan->push_back(
          SchedulingDecision(*(optimal_plan[i].alg_ptr),
                             hype::core::EstimatedTime(optimal_plan[i].cost),
                             optimal_plan[i].feature_vector));
      //                            SchedulingDecision
      //                            s(*(optimal_plan[i].first),
      //                            hype::core::EstimatedTime(optimal_plan[i].second),
      //                            op_seq[i].first.getFeatureVector());
      //                            result_plan->push_back(s);
    }

    addSchedulingDecisionVectorToLoadTracker(result_plan);
    return result_plan;
  }

  return nullptr;
}

//                #define MAX_CHILDS 10
//
//                struct QEP_Node{
//                        OperatorSpecification op_spec;
//                        DeviceConstraint dev_constr;
//                        QEP_Node* parent;
//                        QEP_Node* childs[MAX_CHILDS];
//                        size_t number_childs;
//                        Algorithm* alg_ptr;
//                        double cost;
//
//
//                        QEP_Node(const QEP_Node& other) : op_spec
//                        (other.op_spec),
//                                                           dev_constr(other.dev_constr),
//                                                           parent(NULL),
//                                                           childs(),
//                                                           number_childs(other.number_childs),
//                                                           alg_ptr(other.alg_ptr),
//                                                           cost(other.cost){
//
//                            for(unsigned int i=0;i<other.number_childs;++i){
//                                //call copy constructor
//                                childs[i]=new QEP_Node(*other.childs[i]);
//                                childs[i]->parent=this;
//                            }
//
//
//                        }
//
//                        QEP_Node& operator=(const QEP_Node& other){
//                            //check for self assignment
//                            if(this!=&other){
//                                cleanupChilds();
//
//                                this->alg_ptr=other.alg_ptr;
//                                this->cost=other.cost;
//                                this->dev_constr=other.dev_constr;
//                                this->op_spec=other.op_spec;
//                                this->number_childs=other.number_childs;
//                                this->parent=other.parent;
//
//                                for(unsigned int
//                                i=0;i<other.number_childs;++i){
//                                    //call copy constructor
//                                    childs[i]=new QEP_Node(*other.childs[i]);
//                                    childs[i]->parent=this;
//                                }
//
//                            }
//                        }
//
//                        void assignAlgorithm(Algorithm* alg_ptr){
//                            this->alg_ptr=alg_ptr;
//                            this->cost=alg_ptr->getEstimatedExecutionTime(op_spec.getFeatureVector()).getTimeinNanoseconds();
//                        }
//
//                        void setChilds(QEP_Node** childs, size_t
//                        number_childs){
//                            assert(number_childs<=MAX_CHILDS);
//                            cleanupChilds();
//                            for(unsigned int i=0;i<number_childs;++i){
//                                this->childs[i]=childs[i];
//                                this->childs[i]->parent=this;
//                            }
//                            this->number_childs=number_childs;
//                        }
//
//                        void setParent(QEP_Node* parent){
//                            this->parent=parent;
//                        }
//
//                        ~QEP_Node(){
//                            for(unsigned int i=0;i<number_childs;++i){
//                                delete childs[i];
//                            }
//                        }
//
//                        std::string toString(){
//                            std::stringstream ss;
//                            if(alg_ptr){
//                                ss << alg_ptr->getName() << ": " <<
//                                this->cost;
//                                //ss << alg_ptr->getName() << "(" <<
//                                op_spec.getFeatureVector() << "): " <<
//                                this->cost;
//                            }else{
//                                ss << op_spec.getOperatorName();
//                                //ss << op_spec.getOperatorName() << "(" <<
//                                op_spec.getFeatureVector() << ")";
//                            }
//                            return ss.str();
//                        }
//
//                        void cleanupChilds(){
//                            for(unsigned int i=0;i<number_childs;++i){
//                                if( childs[i]) delete childs[i];
//                            }
//                        }
//
//                };
//
//                class QEP{
//                public:
//                    QEP(QEP_Node* root) : root_(root){
//
//                    }
//
//                    QEP(const QEP& other) : root_(new QEP_Node(*other.root_)){
//
//
//                    }
//
//                    std::string toString(unsigned int indent=1){
//                        std::stringstream ss;
//
//
//                        for(unsigned int i=0;i<indent;++i){
//                            ss << "\t";
//                        }
//
//                        ss << root_->toString() << endl;
//
//                        QEP_Node** childs=root_->childs;
//                        size_t number_childs=root_->number_childs;
//
//                        for(unsigned int i=0;i<number_childs;++i){
//                               ss << QEP(childs[i]).toString(indent+1);
//                        }
//
//
//                        return ss.str();
//                    }
//
//                    ~QEP(){
//                        if(root_)
//                                delete root_;
//                    }
//                private:
//                    QEP_Node* root_;
//                };
//
//                void optimizeQueryPlan(QEP& plan);

//		const SchedulingDecision
// Scheduler::getOptimalAlgorithmName(const
// std::string& name_of_operation, const Tuple& input_values,
// DeviceTypeConstraint dev_constr){
//
//			boost::lock_guard<boost::mutex> lock(global_mutex);
//
//			std::map<std::string,boost::shared_ptr<Operation>
//>::iterator it;
//
//			it=map_operationname_to_operation_.find(name_of_operation);
//
//			boost::shared_ptr<Operation> op;
//
//			if(it==map_operationname_to_operation_.end()){
////operation
// does not exist
//				std::cout << "FATAL ERROR: Operation " <<
// name_of_operation  <<  " does not exist!!!" << std::endl;
//				exit(-1);
//			}//else{
//
//				op=it->second; //boost::shared_ptr<Operation>
//(it->second);
//				if(!op){
//					std::cout << "FATAL ERROR: Operation "
//<<
// name_of_operation  <<  ": NULL Pointer!" << std::endl;
//					exit(-1);
//				}
//				return op->getOptimalAlgorithm(input_values,
// dev_constr);
//			//}
//
//		   //return
// SchedulingDecision(std::string(),EstimatedTime(-1.0),Tuple());
//		}

bool Scheduler::addObservation(const SchedulingDecision& sched_dec,
                               const double& measured_execution_time) {
  boost::lock_guard<boost::mutex> lock(global_mutex);

  if (sched_dec.getNameofChoosenAlgorithm() != "COPY")
    this->proc_devs_.removeSchedulingDecision(sched_dec);

  MemoryChunkPtr mem_chunk = sched_dec.getMemoryChunk();
  if (mem_chunk) {
    DeviceMemoryPtr dev_mem =
        DeviceMemories::instance().getDeviceMemory(mem_chunk->mem_id);
    dev_mem->releaseMemory(mem_chunk);
  }

  // Scheduling decisions return the external algorithm name, so to compute the
  // internal algorithm name, we need the external name and the device
  // specification to get a unique string for the algorithm associated with the
  // processing device
  const std::string name_of_algorithm(
      hype::util::toInternalAlgName(sched_dec.getNameofChoosenAlgorithm(),
                                    sched_dec.getDeviceSpecification()));
  const MeasurementPair mp(sched_dec.getFeatureValues(),
                           MeasuredTime(measured_execution_time),
                           sched_dec.getEstimatedExecutionTimeforAlgorithm());

  MapNameToOperation::iterator it;
  for (it = map_operationname_to_operation_.begin();
       it != map_operationname_to_operation_.end(); it++) {
    if (it->second->hasAlgorithm(name_of_algorithm))
      return it->second->addObservation(name_of_algorithm, mp);
  }
  return false;
}

EstimatedTime Scheduler::getEstimatedExecutionTime(
    const OperatorSpecification& op_spec, const std::string& alg_name) {
  boost::lock_guard<boost::mutex> lock(global_mutex);
  AlgorithmPtr alg_ptr = getAlgorithm(alg_name);
  if (!alg_ptr) {
    std::cout
        << "FATAL ERROR! Scheduler::getEstimatedExecutionTime(): Algorithm '"
        << alg_name << "' not found for operation '"
        << op_spec.getOperatorName() << "'!" << std::endl;
    std::exit(-1);
  }
  return alg_ptr->getEstimatedExecutionTime(op_spec.getFeatureVector());
}

void Scheduler::print() {
  boost::lock_guard<boost::mutex> lock(global_mutex);
  cout << "HyPE Status:" << endl;
  MapNameToOperation::iterator it;
  for (it = map_operationname_to_operation_.begin();
       it != map_operationname_to_operation_.end(); ++it) {
    //                        std::cout << "Operation: '" <<
    //                        it->second->getName() << "'" << std::endl;
    std::cout << "Operation: '" << it->second->toString() << "'" << std::endl;

    std::vector<AlgorithmPtr> algs = it->second->getAlgorithms();
    for (unsigned int i = 0; i < algs.size(); ++i) {
      std::cout << algs[i]->toString(1) << endl;  // algs[i]->getName() << endl;
    }
  }
  proc_devs_.print();
}

void Scheduler::setGlobalLoadAdaptionPolicy(
    RecomputationHeuristic recomp_heur) {
  boost::lock_guard<boost::mutex> lock(global_mutex);
  MapNameToOperation::iterator it;
  for (it = map_operationname_to_operation_.begin();
       it != map_operationname_to_operation_.end(); ++it) {
    std::vector<AlgorithmPtr> algs = it->second->getAlgorithms();
    std::string name_of_recomputation_strategy = util::getName(recomp_heur);
    for (unsigned int i = 0; i < algs.size(); ++i) {
      algs[i]->setRecomputationHeuristic(
          getNewRecomputationHeuristicbyName(name_of_recomputation_strategy));
      // std::cout << algs[i]->toString(1) << endl; //algs[i]->getName() <<
      // endl;
    }
  }
}

const Scheduler::MapNameToOperation& Scheduler::getOperatorMap() {
  return this->map_operationname_to_operation_;
}

bool Scheduler::registerMemoryCostModel(
    const AlgorithmSpecification& alg_spec, const DeviceSpecification& dev_spec,
    MemoryCostModelFuncPtr memory_cost_model) {
  boost::lock_guard<boost::mutex> lock(global_mutex);

  std::string internal_algorithm_name =
      hype::util::toInternalAlgName(alg_spec.getAlgorithmName(), dev_spec);
  AlgorithmPtr alg_ptr = this->getAlgorithm(internal_algorithm_name);
  if (!alg_ptr) {
    return false;
  }
  alg_ptr->setMemoryCostModel(memory_cost_model);
  return true;
}

void Scheduler::addIntoGlobalOperatorStream(queryprocessing::NodePtr op) {
  // boost::lock_guard<boost::mutex> lock(global_mutex);
  // boost::lock_guard<boost::mutex> lock(global_mutex);
  {
    boost::lock_guard<boost::mutex> lock2(this->operator_stream_mutex_);
    this->global_operator_stream_.push(op);
  }
  // notify scheduling thread that there is work to do
  this->scheduling_thread_cond_var_.notify_one();
}

void Scheduler::addIntoGlobalOperatorStream(
    const std::list<queryprocessing::NodePtr>& operator_list) {
  // boost::lock_guard<boost::mutex> lock(global_mutex);
  {  // boost::lock_guard<boost::mutex> lock(global_mutex);
    boost::lock_guard<boost::mutex> lock2(this->operator_stream_mutex_);

    std::list<queryprocessing::NodePtr>::const_iterator cit;
    for (cit = operator_list.begin(); cit != operator_list.end(); ++cit) {
      this->global_operator_stream_.push(*cit);
    }
  }
  // notify scheduling thread that there is work to do
  this->scheduling_thread_cond_var_.notify_one();
}

const std::list<std::pair<std::string, double> >
Scheduler::getAverageEstimationErrors() const {
  boost::lock_guard<boost::mutex> lock(global_mutex);

  std::list<std::pair<std::string, double> > result;
  MapNameToOperation::const_iterator it;
  for (it = map_operationname_to_operation_.begin();
       it != map_operationname_to_operation_.end(); ++it) {
    //                        std::cout << "Operation: '" <<
    //                        it->second->getName() << "'" << std::endl;
    // std::cout << "Operation: '" << it->second->toString() << "'" <<
    // std::endl;

    std::vector<AlgorithmPtr> algs = it->second->getAlgorithms();
    for (unsigned int i = 0; i < algs.size(); ++i) {
      result.push_back(std::make_pair(
          algs[i]->getName(),
          algs[i]
              ->getAlgorithmStatistics()
              .getAverageRelativeError()));  // << endl; //algs[i]->getName() <<
                                             // endl;
    }
  }
  return result;
}
const std::list<std::pair<std::string, double> >
Scheduler::getTotalProcessorTimes() const {
  boost::lock_guard<boost::mutex> lock(global_mutex);
  return this->proc_devs_.getTotalProcessorTimes();
}

/*
bool Scheduler::addObservation(const std::string& name_of_algorithm, const
MeasurementPair& mp){

        boost::lock_guard<boost::mutex> lock(global_mutex);

        this->proc_devs_.removeSchedulingDecision(sched_dec);

        MapNameToOperation::iterator it;
        for(it= map_operationname_to_operation_.begin(); it
!=map_operationname_to_operation_.end(); it++){
                if(it->second->hasAlgorithm(name_of_algorithm))
                         return
it->second->addObservation(name_of_algorithm,mp);
        }
        return false;
}*/

/*################################################################################################*/
/*############################## PRIVATE UTIL FUNCTIONS
 * ##########################################*/
/*################################################################################################*/

void Scheduler::scheduling_thread() {
  while (true) {
    {
      // boost::unique_lock<boost::mutex> lock(this->scheduling_thread_mutex_);
      boost::unique_lock<boost::mutex> lock(this->operator_stream_mutex_);
      while (this->global_operator_stream_.empty()) {
        this->scheduling_thread_cond_var_.wait(lock);
        if (this->terminate_threads_) {
          return;
        }
        //                                int res=0;
        //                                res=pthread_cond_wait(this->scheduling_thread_cond_var_.native_handle(),this->operator_stream_mutex_.native_handle());
        //                                boost::this_thread::interruption_point();
        //                                if(res)
        //                                {
        //                                    boost::throw_exception(condition_error());
        //                                }
      }
      //                        }
      //                        {
      //                            boost::lock_guard<boost::mutex>
      //                            lock(this->operator_stream_mutex_);
      while (!global_operator_stream_.empty()) {
        NodePtr logical_operator = global_operator_stream_.front();
        global_operator_stream_.pop();

        Tuple t = logical_operator->getFeatureVector();
        //                                if (logical_operator->getLeft()){ //if
        //                                left child is valid (has to be by
        //                                convention!), add input data size
        //                                    t.push_back(logical_operator->getLeft()->getOutputResultSize());
        //                                    if (logical_operator->getRight())
        //                                    { //if right child is valid (not
        //                                    null), add input data size for it
        //                                    as well
        //                                        t.push_back(logical_operator->getRight()->getOutputResultSize());
        //                                    }
        //                                }

        OperatorSpecification op_spec(
            logical_operator->getOperationName(), t,
            // parameters are the same, because in the query processing engine,
            // we model copy operations explicitely, so the copy cost have to be
            // zero
            hype::PD_Memory_0,   // input data is in CPU RAM
            hype::PD_Memory_0);  // output data has to be stored in CPU RAM

        // DeviceConstraint dev_constr;

        //                                OperatorPtr op;
        //                                SchedulingDecision* sched_dec=NULL;
        //                                //we purposefully break the chain if
        //                                one algorithm is much faster than the
        //                                other
        //                                SchedulingDecision sched_dec_cpu_only
        //                                =
        //                                hype::Scheduler::instance().getOptimalAlgorithm(op_spec,
        //                                hype::CPU_ONLY);
        //                                SchedulingDecision sched_dec_gpu_only
        //                                =
        //                                hype::Scheduler::instance().getOptimalAlgorithm(op_spec,
        //                                logical_operator->getDeviceConstraint());
        //                                //hype::GPU_ONLY);
        //                                if(sched_dec_gpu_only.getDeviceSpecification().getDeviceType()==CPU){
        //                                    //not enough memory in GPU
        //                                        op =
        //                                        logical_operator->getPhysicalOperator(sched_dec_cpu_only);
        //                                        sched_dec=new
        //                                        SchedulingDecision(sched_dec_cpu_only);
        //                                        core::Scheduler::instance().getProcessingDevices().removeSchedulingDecision(sched_dec_gpu_only);
        //                                }else{
        //                                    if(sched_dec_gpu_only.getEstimatedExecutionTimeforAlgorithm().getTimeinNanoseconds()<sched_dec_cpu_only.getEstimatedExecutionTimeforAlgorithm().getTimeinNanoseconds()){
        //                                        op =
        //                                        logical_operator->getPhysicalOperator(sched_dec_gpu_only);
        //                                        sched_dec=new
        //                                        SchedulingDecision(sched_dec_gpu_only);
        //                                        core::Scheduler::instance().getProcessingDevices().removeSchedulingDecision(sched_dec_cpu_only);
        //                                    }else{
        //                                        op =
        //                                        logical_operator->getPhysicalOperator(sched_dec_cpu_only);
        //                                        sched_dec=new
        //                                        SchedulingDecision(sched_dec_cpu_only);
        //                                        core::Scheduler::instance().getProcessingDevices().removeSchedulingDecision(sched_dec_gpu_only);
        //                                    }
        //                                }

        SchedulingDecision sched_dec =
            hype::Scheduler::instance().getOptimalAlgorithm(
                op_spec, logical_operator->getDeviceConstraint());

        OperatorPtr op = logical_operator->getPhysicalOperator(sched_dec);
        op->setLogicalOperator(logical_operator);
        op->setOutputStream(logical_operator->getOutputStream());

#ifndef HYPE_ALTERNATE_QUERY_CHOPPING

        queryprocessing::ProcessingDevicePtr proc_dev =
            this->proc_devs_.getProcessingDevice(
                op->getDeviceSpecification().getProcessingDeviceID());
        queryprocessing::VirtualProcessingDevicePtr virt_proc_dev =
            this->proc_devs_.getVirtualProcessingDevice(
                op->getDeviceSpecification().getProcessingDeviceID());
        assert(proc_dev != NULL);
        assert(virt_proc_dev != NULL);

        if (!hype::core::quiet && hype::core::verbose && hype::core::debug)
          std::cout << "Scheduling Thread: schedule operator "
                    << logical_operator->toString(true) << " to device "
                    << (int)op->getDeviceSpecification().getProcessingDeviceID()
                    << " (" << util::getName(
                                   op->getDeviceSpecification().getDeviceType())
                    << ")" << std::endl;

        proc_dev->addOperator(op);
        virt_proc_dev->addRunningOperation(sched_dec);
#else
        DeviceOperatorQueuePtr dev_op_queue =
            DeviceOperatorQueues::instance().getDeviceOperatorQueue(
                sched_dec.getDeviceSpecification().getMemoryID());
        dev_op_queue->addOperator(op);
        queryprocessing::VirtualProcessingDevicePtr virt_proc_dev =
            this->proc_devs_.getVirtualProcessingDevice(
                op->getDeviceSpecification().getProcessingDeviceID());
        assert(virt_proc_dev != NULL);
        virt_proc_dev->addRunningOperation(sched_dec);
        if (!hype::core::quiet && hype::core::verbose && hype::core::debug)
          std::cout << "Scheduling Thread: schedule operator "
                    << logical_operator->toString(true) << " to device group "
                    << (int)op->getDeviceSpecification().getMemoryID()
                    << " (Devices attached to memory "
                    << (int)op->getDeviceSpecification().getMemoryID() << ")"
                    << std::endl;
#endif
      }
    }
  }
}

DeviceSpecification& Scheduler::getDeviceSpecificationforCopyType(
    const std::string& copy_type) {
  if (copy_type == "COPY_CPU_CP") {
    return this->dma_cpu_cp_;
  } else if (copy_type == "COPY_CP_CPU") {
    return this->dma_cp_cpu_;
  } else if (copy_type == "COPY_CP_CP") {
    return this->dma_cp_cpu_;
  } else {
    HYPE_FATAL_ERROR(std::string("INVALID COPY OPERATION TYPE: ") + copy_type,
                     std::cout);
    return this->dma_cp_cpu_;
  }
}

void Scheduler::insertCopyOperationAtBegin(
    SchedulingDecisionVectorPtr result_plan,
    const SchedulingDecision& first_operator) {
  if (first_operator.getDeviceSpecification().getMemoryID() !=
      hype::PD_Memory_0) {
    std::string copy_type = util::getCopyOperationType(
        hype::PD0,
        first_operator.getDeviceSpecification().getProcessingDeviceID());
    assert(!copy_type.empty());
    DeviceSpecification dev_spec = getDeviceSpecificationforCopyType(copy_type);
    AlgorithmPtr copy_alg_ptr =
        this->getAlgorithm(util::toInternalAlgName(copy_type, dev_spec));
    assert(copy_alg_ptr != NULL);
    // cout << "COPY Operator " << copy_alg_ptr.get() << endl;
    Tuple t;
    t.push_back(first_operator.getFeatureValues().front());
    SchedulingDecision copy_operator(
        *copy_alg_ptr, copy_alg_ptr->getEstimatedExecutionTime(t), t);
    result_plan->push_back(copy_operator);
  }
}

void Scheduler::insertCopyOperationAtBegin(
    InternalPhysicalPlan& result_plan,
    const InternalPhysicalOperator& first_operator) {
  if (first_operator.alg_ptr->getDeviceSpecification().getMemoryID() !=
      hype::PD_Memory_0) {
    std::string copy_type = util::getCopyOperationType(
        hype::PD0, first_operator.alg_ptr->getDeviceSpecification()
                       .getProcessingDeviceID());
    assert(!copy_type.empty());
    DeviceSpecification dev_spec = getDeviceSpecificationforCopyType(copy_type);
    AlgorithmPtr copy_alg_ptr =
        this->getAlgorithm(util::toInternalAlgName(copy_type, dev_spec));
    assert(copy_alg_ptr != NULL);
    // cout << "COPY Operator " << copy_alg_ptr.get() << endl;
    Tuple t;
    t.push_back(first_operator.feature_vector.front());
    InternalPhysicalOperator copy_operator(
        copy_alg_ptr, t,
        copy_alg_ptr->getEstimatedExecutionTime(t).getTimeinNanoseconds());
    result_plan.push_back(copy_operator);
  }
}

void Scheduler::insertCopyOperationAtEnd(
    SchedulingDecisionVectorPtr result_plan,
    const SchedulingDecision& last_operator) {
  if (last_operator.getDeviceSpecification().getMemoryID() !=
      hype::PD_Memory_0) {
    // cout << "Insert COPY Operation to transfer result to CPU" << endl;
    std::string copy_type = util::getCopyOperationType(
        last_operator.getDeviceSpecification().getProcessingDeviceID(),
        hype::PD0);  // assume that PD0 is CPU!
    assert(!copy_type.empty());
    DeviceSpecification dev_spec = getDeviceSpecificationforCopyType(copy_type);
    AlgorithmPtr copy_alg_ptr =
        this->getAlgorithm(util::toInternalAlgName(copy_type, dev_spec));
    assert(copy_alg_ptr != NULL);
    // cout << "COPY Operator " << copy_alg_ptr.get() << endl;
    Tuple t;
    t.push_back(last_operator.getFeatureValues().front());
    SchedulingDecision copy_operator(
        *copy_alg_ptr, copy_alg_ptr->getEstimatedExecutionTime(t), t);
    result_plan->push_back(copy_operator);
  }
}

void Scheduler::insertCopyOperationAtEnd(
    InternalPhysicalPlan& result_plan,
    const InternalPhysicalOperator& last_operator) {
  if (last_operator.alg_ptr->getDeviceSpecification().getMemoryID() !=
      hype::PD_Memory_0) {
    // cout << "Insert COPY Operation to transfer result to CPU" << endl;
    std::string copy_type = util::getCopyOperationType(
        last_operator.alg_ptr->getDeviceSpecification().getProcessingDeviceID(),
        hype::PD0);  // assume that PD0 is CPU!
    assert(!copy_type.empty());
    DeviceSpecification dev_spec = getDeviceSpecificationforCopyType(copy_type);
    AlgorithmPtr copy_alg_ptr =
        this->getAlgorithm(util::toInternalAlgName(copy_type, dev_spec));
    assert(copy_alg_ptr != NULL);
    // cout << "COPY Operator " << copy_alg_ptr.get() << endl;
    Tuple t;
    t.push_back(last_operator.feature_vector.front());
    InternalPhysicalOperator copy_operator(
        copy_alg_ptr, t,
        copy_alg_ptr->getEstimatedExecutionTime(t).getTimeinNanoseconds());
    // SchedulingDecision
    // copy_operator(*copy_alg_ptr,copy_alg_ptr->getEstimatedExecutionTime(t),t);
    result_plan.push_back(copy_operator);
  }
}

void Scheduler::insertCopyOperationInsidePlan(
    SchedulingDecisionVectorPtr result_plan,
    const SchedulingDecision& current_operator) {
  if (!util::isCopyOperation(result_plan->back())) {
    if (current_operator.getDeviceSpecification().getMemoryID() !=
        result_plan->back().getDeviceSpecification().getMemoryID()) {
      std::string copy_type = util::getCopyOperationType(
          result_plan->back().getDeviceSpecification().getProcessingDeviceID(),
          current_operator.getDeviceSpecification().getProcessingDeviceID());
      assert(!copy_type.empty());
      DeviceSpecification dev_spec =
          getDeviceSpecificationforCopyType(copy_type);
      AlgorithmPtr copy_alg_ptr =
          this->getAlgorithm(util::toInternalAlgName(copy_type, dev_spec));
      assert(copy_alg_ptr != NULL);
      Tuple t;

      t.push_back(current_operator.getFeatureValues().front());

      SchedulingDecision copy_operator(
          (*(copy_alg_ptr.get())), copy_alg_ptr->getEstimatedExecutionTime(t),
          t);
      cout << "insert " << copy_operator.getNameofChoosenAlgorithm()
           << " between " << result_plan->back().getNameofChoosenAlgorithm()
           << " and " << current_operator.getNameofChoosenAlgorithm() << endl;
      result_plan->push_back(copy_operator);
    }
  }
}

void Scheduler::insertCopyOperationInsidePlan(
    InternalPhysicalPlan& result_plan,
    const InternalPhysicalOperator& current_operator) {
  if (!util::isCopyOperation(result_plan.back().alg_ptr->getName())) {
    if (current_operator.alg_ptr->getDeviceSpecification().getMemoryID() !=
        result_plan.back().alg_ptr->getDeviceSpecification().getMemoryID()) {
      std::string copy_type = util::getCopyOperationType(
          result_plan.back()
              .alg_ptr->getDeviceSpecification()
              .getProcessingDeviceID(),
          current_operator.alg_ptr->getDeviceSpecification()
              .getProcessingDeviceID());
      assert(!copy_type.empty());
      DeviceSpecification dev_spec =
          getDeviceSpecificationforCopyType(copy_type);
      AlgorithmPtr copy_alg_ptr =
          this->getAlgorithm(util::toInternalAlgName(copy_type, dev_spec));
      assert(copy_alg_ptr != NULL);
      Tuple t;

      t.push_back(current_operator.feature_vector.front());
      InternalPhysicalOperator copy_operator(
          copy_alg_ptr, t,
          copy_alg_ptr->getEstimatedExecutionTime(t).getTimeinNanoseconds());
      // SchedulingDecision copy_operator( (*(copy_alg_ptr.get())),
      // copy_alg_ptr->getEstimatedExecutionTime(t),t);
      // cout << "insert " << copy_operator.getNameofChoosenAlgorithm() << "
      // between " << result_plan->back().getNameofChoosenAlgorithm() << " and "
      // << current_operator.getNameofChoosenAlgorithm()   << endl;
      result_plan.push_back(copy_operator);
    }
  }
}

/*################################################################################################*/
/*########################## END PRIVATE UTIL FUNCTIONS
 * ##########################################*/
/*################################################################################################*/

Scheduler::ProcessingDevices& Scheduler::getProcessingDevices() {
  return this->proc_devs_;
}

Scheduler::ProcessingDevices::ProcessingDevices()
    : virt_comp_devs_(), processing_device_mutex_() {}

Scheduler::ProcessingDevices::~ProcessingDevices() {
  //    Devices::iterator it;
  Devices::iterator it;

  for (it = virt_comp_devs_.begin(); it != virt_comp_devs_.end(); ++it) {
    PhysicalDevices::iterator phy_it;
    for (phy_it = it->second.second.begin(); phy_it != it->second.second.end();
         ++phy_it) {
      (*phy_it)->stop();
    }
  }
  //    for(it=phy_comp_devs_.begin();it!=phy_comp_devs_.end();++it){
  //        it->second->stop();
  //    }

  //    PhysicalDevices::iterator it;
  //    for(it=phy_comp_devs_.begin();it!=phy_comp_devs_.end();++it){
  //        it->second->stop();
  //    }
  // this->phy_comp_devs_
}

bool Scheduler::ProcessingDevices::addDevice(
    const DeviceSpecification& dev_spec) {
  boost::lock_guard<boost::mutex> lock(processing_device_mutex_);
  assert(!this->exists(dev_spec));

  //        Devices::iterator it;

  //        for(it=virt_comp_devs_.begin();it!=virt_comp_devs_.end();++it){
  //            PhysicalDevices::iterator phy_it;
  //            for(phy_it=(*it)->second.second.begin();phy_it!=(*it)->second.second.end();++phy_it){
  //                (*phy_it)->stop();
  //            }
  //        }

  //        Devices virt_comp_devs_ =
  //        core::Scheduler::instance().getProcessingDevices().getDevices();
  Devices::iterator it;

  it = virt_comp_devs_.find(dev_spec.getProcessingDeviceID());
  if (it == virt_comp_devs_.end()) {
    typedef std::vector<queryprocessing::ProcessingDevicePtr> PhysicalDevices;
    typedef std::pair<queryprocessing::VirtualProcessingDevicePtr,
                      PhysicalDevices>
        LogicalDevice;
    typedef std::map<ProcessingDeviceID, LogicalDevice> Devices;

    PhysicalDevices phy_devs;
    unsigned int number_of_worker_threads = 1;
    if (dev_spec.getDeviceType() == hype::CPU) {
      number_of_worker_threads = boost::thread::hardware_concurrency();
    } else if (dev_spec.getDeviceType() == hype::GPU) {
      number_of_worker_threads = 3;
    }
    for (unsigned int i = 0; i < number_of_worker_threads; ++i) {
      ProcessingDevicePtr phy_dev(new ProcessingDevice(dev_spec));
      phy_dev->start();
      // phy_devs.insert(phy_dev);
      phy_devs.push_back(phy_dev);
    }

    this->virt_comp_devs_.insert(std::make_pair(
        dev_spec.getProcessingDeviceID(),
        LogicalDevice(
            VirtualProcessingDevicePtr(new VirtualProcessingDevice(dev_spec)),
            phy_devs)));
  }

  //	this->virt_comp_devs_.insert(std::make_pair(dev_spec.getProcessingDeviceID(),
  //                                     VirtualProcessingDevicePtr( new
  //                                     VirtualProcessingDevice(dev_spec)))
  //				    );
  //        ProcessingDevicePtr phy_dev( new ProcessingDevice(dev_spec));
  //        this->phy_comp_devs_.insert(std::make_pair(dev_spec.getProcessingDeviceID(),
  //                                    phy_dev)
  //				    );
  //        phy_dev->start();
  return true;
}

bool Scheduler::ProcessingDevices::exists(
    const DeviceSpecification& dev_spec) const throw() {
  Devices::const_iterator cit;
  cit = virt_comp_devs_.find(dev_spec.getProcessingDeviceID());
  if (cit != virt_comp_devs_.end()) {
    return true;
  } else {
    return false;
  }
  //	for(cit=this->virt_comp_devs_.begin();cit!=this->virt_comp_devs_.end();++cit){
  ////			VirtualProcessingDevicePtr
  /// virtual_proc_dev_ptr=cit->second;
  ////			DeviceSpecification dev_spec_current =
  /// virtual_proc_dev_ptr->getDeviceSpecification();
  ////			if(dev_spec_current==dev_spec){
  ////					return true;
  ////			}
  //			if(cit->second->getDeviceSpecification()==dev_spec){
  //					return true;
  //			}
  //	}
  //	return false;
}

const Scheduler::ProcessingDevices::Devices&
Scheduler::ProcessingDevices::getDevices() const throw() {
  return this->virt_comp_devs_;
}

ProcessingDeviceID Scheduler::ProcessingDevices::getProcessingDeviceID(
    ProcessingDeviceMemoryID mem_id) const {
  boost::lock_guard<boost::mutex> lock(processing_device_mutex_);
  Devices::const_iterator it;
  for (it = virt_comp_devs_.begin(); it != virt_comp_devs_.end(); ++it) {
    // return processing device ID of first processor
    // that has the desired memory id
    if (mem_id == it->second.first->getDeviceSpecification().getMemoryID()) {
      return it->first;
    }
  }
  HYPE_FATAL_ERROR("Found no Processor with memory ID " << (int)mem_id << "!",
                   std::cerr);
  return hype::PD0;
}

const std::list<std::pair<std::string, double> >
Scheduler::ProcessingDevices::getTotalProcessorTimes() const {
  std::list<std::pair<std::string, double> > result;

  Devices::const_iterator it;

  for (it = virt_comp_devs_.begin(); it != virt_comp_devs_.end(); ++it) {
    PhysicalDevices::const_iterator phy_it;
    for (phy_it = it->second.second.begin(); phy_it != it->second.second.end();
         ++phy_it) {
      //                (*phy_it)->stop();
      std::string processor_name =
          util::getName(
              it->second.first->getDeviceSpecification().getDeviceType()) +
          boost::lexical_cast<std::string>(
              (int)(*phy_it)->getProcessingDeviceID());
      result.push_back(
          std::make_pair(processor_name, (*phy_it)->getTotalProcessingTime()));
    }
  }

  //    PhysicalDevices::const_iterator it;
  //    for(it=phy_comp_devs_.begin();it!=phy_comp_devs_.end();++it){
  //	Devices::const_iterator cit;
  //	cit=virt_comp_devs_.find(it->second->getProcessingDeviceID());
  //        VirtualProcessingDevicePtr virt_dev = cit->second;
  //        //this->getVirtualProcessingDevice(it->second->getProcessingDeviceID());
  //        assert(virt_dev!=NULL);
  //        std::string processor_name =
  //        util::getName(virt_dev->getDeviceSpecification().getDeviceType())+boost::lexical_cast<std::string>((int)it->second->getProcessingDeviceID());
  //        result.push_back(std::make_pair(processor_name,it->second->getTotalProcessingTime()));
  //    }
  return result;
}

VirtualProcessingDevicePtr
Scheduler::ProcessingDevices::getVirtualProcessingDevice(
    ProcessingDeviceID dev_id) {
  Devices::iterator it;

  it = virt_comp_devs_.find(dev_id);

  VirtualProcessingDevicePtr virt_dev_ptr;

  if (it == virt_comp_devs_.end()) {  // operation does not exist
    std::cout << "FATAL ERROR: Processing Device with ID " << dev_id
              << " does not exist!!!" << std::endl;
    exit(-1);
  }

  virt_dev_ptr = it->second.first;

  return virt_dev_ptr;
}

queryprocessing::ProcessingDevicePtr
Scheduler::ProcessingDevices::getProcessingDevice(ProcessingDeviceID dev_id) {
  core::Scheduler::ProcessingDevices::Devices virt_comp_devs_ =
      core::Scheduler::instance().getProcessingDevices().getDevices();
  core::Scheduler::ProcessingDevices::Devices::iterator it;

  it = virt_comp_devs_.find(dev_id);
  if (it != virt_comp_devs_.end()) {
    if (!it->second.second.empty()) return *(it->second.second.begin());
  }
  return ProcessingDevicePtr();

  //			PhysicalDevices::iterator it;
  //
  //			it=phy_comp_devs_.find(dev_id);
  //
  //			ProcessingDevicePtr phy_dev_ptr;
  //
  //			if(it==phy_comp_devs_.end()){ //operation does not exist
  //				std::cout << "FATAL ERROR: Processing Device
  // with
  // ID
  //"
  //<<
  // dev_id <<  " does not exist!!!" << std::endl;
  //				return queryprocessing::ProcessingDevicePtr();
  ////return
  // NULL
  // Pointer //exit(-1);
  //			}
  //
  //			phy_dev_ptr=it->second;
  //
  //			return phy_dev_ptr;
}

bool Scheduler::ProcessingDevices::addSchedulingDecision(
    const SchedulingDecision& sched_dec) {
  boost::lock_guard<boost::mutex> lock(processing_device_mutex_);
  ProcessingDeviceID dev_id =
      sched_dec.getDeviceSpecification().getProcessingDeviceID();
  VirtualProcessingDevicePtr vir_proc_dev =
      this->getVirtualProcessingDevice(dev_id);
  if (!vir_proc_dev) {
    cout << "Error! Could not find Processing Device for Device ID '" << dev_id
         << "'" << endl;
    return false;
  }
  return vir_proc_dev->addRunningOperation(sched_dec);
}

bool Scheduler::ProcessingDevices::removeSchedulingDecision(
    const SchedulingDecision& sched_dec) {
  boost::lock_guard<boost::mutex> lock(processing_device_mutex_);
  ProcessingDeviceID dev_id =
      sched_dec.getDeviceSpecification().getProcessingDeviceID();
  VirtualProcessingDevicePtr vir_proc_dev =
      this->getVirtualProcessingDevice(dev_id);
  if (!vir_proc_dev) {
    cout << "Error! Could not find Processing Device for Device ID '" << dev_id
         << "'" << endl;
    return false;
  }
  return vir_proc_dev->removeFinishedOperation(sched_dec);
}

void Scheduler::ProcessingDevices::print() const throw() {
  boost::lock_guard<boost::mutex> lock(processing_device_mutex_);
  Devices::const_iterator cit;
  // PhysicalDevices::const_iterator cit_phy;
  // assert(virt_comp_devs_.size()==phy_comp_devs_.size());

  for (cit = virt_comp_devs_.begin(); cit != virt_comp_devs_.end(); ++cit) {
    cit->second.first->print();
    PhysicalDevices::const_iterator phy_it;
    for (phy_it = cit->second.second.begin();
         phy_it != cit->second.second.end(); ++phy_it) {
      std::cout << "\t"
                << "Total Processing Time: "
                << (*phy_it)->getTotalProcessingTime() / (1000 * 1000 * 1000)
                << "s" << std::endl;
    }
  }

  //						for(cit=
  // virt_comp_devs_.begin(),
  // cit_phy=phy_comp_devs_.begin();cit!=
  // virt_comp_devs_.end();++cit,++cit_phy){
  //							cit->second.first->print();
  //                                                        std::cout << "\t" <<
  //                                                        "Total Processing
  //                                                        Time: " <<
  //                                                        cit_phy->second->getTotalProcessingTime()/(1000*1000*1000)
  //                                                        << "s" << std::endl;
  //						}
}

}  // end namespace core
}  // end namespace hype
