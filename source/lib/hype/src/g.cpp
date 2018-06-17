
#include <stdlib.h>
#include <core/scheduler.hpp>
#include <plugins/pluginloader.hpp>

#include <boost/thread.hpp>
#include <util/algorithm_name_conversion.hpp>

using namespace std;

namespace hype {
namespace core {

using namespace queryprocessing;

boost::mutex global_mutex;

Scheduler::Scheduler()
    : map_operationname_to_operation_(),
      map_statisticalmethodname_to_statisticalmethod_(),
      proc_devs_() {
  // is called once, because Scheduler is a Singelton
  PluginLoader::loadPlugins();
}

bool Scheduler::addAlgorithm(const AlgorithmSpecification& alg_spec,
                             const DeviceSpecification& dev_spec) {
  boost::lock_guard<boost::mutex> lock(global_mutex);

  std::string internal_algorithm_name =
      hype::util::toInternalAlgName(alg_spec.getAlgorithmName(), dev_spec);

  if (!this->proc_devs_.exists(dev_spec)) {
    if (!this->proc_devs_.addDevice(dev_spec)) {
      std::cout << "FATAL ERROR! Failed to add Processing Device while "
                   "creatign Algorithm '"
                << alg_spec.getAlgorithmName() << "'" << std::endl;
    }
  }

  std::string name_of_operation = alg_spec.getOperationName();

  std::map<std::string, boost::shared_ptr<Operation> >::iterator it;

  it = map_operationname_to_operation_.find(name_of_operation);

  boost::shared_ptr<Operation> op;

  if (it == map_operationname_to_operation_.end()) {  // operation does not
                                                      // exist
    op = boost::shared_ptr<Operation>(new Operation(name_of_operation));
    map_operationname_to_operation_[name_of_operation] = op;
  } else {
    op = it->second;  // boost::shared_ptr<Operation> (it->second);
  }
  return op->addAlgorithm(internal_algorithm_name, dev_spec,
                          alg_spec.getStatisticalMethodName(),
                          alg_spec.getRecomputationHeuristicName());
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

typedef std::vector<
    std::pair<const OperatorSpecification&, const DeviceConstraint&> >
    OperatorSequence;

enum QueryOptimizationHeuristic {
  GREEDY_HEURISTIC,
  BACKTRACKING,
  TWO_COPY_HEURISTIC
};

typedef std::vector<SchedulingDecision> SchedulingDecisionVector;
typedef boost::shared_ptr<SchedulingDecisionVector> SchedulingDecisionVectorPtr;

const SchedulingDecisionVectorPtr Scheduler::getOptimalAlgorithm(
    const OperatorSequence& op_seq, QueryOptimizationHeuristic heuristic) {
  SchedulingDecisionVectorPtr result_plan(op_seq.size());

  if (heuristic == GREEDY_HEURISTIC) {
    for (unsigned int i = 0; i < op_seq.size(); ++i) {
      result_plan[i] =
          this->getOptimalAlgorithm(op_seq[i].first, op_seq[i].second);
    }
    return result_plan;
  } else if (heuristic == BACKTRACKING) {
    // stores for each operators the number of possible algorithms to choose
    // from
    vector<int> number_of_algorithms(op_seq.size());
    for (unsigned int i = 0; i < op_seq.size(); ++i) {
      this->map_operationname_to_operation_.find()
    }
  }
}

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

  this->proc_devs_.removeSchedulingDecision(sched_dec);

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
  cout << "HyPE Status:" << endl;
  MapNameToOperation::iterator it;
  for (it = map_operationname_to_operation_.begin();
       it != map_operationname_to_operation_.end(); ++it) {
    std::cout << "Operation: '" << it->second->getName() << "'" << std::endl;
    std::vector<AlgorithmPtr> algs = it->second->getAlgorithms();
    for (unsigned int i = 0; i < algs.size(); ++i) {
      std::cout << "\t" << algs[i]->getName() << endl;
    }
  }
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

Scheduler::ProcessingDevices& Scheduler::getProcessingDevices() {
  return this->proc_devs_;
}

Scheduler::ProcessingDevices::ProcessingDevices()
    : virt_comp_devs_(), phy_comp_devs_() {}

bool Scheduler::ProcessingDevices::addDevice(
    const DeviceSpecification& dev_spec) {
  assert(!this->exists(dev_spec));
  this->virt_comp_devs_.insert(std::make_pair(
      dev_spec.getProcessingDeviceID(),
      VirtualProcessingDevicePtr(new VirtualProcessingDevice(dev_spec))));
  this->phy_comp_devs_.insert(std::make_pair(
      dev_spec.getProcessingDeviceID(),
      ProcessingDevicePtr(
          new ProcessingDevice(dev_spec.getProcessingDeviceID()))));
  return true;
}

bool Scheduler::ProcessingDevices::exists(
    const DeviceSpecification& dev_spec) const throw() {
  Devices::const_iterator cit;
  for (cit = this->virt_comp_devs_.begin(); cit != this->virt_comp_devs_.end();
       ++cit) {
    //			VirtualProcessingDevicePtr
    // virtual_proc_dev_ptr=cit->second;
    //			DeviceSpecification dev_spec_current =
    // virtual_proc_dev_ptr->getDeviceSpecification();
    //			if(dev_spec_current==dev_spec){
    //					return true;
    //			}
    if (cit->second->getDeviceSpecification() == dev_spec) {
      return true;
    }
  }
  return false;
}

const Scheduler::ProcessingDevices::Devices&
Scheduler::ProcessingDevices::getDevices() const throw() {
  return this->virt_comp_devs_;
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

  virt_dev_ptr = it->second;

  return virt_dev_ptr;
}

queryprocessing::ProcessingDevicePtr
Scheduler::ProcessingDevices::getProcessingDevice(ProcessingDeviceID dev_id) {
  PhysicalDevices::iterator it;

  it = phy_comp_devs_.find(dev_id);

  ProcessingDevicePtr phy_dev_ptr;

  if (it == phy_comp_devs_.end()) {  // operation does not exist
    std::cout << "FATAL ERROR: Processing Device with ID " << dev_id
              << " does not exist!!!" << std::endl;
    return queryprocessing::ProcessingDevicePtr();  // return NULL Pointer
                                                    // //exit(-1);
  }

  phy_dev_ptr = it->second;

  return phy_dev_ptr;
}

bool Scheduler::ProcessingDevices::addSchedulingDecision(
    const SchedulingDecision& sched_dec) {
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
  Devices::const_iterator cit;
  for (cit = virt_comp_devs_.begin(); cit != virt_comp_devs_.end(); ++cit) {
    cit->second->print();
  }
}

}  // end namespace core
}  // end namespace hype
