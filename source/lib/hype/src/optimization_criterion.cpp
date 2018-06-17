#include <boost/shared_ptr.hpp>
#include <core/operation.hpp>
#include <core/optimization_criterion.hpp>
#include <core/scheduler.hpp>
#include <core/time_measurement.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <config/configuration.hpp>
#include <query_processing/processing_device.hpp>
#include <query_processing/virtual_processing_device.hpp>

#include "core/device_memory.hpp"

using namespace std;
namespace hype {
namespace core {
const SchedulingDecision
hype::core::OptimizationCriterion_Internal::getOptimalAlgorithm(
    const Tuple& input_values, Operation& op, DeviceTypeConstraint dev_constr) {
  // skip training mechanism!
  if (dev_constr != hype::ANY_DEVICE) {
    SchedulingDecision sched_dec =
        this->getOptimalAlgorithm_internal(input_values, op, dev_constr);
    core::Scheduler::instance().getProcessingDevices().addSchedulingDecision(
        sched_dec);
    return sched_dec;
  }

  std::vector<AlgorithmPtr> alg_ptrs = op.getAlgorithms();

  /*
  for(unsigned int i=0; i<alg_ptrs.size(); i++) {
                  if(alg_ptrs[i]->getNumberOfDecisionsforThisAlgorithm()==Runtime_Configuration::instance().getTrainingLength()){
                          while(alg_ptrs[i]->getNumberOfTerminatedExecutions()<alg_ptrs[i]->getNumberOfDecisionsforThisAlgorithm()){
                                  //wait until trainingphase is over?!
                          }
                  }

  }*/

  // executed once for round robin training
  if (map_algorithmname_to_number_of_executions_.empty()) {
    std::vector<AlgorithmPtr> algs = op.getAlgorithms();
    std::vector<AlgorithmPtr>::const_iterator it;
    for (it = algs.begin(); it != algs.end(); ++it) {
      map_algorithmname_to_number_of_executions_[(*it)->getName()] =
          0;  // init map
    }
  }

  bool all_algorithms_finished_training = true;
  for (unsigned int i = 0; i < alg_ptrs.size(); i++) {
    // cout << "Algorithm " << alg_ptrs[i]->getName() << "\tTraining: " <<
    // alg_ptrs[i]->inTrainingPhase()  << " \t Dec: " <<
    // alg_ptrs[i]->getNumberOfDecisionsforThisAlgorithm()
    //<< " \t Exec: " << alg_ptrs[i]->getNumberOfTerminatedExecutions() << endl;
    if (alg_ptrs[i]->inTrainingPhase()) {
      all_algorithms_finished_training = false;
    }
  }
  // cout << "all algorithms finsihed training: " 	<<
  // all_algorithms_finished_training << endl;
  if (!all_algorithms_finished_training) {
    std::map<std::string, unsigned int>::iterator it;
    std::string alg_with_min_executions;
    unsigned int min_execution = std::numeric_limits<unsigned int>::max();
    for (it = map_algorithmname_to_number_of_executions_.begin();
         it != map_algorithmname_to_number_of_executions_.end(); ++it) {
      if (it->second < min_execution) {
        min_execution = it->second;
        alg_with_min_executions = it->first;
      }
    }
    AlgorithmPtr pointer_to_choosen_algorithm =
        op.getAlgorithm(alg_with_min_executions);
    assert(pointer_to_choosen_algorithm != NULL);
    // cout << "Training: Choosing: " << alg_with_min_executions << endl;
    map_algorithmname_to_number_of_executions_[alg_with_min_executions]++;
    // return
    // SchedulingDecision(*pointer_to_choosen_algorithm,EstimatedTime(-1),input_values);
    SchedulingDecision sched_dec(*pointer_to_choosen_algorithm,
                                 EstimatedTime(-1), input_values);
    core::Scheduler::instance().getProcessingDevices().addSchedulingDecision(
        sched_dec);
    return sched_dec;
  }

  for (unsigned int i = 0; i < alg_ptrs.size(); i++) {
    if (!quiet && verbose)
      cout << "Algorithm: " << alg_ptrs[i]->getName()
           << "   In Training Phase: " << alg_ptrs[i]->inTrainingPhase()
           << endl;
#ifdef TIMESTAMP_BASED_LOAD_ADAPTION
    // FEATURE: Timestamp based load adaption (triggers retraining)
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // if algorithm was not executed for a long time
    // (Configuration::maximal_time_where_algorithm_was_not_choosen times),
    // retrain algorithm
    if (alg_ptrs[i]->getTimeOfLastExecution() +
            Configuration::maximal_time_where_algorithm_was_not_choosen <
        op.getCurrentTimestamp()) {
      cout << "Operation execution number: " << op.getCurrentTimestamp()
           << endl;
      alg_ptrs[i]->retrain();
    }
#endif
#ifdef LOAD_MODIFICATOR_BASED_LOAD_ADAPTION
    // FEATURE: Load Modification factor based load adaption
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (alg_ptrs[i]->getLoadChangeEstimator().getLoadModificator() > 2 ||
        alg_ptrs[i]->getLoadChangeEstimator().getLoadModificator() <
            0.5f) {  // execution times increased by factor of 2 -> sifnificant
                     // load change, retrain all algorithms?
      cout << "Operation execution number: " << op.getCurrentTimestamp()
           << "\tAlgorithm: " << alg_ptrs[i]->getName() << "\t"
           << "Significant load change confirmed: "
           << alg_ptrs[i]->getLoadChangeEstimator().getLoadModificator()
           << endl;
    }
#endif
    // train algorithms in round robin manner
    /*if(alg_ptrs[i]->inTrainingPhase()) {
            return
    SchedulingDecision(alg_ptrs[i]->getName(),EstimatedTime(-1),input_values);
    }*/

    if (alg_ptrs[i]->inRetrainingPhase()) {
      // return
      // SchedulingDecision(*alg_ptrs[i],EstimatedTime(alg_ptrs[i]->getEstimatedExecutionTime(input_values)),input_values);
      SchedulingDecision sched_dec(
          *alg_ptrs[i],
          EstimatedTime(alg_ptrs[i]->getEstimatedExecutionTime(input_values)),
          input_values);
      core::Scheduler::instance().getProcessingDevices().addSchedulingDecision(
          sched_dec);
      return sched_dec;
    }  //*/
  }

  SchedulingDecision sched_dec =
      this->getOptimalAlgorithm_internal(input_values, op, dev_constr);
  core::Scheduler::instance().getProcessingDevices().addSchedulingDecision(
      sched_dec);
  // TODO: ADD emory allocation book keeping
  //                         if(Runtime_Configuration::instance().getTrackMemoryUsage()){
  //                            //MemoryChunkPtr mem_chunk =
  //                            sched_dec.getDeviceSpecification().getMemoryID();
  //
  //                            DeviceMemoryPtr dev_mem =
  //                            DeviceMemories::instance().getDeviceMemory(sched_dec.getDeviceSpecification().getMemoryID());
  //                            MemoryChunkPtr mem_chunk =
  //                            dev_mem->allocateMemory(0);
  //
  //                         }
  return sched_dec;

  // return this->getOptimalAlgorithm_internal(input_values, op, dev_constr);
}
const std::string& OptimizationCriterion_Internal::getName() const {
  return name_of_optimization_criterion_;
}

OptimizationCriterion_Internal::OptimizationCriterion_Internal(
    const std::string& name_of_optimization_criterion,
    const std::string& name_of_operation)
    : map_algorithmname_to_number_of_executions_(),
      name_of_optimization_criterion_(name_of_optimization_criterion),
      name_of_operation_(name_of_operation) {}

OptimizationCriterion_Internal::~OptimizationCriterion_Internal() {}

boost::shared_ptr<OptimizationCriterion_Internal>
getNewOptimizationCriterionbyName(
    const std::string& name_of_optimization_criterion) {
  OptimizationCriterion_Internal* ptr =
      OptimizationCriterionFactorySingleton::Instance().CreateObject(
          name_of_optimization_criterion);  //.( 1, createProductNull );
  return boost::shared_ptr<OptimizationCriterion_Internal>(ptr);
}

OptimizationCriterionFactory&
OptimizationCriterionFactorySingleton::Instance() {
  static OptimizationCriterionFactory factory;
  return factory;
}
}  // end namespace core
}  // end namespace hype
