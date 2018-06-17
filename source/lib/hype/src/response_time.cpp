// default includes
#include <limits>

#include <config/global_definitions.hpp>

#include <core/operation.hpp>

#include <plugins/optimization_criterias/response_time.hpp>
#include <util/get_name.hpp>
#include <util/utility_functions.hpp>

#ifdef DUMP_ESTIMATIONS
#include <fstream>
#endif

//#define TIMESTAMP_BASED_LOAD_ADAPTION
//#define PROBABILITY_BASED_LOAD_ADAPTION
//#define LOAD_MODIFICATOR_BASED_LOAD_ADAPTION

using namespace std;

namespace hype {
namespace core {

ResponseTime::ResponseTime(const std::string& name_of_operation)
    : OptimizationCriterion_Internal(std::string("Response Time"),
                                     name_of_operation) {
  // OptimizationCriterionFactorySingleton::Instance().Register("Response
  // Time",&ResponseTime::create);
}

const SchedulingDecision ResponseTime::getOptimalAlgorithm_internal(
    const Tuple& input_values, Operation& op, DeviceTypeConstraint dev_constr) {
  /*
  std::vector<AlgorithmPtr> alg_ptrs = op.getAlgorithms();



  for(int i=0;i<alg_ptrs.size();i++){
          //if(!quiet && verbose)
                  cout << "Algorithm: " << alg_ptrs[i]->getName() << "   In
  Training Phase: " << alg_ptrs[i]->inTrainingPhase() << endl;
  #ifdef TIMESTAMP_BASED_LOAD_ADAPTION
          //FEATURE: Timestamp based load adaption (triggers retraining)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          //if algorithm was not executed for a long time
  (Configuration::maximal_time_where_algorithm_was_not_choosen times), retrain
  algorithm
          if(alg_ptrs[i]->getTimeOfLastExecution()+Configuration::maximal_time_where_algorithm_was_not_choosen<op.getCurrentTimestamp()){
                  cout << "Operation execution number: " <<
  op.getCurrentTimestamp() << endl;
                  alg_ptrs[i]->retrain();
          }
  #endif
  #ifdef LOAD_MODIFICATOR_BASED_LOAD_ADAPTION
          //FEATURE: Load Modification factor based load adaption
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          if(alg_ptrs[i]->getLoadChangeEstimator().getLoadModificator()>2
                  ||
  alg_ptrs[i]->getLoadChangeEstimator().getLoadModificator()<0.5f){ //execution
  times increased by factor of 2 -> sifnificant load change, retrain all
  algorithms?
                  cout << "Operation execution number: " <<
  op.getCurrentTimestamp() << "\tAlgorithm: " << alg_ptrs[i]->getName() << "\t"
                       << "Significant load change confirmed: " <<
  alg_ptrs[i]->getLoadChangeEstimator().getLoadModificator() << endl;
          }
  #endif
          //train algorithms in round robin manner
          if(alg_ptrs[i]->inTrainingPhase()){
                  return
  SchedulingDecision(alg_ptrs[i]->getName(),EstimatedTime(-1),input_values);
          }

          if(alg_ptrs[i]->inRetrainingPhase()){
                  return
  SchedulingDecision(alg_ptrs[i]->getName(),EstimatedTime(alg_ptrs[i]->getEstimatedExecutionTime(input_values)),input_values);
          }//
  }
  */
  // assert(dev_constr==stemod::ANY_DEVICE);
  if (!quiet && verbose && debug) {
    if (dev_constr == hype::CPU_ONLY) {
      cout << "only CPU algorithms allowed for operation'" << op.getName()
           << "' !" << endl;
    } else if (dev_constr == hype::GPU_ONLY) {
      cout << "only GPU algorithms allowed for operation'" << op.getName()
           << "' !" << endl;
    }
  }

  std::vector<AlgorithmPtr> alg_ptrs = op.getAlgorithms();

  double min_time = std::numeric_limits<double>::max();
  AlgorithmPtr optimal_algorithm;
  MemoryChunkPtr mem_chunk;
#ifndef HYPE_USE_MEMORY_COST_MODELS
  for (unsigned int i = 0; i < alg_ptrs.size(); ++i) {
    double t_est = alg_ptrs[i]
                       ->getEstimatedExecutionTime(input_values)
                       .getTimeinNanoseconds();
    if (util::satisfiesDeviceConstraint(alg_ptrs[i]->getDeviceSpecification(),
                                        dev_constr))
      if (t_est < min_time) {
        min_time = t_est;
        optimal_algorithm = alg_ptrs[i];
      }
  }
  if (optimal_algorithm == NULL) {
    // if the optimizer wanted to force a co-processor, but
    // the co-processor has not enough memory, we fall back
    // to use the CPU
    // this violates the device constrained, but its the
    // best we can do in such a situation
    HYPE_WARNING(
        "Unsatisfyable Device Constrained! Falling back to CPU operator for "
        "operation"
            << op.getName(),
        std::cout);
    min_time = std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < alg_ptrs.size(); ++i) {
      double t_est = alg_ptrs[i]
                         ->getEstimatedExecutionTime(input_values)
                         .getTimeinNanoseconds();
      if (util::satisfiesDeviceConstraint(alg_ptrs[i]->getDeviceSpecification(),
                                          hype::CPU_ONLY))
        if (t_est < min_time) {
          min_time = t_est;
          optimal_algorithm = alg_ptrs[i];
        }
    }
  }
  assert(optimal_algorithm != NULL);
#else
  for (unsigned int i = 0; i < alg_ptrs.size(); ++i) {
    double t_est = alg_ptrs[i]
                       ->getEstimatedExecutionTime(input_values)
                       .getTimeinNanoseconds();
    if (util::satisfiesDeviceConstraint(alg_ptrs[i]->getDeviceSpecification(),
                                        dev_constr)) {
      //					if(t_est<min_time) {
      //						min_time=t_est;
      //						optimal_algorithm=alg_ptrs[i];
      //					}
      size_t required_memory =
          alg_ptrs[i]->getEstimatedRequiredMemoryCapacity(input_values);
      // size_t available_memory =
      // alg_ptrs[i]->getDeviceSpecification().getAvailableMemoryCapacity();
      ProcessingDeviceMemoryID mem_id =
          alg_ptrs[i]->getDeviceSpecification().getMemoryID();
      DeviceMemoryPtr dev_mem =
          DeviceMemories::instance().getDeviceMemory(mem_id);
      size_t available_memory = dev_mem->getEstimatedFreeMemoryInBytes();
      if (!quiet && verbose && debug) {
        std::cout << "SRT: Algorithm " << alg_ptrs[i]->getName() << " needs "
                  << double(required_memory) / (1024 * 1024) << "MB"
                  << " with " << double(available_memory) / (1024 * 1024)
                  << "MB available" << std::endl;
      }
      if (required_memory <= available_memory) {
        if (t_est < min_time) {
          min_time = t_est;
          optimal_algorithm = alg_ptrs[i];
        }
      } else {
        HYPE_WARNING(
            "Unable to use processing device "
                << util::getName(
                       alg_ptrs[i]->getDeviceSpecification().getDeviceType())
                << (unsigned int)alg_ptrs[i]
                       ->getDeviceSpecification()
                       .getProcessingDeviceID()
                << ", because of insufficent free memory!",
            std::cout);
      }
    }
  }

  if (optimal_algorithm &&
      optimal_algorithm->getDeviceSpecification().getDeviceType() ==
          hype::GPU) {
    if (Runtime_Configuration::instance().getTrackMemoryUsage()) {
      // MemoryChunkPtr mem_chunk =
      // sched_dec.getDeviceSpecification().getMemoryID();
      DeviceMemoryPtr dev_mem = DeviceMemories::instance().getDeviceMemory(
          optimal_algorithm->getDeviceSpecification().getMemoryID());
      mem_chunk = dev_mem->allocateMemory(
          optimal_algorithm->getEstimatedRequiredMemoryCapacity(input_values));
    }
  }
  if (optimal_algorithm == NULL) {
    // if the optimizer wanted to force a co-processor, but
    // the co-processor has not enough memory, we fall back
    // to use the CPU
    // this violates the device constrained, but its the
    // best we can do in such a situation
    HYPE_WARNING(
        "Unsatisfyable Device Constrained, because co-processor has not enough "
        "memory! Falling back to CPU operator for operation"
            << op.getName(),
        std::cout);
    min_time = std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < alg_ptrs.size(); ++i) {
      double t_est = alg_ptrs[i]
                         ->getEstimatedExecutionTime(input_values)
                         .getTimeinNanoseconds();
      if (util::satisfiesDeviceConstraint(alg_ptrs[i]->getDeviceSpecification(),
                                          hype::CPU_ONLY))
        if (t_est < min_time) {
          min_time = t_est;
          optimal_algorithm = alg_ptrs[i];
        }
    }
  }
  assert(optimal_algorithm != NULL);
#endif

  // assert(optimal_algorithm!=NULL);
  if (!quiet && verbose)
    cout << "Choosing " << optimal_algorithm->getName() << " for operation "
         << op.getName() << endl;
  return SchedulingDecision(*optimal_algorithm, EstimatedTime(min_time),
                            input_values, mem_chunk);

  /*
  std::map<double,std::string> map_execution_times_to_algorithm_name =
  op.getEstimatedExecutionTimesforAlgorithms(input_values);
  if(map_execution_times_to_algorithm_name.empty()) {
          std::cout << "FATAL ERROR! no algorithm to choose from!!!" <<
  std::endl;
          std::cout << "File: " <<  __FILE__ << " Line: " << __LINE__ <<
  std::endl;
          exit(-1);
  }
  std::map<double,std::string>::iterator it;
  double min_time=std::numeric_limits<double>::max();
  cout << "AlgorithmMap: " << endl;
  for(it=map_execution_times_to_algorithm_name.begin();
  it!=map_execution_times_to_algorithm_name.end(); ++it) {
          cout << "Algorithm: '" << it->second << "'	Estimated Execution
  Time: " << it->first << endl;
  }

  for(it=map_execution_times_to_algorithm_name.begin();
  it!=map_execution_times_to_algorithm_name.end(); ++it) {
          if(!quiet && verbose)
                  cout << "Algorithm: '" << it->second << "'	Estimated
  Execution Time: " << it->first << endl;

          cout << "number_of algoritms" <<
  map_execution_times_to_algorithm_name.size() << endl;

          if(dev_constr==stemod::CPU_ONLY) {
                  cout << "only CPU algorithms allowed!" << endl;
          }

          if(dev_constr==stemod::GPU_ONLY) {
                  cout << "only GPU algorithms allowed!" << endl;
          }

          cout << "DEBUG: CPU_ONLY "<< bool((dev_constr==stemod::CPU_ONLY &&
  op.getAlgorithm(it->second)->getComputeDevice()==stemod::CPU))
               << " GPU_ONLY " << bool((dev_constr==stemod::GPU_ONLY &&
  op.getAlgorithm(it->second)->getComputeDevice()==stemod::GPU)) << endl;

          if( (dev_constr==stemod::CPU_ONLY &&
  op.getAlgorithm(it->second)->getComputeDevice()==stemod::CPU)
              || (dev_constr==stemod::GPU_ONLY &&
  op.getAlgorithm(it->second)->getComputeDevice()==stemod::GPU)
              || (dev_constr==stemod::ANY_DEVICE) )
                  if(it->first<min_time) {
                          min_time=it->first;
                  }

  #ifdef DUMP_ESTIMATIONS
          std::string path = "output/";
          path+=op.getName()+"/";
          path+=it->second+".estimations";
          fstream file(path.c_str(),fstream::out | fstream::app);
          for(unsigned int i=0; i<input_values.size(); ++i) {
                  file << input_values[i] << ",";
          }
          file << it->first << "," << it->second << endl;
          file.close();
  #endif
  }

  if(min_time==std::numeric_limits<double>::max()) {
          std::cout << "No suitable Algorithm found that fullfills specified
  contrained: " << std::endl;
          exit(0);
  }

  #ifdef PROBABILITY_BASED_LOAD_ADAPTION
  //FEATURE: probability based load adaption (pick optimal algorithm in X% of
  all cases and pick another one in the other cases) %%%%%%%%%%%%%%%%%%%%%%%%%%%
  //		if(rand()%5000==0){
  //
  for(it=map_execution_times_to_algorithm_name.begin();it!=map_execution_times_to_algorithm_name.end();it++){
  //				if(!quiet && verbose) cout << "Algorithm: '" <<
  it->second << "'	Estimated Execution Time: " << it->first << endl;
  //				cout << "Recoice: " << << endl;
  //				if(!(it->first<min_time)){
  //					min_time=it->first;
  //				}
  //			}
  //		}
  #endif
  //

  std::string name_of_algorithm =
  map_execution_times_to_algorithm_name[min_time];
  AlgorithmPtr pointer_to_choosen_algorithm=op.getAlgorithm(name_of_algorithm);
  assert(pointer_to_choosen_algorithm!=NULL);
  if(!quiet && verbose) cout << "Choosing " << name_of_algorithm << " for
  operation " << op.getName() << endl;
  return
  SchedulingDecision(*pointer_to_choosen_algorithm,EstimatedTime(min_time),input_values);
  */
}

}  // end namespace core
}  // end namespace hype
