// default includes
#include <limits>

#include <config/global_definitions.hpp>

#include <core/operation.hpp>
#include <core/scheduler.hpp>

#include <plugins/optimization_criterias/response_time_advanced.hpp>

#include <query_processing/processing_device.hpp>
#include <util/utility_functions.hpp>

#include <util/get_name.hpp>

#ifdef DUMP_ESTIMATIONS
#include <fstream>
#endif

#ifdef HYPE_USE_MEMORY_COST_MODELS
#include <core/device_memory.hpp>
#endif

//#define TIMESTAMP_BASED_LOAD_ADAPTION
//#define PROBABILITY_BASED_LOAD_ADAPTION
//#define LOAD_MODIFICATOR_BASED_LOAD_ADAPTION

using namespace std;

namespace hype {
namespace core {

WaitingTimeAwareResponseTime::WaitingTimeAwareResponseTime(
    const std::string& name_of_operation)
    : OptimizationCriterion_Internal(
          std::string("WaitingTimeAwareResponseTime"), name_of_operation) {
  // OptimizationCriterionFactorySingleton::Instance().Register("Response
  // Time",&WaitingTimeAwareResponseTime::create);
}

const SchedulingDecision
WaitingTimeAwareResponseTime::getOptimalAlgorithm_internal(
    const Tuple& input_values, Operation& op, DeviceTypeConstraint dev_constr) {
  // idea: choose algorithm with minimal finishing time (including expected
  // waiting time on devices!)

  // assert(dev_constr==hype::ANY_DEVICE);

  std::vector<AlgorithmPtr> alg_ptrs = op.getAlgorithms();

  core::Scheduler& scheduler = core::Scheduler::instance();

  const core::Scheduler::ProcessingDevices::Devices& processing_devices =
      scheduler.getProcessingDevices().getDevices();

  // compute, how long all processing devices need to finish their jobs
  // std::vector<double>
  // estimated_finishing_times_pro_dev(processing_devices.size()); //finishing
  // times for processing devices
  std::map<unsigned int, double> estimated_finishing_times_pro_dev;
  core::Scheduler::ProcessingDevices::Devices::const_iterator cit;

  if (!quiet) scheduler.getProcessingDevices().print();

  for (cit = processing_devices.begin(); cit != processing_devices.end();
       ++cit) {
    // check wheter the ProcessingDevice ID fits in the array (we assume the
    // device ids have to be defined consecutively)
    // assert(cit->first<estimated_finishing_times_pro_dev.size());
    estimated_finishing_times_pro_dev[cit->first] =
        cit->second.first->getEstimatedFinishingTime();
  }
  if (!quiet) {
    cout << "Estimated Times per Processing Device: " << endl;
    for (unsigned int i = 0; i < estimated_finishing_times_pro_dev.size();
         ++i) {
      cout << "PD ID: " << i << " Estimated Finishing Time: "
           << estimated_finishing_times_pro_dev[i] << endl;
    }
  }

  // compute how long algorithms would take to terminate
  std::vector<double> estimated_finishing_times_of_algorithms(alg_ptrs.size());

  for (unsigned int i = 0; i < estimated_finishing_times_of_algorithms.size();
       ++i) {
    estimated_finishing_times_of_algorithms[i] =
        alg_ptrs[i]
            ->getEstimatedExecutionTime(input_values)
            .getTimeinNanoseconds();
    unsigned int array_index =
        alg_ptrs[i]->getDeviceSpecification().getProcessingDeviceID();
    estimated_finishing_times_of_algorithms[i] +=
        estimated_finishing_times_pro_dev[array_index];
  }

  if (!quiet) {
    cout << "Estimated Times per Algorithm: " << endl;
    for (unsigned int i = 0; i < estimated_finishing_times_of_algorithms.size();
         ++i) {
      cout << "PD ID: " << i << " Estimated Finishing Time: "
           << estimated_finishing_times_of_algorithms[0] << endl;
    }
  }
#ifndef HYPE_USE_MEMORY_COST_MODELS
  AlgorithmPtr optimal_algorithm_ptr;
  double min_time = std::numeric_limits<double>::max();
  // for(unsigned int i=0;i<estimated_finishing_times_of_algorithms.size();++i){
  for (int i = estimated_finishing_times_of_algorithms.size() - 1; i >= 0;
       --i) {
    if (util::satisfiesDeviceConstraint(alg_ptrs[i]->getDeviceSpecification(),
                                        dev_constr)) {
      if (min_time > estimated_finishing_times_of_algorithms[i]) {
        min_time = estimated_finishing_times_of_algorithms[i];
        optimal_algorithm_ptr = alg_ptrs[i];
      }
    }
  }
  if (!optimal_algorithm_ptr) {
    // if the optimizer wanted to force a co-processor, but
    // the co-processor has not enough memory, we fall back
    // to use the CPU
    // this violates the device constrained, but its the
    // best we can do in such a situation
    if (!quiet)
      HYPE_WARNING(
          "Unsatisfyable Device Constrained, because co-processor has not "
          "enough memory! Falling back to CPU operator for operation"
              << op.getName(),
          std::cout);
    // HYPE_FATAL_ERROR("No algorithm found that satisfies device
    // constraint!",std::cout);
    min_time = std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < estimated_finishing_times_of_algorithms.size();
         ++i) {
      if (util::satisfiesDeviceConstraint(alg_ptrs[i]->getDeviceSpecification(),
                                          CPU_ONLY)) {
        if (min_time > estimated_finishing_times_of_algorithms[i]) {
          min_time = estimated_finishing_times_of_algorithms[i];
          optimal_algorithm_ptr = alg_ptrs[i];
        }
      }
    }
  }
  if (!optimal_algorithm_ptr) {
    HYPE_FATAL_ERROR("No algorithm found that satisfies device constraint!",
                     std::cout);
  }

  return SchedulingDecision(
      *optimal_algorithm_ptr,
      optimal_algorithm_ptr->getEstimatedExecutionTime(input_values),
      input_values);

#else

  AlgorithmPtr optimal_algorithm_ptr;
  double min_time = std::numeric_limits<double>::max();
  for (unsigned int i = 0; i < estimated_finishing_times_of_algorithms.size();
       ++i) {
    if (util::satisfiesDeviceConstraint(alg_ptrs[i]->getDeviceSpecification(),
                                        dev_constr)) {
      if (min_time > estimated_finishing_times_of_algorithms[i]) {
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
          std::cout << "WTAR: Algorithm " << alg_ptrs[i]->getName() << " needs "
                    << double(required_memory) / (1024 * 1024) << "MB"
                    << " with " << double(available_memory) / (1024 * 1024)
                    << "MB available" << std::endl;
        }
        if (required_memory <= available_memory) {
          min_time = estimated_finishing_times_of_algorithms[i];
          optimal_algorithm_ptr = alg_ptrs[i];
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
  }
  MemoryChunkPtr mem_chunk;
  if (optimal_algorithm_ptr &&
      optimal_algorithm_ptr->getDeviceSpecification().getDeviceType() ==
          hype::GPU) {
    if (Runtime_Configuration::instance().getTrackMemoryUsage()) {
      // MemoryChunkPtr mem_chunk =
      // sched_dec.getDeviceSpecification().getMemoryID();

      DeviceMemoryPtr dev_mem = DeviceMemories::instance().getDeviceMemory(
          optimal_algorithm_ptr->getDeviceSpecification().getMemoryID());
      mem_chunk = dev_mem->allocateMemory(
          optimal_algorithm_ptr->getEstimatedRequiredMemoryCapacity(
              input_values));
    }
  }
  // assert(optimal_algorithm_ptr!=NULL);
  if (!optimal_algorithm_ptr) {
    // if the optimizer wanted to force a co-processor, but
    // the co-processor has not enough memory, we fall back
    // to use the CPU
    // this violates the device constrained, but its the
    // best we can do in such a situation
    if (!quiet)
      HYPE_WARNING(
          "Unsatisfyable Device Constrained, because co-processor has not "
          "enough memory! Falling back to CPU operator for operation"
              << op.getName(),
          std::cout);
    // HYPE_FATAL_ERROR("No algorithm found that satisfies device
    // constraint!",std::cout);
    min_time = std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < estimated_finishing_times_of_algorithms.size();
         ++i) {
      if (util::satisfiesDeviceConstraint(alg_ptrs[i]->getDeviceSpecification(),
                                          CPU_ONLY)) {
        if (min_time > estimated_finishing_times_of_algorithms[i]) {
          min_time = estimated_finishing_times_of_algorithms[i];
          optimal_algorithm_ptr = alg_ptrs[i];
        }
      }
    }
    // assert(optimal_algorithm_ptr!=NULL);
    if (!optimal_algorithm_ptr) {
      HYPE_FATAL_ERROR("No algorithm found that satisfies device constraint!",
                       std::cout);
    }
  }
  return SchedulingDecision(
      *optimal_algorithm_ptr,
      optimal_algorithm_ptr->getEstimatedExecutionTime(input_values),
      input_values, mem_chunk);

#endif
}

/*
        const SchedulingDecision
   WaitingTimeAwareResponseTime::getOptimalAlgorithm_internal(const Tuple&
   input_values, Operation& op, DeviceTypeConstraint dev_constr){
                //idea: choose algorithm with minimal finishing time (including
   expected waiting time on device!)

                assert(dev_constr==hype::ANY_DEVICE);

                std::vector<AlgorithmPtr> alg_ptrs = op.getAlgorithms();

                std::vector<double> estimated_finishing_times_(alg_ptrs.size());

                std::map<hype::DeviceSpecification,double>
   map_compute_device_to_estimated_waiting_time;

                map_compute_device_to_estimated_waiting_time.insert(
                        std::make_pair<hype::DeviceSpecification,double>(hype::CPU,hype::queryprocessing::getProcessingDevice(hype::CPU).getEstimatedTimeUntilIdle())
                );

                map_compute_device_to_estimated_waiting_time.insert(
                        std::make_pair<hype::DeviceSpecification,double>(hype::GPU,hype::queryprocessing::getProcessingDevice(hype::GPU).getEstimatedTimeUntilIdle())
                );

                for(unsigned int i=0;i<estimated_finishing_times_.size();++i){
                        //get estimated waiting time, until the operator can
   start execution
                        estimated_finishing_times_[i]=map_compute_device_to_estimated_waiting_time[alg_ptrs[i]->getComputeDevice()];
                        //debug output
                        if(!quiet && verbose)
                                cout << "estimated_finishing_time of Algorithm "
   << alg_ptrs[i]->getName() << ": " << estimated_finishing_times_[i] << "ns" <<
   endl;
                        //add the estiamted time, the operator itself needs for
   execution
                        estimated_finishing_times_[i]+=alg_ptrs[i]->getEstimatedExecutionTime(input_values).getTimeinNanoseconds();
                        //debug output
                        if(!quiet && verbose)
                                cout << "estimated_finishing_time of Algorithm "
   <<alg_ptrs[i]->getName() << " (including Algorithm execution time): " <<
   estimated_finishing_times_[i] << "ns" << endl;
                }

                AlgorithmPtr optimal_algorithm_ptr;
                double min_time=std::numeric_limits<double>::max();
                for(unsigned int i=0;i<estimated_finishing_times_.size();++i){
                        if(min_time>estimated_finishing_times_[i]){
                                min_time=estimated_finishing_times_[i];
                                optimal_algorithm_ptr=alg_ptrs[i];
                        }
                }
                assert(optimal_algorithm_ptr!=NULL);

                if(!quiet && verbose)
                        cout << "Choosing " << optimal_algorithm_ptr->getName()
   << " for operation " << op.getName() << endl;
                return
   SchedulingDecision(*optimal_algorithm_ptr,optimal_algorithm_ptr->getEstimatedExecutionTime(input_values),input_values);

        }
         */

}  // end namespace core
}  // end namespace hype
