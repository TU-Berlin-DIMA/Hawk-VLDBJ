// default includes
#include <config/global_definitions.hpp>
#include <core/operation.hpp>
#include <functional>
#include <limits>
#include <plugins/optimization_criterias/probability_based_outsourcing.hpp>
#include <query_processing/processing_device.hpp>

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

ProbabilityBasedOutsourcing::ProbabilityBasedOutsourcing(
    const std::string& name_of_operation)
    : OptimizationCriterion_Internal(std::string("ProbabilityBasedOutsourcing"),
                                     name_of_operation),
      random_number_generator_() {
  // OptimizationCriterionFactorySingleton::Instance().Register("Response
  // Time",&ProbabilityBasedOutsourcing::create);
}

const SchedulingDecision
ProbabilityBasedOutsourcing::getOptimalAlgorithm_internal(
    const Tuple& input_values, Operation& op, DeviceTypeConstraint dev_constr) {
  // idea: compute a probability for each algorithm depending on its estimated
  // execution time

  // assert(dev_constr==hype::ANY_DEVICE);

  std::vector<AlgorithmPtr> alg_ptrs_unfiltered = op.getAlgorithms();
  std::vector<AlgorithmPtr> alg_ptrs;
  // pre filter algorithms, so only algorithms are considered that satisfy the
  // device constraint
  for (unsigned int i = 0; i < alg_ptrs_unfiltered.size(); i++) {
    if (util::satisfiesDeviceConstraint(
            alg_ptrs_unfiltered[i]->getDeviceSpecification(), dev_constr)) {
      alg_ptrs.push_back(alg_ptrs_unfiltered[i]);
    }
    //
  }
  if (alg_ptrs.empty()) {
    HYPE_FATAL_ERROR(
        "Unsatisfiable device constraint: " << util::getName(dev_constr),
        std::cerr);
  }

  std::vector<double> estimated_execution_times(alg_ptrs.size());
  std::vector<double> probabilities(alg_ptrs.size());

  std::map<DeviceSpecification, double>
      map_compute_device_to_estimated_waiting_time;

  for (unsigned int i = 0; i < alg_ptrs.size(); i++) {
    estimated_execution_times[i] = alg_ptrs[i]
                                       ->getEstimatedExecutionTime(input_values)
                                       .getTimeinNanoseconds();
  }

  double est_Execution_Time_SUM =
      std::accumulate(estimated_execution_times.begin(),
                      estimated_execution_times.end(), double(0));

  for (unsigned int i = 0; i < alg_ptrs.size(); i++) {
    probabilities[i] =
        1 - (estimated_execution_times[i] / est_Execution_Time_SUM);
  }

  if (!quiet && verbose && debug) {
    cout << "Probabilities: " << endl;
    for (unsigned int i = 0; i < alg_ptrs.size(); i++) {
      cout << alg_ptrs[i]->getName() << ":\t" << probabilities[i] << endl;
    }
  }

  boost::random::discrete_distribution<> dist(probabilities.begin(),
                                              probabilities.end());

  int selected_algorithm_index = dist(random_number_generator_);

  AlgorithmPtr optimal_algorithm_ptr = alg_ptrs[selected_algorithm_index];

  assert(optimal_algorithm_ptr != NULL);

  if (!quiet && verbose)
    cout << "Choosing " << optimal_algorithm_ptr->getName() << " for operation "
         << op.getName() << endl;
  return SchedulingDecision(
      *optimal_algorithm_ptr,
      EstimatedTime(estimated_execution_times[selected_algorithm_index]),
      input_values);

  /*
  map_compute_device_to_estimated_waiting_time.insert(
          std::make_pair<stemod::ComputeDevice,double>(stemod::CPU,stemod::queryprocessing::getProcessingDevice(stemod::CPU).getEstimatedTimeUntilIdle())
  );

  map_compute_device_to_estimated_waiting_time.insert(
          std::make_pair<stemod::ComputeDevice,double>(stemod::GPU,stemod::queryprocessing::getProcessingDevice(stemod::GPU).getEstimatedTimeUntilIdle())
  );

  for(unsigned int i=0;i<estimated_finishing_times_.size();++i){
          //get estimated waiting time, until the operator can start execution
          estimated_finishing_times_[i]=map_compute_device_to_estimated_waiting_time[alg_ptrs[i]->getComputeDevice()];
          //debug output
          if(!quiet && verbose)
                  cout << "estimated_finishing_time of Algorithm " <<
  alg_ptrs[i]->getName() << ": " << estimated_finishing_times_[i] << "ns" <<
  endl;
          //add the estiamted time, the operator itself needs for execution
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
          cout << "Choosing " << optimal_algorithm_ptr->getName() << " for
  operation " << op.getName() << endl;
  return
  SchedulingDecision(*optimal_algorithm_ptr,optimal_algorithm_ptr->getEstimatedExecutionTime(input_values),input_values);
  */
}

}  // end namespace core
}  // end namespace hype
