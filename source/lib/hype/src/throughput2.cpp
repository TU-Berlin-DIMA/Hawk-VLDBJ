// default includes
#include <limits>

#include <config/global_definitions.hpp>

#include <core/operation.hpp>

#include <plugins/optimization_criterias/throughput2.hpp>

//#define TIMESTAMP_BASED_LOAD_ADAPTION
//#define PROBABILITY_BASED_LOAD_ADAPTION
//#define LOAD_MODIFICATOR_BASED_LOAD_ADAPTION

using namespace std;

namespace hype {
namespace core {

// speichert zeit auf cpu und gpu (momentan bloß pro algorithmus, aber spielt
// keine Rolle wenn wir eh nur einen CPU und einen GPU algorithmus haben)
std::map<std::string, double> object_map;

Throughput2::Throughput2(std::string name_of_operation)
    : OptimizationCriterion_Internal(std::string("Throughput2"),
                                     name_of_operation) {
  // OptimizationCriterionFactorySingleton::Instance().Register("Response
  // Time",&Throughput2::create);
}

const SchedulingDecision Throughput2::getOptimalAlgorithm_internal(
    const Tuple& input_values, Operation& op, DeviceTypeConstraint dev_constr) {
  assert(dev_constr == hype::ANY_DEVICE);
  double maximal_time_where_algorithm_was_not_choosen = 2;  // 5;//2;
  // //Configuration::maximal_time_where_algorithm_was_not_choosen;

  std::vector<AlgorithmPtr> alg_ptrs = op.getAlgorithms();

  std::map<double, std::string> map_execution_times_to_algorithm_name =
      op.getEstimatedExecutionTimesforAlgorithms(input_values);
  if (map_execution_times_to_algorithm_name.empty()) {
    std::cout << "FATAL ERROR! no algorithm to choose from!!!" << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }

  ////////////////////////////////7

  //%%%%%%%%%%%%%%%%%%%%%%

  std::map<double, std::string>::iterator it;
  double min_time = std::numeric_limits<double>::max();
  std::string fastest_algorithm_name;
  for (it = map_execution_times_to_algorithm_name.begin();
       it != map_execution_times_to_algorithm_name.end(); it++) {
    if (!quiet && verbose)
      cout << "Algorithm: '" << it->second
           << "'	Estimated Execution Time: " << it->first << endl;
    if (it->first + object_map[it->second] <
        min_time) {  // CHANGED! object map added
      min_time = it->first;
      fastest_algorithm_name = it->second;
    }
  }

  for (unsigned int i = 0; i < alg_ptrs.size(); i++) {
    if (!quiet && verbose)
      cout << "Algorithm: " << alg_ptrs[i]->getName()
           << "   In Training Phase: " << alg_ptrs[i]->inTrainingPhase()
           << endl;

    // train algorithms in round robin manner
    if (alg_ptrs[i]->inTrainingPhase()) {
      return SchedulingDecision(*alg_ptrs[i], EstimatedTime(-1), input_values);
    }
  }

  for (unsigned int i = 0; i < alg_ptrs.size(); i++) {
    if (alg_ptrs[i]->getName() == fastest_algorithm_name) {
      object_map[fastest_algorithm_name] +=
          min_time;  // CHANGED! object map added
      continue;
    }
    // FEATURE: Timestamp based load adaption (triggers retraining)
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // if algorithm was not executed for a long time
    // (Configuration::maximal_time_where_algorithm_was_not_choosen times),
    // retrain algorithm
    if (alg_ptrs[i]->getTimeOfLastExecution() +
            maximal_time_where_algorithm_was_not_choosen <
        op.getCurrentTimestamp()) {
      if (!quiet)
        cout << "Operation execution number: " << op.getCurrentTimestamp()
             << endl;
      double estimation =
          std::max(double(0), alg_ptrs[i]
                                  ->getEstimatedExecutionTime(input_values)
                                  .getTimeinNanoseconds());
      double percenaged_slowdown = (estimation - min_time) / min_time;
      if (!quiet)
        cout << "[DEBUG] estimation: " << estimation
             << " minimal time: " << min_time
             << " with slowdown: " << percenaged_slowdown << endl;
      assert(!alg_ptrs[i]->inTrainingPhase());
      assert(estimation >= min_time);
      // assert(percenaged_slowdown>=0);
      if (percenaged_slowdown < 5.0 * 2 && percenaged_slowdown > -5.0 * 2) {
        // if(!quiet)
        if (!quiet)
          cout << "choose not optimal Algorithm: " << alg_ptrs[i]->getName()
               << " with slowdown: " << percenaged_slowdown << endl;
        alg_ptrs[i]->retrain();
      }
    }

    /*! \todo is this important?*/
    if (alg_ptrs[i]->inRetrainingPhase()) {
      return SchedulingDecision(
          *alg_ptrs[i],
          EstimatedTime(alg_ptrs[i]->getEstimatedExecutionTime(input_values)),
          input_values);
    }  //*/
  }

  std::string name_of_algorithm =
      map_execution_times_to_algorithm_name[min_time];
  AlgorithmPtr pointer_to_choosen_algorithm =
      op.getAlgorithm(name_of_algorithm);
  assert(pointer_to_choosen_algorithm != NULL);
  if (!quiet && verbose)
    cout << "Choosing " << name_of_algorithm << " for operation "
         << op.getName() << endl;
  return SchedulingDecision(*pointer_to_choosen_algorithm,
                            EstimatedTime(min_time), input_values);
}

}  // end namespace core
}  // end namespace hype
