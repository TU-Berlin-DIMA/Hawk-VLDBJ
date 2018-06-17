// default includes
#include <limits>

#include <config/configuration.hpp>
#include <config/global_definitions.hpp>

#include <core/operation.hpp>

#include <plugins/optimization_criterias/throughput.hpp>

//#define TIMESTAMP_BASED_LOAD_ADAPTION
//#define PROBABILITY_BASED_LOAD_ADAPTION
//#define LOAD_MODIFICATOR_BASED_LOAD_ADAPTION

using namespace std;

namespace hype {
namespace core {

// threshold based outsourcing
Throughput::Throughput(std::string name_of_operation)
    : OptimizationCriterion_Internal(std::string("Throughput"),
                                     name_of_operation) {
  // OptimizationCriterionFactorySingleton::Instance().Register("Response
  // Time",&Throughput::create);
}

const SchedulingDecision Throughput::getOptimalAlgorithm_internal(
    const Tuple& input_values, Operation& op, DeviceTypeConstraint dev_constr) {
  assert(dev_constr == hype::ANY_DEVICE);
  // double maximal_time_where_algorithm_was_not_choosen=2; //5;//2;
  // //Configuration::maximal_time_where_algorithm_was_not_choosen;

  /*! \todo severe problem: the logical counter of operation can jump, because
   * it is updated after algorithm termination,
   * so the decision for different algorithms can have the same logical
   * timestamp for an operation, which is FALSE.
   * The relevant information has to be updated directly AFTER THE DECISION,
   * BEFORE the algorithm terminates.*/

  std::vector<AlgorithmPtr> alg_ptrs = op.getAlgorithms();

  std::map<double, std::string> map_execution_times_to_algorithm_name =
      op.getEstimatedExecutionTimesforAlgorithms(input_values);
  if (map_execution_times_to_algorithm_name.empty()) {
    std::cout << "FATAL ERROR! no algorithm to choose from!!!" << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }
  std::map<double, std::string>::iterator it;
  double min_time = std::numeric_limits<double>::max();
  std::string fastest_algorithm_name;
  for (it = map_execution_times_to_algorithm_name.begin();
       it != map_execution_times_to_algorithm_name.end(); it++) {
    if (!quiet && verbose)
      cout << "Algorithm: '" << it->second
           << "'	Estimated Execution Time: " << it->first << endl;
    if (it->first < min_time) {
      min_time = it->first;
      fastest_algorithm_name = it->second;
    }
  }

  AlgorithmPtr optimal_algorithm_ptr;
  for (unsigned int i = 0; i < alg_ptrs.size(); i++) {
    if (alg_ptrs[i]->getName() == fastest_algorithm_name) {
      optimal_algorithm_ptr = alg_ptrs[i];
    }
  }
  assert(optimal_algorithm_ptr != NULL);

  for (unsigned int i = 0; i < alg_ptrs.size(); i++) {
    if (alg_ptrs[i]->getName() == fastest_algorithm_name) continue;
    if (!quiet && verbose && debug)
      cout << "[DEBUG] "
              "stemod::core::Throughput::getOptimalAlgorithm_internal(): "
           << alg_ptrs[i]->getName() << ": "
           << alg_ptrs[i]->getTimeOfLastExecution() << "+"
           << Runtime_Configuration::instance().getAlgorithmMaximalIdleTime()
           << "<" << op.getCurrentTimestamp() << endl;

    // FEATURE: Timestamp based load adaption (triggers retraining)
    // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // if algorithm was not executed for a long time
    // (Configuration::maximal_time_where_algorithm_was_not_choosen times),
    // retrain algorithm
    // if(alg_ptrs[i]->getTimeOfLastExecution()
    // +maximal_time_where_algorithm_was_not_choosen<op.getCurrentTimestamp()){
    if (alg_ptrs[i]->getTimeOfLastExecution() +
            Runtime_Configuration::instance().getAlgorithmMaximalIdleTime() <
        op.getCurrentTimestamp()) {
      if (!quiet)
        cout << "Operation execution number: " << op.getCurrentTimestamp()
             << endl;
      double estimation =
          std::max(double(0), alg_ptrs[i]
                                  ->getEstimatedExecutionTime(input_values)
                                  .getTimeinNanoseconds());
      double percenaged_slowdown = (estimation - min_time) / min_time;
      if (!quiet && verbose && debug)
        cout << "[DEBUG] "
                "stemod::core::Throughput::getOptimalAlgorithm_internal(): "
                "estimation: "
             << estimation << " minimal time: " << min_time
             << " with slowdown: " << percenaged_slowdown << endl;
      assert(!alg_ptrs[i]->inTrainingPhase());
      assert(estimation >= min_time);
      // assert(percenaged_slowdown>=0);
      if (!quiet && verbose && debug)
        cout << "[DEBUG] "
                "stemod::core::Throughput::getOptimalAlgorithm_internal(): max "
                "percentaged slowdown: "
             << Runtime_Configuration::instance()
                    .getMaximalSlowdownOfNonOptimalAlgorithm()
             << endl;
      // cout << "Condition met 1: " <<
      // bool(percenaged_slowdown<Runtime_Configuration::instance().getMaximalSlowdownOfNonOptimalAlgorithm())
      // << endl;
      // cout << "Condition met 2: " <<
      // bool(percenaged_slowdown>(-1*double(Runtime_Configuration::instance().getMaximalSlowdownOfNonOptimalAlgorithm())))
      //<< "	" <<
      // double(-1*double(Runtime_Configuration::instance().getMaximalSlowdownOfNonOptimalAlgorithm()))
      //<< endl;
      if (percenaged_slowdown <
              Runtime_Configuration::instance()
                  .getMaximalSlowdownOfNonOptimalAlgorithm() &&
          percenaged_slowdown >
              -1 * double(Runtime_Configuration::instance()
                              .getMaximalSlowdownOfNonOptimalAlgorithm())) {
        // if(!quiet)
        // if(!quiet)
        if (!quiet && verbose && debug)
          cout << "choose not optimal Algorithm: " << alg_ptrs[i]->getName()
               << " with slowdown: " << percenaged_slowdown << endl;
        // update timestamp of last execution, assign current operation
        // timestamp to algorithm and increment execution counter of operation
        // (ONLY in an optimization criterion!)
        alg_ptrs[i]->setTimeOfLastExecution(op.getNextTimestamp());
        // return Scheduling Decision
        return SchedulingDecision(
            *alg_ptrs[i],
            EstimatedTime(alg_ptrs[i]->getEstimatedExecutionTime(input_values)),
            input_values);
        // alg_ptrs[i]->retrain();
      }
    }

    if (alg_ptrs[i]->inRetrainingPhase()) {
      return SchedulingDecision(
          *alg_ptrs[i],
          EstimatedTime(alg_ptrs[i]->getEstimatedExecutionTime(input_values)),
          input_values);
    }  //*/
  }

  std::string name_of_algorithm =
      map_execution_times_to_algorithm_name[min_time];
  if (!quiet && verbose)
    cout << "Choosing " << name_of_algorithm << " for operation "
         << op.getName() << endl;
  /*update timestamp of last execution, assign current operation timestamp to
    algorithm
    and increment execution counter of operation (ONLY in an optimization
    criterion!)*/
  optimal_algorithm_ptr->setTimeOfLastExecution(op.getNextTimestamp());
  return SchedulingDecision(*optimal_algorithm_ptr, EstimatedTime(min_time),
                            input_values);
}

}  // end namespace core
}  // end namespace hype
