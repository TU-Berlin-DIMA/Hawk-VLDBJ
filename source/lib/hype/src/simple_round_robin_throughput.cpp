// default includes
#include <limits>

#include <config/global_definitions.hpp>

#include <core/operation.hpp>

#include <plugins/optimization_criterias/simple_round_robin_throughput.hpp>

//#define TIMESTAMP_BASED_LOAD_ADAPTION
//#define PROBABILITY_BASED_LOAD_ADAPTION
//#define LOAD_MODIFICATOR_BASED_LOAD_ADAPTION

using namespace std;

namespace hype {
namespace core {

std::map<std::string, unsigned int> map_algorithmname_to_number_of_executions;

SimpleRoundRobin::SimpleRoundRobin(std::string name_of_operation)
    : OptimizationCriterion_Internal(std::string("Simple Round Robin"),
                                     name_of_operation) {
  // OptimizationCriterionFactorySingleton::Instance().Register("Response
  // Time",&SimpleRoundRobin::create);
}

const SchedulingDecision SimpleRoundRobin::getOptimalAlgorithm_internal(
    const Tuple& input_values, Operation& op, DeviceTypeConstraint dev_constr) {
  assert(dev_constr == hype::ANY_DEVICE);

  std::vector<AlgorithmPtr> alg_ptrs = op.getAlgorithms();

  if (map_algorithmname_to_number_of_executions.empty()) {
    std::vector<AlgorithmPtr> algs = op.getAlgorithms();
    std::vector<AlgorithmPtr>::const_iterator it;
    for (it = algs.begin(); it != algs.end(); ++it) {
      map_algorithmname_to_number_of_executions[(*it)->getName()] =
          0;  // init map
    }
  }

  std::map<std::string, unsigned int>::iterator it;
  std::string alg_with_min_executions;
  unsigned int min_execution = std::numeric_limits<unsigned int>::max();
  for (it = map_algorithmname_to_number_of_executions.begin();
       it != map_algorithmname_to_number_of_executions.end(); ++it) {
    if (it->second < min_execution) {
      min_execution = it->second;
      alg_with_min_executions = it->first;
    }
  }

  AlgorithmPtr pointer_to_choosen_algorithm =
      op.getAlgorithm(alg_with_min_executions);
  assert(pointer_to_choosen_algorithm != NULL);
  // cout << "SimpleRoundRobin: Choosing: " << alg_with_min_executions << endl;
  map_algorithmname_to_number_of_executions[alg_with_min_executions]++;
  return SchedulingDecision(
      *pointer_to_choosen_algorithm,
      EstimatedTime(pointer_to_choosen_algorithm->getEstimatedExecutionTime(
          input_values)),
      input_values);
  /*
          uint64_t oldest_timestamp=std::numeric_limits<uint64_t>::max();


          std::string algorithm_name_with_oldest_timestamp;
          uint64_t oldest_timestamp=std::numeric_limits<uint64_t>::max();
//older -> smaller  //works for physical and logical clocks
          for(unsigned int i=0;i<alg_ptrs.size();i++){
                  if(!quiet && verbose) cout << "Algorithm: " <<
alg_ptrs[i]->getName() << "   In Training Phase: " <<
alg_ptrs[i]->inTrainingPhase() << endl;
                  if(alg_ptrs[i]->getTimeOfLastExecution()<oldest_timestamp){
                          oldest_timestamp=alg_ptrs[i]->getTimeOfLastExecution();
                          algorithm_name_with_oldest_timestamp=alg_ptrs[i]->getName();
                  }
          }

          std::map<double,std::string> map_execution_times_to_algorithm_name =
op.getEstimatedExecutionTimesforAlgorithms(input_values);
          if(map_execution_times_to_algorithm_name.empty()){
                   std::cout << "FATAL ERROR! no algorithm to choose from!!!" <<
std::endl;
                   std::cout << "File: " <<  __FILE__ << " Line: " << __LINE__
<< std::endl;
                   exit(-1);
          }

          std::map<double,std::string>::iterator it;
//		double min_time=std::numeric_limits<double>::max();
//		std::string fastest_algorithm_name;
          for(it=map_execution_times_to_algorithm_name.begin();it!=map_execution_times_to_algorithm_name.end();it++){
                  if(!quiet && verbose) cout << "Algorithm: '" << it->second <<
"'	Estimated Execution Time: " << it->first << endl;
                  if(it->second==algorithm_name_with_oldest_timestamp){
//CHANGED! object map added
                          if(!quiet && verbose) cout << "Choosing " <<
algorithm_name_with_oldest_timestamp << " for operation " << op.getName() <<
endl;
                          //cout << "Choosing " <<
algorithm_name_with_oldest_timestamp << " for operation " << op.getName() <<
endl;
                          return
SchedulingDecision(algorithm_name_with_oldest_timestamp,EstimatedTime(it->first),input_values);
                  }
          }
          cout << "FATAL ERROR! In SchedulingDecision
SimpleRoundRobin::getOptimalAlgorithm_internal(): unable to choose algorithm! in
file"
          << __FILE__ << ":" << __LINE__ << endl;
          exit(-1);
          return SchedulingDecision("",EstimatedTime(-1),Tuple());
          */

  //		////////////////////////////////7
  //
  //
  //
  //
  //		//%%%%%%%%%%%%%%%%%%%%%%
  //

  //
  //		for(unsigned int i=0;i<alg_ptrs.size();i++){
  //			if(!quiet && verbose) cout << "Algorithm: " <<
  // alg_ptrs[i]->getName() << "   In Training Phase: " <<
  // alg_ptrs[i]->inTrainingPhase() << endl;

  //				//train algorithms in round robin manner
  //			if(alg_ptrs[i]->inTrainingPhase()){
  //				return
  // SchedulingDecision(alg_ptrs[i]->getName(),EstimatedTime(-1),input_values);
  //			}
  //		}
  //
  //		for(unsigned int i=0;i<alg_ptrs.size();i++){
  //			if(alg_ptrs[i]->getName()==fastest_algorithm_name){
  //				object_map[fastest_algorithm_name]+=min_time;
  ////CHANGED!
  // object map added
  //				continue;
  //			}
  //			//FEATURE: Timestamp based load adaption (triggers
  // retraining)
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  //			//if algorithm was not executed for a long time
  //(Configuration::maximal_time_where_algorithm_was_not_choosen times), retrain
  // algorithm
  //			if(alg_ptrs[i]->getTimeOfLastExecution()+maximal_time_where_algorithm_was_not_choosen<op.getCurrentTimestamp()){
  //				if(!quiet) cout << "Operation execution number:
  //"
  //<<
  // op.getCurrentTimestamp() << endl;
  //				double
  // estimation=std::max(double(0),alg_ptrs[i]->getEstimatedExecutionTime(input_values).getTimeinNanoseconds());
  //				double
  // percenaged_slowdown=(estimation-min_time)/min_time;
  //				if(!quiet) cout << "[DEBUG] estimation: " <<
  // estimation
  //<<
  //"
  // minimal time: "<< min_time  << " with slowdown: " << percenaged_slowdown <<
  // endl;
  //				assert(!alg_ptrs[i]->inTrainingPhase());
  //				assert(estimation>=min_time);
  //				//assert(percenaged_slowdown>=0);
  //				if(percenaged_slowdown<5.0*2 &&
  // percenaged_slowdown>-5.0*2){
  //					//if(!quiet)
  //					if(!quiet) cout << "choose not optimal
  // Algorithm:
  //"
  //<<
  // alg_ptrs[i]->getName() << " with slowdown: " << percenaged_slowdown <<
  // endl;
  //					alg_ptrs[i]->retrain();
  //				}
  //			}

  //			/*! \todo is this important?*/
  //			if(alg_ptrs[i]->inRetrainingPhase()){
  //				return
  // SchedulingDecision(alg_ptrs[i]->getName(),EstimatedTime(alg_ptrs[i]->getEstimatedExecutionTime(input_values)),input_values);
  //			}//*/
  //		}

  //
  //		std::string name_of_algorithm =
  // map_execution_times_to_algorithm_name[min_time];
  //		if(!quiet && verbose) cout << "Choosing " << name_of_algorithm
  //<<
  //"
  // for operation " << op.getName() << endl;
  //		return
  // SchedulingDecision(name_of_algorithm,EstimatedTime(min_time),input_values);
}

}  // end namespace core
}  // end namespace hype
