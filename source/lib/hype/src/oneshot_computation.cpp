
#include <core/algorithm.hpp>
#include <plugins/recomputation_heuristics/oneshot_computation.hpp>

namespace hype {
namespace core {

Oneshotcomputation::Oneshotcomputation()
    : RecomputationHeuristic_Internal("Oneshot Recomputation"),
      once_(true),
      counter_(0) {
  // RecomputationHeuristicFactorySingleton::Instance().Register("Oneshot
  // Recomputation",&Oneshotcomputation::create);
}

// returns false, because the base class takes care of the initial training
// phase
bool Oneshotcomputation::internal_recompute(Algorithm& algorithm) {
  //		//std::cout << "RECOMPUTE???" << std::endl;
  //		//static unsigned int counter=0;
  //		if(once_){
  //		counter_++;
  //		if(counter_>=200){
  //			counter_=0;
  //			//std::cout << "RECOMPUTE APPROXIMATION FUNCTION!!!" <<
  // std::endl;
  //			once_=false;
  //			return true;
  //		}
  //		}
  // std::cout << algorithm.getName() << std::endl;

  return false;
}

}  // end namespace core
}  // end namespace hype
