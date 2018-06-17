
#include <util/get_name.hpp>

namespace hype {
namespace util {

const std::string getName(StatisticalMethod x) {
  const char* const statistical_method_names[] = {
      "Least Squares 1D", "Least Squares 2D", "KNN_Regression"};

  return std::string(statistical_method_names[x]);
};

const std::string getName(RecomputationHeuristic x) {
  const char* const recomputation_heuristic_names[] = {
      "Oneshot Recomputation", "Periodic Recomputation"};

  return std::string(recomputation_heuristic_names[x]);
  //
  //		struct RecomputationHeuristics{
  //		enum RecomputationHeuristic{
  //			No_Recomputation,
  //			Periodic
  //		};
  //	};
};

const std::string getName(OptimizationCriterion x) {
  const char* const optimization_criterion_names[] = {
      "Response Time",      "WaitingTimeAwareResponseTime", "Throughput",
      "Simple Round Robin", "ProbabilityBasedOutsourcing",  "Throughput2"};

  return std::string(optimization_criterion_names[x]);

  //	struct OptimizationCriterions{
  //		enum OptimizationCriterion{
  //			ResponseTime,
  //			WaitingTimeAwareResponseTime,
  //			Throughput,
  //			Simple_Round_Robin,
  //			ProbabilityBasedOutsourcing,
  //			Throughput2
  //		};
  //	};
};

const std::string getName(ProcessingDeviceType x) {
  //	enum ProcessingDeviceType{CPU,GPU,FPGA,NP,XEON_PHI,DMA};
  ////CPU,GPU,FPGA, Network Processor
  const char* const names[] = {"CPU", "GPU", "FPGA", "NP", "XEON_PHI", "DMA"};

  return std::string(names[x]);
}

const std::string getName(DeviceTypeConstraint x) {
  //	enum
  // DeviceTypeConstraint{ANY_DEVICE,CPU_ONLY,GPU_ONLY,FPGA_ONLY,NP_ONLY,XEON_PHI_ONLY};
  const char* const names[] = {"ANY_DEVICE", "CPU_ONLY", "GPU_ONLY",
                               "FPGA_ONLY",  "NP_ONLY",  "XEON_PHI_ONLY"};

  return std::string(names[x]);
}

//                typedef enum {
//                    GREEDY_HEURISTIC,   //Use Greedy Heuristic from Bress et
//                    al. 2012 (ADBIS 2012)
//                    BACKTRACKING,       //consider all possible plans
//                    TWO_COPY_HEURISTIC, //explores a limited optimization
//                    space according to Bress et al. 2012 (Control and
//                    Cybernetics Journal)
//                    CPU_CP_SEQUENCE_ALLOCATION //creates a plan for each
//                    (co-)processor
//                    } QueryOptimizationHeuristic;

const std::string getName(QueryOptimizationHeuristic x) {
  //	enum
  // DeviceTypeConstraint{ANY_DEVICE,CPU_ONLY,GPU_ONLY,FPGA_ONLY,NP_ONLY,XEON_PHI_ONLY};
  const char* const names[] = {"GREEDY_HEURISTIC",
                               "BACKTRACKING",
                               "TWO_COPY_HEURISTIC",
                               "CPU_CP_SEQUENCE_ALLOCATION",
                               "INTERACTIVE_USER_OPTIMIZATION",
                               "GREEDY_CHAINER_HEURISTIC",
                               "CRITICAL_PATH_HEURISTIC",
                               "BEST_EFFORT_GPU_HEURISTIC"};

  return std::string(names[x]);
}

}  // end namespace util
}  // end namespace hype
