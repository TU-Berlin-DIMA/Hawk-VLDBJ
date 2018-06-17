/*
//Statistical Methods
#include <plugins/statistical_methods/least_squares.hpp>
#include <plugins/statistical_methods/multi_dim_least_squares.hpp>
//Recomputation Heuristics
#include <plugins/recomputation_heuristics/oneshot_computation.hpp>
#include <plugins/recomputation_heuristics/periodic_recomputation.hpp>
//Optimization Criterias
#include <plugins/optimization_criterias/response_time.hpp>
#include <plugins/optimization_criterias/simple_round_robin_throughput.hpp>
#include <plugins/optimization_criterias/throughput.hpp>
#include <plugins/optimization_criterias/throughput2.hpp>
*/
namespace hype {
  namespace core {

    class PluginLoader {
     public:
      /*! \todo enum basiertes Interface bauen!*/
      static bool loadPlugins();
    };

  }  // end namespace core
}  // end namespace hype
