

#include <core/runtime_configuration.hpp>

namespace CoGaDB {

RuntimeConfiguration::RuntimeConfiguration()
    : path_to_database_("./data"),
      optimizer_name_("no_join_order_optimizer"),
      print_query_plan_(false),
      enable_profiling_(false),
      enable_query_chopping_(false),
      global_device_constraint_(hype::ANY_DEVICE),
      two_phase_physical_optimization_parallelization_mode(SERIAL),
      hybrid_optimizer_heuristic(hype::GREEDY_HEURISTIC),
      gpu_buffer_management_strategy_(LEAST_RECENTLY_USED),
      table_loader_mode_(LOAD_ALL_COLUMNS)  //(LOAD_NO_COLUMNS)
{}

RuntimeConfiguration& RuntimeConfiguration::instance() {
  static RuntimeConfiguration runtime_config;
  return runtime_config;
}

void RuntimeConfiguration::setPathToDatabase(const std::string& val) {
  path_to_database_ = val;
}

const std::string& RuntimeConfiguration::getPathToDatabase() {
  return path_to_database_;
}

void RuntimeConfiguration::setOptimizer(const std::string& new_optimizer_name) {
  optimizer_name_ = new_optimizer_name;
}
const std::string& RuntimeConfiguration::getOptimizer() {
  return optimizer_name_;
}

void RuntimeConfiguration::setQueryOptimizationHeuristic(
    hype::QueryOptimizationHeuristic opt_heu) {
  this->hybrid_optimizer_heuristic = opt_heu;
}
hype::QueryOptimizationHeuristic
RuntimeConfiguration::getQueryOptimizationHeuristic() const {
  return this->hybrid_optimizer_heuristic;
}

void RuntimeConfiguration::
    setParallelizationModeForTwoPhasePhysicalOptimizationQueryPlans(
        ParallelizationMode value) {
  this->two_phase_physical_optimization_parallelization_mode = value;
}
ParallelizationMode RuntimeConfiguration::
    getParallelizationModeForTwoPhasePhysicalOptimizationQueryPlans() const {
  return this->two_phase_physical_optimization_parallelization_mode;
}

GPUBufferManagementStrategy
RuntimeConfiguration::getGPUBufferManagementStrategy() const {
  return this->gpu_buffer_management_strategy_;
}
void RuntimeConfiguration::setGPUBufferManagementStrategy(
    GPUBufferManagementStrategy value) {
  this->gpu_buffer_management_strategy_ = value;
}

BufferManagementStrategy RuntimeConfiguration::getBufferManagementStrategy()
    const {
  return getGPUBufferManagementStrategy();
}
void RuntimeConfiguration::setBufferManagementStrategy(
    BufferManagementStrategy value) {
  return setGPUBufferManagementStrategy(value);
}

TableLoaderMode RuntimeConfiguration::getTableLoaderMode() const {
  return this->table_loader_mode_;
}

void RuntimeConfiguration::setTableLoaderMode(TableLoaderMode value) {
  this->table_loader_mode_ = value;
}

void RuntimeConfiguration::setPrintQueryPlan(bool value) {
  this->print_query_plan_ = value;
}
bool RuntimeConfiguration::getPrintQueryPlan() const {
  return this->print_query_plan_;
}
bool RuntimeConfiguration::getProfileQueries() const {
  return this->enable_profiling_;
}
void RuntimeConfiguration::setProfileQueries(bool new_value) {
  this->enable_profiling_ = new_value;
}
void RuntimeConfiguration::setGlobalDeviceConstraint(
    const hype::DeviceConstraint& value) {
  this->global_device_constraint_ = value;
}
hype::DeviceConstraint RuntimeConfiguration::getGlobalDeviceConstraint() const {
  return this->global_device_constraint_;
}

void RuntimeConfiguration::setQueryChoppingEnabled(bool value) {
  this->enable_query_chopping_ = value;
}

bool RuntimeConfiguration::isQueryChoppingEnabled() const {
  return this->enable_query_chopping_;
}
}  // end namespace CogaDB
