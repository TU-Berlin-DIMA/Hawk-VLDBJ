
#include <core/specification.hpp>
#include <util/get_name.hpp>

namespace hype {
namespace core {

using namespace util;

ProcessingDeviceMemoryID getMemory(ProcessingDeviceID) {
  return ProcessingDeviceMemoryID();
}

std::vector<ProcessingDeviceID> getProcessingDevices(ProcessingDeviceMemoryID) {
  return std::vector<ProcessingDeviceID>();
}

/************** Algorithm Specification **********/

AlgorithmSpecification::AlgorithmSpecification(
    const std::string& alg_name, const std::string& op_name,
    StatisticalMethod stat_meth, RecomputationHeuristic recomp_heur,
    OptimizationCriterion opt_crit)
    : alg_name_(alg_name),
      op_name_(op_name),
      stat_meth_(stat_meth),
      recomp_heur_(recomp_heur),
      opt_crit_(opt_crit) {}

const std::string& AlgorithmSpecification::getAlgorithmName() const throw() {
  return this->alg_name_;
}

const std::string& AlgorithmSpecification::getOperationName() const throw() {
  return this->op_name_;
}

const std::string AlgorithmSpecification::getOptimizationCriterionName() const
    throw() {
  return getName(this->opt_crit_);
}

const std::string AlgorithmSpecification::getRecomputationHeuristicName() const
    throw() {
  return getName(this->recomp_heur_);
}

const std::string AlgorithmSpecification::getStatisticalMethodName() const
    throw() {
  return getName(this->stat_meth_);
}

/************** Operator Specification **********/

OperatorSpecification::OperatorSpecification(
    const std::string& operator_name, const Tuple& feature_vector,
    ProcessingDeviceMemoryID location_of_input_data,
    ProcessingDeviceMemoryID location_for_output_data)
    : operator_name_(operator_name),
      feature_vector_(feature_vector),
      location_of_input_data_(location_of_input_data),
      location_for_output_data_(location_for_output_data) {}

const Tuple& OperatorSpecification::getFeatureVector() const throw() {
  return this->feature_vector_;
}

ProcessingDeviceMemoryID OperatorSpecification::getMemoryLocation() const
    throw() {
  return this->location_of_input_data_;
}

const std::string& OperatorSpecification::getOperatorName() const throw() {
  return this->operator_name_;
}

/************** Device Specification **********/

DeviceSpecification::DeviceSpecification(
    ProcessingDeviceID pd, ProcessingDeviceType pd_t,
    ProcessingDeviceMemoryID pd_m,
    AvailableMemoryFuncPtr get_avail_memory_func_ptr,
    size_t total_memory_capacity_in_byte)
    : pd_(pd),
      pd_t_(pd_t),
      pd_m_(pd_m),
      get_avail_memory_func_ptr_(get_avail_memory_func_ptr),
      total_memory_capacity_in_byte_(total_memory_capacity_in_byte) {}

ProcessingDeviceMemoryID DeviceSpecification::getMemoryID() const throw() {
  return this->pd_m_;
}

size_t DeviceSpecification::getAvailableMemoryCapacity() const {
  if (this->get_avail_memory_func_ptr_) {
    return (*this->get_avail_memory_func_ptr_)();
  } else {
    return 0;
  }
}

size_t DeviceSpecification::getTotalMemoryCapacity() const {
  return this->total_memory_capacity_in_byte_;
}

ProcessingDeviceID DeviceSpecification::getProcessingDeviceID() const throw() {
  return this->pd_;
}

ProcessingDeviceType DeviceSpecification::getDeviceType() const throw() {
  return this->pd_t_;
}

DeviceSpecification::operator ProcessingDeviceID() { return this->pd_; }
DeviceSpecification::operator ProcessingDeviceType() { return this->pd_t_; }
DeviceSpecification::operator ProcessingDeviceMemoryID() { return this->pd_m_; }

bool DeviceSpecification::operator==(
    const DeviceSpecification& dev_spec) const {
  return (this->pd_ == dev_spec.pd_) && (this->pd_t_ == dev_spec.pd_t_) &&
         (this->pd_m_ == dev_spec.pd_m_);
}

/************** Device Constraint **********/

DeviceConstraint::DeviceConstraint(DeviceTypeConstraint dev_constr,
                                   ProcessingDeviceMemoryID pd_mem_constr)
    : dev_constr_(dev_constr), pd_mem_constr_(pd_mem_constr) {}

DeviceTypeConstraint DeviceConstraint::getDeviceTypeConstraint() const {
  return this->dev_constr_;
}

DeviceConstraint::operator DeviceTypeConstraint() { return this->dev_constr_; }
DeviceConstraint::operator ProcessingDeviceMemoryID() {
  return this->pd_mem_constr_;
}

DeviceConstraint::operator DeviceTypeConstraint() const {
  return this->dev_constr_;
}
DeviceConstraint::operator ProcessingDeviceMemoryID() const {
  return this->pd_mem_constr_;
}

}  // end namespace core
}  // end namespace hype
