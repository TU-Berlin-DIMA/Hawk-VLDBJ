

#include <cassert>
#include <core/scheduler.hpp>
#include <iostream>
#include <util/utility_functions.hpp>

namespace hype {
namespace util {
using namespace core;
bool satisfiesDeviceConstraint(const DeviceSpecification& dev_spec,
                               const DeviceConstraint& dev_constr) {
  if (dev_constr == hype::ANY_DEVICE ||
      (dev_constr == hype::CPU_ONLY && dev_spec.getDeviceType() == hype::CPU) ||
      (dev_constr == hype::GPU_ONLY && dev_spec.getDeviceType() == hype::GPU) ||
      (dev_constr == hype::FPGA_ONLY &&
       dev_spec.getDeviceType() == hype::FPGA) ||
      (dev_constr == hype::XEON_PHI_ONLY &&
       dev_spec.getDeviceType() == hype::XEON_PHI) ||
      (dev_constr == hype::NP_ONLY && dev_spec.getDeviceType() == hype::NP)) {
    return true;
  } else {
    return false;
  }
}

DeviceTypeConstraint getDeviceConstraintForProcessingDeviceType(
    const ProcessingDeviceType& pt_type) {
  if (pt_type == hype::CPU) return hype::CPU_ONLY;
  if (pt_type == hype::GPU) return hype::GPU_ONLY;
  if (pt_type == hype::FPGA) return hype::FPGA_ONLY;
  if (pt_type == hype::XEON_PHI) return hype::XEON_PHI_ONLY;
  if (pt_type == hype::NP) return hype::NP_ONLY;
  HYPE_FATAL_ERROR("Unkown Processing Device Type!", std::cerr);
  return hype::ANY_DEVICE;
}

bool isChainBreaker(const std::string& operator_name) {
  if (operator_name == "AddConstantValueColumn" ||
      operator_name == "COPY_CP_CP" || operator_name == "CPU_COLUMN_SCAN" ||
      operator_name == "INVISIBLE_JOIN" || operator_name == "PROJECTION" ||
      operator_name == "ColumnComparatorOperation"
      //                || operator_name=="COMPLEX_SELECTION"
      || operator_name == "COPY_CP_CPU" || operator_name == "CROSS_JOIN"
      //                || operator_name=="JOIN"
      || operator_name == "RENAME" || operator_name == "UDF"
      //                || operator_name=="ColumnAlgebraOperator"
      || operator_name == "COPY_CPU_CP" || operator_name == "SCAN" ||
      operator_name == "ColumnConstantOperator"
      // FIXME: workaround!!!!
      //                || operator_name=="PositionList_Operator"
      || operator_name == "COLUMN_CONSTANT_FILTER") {
    return true;
  } else {
    return false;
  }
  // CPU_ColumnAlgebra_Operator  GROUPBY PositionList_Operator  SELECTION
}

bool isCoprocessor(ProcessingDeviceID id) { return !isCPU(id); }
bool isCPU(ProcessingDeviceID id) {
  ProcessingDeviceType pd_type = core::Scheduler::instance()
                                     .getProcessingDevices()
                                     .getVirtualProcessingDevice(id)
                                     ->getDeviceSpecification()
                                     .getDeviceType();
  if (pd_type == CPU) {
    return true;
  } else {
    return false;
  }
}

hype::ProcessingDeviceMemoryID getMemoryID(ProcessingDeviceID id) {
  return core::Scheduler::instance()
      .getProcessingDevices()
      .getVirtualProcessingDevice(id)
      ->getDeviceSpecification()
      .getMemoryID();
}

ProcessingDeviceID getProcessingDeviceID(
    hype::ProcessingDeviceMemoryID mem_id) {
  return core::Scheduler::instance()
      .getProcessingDevices()
      .getProcessingDeviceID(mem_id);
}

std::string getCopyOperationType(ProcessingDeviceID id1,
                                 ProcessingDeviceID id2) {
  assert(id1 != id2);
  if (isCPU(id1) && isCPU(id2)) {
    return "";
  } else if (isCPU(id1) && isCoprocessor(id2)) {
    return "COPY_CPU_CP";
  } else if (isCoprocessor(id1) && isCPU(id2)) {
    return "COPY_CP_CPU";
  } else if (isCoprocessor(id1) && isCoprocessor(id2)) {
    return "COPY_CP_CP";
  } else {
    HYPE_FATAL_ERROR(
        "INVALID COMBINATION of PROCESSING DEVICE TYPES! MAYBE BUG IN "
        "hype::util::isCPU() OR hype::util::isCoprocessor() ?",
        std::cout);
  }
}

bool isCopyOperation(const core::SchedulingDecision& sched_dec) {
  std::string name = sched_dec.getNameofChoosenAlgorithm();
  if (name == "COPY_CPU_CP" || name == "COPY_CP_CPU" || name == "COPY_CP_CP") {
    return true;
  } else {
    return false;
  }
}

bool isCopyOperation(const std::string& name) {
  if (name == "COPY_CPU_CP" || name == "COPY_CP_CPU" || name == "COPY_CP_CP") {
    return true;
  } else {
    return false;
  }
}

}  // end namespace util
}  // end namespace hype
