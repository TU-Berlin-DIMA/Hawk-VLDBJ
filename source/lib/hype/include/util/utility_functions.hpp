
#pragma once

#include <iostream>
//#include <core/scheduler.hpp>
#include <core/scheduling_decision.hpp>
#include <core/specification.hpp>

namespace hype {
  namespace util {

    bool satisfiesDeviceConstraint(const core::DeviceSpecification& dev_spec,
                                   const core::DeviceConstraint& dev_constr);
    DeviceTypeConstraint getDeviceConstraintForProcessingDeviceType(
        const ProcessingDeviceType& pt_type);
    bool isChainBreaker(const std::string& operator_name);
    bool isCoprocessor(ProcessingDeviceID id);
    bool isCPU(ProcessingDeviceID id);
    hype::ProcessingDeviceMemoryID getMemoryID(ProcessingDeviceID id);
    ProcessingDeviceID getProcessingDeviceID(
        hype::ProcessingDeviceMemoryID mem_id);
    std::string getCopyOperationType(ProcessingDeviceID id1,
                                     ProcessingDeviceID id2);
    bool isCopyOperation(const core::SchedulingDecision& sched_dec);
    bool isCopyOperation(const std::string& name);
  }  // end namespace util
}  // end namespace hype