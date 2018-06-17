
#pragma once

#include <core/scheduler.hpp>
#include <core/specification.hpp>
#include <iostream>

namespace hype {
  namespace util {

    void print(const core::AlgorithmSpecification& alg_spec, std::ostream& out);
    void print(const core::OperatorSpecification& op_spec, std::ostream& out);
    void print(const core::DeviceSpecification& dev_spec, std::ostream& out);
    void print(const core::DeviceConstraint& dev_constr, std::ostream& out);
    void print(const core::SchedulingDecision& dev_constr, std::ostream& out);

    void print(const core::OperatorSequence& op_seq, std::ostream& out);

    void print(core::SchedulingDecisionVectorPtr plan, std::ostream& out);

    void print(const core::InternalPhysicalOperator& phy_op, std::ostream& out);
    void print(const core::InternalPhysicalPlan& phy_plan, std::ostream& out);

  }  // end namespace util
  std::ostream& operator<<(std::ostream& out, const core::Tuple& pair);
  std::stringstream& operator<<(std::stringstream& out,
                                const core::Tuple& tuple);
}  // end namespace hype