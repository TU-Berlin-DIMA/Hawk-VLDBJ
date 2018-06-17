

#include <util/get_name.hpp>
#include <util/print.hpp>

namespace hype {
namespace util {

using namespace std;
using namespace hype::core;

void print(const AlgorithmSpecification& alg_spec, std::ostream& out) {}
void print(const OperatorSpecification& op_spec, std::ostream& out) {}
void print(const DeviceSpecification& dev_spec, std::ostream& out) {}
void print(const DeviceConstraint& dev_constr, std::ostream& out) {}

void print(const SchedulingDecision& sched_dec, std::ostream& out) {
  out << sched_dec.getNameofChoosenAlgorithm() << " ("
      << sched_dec.getFeatureValues() << "; "
      << "ET: "
      << sched_dec.getEstimatedExecutionTimeforAlgorithm()
                 .getTimeinNanoseconds() /
             (1000 * 1000)
      << "ms; ";
  if (sched_dec.getDeviceSpecification().getProcessingDeviceID() == PD_DMA0 ||
      sched_dec.getDeviceSpecification().getProcessingDeviceID() == PD_DMA1) {
    out << "PD: "
        << util::getName(sched_dec.getDeviceSpecification().getDeviceType())
        << ((int)sched_dec.getDeviceSpecification().getProcessingDeviceID() -
            PD_DMA0)
        << ", ";
  } else {
    out << "PD: "
        << util::getName(sched_dec.getDeviceSpecification().getDeviceType())
        << (int)sched_dec.getDeviceSpecification().getProcessingDeviceID()
        << ", ";
  }
  if (sched_dec.getDeviceSpecification().getMemoryID() == PD_NO_Memory) {
    out << "NO_Memory";
  } else {
    out << "MemoryID" << sched_dec.getDeviceSpecification().getMemoryID();
  }
  out << ")" << std::endl;
}

void print(const OperatorSequence& op_seq, std::ostream& out) {
  out << "============================================================"
      << std::endl;
  out << "Operator Sequence:" << std::endl;
  for (unsigned int i = 0; i < op_seq.size(); ++i) {
    out << op_seq[i].first.getOperatorName() << " ("
        << op_seq[i].first.getFeatureVector()
        << "; Input Data MemoryID: " << (int)op_seq[i].first.getMemoryLocation()
        << "; " << getName(op_seq[i].second.getDeviceTypeConstraint()) << ")"
        << std::endl;
  }
  out << "============================================================"
      << std::endl;
}

void print(SchedulingDecisionVectorPtr plan, std::ostream& out) {
  out << "============================================================"
      << std::endl;
  out << "Physical Plan for Operator Sequence:" << std::endl;
  double total_cost = 0;
  for (unsigned int i = 0; i < plan->size(); i++) {
    out << (*plan)[i].getNameofChoosenAlgorithm() << " ("
        << (*plan)[i].getFeatureValues() << "; "
        << "ET: "
        << (*plan)[i]
                   .getEstimatedExecutionTimeforAlgorithm()
                   .getTimeinNanoseconds() /
               (1000 * 1000)
        << "ms; ";
    if ((*plan)[i].getDeviceSpecification().getProcessingDeviceID() ==
            PD_DMA0 ||
        (*plan)[i].getDeviceSpecification().getProcessingDeviceID() ==
            PD_DMA1) {
      out << "PD: "
          << util::getName((*plan)[i].getDeviceSpecification().getDeviceType())
          << ((int)(*plan)[i].getDeviceSpecification().getProcessingDeviceID() -
              PD_DMA0)
          << ", ";
    } else {
      out << "PD: "
          << util::getName((*plan)[i].getDeviceSpecification().getDeviceType())
          << (int)(*plan)[i].getDeviceSpecification().getProcessingDeviceID()
          << ", ";
    }
    if ((*plan)[i].getDeviceSpecification().getMemoryID() == PD_NO_Memory) {
      out << "NO_Memory";
    } else {
      out << "MemoryID" << (*plan)[i].getDeviceSpecification().getMemoryID();
    }
    out << ")" << std::endl;
    total_cost += (*plan)[i]
                      .getEstimatedExecutionTimeforAlgorithm()
                      .getTimeinNanoseconds();
  }
  out << "------------------------------------------------------------"
      << std::endl;
  out << "Total Execution Cost: " << total_cost / (1000 * 1000) << "ms" << endl;
  out << "============================================================"
      << std::endl;
}

std::string toString(const InternalPhysicalOperator& phy_op) {
  std::stringstream ss;
  ss << phy_op.alg_ptr->getName() << "(" << phy_op.feature_vector
     << "): " << (phy_op.cost) / (1000 * 1000) << "ms";  // << endl;
  return ss.str();
}

void print(const InternalPhysicalOperator& phy_op, std::ostream& out) {
  out << toString(phy_op) << endl;
}

void print(const InternalPhysicalPlan& phy_plan, std::ostream& out) {
  out << "InternalPhysicalPlan:" << endl;
  double cost = 0;
  for (unsigned int i = 0; i < phy_plan.size(); ++i) {
    out << toString(phy_plan[i]) << endl;
    cost += phy_plan[i].cost;
  }
  out << "Cost: " << (cost) / (1000 * 1000) << "ms" << endl;
}

}  // end namespace util

std::ostream& operator<<(std::ostream& out, const core::Tuple& tuple) {
  // feature vector
  out << "FV=[";
  for (unsigned int i = 0; i < tuple.size(); i++) {
    out << tuple[i];
    if (i < tuple.size() - 1) out << ",";
  }
  out << "]";

  return out;
}

std::stringstream& operator<<(std::stringstream& out,
                              const core::Tuple& tuple) {
  // feature vector
  out << "FV=[";
  for (unsigned int i = 0; i < tuple.size(); i++) {
    out << tuple[i];
    if (i < tuple.size() - 1) out << ",";
  }
  out << "]";

  return out;
}

}  // end namespace hype
