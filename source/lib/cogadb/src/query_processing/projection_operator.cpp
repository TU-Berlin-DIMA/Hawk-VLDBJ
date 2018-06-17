
#include <query_processing/projection_operator.hpp>
#include <util/hardware_detector.hpp>

#include "query_compilation/code_generator.hpp"
#include "query_compilation/query_context.hpp"

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Projection_operator=physical_operator::map_init_function_Projection_operator;

namespace physical_operator {

TypedOperatorPtr create_CPU_Projection_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Projection& log_Projection_ref =
      static_cast<logical_operator::Logical_Projection&>(logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create CPU_Projection_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new CPU_Projection_Operator(
      sched_dec, left_child, log_Projection_ref.getColumnList()));
}

TypedOperatorPtr create_GPU_Projection_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Projection& log_Projection_ref =
      static_cast<logical_operator::Logical_Projection&>(logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create GPU_Projection_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new GPU_Projection_Operator(
      sched_dec, left_child, log_Projection_ref.getColumnList()));
}

Physical_Operator_Map_Ptr map_init_function_projection_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for PROJECTION operator!"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("PROJECTION","CPU_Projection_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification projection_alg_spec_cpu(
      "CPU_Projection_Algorithm", "PROJECTION", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification projection_alg_spec_cpu(
      "CPU_Projection_Algorithm", "PROJECTION", hype::Least_Squares_1D,
      hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(projection_alg_spec_cpu,
                                               dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
      // hype::Scheduler::instance().addAlgorithm(group_by_alg_spec_gpu,dev_specs[i]);
    }
  }

  // hype::Scheduler::instance().addAlgorithm("SELECTION","GPU_Projection_Algorithm",hype::GPU,"Least
  // Squares 1D","Periodic Recomputation");
  map["CPU_Projection_Algorithm"] = create_CPU_Projection_Operator;
  // map["GPU_Projection_Algorithm"]=create_GPU_Projection_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

namespace logical_operator {

void Logical_Projection::produce_impl(CodeGeneratorPtr code_gen,
                                      QueryContextPtr context) {
  context->clearProjectionList();

  for (size_t i = 0; i < attribute_references_.size(); ++i) {
    std::string result =
        createFullyQualifiedColumnIdentifier(attribute_references_[i]);
    context->addColumnToProjectionList(attribute_references_[i]);
    context->addAccessedColumn(result);
  }

  left_->produce(code_gen, context);
}

void Logical_Projection::consume_impl(CodeGeneratorPtr code_gen,
                                      QueryContextPtr context) {
  if (this->parent_) {
    this->parent_->consume(code_gen, context);
  }
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
