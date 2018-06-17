
#include <query_processing/generate_constant_column_operator.hpp>
#include <util/hardware_detector.hpp>

namespace CoGaDB {
namespace query_processing {
namespace physical_operator {

TypedOperatorPtr create_AddConstantValueColumn_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_AddConstantValueColumn&
      log_AddConstantValueColumn_ref =
          static_cast<logical_operator::Logical_AddConstantValueColumn&>(
              logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create CPU_AddConstantValueColumn_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new AddConstantValueColumn_Operator(
      sched_dec, left_child, log_AddConstantValueColumn_ref.getColumnName(),
      log_AddConstantValueColumn_ref.getAttributeType(),
      log_AddConstantValueColumn_ref.getConstantValue()));
}

Physical_Operator_Map_Ptr map_init_function_addconstantvaluecolumn_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout
        << "calling map init function for AddConstantValueColumn operator!"
        << std::endl;
// hype::Scheduler::instance().addAlgorithm("PROJECTION","CPU_AddConstantValueColumn_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification projection_alg_spec_cpu(
      "CPU_AddConstantValueColumn_Algorithm", "AddConstantValueColumn",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification projection_alg_spec_gpu(
      "GPU_AddConstantValueColumn_Algorithm", "AddConstantValueColumn",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification projection_alg_spec_cpu(
      "CPU_AddConstantValueColumn_Algorithm", "AddConstantValueColumn",
      hype::Least_Squares_1D, hype::Periodic);

  hype::AlgorithmSpecification projection_alg_spec_gpu(
      "GPU_AddConstantValueColumn_Algorithm", "AddConstantValueColumn",
      hype::Least_Squares_1D, hype::Periodic);
#endif
  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(projection_alg_spec_cpu,
                                               dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
      hype::Scheduler::instance().addAlgorithm(projection_alg_spec_gpu,
                                               dev_specs[i]);
    }
  }

  map["CPU_AddConstantValueColumn_Algorithm"] =
      create_AddConstantValueColumn_Operator;
  map["GPU_AddConstantValueColumn_Algorithm"] =
      create_AddConstantValueColumn_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

namespace logical_operator {

void Logical_AddConstantValueColumn::produce_impl(CodeGeneratorPtr code_gen,
                                                  QueryContextPtr context) {
  /* by pass operator that is not relevant for query compiler */
  left_->produce(code_gen, context);
}

void Logical_AddConstantValueColumn::consume_impl(CodeGeneratorPtr code_gen,
                                                  QueryContextPtr context) {
  /* by pass operator that is not relevant for query compiler */
  if (parent_->getParent()) {
    parent_->getParent()->consume(code_gen, context);
  }
}
}

}  // end namespace query_processing
}  // end namespace CogaDB
