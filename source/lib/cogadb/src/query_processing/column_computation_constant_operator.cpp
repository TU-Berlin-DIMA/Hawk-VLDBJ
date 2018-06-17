
#include <query_processing/column_computation_constant_operator.hpp>
#include <util/hardware_detector.hpp>

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_sort_operator=physical_operator::map_init_function_column_constant_operator;

namespace physical_operator {

TypedOperatorPtr create_CPU_ColumnConstant_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_ColumnConstantOperator& log_sort_ref =
      static_cast<logical_operator::Logical_ColumnConstantOperator&>(
          logical_node);
  // std::cout << "create CPU_ColumnConstant_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new CPU_ColumnConstantOperator(
      sched_dec, left_child, log_sort_ref.getColumnName(),
      log_sort_ref.getValue(), log_sort_ref.getResultColumnName(),
      log_sort_ref.getColumnAlgebraOperation()));
}

Physical_Operator_Map_Ptr map_init_function_column_constant_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for Sort Operator!" << std::endl;
// hype::Scheduler::instance().addAlgorithm("SORT","CPU_Sort_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("SORT","GPU_Sort_Algorithm",hype::GPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification sort_alg_spec_cpu(
      "CPU_ColumnConstant_Algorithm", "ColumnConstantOperator",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification sort_alg_spec_gpu(
      "GPU_ColumnConstant_Algorithm", "ColumnConstantOperator",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification sort_alg_spec_cpu(
      "CPU_ColumnConstant_Algorithm", "ColumnConstantOperator",
      hype::Least_Squares_1D, hype::Periodic);

  hype::AlgorithmSpecification sort_alg_spec_gpu(
      "GPU_ColumnConstant_Algorithm", "ColumnConstantOperator",
      hype::Least_Squares_1D, hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(sort_alg_spec_cpu, dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
      hype::Scheduler::instance().addAlgorithm(sort_alg_spec_gpu, dev_specs[i]);
#endif
    }
  }

  map["CPU_ColumnConstant_Algorithm"] = create_CPU_ColumnConstant_Operator;
  map["GPU_ColumnConstant_Algorithm"] = create_CPU_ColumnConstant_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
