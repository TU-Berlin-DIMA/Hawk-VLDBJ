
#include <core/data_dictionary.hpp>
#include <core/processor_data_cache.hpp>
#include <query_processing/selection_operator.hpp>
#include <util/hardware_detector.hpp>

#include "query_compilation/code_generator.hpp"
#include "query_compilation/predicate_expression.hpp"

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Selection_operator=physical_operator::map_init_function_Selection_operator;

namespace physical_operator {

TypedOperatorPtr create_CPU_Selection_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Selection& log_selection_ref =
      static_cast<logical_operator::Logical_Selection&>(logical_node);
  // std::cout << "create CPU_Selection_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new CPU_Selection_Operator(
      sched_dec, left_child, log_selection_ref.getPredicate(),
      log_selection_ref.getMaterializationStatus()));
}

TypedOperatorPtr create_CPU_ParallelSelection_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Selection& log_selection_ref =
      static_cast<logical_operator::Logical_Selection&>(logical_node);
  // std::cout << "create CPU_Selection_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new CPU_ParallelSelection_Operator(
      sched_dec, left_child, log_selection_ref.getPredicate(),
      log_selection_ref.getMaterializationStatus()));
}

TypedOperatorPtr create_GPU_Selection_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Selection& log_selection_ref =
      static_cast<logical_operator::Logical_Selection&>(logical_node);
  // std::cout << "create GPU_Selection_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new GPU_Selection_Operator(
      sched_dec, left_child, log_selection_ref.getPredicate(),
      log_selection_ref.getMaterializationStatus()));
}

Physical_Operator_Map_Ptr map_init_function_selection_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for SELECTION operator!"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SELECTION","CPU_Selection_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("SELECTION","GPU_Selection_Algorithm",hype::GPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_Selection_Algorithm", "SELECTION", hype::KNN_Regression,
      hype::Periodic);
  hype::AlgorithmSpecification selection_alg_spec_cpu_parallel(
      "CPU_ParallelSelection_Algorithm", "SELECTION", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_Selection_Algorithm", "SELECTION", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_Selection_Algorithm", "SELECTION", hype::Least_Squares_1D,
      hype::Periodic);
  hype::AlgorithmSpecification selection_alg_spec_cpu_parallel(
      "CPU_ParallelSelection_Algorithm", "SELECTION", hype::Least_Squares_1D,
      hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_Selection_Algorithm", "SELECTION", hype::Least_Squares_1D,
      hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu,
                                               dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu_parallel,
                                               dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_gpu,
                                               dev_specs[i]);
#endif
    }
  }

  map["CPU_Selection_Algorithm"] = create_CPU_Selection_Operator;
  map["CPU_ParallelSelection_Algorithm"] =
      create_CPU_ParallelSelection_Operator;
  map["GPU_Selection_Algorithm"] = create_GPU_Selection_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

namespace logical_operator {

bool Logical_Selection::isInputDataCachedInGPU() {
  std::list<std::pair<ColumnPtr, TablePtr> > columns =
      DataDictionary::instance().getColumnsforColumnName(
          this->pred_.getColumn1Name());
  assert(columns.size() == 1);
  ColumnPtr col = columns.front().first;
  assert(col != NULL);
  hype::ProcessingDeviceMemoryID mem_id = getMemoryIDForDeviceID(0);
  return DataCacheManager::instance().getDataCache(mem_id).isCached(col);
}
}
}  // end namespace query_processing

}  // end namespace CogaDB
