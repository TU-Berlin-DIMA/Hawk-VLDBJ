
#include <query_processing/join_operator.hpp>
#include <util/hardware_detector.hpp>

#include "query_processing/pk_fk_join_operator.hpp"

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Selection_operator=physical_operator::map_init_function_Selection_operator;

namespace physical_operator {

TypedOperatorPtr create_CPU_NestedLoopPK_FK_Join_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_PK_FK_Join& log_join_ref =
      static_cast<logical_operator::Logical_PK_FK_Join&>(logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create CPU_NestedLoopPK_FK_Join_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child != NULL);  // binary operator
  return TypedOperatorPtr(new CPU_NestedLoopPK_FK_Join_Operator(
      sched_dec, left_child, right_child, log_join_ref.getPKColumnName(),
      log_join_ref.getFKColumnName(), log_join_ref.getMaterializationStatus()));
}

TypedOperatorPtr create_CPU_SortMergePK_FK_Join_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_PK_FK_Join& log_join_ref =
      static_cast<logical_operator::Logical_PK_FK_Join&>(logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create CPU_SortMergePK_FK_Join_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child != NULL);  // binary operator
  return TypedOperatorPtr(new CPU_SortMergePK_FK_Join_Operator(
      sched_dec, left_child, right_child, log_join_ref.getPKColumnName(),
      log_join_ref.getFKColumnName(), log_join_ref.getMaterializationStatus()));
}

TypedOperatorPtr create_CPU_HashPK_FK_Join_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_PK_FK_Join& log_join_ref =
      static_cast<logical_operator::Logical_PK_FK_Join&>(logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create CPU_HashPK_FK_Join_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child != NULL);  // binary operator
  return TypedOperatorPtr(new CPU_HashPK_FK_Join_Operator(
      sched_dec, left_child, right_child, log_join_ref.getPKColumnName(),
      log_join_ref.getFKColumnName(), log_join_ref.getMaterializationStatus()));
}

TypedOperatorPtr create_CPU_Parallel_HashPK_FK_Join_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_PK_FK_Join& log_join_ref =
      static_cast<logical_operator::Logical_PK_FK_Join&>(logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create CPU_Parallel_HashPK_FK_Join_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child != NULL);  // binary operator
  return TypedOperatorPtr(new CPU_Parallel_HashPK_FK_Join_Operator(
      sched_dec, left_child, right_child, log_join_ref.getPKColumnName(),
      log_join_ref.getFKColumnName(), log_join_ref.getMaterializationStatus()));
}

TypedOperatorPtr create_GPU_PK_FK_Join_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_PK_FK_Join& log_join_ref =
      static_cast<logical_operator::Logical_PK_FK_Join&>(logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create GPU_PK_FK_Join_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child != NULL);  // binary operator
  return TypedOperatorPtr(new GPU_PK_FK_Join_Operator(
      sched_dec, left_child, right_child, log_join_ref.getPKColumnName(),
      log_join_ref.getFKColumnName(), log_join_ref.getMaterializationStatus()));
}

Physical_Operator_Map_Ptr map_init_function_pk_fk_join_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for PK_FK_JOIN operator!"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("PK_FK_JOIN","CPU_NestedLoopPK_FK_Join_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");
// stemod::Scheduler::instance().addAlgorithm("PK_FK_JOIN","CPU_SortMergePK_FK_Join_Algorithm",stemod::CPU,"Least
// Squares 2D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("PK_FK_JOIN","CPU_HashPK_FK_Join_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification join_alg_spec_cpu_nlj(
      "CPU_NestedLoopPK_FK_Join_Algorithm", "PK_FK_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_hashjoin(
      "CPU_HashPK_FK_Join_Algorithm", "PK_FK_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_parallel_hashjoin(
      "CPU_Parallel_HashPK_FK_Join_Algorithm", "PK_FK_JOIN",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_gpu_sort_merge_join(
      "GPU_SortMergePK_FK_Join_Algorithm", "PK_FK_JOIN", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification join_alg_spec_cpu_nlj(
      "CPU_NestedLoopPK_FK_Join_Algorithm", "PK_FK_JOIN",
      hype::Multilinear_Fitting_2D, hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_hashjoin(
      "CPU_HashPK_FK_Join_Algorithm", "PK_FK_JOIN",
      hype::Multilinear_Fitting_2D, hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_parallel_hashjoin(
      "CPU_Parallel_HashPK_FK_Join_Algorithm", "PK_FK_JOIN",
      hype::Multilinear_Fitting_2D, hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_gpu_sort_merge_join(
      "GPU_SortMergePK_FK_Join_Algorithm", "PK_FK_JOIN",
      hype::Multilinear_Fitting_2D, hype::Periodic);
#endif
  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(join_alg_spec_cpu_hashjoin,
                                               dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(
          join_alg_spec_cpu_parallel_hashjoin, dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
#ifdef ENABLE_GPU_PK_FK_JOIN
      hype::Scheduler::instance().addAlgorithm(
          join_alg_spec_gpu_sort_merge_join, dev_specs[i]);
#endif
#endif
    }
  }

  // stemod::Scheduler::instance().addAlgorithm("SELECTION","GPU_Selection_Algorithm",stemod::GPU,"Least
  // Squares 1D","Periodic Recomputation");
  map["CPU_NestedLoopPK_FK_Join_Algorithm"] =
      create_CPU_NestedLoopPK_FK_Join_Operator;
  // map["CPU_SortMergePK_FK_Join_Algorithm"]=create_CPU_SortMergePK_FK_Join_Operator;
  map["GPU_SortMergePK_FK_Join_Algorithm"] = create_GPU_PK_FK_Join_Operator;
  map["CPU_HashPK_FK_Join_Algorithm"] = create_CPU_HashPK_FK_Join_Operator;
  map["CPU_Parallel_HashPK_FK_Join_Algorithm"] =
      create_CPU_Parallel_HashPK_FK_Join_Operator;
  // map["GPU_Selection_Algorithm"]=create_GPU_Selection_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
