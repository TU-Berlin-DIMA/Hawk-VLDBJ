
#include <query_processing/column_processing/column_comparator.hpp>
#include <util/hardware_detector.hpp>

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Selection_operator=physical_operator::map_init_function_Selection_operator;

namespace physical_operator {

column_processing::cpu::TypedOperatorPtr create_CPU_ColumnComparatorOperator(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr right_child) {
  logical_operator::Logical_ColumnComparatorOperation& log_algebra_ref =
      static_cast<logical_operator::Logical_ColumnComparatorOperation&>(
          logical_node);
  if (!quiet && debug && verbose)
    std::cout << "create CPU_Columnalgebra_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "[HyPE Library]: Fatal Error! in "
                 "query_processing::physical_operator::create_CPU_"
                 "ColumnAlgebraOperator()"
              << std::endl;
    std::cout << "Left Child of Node " << logical_node.getOperationName()
              << " (" << log_algebra_ref.getPredicate().toString()
              << ") is Invalid!" << std::endl;
    exit(-1);
  }

  assert(right_child != NULL);  // binary operator
  return column_processing::cpu::TypedOperatorPtr(new ColumnComparatorOperation(
      sched_dec, left_child, right_child, log_algebra_ref.getPredicate(),
      log_algebra_ref.getMaterializationStatus()));
}

////dass sind GPU Columns!!!!!!!!!! -> funktioniert so nicht!!!!!
// ColumnWise_TypedOperatorPtr
// create_GPU_ColumnAlgebraOperator(ColumnWise_TypedLogicalNode& logical_node,
// const hype::SchedulingDecision&, ColumnWise_TypedOperatorPtr left_child,
// ColumnWise_TypedOperatorPtr right_child)
//	{
//		logical_operator::Logical_ColumnComparatorOperation&
// log_algebra_ref =
// static_cast<logical_operator::Logical_ColumnComparatorOperation&>(logical_node);
//		std::cout << "create CPU_Columnalgebra_Operator!" << std::endl;
//		if(!left_child) {
//			std::cout << "Error!" << std::endl;
//			exit(-1);
//		}
//

//
//		assert(right_child!=NULL); //binary operator
//		return ColumnWise_TypedOperatorPtr(new GPU_Predicate(sched_dec,
//		                        left_child,
//										right_child,
//		                        log_algebra_ref.getPredicate()
//		                        log_algebra_ref.getMaterializationStatus())
//);
//	}

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_cpu_column_comparison_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
// std::cout << "calling map init function for CPU_ColumnAlgebraOperator
// operator!" << std::endl;
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_NestedLoopJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");
// stemod::Scheduler::instance().addAlgorithm("JOIN","CPU_SortMergeJoin_Algorithm",stemod::CPU,"Least
// Squares 2D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_HashJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification col_algebra_alg_spec_cpu(
      "CPU_ColumnComparator_Algorithm", "ColumnComparatorOperation",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification col_algebra_alg_spec_cpu(
      "CPU_ColumnComparator_Algorithm", "ColumnComparatorOperation",
      hype::Multilinear_Fitting_2D, hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(col_algebra_alg_spec_cpu,
                                               dev_specs[i]);
      // hype::Scheduler::instance().addAlgorithm(join_alg_spec_cpu_hashjoin,dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
      // hype::Scheduler::instance().addAlgorithm(group_by_alg_spec_gpu,dev_specs[i]);
    }
  }

  map["CPU_ColumnComparator_Algorithm"] = create_CPU_ColumnComparatorOperator;

  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

}  // end namespace physical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
