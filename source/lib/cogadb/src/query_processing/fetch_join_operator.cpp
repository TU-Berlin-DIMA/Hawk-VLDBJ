
#include <query_processing/join_operator.hpp>
#include <util/hardware_detector.hpp>

#include "query_processing/fetch_join_operator.hpp"

#include <query_processing/scan_operator.hpp>

#include <core/processor_data_cache.hpp>
#include <lookup_table/join_index.hpp>

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Selection_operator=physical_operator::map_init_function_Selection_operator;

namespace physical_operator {

TypedOperatorPtr create_CPU_Fetch_Join_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Fetch_Join& log_join_ref =
      static_cast<logical_operator::Logical_Fetch_Join&>(logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create CPU_Fetch_Join_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child != NULL);  // binary operator
  return TypedOperatorPtr(new CPU_Fetch_Join_Operator(
      sched_dec, left_child, right_child, log_join_ref.getPKColumnName(),
      log_join_ref.getFKColumnName(), log_join_ref.getMaterializationStatus()));
}

Physical_Operator_Map_Ptr map_init_function_fetch_join_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for FETCH_JOIN operator!"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("FETCH_JOIN","CPU_NestedLoopFetch_Join_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");
// stemod::Scheduler::instance().addAlgorithm("FETCH_JOIN","CPU_SortMergeFetch_Join_Algorithm",stemod::CPU,"Least
// Squares 2D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("FETCH_JOIN","CPU_HashFetch_Join_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification alg_spec_cpu_fetch_join(
      "CPU_Fetch_Join_Algorithm", "FETCH_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification alg_spec_gpu_fetch_join(
      "GPU_Fetch_Join_Algorithm", "FETCH_JOIN", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification alg_spec_cpu_fetch_join(
      "CPU_Fetch_Join_Algorithm", "FETCH_JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);

  hype::AlgorithmSpecification alg_spec_gpu_fetch_join(
      "GPU_Fetch_Join_Algorithm", "FETCH_JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);
#endif
  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(alg_spec_cpu_fetch_join,
                                               dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
#ifdef ENABLE_GPU_FETCH_JOIN
      hype::Scheduler::instance().addAlgorithm(alg_spec_gpu_fetch_join,
                                               dev_specs[i]);
//                        hype::Scheduler::instance().registerMemoryCostModel(alg_spec_gpu_fetch_join,
//                        dev_specs[i],
//                        &gpu::GPU_Operators_Memory_Cost_Models::tableFetchJoin);
#endif
#endif
    }
  }
  // stemod::Scheduler::instance().addAlgorithm("SELECTION","GPU_Selection_Algorithm",stemod::GPU,"Least
  // Squares 1D","Periodic Recomputation");
  map["CPU_Fetch_Join_Algorithm"] = create_CPU_Fetch_Join_Operator;
  map["GPU_Fetch_Join_Algorithm"] = create_CPU_Fetch_Join_Operator;

  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

namespace logical_operator {

bool Logical_Fetch_Join::isInputDataCachedInGPU() {
  //                if(this->getLeft()->getPhysicalOperator() &&
  //                this->getRight()->getPhysicalOperator()){
  //                    this->getLeft()->getPhysicalOperator()->
  //                }
  if (this->getRight()->getOperationName() != "SCAN") {
    COGADB_FATAL_ERROR(
        "Fetch_Join has not a SCAN as right child (which should be the fact "
        "table)",
        "");
  }
  boost::shared_ptr<Logical_Scan> right_scan =
      boost::dynamic_pointer_cast<Logical_Scan>(this->getRight());
  assert(right_scan != NULL);
  TablePtr fact_table = right_scan->getTablePtr();
  TablePtr dimension_table;
  NodePtr child = this->getLeft();
  while (child &&
         child->getOperationName() !=
             "SCAN") {  // || child->getOperationName()=="CPU_COLUMN_SCAN"){
    child = child->getLeft();
  }
  assert(child != NULL);
  boost::shared_ptr<Logical_Scan> left_scan =
      boost::dynamic_pointer_cast<Logical_Scan>(child);
  dimension_table = left_scan->getTablePtr();

  JoinIndexPtr join_index = JoinIndexes::instance().getJoinIndex(
      dimension_table, this->join_pk_column_name_, fact_table,
      this->join_fk_column_name_);
  assert(join_index != NULL);
  hype::ProcessingDeviceMemoryID mem_id = getMemoryIDForDeviceID(0);
  return DataCacheManager::instance().getDataCache(mem_id).isCached(join_index);
}
}

}  // end namespace query_processing

}  // end namespace CogaDB
