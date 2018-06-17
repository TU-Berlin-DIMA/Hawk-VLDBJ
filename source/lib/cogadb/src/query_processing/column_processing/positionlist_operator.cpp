
#include <query_processing/column_processing/positionlist_operator.hpp>
#include <query_processing/query_processor.hpp>
#include <util/hardware_detector.hpp>

#include "backends/processor_backend.hpp"

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Selection_operator=physical_operator::map_init_function_Selection_operator;

namespace physical_operator {

column_processing::cpu::TypedOperatorPtr create_CPU_PositionList_Operator(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr right_child) {
  logical_operator::Logical_PositionList_Operator& log_algebra_ref =
      static_cast<logical_operator::Logical_PositionList_Operator&>(
          logical_node);
  if (!quiet && debug && verbose)
    std::cout << "create CPU_Columnalgebra_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "[HyPE Library]: Fatal Error! in "
                 "query_processing::physical_operator::create_CPU_PositionList_"
                 "Union_Operator()"
              << std::endl;
    std::cout << "Left Child of Node " << logical_node.getOperationName()
              << " is Invalid!" << std::endl;
    exit(-1);
  }

  assert(right_child != NULL);  // binary operator
  return column_processing::cpu::TypedOperatorPtr(new CPU_PositionList_Operator(
      sched_dec, left_child, right_child,
      log_algebra_ref.getPositionListOperation(),
      log_algebra_ref.getMaterializationStatus()));
}

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_cpu_positionlist_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
// std::cout << "calling map init function for CPU_PositionList_Union_Operator
// operator!" << std::endl;
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_NestedLoopJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");
// stemod::Scheduler::instance().addAlgorithm("JOIN","CPU_SortMergeJoin_Algorithm",stemod::CPU,"Least
// Squares 2D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_HashJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification col_algebra_alg_spec_cpu(
      "CPU_PositionList_Operator", "PositionList_Operator",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification col_algebra_alg_spec_gpu(
      "GPU_PositionList_Operator", "PositionList_Operator",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification col_algebra_alg_spec_cpu(
      "CPU_PositionList_Operator", "PositionList_Operator",
      hype::Multilinear_Fitting_2D, hype::Periodic);

  hype::AlgorithmSpecification col_algebra_alg_spec_gpu(
      "GPU_PositionList_Operator", "PositionList_Operator",
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
#ifdef ENABLE_GPU_ACCELERATION
      hype::Scheduler::instance().addAlgorithm(col_algebra_alg_spec_gpu,
                                               dev_specs[i]);
#endif
      // hype::Scheduler::instance().addAlgorithm(group_by_alg_spec_gpu,dev_specs[i]);
    }
  }

  map["CPU_PositionList_Operator"] = create_CPU_PositionList_Operator;
  map["GPU_PositionList_Operator"] = create_CPU_PositionList_Operator;

  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

bool CPU_PositionList_Operator::execute() {
  if (!quiet && debug && verbose)
    std::cout << "Execute Column Operator CPU" << std::endl;

  PositionListOperator* pos_list_op_left =
      dynamic_cast<PositionListOperator*>(this->left_child_.get());
  PositionListOperator* pos_list_op_right =
      dynamic_cast<PositionListOperator*>(this->right_child_.get());

  assert(pos_list_op_left != NULL);
  assert(pos_list_op_right != NULL);

  hype::ProcessingDeviceID id =
      sched_dec_.getDeviceSpecification().getProcessingDeviceID();
  hype::ProcessingDeviceMemoryID mem_id = hype::util::getMemoryID(id);
  ProcessorSpecification proc_spec(id);

  SetOperation set_op;
  if (op_ == POSITIONLIST_INTERSECTION) {
    set_op = INTERSECT;
  } else if (op_ == POSITIONLIST_UNION) {
    set_op = UNION;
  } else {
    COGADB_FATAL_ERROR("Invalid Set Operation!", "");
  }

  SetOperationParam param(proc_spec, set_op);

  PositionListPtr tids;

  PositionListPtr input_tids_left;
  input_tids_left =
      copy_if_required(pos_list_op_left->getResultPositionList(), mem_id);
  if (!input_tids_left) return false;

  PositionListPtr input_tids_right;
  input_tids_right =
      copy_if_required(pos_list_op_right->getResultPositionList(), mem_id);
  if (!input_tids_right) return false;

  ProcessorBackend<TID>* backend = ProcessorBackend<TID>::get(id);
  tids = backend->computePositionListSetOperation(input_tids_left,
                                                  input_tids_right, param);

  if (tids) {
    // this->result_ =
    // createLookupArrayForColumn(this->getInputDataLeftChild(),tids);
    this->tids_ = tids;
    this->result_size_ = tids->size();
    return true;
  } else {
    return false;
  }
  //                    if (this->result_)
  //                        return true;
  //                    else
  //                        return false;
}

}  // end namespace physical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
