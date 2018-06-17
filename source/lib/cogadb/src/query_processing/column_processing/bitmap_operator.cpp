
#include <backends/gpu/memory_cost_models.hpp>
#include <backends/processor_backend.hpp>
#include <query_processing/column_processing/bitmap_operator.hpp>
#include <query_processing/query_processor.hpp>
#include <util/hardware_detector.hpp>

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Selection_operator=physical_operator::map_init_function_Selection_operator;

namespace physical_operator {

CPU_Bitmap_Operator::CPU_Bitmap_Operator(
    const hype::SchedulingDecision& sched_dec, TypedOperatorPtr left_child,
    TypedOperatorPtr right_child, BitmapOperation op,
    MaterializationStatus mat_stat)
    : BinaryOperator<ColumnPtr, ColumnPtr, ColumnPtr>(sched_dec, left_child,
                                                      right_child),
      BitmapOperator(),
      op_(op) {}

bool CPU_Bitmap_Operator::execute() {
  if (!quiet && debug && verbose)
    std::cout << "Execute Column Operator CPU" << std::endl;

  BitmapOperator* left_bitmap_op =
      dynamic_cast<BitmapOperator*>(this->left_child_.get());
  BitmapOperator* right_bitmap_op =
      dynamic_cast<BitmapOperator*>(this->right_child_.get());
  assert(left_bitmap_op != NULL);
  assert(right_bitmap_op != NULL);

  hype::ProcessingDeviceID id =
      sched_dec_.getDeviceSpecification().getProcessingDeviceID();
  hype::ProcessingDeviceMemoryID mem_id = hype::util::getMemoryID(id);
  ProcessorSpecification proc_spec(id);
  BitmapOperationParam param(proc_spec, this->op_);

  BitmapPtr placed_input_bitmap_left;
  placed_input_bitmap_left =
      copy_if_required(left_bitmap_op->getResultBitmap(), mem_id);
  if (!placed_input_bitmap_left) return false;

  BitmapPtr placed_input_bitmap_right;
  placed_input_bitmap_right =
      copy_if_required(right_bitmap_op->getResultBitmap(), mem_id);
  if (!placed_input_bitmap_right) return false;

  ProcessorBackend<TID>* backend = ProcessorBackend<TID>::get(id);
  this->cpu_bitmap_ = backend->computeBitmapSetOperation(
      placed_input_bitmap_left, placed_input_bitmap_right, param);
  if (!this->cpu_bitmap_) return false;
  this->result_size_ = -1;
  return true;
}

CPU_Bitmap_Operator::~CPU_Bitmap_Operator() {}

column_processing::cpu::TypedOperatorPtr create_CPU_Bitmap_Operator(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr right_child) {
  logical_operator::Logical_Bitmap_Operator& log_algebra_ref =
      static_cast<logical_operator::Logical_Bitmap_Operator&>(logical_node);
  if (!quiet && debug && verbose)
    std::cout << "create CPU_Columnalgebra_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "[HyPE Library]: Fatal Error! in "
                 "query_processing::physical_operator::create_CPU_Bitmap_Union_"
                 "Operator()"
              << std::endl;
    std::cout << "Left Child of Node " << logical_node.getOperationName()
              << " is Invalid!" << std::endl;
    exit(-1);
  }

  assert(right_child != NULL);  // binary operator
  return column_processing::cpu::TypedOperatorPtr(new CPU_Bitmap_Operator(
      sched_dec, left_child, right_child, log_algebra_ref.getBitmapOperation(),
      log_algebra_ref.getMaterializationStatus()));
}

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_cpu_bitmap_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
// std::cout << "calling map init function for CPU_Bitmap_Union_Operator
// operator!" << std::endl;
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_NestedLoopJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");
// stemod::Scheduler::instance().addAlgorithm("JOIN","CPU_SortMergeJoin_Algorithm",stemod::CPU,"Least
// Squares 2D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_HashJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification col_algebra_alg_spec_cpu(
      "CPU_Bitmap_Operator", "Bitmap_Operator", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification col_algebra_alg_spec_gpu(
      "GPU_Bitmap_Operator", "Bitmap_Operator", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification col_algebra_alg_spec_cpu(
      "CPU_Bitmap_Operator", "Bitmap_Operator", hype::Multilinear_Fitting_2D,
      hype::Periodic);

  hype::AlgorithmSpecification col_algebra_alg_spec_gpu(
      "GPU_Bitmap_Operator", "Bitmap_Operator", hype::Multilinear_Fitting_2D,
      hype::Periodic);
#endif
  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(col_algebra_alg_spec_cpu,
                                               dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
      hype::Scheduler::instance().addAlgorithm(col_algebra_alg_spec_gpu,
                                               dev_specs[i]);
      hype::Scheduler::instance().registerMemoryCostModel(
          col_algebra_alg_spec_gpu, dev_specs[i],
          &gpu::GPU_Operators_Memory_Cost_Models::bitwiseAND);
#endif
    }
  }

  map["CPU_Bitmap_Operator"] = create_CPU_Bitmap_Operator;
  map["GPU_Bitmap_Operator"] =
      create_CPU_Bitmap_Operator;  // create_GPU_Bitmap_Operator;

  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

}  // end namespace physical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
