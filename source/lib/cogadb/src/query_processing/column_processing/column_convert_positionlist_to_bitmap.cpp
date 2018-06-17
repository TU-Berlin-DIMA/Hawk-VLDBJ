
#include <core/lookup_array.hpp>
#include <query_processing/column_processing/column_convert_positionlist_to_bitmap.hpp>
#include <util/hardware_detector.hpp>

#include <core/data_dictionary.hpp>

#include <backends/gpu/memory_cost_models.hpp>
#include <hardware_optimizations/primitives.hpp>
#include <query_processing/query_processor.hpp>
#include "compression/dictionary_compressed_column.hpp"
#include "core/gpu_column_cache.hpp"
#include "lookup_table/join_index.hpp"
#include "query_processing/operator_extensions.hpp"

namespace CoGaDB {

namespace query_processing {
// Map_Init_Function
// init_function_column_convert_positionlist_to_Bitmap=physical_operator::map_init_function_column_convert_positionlist_to_Bitmap;

namespace physical_operator {

column_processing::cpu::TypedOperatorPtr
create_CPU_column_convert_positionlist_to_Bitmap(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr right_child) {
  logical_operator::Logical_Column_Convert_PositionList_To_Bitmap&
      log_sort_ref = static_cast<
          logical_operator::Logical_Column_Convert_PositionList_To_Bitmap&>(
          logical_node);
  // std::cout << "create SCAN Operator!" << std::endl;

  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator

  return column_processing::cpu::TypedOperatorPtr(
      new CPU_column_convert_positionlist_to_Bitmap(
          sched_dec, left_child, log_sort_ref.number_of_rows_));
}

column_processing::cpu::TypedOperatorPtr
create_GPU_column_convert_positionlist_to_Bitmap(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr right_child) {
  logical_operator::Logical_Column_Convert_PositionList_To_Bitmap&
      log_sort_ref = static_cast<
          logical_operator::Logical_Column_Convert_PositionList_To_Bitmap&>(
          logical_node);
  // std::cout << "create SCAN Operator!" << std::endl;

  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator

  return column_processing::cpu::TypedOperatorPtr(
      new GPU_column_convert_positionlist_to_Bitmap(
          sched_dec, left_child, log_sort_ref.number_of_rows_));
}

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_column_convert_positionlist_to_Bitmap() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function! (CONVERT_POSITIONLIST_TO_BITMAP "
                 "OPERATION)"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SCAN","TABLE_SCAN",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_CONVERT_POSITIONLIST_TO_BITMAP", "CONVERT_POSITIONLIST_TO_BITMAP",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_CONVERT_POSITIONLIST_TO_BITMAP", "CONVERT_POSITIONLIST_TO_BITMAP",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_CONVERT_POSITIONLIST_TO_BITMAP", "CONVERT_POSITIONLIST_TO_BITMAP",
      hype::Least_Squares_1D, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_CONVERT_POSITIONLIST_TO_BITMAP", "CONVERT_POSITIONLIST_TO_BITMAP",
      hype::Least_Squares_1D, hype::Periodic);
#endif

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu,
                                               dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_gpu,
                                               dev_specs[i]);
      hype::Scheduler::instance().registerMemoryCostModel(
          selection_alg_spec_gpu, dev_specs[i],
          &gpu::GPU_Operators_Memory_Cost_Models::positionlistToBitmap);
#endif
    }
  }
  map["CPU_CONVERT_POSITIONLIST_TO_BITMAP"] =
      create_CPU_column_convert_positionlist_to_Bitmap;
  map["GPU_CONVERT_POSITIONLIST_TO_BITMAP"] =
      create_GPU_column_convert_positionlist_to_Bitmap;
  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

CPU_column_convert_positionlist_to_Bitmap::
    CPU_column_convert_positionlist_to_Bitmap(
        const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
        size_t number_of_rows)
    : UnaryOperator<ColumnPtr, ColumnPtr>(sched_dec, child),
      BitmapOperator(),
      number_of_rows_(number_of_rows) {
  // this->result_=getTablebyName(table_name_);
}

bool CPU_column_convert_positionlist_to_Bitmap::execute() {
  COGADB_FATAL_ERROR(
      "Called unimplemented method! This function was not yet refactored to "
      "the new interface!",
      "");
  return false;
  //                if (!quiet && verbose && debug) std::cout << "Execute Filter
  //                Operation" << std::endl;
  //
  //                    PositionListOperator* input_tids =
  //                    dynamic_cast<PositionListOperator*>(this->child_.get());
  //                    assert(input_tids!=NULL);
  //                    assert(input_tids->hasResultPositionList() ||
  //                    input_tids->hasCachedResult_GPU_PositionList());
  //
  //                    PositionListPtr tids;
  //
  //                    if(!input_tids->hasResultPositionList() &&
  //                    input_tids->hasCachedResult_GPU_PositionList()){
  //                        tids=copy_PositionList_device_to_host(input_tids->getResult_GPU_PositionList());
  //                    }else{
  //                        tids = input_tids->getResultPositionList();
  //                    }
  //
  //                    assert(tids!=NULL);
  //
  //                    this->cpu_bitmap_ =
  //                    CoGaDB::CDK::convertPositionListToBitmap(tids,
  //                    this->number_of_rows_);
  //                    this->result_size_=this->number_of_rows_;
  //                    //CDK::selection::countSetBitsInBitmap(this->cpu_bitmap_->data(),this->cpu_bitmap_->size());
  //
  //                if (this->cpu_bitmap_) {
  //                    return true;
  //                } else {
  //                    return false;
  //                }
}

CPU_column_convert_positionlist_to_Bitmap::
    ~CPU_column_convert_positionlist_to_Bitmap() {}

/*GPU Operation*/
GPU_column_convert_positionlist_to_Bitmap::
    GPU_column_convert_positionlist_to_Bitmap(
        const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
        size_t number_of_rows)
    : UnaryOperator<ColumnPtr, ColumnPtr>(sched_dec, child),
      BitmapOperator(),
      number_of_rows_(number_of_rows) {
  // this->result_=getTablebyName(table_name_);
}

bool GPU_column_convert_positionlist_to_Bitmap::execute() {
  COGADB_FATAL_ERROR(
      "Called unimplemented method! This function was not yet refactored to "
      "the new interface!",
      "");
  return false;

  //                if (!quiet && verbose && debug) std::cout << "Execute GPU
  //                Operation" << std::endl;
  //                COGADB_EXECUTE_GPU_OPERATOR("CONVERT_POSITIONLIST_TO_BITMAP");
  //
  //                    PositionListOperator* input_tids =
  //                    dynamic_cast<PositionListOperator*>(this->child_.get());
  //                    assert(input_tids!=NULL);
  //                    assert(input_tids->hasResultPositionList() ||
  //                    input_tids->hasCachedResult_GPU_PositionList());
  //
  //                    gpu::GPU_PositionlistPtr gpu_tids;
  //
  //                    if(input_tids->hasResultPositionList() &&
  //                    !input_tids->hasCachedResult_GPU_PositionList()){
  //                        gpu_tids =
  //                        gpu::copy_PositionList_host_to_device(input_tids->getResultPositionList());
  //                    }else{
  //                        gpu_tids = input_tids->getResult_GPU_PositionList();
  //                    }
  //
  //                    if(!gpu_tids){
  //                        //GPU operator aborted, execute operator on CPU
  //                        COGADB_ABORT_GPU_OPERATOR("CONVERT_POSITIONLIST_TO_BITMAP");
  //                        PositionListPtr tids;
  //                        if(!input_tids->hasResultPositionList() &&
  //                        input_tids->hasCachedResult_GPU_PositionList()){
  //                            tids=copy_PositionList_device_to_host(input_tids->getResult_GPU_PositionList());
  //                        }else{
  //                            tids = input_tids->getResultPositionList();
  //                        }
  //                        assert(tids!=NULL);
  //                        this->cpu_bitmap_ =
  //                        CoGaDB::CDK::convertPositionListToBitmap(tids,
  //                        this->number_of_rows_);
  //                        this->result_size_=tids->size();
  //                        return true;
  //                    }
  //
  ////                    this->cpu_bitmap_ =
  /// CoGaDB::CDK::convertPositionListToBitmap(tids, this->number_of_rows_);
  ////
  /// this->result_size_=CDK::selection::countSetBitsInBitmap(this->cpu_bitmap_->data(),this->cpu_bitmap_->size());
  //                    this->gpu_bitmap_ =
  //                    gpu::GPU_Operators::convertPositionListToBitmap(gpu_tids,
  //                    this->number_of_rows_);
  //                    this->result_size_=gpu_tids->size(); //number of rows
  //                    does not change!
  //
  //                    if(!gpu_bitmap_){
  //                        //GPU operator aborted, execute operator on CPU
  //                        PositionListPtr tids;
  //                        if(!input_tids->hasResultPositionList() &&
  //                        input_tids->hasCachedResult_GPU_PositionList()){
  //                            tids=copy_PositionList_device_to_host(input_tids->getResult_GPU_PositionList());
  //                        }else{
  //                            tids = input_tids->getResultPositionList();
  //                        }
  //                        assert(tids!=NULL);
  //                        this->cpu_bitmap_ =
  //                        CoGaDB::CDK::convertPositionListToBitmap(tids,
  //                        this->number_of_rows_);
  //                        this->result_size_=tids->size();
  //                        return true;
  //                    }
  //
  ////#define VALIDATE_GPU_RESULTS_ON_CPU
  //#ifdef VALIDATE_GPU_RESULTS_ON_CPU
  //                    PositionListPtr tids;
  //                    tids=copy_PositionList_device_to_host(gpu_tids);
  //                    assert(tids!=NULL);
  //
  //                    this->cpu_bitmap_ =
  //                    CoGaDB::CDK::convertPositionListToBitmap(tids,
  //                    this->number_of_rows_);
  //
  //                    BitmapPtr result_bitmap_build_on_gpu =
  //                    gpu::copy_Bitmap_device_to_host(this->gpu_bitmap_);
  //                    assert(*(this->cpu_bitmap_)==*result_bitmap_build_on_gpu);
  //#endif
  //
  //
  //
  //                if (this->gpu_bitmap_) {
  //                    return true;
  //                } else {
  //                    return false;
  //                }
}

void GPU_column_convert_positionlist_to_Bitmap::releaseInputData() {
  PositionListOperator* input_tids =
      dynamic_cast<PositionListOperator*>(this->child_.get());
  assert(input_tids != NULL);
  input_tids->releaseResultData();
  // this->TypedOperator<ColumnPtr>::releaseInputData();
}
bool GPU_column_convert_positionlist_to_Bitmap::isInputDataCachedInGPU() {
  //                    PositionListOperator* input_tids =
  //                    dynamic_cast<PositionListOperator*>(this->child_.get());
  //                    assert(input_tids!=NULL);
  return false;  // input_tids->hasCachedResult_GPU_PositionList();
}

GPU_column_convert_positionlist_to_Bitmap::
    ~GPU_column_convert_positionlist_to_Bitmap() {}

}  // end namespace physical_operator

namespace logical_operator {

Logical_Column_Convert_PositionList_To_Bitmap::
    Logical_Column_Convert_PositionList_To_Bitmap(
        size_t number_of_rows, hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<
          ColumnPtr,
          physical_operator::
              map_init_function_column_convert_positionlist_to_Bitmap>(
          false, dev_constr),
      number_of_rows_(number_of_rows) {}

unsigned int
Logical_Column_Convert_PositionList_To_Bitmap::getOutputResultSize() const {
  return 10;
}

double Logical_Column_Convert_PositionList_To_Bitmap::getCalculatedSelectivity()
    const {
  return 0.3;
}

std::string Logical_Column_Convert_PositionList_To_Bitmap::getOperationName()
    const {
  return "CONVERT_POSITIONLIST_TO_BITMAP";
}

std::string Logical_Column_Convert_PositionList_To_Bitmap::toString(
    bool verbose) const {
  std::string result = "CONVERT_POSITIONLIST_TO_BITMAP";
  return result;
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
