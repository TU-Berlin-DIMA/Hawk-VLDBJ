
#include <core/lookup_array.hpp>
#include <query_processing/column_processing/column_convert_bitmap_to_positionlist.hpp>
#include <util/hardware_detector.hpp>

#include <core/data_dictionary.hpp>

#include <backends/gpu/memory_cost_models.hpp>
#include <hardware_optimizations/primitives.hpp>
#include <query_processing/query_processor.hpp>
#include "backends/processor_backend.hpp"
#include "compression/dictionary_compressed_column.hpp"
#include "core/gpu_column_cache.hpp"

namespace CoGaDB {

namespace query_processing {
// Map_Init_Function
// init_function_column_convert_bitmap_to_positionlist=physical_operator::map_init_function_column_convert_bitmap_to_positionlist;

namespace physical_operator {

column_processing::cpu::TypedOperatorPtr
create_CPU_column_convert_bitmap_to_positionlist(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr right_child) {
  //                logical_operator::Logical_Column_Convert_Bitmap_To_PositionList&
  //                log_sort_ref =
  //                static_cast<logical_operator::Logical_Column_Convert_Bitmap_To_PositionList&>
  //                (logical_node);
  // std::cout << "create SCAN Operator!" << std::endl;

  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator

  return column_processing::cpu::TypedOperatorPtr(
      new CPU_column_convert_bitmap_to_positionlist(sched_dec, left_child));
}

//            column_processing::cpu::TypedOperatorPtr
//            create_GPU_column_convert_bitmap_to_positionlist(column_processing::cpu::TypedLogicalNode&
//            logical_node, const hype::SchedulingDecision& sched_dec,
//            column_processing::cpu::TypedOperatorPtr left_child,
//            column_processing::cpu::TypedOperatorPtr right_child) {
//                logical_operator::Logical_Column_Convert_Bitmap_To_PositionList&
//                log_sort_ref =
//                static_cast<logical_operator::Logical_Column_Convert_Bitmap_To_PositionList&>
//                (logical_node);
//                //std::cout << "create SCAN Operator!" << std::endl;
//
//                if(!left_child) {
//                    std::cout << "Error! File: " << __FILE__ << " Line: " <<
//                    __LINE__ << std::endl;
//                        exit(-1);
//                }
//                assert(right_child==NULL); //unary operator
//
//                return column_processing::cpu::TypedOperatorPtr(new
//                GPU_column_convert_bitmap_to_positionlist(sched_dec,
//                left_child));
//            }

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_column_convert_bitmap_to_positionlist() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function! (CONVERT_BITMAP_TO_POSITIONLIST "
                 "OPERATION)"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SCAN","TABLE_SCAN",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_CONVERT_BITMAP_TO_POSITIONLIST", "CONVERT_BITMAP_TO_POSITIONLIST",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_CONVERT_BITMAP_TO_POSITIONLIST", "CONVERT_BITMAP_TO_POSITIONLIST",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_CONVERT_BITMAP_TO_POSITIONLIST", "CONVERT_BITMAP_TO_POSITIONLIST",
      hype::Least_Squares_1D, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_CONVERT_BITMAP_TO_POSITIONLIST", "CONVERT_BITMAP_TO_POSITIONLIST",
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
          &gpu::GPU_Operators_Memory_Cost_Models::bitmapToPositionList);
#endif
    }
  }
  map["CPU_CONVERT_BITMAP_TO_POSITIONLIST"] =
      create_CPU_column_convert_bitmap_to_positionlist;
  map["GPU_CONVERT_BITMAP_TO_POSITIONLIST"] =
      create_CPU_column_convert_bitmap_to_positionlist;  // create_GPU_column_convert_bitmap_to_positionlist;
  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

CPU_column_convert_bitmap_to_positionlist::
    CPU_column_convert_bitmap_to_positionlist(
        const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child)
    : UnaryOperator<ColumnPtr, ColumnPtr>(sched_dec, child),
      PositionListOperator() {
  // this->result_=getTablebyName(table_name_);
}

bool CPU_column_convert_bitmap_to_positionlist::execute() {
  if (!quiet && verbose && debug)
    std::cout << "Execute Filter Operation" << std::endl;

  BitmapOperator* bitmap_op = dynamic_cast<BitmapOperator*>(this->child_.get());
  assert(bitmap_op != NULL);

  hype::ProcessingDeviceID id =
      sched_dec_.getDeviceSpecification().getProcessingDeviceID();
  hype::ProcessingDeviceMemoryID mem_id = hype::util::getMemoryID(id);
  ProcessorSpecification proc_spec(id);

  BitmapPtr placed_input_bitmap;
  placed_input_bitmap = copy_if_required(bitmap_op->getResultBitmap(), mem_id);
  if (!placed_input_bitmap) {
    this->has_aborted_ = true;
    return false;
  }

  ProcessorBackend<TID>* backend = ProcessorBackend<TID>::get(id);
  this->tids_ =
      backend->convertBitmapToPositionList(placed_input_bitmap, proc_spec);
  if (!this->tids_) {
    this->has_aborted_ = true;
    return false;
  }
  this->result_size_ = tids_->size();
  return true;
}

CPU_column_convert_bitmap_to_positionlist::
    ~CPU_column_convert_bitmap_to_positionlist() {}

//            /*GPU Operation*/
//            GPU_column_convert_bitmap_to_positionlist::GPU_column_convert_bitmap_to_positionlist(const
//            hype::SchedulingDecision& sched_dec, TypedOperatorPtr child) :
//            UnaryOperator<ColumnPtr, ColumnPtr>(sched_dec, child),
//            PositionListOperator() {
//                //this->result_=getTablebyName(table_name_);
//            }
//
//            bool GPU_column_convert_bitmap_to_positionlist::execute() {
//                if (!quiet && verbose && debug) std::cout << "Execute Filter
//                Operation" << std::endl;
//
//                    COGADB_EXECUTE_GPU_OPERATOR("CONVERT_BITMAP_TO_POSITIONLIST");
//
//
//                    BitmapOperator* bitmap_op =
//                    dynamic_cast<BitmapOperator*>(this->child_.get());
//
//                    assert(bitmap_op!=NULL);
//                    assert(bitmap_op->hasResultBitmap() ||
//                    bitmap_op->hasCachedResult_GPU_Bitmap());
//
//                    gpu::GPU_BitmapPtr input_bitmap;
//
//                    if(bitmap_op->hasResultBitmap() &&
//                    !bitmap_op->hasCachedResult_GPU_Bitmap()){
//                        input_bitmap=gpu::copy_Bitmap_host_to_device(bitmap_op->getResultBitmap());
//                    }else if(bitmap_op->hasCachedResult_GPU_Bitmap()){
//                        input_bitmap=bitmap_op->getResult_GPU_Bitmap();
//                    }
//
//                   //assert(input_bitmap!=NULL);
//
//
//                    this->gpu_tids_ =
//                    gpu::GPU_Operators::convertBitmapToPositionList(input_bitmap);
//                    if(this->gpu_tids_){
//                         this->result_size_ = gpu_tids_->size();
//                    }else{
//                        //GPU operator aborted, execute operator on CPU
//                        COGADB_ABORT_GPU_OPERATOR("CONVERT_BITMAP_TO_POSITIONLIST");
//                        BitmapPtr input_bitmap;
//                        if(!bitmap_op->hasResultBitmap() &&
//                        bitmap_op->hasCachedResult_GPU_Bitmap()){
//                            input_bitmap=gpu::copy_Bitmap_device_to_host(bitmap_op->getResult_GPU_Bitmap());
//                        }else if(bitmap_op->hasResultBitmap()){
//                            input_bitmap=bitmap_op->getResultBitmap();
//                        }
//                        assert(input_bitmap!=NULL);
//
//                        this->tids_ =
//                        CDK::convertBitmapToPositionList(input_bitmap);
//                        assert(this->tids_!=NULL);
//                        this->result_size_ = tids_->size();
//                    }
//
////#define VALIDATE_GPU_RESULTS_ON_CPU
//#ifdef VALIDATE_GPU_RESULTS_ON_CPU
//                    assert(this->gpu_tids_!=NULL);
//                    BitmapPtr bitmap=bitmap_op->getResultBitmap();
//                    this->cpu_tids_ =
//                    CDK::convertBitmapToPositionList(bitmap);
//                    PositionListPtr reference_matching_fact_table_tids =
//                    gpu::copy_PositionList_device_to_host(this->gpu_tids_);
//                    assert((*reference_matching_fact_table_tids)==(*(this->cpu_tids_)));
//#endif
//
//                if (this->gpu_tids_) {
//                    return true;
//                } else {
//                    return false;
//                }
//            }
//
//            void
//            GPU_column_convert_bitmap_to_positionlist::releaseInputData(){
//                BitmapOperator* bitmap_op =
//                dynamic_cast<BitmapOperator*>(this->child_.get());
//                assert(bitmap_op!=NULL);
//                bitmap_op->releaseResultData();
//                //this->TypedOperator<ColumnPtr>::releaseInputData();
//            }
//            bool
//            GPU_column_convert_bitmap_to_positionlist::isInputDataCachedInGPU(){
//                BitmapOperator* bitmap_op =
//                dynamic_cast<BitmapOperator*>(this->child_.get());
//                assert(bitmap_op!=NULL);
//                return bitmap_op->hasCachedResult_GPU_Bitmap();
//            }
//            GPU_column_convert_bitmap_to_positionlist::~GPU_column_convert_bitmap_to_positionlist(){
//            }

}  // end namespace physical_operator

namespace logical_operator {

Logical_Column_Convert_Bitmap_To_PositionList::
    Logical_Column_Convert_Bitmap_To_PositionList(
        hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<
          ColumnPtr,
          physical_operator::
              map_init_function_column_convert_bitmap_to_positionlist>(
          false, dev_constr) {}

unsigned int
Logical_Column_Convert_Bitmap_To_PositionList::getOutputResultSize() const {
  return 10;
}

double Logical_Column_Convert_Bitmap_To_PositionList::getCalculatedSelectivity()
    const {
  return 0.3;
}

std::string Logical_Column_Convert_Bitmap_To_PositionList::getOperationName()
    const {
  return "CONVERT_BITMAP_TO_POSITIONLIST";
}

std::string Logical_Column_Convert_Bitmap_To_PositionList::toString(
    bool verbose) const {
  std::string result = "CONVERT_BITMAP_TO_POSITIONLIST";
  return result;
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
