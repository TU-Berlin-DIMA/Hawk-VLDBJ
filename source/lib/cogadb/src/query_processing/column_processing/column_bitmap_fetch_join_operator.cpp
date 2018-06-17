
#include <query_processing/column_processing/column_bitmap_fetch_join_operator.hpp>

#include <lookup_table/join_index.hpp>
#include <util/hardware_detector.hpp>

#include <util/time_measurement.hpp>

#include <hardware_optimizations/primitives.hpp>
#include "query_processing/operator_extensions.hpp"

//#include "core/gpu_column_cache.hpp"
#include <core/processor_data_cache.hpp>

#include <backends/processor_backend.hpp>
#include <query_processing/query_processor.hpp>

namespace CoGaDB {

namespace query_processing {

namespace physical_operator {

column_processing::cpu::TypedOperatorPtr
create_column_bitmap_fetch_join_operator(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr) {
  logical_operator::Logical_Column_Bitmap_Fetch_Join& log_sort_ref =
      static_cast<logical_operator::Logical_Column_Bitmap_Fetch_Join&>(
          logical_node);
  return column_processing::cpu::TypedOperatorPtr(
      new column_bitmap_fetch_join_operator(
          sched_dec, left_child, log_sort_ref.getPK_FK_Join_Predicate()));
}

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_column_bitmap_fetch_join_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
  if (!quiet)
    std::cout
        << "calling map init function! (COLUMN_BITMAP_FETCH_JOIN OPERATION)"
        << std::endl;

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_BITMAP_FETCH_JOIN", "COLUMN_BITMAP_FETCH_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_BITMAP_FETCH_JOIN", "COLUMN_BITMAP_FETCH_JOIN", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_BITMAP_FETCH_JOIN", "COLUMN_BITMAP_FETCH_JOIN",
      hype::Multilinear_Fitting_2D, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_BITMAP_FETCH_JOIN", "COLUMN_BITMAP_FETCH_JOIN",
      hype::Multilinear_Fitting_2D, hype::Periodic);
#endif

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu,
                                               dev_specs[i]);
    }
    if (dev_specs[i].getDeviceType() == hype::GPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_gpu,
                                               dev_specs[i]);
    }
  }

  map["CPU_BITMAP_FETCH_JOIN"] = create_column_bitmap_fetch_join_operator;
  map["GPU_BITMAP_FETCH_JOIN"] =
      create_column_bitmap_fetch_join_operator;  // create_gpu_column_bitmap_fetch_join_operator;
  // map["GPU_Algorithm"]=create_GPU_SORT_Operator;
  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

column_bitmap_fetch_join_operator::column_bitmap_fetch_join_operator(
    const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
    PK_FK_Join_Predicate pk_fk_join_pred)
    : UnaryOperator<ColumnPtr, ColumnPtr>(sched_dec, child),
      BitmapOperator(),
      pk_fk_join_pred_(pk_fk_join_pred) {}

bool column_bitmap_fetch_join_operator::execute() {
  if (!quiet && verbose && debug)
    std::cout << "Execute Column_Bitmap_Fetch_Join" << std::endl;

  JoinIndexPtr join_index = JoinIndexes::instance().getJoinIndex(
      getTablebyName(pk_fk_join_pred_.join_pk_table_name),
      pk_fk_join_pred_.join_pk_column_name,
      getTablebyName(pk_fk_join_pred_.join_fk_table_name),
      pk_fk_join_pred_.join_fk_column_name);
  assert(
      getTablebyName(pk_fk_join_pred_.join_fk_table_name)->getNumberofRows() ==
      join_index->second->getPositionList()->size());

  PositionListOperator* pos_list_op =
      dynamic_cast<PositionListOperator*>(this->child_.get());
  assert(pos_list_op != NULL);

  hype::ProcessingDeviceID id =
      sched_dec_.getDeviceSpecification().getProcessingDeviceID();
  ProcessorSpecification proc_spec(id);
  FetchJoinParam param(proc_spec);

  PositionListPtr placed_input_tids =
      copy_if_required(pos_list_op->getResultPositionList(), proc_spec);
  if (!placed_input_tids) {
    return false;
  }

  JoinIndexPtr placed_join_index = copy_if_required(join_index, proc_spec);
  ProcessorBackend<TID>* backend =
      ProcessorBackend<TID>::get(proc_spec.proc_id);
  this->cpu_bitmap_ =
      backend->bitmap_fetch_join(placed_join_index, placed_input_tids, param);
  if (!this->cpu_bitmap_) {
    this->has_aborted_ = true;
    if (param.proc_spec.proc_id == hype::PD0) {
      COGADB_FATAL_ERROR("Could not compute bitmap_fetch_join!", "");
    } else {
      COGADB_ERROR("Could not compute bitmap_fetch_join!", "");
    }
    return false;
  }

  return true;
}

column_bitmap_fetch_join_operator::~column_bitmap_fetch_join_operator() {}

//                        gpu_column_bitmap_fetch_join_operator::gpu_column_bitmap_fetch_join_operator(const
//                        hype::SchedulingDecision& sched_dec, TypedOperatorPtr
//                        child, PK_FK_Join_Predicate pk_fk_join_pred) :
//                        UnaryOperator<ColumnPtr,ColumnPtr>(sched_dec, child),
//                        BitmapOperator(),
//                                pk_fk_join_pred_(pk_fk_join_pred){
//                        }
//
//            bool gpu_column_bitmap_fetch_join_operator::execute() {
//
////#define VALIDATE_GPU_RESULTS_ON_CPU
//
//                if (!quiet && verbose && debug) std::cout << "Execute
//                Column_Fetch_Join" << std::endl;
//
//                return false;
//
////                COGADB_EXECUTE_GPU_OPERATOR("Column_Bitmap_Fetch_Join");
////
////                PositionListOperator* pos_list_op =
/// dynamic_cast<PositionListOperator*> (this->child_.get());
////                assert(pos_list_op != NULL);
////                assert(pos_list_op->hasResultPositionList() ||
/// pos_list_op->hasCachedResult_GPU_PositionList());
////
////                Timestamp begin_gpu_fetch_join = getTimestamp();
////                gpu::GPU_PositionlistPtr gpu_input_tids;
////                if (!pos_list_op->hasCachedResult_GPU_PositionList()) {
////                    gpu_input_tids =
/// gpu::copy_PositionList_host_to_device(pos_list_op->getResultPositionList());
////                } else {
////                    gpu_input_tids =
/// pos_list_op->getResult_GPU_PositionList();
////                }
////                //assert(gpu_input_tids != NULL);
////
////                JoinIndexPtr join_index =
/// JoinIndexes::instance().getJoinIndex(getTablebyName(pk_fk_join_pred_.join_pk_table_name),
/// pk_fk_join_pred_.join_pk_column_name,
/// getTablebyName(pk_fk_join_pred_.join_fk_table_name),
/// pk_fk_join_pred_.join_fk_column_name);
////
//////                //CPU cross check
////#ifdef VALIDATE_GPU_RESULTS_ON_CPU
////                Timestamp begin_cpu_fetch_join = getTimestamp();
////                PositionListPtr input_tids =
/// gpu::copy_PositionList_device_to_host(gpu_input_tids);
////                PositionListPtr matching_fact_table_tids =
/// CoGaDB::fetchMatchingTIDsFromJoinIndex(join_index, input_tids);
////
/// std::sort(matching_fact_table_tids->begin(),matching_fact_table_tids->end());
////                //BitmapPtr
/// ref_cpu_bitmap=createBitmapOfMatchingTIDsFromJoinIndex(join_index,input_tids);
////                //PositionListPtr matching_fact_table_tids =
/// CDK::convertBitmapToPositionList(ref_cpu_bitmap);
////                this->result_size_ = matching_fact_table_tids->size();
////                Timestamp end_cpu_fetch_join = getTimestamp();
////                std::cout << "CPU: #matching fact table rows: " <<
/// matching_fact_table_tids->size() << std::endl;
////#endif
////
////                //process on GPU
////                //                gpu::GPU_JoinIndexPtr gpu_join_index =
/// gpu::copy_JoinIndex_host_to_device(join_index);
////                gpu::GPU_JoinIndexPtr gpu_join_index =
/// GPU_Column_Cache::instance().getGPUJoinIndex(join_index);
////                gpu::GPU_BitmapPtr gpu_bitmap =
/// gpu::GPU_Operators::createBitmapOfMatchingTIDsFromJoinIndex(gpu_join_index,
/// gpu_input_tids);
////                //                        PositionListPtr
/// reference_matching_fact_table_tids=copy_PositionList_device_to_host(gpu_matching_fact_table_tids);
////                // assert(reference_matching_fact_table_tids!=NULL);
////                //
/// assert((*reference_matching_fact_table_tids)==(*matching_fact_table_tids));
////                Timestamp end_gpu_fetch_join = getTimestamp();
////
////                //check whether GPU operator was successfull
////                if (!gpu_bitmap) {
////                    //ok, GPU operator aborted, execute operator on CPU
////                    COGADB_ABORT_GPU_OPERATOR("Column_Bitmap_Fetch_Join");
////
////                    //PositionListPtr input_tids =
/// gpu::copy_PositionList_device_to_host(gpu_input_tids);
////                    PositionListOperator* pos_list_op =
/// dynamic_cast<PositionListOperator*> (this->child_.get());
////                    assert(pos_list_op != NULL);
////                    assert(pos_list_op->hasResultPositionList() ||
/// pos_list_op->hasCachedResult_GPU_PositionList());
////                    PositionListPtr input_tids;
////                    if (!pos_list_op->hasResultPositionList() &&
/// pos_list_op->hasCachedResult_GPU_PositionList()) {
////                        input_tids =
/// gpu::copy_PositionList_device_to_host(pos_list_op->getResult_GPU_PositionList());
////                    } else {
////                        input_tids = pos_list_op->getResultPositionList();
////                    }
////
////                    //FIXME: use bitmap fetch join routine
////                    PositionListPtr matching_fact_table_tids =
/// CoGaDB::fetchMatchingTIDsFromJoinIndex(join_index, input_tids);
////                    assert(matching_fact_table_tids != NULL);
////                    this->cpu_bitmap_ =
/// CoGaDB::CDK::convertPositionListToBitmap(matching_fact_table_tids,
/// join_index->second->getPositionList()->size());
////                    this->result_size_ = matching_fact_table_tids->size();
////                    //assert(cpu_tids_!=NULL);
////                    return true;
////                }
////
////#ifdef INVISIBLE_JOIN_USE_POSITIONLIST_ONLY_PLANS
////                //these kind of plans rely on a sorted output of the fetch
/// join
////                gpu::GPU_Operators::sortPositionList(gpu_bitmap);
////#endif
////
////                this->gpu_bitmap_ = gpu_bitmap;
////                this->result_size_ = gpu_bitmap->size();
////
////                //std::cout << "CPU Fetch Join: " <<
/// double(end_cpu_fetch_join-begin_cpu_fetch_join)/(1000*1000) << "ms" <<
/// std::endl;
////                std::cout << "GPU Bitmap Fetch Join: " <<
/// double(end_gpu_fetch_join - begin_gpu_fetch_join) / (1000 * 1000) << "ms" <<
/// std::endl;
////
////
////#ifdef VALIDATE_GPU_RESULTS_ON_CPU
////                if(gpu_bitmap){
////                    BitmapPtr result_bitmap =
/// gpu::copy_Bitmap_device_to_host(gpu_bitmap);
////                    assert(result_bitmap!=NULL);
////                    //PositionListPtr reference_matching_fact_table_tids =
/// gpu::copy_PositionList_device_to_host(gpu_bitmap);
////                    PositionListPtr reference_matching_fact_table_tids =
/// CDK::convertBitmapToPositionList(result_bitmap);
////                    if(! ((*reference_matching_fact_table_tids) ==
///(*matching_fact_table_tids))){
////                        COGADB_ERROR("Incorrect Result of GPU Bitmap Fetch
/// Join!","");
////                        std::cerr << "Number of Rows CPU: " <<
/// matching_fact_table_tids->size() << std::endl;
////                        std::cerr << "Number of Rows GPU: " <<
/// reference_matching_fact_table_tids->size() << std::endl;
////                        TID* cpu_array=matching_fact_table_tids->data();
////                        TID*
/// gpu_array=reference_matching_fact_table_tids->data();
////                        for(size_t i=0;
/// i<std::min(matching_fact_table_tids->size(),reference_matching_fact_table_tids->size());
///++i){
////                            if(cpu_array[i]!=gpu_array[i]){
////                                std::cerr << "Variant at Position " << i <<
///": CPU: " <<  cpu_array[i] << " GPU: " << gpu_array[i] << std::endl;
////                            }
////                        }
////                        COGADB_FATAL_ERROR("Terminate CoGaDB...","");
////                    }
////                    std::cout << "CPU Fetch Join: " <<
/// double(end_cpu_fetch_join - begin_cpu_fetch_join) / (1000 * 1000) << "ms" <<
/// std::endl;
////                }
////#endif
////
////                return true;
//            }
//
//                        void
//                        gpu_column_bitmap_fetch_join_operator::releaseInputData(){
//                                PositionListOperator* input_tids =
//                                dynamic_cast<PositionListOperator*>(this->child_.get());
//                                assert(input_tids!=NULL);
//                                input_tids->releaseResultData();
//                                //this->TypedOperator<ColumnPtr>::releaseInputData();
//                        }
//                        bool
//                        gpu_column_bitmap_fetch_join_operator::isInputDataCachedInGPU(){
//                                PositionListOperator* input_tids =
//                                dynamic_cast<PositionListOperator*>(this->child_.get());
//                                JoinIndexPtr join_index =
//                                JoinIndexes::instance().getJoinIndex(getTablebyName(pk_fk_join_pred_.join_pk_table_name),
//                                pk_fk_join_pred_.join_pk_column_name,
//                                getTablebyName(pk_fk_join_pred_.join_fk_table_name),
//                                pk_fk_join_pred_.join_fk_column_name);
//                                assert(input_tids!=NULL);
//                                assert(join_index!=NULL);
//                                return
//                                input_tids->hasCachedResult_GPU_PositionList()
//                                &&
//                                CoGaDB::GPU_Column_Cache::instance().isCached(join_index);
//                        }
//
//			gpu_column_bitmap_fetch_join_operator::~gpu_column_bitmap_fetch_join_operator()
//{}
//

}  // end namespace physical_operator

namespace logical_operator {
Logical_Column_Bitmap_Fetch_Join::Logical_Column_Bitmap_Fetch_Join(
    const PK_FK_Join_Predicate& pk_fk_join_pred,
    hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<ColumnPtr,
                     physical_operator::
                         map_init_function_column_bitmap_fetch_join_operator>(
          false, dev_constr),
      pk_fk_join_pred_(pk_fk_join_pred) {}

unsigned int Logical_Column_Bitmap_Fetch_Join::getOutputResultSize() const {
  return 10;
}

double Logical_Column_Bitmap_Fetch_Join::getCalculatedSelectivity() const {
  return 1.0;
}
std::string Logical_Column_Bitmap_Fetch_Join::getOperationName() const {
  return "COLUMN_BITMAP_FETCH_JOIN";
}
const PK_FK_Join_Predicate
Logical_Column_Bitmap_Fetch_Join::getPK_FK_Join_Predicate() const {
  return pk_fk_join_pred_;
}

std::string Logical_Column_Bitmap_Fetch_Join::toString(bool verbose) const {
  std::string result = "COLUMN_BITMAP_FETCH_JOIN";
  if (verbose) {
    result += " ON JOIN_INDEX(";
    result += this->pk_fk_join_pred_.join_pk_table_name;
    result += ".";
    result += this->pk_fk_join_pred_.join_pk_column_name;
    result += ",";
    result += this->pk_fk_join_pred_.join_fk_table_name;
    result += ".";
    result += this->pk_fk_join_pred_.join_fk_column_name;
    result += ")";
  }
  return result;
}

const hype::Tuple Logical_Column_Bitmap_Fetch_Join::getFeatureVector() const {
  hype::Tuple t;
  if (this->left_) {  // if left child is valid (has to be by convention!), add
                      // input data size
    // if we already know the correct input data size, because the child node
    // was already executed
    // during query chopping, we use the real cardinality, other wise we call
    // the estimator
    if (this->left_->getPhysicalOperator()) {
      t.push_back(this->left_->getPhysicalOperator()
                      ->getResultSize());  // ->result_size_;

#ifdef HYPE_INCLUDE_INPUT_DATA_LOCALITY_IN_FEATURE_VECTOR
      t.push_back(const_cast<Logical_Column_Bitmap_Fetch_Join*>(this)
                      ->isInputDataCachedInGPU());
#endif
    } else {
      t.push_back(this->left_->getOutputResultSize());
    }
  } else {
    HYPE_FATAL_ERROR("Invalid Left Child!", std::cout);
  }

  // size_t number_of_dimension_tids = this->left_->getOutputResultSize();
  // TablePtr pk_table =
  // getTablebyName(this->getPK_FK_Join_Predicate().join_pk_table_name);
  TablePtr fk_table =
      getTablebyName(this->getPK_FK_Join_Predicate().join_fk_table_name);
  size_t number_of_fact_table_tids = fk_table->getNumberofRows();

  // t.push_back(double(number_of_dimension_tids));
  t.push_back(double(number_of_fact_table_tids));

  return t;
}

bool Logical_Column_Bitmap_Fetch_Join::isInputDataCachedInGPU() {
  JoinIndexPtr join_index = JoinIndexes::instance().getJoinIndex(
      getTablebyName(pk_fk_join_pred_.join_pk_table_name),
      pk_fk_join_pred_.join_pk_column_name,
      getTablebyName(pk_fk_join_pred_.join_fk_table_name),
      pk_fk_join_pred_.join_fk_column_name);
  assert(join_index != NULL);
  hype::ProcessingDeviceMemoryID mem_id = getMemoryIDForDeviceID(0);
  return DataCacheManager::instance().getDataCache(mem_id).isCached(join_index);
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
