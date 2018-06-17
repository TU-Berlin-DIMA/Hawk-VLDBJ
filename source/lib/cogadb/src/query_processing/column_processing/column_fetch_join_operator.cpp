
#include <query_processing/column_processing/column_fetch_join_operator.hpp>

#include <lookup_table/join_index.hpp>
#include <util/hardware_detector.hpp>

#include <util/time_measurement.hpp>

#include <tbb/parallel_sort.h>
#include <core/processor_data_cache.hpp>
#include <query_processing/invisible_join_operator.hpp>

#include <backends/gpu/memory_cost_models.hpp>
#include <backends/processor_backend.hpp>

namespace CoGaDB {

namespace query_processing {
// Map_Init_Function
// init_function_column_fetch_join_operator=physical_operator::map_init_function_column_fetch_join_operator;

PK_FK_Join_Predicate::PK_FK_Join_Predicate(
    const std::string& join_pk_table_name_a,
    const std::string& join_pk_column_name_a,
    const std::string& join_fk_table_name_a,
    const std::string& join_fk_column_name_a)
    : join_pk_table_name(join_pk_table_name_a),
      join_pk_column_name(join_pk_column_name_a),
      join_fk_table_name(join_fk_table_name_a),
      join_fk_column_name(join_fk_column_name_a) {}

namespace physical_operator {

column_processing::cpu::TypedOperatorPtr create_column_fetch_join_operator(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr) {
  logical_operator::Logical_Column_Fetch_Join& log_sort_ref =
      static_cast<logical_operator::Logical_Column_Fetch_Join&>(logical_node);
  // std::cout << "create SCAN Operator!" << std::endl;
  /*
  if(!left_child) {
          std::cout << "Error!" << std::endl;
          exit(-1);
  }
  assert(right_child==NULL); //unary operator
   */
  //				return column_processing::cpu::TypedOperatorPtr(
  // new
  // column_fetch_join_operator(sched_dec,
  // log_sort_ref.getTableName(),log_sort_ref.getColumnName()) );
  return column_processing::cpu::TypedOperatorPtr(
      new column_fetch_join_operator(sched_dec, left_child,
                                     log_sort_ref.getPK_FK_Join_Predicate()));
}

//            column_processing::cpu::TypedOperatorPtr
//            create_gpu_column_fetch_join_operator(column_processing::cpu::TypedLogicalNode&
//            logical_node, const hype::SchedulingDecision& sched_dec,
//            column_processing::cpu::TypedOperatorPtr left_child,
//            column_processing::cpu::TypedOperatorPtr) {
//                logical_operator::Logical_Column_Fetch_Join& log_sort_ref =
//                static_cast<logical_operator::Logical_Column_Fetch_Join&>
//                (logical_node);
//                //std::cout << "create SCAN Operator!" << std::endl;
//                /*
//                if(!left_child) {
//                        std::cout << "Error!" << std::endl;
//                        exit(-1);
//                }
//                assert(right_child==NULL); //unary operator
//                 */
//                //				return
//                column_processing::cpu::TypedOperatorPtr( new
//                gpu_column_fetch_join_operator(sched_dec,
//                log_sort_ref.getTableName(),log_sort_ref.getColumnName()) );
//                return column_processing::cpu::TypedOperatorPtr(new
//                gpu_column_fetch_join_operator(sched_dec, left_child,
//                log_sort_ref.getPK_FK_Join_Predicate()));
//            }

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_column_fetch_join_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function! (COLUMN_FETCH_JOIN OPERATION)"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SCAN","TABLE_SCAN",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_COLUMN_FETCH_JOIN", "COLUMN_FETCH_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_COLUMN_FETCH_JOIN", "COLUMN_FETCH_JOIN", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_COLUMN_FETCH_JOIN", "COLUMN_FETCH_JOIN",
      hype::Multilinear_Fitting_2D, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_COLUMN_FETCH_JOIN", "COLUMN_FETCH_JOIN",
      hype::Multilinear_Fitting_2D, hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

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
          &gpu::GPU_Operators_Memory_Cost_Models::columnFetchJoin);
#endif
    }
  }

  map["CPU_COLUMN_FETCH_JOIN"] = create_column_fetch_join_operator;
  map["GPU_COLUMN_FETCH_JOIN"] =
      create_column_fetch_join_operator;  // create_gpu_column_fetch_join_operator;
  // map["GPU_Algorithm"]=create_GPU_SORT_Operator;
  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

column_fetch_join_operator::column_fetch_join_operator(
    const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
    PK_FK_Join_Predicate pk_fk_join_pred)
    : UnaryOperator<ColumnPtr, ColumnPtr>(sched_dec, child),
      PositionListOperator(),
      pk_fk_join_pred_(pk_fk_join_pred) {}

bool column_fetch_join_operator::execute() {
  if (!quiet && verbose && debug)
    std::cout << "Execute Column_Fetch_Join" << std::endl;

  PositionListPtr input_tids;
  PositionListOperator* pos_list_op =
      dynamic_cast<PositionListOperator*>(this->child_.get());
  assert(pos_list_op != NULL);
  input_tids = pos_list_op->getResultPositionList();
  assert(input_tids != NULL);

  hype::ProcessingDeviceID id =
      sched_dec_.getDeviceSpecification().getProcessingDeviceID();
  JoinIndexPtr join_index = JoinIndexes::instance().getJoinIndex(
      getTablebyName(pk_fk_join_pred_.join_pk_table_name),
      pk_fk_join_pred_.join_pk_column_name,
      getTablebyName(pk_fk_join_pred_.join_fk_table_name),
      pk_fk_join_pred_.join_fk_column_name);

  ProcessorSpecification proc_spec(id);
  FetchJoinParam param(proc_spec);
  PositionListPtr placed_input_tids = copy_if_required(input_tids, proc_spec);
  if (!placed_input_tids) {
    return false;
  }

  JoinIndexPtr placed_join_index = copy_if_required(join_index, proc_spec);
  ProcessorBackend<TID>* backend =
      ProcessorBackend<TID>::get(proc_spec.proc_id);
  PositionListPtr matching_fact_table_tids =
      backend->tid_fetch_join(placed_join_index, placed_input_tids, param);

//                PositionListPtr matching_fact_table_tids =
//                CoGaDB::fetchMatchingTIDsFromJoinIndex(join_index,
//                input_tids);

#ifdef INVISIBLE_JOIN_USE_POSITIONLIST_ONLY_PLANS
  // these kind of plans rely on a sorted output of the fetch join
  tbb::parallel_sort(matching_fact_table_tids->begin(),
                     matching_fact_table_tids->end());
#endif

  this->tids_ = matching_fact_table_tids;
  this->result_size_ = matching_fact_table_tids->size();

  return true;
}

column_fetch_join_operator::~column_fetch_join_operator() {}

//
//
//            gpu_column_fetch_join_operator::gpu_column_fetch_join_operator(const
//            hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
//            PK_FK_Join_Predicate pk_fk_join_pred) : UnaryOperator<ColumnPtr,
//            ColumnPtr>(sched_dec, child), PositionListOperator(),
//            pk_fk_join_pred_(pk_fk_join_pred) {
//            }
//
//            bool gpu_column_fetch_join_operator::execute() {
//                if (!quiet && verbose && debug) std::cout << "Execute
//                Column_Fetch_Join" << std::endl;
//
//
//
////                COGADB_EXECUTE_GPU_OPERATOR("Column_Fetch_Join");
//
//                return false;
////                //assert(this->getInputData()!=NULL);
////                //this->getInputData()->getName() << std::endl;
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
////                //CPU cross check
////#ifdef VALIDATE_GPU_RESULTS_ON_CPU
////                Timestamp begin_cpu_fetch_join = getTimestamp();
////                PositionListPtr input_tids =
/// gpu::copy_PositionList_device_to_host(gpu_input_tids);
////                //JoinIndexPtr join_index =
/// JoinIndexes::instance().getJoinIndex(getTablebyName(pk_fk_join_pred_.join_pk_table_name),
/// pk_fk_join_pred_.join_pk_column_name,
/// getTablebyName(pk_fk_join_pred_.join_fk_table_name),pk_fk_join_pred_.join_fk_column_name);
////                PositionListPtr matching_fact_table_tids =
/// CoGaDB::fetchMatchingTIDsFromJoinIndex(join_index,input_tids);
////                this->cpu_tids_=matching_fact_table_tids;
////                this->result_size_=matching_fact_table_tids->size();
////                Timestamp end_cpu_fetch_join = getTimestamp();
////                std::cout << "CPU: #matching fact table rows: " <<
/// matching_fact_table_tids->size() << std::endl;
////#endif
////
////                //process on GPU
//////                gpu::GPU_JoinIndexPtr gpu_join_index =
/// gpu::copy_JoinIndex_host_to_device(join_index);
////                gpu::GPU_JoinIndexPtr gpu_join_index =
/// GPU_Column_Cache::instance().getGPUJoinIndex(join_index);
////                gpu::GPU_PositionlistPtr gpu_matching_fact_table_tids =
/// gpu::GPU_Operators::fetchMatchingTIDsFromJoinIndex(gpu_join_index,
/// gpu_input_tids);
////                //                        PositionListPtr
/// reference_matching_fact_table_tids=copy_PositionList_device_to_host(gpu_matching_fact_table_tids);
////                // assert(reference_matching_fact_table_tids!=NULL);
////                //
/// assert((*reference_matching_fact_table_tids)==(*matching_fact_table_tids));
////                Timestamp end_gpu_fetch_join = getTimestamp();
////
////                //check whether GPU operator was successfull
////                if(!gpu_matching_fact_table_tids){
////                    //ok, GPU operator aborted, execute operator on CPU
////                    COGADB_ABORT_GPU_OPERATOR("Column_Fetch_Join");
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
////                    //assert(input_tids != NULL);
////
////                    //JoinIndexPtr join_index =
/// JoinIndexes::instance().getJoinIndex(getTablebyName(pk_fk_join_pred_.join_pk_table_name),
/// pk_fk_join_pred_.join_pk_column_name,
/// getTablebyName(pk_fk_join_pred_.join_fk_table_name),pk_fk_join_pred_.join_fk_column_name);
////                    PositionListPtr matching_fact_table_tids =
/// CoGaDB::fetchMatchingTIDsFromJoinIndex(join_index,input_tids);
////                    assert(matching_fact_table_tids!=NULL);
////                    this->tids_=matching_fact_table_tids;
////                    this->result_size_=matching_fact_table_tids->size();
////                    //assert(cpu_tids_!=NULL);
////                    return true;
////                }
////
////#ifdef INVISIBLE_JOIN_USE_POSITIONLIST_ONLY_PLANS
////                //these kind of plans rely on a sorted output of the fetch
/// join
//// gpu::GPU_Operators::sortPositionList(gpu_matching_fact_table_tids);
////#endif
////
////                this->gpu_tids_ = gpu_matching_fact_table_tids;
////                this->result_size_ = gpu_matching_fact_table_tids->size();
////
////                //std::cout << "CPU Fetch Join: " <<
/// double(end_cpu_fetch_join-begin_cpu_fetch_join)/(1000*1000) << "ms" <<
/// std::endl;
////                std::cout << "GPU Fetch Join: " << double(end_gpu_fetch_join
///- begin_gpu_fetch_join) / (1000 * 1000) << "ms" << std::endl;
////
////
////#ifdef VALIDATE_GPU_RESULTS_ON_CPU
////                PositionListPtr reference_matching_fact_table_tids =
/// gpu::copy_PositionList_device_to_host(gpu_matching_fact_table_tids);
//// assert((*reference_matching_fact_table_tids)==(*matching_fact_table_tids));
////                std::cout << "CPU Fetch Join: " <<
/// double(end_cpu_fetch_join-begin_cpu_fetch_join)/(1000*1000) << "ms" <<
/// std::endl;
////#endif
////
//////                                        Timestamp begin_gpu_fetch_join =
/// getTimestamp();
//////                                        //gpu::GPU_JoinIndexPtr
/// gpu_join_index=gpu::copy_JoinIndex_host_to_device(join_index);
//////                                        gpu::GPU_PositionlistPtr
/// gpu_input_tids=gpu::copy_PositionList_host_to_device(input_tids);
//////                		        gpu::GPU_PositionlistPtr
/// gpu_matching_fact_table_tids =
/// gpu::GPU_Operators::fetchMatchingTIDsFromJoinIndex(gpu_join_index,
/// gpu_input_tids);
//////                                        PositionListPtr
/// reference_matching_fact_table_tids=copy_PositionList_device_to_host(gpu_matching_fact_table_tids);
////// assert(reference_matching_fact_table_tids!=NULL);
//////
/// assert((*reference_matching_fact_table_tids)==(*matching_fact_table_tids));
//////                                        Timestamp end_gpu_fetch_join =
/// getTimestamp();
////
////
////                                        //std::cout << "GPU Fetch Join: " <<
/// double(end_gpu_fetch_join-begin_gpu_fetch_join)/(1000*1000) << "ms" <<
/// std::endl;
////
////                return true;
//            }

//            void gpu_column_fetch_join_operator::releaseInputData(){
//                    PositionListOperator* input_tids =
//                    dynamic_cast<PositionListOperator*>(this->child_.get());
//                    assert(input_tids!=NULL);
//                    input_tids->releaseResultData();
//                    //this->TypedOperator<ColumnPtr>::releaseInputData();
//            }
//            bool gpu_column_fetch_join_operator::isInputDataCachedInGPU(){
//                    PositionListOperator* input_tids =
//                    dynamic_cast<PositionListOperator*>(this->child_.get());
//                    JoinIndexPtr join_index =
//                    JoinIndexes::instance().getJoinIndex(getTablebyName(pk_fk_join_pred_.join_pk_table_name),
//                    pk_fk_join_pred_.join_pk_column_name,
//                    getTablebyName(pk_fk_join_pred_.join_fk_table_name),
//                    pk_fk_join_pred_.join_fk_column_name);
//                    assert(input_tids!=NULL);
//                    assert(join_index!=NULL);
//                    return input_tids->hasCachedResult_GPU_PositionList() &&
//                    CoGaDB::GPU_Column_Cache::instance().isCached(join_index);
//            }
//
//            gpu_column_fetch_join_operator::~gpu_column_fetch_join_operator()
//            {
//            }

}  // end namespace physical_operator

namespace logical_operator {

Logical_Column_Fetch_Join::Logical_Column_Fetch_Join(
    const PK_FK_Join_Predicate& pk_fk_join_pred,
    hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<
          ColumnPtr,
          physical_operator::map_init_function_column_fetch_join_operator>(
          false, dev_constr),
      pk_fk_join_pred_(pk_fk_join_pred) {}

unsigned int Logical_Column_Fetch_Join::getOutputResultSize() const {
  return 10;
}

double Logical_Column_Fetch_Join::getCalculatedSelectivity() const {
  return 1.0;
}

std::string Logical_Column_Fetch_Join::getOperationName() const {
  return "COLUMN_FETCH_JOIN";
}

const PK_FK_Join_Predicate Logical_Column_Fetch_Join::getPK_FK_Join_Predicate()
    const {
  return pk_fk_join_pred_;
}

std::string Logical_Column_Fetch_Join::toString(bool verbose) const {
  std::string result = "COLUMN_FETCH_JOIN";
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

//            const hype::Tuple Logical_Column_Fetch_Join::getFeatureVector()
//            const{
//
//                size_t number_of_dimension_tids =
//                this->left_->getOutputResultSize();
//                //TablePtr pk_table =
//                getTablebyName(this->getPK_FK_Join_Predicate().join_pk_table_name);
//                TablePtr fk_table =
//                getTablebyName(this->getPK_FK_Join_Predicate().join_fk_table_name);
//                size_t number_of_fact_table_tids =
//                fk_table->getNumberofRows();
//
//                hype::Tuple t;
//                t.push_back(double(number_of_dimension_tids));
//                t.push_back(double(number_of_fact_table_tids));
//
//                return t;
//            }

const hype::Tuple Logical_Column_Fetch_Join::getFeatureVector() const {
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
      //                        JoinIndexPtr join_index =
      //                        JoinIndexes::instance().getJoinIndex(getTablebyName(pk_fk_join_pred_.join_pk_table_name),
      //                        pk_fk_join_pred_.join_pk_column_name,
      //                        getTablebyName(pk_fk_join_pred_.join_fk_table_name),
      //                        pk_fk_join_pred_.join_fk_column_name);
      //                        assert(join_index!=NULL);
      t.push_back(const_cast<Logical_Column_Fetch_Join*>(this)
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

bool Logical_Column_Fetch_Join::isInputDataCachedInGPU() {
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
