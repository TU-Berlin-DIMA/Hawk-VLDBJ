
#include <core/lookup_array.hpp>
#include <query_processing/column_processing/column_bitmap_selection_operator.hpp>
#include <util/hardware_detector.hpp>

#include <core/data_dictionary.hpp>

#include <hardware_optimizations/primitives.hpp>
#include "compression/dictionary_compressed_column.hpp"
#include "core/gpu_column_cache.hpp"

namespace CoGaDB {

namespace query_processing {
// Map_Init_Function
// init_function_column_bitmap_selection_operator=physical_operator::map_init_function_column_bitmap_selection_operator;

namespace physical_operator {

column_processing::cpu::TypedOperatorPtr
create_CPU_column_bitmap_selection_operator(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr right_child) {
  logical_operator::Logical_Column_Bitmap_Selection& log_sort_ref =
      static_cast<logical_operator::Logical_Column_Bitmap_Selection&>(
          logical_node);
  // std::cout << "create SCAN Operator!" << std::endl;

  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator

  return column_processing::cpu::TypedOperatorPtr(
      new CPU_column_bitmap_selection_operator(sched_dec, left_child,
                                               log_sort_ref.getPredicate()));
}

column_processing::cpu::TypedOperatorPtr
create_GPU_column_bitmap_selection_operator(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr right_child) {
  logical_operator::Logical_Column_Bitmap_Selection& log_sort_ref =
      static_cast<logical_operator::Logical_Column_Bitmap_Selection&>(
          logical_node);
  // std::cout << "create SCAN Operator!" << std::endl;

  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator

  return column_processing::cpu::TypedOperatorPtr(
      new GPU_column_bitmap_selection_operator(sched_dec, left_child,
                                               log_sort_ref.getPredicate()));
}

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_column_bitmap_selection_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
  if (!quiet)
    std::cout
        << "calling map init function! (COLUMN_BITMAP_SELECTION OPERATION)"
        << std::endl;
  // hype::Scheduler::instance().addAlgorithm("SCAN","TABLE_SCAN",hype::CPU,"Least
  // Squares 1D","Periodic Recomputation");
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_COLUMN_BITMAP_SELECTION", "COLUMN_BITMAP_SELECTION",
      hype::Least_Squares_1D, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_COLUMN_BITMAP_SELECTION", "COLUMN_BITMAP_SELECTION",
      hype::Least_Squares_1D, hype::Periodic);

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu,
                                               dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_gpu,
                                               dev_specs[i]);
    }
  }
  map["CPU_COLUMN_BITMAP_SELECTION"] =
      create_CPU_column_bitmap_selection_operator;
  /*TODO: implement bitmap selection operator*/
  // map["GPU_COLUMN_BITMAP_SELECTION"] =
  // create_GPU_column_bitmap_selection_operator;
  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

CPU_column_bitmap_selection_operator::CPU_column_bitmap_selection_operator(
    const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
    const Predicate& pred)
    : UnaryOperator<ColumnPtr, ColumnPtr>(sched_dec, child),
      BitmapOperator(),
      pred_(pred) {
  // this->result_=getTablebyName(table_name_);
}

bool CPU_column_bitmap_selection_operator::execute() {
  if (!quiet && verbose && debug)
    std::cout << "Execute Filter Operation" << std::endl;

  assert(pred_.getPredicateType() == ValueConstantPredicate);
  ColumnPtr col = this->getInputData();
  // std::cout << "Input Column: " << col.get() << std::endl;
  if (col == NULL)
    COGADB_FATAL_ERROR(
        "Column_BITMAP_SELECTION: Invalid Input Column! NULL Pointer...", "");
  hype::ProcessingDeviceID id =
      sched_dec_.getDeviceSpecification().getProcessingDeviceID();
  ProcessorSpecification proc_spec(id);
  SelectionParam param(proc_spec, ValueConstantPredicate, pred_.getConstant(),
                       pred_.getValueComparator());
  this->cpu_bitmap_ = col->bitmap_selection(param);
  this->result_size_ = CDK::selection::countSetBitsInBitmap(
      this->cpu_bitmap_->data(), this->cpu_bitmap_->size());

  if (this->cpu_bitmap_) {
    return true;
  } else {
    return false;
  }
}

CPU_column_bitmap_selection_operator::~CPU_column_bitmap_selection_operator() {}

/*GPU Operation*/
GPU_column_bitmap_selection_operator::GPU_column_bitmap_selection_operator(
    const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
    const Predicate& pred)
    : UnaryOperator<ColumnPtr, ColumnPtr>(sched_dec, child),
      BitmapOperator(),
      pred_(pred) {
  // this->result_=getTablebyName(table_name_);
}

bool GPU_column_bitmap_selection_operator::execute() {
  if (!quiet && verbose && debug)
    std::cout << "Execute Filter Operation" << std::endl;

  //                assert(pred_.getPredicateType()==ValueConstantPredicate);
  //                //this->result_=this->getInputData()->selection(pred_.getConstant(),pred_.getValueComparator());
  //                ColumnPtr col = this->getInputData();
  //#ifndef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
  //                assert(col->getType()!=VARCHAR);
  //#else
  //                    uint32_t filter_id=std::numeric_limits<uint32_t>::max();
  //                    if(col->getType()==VARCHAR){
  //                    shared_pointer_namespace::shared_ptr<DictionaryCompressedColumn<std::string>
  //                    > host_col =
  //                    shared_pointer_namespace::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>
  //                    > (col);
  //                    assert(host_col!=NULL);
  //                    std::string filter_val =
  //                    boost::any_cast<std::string>(pred_.getConstant());
  //                    std::pair<bool,uint32_t> ret =
  //                    host_col->getDictionaryID(filter_val);
  //                    //if filter_Val exist in dictionary, set filter_id to
  //                    returned id
  //                    if(ret.first){
  //                        filter_id=ret.second;
  //                    }
  //                    }
  //                    //comp_val=boost::any(filter_id);
  //#endif
  //                //ensure that col is a dense value array, other columns
  //                cannot be processed on GPU
  //                if(col->getType()!=VARCHAR && !col->isMaterialized())
  //                col=col->materialize();
  //                //gpu::copy_column_host_to_device(ColumnPtr host_column)
  //                gpu::GPU_Base_ColumnPtr gpu_col =
  //                CoGaDB::GPU_Column_Cache::instance().getGPUColumn(col);
  //                if(!gpu_col){
  //                    std::cout << "Error! In
  //                    GPU_column_bitmap_selection_operator::execute: Data
  //                    Transfer to GPU Failed!" << std::endl;
  //                    return false;
  //                }
  //                gpu::GPU_PositionlistPtr gpu_tids;
  //#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
  //                //if it is a VARCHAR, then translate the query from string
  //                to the compressed value
  //                //We only support EQUALITY evaluation on columns!!!
  //                if(gpu_col->getType()==VARCHAR){
  //                    gpu_tids =
  //                    gpu::GPU_Operators::selection(gpu_col,boost::any(filter_id),
  //                    pred_.getValueComparator());
  //                }else{
  //                    gpu_tids =
  //                    gpu::GPU_Operators::selection(gpu_col,pred_.getConstant(),
  //                    pred_.getValueComparator());
  //                }
  //#else
  //                gpu_tids =
  //                gpu::GPU_Operators::selection(gpu_col,pred_.getConstant(),
  //                pred_.getValueComparator());
  //#endif
  //
  //                if(!gpu_tids){
  //                    std::cout << "Error! In
  //                    GPU_column_bitmap_selection_operator::execute: GPU
  //                    Selection Operator Failed!" << std::endl;
  //                    return false;
  //                }
  //                PositionListPtr tids =
  //                gpu::copy_PositionList_device_to_host(gpu_tids);
  //                if(!tids){
  //                    std::cout << "Error! In
  //                    GPU_column_bitmap_selection_operator::execute: Data
  //                    Transfer to CPU Failed!" << std::endl;
  //                    return false;
  //                }
  //                if(!CoGaDB::quiet && CoGaDB::verbose)
  //                std::cout << "GPU SCAN on Column " << col->getName() << "
  //                returned " << tids->size() << "rows" << std::endl;
  //                //create a LookupArray
  //                this->result_=createLookupArrayForColumn(col,tids);

  if (this->result_) {
    return true;
  } else {
    return false;
  }
}

GPU_column_bitmap_selection_operator::~GPU_column_bitmap_selection_operator() {}

}  // end namespace physical_operator

namespace logical_operator {

Logical_Column_Bitmap_Selection::Logical_Column_Bitmap_Selection(
    const Predicate& pred, hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<ColumnPtr,
                     physical_operator::
                         map_init_function_column_bitmap_selection_operator>(
          false, dev_constr),
      pred_(pred) {
  if (pred_.getPredicateType() == ValueConstantPredicate) {
    std::list<std::pair<ColumnPtr, TablePtr> > columns =
        DataDictionary::instance().getColumnsforColumnName(
            pred_.getColumn1Name());
    // assume unique column names
    assert(columns.size() == 1);
    if (columns.front().first->getType() == VARCHAR &&
        pred_.getValueComparator() != EQUAL) {
      if (!quiet && verbose)
        std::cout << "Set Device Constraint CPU ONLY for operator '"
                  << this->toString(true) << "'" << std::endl;
      this->dev_constr_ = hype::DeviceConstraint(hype::CPU_ONLY);
    }
    //                    std::list<std::pair<ColumnPtr,TablePtr> >::iterator
    //                    it;
    //                    for(it=columns.begin();it!=columns.end();++it){
    //
    //                    }
  }
}

unsigned int Logical_Column_Bitmap_Selection::getOutputResultSize() const {
  return 10;
}

double Logical_Column_Bitmap_Selection::getCalculatedSelectivity() const {
  return 0.3;
}

std::string Logical_Column_Bitmap_Selection::getOperationName() const {
  return "COLUMN_BITMAP_SELECTION";
}

const Predicate& Logical_Column_Bitmap_Selection::getPredicate() const {
  return pred_;
}

std::string Logical_Column_Bitmap_Selection::toString(bool verbose) const {
  std::string result = "COLUMN_BITMAP_SELECTION";
  if (verbose) {
    result += " (";
    result += pred_.toString();
    result += ")";
  }
  return result;
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
