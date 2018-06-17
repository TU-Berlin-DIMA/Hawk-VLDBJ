
#include <core/lookup_array.hpp>
#include <query_processing/column_processing/cpu_column_constant_filter_operator.hpp>
#include <util/hardware_detector.hpp>

#include <core/data_dictionary.hpp>

#include <query_processing/query_processor.hpp>
#include "backends/gpu/memory_cost_models.hpp"
#include "compression/dictionary_compressed_column.hpp"
#include "core/processor_data_cache.hpp"

namespace CoGaDB {

namespace query_processing {
// Map_Init_Function
// init_function_column_constant_filter_operator=physical_operator::map_init_function_column_constant_filter_operator;

namespace physical_operator {

column_processing::cpu::TypedOperatorPtr
create_CPU_column_constant_filter_operator(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr left_child,
    column_processing::cpu::TypedOperatorPtr right_child) {
  logical_operator::Logical_Column_Constant_Filter& log_sort_ref =
      static_cast<logical_operator::Logical_Column_Constant_Filter&>(
          logical_node);
  // std::cout << "create SCAN Operator!" << std::endl;

  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator

  return column_processing::cpu::TypedOperatorPtr(
      new CPU_column_constant_filter_operator(sched_dec, left_child,
                                              log_sort_ref.getPredicate()));
}

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_column_constant_filter_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function! (COLUMN_CONSTANT_FILTER OPERATION)"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SCAN","TABLE_SCAN",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_COLUMN_CONSTANT_FILTER", "COLUMN_CONSTANT_FILTER",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_COLUMN_CONSTANT_FILTER", "COLUMN_CONSTANT_FILTER",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_COLUMN_CONSTANT_FILTER", "COLUMN_CONSTANT_FILTER",
      hype::Least_Squares_1D, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_COLUMN_CONSTANT_FILTER", "COLUMN_CONSTANT_FILTER",
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
          &CoGaDB::gpu::GPU_Operators_Memory_Cost_Models::columnSelection);
#endif
    }
  }
  map["CPU_COLUMN_CONSTANT_FILTER"] =
      create_CPU_column_constant_filter_operator;
  map["GPU_COLUMN_CONSTANT_FILTER"] =
      create_CPU_column_constant_filter_operator;  // create_GPU_column_constant_filter_operator;
  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

CPU_column_constant_filter_operator::CPU_column_constant_filter_operator(
    const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
    const Predicate& pred)
    : UnaryOperator<ColumnPtr, ColumnPtr>(sched_dec, child),
      PositionListOperator(),
      pred_(pred) {
  // this->result_=getTablebyName(table_name_);
}

bool CPU_column_constant_filter_operator::execute() {
  if (!quiet && verbose && debug)
    std::cout << "Execute Filter Operation" << std::endl;

  assert(pred_.getPredicateType() == ValueConstantPredicate);
  ColumnPtr col = this->getInputData();

  hype::ProcessingDeviceID id =
      sched_dec_.getDeviceSpecification().getProcessingDeviceID();
  ProcessorSpecification proc_spec(id);
  SelectionParam param(proc_spec, ValueConstantPredicate, pred_.getConstant(),
                       pred_.getValueComparator());

  col = copy_if_required(col, proc_spec);
  if (!col) {
    this->has_aborted_ = true;
    return false;
  }
  PositionListPtr tids = col->selection(param);
  if (!tids) return false;
  // create a LookupArray
  // this->result_=createLookupArrayForColumn(col,tids);
  this->tids_ = tids;
  result_size_ = tids->size();
  return true;
}

CPU_column_constant_filter_operator::~CPU_column_constant_filter_operator() {}

}  // end namespace physical_operator

namespace logical_operator {

Logical_Column_Constant_Filter::Logical_Column_Constant_Filter(
    const Predicate& pred, hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<
          ColumnPtr,
          physical_operator::map_init_function_column_constant_filter_operator>(
          false, dev_constr),
      pred_(pred) {
  if (pred_.getPredicateType() == ValueConstantPredicate) {
    std::list<std::pair<ColumnPtr, TablePtr> > columns =
        DataDictionary::instance().getColumnsforColumnName(
            pred_.getColumn1Name());
    // assume unique column names
    if (columns.size() != 1) {
      if (columns.size() > 1) {
        COGADB_FATAL_ERROR("Column name not unique! Found Column "
                               << pred_.getColumn1Name() << " more than once ("
                               << columns.size() << ") in database!",
                           "");
      } else {
        //                    		COGADB_FATAL_ERROR("Column " <<
        //                    pred_.getColumn1Name() << " not found in
        //                    database!","");
        // we did not find this column, so we cannot assume it is
        // GPU compatible, and hence, we set a CPU_ONLY constraint
        this->dev_constr_ = hype::DeviceConstraint(hype::CPU_ONLY);
      }
    }
    //                    assert(columns.size()==1);
    if (columns.size() == 1 && columns.front().first->getType() == VARCHAR &&
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

unsigned int Logical_Column_Constant_Filter::getOutputResultSize() const {
  return getCalculatedSelectivity() * this->getLeft()->getOutputResultSize();
}

double Logical_Column_Constant_Filter::getCalculatedSelectivity() const {
  return 0.3;
}

std::string Logical_Column_Constant_Filter::getOperationName() const {
  return "COLUMN_CONSTANT_FILTER";
}

const Predicate& Logical_Column_Constant_Filter::getPredicate() const {
  return pred_;
}

std::string Logical_Column_Constant_Filter::toString(bool verbose) const {
  std::string result = "COLUMN_CONSTANT_FILTER";
  if (verbose) {
    result += " (";
    result += pred_.toString();
    result += ")";
  }
  return result;
}

bool Logical_Column_Constant_Filter::isInputDataCachedInGPU() {
  std::list<std::pair<ColumnPtr, TablePtr> > columns =
      DataDictionary::instance().getColumnsforColumnName(
          this->pred_.getColumn1Name());
  assert(columns.size() <= 1);
  // if we do not find the column in the database,
  // we assume it is not cached on the GPU
  if (columns.empty()) return false;
  ColumnPtr col = columns.front().first;
  assert(col != NULL);
  hype::ProcessingDeviceMemoryID mem_id = getMemoryIDForDeviceID(0);
  return DataCacheManager::instance().getDataCache(mem_id).isCached(col);
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
