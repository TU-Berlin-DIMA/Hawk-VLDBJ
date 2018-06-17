
#include <query_processing/column_processing/cpu_columnscan_operator.hpp>
#include <util/hardware_detector.hpp>

namespace CoGaDB {

namespace query_processing {
// Map_Init_Function
// init_function_column_scan_operator=physical_operator::map_init_function_column_scan_operator;

namespace physical_operator {

column_processing::cpu::TypedOperatorPtr create_column_scan_operator(
    column_processing::cpu::TypedLogicalNode& logical_node,
    const hype::SchedulingDecision& sched_dec,
    column_processing::cpu::TypedOperatorPtr,
    column_processing::cpu::TypedOperatorPtr) {
  logical_operator::Logical_Column_Scan& log_sort_ref =
      static_cast<logical_operator::Logical_Column_Scan&>(logical_node);
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
  // column_scan_operator(sched_dec,
  // log_sort_ref.getTableName(),log_sort_ref.getColumnName()) );
  return column_processing::cpu::TypedOperatorPtr(new column_scan_operator(
      sched_dec, log_sort_ref.getTablePtr(), log_sort_ref.getColumnName()));
}

column_processing::cpu::Physical_Operator_Map_Ptr
map_init_function_column_scan_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  column_processing::cpu::Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function! (CPU_COLUMN_SCAN OPERATION)"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SCAN","TABLE_SCAN",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_COLUMN_SCAN", "CPU_COLUMN_SCAN", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_COLUMN_SCAN", "CPU_COLUMN_SCAN", hype::Least_Squares_1D,
      hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu,
                                               dev_specs[i]);
    }
  }

  map["CPU_COLUMN_SCAN"] = create_column_scan_operator;
  // map["GPU_Algorithm"]=create_GPU_SORT_Operator;
  return column_processing::cpu::Physical_Operator_Map_Ptr(
      new column_processing::cpu::Physical_Operator_Map(map));
}

column_scan_operator::column_scan_operator(
    const hype::SchedulingDecision& sched_dec, const std::string& table_name,
    const std::string& column_name)
    : UnaryOperator<ColumnPtr, ColumnPtr>(
          sched_dec, column_processing::cpu::TypedOperatorPtr()),
      table_name_(table_name),
      column_name_(column_name),
      table_ptr_(getTablebyName(table_name)) {
  if (table_ptr_ == NULL) {
    COGADB_FATAL_ERROR("TablePtr is zero, cannot scan column of unknown Table!",
                       "");
  }
  // this->result_=getTablebyName(table_name_);
}

column_scan_operator::column_scan_operator(
    const hype::SchedulingDecision& sched_dec, TablePtr table_ptr,
    const std::string& column_name)
    : UnaryOperator<ColumnPtr, ColumnPtr>(
          sched_dec, column_processing::cpu::TypedOperatorPtr()),
      table_name_(),
      column_name_(column_name),
      table_ptr_(table_ptr) {
  if (table_ptr_ == NULL) {
    COGADB_FATAL_ERROR("TablePtr is zero, cannot scan column of unknown Table!",
                       "");
  }

  table_name_ = table_ptr_->getName();
}

bool column_scan_operator::execute() {
  if (!quiet && verbose && debug) std::cout << "Execute Scan" << std::endl;
  // const TablePtr sort(TablePtr table, const std::string& column_name,
  // SortOrder order=ASCENDING, MaterializationStatus mat_stat=MATERIALIZE,
  // ComputeDevice comp_dev=CPU);

  //				TablePtr ptr=getTablebyName(table_name_);
  //				if(!ptr) return false;
  //				this->result_=ptr->getColumnbyName(column_name_);
  // std::cout << "SCAN: Table " <<  table_ptr_->getName() << "(" << (void*)
  // table_ptr_.get() << ") on Column " << column_name_ << std::endl;
  this->result_ = table_ptr_->getColumnbyName(column_name_);
  // std::cout << "Result: " << (void*) this->result_.get() << std::endl;
  if (this->result_ == NULL) {
    COGADB_FATAL_ERROR(std::string("Could not find Column ") + column_name_ +
                           std::string(" in table ") + table_ptr_->getName(),
                       "");
  }
  if (this->result_) {
    result_size_ = result_->size();
    return true;
  } else {
    return false;
  }
}

void column_scan_operator::releaseInputData() {
  // do nothing
}

column_scan_operator::~column_scan_operator() {}

}  // end namespace physical_operator

namespace logical_operator {

Logical_Column_Scan::Logical_Column_Scan(const std::string& table_name,
                                         const std::string& column_name)
    : TypedNode_Impl<
          ColumnPtr,
          physical_operator::map_init_function_column_scan_operator>(),
      table_name_(table_name),
      column_name_(column_name),
      table_ptr_(getTablebyName(table_name_)) {
  assert(table_ptr_ != NULL);
}

Logical_Column_Scan::Logical_Column_Scan(TablePtr table,
                                         const std::string& column_name)
    : table_name_(), column_name_(column_name), table_ptr_(table) {
  assert(table_ptr_ != NULL);
  table_name_ = table_ptr_->getName();
}

unsigned int Logical_Column_Scan::getOutputResultSize() const {
  return table_ptr_->getColumnbyName(column_name_)->size();
}

double Logical_Column_Scan::getCalculatedSelectivity() const { return 1.0; }

std::string Logical_Column_Scan::getOperationName() const {
  return "CPU_COLUMN_SCAN";
}

const std::string& Logical_Column_Scan::getTableName() const {
  return table_name_;
}

const TablePtr Logical_Column_Scan::getTablePtr() { return table_ptr_; }

const std::string& Logical_Column_Scan::getColumnName() const {
  return column_name_;
}

std::string Logical_Column_Scan::toString(bool verbose) const {
  std::string result = "CPU_COLUMN_SCAN";
  if (verbose) {
    result += " on column '";
    result += this->column_name_;
    result += "' from table '";
    result += this->table_name_;
    result += "'";
  }
  return result;
}

//	//redefining virtual function -> workaround so that scan operator can be
// processed uniformely with other operators
//	column_processing::cpu::TypedOperatorPtr
// Logical_Column_Scan::getOptimalOperator(column_processing::cpu::TypedOperatorPtr
// left_child, column_processing::cpu::TypedOperatorPtr right_child,
// hype::DeviceTypeConstraint dev_constr){
//		hype::Tuple t;
//
//		t.push_back(table_ptr_->getNumberofRows());
//
//		return this->operator_mapper_.getPhysicalOperator(*this, t,
// left_child, right_child, dev_constr); //this->getOperationName(), t,
// left_child, right_child);
//	}

const hype::Tuple Logical_Column_Scan::getFeatureVector() const {
  hype::Tuple t;
  t.push_back(table_ptr_->getNumberofRows());
  return t;
}

hype::query_optimization::QEP_Node* Logical_Column_Scan::toQEP_Node() {
  hype::Tuple t;

  t.push_back(table_ptr_->getNumberofRows());

  hype::OperatorSpecification op_spec(
      this->getOperationName(), t,
      // parameters are the same, because in the query processing engine, we
      // model copy oeprations explicitely, so the copy cost have to be zero
      hype::PD_Memory_0,   // input data is in CPU RAM
      hype::PD_Memory_0);  // output data has to be stored in CPU RAM

  hype::query_optimization::QEP_Node* node =
      new hype::query_optimization::QEP_Node(op_spec, this->dev_constr_);
  return node;  // hype::query_optimization::QEPPtr(new
                // hype::query_optimization::QEP(node));
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
