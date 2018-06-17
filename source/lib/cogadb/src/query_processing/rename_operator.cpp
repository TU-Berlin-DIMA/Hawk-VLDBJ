
#include <query_processing/rename_operator.hpp>
#include <util/hardware_detector.hpp>

#include "config/global_definitions.hpp"

namespace CoGaDB {

namespace query_processing {
// Map_Init_Function
// init_function_rename_operator=physical_operator::map_init_function_rename_operator;

namespace physical_operator {

TypedOperatorPtr create_rename_operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr) {
  logical_operator::Logical_Rename& log_sort_ref =
      static_cast<logical_operator::Logical_Rename&>(logical_node);
  // std::cout << "create RENAME Operator!" << std::endl;
  /*
  if(!left_child) {
          std::cout << "Error!" << std::endl;
          exit(-1);
  }
  assert(right_child==NULL); //unary operator
   */
  return TypedOperatorPtr(
      new rename_operator(sched_dec, left_child, log_sort_ref.getRenameList()));
}

Physical_Operator_Map_Ptr map_init_function_rename_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function! (RENAME OPERATION)" << std::endl;
// hype::Scheduler::instance().addAlgorithm("RENAME","RENAME",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_RENAME", "RENAME", hype::KNN_Regression, hype::No_Recomputation);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_RENAME", "RENAME", hype::Least_Squares_1D, hype::No_Recomputation);
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

  map["CPU_RENAME"] = create_rename_operator;
  // map["GPU_Algorithm"]=create_GPU_SORT_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

rename_operator::rename_operator(const hype::SchedulingDecision& sched_dec,
                                 TypedOperatorPtr left_child,
                                 RenameList rename_list)
    : UnaryOperator<TablePtr, TablePtr>(sched_dec, left_child),
      rename_list_(rename_list) {}

bool rename_operator::execute() {
  if (!quiet && verbose && debug) std::cout << "Execute RENAME" << std::endl;
  // const TablePtr sort(TablePtr table, const std::string& column_name,
  // SortOrder order=ASCENDING, MaterializationStatus mat_stat=MATERIALIZE,
  // ComputeDevice comp_dev=CPU);
  this->result_ = this->getInputData();
  if (this->result_) {
    if (!this->result_->renameColumns(rename_list_)) return false;
    setResultSize(this->result_->getNumberofRows());
    return true;
  } else
    return false;
}

rename_operator::~rename_operator() {}

}  // end namespace physical_operator

namespace logical_operator {

Logical_Rename::Logical_Rename(const RenameList& rename_list)
    : TypedNode_Impl<TablePtr,
                     physical_operator::map_init_function_rename_operator>(),
      rename_list_(rename_list) {}

unsigned int Logical_Rename::getOutputResultSize() const {
  // TablePtr table_ptr = getTablebyName(table_name_);
  // assert(table_ptr != NULL);
  assert(this->left_ != NULL);
  return this->left_->getOutputResultSize();  // table_->getNumberofRows();
}

double Logical_Rename::getCalculatedSelectivity() const { return 1.0; }

std::string Logical_Rename::getOperationName() const { return "RENAME"; }

std::string Logical_Rename::toString(bool verbose) const {
  std::stringstream ss;
  ss << getOperationName() << " (";
  RenameList::const_iterator cit;

  for (cit = rename_list_.begin(); cit != rename_list_.end(); ++cit) {
    ss << cit->first << "->" << cit->second;
    if (cit != --rename_list_.end()) ss << ", ";
  }
  ss << ")";
  return ss.str();  //+table_->getName();
}

const RenameList& Logical_Rename::getRenameList() { return this->rename_list_; }

void Logical_Rename::produce_impl(CodeGeneratorPtr code_gen,
                                  QueryContextPtr context) {
  COGADB_FATAL_ERROR("Called function that is not yet implemented!", "");
  //                RenameList::const_iterator cit;
  //
  //
  //                for(cit=rename_list_.begin();cit!=rename_list_.end();++cit){
  ////                    ss << cit->first << "->" << cit->second;
  //                    context->addColumnToProjectionList(cit->first);
  //                    context->addAccessedColumn(cit->first);
  //                }
  //
  //
  //
  //                left_->produce(code_gen, context);
}

void Logical_Rename::consume_impl(CodeGeneratorPtr code_gen,
                                  QueryContextPtr context) {
  if (this->parent_) {
    this->parent_->consume(code_gen, context);
  }
}

//	//redefining virtual function -> workaround so that scan operator can be
// processed uniformely with other operators
//	TypedOperatorPtr Logical_Rename::getOptimalOperator(TypedOperatorPtr
// left_child, TypedOperatorPtr right_child, hype::DeviceTypeConstraint
// dev_constr){
//		hype::Tuple t;
//
//		t.push_back(getOutputResultSize());
//
//		return this->operator_mapper_.getPhysicalOperator(*this, t,
// left_child, right_child, dev_constr); //this->getOperationName(), t,
// left_child, right_child);
//	}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
