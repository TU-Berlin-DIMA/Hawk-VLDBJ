
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/column_computation_algebra_operator.hpp>
#include <util/code_generation.hpp>
#include <util/hardware_detector.hpp>

#include <core/data_dictionary.hpp>
#include <core/variable_manager.hpp>

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_sort_operator=physical_operator::map_init_function_column_algebra_operator;

namespace physical_operator {

TypedOperatorPtr create_CPU_ColumnAlgebra_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_ColumnAlgebraOperator& log_sort_ref =
      static_cast<logical_operator::Logical_ColumnAlgebraOperator&>(
          logical_node);
  // std::cout << "create CPU_ColumnAlgebra_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new CPU_ColumnAlgebraOperator(
      sched_dec, left_child, log_sort_ref.getColumn1Name(),
      log_sort_ref.getColumn2Name(), log_sort_ref.getResultColumnName(),
      log_sort_ref.getColumnAlgebraOperation()));
}

Physical_Operator_Map_Ptr map_init_function_column_algebra_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for ColumnAlgebra Operator!"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SORT","CPU_Sort_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("SORT","GPU_Sort_Algorithm",hype::GPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification sort_alg_spec_cpu(
      "CPU_ColumnAlgebra_Algorithm", "ColumnAlgebraOperator",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification sort_alg_spec_gpu(
      "GPU_ColumnAlgebra_Algorithm", "ColumnAlgebraOperator",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification sort_alg_spec_cpu(
      "CPU_ColumnAlgebra_Algorithm", "ColumnAlgebraOperator",
      hype::Least_Squares_1D, hype::Periodic);

  hype::AlgorithmSpecification sort_alg_spec_gpu(
      "GPU_ColumnAlgebra_Algorithm", "ColumnAlgebraOperator",
      hype::Least_Squares_1D, hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(sort_alg_spec_cpu, dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
      hype::Scheduler::instance().addAlgorithm(sort_alg_spec_gpu, dev_specs[i]);
#endif
    }
  }

  map["CPU_ColumnAlgebra_Algorithm"] = create_CPU_ColumnAlgebra_Operator;
  map["GPU_ColumnAlgebra_Algorithm"] = create_CPU_ColumnAlgebra_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

namespace logical_operator {

Logical_ColumnAlgebraOperator::Logical_ColumnAlgebraOperator(
    const std::string& column1_name, const std::string& column2_name,
    const std::string& result_col_name, ColumnAlgebraOperation operation,
    hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<
          TablePtr,
          physical_operator::map_init_function_column_algebra_operator>(
          false, dev_constr),
      column1_name_(column1_name),
      column2_name_(column2_name),
      result_column_name_(result_col_name),
      operation_(operation) {
  column1_name_ = convertToFullyQualifiedNameIfRequired(column1_name_);
  column2_name_ = convertToFullyQualifiedNameIfRequired(column2_name_);
}

unsigned int Logical_ColumnAlgebraOperator::getOutputResultSize() const {
  return 10;
}

double Logical_ColumnAlgebraOperator::getCalculatedSelectivity() const {
  return 0.1;
}

std::string Logical_ColumnAlgebraOperator::getOperationName() const {
  return "ColumnAlgebraOperator";
}

std::string Logical_ColumnAlgebraOperator::toString(bool verbose) const {
  std::string result = "ColumnAlgebraOperator";
  if (verbose) {
    result += " (";
    result += result_column_name_;
    result += "=";
    result += util::getName(operation_);
    result += "(";
    result += column1_name_;
    result += ",";
    result += column2_name_;
    result += ")";

    result += ")";
  }
  return result;
}

const std::string& Logical_ColumnAlgebraOperator::getColumn1Name() {
  return column1_name_;
}

const std::string& Logical_ColumnAlgebraOperator::getColumn2Name() {
  return column2_name_;
}

const std::string& Logical_ColumnAlgebraOperator::getResultColumnName() {
  return result_column_name_;
}

CoGaDB::ColumnAlgebraOperation
Logical_ColumnAlgebraOperator::getColumnAlgebraOperation() {
  return operation_;
}

const std::list<std::string>
Logical_ColumnAlgebraOperator::getNamesOfReferencedColumns() const {
  std::list<std::string> result;
  result.push_back(column1_name_);
  result.push_back(column2_name_);
  return result;
}

void Logical_ColumnAlgebraOperator::produce_impl(CodeGeneratorPtr code_gen,
                                                 QueryContextPtr context) {
  /* add the attributes accessed by this operator to the list in
   * the query context */
  std::list<std::string> referenced_colums = getNamesOfReferencedColumns();
  std::list<std::string>::const_iterator cit;
  for (cit = referenced_colums.begin(); cit != referenced_colums.end(); ++cit) {
    context->addAccessedColumn(*cit);
    context->addColumnToProjectionList(*cit);
  }

  left_->produce(code_gen, context);
}

void Logical_ColumnAlgebraOperator::consume_impl(CodeGeneratorPtr code_gen,
                                                 QueryContextPtr context) {
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");

  AttributeReferencePtr attr_1 =
      getAttributeReference(this->column1_name_, code_gen, context);
  AttributeReferencePtr attr_2 =
      getAttributeReference(this->column2_name_, code_gen, context);

  if (!attr_1) {
    code_gen->print();
    COGADB_FATAL_ERROR("[MAP]: Attribute '" << column1_name_ << "' not found!",
                       "");
  }
  if (!attr_2) {
    code_gen->print();
    COGADB_FATAL_ERROR("[MAP]: Attribute '" << column2_name_ << "' not found!",
                       "");
  }

  code_gen->addToScannedAttributes(*attr_1);
  code_gen->addToScannedAttributes(*attr_2);

  std::pair<bool, AttributeReference> ret =
      code_gen->consumeAlgebraComputation(*attr_1, *attr_2, operation_);

  if (!ret.first) {
    COGADB_FATAL_ERROR("", "");
  } else {
    std::string computed_column_name = "(";
    computed_column_name.append(column1_name_);
    computed_column_name.append(toCPPOperator(operation_));
    computed_column_name.append(column2_name_);
    computed_column_name.append(")");
    AttributeReferencePtr comp_attr(new AttributeReference(ret.second));
    assert(comp_attr != NULL);
    context->addComputedAttribute(computed_column_name, comp_attr);
    std::set<std::string> unresolved_projections =
        context->getUnresolvedProjectionAttributes();
    std::set<std::string>::const_iterator cit =
        unresolved_projections.find(computed_column_name);
    if (cit != unresolved_projections.end()) {
      code_gen->addAttributeProjection(ret.second);
      if (debug_code_generator) {
        std::cout << "[MAP]: resolved attribute '" << computed_column_name
                  << "' (" << CoGaDB::toString(*comp_attr) << ")" << std::endl;
      }
    }
    if (debug_code_generator) {
      std::cout << "Col name: " << computed_column_name << std::endl;
      std::cout << "Unresolved Projection Attributes: " << std::endl;
      for (cit = unresolved_projections.begin();
           cit != unresolved_projections.end(); ++cit) {
        std::cout << "\t" << *cit << std::endl;
      }
    }
  }
  if (parent_) {
    parent_->consume(code_gen, context);
  }
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
