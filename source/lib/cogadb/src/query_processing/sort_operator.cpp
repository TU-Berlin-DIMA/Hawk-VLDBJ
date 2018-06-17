
#include <query_compilation/code_generator.hpp>
#include <query_compilation/predicate_expression.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/sort_operator.hpp>
#include <util/hardware_detector.hpp>
#include <util/query_compilation.hpp>

namespace CoGaDB {

namespace query_processing {

namespace physical_operator {

TypedOperatorPtr create_CPU_SORT_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Sort& log_sort_ref =
      static_cast<logical_operator::Logical_Sort&>(logical_node);
  // std::cout << "create CPU_SORT_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(
      new CPU_Sort_Operator(sched_dec, left_child,
                            // log_sort_ref.getColumnNames(),
                            // log_sort_ref.getSortOrder(),
                            log_sort_ref.getSortAttributes(),
                            log_sort_ref.getMaterializationStatus()));
}

TypedOperatorPtr create_GPU_SORT_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Sort& log_sort_ref =
      static_cast<logical_operator::Logical_Sort&>(logical_node);
  // std::cout << "create GPU_SORT_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(
      new GPU_Sort_Operator(sched_dec, left_child,
                            // log_sort_ref.getColumnNames(),
                            // log_sort_ref.getSortOrder(),
                            log_sort_ref.getSortAttributes(),
                            log_sort_ref.getMaterializationStatus()));
}

Physical_Operator_Map_Ptr map_init_function_sort_operator() {
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for Sort Operator!" << std::endl;

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification sort_alg_spec_cpu(
      "CPU_Sort_Algorithm", "SORT", hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification sort_alg_spec_gpu(
      "GPU_Sort_Algorithm", "SORT", hype::KNN_Regression, hype::Periodic);
#else

  hype::AlgorithmSpecification sort_alg_spec_cpu("CPU_Sort_Algorithm", "SORT",
                                                 hype::Multilinear_Fitting_2D,
                                                 hype::Periodic);

  hype::AlgorithmSpecification sort_alg_spec_gpu(
      "GPU_Sort_Algorithm", "SORT",
      hype::Multilinear_Fitting_2D,  // Least_Squares_1D,
      hype::Periodic);
#endif

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(sort_alg_spec_cpu, dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
//                        hype::Scheduler::instance().addAlgorithm(sort_alg_spec_gpu,
//                        dev_specs[i]);
#endif
    }
  }

  map["CPU_Sort_Algorithm"] = create_CPU_SORT_Operator;
  map["GPU_Sort_Algorithm"] = create_GPU_SORT_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

namespace logical_operator {

Logical_Sort::Logical_Sort(const SortAttributeList& sort_attributes,
                           MaterializationStatus mat_stat,
                           hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<TablePtr,
                     physical_operator::map_init_function_sort_operator>(
          false, dev_constr),
      sort_attributes_(),
      mat_stat_(mat_stat) {
  SortAttributeList::const_iterator cit;
  for (cit = sort_attributes.begin(); cit != sort_attributes.end(); ++cit) {
    sort_attributes_.push_back(SortAttribute(
        convertToFullyQualifiedNameIfRequired(cit->first), cit->second));
  }
}

unsigned int Logical_Sort::getOutputResultSize() const {
  return this->left_->getOutputResultSize();
}

double Logical_Sort::getCalculatedSelectivity() const { return 1; }

std::string Logical_Sort::getOperationName() const { return "SORT"; }

std::string Logical_Sort::toString(bool verbose) const {
  std::string result = "SORT";
  if (verbose) {
    result += " BY (";

    std::string asc("ASCENDING");
    std::string desc("DESCENDING");

    SortAttributeList::const_iterator cit;
    for (cit = sort_attributes_.begin(); cit != sort_attributes_.end(); ++cit) {
      result += cit->first;
      result += " ORDER ";

      if (cit->second == ASCENDING)
        result += asc;
      else
        result += desc;

      if (cit != --sort_attributes_.end()) result += ",";
    }
    result += ")";
  }
  return result;
}

const std::list<SortAttribute>& Logical_Sort::getSortAttributes() {
  return sort_attributes_;
}

MaterializationStatus Logical_Sort::getMaterializationStatus() {
  return mat_stat_;
}

void Logical_Sort::produce_impl(CodeGeneratorPtr code_gen,
                                QueryContextPtr context) {
  /* The sort operator is a pipeline breaker. Thus, we
   * generate a new code generator and a new query context */
  std::list<std::string> accessed_attributes =
      this->getNamesOfReferencedColumns();
  BulkOperatorFunction function =
      boost::bind(BaseTable::sort, _1, sort_attributes_, mat_stat_, CPU);
  produceBulkProcessingOperator(code_gen, context, this->left_, function,
                                accessed_attributes);
}

void Logical_Sort::consume_impl(CodeGeneratorPtr code_gen,
                                QueryContextPtr context) {
  (void)code_gen;
  (void)context;
  /* this is a pipeline breaker, so we must not call consume of the parent
   * operator! */
  /* as we currently do not generate code for sort operations, we
   * do not do anything in this function*/
}

const hype::Tuple Logical_Sort::getFeatureVector() const {
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
    } else {
      t.push_back(this->left_->getOutputResultSize());
    }
    t.push_back(sort_attributes_.size());
  }
  return t;
}

const std::list<std::string> Logical_Sort::getNamesOfReferencedColumns() const {
  std::list<std::string> ref_cols;
  std::list<SortAttribute>::const_iterator cit;
  for (cit = this->sort_attributes_.begin();
       cit != this->sort_attributes_.end(); ++cit) {
    ref_cols.push_back(cit->first);
  }
  return ref_cols;
}

}  // end namespace logical_operator
}  // end namespace query_processing
}  // end namespace CoGaDB
