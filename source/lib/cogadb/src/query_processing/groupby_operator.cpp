
#include <query_processing/groupby_operator.hpp>
#include <util/hardware_detector.hpp>

#include <core/variable_manager.hpp>
#include <util/code_generation.hpp>
#include "query_compilation/code_generator.hpp"
#include "query_compilation/query_context.hpp"
#include "statistics/column_statistics.hpp"

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Groupby_operator=physical_operator::map_init_function_groupby_operator;

namespace physical_operator {

bool isHashableAggregationFunction(const AggregationFunction& agg_func) {
  if (agg_func == COUNT || agg_func == SUM || agg_func == MIN ||
      agg_func == MAX || agg_func == AVERAGE || agg_func == AGG_GENOTYPE ||
      agg_func == AGG_CONCAT_BASES || agg_func == AGG_IS_HOMOPOLYMER ||
      agg_func == AGG_GENOTYPE_STATISTICS) {
    return true;
  }
  return false;
}

AggregationAlgorithm getAggregationAlgorithm(
    const std::list<ColumnAggregation>& aggregation_functions) {
  std::list<ColumnAggregation>::const_iterator cit;

  std::map<AggregationAlgorithm, size_t> counter_map;

  for (cit = aggregation_functions.begin(); cit != aggregation_functions.end();
       ++cit) {
    if (isHashableAggregationFunction(cit->second.first)) {
      counter_map[HASH_BASED_AGGREGATION]++;
    } else {
      counter_map[SORT_BASED_AGGREGATION]++;
    }
  }
  // if we either have an aggregation function that requires sorted data,
  // or we have at least four aggregation functions, we
  // choose a sorting based strategies
  // In the first case, we have no choice
  // In the second case, we expect that the sorting overhead
  // is amortized by the faster aggregation
  if (counter_map[SORT_BASED_AGGREGATION] > 0
      //                        || counter_map[HASH_BASED_AGGREGATION] >= 4
      ) {
    return SORT_BASED_AGGREGATION;
  } else {
    return HASH_BASED_AGGREGATION;
  }
}

bool CPU_Groupby_Operator::execute() {
  hype::ProcessingDeviceID id =
      sched_dec_.getDeviceSpecification().getProcessingDeviceID();
  ProcessorSpecification proc_spec(id);

  AggregationFunctions aggregation_functions;
  std::list<ColumnAggregation>::const_iterator cit;
  // we cannot combine hash-based and sort-based aggregation
  //(at least without introducing other overhead)
  // thus, we need to decide on one suitable algorithm for this
  // group by operation
  AggregationAlgorithm agg_alg =
      getAggregationAlgorithm(aggregation_functions_);
  // we only support sorting-based aggregation on co-processors
  if (hype::util::isCoprocessor(proc_spec.proc_id)) {
    agg_alg = SORT_BASED_AGGREGATION;
  }
  // Check special case where we group by a single column and
  // this column is sorted, then we can omit the sort step.
  // For now, we limit this optimization to persistent tables.
  // Support for intermediate tables would requrie a book
  // keeping which operators are oder preserving (e.g., selections).
  if (this->getInputData() && this->getInputData()->isMaterialized() &&
      grouping_columns_.size() == 1) {
    ColumnPtr col =
        this->getInputData()->getColumnbyName(grouping_columns_.front());
    if (col) {
      if (col->getColumnStatistics().statistics_up_to_date_) {
        if (col->getColumnStatistics().is_sorted_ascending_) {
          agg_alg = SORT_BASED_AGGREGATION;
        }
      }
    }
  }
  //                        agg_alg = SORT_BASED_AGGREGATION;
  for (cit = aggregation_functions_.begin();
       cit != aggregation_functions_.end(); ++cit) {
    aggregation_functions.push_back(std::make_pair(
        cit->first, AggregationParam(proc_spec, cit->second.first, agg_alg,
                                     cit->second.second)));
  }

  GroupbyParam param(proc_spec, grouping_columns_, aggregation_functions);

  this->result_ = BaseTable::groupby(this->getInputData(), param);

  if (this->result_) {
    setResultSize(((TablePtr) this->result_)->getNumberofRows());
    return true;
  } else
    this->has_aborted_ = true;
  return false;
}

TypedOperatorPtr create_CPU_Groupby_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Groupby& log_Groupby_ref =
      static_cast<logical_operator::Logical_Groupby&>(logical_node);
  if (!quiet && verbose)
    std::cout << "create CPU_Groupby_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new CPU_Groupby_Operator(
      sched_dec, left_child, log_Groupby_ref.getGroupingColumns(),
      log_Groupby_ref.getColumnAggregationFunctions(),
      log_Groupby_ref.getMaterializationStatus()));
}

TypedOperatorPtr create_GPU_Groupby_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_Groupby& log_Groupby_ref =
      static_cast<logical_operator::Logical_Groupby&>(logical_node);
  if (!quiet && verbose)
    std::cout << "create GPU_Groupby_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }
  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new GPU_Groupby_Operator(
      sched_dec, left_child, log_Groupby_ref.getGroupingColumns(),
      log_Groupby_ref.getColumnAggregationFunctions(),
      log_Groupby_ref.getMaterializationStatus()));
}

Physical_Operator_Map_Ptr map_init_function_groupby_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for GROUPBY operator!" << std::endl;
// hype::Scheduler::instance().addAlgorithm("GROUPBY","CPU_Groupby_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("GROUPBY","GPU_Groupby_Algorithm",hype::GPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification group_by_alg_spec_cpu(
      "CPU_Groupby_Algorithm", "GROUPBY",
      hype::KNN_Regression,  // hype::Multilinear_Fitting_2D,
                             // //hype::Least_Squares_1D,
      hype::Periodic);

  hype::AlgorithmSpecification group_by_alg_spec_gpu(
      "GPU_Groupby_Algorithm", "GROUPBY",
      hype::KNN_Regression,  // hype::Multilinear_Fitting_2D,
                             // //hype::Least_Squares_1D,
      hype::Periodic);
#else
  hype::AlgorithmSpecification group_by_alg_spec_cpu(
      "CPU_Groupby_Algorithm", "GROUPBY", hype::Least_Squares_1D,
      hype::Periodic);

  hype::AlgorithmSpecification group_by_alg_spec_gpu(
      "GPU_Groupby_Algorithm", "GROUPBY", hype::Least_Squares_1D,
      hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(group_by_alg_spec_cpu,
                                               dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
      hype::Scheduler::instance().addAlgorithm(group_by_alg_spec_gpu,
                                               dev_specs[i]);
#endif
    }
  }

  // hype::Scheduler::instance().addAlgorithm("SELECTION","GPU_Groupby_Algorithm",hype::GPU,"Least
  // Squares 1D","Periodic Recomputation");
  map["CPU_Groupby_Algorithm"] = create_CPU_Groupby_Operator;
  map["GPU_Groupby_Algorithm"] =
      create_CPU_Groupby_Operator;  // create_GPU_Groupby_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

namespace logical_operator {

Logical_Groupby::Logical_Groupby(
    const std::list<std::string>& grouping_columns,
    const std::list<ColumnAggregation>& aggregation_functions,
    MaterializationStatus mat_stat, hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<TablePtr,
                     physical_operator::map_init_function_groupby_operator>(
          false, dev_constr),
      grouping_columns_(grouping_columns),
      aggregation_functions_(aggregation_functions),
      mat_stat_(mat_stat) {}

unsigned int Logical_Groupby::getOutputResultSize() const { return 10; }

double Logical_Groupby::getCalculatedSelectivity() const { return 0.3; }

std::string Logical_Groupby::getOperationName() const { return "GROUPBY"; }

std::string Logical_Groupby::toString(bool verbose) const {
  std::string result = "GROUPBY";
  if (verbose) {
    result += " (";
    std::list<std::string>::const_iterator cit;
    for (cit = grouping_columns_.begin(); cit != grouping_columns_.end();
         ++cit) {
      result += *cit;
      if (cit != --grouping_columns_.end()) result += ",";
    }
    result += ")";
    result += " USING (";
    std::list<ColumnAggregation>::const_iterator agg_func_cit;
    for (agg_func_cit = aggregation_functions_.begin();
         agg_func_cit != aggregation_functions_.end(); ++agg_func_cit) {
      result += CoGaDB::util::getName(agg_func_cit->second.first);
      result += "(";
      result += agg_func_cit->first;
      result += ")";
    }
    result += ")";
  }
  return result;
}

const std::list<std::string> Logical_Groupby::getNamesOfReferencedColumns()
    const {
  std::list<std::string> result(grouping_columns_.begin(),
                                grouping_columns_.end());
  std::list<ColumnAggregation>::const_iterator agg_func_cit;
  for (agg_func_cit = aggregation_functions_.begin();
       agg_func_cit != aggregation_functions_.end(); ++agg_func_cit) {
    result.push_back(agg_func_cit->first);
    //                        std::cout << "Aggregated Column: " <<
    //                        agg_func_cit->first << std::endl;
  }
  return result;
}

const std::list<std::string>& Logical_Groupby::getGroupingColumns() {
  return grouping_columns_;
}

const std::list<ColumnAggregation>&
Logical_Groupby::getColumnAggregationFunctions() {
  return aggregation_functions_;
}

const MaterializationStatus& Logical_Groupby::getMaterializationStatus() const {
  return mat_stat_;
}

const hype::Tuple Logical_Groupby::getFeatureVector() const {
  hype::Tuple t;
  if (this->left_) {  // if left child is valid (has to be by convention!), add
                      // input data size
    // if we already know the correct input data size, because the child node
    // was already executed
    // during query chopping, we use the real cardinality, other wise we call
    // the estimator
    if (this->left_->getPhysicalOperator()) {
      // for the learning algorithms, it is helpful to
      // artificially adjust the points in multidimensional space
      // so proper regression models can be build
      // we use the logarithm function to destribute the points more equally

      double input_size = this->left_->getPhysicalOperator()->getResultSize();
      //                        if(input_size>0){
      //                            input_size = log(input_size)*10;
      //                        }
      t.push_back(input_size);
      t.push_back(this->aggregation_functions_.size() * input_size);
      // t.push_back(this->grouping_columns_.size()*10);

    } else {
      return this->Node::getFeatureVector();
      // t.push_back(this->left_->getOutputResultSize());
    }
  } else {
    HYPE_FATAL_ERROR("Invalid Left Child!", std::cout);
  }

  return t;
}

void Logical_Groupby::produce_impl(CodeGeneratorPtr code_gen,
                                   QueryContextPtr context) {
  /* The build of the aggregation hash table is a pipeline breaker.
   * Thus, we generate a new code generator and a new query context. */
  ProjectionParam param;
  CodeGeneratorPtr code_gen_hash_aggregate;
  QueryContextPtr context_hash_aggregate;

  /* if this operator is the root, then we use the code generator
   and context passed to us */
  code_gen_hash_aggregate = code_gen;
  context_hash_aggregate = context;

  /* add the attributes accessed by this operator to the list in
   * the query context */
  std::list<std::string> referenced_colums = getNamesOfReferencedColumns();
  std::list<std::string>::const_iterator cit;
  for (cit = referenced_colums.begin(); cit != referenced_colums.end(); ++cit) {
    context_hash_aggregate->addAccessedColumn(*cit);
    context_hash_aggregate->addColumnToProjectionList(*cit);
  }
  left_->produce(code_gen_hash_aggregate, context_hash_aggregate);
}

void Logical_Groupby::consume_impl(CodeGeneratorPtr code_gen,
                                   QueryContextPtr context) {
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");
  /* convert data structures to data structures for the code
   * generator */
  ProcessorSpecification proc_spec(hype::PD0);
  GroupingAttributes grouping_columns;
  std::list<std::string>::const_iterator cit_groups;
  std::list<ColumnAggregation>::const_iterator cit_aggr;
  for (cit_groups = grouping_columns_.begin();
       cit_groups != grouping_columns_.end(); ++cit_groups) {
    AttributeReferencePtr attr =
        getAttributeReference(*cit_groups, code_gen, context);
    if (!attr) {
      code_gen->print();
      COGADB_FATAL_ERROR("Attribute " << *cit_groups << " not found!", "");
    }
    grouping_columns.push_back(*attr);
  }

  AggregateSpecifications agg_specs;
  for (cit_aggr = aggregation_functions_.begin();
       cit_aggr != aggregation_functions_.end(); ++cit_aggr) {
    //                    AttributeReferencePtr scan_attr =
    //                    code_gen->getScannedAttributeByName(cit_aggr->first);
    AttributeReferencePtr scan_attr =
        getAttributeReference(cit_aggr->first, code_gen, context);
    //                    assert(scan_attr != NULL);
    if (!scan_attr) {
      COGADB_FATAL_ERROR(
          "Could not retrieve scan attribute reference for column "
              << cit_aggr->first,
          "");
    }
    //                    AttributeType type = DOUBLE;
    //                    if(cit_aggr->second.first==COUNT)
    //                        type=INT;
    /* create computed attribute */
    //                    AttributeReferencePtr result_attr(new
    //                    AttributeReference(scan_attr->getUnversionedAttributeName(),
    //                            type,
    //                            cit_aggr->second.second));
    AggregateSpecificationPtr aggr_spec = createAggregateSpecification(
        *scan_attr, cit_aggr->second.first, cit_aggr->second.second);

    //                    AggregateSpecification aggr_spec(*scan_attr,
    //                            result_attr,
    //                            cit_aggr->second.first);

    agg_specs.push_back(aggr_spec);
  }

  GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);

  if (grouping_columns_.empty()) {
    /* aggregation without groupby */
    if (!code_gen->consumeAggregate(agg_specs)) {
      COGADB_FATAL_ERROR("Failed to consume Aggregate Operation!", "");
    }
  } else {
    /* aggregation with groupby */
    if (!code_gen->consumeHashGroupAggregate(groupby_param)) {
      COGADB_FATAL_ERROR("Failed to consume Hash Aggregate Operation!", "");
    }
  }
  if (debug_code_generator) code_gen->print();
  /* This operator is a pipeline breaker, so we do not call consume
   * of the parent operator! */
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CoGaDB
