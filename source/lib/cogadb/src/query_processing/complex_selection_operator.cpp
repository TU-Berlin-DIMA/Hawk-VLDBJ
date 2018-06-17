
#include <query_processing/complex_selection_operator.hpp>
#include <query_processing/query_processor.hpp>
#include <util/hardware_detector.hpp>

#include <core/data_dictionary.hpp>
#include <statistics/column_statistics.hpp>
#include <statistics/selectivity_estimator.hpp>

#include "query_compilation/code_generator.hpp"
#include "query_compilation/predicate_expression.hpp"
#include "query_compilation/query_context.hpp"

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_ComplexSelection_operator=physical_operator::map_init_function_ComplexSelection_operator;

namespace physical_operator {

bool CPU_ComplexSelection_Operator::execute() {
#ifdef ENABLE_TWO_PHASE_PHYSICAL_OPTIMIZATION
  // if there is one disjunction, we need the full query plan
  if (knf_expr_.disjunctions.size() > 1) {
    this->result_ = query_processing::two_phase_physical_optimization_selection(
        this->getInputData(), knf_expr_, dev_constr_, mat_stat_,
        CoGaDB::RuntimeConfiguration::instance()
            .getParallelizationModeForTwoPhasePhysicalOptimizationQueryPlans(),
        this->out);
    // if we have just one disjunction, it is possible that we split up the
    // query in selections for single disjunctions, which we execute serially
    // the other case is that the user specified a condition that can be
    // represented by a single disjunction
    // in both cases, it is correct to create a column based plan for the
    // disjunction and construct a Lookup Table from the result
  } else if (knf_expr_.disjunctions.size() == 1) {
    hype::ProcessingDeviceID id =
        sched_dec_.getDeviceSpecification().getProcessingDeviceID();
    ProcessorSpecification proc_spec(id);
    // SelectionParam param(proc_spec, ValueConstantPredicate,
    // knf_expr_.disjunctions[0], comp);

    this->result_ =
        BaseTable::selection(this->getInputData(), knf_expr_.disjunctions[0],
                             proc_spec, dev_constr_);
  } else {
    COGADB_FATAL_ERROR("Complex Selection has empty conditions!", "");
  }
#else
  this->result_ =
      BaseTable::selection(this->getInputData(), knf_expr_, mat_stat_, SERIAL);
#endif
  if (this->result_) {
    setResultSize(((TablePtr) this->result_)->getNumberofRows());
    return true;
  } else {
    return false;
  }
}

TypedOperatorPtr create_CPU_ComplexSelection_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_ComplexSelection& log_selection_ref =
      static_cast<logical_operator::Logical_ComplexSelection&>(logical_node);
  // std::cout << "create CPU_ComplexSelection_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new CPU_ComplexSelection_Operator(
      sched_dec, left_child, log_selection_ref.getKNF_Selection_Expression(),
      log_selection_ref.getDeviceConstraint(),
      log_selection_ref.getMaterializationStatus()));
}

Physical_Operator_Map_Ptr map_init_function_complex_selection_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for SELECTION operator!"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SELECTION","CPU_ComplexSelection_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("SELECTION","GPU_ComplexSelection_Algorithm",hype::GPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_ComplexSelection_Algorithm", "COMPLEX_SELECTION",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification selection_alg_spec_gpu(
      "GPU_ComplexSelection_Algorithm", "COMPLEX_SELECTION",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_ComplexSelection_Algorithm", "COMPLEX_SELECTION",
      hype::Least_Squares_1D, hype::Periodic);
#endif
  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu,
                                               dev_specs[i]);
      // hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu_parallel,dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_gpu,
                                               dev_specs[i]);
    }
  }

  map["CPU_ComplexSelection_Algorithm"] = create_CPU_ComplexSelection_Operator;
  map["GPU_ComplexSelection_Algorithm"] = create_CPU_ComplexSelection_Operator;
  // map["CPU_ParallelComplexSelection_Algorithm"]=create_CPU_ParallelComplexSelection_Operator;
  // map["GPU_ComplexSelection_Algorithm"]=create_GPU_ComplexSelection_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

namespace logical_operator {

bool Logical_ComplexSelection::isInputDataCachedInGPU() {
  KNF_Selection_Expression knf = this->getKNF_Selection_Expression();
  bool is_everything_cached = true;

  for (unsigned int i = 0; i < knf.disjunctions.size(); ++i) {
    for (unsigned int j = 0; j < knf.disjunctions[i].size(); ++j) {
      AttributeReferencePtr attr = getAttributeFromColumnIdentifier(
          knf.disjunctions[i][j].getColumn1Name());
      if (!attr) {
        COGADB_WARNING("Could not find Column '"
                           << knf.disjunctions[i][j].getColumn1Name()
                           << "' in data dictionary! I will assume the columns "
                              "is not cached!",
                       "");
        return false;
      }
      ColumnPtr col = attr->getColumn();
      assert(col != NULL);
      hype::ProcessingDeviceMemoryID mem_id = getMemoryIDForDeviceID(0);
      is_everything_cached =
          is_everything_cached &&
          DataCacheManager::instance().getDataCache(mem_id).isCached(col);
    }
  }
  if (is_everything_cached) {
    if (this->getLeft()->getOperationName() == "SCAN") {
      return true;
    } else {
      return hype::queryprocessing::Node::isInputDataCachedInGPU();
    }
  } else {
    return false;
  }
}

const std::list<std::string>
Logical_ComplexSelection::getNamesOfReferencedColumns() const {
  return knf_expr_.getReferencedColumnNames();
}

}  // end namespace logical_operator

double estimateSelectivity(const Predicate& p) {
  if (p.getPredicateType() == ValueValuePredicate) {
    return 0.1;
  }

  double est_sel = 0.1;

  std::list<std::pair<ColumnPtr, TablePtr> > columns =
      DataDictionary::instance().getColumnsforColumnName(p.getColumn1Name());
  assert(columns.size() <= 1);
  if (columns.empty()) return 0.1;

  if (columns.front().first) {
    if (columns.front().first->getColumnStatistics().sel_est_) {
      est_sel = columns.front()
                    .first->getColumnStatistics()
                    .sel_est_->getEstimatedSelectivity(p);
    }
  }

  return est_sel;
}

double estimateSelectivity(const Disjunction& disjunction) { return 0.1; }

double estimateSelectivity(const KNF_Selection_Expression& knf) {
  // detect range predicates
  return 0.1;
}

namespace logical_operator {

double Logical_ComplexSelection::getCalculatedSelectivity() const {
  return 0.1;
}

void Logical_ComplexSelection::produce_impl(CodeGeneratorPtr code_gen,
                                            QueryContextPtr context) {
  std::list<std::string> ref_cols = this->getNamesOfReferencedColumns();
  std::list<std::string>::const_iterator cit;
  for (cit = ref_cols.begin(); cit != ref_cols.end(); ++cit) {
    context->addAccessedColumn(*cit);
  }

  left_->produce(code_gen, context);
}

void Logical_ComplexSelection::consume_impl(CodeGeneratorPtr code_gen,
                                            QueryContextPtr context) {
  std::vector<PredicateExpressionPtr> conjunctions;
  for (unsigned int i = 0; i < knf_expr_.disjunctions.size(); ++i) {
    std::vector<PredicateExpressionPtr> disjunctions;

    for (unsigned int j = 0; j < knf_expr_.disjunctions[i].size(); ++j) {
      if (knf_expr_.disjunctions[i][j].getPredicateType() ==
          ValueConstantPredicate) {
        //                            std::list<std::pair<ColumnPtr,TablePtr> >
        //                            columns
        //                                    =
        //                                    DataDictionary::instance().getColumnsforColumnName(
        //                                    knf_expr_.disjunctions[i][j].getColumn1Name());
        //                            assert(columns.size()==1);
        //
        //                            AttributeReferencePtr attr =
        //                            createInputAttribute(
        //                                    columns.front().second,
        //                                    knf_expr_.disjunctions[i][j].getColumn1Name());

        AttributeReferencePtr attr = getAttributeFromColumnIdentifier(
            knf_expr_.disjunctions[i][j].getColumn1Name());
        //                                    columns.front().second,
        //                                    knf_expr_.disjunctions[i][j].getColumn1Name());
        assert(attr != NULL);
        PredicateExpressionPtr pred_expr =
            createColumnConstantComparisonPredicateExpression(
                attr, knf_expr_.disjunctions[i][j].getConstant(),
                knf_expr_.disjunctions[i][j].getValueComparator());
        disjunctions.push_back(pred_expr);
      } else if (knf_expr_.disjunctions[i][j].getPredicateType() ==
                 ValueRegularExpressionPredicate) {
        COGADB_FATAL_ERROR("Currently Unsupported", "");
      } else if (knf_expr_.disjunctions[i][j].getPredicateType() ==
                 ValueValuePredicate) {
        COGADB_FATAL_ERROR("Currently Unsupported", "");
      } else {
        COGADB_FATAL_ERROR("Unkown Predicate Type!", "");
      }
    }

    conjunctions.push_back(createPredicateExpression(disjunctions, LOGICAL_OR));
  }

  code_gen->consumeSelection(
      createPredicateExpression(conjunctions, LOGICAL_AND));

  if (parent_) {
    parent_->consume(code_gen, context);
  }
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
