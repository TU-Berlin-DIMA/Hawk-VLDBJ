
#include <query_processing/cross_join_operator.hpp>
#include <util/hardware_detector.hpp>

#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Selection_operator=physical_operator::map_init_function_Selection_operator;

namespace physical_operator {

TypedOperatorPtr create_CPU_CrossJoin_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  if (!quiet && verbose && debug)
    std::cout << "create CPU_CrossJoin_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child != NULL);  // binary operator
  return TypedOperatorPtr(
      new CPU_CrossJoin_Operator(sched_dec, left_child, right_child));
}

Physical_Operator_Map_Ptr map_init_function_crossjoin_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for JOIN operator!" << std::endl;
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_NestedLoopJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");
// stemod::Scheduler::instance().addAlgorithm("JOIN","CPU_SortMergeJoin_Algorithm",stemod::CPU,"Least
// Squares 2D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_HashJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification join_alg_spec_cpu_nlj(
      "CPU_CrossJoin_Algorithm", "CROSS_JOIN", hype::KNN_Regression,
      hype::Periodic);
#else
  hype::AlgorithmSpecification join_alg_spec_cpu_nlj(
      "CPU_CrossJoin_Algorithm", "CROSS_JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(join_alg_spec_cpu_nlj,
                                               dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
      // hype::Scheduler::instance().addAlgorithm(group_by_alg_spec_gpu,dev_specs[i]);
    }
  }
  // stemod::Scheduler::instance().addAlgorithm("SELECTION","GPU_Selection_Algorithm",stemod::GPU,"Least
  // Squares 1D","Periodic Recomputation");
  map["CPU_CrossJoin_Algorithm"] = create_CPU_CrossJoin_Operator;

  // map["GPU_Selection_Algorithm"]=create_GPU_Selection_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

namespace logical_operator {

Logical_CrossJoin::Logical_CrossJoin()
    : TypedNode_Impl<TablePtr,
                     physical_operator::map_init_function_crossjoin_operator>(
          false, hype::ANY_DEVICE) {}

unsigned int Logical_CrossJoin::getOutputResultSize() const {
  return left_->getOutputResultSize() * right_->getOutputResultSize();
}

double Logical_CrossJoin::getCalculatedSelectivity() const { return 1; }

std::string Logical_CrossJoin::getOperationName() const { return "CROSS_JOIN"; }

void Logical_CrossJoin::produce_impl(CodeGeneratorPtr code_gen,
                                     QueryContextPtr context) {
  /* The build of the hash table is a pipeline breaker. Thus, we
   * generate a new code generator and a new query context */
  ProjectionParam param;
  CodeGeneratorPtr code_gen_build_phase =
      createCodeGenerator(code_gen->getCodeGeneratorType(), param);
  QueryContextPtr context_build_phase =
      createQueryContext(BUILD_HASH_TABLE_PIPELINE);

  context_build_phase->fetchInformationFromParentContext(context);
  //                /* pass down all referenced columns up to now to new query
  //                context */
  //                std::vector<std::string> accessed_columns =
  //                context->getAccessedColumns();
  //                for (size_t i = 0; i < accessed_columns.size(); ++i) {
  //                    context_build_phase->addAccessedColumn(accessed_columns[i]);
  //                }
  //                /* pass down projection list to sub pipeline */
  //                std::vector<std::string> projected_columns =
  //                context->getProjectionList();
  //                for (size_t i = 0; i < projected_columns.size(); ++i) {
  //                    context_build_phase->addColumnToProjectionList(projected_columns[i]);
  //                }
  /* create the build pipeline */
  left_->produce(code_gen_build_phase, context_build_phase);

  PipelinePtr build_pipeline = code_gen_build_phase->compile();
  if (!build_pipeline) {
    COGADB_FATAL_ERROR("Compiling code for hash table build failed!", "");
  }
  TablePtr result = execute(build_pipeline, context_build_phase);
  if (!result) {
    COGADB_FATAL_ERROR(
        "Execution of compiled code for hash table build failed!", "");
  }
  //                context->updateStatistics(build_pipeline);
  context->updateStatistics(context_build_phase);
  storeResultTableAttributes(code_gen, context, result);
  //                /* we later require the attribute references to the computed
  //                result,
  //                 so we store it in the query context */
  //                std::vector<ColumnProperties> col_props =
  //                result->getPropertiesOfColumns();
  //                for (size_t i = 0; i < col_props.size(); ++i) {
  //                    AttributeReferencePtr attr = createInputAttribute(
  //                            result,
  //                            col_props[i].name);
  //                    context->addReferencedAttributeFromOtherPipeline(attr);
  //                    code_gen->addToScannedAttributes(*attr);
  //                }
  //                /* add attribute references of result table to probe
  //                pipelines
  //                 * output schema, if we find them in the list of projected
  //                 columns */
  //                for (size_t i = 0; i < projected_columns.size(); ++i) {
  //                    if (result->hasColumn(projected_columns[i])) {
  //                        AttributeReferencePtr attr = createInputAttribute(
  //                                result,
  //                                projected_columns[i]);
  //                        code_gen->addAttributeProjection(*attr);
  //                    }
  //                }
  right_->produce(code_gen, context);
}

void Logical_CrossJoin::consume_impl(CodeGeneratorPtr code_gen,
                                     QueryContextPtr context) {
  if (context->getPipelineType() == BUILD_HASH_TABLE_PIPELINE) {
    /* we must not call consume of the parent operator here, we do it in the
     * other branch, because
     * we need to do it once per operator. */

  } else {
    std::vector<AttributeReferencePtr> attrs =
        context->getReferencedAttributeFromOtherPipelines();
    assert(attrs.size() > 0);
    if (!code_gen->consumeCrossJoin(*attrs.front())) {
      COGADB_FATAL_ERROR("Failed to generate code for cross join!", "");
    }

    /* we can pipeline the probe, so consume the parent operator */
    if (this->parent_) {
      this->parent_->consume(code_gen, context);
    }
  }
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
