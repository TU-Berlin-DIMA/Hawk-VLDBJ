#include <util/query_compilation.hpp>

namespace CoGaDB {
namespace query_processing {

void produceBulkProcessingOperator(
    CodeGeneratorPtr code_gen, QueryContextPtr context, NodePtr left_child,
    BulkOperatorFunction function,
    const std::list<std::string>& accessed_attributes) {
  /* This function assumes a pipeline breaking operator. Thus, we
   * generate a new code generator and a new query context */
  ProjectionParam param;
  CodeGeneratorPtr code_gen_build_phase =
      createCodeGenerator(code_gen->getCodeGeneratorType(), param);
  QueryContextPtr context_build_phase = createQueryContext(NORMAL_PIPELINE);
  /* add the attributes accessed by this operator to the list in
   * the query context */
  std::list<std::string>::const_iterator cit;
  for (cit = accessed_attributes.begin(); cit != accessed_attributes.end();
       ++cit) {
    context->addAccessedColumn(*cit);
    context_build_phase->addColumnToProjectionList(*cit);
  }
  context_build_phase->fetchInformationFromParentContext(context);
  /* create the build pipeline */
  left_child->produce(code_gen_build_phase, context_build_phase);
  TablePtr result;
  PipelinePtr build_pipeline = code_gen_build_phase->compile();
  if (!build_pipeline) {
    COGADB_FATAL_ERROR("Compiling code for pipeline running "
                           << "before sort failed!",
                       "");
  }
  result = execute(build_pipeline, context_build_phase);
  if (!result) {
    COGADB_FATAL_ERROR("Execution of compiled code for pipeline "
                           << "running before sort failed!!",
                       "");
  }
  //                context->updateStatistics(build_pipeline);
  context->updateStatistics(context_build_phase);
  Timestamp begin = getTimestamp();
  result = function(result);
  Timestamp end = getTimestamp();
  if (!result) {
    COGADB_FATAL_ERROR("Failed to sort table!", "");
  }
  double sort_execution_time = double(end - begin) / (1000 * 1000 * 1000);
  context->addExecutionTime(sort_execution_time);
  storeResultTableAttributes(code_gen, context, result);
  code_gen->createForLoop(result, 1);
}

}  // end namespace query_processing

}  // end namespace CoGaDB
