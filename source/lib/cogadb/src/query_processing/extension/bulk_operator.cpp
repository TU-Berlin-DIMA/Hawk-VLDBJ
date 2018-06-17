

#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/store_table_operator.hpp>

namespace CoGaDB {
namespace query_processing {

Physical_Operator_Map_Ptr map_init_function_dummy() {
  return Physical_Operator_Map_Ptr();
}

namespace logical_operator {
Logical_BulkOperator::Logical_BulkOperator()
    : TypedNode_Impl<TablePtr, map_init_function_dummy>(false,
                                                        hype::ANY_DEVICE) {}

void Logical_BulkOperator::produce_impl(CodeGeneratorPtr code_gen,
                                        QueryContextPtr context) {
  /* The sort operator is a pipeline breaker. Thus, we
   * generate a new code generator and a new query context */
  ProjectionParam param;
  CodeGeneratorPtr new_code_gen =
      createCodeGenerator(code_gen->getCodeGeneratorType(), param);
  QueryContextPtr new_context = createQueryContext(NORMAL_PIPELINE);
  /* add the attributes accessed by this operator to the list in
   * the query context */
  std::list<std::string> referenced_columns =
      this->getNamesOfReferencedColumns();
  std::list<std::string>::const_iterator cit;
  for (cit = referenced_columns.begin(); cit != referenced_columns.end();
       ++cit) {
    context->addAccessedColumn(*cit);
    new_context->addColumnToProjectionList(*cit);
  }

  new_context->fetchInformationFromParentContext(context);

  TablePtr result;
  if (left_) {
    left_->produce(new_code_gen, new_context);
    PipelinePtr build_pipeline = new_code_gen->compile();
    if (!build_pipeline) {
      COGADB_FATAL_ERROR("Compiling code for pipeline running "
                             << "before '" << this->getOperationName()
                             << "' failed!",
                         "");
    }
    result = execute(build_pipeline, new_context);
    if (!result) {
      COGADB_FATAL_ERROR("Execution of compiled code for pipeline "
                             << "running before sort failed!!",
                         "");
    }
    context->updateStatistics(new_context);
  }
  //                storeResultTableAttributes(code_gen, context, result);
  Timestamp begin = getTimestamp();
  result = this->executeBulkOperator(result);
  Timestamp end = getTimestamp();
  if (!result) {
    COGADB_FATAL_ERROR("Failed to execute bulk operator!", "");
  }
  double sort_execution_time = double(end - begin) / (1000 * 1000 * 1000);
  context->addExecutionTime(sort_execution_time);

  storeResultTableAttributes(code_gen, context, result);

  uint32_t version = 1;
  /* only create a scan set if we have follow up operators to not mess up the
   * result
   * and avoid problems with computed attributes
   */
  if (parent_)
    retrieveScannedAndProjectedAttributesFromScannedTable(code_gen, context,
                                                          result, version);

  // create for loop for code_gen
  if (!code_gen->createForLoop(result, version)) {
    COGADB_FATAL_ERROR("Creating of for loop failed!", "");
  }
  if (parent_) {
    parent_->consume(code_gen, context);
  }
}

void Logical_BulkOperator::consume_impl(CodeGeneratorPtr code_gen,
                                        QueryContextPtr context) {
  /* this is a bulk operator, so no consume here! */
}

}  // end namespace logical_operator

}  // end namespace query_processing
}  // end namespace CogaDB
