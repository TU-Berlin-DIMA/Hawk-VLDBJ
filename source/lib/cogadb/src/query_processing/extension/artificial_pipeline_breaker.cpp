
#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/artificial_pipeline_breaker.hpp>

namespace CoGaDB {
namespace query_processing {

namespace logical_operator {

Logical_Artificial_Pipeline_Breaker::Logical_Artificial_Pipeline_Breaker()
    : Logical_BulkOperator() {}

unsigned int Logical_Artificial_Pipeline_Breaker::getOutputResultSize() const {
  return 0;
}

double Logical_Artificial_Pipeline_Breaker::getCalculatedSelectivity() const {
  return 1;
}

std::string Logical_Artificial_Pipeline_Breaker::getOperationName() const {
  return "ARTIFICIAL_PIPELINE_BREAKER";
}

std::string Logical_Artificial_Pipeline_Breaker::toString(bool verbose) const {
  std::string result = this->getOperationName();
  return result;
}

const TablePtr Logical_Artificial_Pipeline_Breaker::executeBulkOperator(
    TablePtr table) const {
  return table;
}

const std::list<std::string>
Logical_Artificial_Pipeline_Breaker::getNamesOfReferencedColumns() const {
  std::list<std::string> cols;

  return cols;
}

}  // end namespace logical_operator

}  // end namespace query_processing
}  // end namespace CogaDB
