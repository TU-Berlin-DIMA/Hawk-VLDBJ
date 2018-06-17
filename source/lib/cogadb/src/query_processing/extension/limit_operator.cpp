

#include <core/attribute_reference.hpp>
#include <core/table.hpp>
#include <core/user_defined_function.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/extension/limit_operator.hpp>

namespace CoGaDB {
namespace query_processing {

namespace logical_operator {

Logical_Limit::Logical_Limit(const uint64_t& max_number_of_rows)
    : Logical_BulkOperator(), max_number_of_rows_(max_number_of_rows) {}

unsigned int Logical_Limit::getOutputResultSize() const {
  return max_number_of_rows_;
}

double Logical_Limit::getCalculatedSelectivity() const { return 1; }

std::string Logical_Limit::getOperationName() const { return "LIMIT"; }

std::string Logical_Limit::toString(bool verbose) const {
  std::string result = this->getOperationName();
  if (verbose) {
    result += " (";
    result += max_number_of_rows_;
    result += ")";
  }
  return result;
}

const TablePtr Logical_Limit::executeBulkOperator(TablePtr table) const {
  assert(table != NULL);

  ProcessorSpecification proc_spec(hype::PD0);
  std::vector<boost::any> param;
  param.push_back(boost::any(size_t(max_number_of_rows_)));
  std::cout << "LIMIT Operator: " << max_number_of_rows_ << std::endl;
  return limit(table, "LIMIT", param, proc_spec);
}

const std::list<std::string> Logical_Limit::getNamesOfReferencedColumns()
    const {
  std::list<std::string> cols;

  return cols;
}

}  // end namespace logical_operator

}  // end namespace query_processing
}  // end namespace CogaDB
