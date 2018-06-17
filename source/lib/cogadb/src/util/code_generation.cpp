
#include <core/variable_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <util/code_generation.hpp>

namespace CoGaDB {

AttributeReferencePtr getAttributeReference(const std::string& column_name,
                                            CodeGeneratorPtr code_gen,
                                            QueryContextPtr context) {
  assert(code_gen != NULL);
  assert(context != NULL);
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");
  if (debug_code_generator) {
    std::cout << "Searching for column '" << column_name << "'" << std::endl;
    code_gen->print();
  }

  AttributeReferencePtr attr = code_gen->getScannedAttributeByName(column_name);
  if (attr) return attr;
  attr = context->getComputedAttribute(column_name);
  return attr;
}

const std::string toCPPOperator(const ColumnAlgebraOperation& op) {
  // enum ColumnAlgebraOperation{ADD,SUB,MUL,DIV};
  const char* const names[] = {"+", "-", "*", "/"};
  return std::string(names[op]);
}

StructField::StructField(const AttributeType& _field_type,
                         const std::string& _field_name,
                         const std::string& _field_init_val)
    : field_type(_field_type),
      field_name(_field_name),
      field_init_val(_field_init_val) {}

}  // end namespace CoGaDB
