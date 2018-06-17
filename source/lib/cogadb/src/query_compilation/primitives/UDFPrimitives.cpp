/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   UDFPrimitives.cpp
 * Author: dbronesk
 *
 * Created on February 8, 2016, 3:37 PM
 */

#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/primitives/UDFPrimitives.hpp>

namespace CoGaDB {

const GeneratedCodePtr MapUDF::getCode(CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  std::stringstream upper;
  std::stringstream lower;

  if (!resultUDF_.return_code) {
    COGADB_FATAL_ERROR(
        "Failed to generate code for Map UDF type: " /* << int(param->map_udf->getMap_UDF_Type())*/,
        "");
  }

  code->upper_code_block_.push_back(resultUDF_.declared_variables);
  code->upper_code_block_.push_back(resultUDF_.generated_code);
  return code;
}

const std::string MapUDF::toString() const { return "MAP_UDF"; }

std::map<std::string, std::string> MapUDF::getInputVariables(
    CodeGenerationTarget target) {
  return getVariableNameAndTypeFromInputAttributeRefVector(
      resultUDF_.scanned_attributes);
}

std::map<std::string, std::string> MapUDF::getOutputVariables(
    CodeGenerationTarget target) {
  return {};
  //  return getVariableNameAndTypeFromAttributeRefVector(
  //      resultUDF_.computed_attributes);
}

}  // namespace CoGaDB
