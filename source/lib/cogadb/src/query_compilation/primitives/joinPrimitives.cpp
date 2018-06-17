/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   joinPrimitives.cpp
 * Author: dbronesk
 *
 * Created on February 8, 2016, 9:25 AM
 */

#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/primitives/joinPrimitives.hpp>

namespace CoGaDB {

const GeneratedCodePtr CrossJoin::getCode(CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  std::stringstream upper;
  std::stringstream lower;

  upper << "for (size_t " << getTupleIDVarName(attr_) << " = 0; "
        << getTupleIDVarName(attr_) << "< "
        << attr_.getTable()->getNumberofRows() << "ULL; ++"
        << getTupleIDVarName(attr_) << ") {";

  lower << "}";

  code->upper_code_block_.push_back(upper.str());
  code->lower_code_block_.push_back(lower.str());
  return code;
}

const std::string CrossJoin::toString() const { return "CROSS_JOIN"; }

std::map<std::string, std::string> CrossJoin::getInputVariables(
    CodeGenerationTarget target) {
  return {};
}

std::map<std::string, std::string> CrossJoin::getOutputVariables(
    CodeGenerationTarget target) {
  return {{getTupleIDVarName(attr_), "size_t"}};
}

}  // namespace CoGaDB
