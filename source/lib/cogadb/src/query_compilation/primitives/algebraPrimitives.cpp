/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   algebraPrimitives.cpp
 * Author: dbronesk
 *
 * Created on February 8, 2016, 3:42 PM
 */
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/primitives/algebraPrimitives.hpp>

namespace CoGaDB {

const GeneratedCodePtr AttributeAttributeOp::getCode(
    CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());

  std::stringstream compute_expr;
  /* declare variable before assignment */
  compute_expr << getCTypeFunctionPostFix(computed_attr_.getAttributeType())
               << " " << getVarName(computed_attr_) << ";" << std::endl;
  /* perform computation, then assign to variable */
  compute_expr << getElementAccessExpression(computed_attr_) << " = "
               << getElementAccessExpression(left_operand_) << " "
               << toCPPOperator(alg_op_) << " "
               << getElementAccessExpression(right_operand_) << ";";

  code->upper_code_block_.push_back(compute_expr.str());
  return code;
}

const std::string AttributeAttributeOp::toString() const {
  std::stringstream str;
  str << "ALGEBRA(" << CoGaDB::toString(computed_attr_) << " = "
      << CoGaDB::toString(left_operand_) << " " << toCPPOperator(alg_op_) << " "
      << CoGaDB::toString(right_operand_) << ")";
  return str.str();
}

std::map<std::string, std::string> AttributeAttributeOp::getInputVariables(
    CodeGenerationTarget target) {
  auto result0 = getVariableNameAndTypeFromAttributeRef(left_operand_);
  auto result1 = getVariableNameAndTypeFromAttributeRef(right_operand_);

  return {result0, result1};
}

std::map<std::string, std::string> AttributeAttributeOp::getOutputVariables(
    CodeGenerationTarget target) {
  // TODO, In the current state we can't return here anything..
  return {};
}

const GeneratedCodePtr AttributeConstantOp::getCode(
    CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  code->declare_variables_code_block_
      << "double " << getElementAccessExpression(computed_attr_) << " = 0.0;"
      << std::endl;

  std::stringstream compute_expr;
  compute_expr << getElementAccessExpression(computed_attr_) << " = "
               << getElementAccessExpression(left_operand_) << " "
               << toCPPOperator(alg_op_) << " " << getConstant(right_operand_)
               << ";";

  code->upper_code_block_.push_back(compute_expr.str());
  return code;
}

const std::string AttributeConstantOp::toString() const {
  std::stringstream str;
  str << "ALGEBRA(" << CoGaDB::toString(computed_attr_) << " = "
      << CoGaDB::toString(left_operand_) << " " << toCPPOperator(alg_op_) << " "
      << getConstant(right_operand_) << ")";
  return str.str();
}

std::map<std::string, std::string> AttributeConstantOp::getInputVariables(
    CodeGenerationTarget target) {
  return {getVariableNameAndTypeFromAttributeRef(left_operand_)};
}

std::map<std::string, std::string> AttributeConstantOp::getOutputVariables(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

const GeneratedCodePtr ConstantAttributeOp::getCode(
    CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  code->declare_variables_code_block_
      << "double " << getElementAccessExpression(computed_attr_) << " = 0.0;"
      << std::endl;

  std::stringstream compute_expr;
  compute_expr << getElementAccessExpression(computed_attr_) << " = "
               << getConstant(left_operand_) << " " << toCPPOperator(alg_op_)
               << " " << getElementAccessExpression(right_operand_) << ";";

  code->upper_code_block_.push_back(compute_expr.str());
  return code;
}

const std::string ConstantAttributeOp::toString() const {
  std::stringstream str;
  str << "ALGEBRA(" << CoGaDB::toString(computed_attr_) << " = "
      << getConstant(left_operand_) << " " << toCPPOperator(alg_op_) << " "
      << CoGaDB::toString(right_operand_) << ")";
  return str.str();
}

std::map<std::string, std::string> ConstantAttributeOp::getInputVariables(
    CodeGenerationTarget target) {
  return {getVariableNameAndTypeFromAttributeRef(right_operand_)};
}

std::map<std::string, std::string> ConstantAttributeOp::getOutputVariables(
    CodeGenerationTarget target) {
  return {};
}
}
