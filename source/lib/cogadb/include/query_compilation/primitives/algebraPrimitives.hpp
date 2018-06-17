/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   algebraPrimitives.hpp
 * Author: dbronesk
 *
 * Created on February 8, 2016, 3:42 PM
 */

#ifndef ALGEBRAPRIMITIVES_HPP
#define ALGEBRAPRIMITIVES_HPP

#include <query_compilation/primitives/instruction.hpp>

namespace CoGaDB {

  class AttributeAttributeOp : public Instruction {
   public:
    AttributeAttributeOp(const AttributeReference& computed_attr,
                         const AttributeReference& left_attr,
                         const AttributeReference& right_attr,
                         const ColumnAlgebraOperation& alg_op)
        : Instruction(ALGEBRA_INSTR),
          computed_attr_(computed_attr),
          left_operand_(left_attr),
          right_operand_(right_attr),
          alg_op_(alg_op) {}
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

   private:
    AttributeReference computed_attr_;
    AttributeReference left_operand_;
    AttributeReference right_operand_;
    ColumnAlgebraOperation alg_op_;
  };

  class ConstantAttributeOp : public Instruction {
   public:
    ConstantAttributeOp(const AttributeReference& computed_attr,
                        const boost::any constant,
                        const AttributeReference& right_attr,
                        const ColumnAlgebraOperation& alg_op)
        : Instruction(ALGEBRA_INSTR),
          computed_attr_(computed_attr),
          left_operand_(constant),
          right_operand_(right_attr),
          alg_op_(alg_op) {}
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

   private:
    AttributeReference computed_attr_;
    boost::any left_operand_;
    AttributeReference right_operand_;
    ColumnAlgebraOperation alg_op_;
  };

  class AttributeConstantOp : public Instruction {
   public:
    AttributeConstantOp(const AttributeReference& computed_attr,
                        const AttributeReference& left_attr,
                        const boost::any constant,
                        const ColumnAlgebraOperation& alg_op)
        : Instruction(ALGEBRA_INSTR),
          computed_attr_(computed_attr),
          left_operand_(left_attr),
          right_operand_(constant),
          alg_op_(alg_op) {}
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

   private:
    AttributeReference computed_attr_;
    AttributeReference left_operand_;
    boost::any right_operand_;
    ColumnAlgebraOperation alg_op_;
  };
}

#endif /* ALGEBRAPRIMITIVES_HPP */
