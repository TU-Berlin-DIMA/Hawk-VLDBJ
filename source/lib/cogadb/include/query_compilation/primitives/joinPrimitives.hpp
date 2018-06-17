/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   joinPrimitives.hpp
 * Author: dbronesk
 *
 * Created on February 8, 2016, 9:25 AM
 */

#ifndef JOINPRIMITIVES_HPP
#define JOINPRIMITIVES_HPP

#include <query_compilation/primitives/instruction.hpp>

namespace CoGaDB {

  class CrossJoin : public Instruction {
   public:
    CrossJoin(const AttributeReference& joinAttribute)
        : Instruction(CROSS_JOIN_INSTR), attr_(joinAttribute) {}

    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

   private:
    AttributeReference attr_;
  };
}

#endif /* JOINPRIMITIVES_HPP */
