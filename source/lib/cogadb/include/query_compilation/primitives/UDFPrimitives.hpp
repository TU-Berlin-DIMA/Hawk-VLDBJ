/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   UDFPrimitives.hpp
 * Author: dbronesk
 *
 * Created on February 8, 2016, 3:37 PM
 */

#ifndef UDFPRIMITIVES_HPP
#define UDFPRIMITIVES_HPP

#include <query_compilation/primitives/instruction.hpp>
#include <query_compilation/user_defined_code.hpp>

namespace CoGaDB {

  class MapUDF : public Instruction {
   public:
    MapUDF(Map_UDF_Result mapUDFCode)
        : Instruction(MAP_UDF_INSTR), resultUDF_(mapUDFCode) {}

    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;

    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

   private:
    Map_UDF_Result resultUDF_;
  };
}

#endif /* UDFPRIMITIVES_HPP */
