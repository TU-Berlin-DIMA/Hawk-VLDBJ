/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   hashPrimitives.hpp
 * Author: dbronesk
 *
 * Created on February 3, 2016, 8:02 PM
 */

#ifndef HASHPRIMITIVES_HPP
#define HASHPRIMITIVES_HPP

#include <query_compilation/hash_table_generator.hpp>
#include <query_compilation/primitives/instruction.hpp>

namespace CoGaDB {

  class HashPut : public Instruction {
   public:
    HashPut(HashTableGeneratorPtr HTGen, const AttributeReference& build_attr);
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

   private:
    HashTableGeneratorPtr htgen_;
    AttributeReference build_attr_;
  };

  class HashProbe : public Instruction {
   public:
    HashProbe(HashTableGeneratorPtr HTGen,
              const AttributeReference& buildAttribute,
              const AttributeReference& probeAttribute);

    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

    bool supportsPredication() const override final;

   private:
    HashTableGeneratorPtr htgen_;
    AttributeReference build_attr_;
    AttributeReference probe_attr_;
  };
}

#endif /* HASHPRIMITIVES_HPP */
