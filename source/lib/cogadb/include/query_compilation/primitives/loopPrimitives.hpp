/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   loopPrimitives.hpp
 * Author: dbronesk
 *
 * Created on February 9, 2016, 1:11 PM
 */

#ifndef LOOPPRIMITIVES_HPP
#define LOOPPRIMITIVES_HPP

#include <query_compilation/primitives/instruction.hpp>

namespace CoGaDB {
  class Loop;
  typedef boost::shared_ptr<Loop> LoopPtr;

  class Materialization : public Instruction {
   public:
    Materialization(ProjectionParam param)
        : Instruction(MATERIALIZATION_INSTR), param_(param) {}
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

    bool supportsPredication() const override final;

   protected:
    ProjectionParam param_;
  };

  class Loop : public Instruction {
   public:
    Loop(TablePtr table, uint32_t version, uint32_t rangeDiv = 1,
         uint32_t step = 1);
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override;
    virtual const std::string toString() const override;
    uint64_t getNumberOfElements() const;
    const std::string getNumberOfElementsExpression() const;
    const std::string getVarNameNumberOfElements() const;
    const std::string getLoopVariableName() const;
    void setLoopVarName(std::string loop_var);
    std::string getLoopVarName() const;
    void setStep(uint32_t step);
    uint32_t getStep() const;
    void setRangeDiv(uint32_t rangeDiv);
    uint32_t getRangeDiv() const;
    uint32_t getVersion() const;
    void setVersion(uint32_t version);
    TablePtr getTable() const;

    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

   protected:
    TablePtr table_;
    uint32_t version_;
    uint32_t rangeDiv_;
    uint32_t step_;
    std::string loop_var_;
  };
  typedef boost::shared_ptr<Loop> LoopPtr;

  class ResidualLoop : public Loop {
   public:
    ResidualLoop(uint32_t loop_var_mult, TablePtr table, uint32_t version,
                 uint32_t rangeDiv = 1, uint32_t step = 1)
        : Loop(table, version, rangeDiv, step), loop_var_mult_(loop_var_mult) {}
    ResidualLoop(LoopPtr before)
        : Loop(before->getTable(), before->getVersion() + 1,
               before->getRangeDiv(), before->getStep()),
          loop_var_mult_(before->getRangeDiv()) {}
    virtual const GeneratedCodePtr getCode(CodeGenerationTarget target);
    virtual const std::string toString() const;
    void setLoop_var_mult(uint32_t loop_var_mult);
    uint32_t getLoop_var_mult() const;

   protected:
    uint32_t loop_var_mult_;
  };

  typedef boost::shared_ptr<ResidualLoop> ResidualLoopPtr;

  class ConstantLoop : public Loop {
   public:
    ConstantLoop(const std::string& parent_loop_var, uint32_t begin,
                 uint32_t end, TablePtr table, uint32_t version,
                 uint32_t loopVarMult = 1, uint32_t rangeDiv = 1,
                 uint32_t step = 1)
        : Loop(table, version, rangeDiv, step),
          begin_(begin),
          end_(end),
          loop_var_mult_(loopVarMult),
          parent_loop_var_(parent_loop_var) {}

    ConstantLoop(uint32_t begin, uint32_t end, LoopPtr before)
        : Loop(before->getTable(), before->getVersion() + 1,
               before->getRangeDiv(), before->getStep()),
          begin_(begin),
          end_(end),
          loop_var_mult_(before->getRangeDiv()),
          parent_loop_var_(before->getLoopVarName()) {}

    virtual const GeneratedCodePtr getCode(CodeGenerationTarget target);
    virtual const std::string toString() const;

    uint64_t getNumberOfElements() const;
    const std::string getNumberOfElementsExpression() const;
    const std::string getVarNameNumberOfElements() const;
    const std::string getLoopVariableName() const;

   protected:
    uint32_t begin_;
    uint32_t end_;
    uint32_t loop_var_mult_;
    const std::string parent_loop_var_;
  };

  typedef boost::shared_ptr<ConstantLoop> ConstantLoopPtr;

  class ProduceTuples : public Instruction {
   public:
    ProduceTuples(const ScanParam& scanned_attributes,
                  const ProjectionParam& projection_attributes,
                  const std::map<std::string, AttributeReferencePtr>&
                      columns_to_decompress)
        : Instruction(PRODUCE_TUPLE_INSTR),
          scanned_attributes_(scanned_attributes),
          param_(projection_attributes),
          columns_to_decompress_(columns_to_decompress) {}
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override;
    virtual const std::string toString() const override;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

   private:
    ScanParam scanned_attributes_;
    ProjectionParam param_;
    std::map<std::string, AttributeReferencePtr> columns_to_decompress_;
  };
}

#endif /* LOOPPRIMITIVES_HPP */
