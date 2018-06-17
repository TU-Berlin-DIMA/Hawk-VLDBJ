
/*
 * File:   instruction.hpp
 * Author: dbronesk
 *
 * Created on February 1, 2016, 2:56 PM
 */

#ifndef INSTRUCTION_HPP
#define INSTRUCTION_HPP

#include <boost/make_shared.hpp>
#include <core/attribute_reference.hpp>
#include <set>
#include <sstream>

namespace CoGaDB {

  enum CodeGenerationTarget {
    C_TARGET_CODE,
    CUDA_C_TARGET_CODE,
    OCL_TARGET_CODE,
    LLVM_TARGET_CODE
  };

  enum InstructionType {
    FILTER_INSTR,
    MATERIALIZATION_INSTR,
    HASH_TABLE_BUILD_INSTR,
    HASH_TABLE_PROBE_INSTR,
    ALGEBRA_INSTR,
    AGGREGATE_INSTR,
    HASH_AGGREGATE_INSTR,
    BITPACKED_GROUPING_KEY_INSTR,
    GENERIC_GROUPING_KEY_INSTR,
    CROSS_JOIN_INSTR,
    LOOP_INSTR,
    PRODUCE_TUPLE_INSTR,
    MAP_UDF_INSTR,
    INCREMENT_INSTR
  };

  class GeneratedCode {
   public:
    std::stringstream kernel_header_and_types_code_block_;
    std::stringstream header_and_types_code_block_;
    std::stringstream fetch_input_code_block_;
    std::stringstream declare_variables_code_block_;
    std::stringstream init_variables_code_block_;
    std::list<std::string> upper_code_block_;
    std::list<std::string> lower_code_block_;
    std::stringstream after_for_loop_code_block_;
    std::stringstream create_result_table_code_block_;
    std::stringstream clean_up_code_block_;
  };

  typedef boost::shared_ptr<GeneratedCode> GeneratedCodePtr;

  bool addToCode(GeneratedCodePtr code, GeneratedCodePtr code_to_add);

  class Instruction {
   public:
    Instruction(const InstructionType& instr_type, uint32_t loop_var = 1,
                uint8_t offset = 0);
    virtual ~Instruction() {}
    virtual const GeneratedCodePtr getCode(CodeGenerationTarget target) = 0;
    virtual const std::string getName();
    virtual const std::string toString() const = 0;
    InstructionType getInstructionType() const;
    void setLoopVar(uint32_t version);
    uint32_t getLoopVar() const;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) = 0;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) = 0;

    virtual bool supportsPredication() const;
    virtual void setPredicationMode(const PredicationMode& pred_mode);
    virtual PredicationMode getPredicationMode() const;

   protected:
    std::pair<std::string, std::string> getVariableNameAndTypeFromAttributeRef(
        const AttributeReference& ref, bool ignore_compressed = false);

    struct AttributeRefAccessWrapper {
      static const AttributeReference& get(const AttributeReference& ref) {
        return ref;
      }

      static const AttributeReference& get(const AttributeReferencePtr& ref) {
        return *ref;
      }
    };

    template <typename T>
    std::map<std::string, std::string>
    getVariableNameAndTypeFromInputAttributeRefVector(
        const std::vector<T>& vec, bool ignore_compressed = false) {
      std::map<std::string, std::string> result;

      for (const auto& attr : vec) {
        if (!isComputed(AttributeRefAccessWrapper::get(attr))) {
          result.insert(getVariableNameAndTypeFromAttributeRef(
              AttributeRefAccessWrapper::get(attr), ignore_compressed));
        }
      }

      return result;
    }

   private:
    InstructionType instr_type_;
    uint32_t loop_var_;
    PredicationMode predication_mode_;
  };

  typedef boost::shared_ptr<Instruction> InstructionPtr;

}  // end namespace CoGaDB

#endif /* INSTRUCTION_HPP */
