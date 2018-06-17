
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/primitives/instruction.hpp>

namespace CoGaDB {

Instruction::Instruction(const InstructionType& instr_type, uint32_t loop_var,
                         uint8_t offset)
    : instr_type_(instr_type), loop_var_(loop_var), predication_mode_() {}

const std::string Instruction::getName() { return ""; }

InstructionType Instruction::getInstructionType() const { return instr_type_; }

void Instruction::setLoopVar(uint32_t loop_var) { this->loop_var_ = loop_var; }

uint32_t Instruction::getLoopVar() const { return loop_var_; }

bool Instruction::supportsPredication() const {
  /* predication support is off by default,
   * needs to be explictely overridden by
   * overriding this function in a base class
   */
  return false;
}

void Instruction::setPredicationMode(const PredicationMode& pred_mode) {
  if (pred_mode == PREDICATED_EXECUTION) {
    if (!this->supportsPredication()) {
      COGADB_FATAL_ERROR(
          "Predication not supported by instruction: " << this->toString(), "");
    }
  }
  predication_mode_ = pred_mode;
}

PredicationMode Instruction::getPredicationMode() const {
  return predication_mode_;
}

std::pair<std::string, std::string>
Instruction::getVariableNameAndTypeFromAttributeRef(
    const AttributeReference& ref, bool ignore_compressed) {
  std::string pointer = "*";

  // TODO
  if (ref.getAttributeReferenceType() == COMPUTED_ATTRIBUTE) {
    pointer = "";
  }

  return std::make_pair(getVarName(ref),
                        getResultType(ref, ignore_compressed) + pointer);
}

bool addToCode(GeneratedCodePtr code, GeneratedCodePtr code_to_add) {
  if (!code || !code_to_add) return false;
  code->header_and_types_code_block_
      << code_to_add->header_and_types_code_block_.str();
  code->fetch_input_code_block_ << code_to_add->fetch_input_code_block_.str();
  code->declare_variables_code_block_
      << code_to_add->declare_variables_code_block_.str();
  code->init_variables_code_block_
      << code_to_add->init_variables_code_block_.str();
  code->upper_code_block_.insert(code->upper_code_block_.end(),
                                 code_to_add->upper_code_block_.begin(),
                                 code_to_add->upper_code_block_.end());
  code->lower_code_block_.insert(code->lower_code_block_.begin(),
                                 code_to_add->lower_code_block_.begin(),
                                 code_to_add->lower_code_block_.end());
  code->after_for_loop_code_block_
      << code_to_add->after_for_loop_code_block_.str();
  code->create_result_table_code_block_
      << code_to_add->create_result_table_code_block_.str();
  code->clean_up_code_block_ << code_to_add->clean_up_code_block_.str();
  code->kernel_header_and_types_code_block_
      << code_to_add->kernel_header_and_types_code_block_.str();
  return true;
}

}  // end namespace CoGaDB
