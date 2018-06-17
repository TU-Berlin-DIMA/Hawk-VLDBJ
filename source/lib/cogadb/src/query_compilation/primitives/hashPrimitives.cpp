/*
 * File:   hashPrimitives.cpp
 * Author: dbronesk
 *
 * Created on February 3, 2016, 8:02 PM
 */

#include <query_compilation/primitives/hashPrimitives.hpp>
#include "query_compilation/code_generators/code_generator_utils.hpp"

namespace CoGaDB {

HashPut::HashPut(HashTableGeneratorPtr HTGen,
                 const AttributeReference& build_attr)
    : Instruction(HASH_TABLE_BUILD_INSTR),
      htgen_(HTGen),
      build_attr_(build_attr) {}

const GeneratedCodePtr HashPut::getCode(CodeGenerationTarget target) {
  auto code = boost::make_shared<GeneratedCode>();
  std::stringstream upper;

  if (target == OCL_TARGET_CODE) {
    code->kernel_header_and_types_code_block_
        << htgen_->generateCodeForHeaderAndTypesBlock();
  } else {
    code->header_and_types_code_block_
        << htgen_->generateCodeForHeaderAndTypesBlock();
  }

  code->declare_variables_code_block_ << htgen_->generateCodeDeclareHashTable();

  std::stringstream number_of_elements;
  if (target == C_TARGET_CODE) {
    size_t estimated_num_elements = 10000;
    if (build_attr_.getTable()) {
      estimated_num_elements = build_attr_.getTable()->getNumberofRows();
    }
    number_of_elements << estimated_num_elements;
  } else if (target == OCL_TARGET_CODE) {
    number_of_elements << "allocated_result_elements";
  }

  code->init_variables_code_block_
      << htgen_->generateCodeInitHashTable(true, number_of_elements.str());

  upper << htgen_->generateCodeInsertIntoHashTable(getLoopVar());
  code->upper_code_block_.push_back(upper.str());

  code->create_result_table_code_block_
      << "C_HashTable* " << getHashTableVarName(build_attr_) << "_system = "
      << "createSystemHashTable(" << htgen_->getHTVarNameForSystemHT()
      << ", (HashTableCleanupFunctionPtr)&"
      << htgen_->generateIdentifierCleanupHandler() << ", \" "
      << htgen_->identifier_ << "\");" << std::endl;

  code->create_result_table_code_block_
      << "if (!addHashTable(result_table, \""
      << build_attr_.getResultAttributeName() << "\", "
      << getHashTableVarName(build_attr_) << "_system)) {"
      << "printf(\"Error adding hash table for attribute '"
      << build_attr_.getResultAttributeName() << "' to result!\\n\");"
      << std::endl
      << "return NULL;" << std::endl
      << "}" << std::endl;

  code->clean_up_code_block_ << "releaseHashTable("
                             << getHashTableVarName(build_attr_) << "_system);"
                             << std::endl;

  return code;
}

const std::string HashPut::toString() const {
  std::stringstream str;
  str << "HASH_PUT(" << CoGaDB::toString(build_attr_) << ", "
      << htgen_->identifier_ << ")";
  return str.str();
}

std::map<std::string, std::string> HashPut::getInputVariables(
    CodeGenerationTarget target) {
  auto extra_input = htgen_->getBuildExtraInputVariables();

  extra_input.insert(getVariableNameAndTypeFromAttributeRef(build_attr_));

  return extra_input;
}

std::map<std::string, std::string> HashPut::getOutputVariables(
    CodeGenerationTarget target) {
  auto extra_output = htgen_->getBuildExtraOutputVariables();

  extra_output.insert(
      {getHashTableVarName(build_attr_), htgen_->getHashTableCType()});

  return extra_output;
}

HashProbe::HashProbe(HashTableGeneratorPtr HTGen,
                     const AttributeReference& buildAttribute,
                     const AttributeReference& probeAttribute)
    : Instruction(HASH_TABLE_PROBE_INSTR),
      htgen_(HTGen),
      build_attr_(buildAttribute),
      probe_attr_(probeAttribute) {}

const GeneratedCodePtr HashProbe::getCode(CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());

  if (target == OCL_TARGET_CODE) {
    code->kernel_header_and_types_code_block_
        << htgen_->generateCodeForHeaderAndTypesBlock();
  } else {
    code->header_and_types_code_block_
        << htgen_->generateCodeForHeaderAndTypesBlock();
  }

  code->declare_variables_code_block_
      << "C_HashTable* generic_hashtable_" << getHashTableVarName(build_attr_)
      << "=getHashTable(" << getTableVarName(build_attr_) << ", "
      << "\"" << createFullyQualifiedColumnIdentifier(
                     boost::make_shared<AttributeReference>(build_attr_))
      << "\");" << std::endl;

  code->declare_variables_code_block_ << htgen_->generateCodeDeclareHashTable();
  code->init_variables_code_block_ << htgen_->generateCodeInitHashTable(false);

  ProbeHashTableCode probeCode =
      htgen_->generateCodeProbeHashTable(probe_attr_, getLoopVar());
  code->upper_code_block_.push_back("TID " + getTupleIDVarName(build_attr_) +
                                    " = 0;");
  code->upper_code_block_.push_back(probeCode.first);
  code->lower_code_block_.push_back(probeCode.second);

  code->clean_up_code_block_ << "releaseHashTable(generic_hashtable_"
                             << getHashTableVarName(build_attr_) << ");"
                             << std::endl;

  return code;
}

const std::string HashProbe::toString() const {
  std::stringstream str;
  str << "HASH_PROBE(HT(" << CoGaDB::toString(build_attr_) << ", "
      << htgen_->identifier_ << "), " << CoGaDB::toString(probe_attr_) << ")";
  return str.str();
}

std::map<std::string, std::string> HashProbe::getInputVariables(
    CodeGenerationTarget target) {
  auto hash_table_input = std::make_pair(getHashTableVarName(build_attr_),
                                         htgen_->getHashTableCType());
  auto probe_input = getVariableNameAndTypeFromAttributeRef(probe_attr_);
  auto extra_input = htgen_->getProbeExtraInputVariables();

  extra_input.insert(hash_table_input);
  extra_input.insert(probe_input);

  return extra_input;
}

std::map<std::string, std::string> HashProbe::getOutputVariables(
    CodeGenerationTarget target) {
  return {};
}

bool HashProbe::supportsPredication() const { return false; }
}
