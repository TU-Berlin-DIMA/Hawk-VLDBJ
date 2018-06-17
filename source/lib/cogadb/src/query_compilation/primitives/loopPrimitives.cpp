/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   loopPrimitives.cpp
 * Author: dbronesk
 *
 * Created on February 9, 2016, 1:11 PM
 */

#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/primitives/loopPrimitives.hpp>

namespace CoGaDB {

const GeneratedCodePtr Loop::getCode(CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  if (target == C_TARGET_CODE) {
    std::stringstream lower;
    std::stringstream for_loop;
    for_loop << "size_t num_elements_" << getTupleIDVarName(table_, version_)
             << " = getNumberOfRows(" << getTableVarName(table_, version_)
             << ")/" << rangeDiv_ << ";" << std::endl;

    for_loop << "size_t " << loop_var_ << ";";

    for_loop << "for (" << loop_var_ << " = 0; " << loop_var_
             << " < num_elements_" << getTupleIDVarName(table_, version_) << ";"
             << loop_var_ << "+=" << step_ << ") {";
    lower << "}";

    code->upper_code_block_.push_back(for_loop.str());
    code->lower_code_block_.push_back(lower.str());
  } else if (target == OCL_TARGET_CODE) {
    std::stringstream lower;
    std::stringstream for_loop;
    for_loop << "size_t num_elements_" << getTupleIDVarName(table_, version_)
             << " = getNumberOfRows(" << getTableVarName(table_, version_)
             << ");" << std::endl;
    for_loop << "size_t result_increment=1;" << std::endl;
    code->upper_code_block_.push_back(for_loop.str());
    code->lower_code_block_.push_back(lower.str());
  } else {
    COGADB_FATAL_ERROR("Cannot handle CodeGenerationTarget: " << target, "");
  }
  return code;
}

const GeneratedCodePtr ResidualLoop::getCode(CodeGenerationTarget) {
  GeneratedCodePtr code(new GeneratedCode());
  std::stringstream lower;
  std::stringstream for_loop;
  for_loop << "size_t full_num_elements_" << getTupleIDVarName(table_, version_)
           << " = getNumberOfRows(" << getTableVarName(table_, version_) << ")/"
           << rangeDiv_ << ";" << std::endl;

  for_loop << "size_t " << loop_var_ << "=" << loop_var_ << "* "
           << loop_var_mult_ << ";";

  for_loop << "for (; " << loop_var_ << " < full_num_elements_"
           << getTupleIDVarName(table_, version_) << ";" << loop_var_
           << "+=" << step_ << ") {";
  lower << "}";

  code->upper_code_block_.push_back(for_loop.str());
  code->lower_code_block_.push_back(lower.str());
  return code;
}

Loop::Loop(TablePtr table, uint32_t version, uint32_t rangeDiv, uint32_t step)
    : Instruction(LOOP_INSTR, version),
      table_(table),
      version_(version),
      rangeDiv_(rangeDiv),
      step_(step),
      loop_var_(getTupleIDVarName(table, version)) {}

const std::string Loop::toString() const {
  std::stringstream str;
  str << "LOOP(" << table_->getName() << " ("
      << static_cast<void*>(table_.get()) << ")"
      << ", rows=" << table_->getNumberofRows() << ", " << loop_var_
      << ", rangeDiv=" << rangeDiv_ << ", step=" << step_ << ")";
  return str.str();
}

void Loop::setLoopVarName(std::string loop_var) { this->loop_var_ = loop_var; }

std::string Loop::getLoopVarName() const { return loop_var_; }

void Loop::setStep(uint32_t step) { this->step_ = step; }

uint32_t Loop::getStep() const { return step_; }

void Loop::setRangeDiv(uint32_t rangeDiv) { this->rangeDiv_ = rangeDiv; }

uint32_t Loop::getRangeDiv() const { return rangeDiv_; }

uint32_t Loop::getVersion() const { return version_; }

TablePtr Loop::getTable() const { return table_; }

void Loop::setVersion(uint32_t version) {
  this->version_ = version;
  loop_var_ = getTupleIDVarName(table_, version_);
}

uint64_t Loop::getNumberOfElements() const { return table_->getNumberofRows(); }

const std::string Loop::getNumberOfElementsExpression() const {
  std::stringstream str;
  str << "size_t num_elements_" << getTupleIDVarName(table_, version_)
      << " = getNumberOfRows(" << getTableVarName(table_, version_) << ");"
      << std::endl;
  return str.str();
}

std::map<std::string, std::string> Loop::getInputVariables(
    CodeGenerationTarget) {
  return {};
}

std::map<std::string, std::string> Loop::getOutputVariables(
    CodeGenerationTarget) {
  return {};
}

const std::string ResidualLoop::toString() const {
  std::stringstream str;
  str << "RESIDUAL_LOOP(" << table_->getName() << " ("
      << static_cast<void*>(table_.get()) << ")"
      << ", rows=" << table_->getNumberofRows() << ", " << loop_var_
      << "LoopMult: " << loop_var_mult_ << ", rangeDiv=" << rangeDiv_
      << ", step=" << step_ << ")";
  return str.str();
}

void ResidualLoop::setLoop_var_mult(uint32_t loop_var_mult) {
  loop_var_mult_ = loop_var_mult;
}

uint32_t ResidualLoop::getLoop_var_mult() const { return loop_var_mult_; }

const std::string ConstantLoop::toString() const {
  std::stringstream str;
  str << "CONSTANT_LOOP(" << table_->getName() << " ("
      << static_cast<void*>(table_.get()) << ")"
      << ", rows=" << table_->getNumberofRows() << ", " << loop_var_
      << ", Begin: " << begin_ << ",  End: " << end_
      << ", rangeDiv=" << rangeDiv_ << ", step=" << step_ << ")";
  return str.str();
}

const std::string Loop::getVarNameNumberOfElements() const {
  std::stringstream str;
  str << "num_elements_" << getTupleIDVarName(table_, version_);
  return str.str();
}

const GeneratedCodePtr Materialization::getCode(CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());

  std::stringstream ss;
  std::string result_index_name;

  if (target == C_TARGET_CODE) {
    ss << "if (current_result_size >= allocated_result_elements) {"
       << std::endl;
    ss << "   allocated_result_elements *= 1.4;" << std::endl;
    ss << getCodeReallocResultMemory(param_);
    ss << "}" << std::endl;
    result_index_name = "current_result_size";
  } else if (target == OCL_TARGET_CODE) {
    if (getPredicationMode() == PREDICATED_EXECUTION) {
      result_index_name =
          "write_pos * result_increment + "
          "(num_elements + 1) * (1 - result_increment)";
    } else {
      result_index_name = "write_pos";
    }
  }

  for (const auto& param : param_) {
    ss << getResultArrayVarName(param) << "[" << result_index_name
       << "] = " << getCompressedElementAccessExpression(param) << ";"
       << std::endl;
  }

  code->declare_variables_code_block_ << getCodeDeclareResultMemory(param_);

  code->init_variables_code_block_ << getCodeReallocResultMemory(param_);
  code->upper_code_block_.push_back(ss.str());
  code->lower_code_block_.push_back("");
  return code;
}

const std::string Materialization::toString() const {
  std::stringstream str;

  if (getPredicationMode() == PREDICATED_EXECUTION) {
    str << "MATERIALIZE_PREDICATED(";
  } else {
    str << "MATERIALIZE(";
  }

  for (size_t i = 0; i < param_.size(); ++i) {
    str << param_[i].getResultAttributeName();
    if (i + 1 < param_.size()) str << ", ";
  }
  str << ")";
  return str.str();
}

std::map<std::string, std::string> Materialization::getInputVariables(
    CodeGenerationTarget) {
  std::map<std::string, std::string> result;

  for (const auto& ref : param_) {
    if (!isComputed(ref)) {
      result.insert(
          std::make_pair(getVarName(ref), getResultType(ref, false) + "*"));
    }
  }

  return result;
}

std::map<std::string, std::string> Materialization::getOutputVariables(
    CodeGenerationTarget) {
  std::map<std::string, std::string> result;

  for (const auto& ref : param_) {
    result.insert(std::make_pair(getResultArrayVarName(ref),
                                 getResultType(ref, false) + "*"));
  }

  return result;
}

bool Materialization::supportsPredication() const { return true; }

const std::string Loop::getLoopVariableName() const {
  std::stringstream str;
  str << getTupleIDVarName(table_, version_);
  return str.str();
}

const GeneratedCodePtr ConstantLoop::getCode(CodeGenerationTarget) {
  GeneratedCodePtr code(new GeneratedCode());
  std::stringstream lower;
  std::stringstream for_loop;
  for_loop << "size_t end_num_elements_" << getTupleIDVarName(table_, version_)
           << " = " << parent_loop_var_ << " * " << loop_var_mult_ << " + "
           << end_ << ";" << std::endl;
  for_loop << "size_t " << loop_var_ << " = " << parent_loop_var_ << " * "
           << loop_var_mult_ << " + " << begin_ << ";" << std::endl;
  ;

  for_loop << "for (; " << loop_var_ << " < "
           << "end_num_elements_" << getTupleIDVarName(table_, version_) << ";"
           << loop_var_ << "+=" << step_ << ") {";
  lower << "}";

  code->upper_code_block_.push_back(for_loop.str());
  code->lower_code_block_.push_back(lower.str());
  return code;
}

const GeneratedCodePtr ProduceTuples::getCode(CodeGenerationTarget) {
  GeneratedCodePtr code(new GeneratedCode());

  std::set<std::string> scanned_tables;
  bool debug_code_generator = false;

  size_t param_id = 0;
  ScanParam::const_iterator cit;
  for (cit = scanned_attributes_.begin(); cit != scanned_attributes_.end();
       ++cit, ++param_id) {
    if (cit->getTable() == NULL) {
      COGADB_FATAL_ERROR("Found no valid TablePtr in Scan Attribute: "
                             << cit->getVersionedAttributeName(),
                         "");
    }

    if (scanned_tables.find(cit->getTable()->getName()) ==
        scanned_tables.end()) {
      // we assume that the same ScanParam passed to this function
      // is also passed to the compiled query function
      code->fetch_input_code_block_ << "C_Table* " << getTableVarName(*cit)
                                    << " = c_tables[" << param_id << "];"
                                    << std::endl;
      code->fetch_input_code_block_ << "assert(" << getTableVarName(*cit)
                                    << " != NULL);" << std::endl;
      scanned_tables.insert(cit->getTable()->getName());

      if (debug_code_generator) {
        std::cout << "[DEBUG:] Produce Tuples for Table "
                  << getTableVarName(*cit) << std::endl;
      }
    }
    code->fetch_input_code_block_
        << "C_Column* " << getInputColumnVarName(*cit) << " = getColumnById("
        << getTableVarName(*cit) << ","
        << getConstant(
               cit->getTable()->getColumnIdbyColumnName(CoGaDB::toString(*cit)))
        << ");" << std::endl;
    code->fetch_input_code_block_ << "uint64_t " << getInputColumnVarName(*cit)
                                  << "_length = getNumberOfRows("
                                  << getTableVarName(*cit) << ");" << std::endl;
    code->clean_up_code_block_
        << "releaseColumn(" << getInputColumnVarName(*cit) << ");" << std::endl;

    code->fetch_input_code_block_
        << "if (!" << getInputColumnVarName(*cit) << ") { printf(\"Column '"
        << getInputColumnVarName(*cit) << "' not found!\\n\"); return NULL; }"
        << std::endl;

    code->fetch_input_code_block_ << getArrayFromColumnCode(*cit);
  }
  /* generate code to decompress the columns we cannot work on directly
   * (e.g., ">" or "<" predicates on string columns) */
  for (std::map<std::string, AttributeReferencePtr>::const_iterator itr(
           columns_to_decompress_.begin());
       itr != columns_to_decompress_.end(); ++itr) {
    code->fetch_input_code_block_ << getArrayFromColumnCode(*itr->second, true);
  }
  return code;
}

const std::string ProduceTuples::toString() const {
  std::stringstream str;
  str << "PRODUCE_TUPLES(";
  for (size_t i = 0; i < scanned_attributes_.size(); ++i) {
    str << CoGaDB::toString(scanned_attributes_[i]);
    //    if(isPersistent(scanned_attributes_[i].getTable() )){
    //      str << " (Persistent)";
    //    }else{
    //      str << " (Intermediate Result TablePtr: " <<
    //      scanned_attributes_[i].getTable() << ")";
    //    }
    if (i + 1 < scanned_attributes_.size()) str << ", ";
  }
  str << ")";
  return str.str();
}

std::map<std::string, std::string> ProduceTuples::getInputVariables(
    CodeGenerationTarget) {
  return {};
}

std::map<std::string, std::string> ProduceTuples::getOutputVariables(
    CodeGenerationTarget) {
  return {};
}
}
