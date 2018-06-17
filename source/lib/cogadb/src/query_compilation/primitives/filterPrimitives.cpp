
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/primitives/filterPrimitives.hpp>

namespace CoGaDB {

const GeneratedCodePtr Filter::getCode(CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  std::stringstream upper;
  std::stringstream lower;
  if (PREDICATED_EXECUTION == getPredicationMode()) {
    upper << "result_increment=" << pred_expr_->getCPPExpression() << ";";
  } else {
    upper << "if(" << pred_expr_->getCPPExpression() << "){";
    lower << "}";
  }
  code->upper_code_block_.push_back(upper.str());
  code->lower_code_block_.push_back(lower.str());
  return code;
}

const std::string Filter::toString() const {
  std::stringstream str;
  if (PREDICATED_EXECUTION == getPredicationMode()) {
    str << "PREDICATED_FILTER";
  } else {
    str << "FILTER";
  }
  str << "(" << pred_expr_->toString() << ")";
  return str.str();
}

std::map<std::string, std::string> Filter::getInputVariables(
    CodeGenerationTarget target) {
  return getVariableNameAndTypeFromInputAttributeRefVector(
      pred_expr_->getScannedAttributes());
}

std::map<std::string, std::string> Filter::getOutputVariables(
    CodeGenerationTarget target) {
  return {};
}

bool Filter::supportsPredication() const { return true; }

const GeneratedCodePtr SSEFilter::getCode(CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  std::stringstream upper;
  std::stringstream lower;

  code->header_and_types_code_block_
      << "#include \"nmmintrin.h\"\n #include \"smmintrin.h\"\n";
  code->fetch_input_code_block_
      << "__m128i maskOfOnes;\n"
      << "__m128i " << sse_result_name_ << ";\n"
      << "maskOfOnes = _mm_cmpeq_epi8(maskOfOnes,maskOfOnes);\n";
  uint32_t prednum = 1;
  std::pair<std::string, std::string> SSEExpr =
      pred_expr_->getSSEExpression(prednum);
  code->fetch_input_code_block_ << SSEExpr.first;
  upper << sse_result_name_ << "= " << SSEExpr.second << ";\n"
        << "if(!_mm_testz_si128(" << sse_result_name_ << ", maskOfOnes)){";
  lower << "}";
  code->upper_code_block_.push_back(upper.str());
  code->lower_code_block_.push_back(lower.str());

  return code;
}

const std::string SSEMaskFilter::toString() const {
  std::stringstream str;
  str << "SSEMask_FILTER(" << pred_expr_->toString() << ")";
  return str.str();
}

const GeneratedCodePtr SSEMaskFilter::getCode(CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  std::stringstream upper;
  std::stringstream lower;

  code->fetch_input_code_block_ << "int SSEmask;\n";

  upper << "SSEmask = _mm_movemask_ps((__m128)" << sse_result_name_ << ");"
        << "if((SSEmask >> ("
        << getTupleIDVarName(
               (pred_expr_->getScannedAttributes()[0])->getTable(),
               getLoopVar())
        << "-" << getTupleIDVarName(
                      (pred_expr_->getScannedAttributes()[0])->getTable(),
                      getLoopVar() - 1)
        << "*4"
        << ")) & 1 ){";
  lower << "}";
  code->upper_code_block_.push_back(upper.str());
  code->lower_code_block_.push_back(lower.str());

  return code;
}

const std::string SSEFilter::toString() const {
  std::stringstream str;
  str << "SIMD_FILTER(" << pred_expr_->toString() << ")";
  return str.str();
}
}
