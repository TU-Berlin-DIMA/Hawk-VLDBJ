

#include <query_compilation/execution_strategy/pipeline.hpp>

#include <iomanip>
#include <list>
#include <set>
#include <sstream>

#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <util/getname.hpp>
#include <util/iostream.hpp>

#include <util/time_measurement.hpp>

#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
//#include <ctime>

#include <google/dense_hash_map>  // streaming operators etc.

#include <core/variable_manager.hpp>
#include <util/code_generation.hpp>
#include <util/shared_library.hpp>

#include <query_compilation/pipeline_info.hpp>

#include <util/opencl_runtime.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

PipelineExecutionStrategy::PipelineExecutionStrategy(
    CodeGenerationTarget target)
    : code_(new GeneratedCode()), target_(target) {}

PipelineExecutionStrategy::~PipelineExecutionStrategy() {}

void PipelineExecutionStrategy::addInstruction(InstructionPtr instr) {
  addInstruction_impl(instr);
}

PlainC::PlainC() : PipelineExecutionStrategy(C_TARGET_CODE) {}

void PlainC::addInstruction_impl(InstructionPtr instr) {
  GeneratedCodePtr gen_code = instr->getCode(C_TARGET_CODE);
  if (!gen_code) {
    COGADB_FATAL_ERROR("", "");
  }
  //        std::cout << "Instr: " << instr->toString() << std::endl;
  //        std::cout << "Code: " << std::endl;
  //        std::cout << "Header and Types: " << std::endl
  //                  << "'" << gen_code->mHeaderAndTypesBlock.str() << "'" <<
  //                  std::endl;
  //        std::cout << "Fetch Input Data: " << std::endl
  //                  << "'" << gen_code->mFetchInputCodeBlock.str() << "'" <<
  //                  std::endl;

  if (!addToCode(code_, gen_code)) {
    COGADB_FATAL_ERROR("", "");
  }
}

const std::pair<std::string, std::string> PlainC::getCode(
    const ProjectionParam& param, const ScanParam& scanned_attributes,
    PipelineEndType pipe_end, const std::string& result_table_name,
    const std::map<std::string, AttributeReferencePtr>& columns_to_decompress) {
  GeneratedCodePtr code = code_;
  std::stringstream out;
  out << "#include <query_compilation/minimal_api_c.h>" << std::endl
      << std::endl;
  /* all imports and declarations */
  out << code->header_and_types_code_block_.str() << std::endl;
  /* write function signature */
  out << "const C_Table* compiled_query(C_Table** c_tables) {" << std::endl;
  out << code->fetch_input_code_block_.str() << std::endl;
  /* all code for query function definition and input array retrieval */
  out << code->declare_variables_code_block_.str() << std::endl;
  // TODO, we need to find a better place for this!
  out << "uint64_t current_result_size = 0;" << std::endl;
  out << "uint64_t allocated_result_elements = 10000;" << std::endl;
  out << code->init_variables_code_block_.str() << std::endl;
  /* add for loop and it's contents */
  std::list<std::string>::const_iterator cit;
  for (cit = code->upper_code_block_.begin();
       cit != code->upper_code_block_.end(); ++cit) {
    out << *cit << std::endl;
  }
  /* if we do not materialize into a hash table during aggregation,
     write result regularely */
  if (pipe_end != MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
    out << "++current_result_size;" << std::endl;
  }
  /* generate closing brackets, pointer chasing, and cleanup operations */
  for (cit = code->lower_code_block_.begin();
       cit != code->lower_code_block_.end(); ++cit) {
    out << *cit << std::endl;
  }
  /* if we do materialize into a hash table during aggregation,
     write copy result from hash table to output arrays */
  if (pipe_end == MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
    std::string temp = code->after_for_loop_code_block_.str();
    out << temp.c_str();
  }
  /* generate code that builds the reslt table using the minimal API */
  out << generateCCodeCreateResultTable(
             param, code->create_result_table_code_block_.str(),
             code->clean_up_code_block_.str(), result_table_name)
      << std::endl;
  out << "}" << std::endl;
  return std::make_pair(out.str(), std::string());
}

}  //  namespace ExecutionStrategy

}  // namespace CoGaDB
