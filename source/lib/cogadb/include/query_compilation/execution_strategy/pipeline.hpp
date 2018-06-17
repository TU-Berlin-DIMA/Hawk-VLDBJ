#ifndef PIPELINE_EXECUTION_STRATEGY_HPP
#define PIPELINE_EXECUTION_STRATEGY_HPP

#include <iomanip>
#include <list>
#include <set>
#include <sstream>

#include <core/attribute_reference.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/primitives/instruction.hpp>
#include <query_compilation/primitives/loopPrimitives.hpp>
#include <util/getname.hpp>
#include <util/iostream.hpp>

namespace CoGaDB {

  namespace ExecutionStrategy {

    class PipelineExecutionStrategy {
     public:
      void addInstruction(InstructionPtr instr);
      virtual const std::pair<std::string, std::string> getCode(
          const ProjectionParam& param, const ScanParam& scanned_attributes,
          PipelineEndType pipe_end, const std::string& result_table_name,
          const std::map<std::string, AttributeReferencePtr>&
              columns_to_decompress) = 0;
      virtual ~PipelineExecutionStrategy();

     protected:
      PipelineExecutionStrategy(CodeGenerationTarget target);
      virtual void addInstruction_impl(InstructionPtr instr) = 0;
      GeneratedCodePtr code_;
      CodeGenerationTarget target_;
    };

    typedef boost::shared_ptr<PipelineExecutionStrategy> PipelinePtr;

    class PlainC : public PipelineExecutionStrategy {
     public:
      PlainC();
      void addInstruction_impl(InstructionPtr instr);
      const std::pair<std::string, std::string> getCode(
          const ProjectionParam& param, const ScanParam& scanned_attributes,
          PipelineEndType pipe_end, const std::string& result_table_name,
          const std::map<std::string, AttributeReferencePtr>&
              columns_to_decompress);
    };

  }  // namespace ExecutionStrategy

}  // end namespace CoGaDB

#endif  // PIPELINE_EXECUTION_STRATEGY_HPP
