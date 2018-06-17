#include <query_compilation/code_generators/multi_stage_code_generator.hpp>

#include <iomanip>
#include <list>
#include <set>
#include <sstream>

#include <core/selection_expression.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <util/getname.hpp>
#include <util/iostream.hpp>

#include <dlfcn.h>
#include <stdlib.h>
#include <core/data_dictionary.hpp>
#include <util/functions.hpp>
#include <util/time_measurement.hpp>

#include <boost/filesystem/operations.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>
#include <ctime>

#include <google/dense_hash_map>  // streaming operators etc.

#include <core/variable_manager.hpp>
#include <util/code_generation.hpp>
#include <util/shared_library.hpp>

#include <query_compilation/execution_strategy/ocl_aggregation.h>
#include <query_compilation/execution_strategy/ocl_aggregation_single_pass.h>
#include <query_compilation/execution_strategy/ocl_grouped_aggregation.h>
#include <query_compilation/execution_strategy/ocl_grouped_aggregation_atomic.h>
#include <query_compilation/execution_strategy/ocl_grouped_aggregation_atomic_workgroup_ht.h>
#include <query_compilation/execution_strategy/ocl_grouped_aggregation_reduce_atomics.h>
#include <query_compilation/execution_strategy/ocl_aggregation_multipass.hpp>
#include <query_compilation/execution_strategy/ocl_grouped_aggregation_fused_scan_global_group.hpp>
#include <query_compilation/execution_strategy/ocl_grouped_aggregation_multipass.hpp>
#include <query_compilation/execution_strategy/ocl_projection_parallel_global_atomic_single_pass.hpp>
#include <query_compilation/execution_strategy/ocl_projection_serial_single_pass.hpp>
#include <query_compilation/execution_strategy/ocl_projection_single_pass_scan.hpp>
#include <query_compilation/execution_strategy/ocl_projection_three_phase.hpp>
#include <query_compilation/execution_strategy/pipeline.hpp>
#include <query_compilation/pipeline_info.hpp>

#include <util/opencl_runtime.hpp>

#include <query_compilation/code_generators/c_code_compiler.hpp>
#include <query_compilation/query_context.hpp>

namespace CoGaDB {

MultiStageCodeGenerator::MultiStageCodeGenerator(const ProjectionParam& _param,
                                                 const TablePtr table,
                                                 uint32_t version)
    : CCodeGenerator(_param, table, version),
      programPrimitives(),
      programPos(0),
      target(C_TARGET_CODE) {
  init();
}

MultiStageCodeGenerator::MultiStageCodeGenerator(const ProjectionParam& _param)
    : CCodeGenerator(_param),
      programPrimitives(),
      programPos(0),
      target(C_TARGET_CODE) {
  init();
}

void MultiStageCodeGenerator::init() {
  this->generator_type = MULTI_STAGE_CODE_GENERATOR;
}

bool MultiStageCodeGenerator::consumeSelection_impl(
    const PredicateExpressionPtr pred_expr) {
  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }

  programPrimitives.push_back(InstructionPtr(new Filter(pred_expr)));

  std::vector<AttributeReferencePtr> columns_to_decompress =
      pred_expr->getColumnsToDecompress();
  for (std::vector<AttributeReferencePtr>::iterator itr(
           columns_to_decompress.begin());
       itr != columns_to_decompress.end(); ++itr) {
    mColumnsToDecompress[(*itr)->getVersionedAttributeName()] = *itr;
  }

  return true;
}

bool MultiStageCodeGenerator::createForLoop_impl(const TablePtr table,
                                                 uint32_t version) {
  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }

  programPrimitives.push_back(InstructionPtr(new Loop(table, version)));

  return true;
}

bool MultiStageCodeGenerator::consumeBuildHashTable_impl(
    const AttributeReference& attr) {
  /* add code for hash table creation before for loop */
  this->ht_gen = CoGaDB::createHashTableGenerator(attr);
  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  programPrimitives.push_back(InstructionPtr(new HashPut(ht_gen, attr)));
  return true;
}

bool MultiStageCodeGenerator::consumeProbeHashTable_impl(
    const AttributeReference& hash_table_attr,
    const AttributeReference& probe_attr) {
  this->ht_gen = CoGaDB::createHashTableGenerator(hash_table_attr);

  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  programPrimitives.push_back(
      InstructionPtr(new HashProbe(ht_gen, hash_table_attr, probe_attr)));

  return true;
}

/*
const std::string MultiStageCodeGenerator::getAggregationCode(const
GroupingAttributes& grouping_columns, const AggregateSpecifications&
aggregation_specs,
                                                     const std::string
access_ht_entry_expression) const {
    return CCodeGenerator::getAggregationCode(grouping_columns,ggregation_specs,
access_ht_entry_expression);
}
 */
bool MultiStageCodeGenerator::consumeAggregate_impl(
    const AggregateSpecifications& param) {
  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  for (size_t i = 0; i < param.size(); ++i) {
    programPrimitives.push_back(InstructionPtr(new Aggregation(param[i])));
  }
  /* appends an instruction that increments the result size, this is required
   so that the result table has exactly one row => "++current_result_size;" */
  programPrimitives.push_back(
      InstructionPtr(new IncrementResultTupleCounter()));
  return true;
}

bool MultiStageCodeGenerator::consumeCrossJoin_impl(
    const AttributeReference& attr) {
  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  programPrimitives.push_back(InstructionPtr(new CrossJoin(attr)));

  return true;
}

bool MultiStageCodeGenerator::consumeNestedLoopJoin_impl(
    const PredicateExpressionPtr pred_expr) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  return false;
}

void MultiStageCodeGenerator::generateCode_BitpackedGroupingKeyComputation(
    const GroupingAttributes& grouping_attrs) {
  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  programPrimitives.push_back(
      InstructionPtr(new BitPackedGroupingKey(grouping_attrs)));
}

void MultiStageCodeGenerator::generateCode_GenericGroupingKeyComputation(
    const GroupingAttributes& grouping_attrs) {
  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  programPrimitives.push_back(
      InstructionPtr(new GenericGroupingKey(grouping_attrs)));
}

const std::string MultiStageCodeGenerator::getCodeWriteResult() const {
  return generateCCodeWriteResult(param);
}

bool MultiStageCodeGenerator::consumeHashGroupAggregate_impl(
    const GroupByAggregateParam& groupby_param) {
  const AggregateSpecifications& aggr_specs = groupby_param.aggregation_specs;
  const GroupingAttributes& grouping_attrs = groupby_param.grouping_attrs;

  if (isBitpackedGroupbyOptimizationApplicable(grouping_attrs)) {
    generateCode_BitpackedGroupingKeyComputation(grouping_attrs);
  } else {
    generateCode_GenericGroupingKeyComputation(grouping_attrs);
  }

  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }

  programPrimitives.push_back(InstructionPtr(
      new HashGroupAggregate(grouping_attrs, aggr_specs, param)));

  grouped_aggregation = true;

  return true;
}

const std::pair<bool, std::vector<AttributeReferencePtr>>
MultiStageCodeGenerator::consumeMapUDF_impl(const Map_UDF_ParamPtr param) {
  ProjectionParam project_param;
  Map_UDF_Result result =
      param->map_udf->generateCode(scanned_attributes, project_param);

  if (!result.return_code) {
    COGADB_FATAL_ERROR("Failed to generate code for Map UDF type: "
                           << int(param->map_udf->getMap_UDF_Type()),
                       "");
  }

  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  programPrimitives.push_back(InstructionPtr(new MapUDF(result)));

  return std::pair<bool, std::vector<AttributeReferencePtr>>(
      true, result.computed_attributes);
}

const std::pair<bool, AttributeReference>
MultiStageCodeGenerator::consumeAlgebraComputation_impl(
    const AttributeReference& left_attr, const AttributeReference& right_attr,
    const ColumnAlgebraOperation& alg_op) {
  AttributeReference computed_attr =
      createComputedAttribute(left_attr, right_attr, alg_op);

  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  programPrimitives.push_back(InstructionPtr(
      new AttributeAttributeOp(computed_attr, left_attr, right_attr, alg_op)));

  return std::make_pair(true, computed_attr);
}

const std::pair<bool, AttributeReference>
MultiStageCodeGenerator::consumeAlgebraComputation_impl(
    const AttributeReference& left_attr, const boost::any constant,
    const ColumnAlgebraOperation& alg_op) {
  AttributeReference computed_attr =
      createComputedAttribute(left_attr, constant, alg_op);
  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  programPrimitives.push_back(InstructionPtr(
      new AttributeConstantOp(computed_attr, left_attr, constant, alg_op)));

  return std::make_pair(true, computed_attr);
}

const std::pair<bool, AttributeReference>
MultiStageCodeGenerator::consumeAlgebraComputation_impl(
    const boost::any constant, const AttributeReference& right_attr,
    const ColumnAlgebraOperation& alg_op) {
  AttributeReference computed_attr =
      createComputedAttribute(constant, right_attr, alg_op);
  if (programPrimitives.size() == 0) {
    programPos = mUpperCodeBlock.size();
  }
  programPrimitives.push_back(InstructionPtr(
      new ConstantAttributeOp(computed_attr, constant, right_attr, alg_op)));
  return std::make_pair(true, computed_attr);
}

void print(const std::list<InstructionPtr>& programPrimitives) {
  std::list<InstructionPtr>::const_iterator primCit;

  std::vector<std::string> instructions;
  size_t max_length = 0;
  for (primCit = programPrimitives.begin(); primCit != programPrimitives.end();
       ++primCit) {
    std::string str = (*primCit)->toString();
    if (str.size() > max_length) {
      max_length = str.size();
    }
    instructions.push_back(str);
  }
  size_t offset = 3;
  std::cout << "+-" << std::string(max_length + offset, '-') << "-+"
            << std::endl;
  std::cout << "| DSL Program:"
            << std::string(max_length - strlen("DSL Program:") + offset, ' ')
            << " |" << std::endl;
  std::cout << "+=" << std::string(max_length + offset, '=') << "=+"
            << std::endl;
  for (size_t i = 0; i < instructions.size(); ++i) {
    std::cout << "| " << i << ". " << instructions[i]
              << std::string(max_length - instructions[i].size(), ' ') << " |"
              << std::endl;
  }
  std::cout << "+-" << std::string(max_length + offset, '-') << "-+"
            << std::endl;
}

void optimizePrimitivePipelineSSE(std::list<InstructionPtr>& program) {
  // TODO: extract into own OptimizingModule
  std::list<InstructionPtr>::iterator primCit1;

  std::list<InstructionPtr> new_program;

  LoopPtr currentLoop;
  uint32_t version = 0;
  for (primCit1 = program.begin(); primCit1 != program.end(); ++primCit1) {
    if ((*primCit1)->getInstructionType() == FILTER_INSTR) {
      SSEFilterPtr toInsert(
          new SSEFilter(boost::static_pointer_cast<Filter>(*primCit1)
                            ->getPredicateExpression()));
      new_program.push_back(toInsert);
      LoopPtr toInsert2(new ConstantLoop(0, 4, currentLoop));
      new_program.push_back(toInsert2);
      version++;
      SSEMaskFilterPtr maskeval(new SSEMaskFilter(toInsert));
      maskeval->setLoopVar(version);
      new_program.push_back(maskeval);
      currentLoop = toInsert2;
    } else if ((*primCit1)->getInstructionType() == LOOP_INSTR) {
      currentLoop = boost::static_pointer_cast<Loop>(*primCit1);
      // loopSave->setVersion(loopSave->getVersion()+1);
      currentLoop->setRangeDiv(4);
      version++;
      new_program.push_back(*primCit1);
    } else {
      new_program.push_back(*primCit1);
    }
  }
  program = new_program;
  return;
}

void optimizePrimitivePipelinePredicateFilter(
    std::list<InstructionPtr>& programs) {
  bool applied_predication = true;

  for (const auto& instr : programs) {
    InstructionType instr_type = instr->getInstructionType();
    if (instr_type != PRODUCE_TUPLE_INSTR && instr_type != LOOP_INSTR) {
      applied_predication &= instr->supportsPredication();

      if (applied_predication) {
        instr->setPredicationMode(PREDICATED_EXECUTION);
      } else {
        COGADB_WARNING("Could not apply predication to the pipeline because \""
                           << instr->getName() << "\" instruction does not "
                           << "support this.",
                       "");
        break;
      }
    }
  }

  if (!applied_predication) {
    // we could not apply predication to all instruction, so reset to branched
    // execution
    for (const auto& instr : programs) {
      instr->setPredicationMode(BRANCHED_EXECUTION);
    }
  }
}

void optimizePrimitivePipeline(std::list<InstructionPtr>& program) {
  //  if (VariableManager::instance().getVariableValueBoolean(
  //          "multi_staged.c_code.enable_vectorization_optimizer")) {
  //    optimizePrimitivePipelineSSE(program);
  //  }

  if (VariableManager::instance().getVariableValueBoolean(
          "code_gen.opt.enable_predication")) {
    optimizePrimitivePipelinePredicateFilter(program);
  }
}

void MultiStageCodeGenerator::printCode(std::ostream&) {
  COGADB_NOT_IMPLEMENTED;
}

const PipelinePtr MultiStageCodeGenerator::skipCompilationOfEmptyPipeline() {
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");

  if (debug_code_generator) {
    std::cout << "[Falcon]: Omit compilation of empty pipeline..." << std::endl;
  }
  PipelineInfoPtr pipe_info = boost::make_shared<PipelineInfo>();
  pipe_info->setSourceTable(input_table);
  pipe_info->setPipelineType(pipe_end);
  pipe_info->setGroupByAggregateParam(groupby_param);

  return PipelinePtr(
      new DummyPipeline(input_table, scanned_attributes, pipe_info));
}

void MultiStageCodeGenerator::setCodeGeneratorTarget() {
  const std::string strategy =
      VariableManager::instance().getVariableValueString(
          "code_gen.exec_strategy");

  if (strategy == "c") {
    target = C_TARGET_CODE;
  } else if (strategy == "opencl") {
    target = OCL_TARGET_CODE;
  } else {
    COGADB_FATAL_ERROR("Unknown pipeline execution strategy: " << strategy, "");
  }
}

void MultiStageCodeGenerator::createExecutionStrategyAndDeviceType(
    ExecutionStrategy::PipelinePtr& exec_strat, cl_device_type& device_type,
    cl_device_id& device_id) {
  switch (target) {
    case C_TARGET_CODE: {
      exec_strat = boost::make_shared<ExecutionStrategy::PlainC>();
      break;
    }
    case OCL_TARGET_CODE: {
      MemoryAccessPattern memory_access_pattern;

      device_id = getOpenCLGlobalDevice();

      const std::string mem_access =
          VariableManager::instance().getVariableValueString(
              "code_gen.memory_access");
      if (mem_access == "sequential") {
        memory_access_pattern = BLOCK_MEMORY_ACCESS;
      } else if (mem_access == "coalesced") {
        memory_access_pattern = COALESCED_MEMORY_ACCESS;
      } else {
        COGADB_FATAL_ERROR("", "");
      }

      if (pipe_end == MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
        if (grouped_aggregation) {
          auto grouped_strategy =
              VariableManager::instance().getVariableValueString(
                  "code_gen.opt.ocl_grouped_aggregation_strategy");

          if (grouped_strategy == "atomic") {
            exec_strat = boost::make_shared<
                ExecutionStrategy::OCLGroupedAggregationAtomic>(
                true, memory_access_pattern, device_id);
          } else if (grouped_strategy == "semaphore") {
            exec_strat =
                boost::make_shared<ExecutionStrategy::OCLGroupedAggregation>(
                    true, memory_access_pattern, device_id);
          } else if (grouped_strategy == "sequential") {
            exec_strat =
                boost::make_shared<ExecutionStrategy::OCLGroupedAggregation>(
                    true, memory_access_pattern, device_id);
          } else if (grouped_strategy == "atomic_workgroup") {
            exec_strat = boost::make_shared<
                ExecutionStrategy::OCLGroupedAggregationAtomicWorkGroupHT>(
                true, memory_access_pattern, device_id);
          } else if (grouped_strategy == "reduce_atomics") {
            exec_strat = boost::make_shared<
                ExecutionStrategy::OCLGroupedAggregationLocalReduceAtomics>(
                true, memory_access_pattern, device_id);
          } else if (grouped_strategy == "multipass") {
            exec_strat = boost::make_shared<
                ExecutionStrategy::OCLGroupedAggregationMultipass>(
                true, memory_access_pattern, device_id);
          } else if (grouped_strategy == "fused_scan_global_group") {
            exec_strat = boost::make_shared<
                ExecutionStrategy::OCLGroupedAggregationFusedScanGlobalGroup>(
                true, memory_access_pattern, device_id);
          } else {
            COGADB_FATAL_ERROR(
                "Unknown strategy \"" << grouped_strategy << "\"!", "");
          }
        } else {
          std::string val = VariableManager::instance().getVariableValueString(
              "code_gen.aggregation_exec_strategy");

          if (val == "single_pass_reduce") {
            exec_strat = boost::make_shared<
                ExecutionStrategy::OCLAggregationSinglePassReduce>(
                true, memory_access_pattern, device_id);
          } else if (val == "global_reduce_kernel") {
            exec_strat = boost::make_shared<ExecutionStrategy::OCLAggregation>(
                true, memory_access_pattern, device_id);
          } else if (val == "multipass") {
            exec_strat =
                boost::make_shared<ExecutionStrategy::OCLAggregationMultipass>(
                    true, memory_access_pattern, device_id);
          } else {
            COGADB_FATAL_ERROR("Unknown strategy \"" << val << "\"!", "");
          }
        }
      } else {
        std::string val = VariableManager::instance().getVariableValueString(
            "code_gen.pipe_exec_strategy");
        if (val == "parallel_three_pass") {
          exec_strat =
              boost::make_shared<ExecutionStrategy::OCLProjectionThreePhase>(
                  true, memory_access_pattern, device_id);
        } else if (val == "serial_single_pass") {
          exec_strat = boost::make_shared<
              ExecutionStrategy::OCLProjectionSerialSinglePass>(
              true, memory_access_pattern, device_id);
        } else if (val == "parallel_global_atomic_single_pass") {
          exec_strat = boost::make_shared<
              ExecutionStrategy::OCLProjectionParallelGlobalAtomicSinglePass>(
              true, memory_access_pattern, device_id);
        } else if (val == "single_pass_scan") {
          exec_strat = boost::make_shared<
              ExecutionStrategy::OCLProjectionSinglePassScan>(
              true, memory_access_pattern, device_id);
        } else {
          COGADB_FATAL_ERROR(
              "Unknown code_gen.pipe_exec_strategy \"" << val << "\"!", "");
        }
      }

      break;
    }
    default: {
      COGADB_FATAL_ERROR(
          "Should never reach this point because we already tested the "
          "strategy",
          "");
    }
  }
}

void MultiStageCodeGenerator::generateCode(
    std::string& host_source_code, std::string& kernel_source_code,
    ExecutionStrategy::PipelinePtr exec_strat) {
  std::list<InstructionPtr> program(programPrimitives);
  program.push_front(boost::make_shared<ProduceTuples>(
      scanned_attributes, param, mColumnsToDecompress));

  if (pipe_end != MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY) {
    program.push_back(boost::make_shared<Materialization>(param));
  }

  optimizePrimitivePipeline(program);

  CoGaDB::print(program);

  std::list<InstructionPtr>::const_iterator primCit;
  for (primCit = program.begin(); primCit != program.end(); ++primCit) {
    exec_strat->addInstruction(*primCit);
  }

  std::pair<std::string, std::string> code =
      exec_strat->getCode(param, scanned_attributes, pipe_end,
                          input_table->getName(), mColumnsToDecompress);

  host_source_code = code.first;
  kernel_source_code = code.second;
}

const PipelinePtr MultiStageCodeGenerator::compile() {
  setCodeGeneratorTarget();
  ExecutionStrategy::PipelinePtr exec_strat;
  cl_device_type device_type;
  cl_device_id device_id;
  createExecutionStrategyAndDeviceType(exec_strat, device_type, device_id);
  return compileOneVariant(exec_strat, device_type, device_id);
}

CodeGenerationTarget MultiStageCodeGenerator::getTarget() const {
  return target;
}

const PipelinePtr MultiStageCodeGenerator::compileOneVariant(
    const ExecutionStrategy::PipelinePtr exec_strat,
    cl_device_type& device_type, cl_device_id& device_id) {
  if (canOmitCompilation()) {
    return skipCompilationOfEmptyPipeline();
  }

  std::string host_source_code;
  std::string kernel_source_code;

  generateCode(host_source_code, kernel_source_code, exec_strat);

  bool show_generated_code =
      VariableManager::instance().getVariableValueBoolean(
          "show_generated_code");
  bool show_generated_kernel_code =
      VariableManager::instance().getVariableValueBoolean(
          "show_generated_kernel_code");
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");
  if (debug_code_generator || show_generated_code ||
      show_generated_kernel_code) {
    std::cout << std::string(80, '*') << std::endl;
    std::cout << "<<< Generated Kernel Code: " << std::endl;
    pretty_print_code(kernel_source_code);
    std::cout << ">>> Generated Kernel Code" << std::endl;
    std::cout << std::string(80, '*') << std::endl;
  }

  PipelineInfoPtr pipe_info = boost::make_shared<PipelineInfo>();
  pipe_info->setSourceTable(input_table);
  pipe_info->setPipelineType(pipe_end);
  pipe_info->setGroupByAggregateParam(groupby_param);

  CCodeCompiler compiler;
  auto compiledc = compiler.compile(host_source_code);
  if (VariableManager::instance().getVariableValueBoolean(
          "code_gen.opt.enable_profiling")) {
    std::cout << "Compile Time (Host): " << compiledc->getCompileTimeInSeconds()
              << "s" << std::endl;
  }

  if (target == C_TARGET_CODE) {
    return boost::make_shared<CPipeline>(scanned_attributes, pipe_info,
                                         compiledc);
  } else if (target == OCL_TARGET_CODE) {
    std::pair<OCL_Execution_ContextPtr, double> compiled_exec_context =
        OCL_Runtime::instance().compileDeviceKernel(device_id,
                                                    kernel_source_code);

    OCL_Execution_ContextPtr execution_context = compiled_exec_context.first;
    assert(execution_context != nullptr);
    if (VariableManager::instance().getVariableValueBoolean(
            "code_gen.opt.enable_profiling")) {
      std::cout << "Compile Time (Kernel): " << compiled_exec_context.second
                << "s" << std::endl;
    }

    return boost::make_shared<OCLPipeline>(
        scanned_attributes, pipe_info, compiledc, compiled_exec_context.second,
        execution_context);
  } else {
    COGADB_FATAL_ERROR("Unknown Target!", "");
  }
}

}  // end namespace CoGaDB
