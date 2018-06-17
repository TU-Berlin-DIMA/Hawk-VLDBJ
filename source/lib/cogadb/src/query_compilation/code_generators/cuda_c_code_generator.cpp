
#include <boost/filesystem/operations.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>
#include <ctime>

#include <util/code_generation.hpp>
#include <util/functions.hpp>
#include <util/time_measurement.hpp>

#include <core/variable_manager.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/cuda_c_code_generator.hpp>
#include <query_compilation/pipeline_info.hpp>
#include <util/shared_library.hpp>

namespace CoGaDB {

const std::string getIndexVarType() { return "uint32_t"; }

const std::string getDeviceInputArrayVarName(
    const AttributeReference& attr_ref) {
  std::stringstream ss;
  ss << "d_" << getInputArrayVarName(attr_ref);
  return ss.str();
}

const std::string getDeviceResultArrayVarName(
    const AttributeReference& attr_ref) {
  std::stringstream ss;
  ss << "d_" << getResultArrayVarName(attr_ref);
  return ss.str();
}

CUDA_C_CodeGenerator::CUDA_C_CodeGenerator(const ProjectionParam& param,
                                           const TablePtr table,
                                           uint32_t version)
    : CPPCodeGenerator(param, table, version) {
  init();
}

CUDA_C_CodeGenerator::CUDA_C_CodeGenerator(const ProjectionParam& param)
    : CPPCodeGenerator(param) {
  init();
}

void CUDA_C_CodeGenerator::init() {
  this->producedProjection = false;
  this->codeIsProduced = false;
  this->generator_type = CUDA_C_CODE_GENERATOR;

  cuda_includes << "#include <cuda_runtime.h>" << std::endl;
  cuda_includes << "#include <cub/cub.cuh>" << std::endl;
  cuda_includes << "#include <vector>" << std::endl;
  cuda_includes << "#include <query_compilation/gpu_utilities/util.cuh>"
                << std::endl;
  cuda_includes << "#include <thrust/scan.h>" << std::endl;
  cuda_includes << "#include <thrust/device_ptr.h>" << std::endl;
  cuda_includes << "#include "
                   "<query_compilation/gpu_utilities/compilation_hashtables/"
                   "GPU_CHT32_2.cuh>";

  blockSize = 128;
  constant_declarations << "const int kBlockSize = " << blockSize << ";"
                        << std::endl;
}

void CUDA_C_CodeGenerator::produceAllocArray(
    std::string datatype, std::string varname, size_t num_elements,
    MemoryLocation location, CodePosition code_pos, bool doFree) {
  this->produceAllocArray(datatype, varname,
                          boost::lexical_cast<std::string>(num_elements),
                          location, code_pos, doFree);
}

void CUDA_C_CodeGenerator::produceAllocArray(
    std::string datatype, std::string varname, std::string num_elements_varname,
    MemoryLocation location, CodePosition code_pos, bool doFree) {
  std::stringstream allocCode;
  std::stringstream freeCode;
  allocCode << datatype << "* " << varname << ";";
  if (location == DEVICE) {
    allocCode << "cudaMalloc((void**)&" << varname << ", sizeof(" << datatype
              << ") * " << num_elements_varname << ");";
    if (doFree) freeCode << "cudaFree(" << varname << ");";
  } else if (location == HOST) {
    allocCode << varname << " = (" << datatype << "*)"
              << "malloc(" << num_elements_varname << "* sizeof(" << datatype
              << "));";
    if (doFree) freeCode << "free(" << varname << ");";
  }
  switch (code_pos) {
    case PRE_COMPUTE:
      pre_compute_allocation_code << allocCode.str() << std::endl;
      break;
    case POST_SCAN:
      post_scan_allocation_code << allocCode.str() << std::endl;
      break;
    case POST_PROJECTION:
      post_projection_allocation_code << allocCode.str() << std::endl;
      break;
  }
  free_memory_code << freeCode.str() << std::endl;
}

void CUDA_C_CodeGenerator::produceDataTransfer(
    std::string datatype, std::string name_dest, std::string name_src,
    size_t num_elements, MemcpyDirection direction, CodePosition code_pos) {
  this->produceDataTransfer(datatype, name_dest, name_src,
                            boost::lexical_cast<std::string>(num_elements),
                            direction, code_pos);
}

void CUDA_C_CodeGenerator::produceDataTransfer(std::string datatype,
                                               std::string name_dest,
                                               std::string name_src,
                                               std::string num_elements_varname,
                                               MemcpyDirection direction,
                                               CodePosition code_pos) {
  std::stringstream transferCode;
  transferCode << "cudaMemcpy(" << name_dest << ", " << name_src << ", "
               << num_elements_varname << "* sizeof(" << datatype << "), ";

  if (direction == H2D)
    transferCode << "cudaMemcpyHostToDevice";
  else if (direction == D2H)
    transferCode << "cudaMemcpyDeviceToHost";

  transferCode << ");" << std::endl;

  switch (code_pos) {
    case PRE_COMPUTE:
      pre_compute_transfer_code << transferCode.str();
      break;
    case POST_SCAN:
      post_scan_transfer_code << transferCode.str();
      break;
    case POST_PROJECTION:
      post_projection_transfer_code << transferCode.str();
      break;
  }
}

void CUDA_C_CodeGenerator::generateCodeComputeStructure() {
  std::stringstream initComputeStructureVars;

  // declare variables
  initComputeStructureVars << "int numKernelBlocks;" << std::endl
                           << "int numKernelThreads;" << std::endl
                           << "size_t current_result_size;" << std::endl;

  // compute number of blocks
  initComputeStructureVars << "dim3 grid( (" << input_table->getNumberofRows()
                           << " + kBlockSize-1) / "
                           << "kBlockSize);" << std::endl
                           << "if (grid.x > 16384) {" << std::endl
                           << "grid.y = (grid.x + 16384 - 1) / 16384;"
                           << std::endl
                           << "grid.x = 16384;" << std::endl
                           << "}" << std::endl
                           << "if(grid.x == 0) grid.x = 1;" << std::endl
                           << "numKernelBlocks = grid.x * grid.y;" << std::endl
                           << "numKernelThreads = numKernelBlocks * kBlockSize;"
                           << std::endl;

  // output compute structure info
  initComputeStructureVars
      << "std::cout << \"numKernelThreads: \" << numKernelThreads << std::endl;"
      << std::endl
      << "std::cout << \"numKernelBlocks: \" << numKernelBlocks << std::endl;"
      << std::endl
      << "std::cout << \"grid.x: \" << grid.x << std::endl;" << std::endl
      << "std::cout << \"grid.y: \" << grid.y << std::endl;" << std::endl
      << "std::cout << \"kBlockSize: \" << kBlockSize << std::endl;"
      << std::endl;

  compute_structure_code = initComputeStructureVars.str();

  produceAllocArray(getIndexVarType(), "d_writePositions", "numKernelThreads",
                    DEVICE, PRE_COMPUTE);

  produceAllocArray("uint32_t", "d_flagArray", "numKernelThreads", DEVICE,
                    PRE_COMPUTE);

  produceAllocArray("size_t", "d_blockCounts", "numKernelBlocks", DEVICE,
                    PRE_COMPUTE);

  produceAllocArray("size_t", "d_scanBlockCounts", "numKernelBlocks", DEVICE,
                    PRE_COMPUTE);
}

bool CUDA_C_CodeGenerator::produceTuples(const ScanParam& param) {
  if (this->tuples_produced) return true;

  std::cout << "CUDA_C_CodeGenerator::produceTuples(..)" << std::endl;

  CPPCodeGenerator::produceTuples(param);

  ScanParam::const_iterator cit;
  for (cit = param.begin(); cit != param.end(); ++cit) {
    AttributeType attr = cit->getAttributeType();

    produceAllocArray(toCPPType(attr), getDeviceInputArrayVarName(*cit),
                      cit->getTable()->getNumberofRows(), DEVICE, PRE_COMPUTE);

    produceDataTransfer(toCPPType(attr), getDeviceInputArrayVarName(*cit),
                        getInputArrayVarName(*cit),
                        cit->getTable()->getNumberofRows(), H2D, PRE_COMPUTE);
  }

  return true;
}

// allocate result with size specified by variable given by its name
void CUDA_C_CodeGenerator::produceAllocateDeviceResult(
    std::string numElementsVarName) {
  for (size_t i = 0; i < param.size(); ++i) {
    AttributeType attr = param[i].getAttributeType();
    produceAllocArray(toCPPType(attr), getDeviceResultArrayVarName(param[i]),
                      numElementsVarName, DEVICE, POST_SCAN);
  }
}

// initialize kernels
bool CUDA_C_CodeGenerator::createForLoop_impl(const TablePtr table,
                                              uint32_t version) {
  std::cout << "CUDA_C_CodeGenerator::createForLoop_impl(..)" << std::endl;

  scan_pipeline_kernel.init("scanPipelineKernel", blockSize, getIndexVarType(),
                            getTupleIDVarName(input_table, input_table_version),
                            input_table->getNumberofRows(), true);

  project_pipeline_kernel.init(
      "projectPipelineKernel", blockSize, getIndexVarType(),
      getTupleIDVarName(input_table, input_table_version), 0, false);

  // add scanned attributes to scan pipeline kernel
  for (auto i = 0u; i < scanned_attributes.size(); i++) {
    scan_pipeline_kernel.addParameter(
        toCPPType(scanned_attributes[i].getAttributeType()) + "*",
        getVarName(scanned_attributes[i]));
  }
  // compute structure
  scan_pipeline_kernel.addParameter(getIndexVarType() + "*", "writePositions");
  scan_pipeline_kernel.addParameter("size_t*", "blockCounts");
  scan_pipeline_kernel.addParameter("uint32_t*", "flagArray");

  // add projected attributes to project pipeline kernel
  for (size_t i = 0; i < param.size(); ++i) {
    AttributeType attr = param[i].getAttributeType();
    project_pipeline_kernel.addParameter(toCPPType(attr) + "* ",
                                         getVarName(param[i]));

    project_pipeline_kernel.addParameter(toCPPType(attr) + "* ",
                                         getResultArrayVarName(param[i]));
  }
  // compute structure
  project_pipeline_kernel.addParameter(getIndexVarType() + "*",
                                       "writePositions");
  project_pipeline_kernel.addParameter("size_t*", "scanBlockCounts");
  project_pipeline_kernel.addParameter("uint32_t*", "flagArray");

  this->generateCodeComputeStructure();

  return true;
}

bool CUDA_C_CodeGenerator::consumeSelection_impl(
    const PredicateExpressionPtr pred_expr) {
  std::cout << "CUDA_C_CodeGenerator::consumeSelection_impl(..)" << std::endl;

  // produce predicate evaluation
  std::stringstream predicateCode;
  predicateCode << "selected = selected && (" << pred_expr->getCPPExpression()
                << ");" << std::endl;
  scan_pipeline_kernel.append_upper(predicateCode.str());

  return true;
}

bool CUDA_C_CodeGenerator::consumeBuildHashTable_impl(
    const AttributeReference& attr) {
  // allocate hashtable
  std::stringstream codeAllocHt;
  codeAllocHt << "GPU_CHT32_2::HashTable* " << getHashTableVarName(attr)
              << " = GPU_CHT32_2::createHashTable("
              << "current_result_size*2, 2.1);" << std::endl;
  allocate_hashtables_code << codeAllocHt.str();

  // add hashtable as kernel parameter
  project_pipeline_kernel.addParameter("GPU_CHT32_2::HashTable",
                                       getHashTableVarName(attr),
                                       "*" + getHashTableVarName(attr));

  // insert element (put key and row index)
  std::stringstream codeInsertKey;
  codeInsertKey << "GPU_CHT32_2::insertEntry(" << getHashTableVarName(attr)
                << ", "
                << "(uint32_t)" << getInputArrayVarName(attr)
                << "[thread_index], "
                << "writePositions[thread_index]);" << std::endl;
  make_hashtable_entries_code << codeInsertKey.str();

  // add hashtable to result
  this->create_result_table_code_block
      << "if(!addHashTable(result_table, \"" << attr.getResultAttributeName()
      << "\", "
      << "createSystemHashTable(" << getHashTableVarName(attr) << "))){"
      << "std::cout << \"Error adding hash table for attribute '"
      << attr.getResultAttributeName() << "' to result!\" << std::endl;"
      << "return TablePtr();"
      << "}" << std::endl;

  this->create_result_table_code_block << "printHashTable(*"
                                       << getHashTableVarName(attr) << ", 20);"
                                       << std::endl;

  return true;
}

bool CUDA_C_CodeGenerator::consumeProbeHashTable_impl(
    const AttributeReference& hash_table_attr,
    const AttributeReference& probe_attr) {
  std::stringstream codeGetHt;

  // get hash table
  codeGetHt << "HashTablePtr generic_hashtable_"
            << hash_table_attr.getVersionedAttributeName() << " = getHashTable("
            << getTableVarName(hash_table_attr) << ", "
            << "\"" << hash_table_attr.getUnversionedAttributeName() << "\");"
            << std::endl;

  codeGetHt << "GPU_CHT32_2::HashTable* "
            << getHashTableVarName(hash_table_attr)
            << " = (GPU_CHT32_2::HashTable*) "
               "getHashTableFromSystemHashTable(generic_hashtable_"
            << hash_table_attr.getVersionedAttributeName() << ");" << std::endl;

  get_hashtables_code << codeGetHt.str();

  // add hashtable as kernel parameter
  scan_pipeline_kernel.addParameter("GPU_CHT32_2::HashTable",
                                    getHashTableVarName(hash_table_attr),
                                    "*" + getHashTableVarName(hash_table_attr));

  project_pipeline_kernel.addParameter(
      "GPU_CHT32_2::HashTable", getHashTableVarName(hash_table_attr),
      "*" + getHashTableVarName(hash_table_attr));

  // probe element in kernels
  std::stringstream codeProbeHt;

  codeProbeHt << "uint32_t " << getTupleIDVarName(hash_table_attr) << ";"
              << std::endl;

  codeProbeHt << getTupleIDVarName(hash_table_attr)
              << " = GPU_CHT32_2::probeKey("
              << getHashTableVarName(hash_table_attr) << ", "
              << getVarName(probe_attr) << "[" << getTupleIDVarName(probe_attr)
              << "]"
              << ");" << std::endl;
  codeProbeHt << "selected = selected && ("
              << getTupleIDVarName(hash_table_attr)
              << " != GPU_CHT32_2::kEmpty);" << std::endl;
  scan_pipeline_kernel.append_upper(codeProbeHt.str());

  codeProbeHt << "uint32_t " << getTupleIDVarName(hash_table_attr) << ";"
              << std::endl;
  get_hashtable_values_code
      << "uint32_t " << getTupleIDVarName(hash_table_attr)
      << " = GPU_CHT32_2::probeKey(" << getHashTableVarName(hash_table_attr)
      << ", " << getVarName(probe_attr) << "[" << getTupleIDVarName(probe_attr)
      << "]"
      << ");" << std::endl;

  free_memory_code << "freeHashTable(" << getHashTableVarName(hash_table_attr)
                   << ");" << std::endl;

  return true;
}

// produces a projection of all specified attributes after selection using
// prefix sum
void CUDA_C_CodeGenerator::produceProjection() {
  if (this->producedProjection == true) return;

  scan_pipeline_kernel.append_lower("flagArray[thread_index] = selected;");
  this->produceScanFlagArray();

  std::stringstream projectElements;
  projectElements << "if(flagArray[thread_index]) {";
  projectElements << get_hashtable_values_code.str();
  projectElements << this->getCodeDenseWriteProjectionElements();
  projectElements << make_hashtable_entries_code.str();
  projectElements << "}";
  project_pipeline_kernel.append_lower(projectElements.str());

  this->producedProjection = true;
}

// global prefix sum of flagarray provides write positions
void CUDA_C_CodeGenerator::produceScanFlagArray() {
  // produce compute structure scan
  // didnt work for unknown reason
  // scanStructureCode
  //   << "void     *d_temp_storage = NULL;" << std::endl
  //   << "size_t   temp_storage_bytes = 0;" << std::endl
  //   << "cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
  //   d_flagArray, d_writePositions, numKernelThreads);" << std::endl
  //   << "cudaMalloc(&d_temp_storage, temp_storage_bytes);" << std::endl
  //   << "cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
  //   d_flagArray, d_writePositions, numKernelThreads);" << std::endl
  //   << "cudaFree(d_temp_storage);" << std::endl;
  std::stringstream scanStructureCode;
  scanStructureCode
      << "thrust::exclusive_scan("
      << "thrust::device_pointer_cast(d_flagArray), "
      << "thrust::device_pointer_cast(d_flagArray + numKernelThreads), "
      << "thrust::device_pointer_cast(d_writePositions));" << std::endl;
  // get final result size
  scanStructureCode << "uint32_t h_flagN;" << std::endl
                    << "uint32_t h_writePosN;" << std::endl
                    << "cudaMemcpy(&h_flagN, "
                       "&(d_flagArray[numKernelThreads-1]), sizeof(uint32_t), "
                       "cudaMemcpyDeviceToHost);"
                    << std::endl
                    << "cudaMemcpy(&h_writePosN, "
                       "&(d_writePositions[numKernelThreads-1]), "
                       "sizeof(uint32_t), cudaMemcpyDeviceToHost);"
                    << std::endl
                    << "current_result_size = h_flagN + h_writePosN;"
                    << std::endl;
  scanStructureCode << "std::cout << \"Final result size is:\" << "
                       "current_result_size << std::endl;";
  scan_compute_structure = scanStructureCode.str();
}

// dense write projected attributes based on write positions
const std::string CUDA_C_CodeGenerator::getCodeDenseWriteProjectionElements() {
  std::stringstream projection;
  for (size_t i = 0; i < param.size(); ++i) {
    projection << getResultArrayVarName(param[i])
               << "[writePositions[thread_index]]"
               << " = " << getVarName(param[i]) << "["
               << getTupleIDVarName(param[i]) << "];" << std::endl;
  }
  return projection.str();
}

// block wide scan
void CUDA_C_CodeGenerator::produceBlockScanSelected(KernelExecution& kernel) {
  std::stringstream blockScanInit;
  blockScanInit << "typedef cub::BlockScan<int, kBlockSize> BlockScan;"
                << std::endl
                << "__shared__ typename BlockScan::TempStorage temp_storage;"
                << std::endl
                << "int flag = 0;" << std::endl
                << "int prefixSum = 0;" << std::endl
                << "int blockAggregate = 0;" << std::endl;

  kernel.prepend_upper(blockScanInit.str());

  // produce block scan
  std::stringstream blockScanCode;
  blockScanCode << "if(selected) flag = 1;" << std::endl
                << "BlockScan(temp_storage).ExclusiveSum("
                << "flag, prefixSum);" << std::endl;

  blockScanCode << "writePositions[thread_index] = prefixSum;" << std::endl
                << "if(threadIdx.x == blockDim.x-1) " << std::endl
                << "blockCounts[block_index] = "
                << "prefixSum + flag;" << std::endl;

  kernel.append_lower(blockScanCode.str());
}

// global prefix sum on the result size of each block
void CUDA_C_CodeGenerator::produceScanBlockResultSizes() {
  std::stringstream scanStructureCode;
  scanStructureCode << "void     *d_temp_storage = NULL;" << std::endl
                    << "size_t   temp_storage_bytes = 0;" << std::endl
                    << "cub::DeviceScan::ExclusiveSum(d_temp_storage, "
                       "temp_storage_bytes, d_blockCounts, d_scanBlockCounts, "
                       "numKernelBlocks);"
                    << std::endl
                    << "cudaMalloc(&d_temp_storage, temp_storage_bytes);"
                    << std::endl
                    << "cub::DeviceScan::ExclusiveSum(d_temp_storage, "
                       "temp_storage_bytes, d_blockCounts, d_scanBlockCounts, "
                       "numKernelBlocks);"
                    << std::endl
                    << "cudaFree(d_temp_storage);" << std::endl;

  scanStructureCode
      << "std::vector<size_t> scannedBlockCounts(numKernelBlocks);" << std::endl
      << "cudaMemcpy(&scannedBlockCounts[0], d_scanBlockCounts, sizeof(size_t) "
         "* numKernelBlocks, cudaMemcpyDeviceToHost);"
      << std::endl
      << "for(int i=0;i<numKernelBlocks;i++) std::cout << "
         "scannedBlockCounts[i] << \", \";"
      << std::endl;
  scan_compute_structure = scanStructureCode.str();
}

// allocate (host) result columns with final size
void CUDA_C_CodeGenerator::produceAllocateResultTable() {
  for (size_t i = 0; i < param.size(); ++i) {
    AttributeType attr = param[i].getAttributeType();
    produceAllocArray(toCPPType(attr), getResultArrayVarName(param[i]),
                      "current_result_size", HOST, POST_PROJECTION, false);
  }
}

// copy final result to host
void CUDA_C_CodeGenerator::produceCopyDeviceResult() {
  for (size_t i = 0; i < param.size(); ++i) {
    AttributeType attr = param[i].getAttributeType();
    produceDataTransfer(toCPPType(attr), getResultArrayVarName(param[i]),
                        getDeviceResultArrayVarName(param[i]),
                        "current_result_size", D2H, POST_PROJECTION);
  }
}

void CUDA_C_CodeGenerator::produceCode() {
  // ----- produce the rest of the code ----
  bool ret = this->produceTuples(this->scanned_attributes);
  if (!ret) COGADB_FATAL_ERROR("Produce Tuples Failed!", "");
  this->produceProjection();

  // ----- changed mode from getcode to produce
  if (!codeIsProduced) {
    this->produceAllocateDeviceResult("current_result_size");
    this->produceAllocateResultTable();
    this->produceCopyDeviceResult();

    this->pre_compute_allocation_code << "gpuErrchk(cudaGetLastError());"
                                      << std::endl;
    this->pre_compute_transfer_code << "gpuErrchk(cudaGetLastError());"
                                    << std::endl;

    if (this->post_scan_allocation_code.str().size() > 0) {
      this->post_scan_allocation_code << "gpuErrchk(cudaGetLastError());"
                                      << std::endl;
    }
    if (this->post_scan_transfer_code.str().size() > 0) {
      this->post_scan_transfer_code << "gpuErrchk(cudaGetLastError());"
                                    << std::endl;
    }

    this->post_projection_allocation_code << "gpuErrchk(cudaGetLastError());"
                                          << std::endl;
    this->post_projection_transfer_code << "gpuErrchk(cudaGetLastError());"
                                        << std::endl;
    this->free_memory_code << "gpuErrchk(cudaGetLastError());" << std::endl;

    codeIsProduced = true;
  }
}

// produce and print cde to out
void CUDA_C_CodeGenerator::printCode(std::ostream& out) {
  produceCode();

  /* all imports and declarations */
  out << this->cuda_includes.str() << std::endl;
  out << this->header_and_types_block.str() << std::endl;
  /* declare and initialize constants */
  out << this->constant_declarations.str() << std::endl;

  // ---------------------- KERNELS --------------------------
  out << this->scan_pipeline_kernel.getKernelCode() << std::endl;
  out << this->project_pipeline_kernel.getKernelCode() << std::endl;

  // ------------------- FUNCTION START: INPUT -----------------------
  //****** prepare input and intermediate data *******

  // inherited: includes function header
  out << this->fetch_input_code_block.str() << std::endl;
  out << this->compute_structure_code << std::endl;

  out << this->pre_compute_allocation_code.str() << std::endl;
  out << this->pre_compute_transfer_code.str() << std::endl;

  out << this->get_hashtables_code.str() << std::endl;

  // ----------------------------- COMPUTE ---------------------------------
  //****** start computation with kernel that counts result sizes *******
  out << this->scan_pipeline_kernel.getInvocationCode() << std::endl;

  //***** scan sizes to get memory and write positions ******
  out << this->scan_compute_structure << std::endl;

  out << this->post_scan_allocation_code.str();
  out << this->post_scan_transfer_code.str() << std::endl;

  out << this->allocate_hashtables_code.str() << std::endl;

  // ****** invoke kernel that writes full dense result to device memory ****
  out << this->project_pipeline_kernel.getInvocationCode() << std::endl;

  // ----------------------------- OUTPUT ---------------------------------
  out << this->post_projection_allocation_code.str() << std::endl;
  out << this->post_projection_transfer_code.str() << std::endl;
  out << this->free_memory_code.str() << std::endl;

  /* generate code that builds the reslt table using the minimal API */
  out << this->createResultTable() << std::endl;

  out << "}" << std::endl;
}

const PipelinePtr CUDA_C_CodeGenerator::compile() {
  ScanParam& param = scanned_attributes;

  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");

  PipelineInfoPtr pipe_info(new PipelineInfo());
  pipe_info->setSourceTable(this->input_table);
  pipe_info->setPipelineType(this->pipe_end);
  pipe_info->setGroupByAggregateParam(this->groupby_param);

  if (canOmitCompilation()) {
    if (debug_code_generator)
      std::cout << "[Falcon]: Omit compilation of empty pipeline..."
                << std::endl;
    return PipelinePtr(
        new DummyPipeline(this->input_table, scanned_attributes, pipe_info));
  }

  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  std::stringstream ss;
  ss << "gen_query_" << uuid;

  std::string filename = ss.str() + ".cu";

  std::ofstream generated_file(filename.c_str(),
                               std::ios::trunc | std::ios::out);
  printCode(generated_file);
  generated_file.close();

  std::string format_command = std::string("astyle ") + filename;
  int ret = system(format_command.c_str());

  std::string cat_command = std::string("cat ") + filename;
  ret = system(cat_command.c_str());

  std::string copy_last_query_command =
      std::string("cp '") + filename + std::string("' last_generated_query.cu");
  ret = system(copy_last_query_command.c_str());

  Timestamp begin_compile = getTimestamp();
  std::string path_to_precompiled_header = "minimal_api.hpp.pch";
  std::string path_to_minimal_api_header =
      std::string(PATH_TO_COGADB_SOURCE_CODE) +
      "/include/query_compilation/minimal_api.hpp";

  bool rebuild_precompiled_header = false;

  if (!boost::filesystem::exists(path_to_precompiled_header)) {
    rebuild_precompiled_header = true;
  } else {
    std::time_t last_access_pch =
        boost::filesystem::last_write_time(path_to_precompiled_header);
    std::time_t last_access_header =
        boost::filesystem::last_write_time(path_to_minimal_api_header);
    /* pre-compiled header outdated? */
    if (last_access_header > last_access_pch) {
      std::cout << "Pre-compiled header '" << path_to_precompiled_header
                << "' is outdated!" << std::endl;
      rebuild_precompiled_header = true;
    }
  }

  if (rebuild_precompiled_header) {
    std::cout
        << "Precompiled Header not found! Building Precompiled Header now..."
        << std::endl;
    std::stringstream precompile_header;

    precompile_header << QUERY_COMPILATION_CXX << " -g -O3 -fpic "
                      << PATH_TO_COGADB_SOURCE_CODE
                      << "/include/query_compilation/minimal_api.hpp -I "
                      << PATH_TO_COGADB_SOURCE_CODE
                      << "/include/  -o minimal_api.hpp.pch" << std::endl;
    ret = system(precompile_header.str().c_str());
    if (ret != 0) {
      std::cout << "Compilation of precompiled header failed!" << std::endl;
      return PipelinePtr();
    } else {
      std::cout << "Compilation of precompiled header successful!" << std::endl;
    }
  }

  std::stringstream compile_command;

  compile_command << "nvcc -g -O3 -I " << PATH_TO_COGADB_SOURCE_CODE
                  << "/include/ -I " << PATH_TO_COGADB_SOURCE_CODE
                  << "/../external_libraries/cub-1.4.1/ -I "
                  << PATH_TO_COGADB_SOURCE_CODE
                  << "/../hype-library/include/ -c --compiler-options -fpic ";

  // compile_command << "nvcc -g -O3 -include minimal_api.hpp -I "
  //                << PATH_TO_COGADB_SOURCE_CODE << "/include/ -I "
  //                << PATH_TO_COGADB_SOURCE_CODE << "/../hype-library/include/
  //                -c --compiler-options -fpic ";
  compile_command << filename << " -o " << ss.str() << ".o";
  //        ret=system("clang -g -I ../cogadb/include/ -I
  //        ../hype-library/include/ -c -fpic gen_query.cpp -o gen_query.o");

  std::cout << compile_command.str() << std::endl;

  ret = system(compile_command.str().c_str());
  if (ret != 0) {
    std::cout << "Compilation Failed!" << std::endl;
    return PipelinePtr();
  } else {
    std::cout << "Compilation Successful!" << std::endl;
  }
  std::stringstream linking_command;
  linking_command << "g++ -shared " << ss.str() << ".o -o " << ss.str() << ".so"
                  << std::endl;
  ret = system(linking_command.str().c_str());

  Timestamp end_compile = getTimestamp();

  //        std::stringstream shared_lib;
  //        shared_lib << "./" << ss.str() << ".so";
  //        std::cout << "Loading shared library '" << shared_lib.str() << "'"
  //        << std::endl;
  //        void *myso = dlopen(shared_lib.str().c_str(), RTLD_NOW);
  //        assert(myso != NULL);
  //        CompiledQueryPtr query = (CompiledQueryPtr) dlsym(myso,
  //        "_Z14compiled_queryRKSt6vectorIN6CoGaDB18AttributeReferenceESaIS1_EEN5boost10shared_ptrINS0_5StateEEE");
  //        //"compiled_query");
  //        error = dlerror();
  //        if (error) {
  //            std::cerr << error << std::endl;
  //            return PipelinePtr();
  //        }
  //        assert(query != NULL);
  //
  //        std::cout << "Attributes with Hash Tables: " << std::endl;
  //        for (size_t i = 0; i < param.size(); ++i) {
  //            std::cout << param[i].getVersionedTableName()
  //                    << "." << param[i].getVersionedAttributeName()
  //                    << ": " << param[i].hasHashTable() << std::endl;
  //        }
  //
  //        double compile_time_in_sec = double(end_compile - begin_compile) /
  //        (1000 * 1000 * 1000);
  //        return PipelinePtr(new SharedLibPipeline(query, param,
  //        compile_time_in_sec, myso, ss.str()));
  std::stringstream shared_lib_path;
  shared_lib_path << "./" << ss.str() << ".so";
  //        if(debug_code_generator)
  //            std::cout << "Loading shared library '" << shared_lib_path.str()
  //            << "'" << std::endl;

  SharedLibraryPtr shared_lib = SharedLibrary::load(shared_lib_path.str());
  assert(shared_lib != NULL);

  std::string symbol_name =
      "_Z14compiled_queryRKSt6vectorIN6CoGaDB18AttributeReferenceESaIS1_"
      "EEN5boost10shared_ptrINS0_5StateEEE";
  SharedCppLibPipelineQueryPtr query =
      shared_lib->getFunction<SharedCppLibPipelineQueryPtr>(symbol_name);
  assert(query != NULL);

  //        if(debug_code_generator){
  //            std::cout << "Attributes with Hash Tables: " << std::endl;
  //            for (size_t i = 0; i < param.size(); ++i) {
  //                std::cout << param[i].getVersionedTableName()
  //                        << "." << param[i].getVersionedAttributeName()
  //                        << ": " << param[i].hasHashTable() << std::endl;
  //            }
  //        }

  double compile_time_in_sec =
      double(end_compile - begin_compile) / (1000 * 1000 * 1000);
  return PipelinePtr(new SharedCppLibPipeline(query, param, compile_time_in_sec,
                                              shared_lib, pipe_info, ss.str()));
}
}
