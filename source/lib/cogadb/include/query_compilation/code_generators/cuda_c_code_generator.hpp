#include <query_compilation/code_generators/cpp_code_generator.hpp>
#include <query_compilation/kernel_execution.hpp>
#include <vector>

namespace CoGaDB {

  enum CodePosition { PRE_COMPUTE, POST_SCAN, POST_PROJECTION };
  enum MemcpyDirection { H2D, D2H };
  enum MemoryLocation { DEVICE, HOST };

  const std::string getDeviceInputArrayVarName(
      const AttributeReference& attr_ref);

  class CUDA_C_CodeGenerator : public CPPCodeGenerator {
   public:
    CUDA_C_CodeGenerator(const ProjectionParam& param, const TablePtr table,
                         uint32_t version = 1);

    CUDA_C_CodeGenerator(const ProjectionParam& param);

    virtual void printCode(std::ostream& out);
    virtual void produceCode();
    virtual bool consumeSelection_impl(const PredicateExpressionPtr pred_expr);
    virtual bool consumeBuildHashTable_impl(const AttributeReference& attr);
    virtual bool consumeProbeHashTable_impl(
        const AttributeReference& hash_table_attr,
        const AttributeReference& probe_attr);

   protected:
    void init();

    bool produceTuples(const ScanParam& param);
    virtual bool createForLoop_impl(const TablePtr table, uint32_t version);
    const PipelinePtr compile();

    void produceAllocateResultTable();
    void produceCopyDeviceResult();
    void produceAllocateDeviceResult(std::string varname);
    const std::string getCodeDenseWriteProjectionElements();

    void generateCodeComputeStructure();
    void produceProjection();
    void produceScanFlagArray();
    void produceScanBlockResultSizes();
    void produceBlockScanSelected(KernelExecution& kernel);

    void produceAllocArray(std::string datatype, std::string name,
                           std::string num_elements_varname,
                           MemoryLocation location, CodePosition pos,
                           bool doFree = true);

    void produceAllocArray(std::string datatype, std::string varname,
                           size_t num_elements, MemoryLocation location,
                           CodePosition code_pos, bool doFree = true);

    void produceDataTransfer(std::string datatype, std::string name_dest,
                             std::string name_src,
                             std::string num_elements_src_varname,
                             MemcpyDirection direction, CodePosition pos);

    void produceDataTransfer(std::string datatype, std::string name_dest,
                             std::string name_src, size_t num_elements,
                             MemcpyDirection direction, CodePosition code_pos);

    // code generation state
    bool codeIsProduced;
    bool producedProjection;

    // for scan and project kernel
    int blockSize;

    // code framework
    std::stringstream cuda_includes;

    // code pieces: datastructure allocation, retrieval and transfer
    std::stringstream constant_declarations;
    std::stringstream pre_compute_allocation_code;
    std::stringstream pre_compute_transfer_code;
    std::stringstream post_scan_allocation_code;
    std::stringstream post_scan_transfer_code;
    std::stringstream post_projection_allocation_code;
    std::stringstream post_projection_transfer_code;
    std::stringstream free_memory_code;
    std::stringstream get_hashtables_code;
    std::stringstream allocate_hashtables_code;

    // code pieces: computation
    KernelExecution scan_pipeline_kernel;
    KernelExecution project_pipeline_kernel;
    std::string scan_compute_structure;
    std::string compute_structure_code;
    std::stringstream pre_scan_sort_code;
    std::stringstream make_hashtable_entries_code;
    std::stringstream get_hashtable_values_code;
  };
}
