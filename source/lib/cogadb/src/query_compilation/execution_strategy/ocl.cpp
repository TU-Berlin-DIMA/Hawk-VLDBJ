#include <query_compilation/execution_strategy/ocl.hpp>

namespace CoGaDB {

namespace ExecutionStrategy {

OCL::OCL(bool use_host_ptr, MemoryAccessPattern mem_access, cl_device_id dev_id)
    : PipelineExecutionStrategy(OCL_TARGET_CODE),
      use_host_ptr_(use_host_ptr),
      mem_access_(mem_access),
      dev_id_(dev_id),
      dev_type_(CL_DEVICE_TYPE_CPU) {
  assert(use_host_ptr_ == true);
  CL_CHECK(clGetDeviceInfo(dev_id_, CL_DEVICE_TYPE, sizeof(cl_device_type),
                           &dev_type_, NULL));
}

std::string OCL::getDefaultIncludeHeaders() const {
  std::stringstream out;
  out << "#include <query_compilation/minimal_api_c.h>" << std::endl;
  out << "#include <query_compilation/ocl_api.h>" << std::endl;

  return out.str();
}

std::string OCL::getFunctionSignature() const {
  return "const C_Table* compiled_query(C_Table** c_tables, "
         "OCL_Execution_Context* context, C_State* state)";
}

}  // namespace ExecutionStrategy

}  // namespace CoGaDB
