#ifndef OCL_EXECUTION_STRATEGY_HPP
#define OCL_EXECUTION_STRATEGY_HPP

#include <core/global_definitions.hpp>

#include <query_compilation/execution_strategy/pipeline.hpp>

namespace CoGaDB {

  namespace ExecutionStrategy {

    class CodeBlock {
     public:
      std::list<std::string> upper_block;
      std::list<std::string> materialize_result_block;
      std::list<std::string> lower_block;
    };

    typedef boost::shared_ptr<CodeBlock> CodeBlockPtr;

    class GeneratedKernel {
     public:
      GeneratedKernel()
          : pre_kernel_call_host_code(),
            kernel_code_blocks(),
            post_kernel_call_host_code() {
        kernel_code_blocks.push_back(boost::make_shared<CodeBlock>());
      }
      std::stringstream pre_kernel_call_host_code;
      std::vector<CodeBlockPtr> kernel_code_blocks;
      std::stringstream post_kernel_call_host_code;
    };

    typedef boost::shared_ptr<GeneratedKernel> GeneratedKernelPtr;

    class OCL : public PipelineExecutionStrategy {
     public:
      OCL(bool use_host_ptr, MemoryAccessPattern mem_access,
          cl_device_id dev_id);

     protected:
      std::string getDefaultIncludeHeaders() const;
      std::string getFunctionSignature() const;

      bool use_host_ptr_;
      MemoryAccessPattern mem_access_;
      cl_device_id dev_id_;
      cl_device_type dev_type_;
    };

    typedef boost::shared_ptr<OCL> OCLPtr;

  }  // namespace ExecutionStrategy

}  // namespace CoGaDB

#endif  // OCL_EXECUTION_STRATEGY_HPP
