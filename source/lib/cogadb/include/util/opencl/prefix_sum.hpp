
#include <boost/thread.hpp>
#include <core/global_definitions.hpp>
#include <util/opencl_runtime.hpp>

namespace CoGaDB {

  class OCL_Kernels {
   public:
    static OCL_Kernels& instance();
    cl_kernel getKernel(const std::string& kernel_name, cl_device_id dev_id,
                        cl_context context);

    void resetCache();

   private:
    OCL_Kernels();

    typedef std::map<cl_device_id, cl_program> Programs;
    Programs programs_;
    boost::mutex mutex_;
  };

  size_t ocl_prefix_sum(cl_command_queue queue, cl_program program,
                        cl_mem cl_output_mem_flags, cl_mem cl_output_prefix_sum,
                        size_t num_elements);

}  // end namespace CoGaDB
