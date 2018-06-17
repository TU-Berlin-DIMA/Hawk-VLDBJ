
/*
 * File:   opencl_runtime.hpp
 * Author: sebastian
 *
 * Created on 26. Februar 2016, 20:25
 */

#ifndef OPENCL_RUNTIME_HPP
#define OPENCL_RUNTIME_HPP

#include <mutex>
#include <query_compilation/ocl_api.hpp>
#include <unordered_map>
#include <vector>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>

namespace CoGaDB {

  enum OCLVendor {
    OCLVendor_Intel,
    OCLVendor_AMD,
    OCLVendor_Nvidia,
    OCLVendor_Unknown
  };

  class OCL_Runtime {
   public:
    static OCL_Runtime& instance();
    cl_device_id getCPU();
    cl_device_id getGPU();
    cl_device_id getDeviceID(cl_device_type type, bool host_unified_memory);
    const std::vector<cl_device_id> getDevicesWithDedicatedMemory();
    const std::pair<OCL_Execution_ContextPtr, double> compileDeviceKernel(
        cl_device_id device, const std::string& kernel_source);

    struct OCL_DeviceStructures {
      OCL_DeviceStructures(
          const boost::compute::context& _context,
          const std::vector<boost::compute::command_queue>& _compute_queues,
          const std::vector<boost::compute::command_queue>&
              _copy_host_to_device_queues,
          const std::vector<boost::compute::command_queue>&
              _copy_device_to_host_queues,
          const std::string& create_strategy);

      ~OCL_DeviceStructures();

      boost::compute::context context;
      std::vector<boost::compute::command_queue> compute_queues;
      std::vector<boost::compute::command_queue> copy_host_to_device_queues;
      std::vector<boost::compute::command_queue> copy_device_to_host_queues;
      std::string create_strategy;
    };

    typedef boost::shared_ptr<OCL_DeviceStructures> OCL_DeviceStructuresPtr;
    const OCL_DeviceStructuresPtr getDeviceStructures(cl_device_id device);

    std::vector<std::string> getAvailableDeviceTypes() const;

    OCLVendor getDeviceVendor(cl_device_id dev_id) const;

   private:
    OCL_Runtime();
    OCL_Runtime(OCL_Runtime&);
    OCL_Runtime& operator=(OCL_Runtime&);

    OCL_DeviceStructuresPtr getDeviceStructuresMultipleCommandQueues(
        cl_device_id device, unsigned int num_threads);

    OCL_DeviceStructuresPtr getDeviceStructuresSubDevicesCommandQueues(
        cl_device_id device, unsigned int num_threads);

    OCL_DeviceStructuresPtr getDeviceStructuresOutOfOrderCommandQueues(
        cl_device_id device, unsigned int num_threads);

    bool hasHostUnifiedMemory(cl_device_id id) const;

    std::vector<cl_platform_id> platforms_;
    std::vector<cl_device_id> devices_;
    std::vector<cl_device_id> cpus_;
    std::vector<cl_device_id> gpus_;
    std::vector<cl_device_id> phis_;
    std::unordered_map<std::string,
                       std::map<cl_device_id, OCL_DeviceStructuresPtr>>
        dev_to_context_map_;
    std::mutex mutex_;
  };

  cl_device_id getOpenCLGlobalDevice();
}

#endif /* OPENCL_RUNTIME_HPP */
