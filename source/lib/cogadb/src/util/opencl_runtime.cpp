
#include <core/global_definitions.hpp>
#include <core/variable_manager.hpp>
#include <util/opencl/prefix_sum.hpp>
#include <util/opencl_runtime.hpp>
#include <util/time_measurement.hpp>

#include <boost/compute/program.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread.hpp>

namespace CoGaDB {

OCL_Runtime::OCL_Runtime()
    : platforms_(), devices_(), cpus_(), gpus_(), phis_() {
  cl_int error = CL_SUCCESS;

  cl_uint num_platforms = 0;

  error = clGetPlatformIDs(0, NULL, &num_platforms);
  if (error == CL_SUCCESS && num_platforms > 0) {
    cl_platform_id* platforms =
        (cl_platform_id*)malloc(sizeof(cl_platform_id*) * num_platforms);

    error = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (error != CL_SUCCESS) {
      COGADB_FATAL_ERROR("Cannot retrieve an OpenCL platform!", "");
    }

    printf("=== %d OpenCL platform(s) found: ===\n", num_platforms);
    for (cl_uint i = 0; i < num_platforms; i++) {
      platforms_.push_back(platforms[i]);

      char buffer[10240];
      printf("  -- %d --\n", i);
      CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE, 10240,
                                 buffer, NULL));
      printf("  PROFILE = %s\n", buffer);
      CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 10240,
                                 buffer, NULL));
      printf("  VERSION = %s\n", buffer);
      CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 10240, buffer,
                                 NULL));
      printf("  NAME = %s\n", buffer);
      CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 10240,
                                 buffer, NULL));
      printf("  VENDOR = %s\n", buffer);
      CL_CHECK(clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS, 10240,
                                 buffer, NULL));
      printf("  EXTENSIONS = %s\n", buffer);

      cl_uint num_devices = 0;
      if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                         &num_devices) != CL_SUCCESS) {
        continue;
      }
      if (num_devices == 0) {
        continue;
      }

      cl_device_id* devices =
          (cl_device_id*)malloc(sizeof(cl_device_id*) * num_devices);
      assert(devices != NULL);

      CL_CHECK(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices,
                              devices, NULL));

      printf("=== %d OpenCL device(s) found on platform:\n", i);
      for (cl_uint j = 0; j < num_devices; j++) {
        devices_.push_back(devices[j]);
        char buffer[10240];
        cl_uint buf_uint;
        cl_ulong buf_ulong;
        printf("  -- %d --\n", j);
        CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer),
                                 buffer, NULL));
        printf("  DEVICE_NAME = %s\n", buffer);
        CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, sizeof(buffer),
                                 buffer, NULL));
        printf("  DEVICE_VENDOR = %s\n", buffer);
        CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(buffer),
                                 buffer, NULL));
        printf("  DEVICE_VERSION = %s\n", buffer);
        CL_CHECK(clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(buffer),
                                 buffer, NULL));
        printf("  DRIVER_VERSION = %s\n", buffer);
        CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                                 sizeof(buf_uint), &buf_uint, NULL));
        printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
        CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                 sizeof(buf_uint), &buf_uint, NULL));
        printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
        CL_CHECK(clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE,
                                 sizeof(buf_ulong), &buf_ulong, NULL));
        printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n",
               (unsigned long long)buf_ulong);

        cl_device_type type;
        clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type),
                        &type, NULL);
        if (type == CL_DEVICE_TYPE_CPU) {
          cpus_.push_back(devices[j]);
        } else if (type == CL_DEVICE_TYPE_GPU) {
          gpus_.push_back(devices[j]);
        } else if (type == CL_DEVICE_TYPE_ACCELERATOR) {
          phis_.push_back(devices[j]);
        } else if (type == CL_DEVICE_TYPE_CUSTOM) {
          COGADB_FATAL_ERROR("Unknown OpenCL Device Type!", "");
        } else {
          COGADB_FATAL_ERROR("Unknown OpenCL Device Type!", "");
        }
      }

      free(devices);
    }
    free(platforms);
  } else {
    COGADB_WARNING(
        "Cannot retrieve an OpenCL platform! OpenCL related features not "
        "available!",
        "");
  }
}

OCL_Runtime& OCL_Runtime::instance() {
  static OCL_Runtime runtime;
  return runtime;
}

cl_device_id OCL_Runtime::getCPU() {
  if (!cpus_.empty()) {
    return cpus_.front();
  }
  return NULL;
}

cl_device_id OCL_Runtime::getGPU() {
  if (!gpus_.empty()) {
    return gpus_.front();
  }
  return NULL;
}

cl_device_id OCL_Runtime::getDeviceID(cl_device_type type,
                                      bool host_unified_memory) {
  cl_device_id device = NULL;
  if (type == CL_DEVICE_TYPE_CPU) {
    if (!cpus_.empty()) {
      device = cpus_.front();
    } else {
      COGADB_FATAL_ERROR("No CPU OpenCL device found!", "");
    }
  } else if (type == CL_DEVICE_TYPE_GPU) {
    if (!gpus_.empty()) {
      for (size_t i = 0; i < gpus_.size(); ++i) {
        if (hasHostUnifiedMemory(gpus_[i]) == host_unified_memory) {
          return gpus_[i];
        }
      }
      std::string error;
      if (host_unified_memory) {
        error = "integrated GPU with host unified memory";
      } else {
        error = "dedicated GPU with dedicated device memory";
      }
      COGADB_FATAL_ERROR("Could not find " << error, "");
    } else {
      COGADB_FATAL_ERROR("No GPU OpenCL device found!", "");
    }
  } else if (type == CL_DEVICE_TYPE_ACCELERATOR) {
    if (!phis_.empty()) {
      return phis_.front();
    } else {
      COGADB_FATAL_ERROR("No Xeon Phi coprocessor found!", "");
      return NULL;
    }
  } else if (type == CL_DEVICE_TYPE_CUSTOM) {
    COGADB_FATAL_ERROR("Unknown OpenCL Device Type!", "");
  } else {
    COGADB_FATAL_ERROR("Unknown OpenCL Device Type!", "");
  }
  return device;
}

bool OCL_Runtime::hasHostUnifiedMemory(cl_device_id id) const {
  return boost::compute::device(id).get_info<bool>(
      CL_DEVICE_HOST_UNIFIED_MEMORY);
}

const std::vector<cl_device_id> OCL_Runtime::getDevicesWithDedicatedMemory() {
  std::vector<cl_device_id> devices;
  for (size_t i = 0; i < devices_.size(); ++i) {
    cl_bool unified_host_memory_tmp = true;
    CL_CHECK(clGetDeviceInfo(devices_[i], CL_DEVICE_HOST_UNIFIED_MEMORY,
                             sizeof(cl_bool), &unified_host_memory_tmp, NULL));
    if (!unified_host_memory_tmp) {
      devices.push_back(devices_[i]);
    }
  }
  return devices;
}

OCL_Runtime::OCL_DeviceStructures::OCL_DeviceStructures(
    const boost::compute::context& _context,
    const std::vector<boost::compute::command_queue>& _compute_queues,
    const std::vector<boost::compute::command_queue>&
        _copy_host_to_device_queues,
    const std::vector<boost::compute::command_queue>&
        _copy_device_to_host_queues,
    const std::string& _create_strategy)
    : context(_context),
      compute_queues(_compute_queues),
      copy_host_to_device_queues(_copy_host_to_device_queues),
      copy_device_to_host_queues(_copy_device_to_host_queues),
      create_strategy(_create_strategy) {
  assert(context != NULL);
}

OCL_Runtime::OCL_DeviceStructures::~OCL_DeviceStructures() {}

const OCL_Runtime::OCL_DeviceStructuresPtr OCL_Runtime::getDeviceStructures(
    cl_device_id device) {
  std::lock_guard<std::mutex> lock(mutex_);

  assert(device != nullptr);

  unsigned int num_threads =
      VariableManager::instance().getVariableValueInteger(
          "code_gen.num_threads");

  auto strategy = VariableManager::instance().getVariableValueString(
      "code_gen.cl_command_queue_strategy");

  auto& strategy_map = dev_to_context_map_[strategy];

  auto it = strategy_map.find(device);

  if (it == strategy_map.end() || strategy != it->second->create_strategy ||
      num_threads > it->second->compute_queues.size()) {
    OCL_DeviceStructuresPtr ptr;

    OCL_Kernels::instance().resetCache();

    if (strategy == "multiple") {
      ptr = getDeviceStructuresMultipleCommandQueues(device, num_threads);
    } else if (strategy == "subdevices") {
      ptr = getDeviceStructuresSubDevicesCommandQueues(device, num_threads);
    } else if (strategy == "outoforder") {
      ptr = getDeviceStructuresOutOfOrderCommandQueues(device, num_threads);
    } else {
      COGADB_FATAL_ERROR("Unknown strategy \"" << strategy << "\"!", "");
    }

    if (it == strategy_map.end()) {
      it = strategy_map.insert(std::make_pair(device, ptr)).first;
    } else {
      it->second = ptr;
    }
  }

  return it->second;
}

OCL_Runtime::OCL_DeviceStructuresPtr
OCL_Runtime::getDeviceStructuresMultipleCommandQueues(
    cl_device_id device, unsigned int num_threads) {
  auto bdevice = boost::compute::device(device);
  auto context = boost::compute::context(bdevice);

  std::vector<boost::compute::command_queue> compute_queues, to_host_queues,
      to_device_queues;
  for (unsigned int i = 0; i < num_threads; ++i) {
    auto compute_queue = boost::compute::command_queue(context, bdevice);
    compute_queues.push_back(compute_queue);

    auto transfer_host_to_device_queue =
        boost::compute::command_queue(context, bdevice);
    to_device_queues.push_back(transfer_host_to_device_queue);

    auto transfer_device_to_host_queue =
        boost::compute::command_queue(context, bdevice);
    to_host_queues.push_back(transfer_device_to_host_queue);
  }

  return boost::make_shared<OCL_DeviceStructures>(
      context, compute_queues, to_device_queues, to_host_queues, "multiple");
}

OCL_Runtime::OCL_DeviceStructuresPtr
OCL_Runtime::getDeviceStructuresSubDevicesCommandQueues(
    cl_device_id device, unsigned int num_threads) {
  auto bdevice = boost::compute::device(device);

  if (!bdevice.check_version(1, 2)) {
    COGADB_FATAL_ERROR("At least OpenCL 1.2 is required!", "");
  }

  static auto sub_devices = bdevice.partition_equally(1);

  if (sub_devices.size() < num_threads) {
    COGADB_FATAL_ERROR(
        "The device only supports "
            << sub_devices.size()
            << " sub devices. So we are not able to create sufficient"
               " command queues!",
        "");
  }

  auto context = boost::compute::context(sub_devices);
  std::vector<boost::compute::command_queue> compute_queues, to_host_queues,
      to_device_queues;
  for (unsigned int i = 0; i < num_threads; ++i) {
    auto& sub_device = sub_devices[i];

    auto compute_queue = boost::compute::command_queue(context, sub_device);
    compute_queues.push_back(compute_queue);

    auto transfer_host_to_device_queue =
        boost::compute::command_queue(context, sub_device);
    to_device_queues.push_back(transfer_host_to_device_queue);

    auto transfer_device_to_host_queue =
        boost::compute::command_queue(context, sub_device);
    to_host_queues.push_back(transfer_device_to_host_queue);
  }

  return boost::make_shared<OCL_DeviceStructures>(
      context, compute_queues, to_device_queues, to_host_queues, "subdevices");
}

OCL_Runtime::OCL_DeviceStructuresPtr
OCL_Runtime::getDeviceStructuresOutOfOrderCommandQueues(
    cl_device_id device, unsigned int num_threads) {
  auto bdevice = boost::compute::device(device);
  auto context = boost::compute::context(bdevice);

  std::vector<boost::compute::command_queue> compute_queues(num_threads),
      to_host_queues(num_threads), to_device_queues(num_threads);

  std::fill_n(
      compute_queues.begin(), num_threads,
      boost::compute::command_queue(
          context, bdevice,
          boost::compute::command_queue::enable_out_of_order_execution));

  std::fill_n(
      to_host_queues.begin(), num_threads,
      boost::compute::command_queue(
          context, bdevice,
          boost::compute::command_queue::enable_out_of_order_execution));

  std::fill_n(
      to_device_queues.begin(), num_threads,
      boost::compute::command_queue(
          context, bdevice,
          boost::compute::command_queue::enable_out_of_order_execution));

  return boost::make_shared<OCL_DeviceStructures>(
      context, compute_queues, to_device_queues, to_host_queues, "outoforder");
}

const std::pair<OCL_Execution_ContextPtr, double>
OCL_Runtime::compileDeviceKernel(cl_device_id device,
                                 const std::string& program_source) {
  assert(device != NULL);
  auto dev_structs = getDeviceStructures(device);
  assert(dev_structs != NULL);

  std::lock_guard<std::mutex> lock(mutex_);

  CoGaDB::Timestamp begin = CoGaDB::getTimestamp();
  auto program = boost::compute::program::create_with_source(
      program_source, dev_structs->context);

  try {
    program.build();

#ifndef NDEBUG
    std::cout << "Kernel Build Log: " << program.build_log() << std::endl;
#endif
  } catch (...) {
    std::cout << "Kernel Source: " << std::endl
              << "'" << program_source << "'" << std::endl;
    std::cout << "CL Compilation failed: " << program.build_log() << std::endl;
  }

  CoGaDB::Timestamp end = CoGaDB::getTimestamp();
  double kernel_compilation_time_in_sec =
      double(end - begin) / (1000 * 1000 * 1000);

  auto exec_context = boost::make_shared<OCL_Execution_Context>(
      dev_structs->context, dev_structs->compute_queues,
      dev_structs->copy_host_to_device_queues,
      dev_structs->copy_device_to_host_queues, program);

  return std::make_pair(exec_context, kernel_compilation_time_in_sec);
}

std::vector<std::string> OCL_Runtime::getAvailableDeviceTypes() const {
  std::vector<std::string> result;

  if (!cpus_.empty()) {
    result.push_back("cpu");
  }

  if (!gpus_.empty()) {
    bool found_igpu = false, found_dgpu = false;

    for (auto id : gpus_) {
      if (hasHostUnifiedMemory(id) && !found_igpu) {
        result.push_back("igpu");
        found_igpu = true;
      } else if (!found_dgpu) {
        result.push_back("dgpu");
        found_dgpu = true;
      }
    }
  }

  if (!phis_.empty()) {
    result.push_back("phi");
  }

  return result;
}

OCLVendor OCL_Runtime::getDeviceVendor(cl_device_id dev_id) const {
  auto name = boost::compute::device(dev_id).vendor();

  if (name == "Intel(R) Corporation" || name == "Intel(R) OpenCL" ||
      name == "GenuineIntel") {
    return OCLVendor_Intel;
  }

  if (name == "Advanced Micro Devices, Inc." || name == "AuthenticAMD") {
    return OCLVendor_AMD;
  }

  if (name == "NVIDIA Corporation" || name == "NVIDIA CUDA") {
    return OCLVendor_Nvidia;
  }

  return OCLVendor_Unknown;
}

cl_device_id getOpenCLGlobalDevice() {
  cl_device_id device_id;
  cl_device_type device_type;
  const std::string dev_type =
      VariableManager::instance().getVariableValueString(
          "code_gen.cl_device_type");
  bool host_unified_memory = true;
  if (dev_type == "cpu") {
    device_type = CL_DEVICE_TYPE_CPU;
    host_unified_memory = true;
  } else if (dev_type == "gpu" || dev_type == "dgpu") {
    device_type = CL_DEVICE_TYPE_GPU;
    host_unified_memory = false;
  } else if (dev_type == "igpu") {
    device_type = CL_DEVICE_TYPE_GPU;
    host_unified_memory = true;
  } else if (dev_type == "phi") {
    device_type = CL_DEVICE_TYPE_ACCELERATOR;
    host_unified_memory = false;
  } else {
    COGADB_FATAL_ERROR("", "");
  }

  device_id =
      OCL_Runtime::instance().getDeviceID(device_type, host_unified_memory);
  assert(device_id != NULL);
  return device_id;
}
}
