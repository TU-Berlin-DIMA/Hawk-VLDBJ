
#include <boost/thread/thread.hpp>
#include <core/global_definitions.hpp>
#include <util/hardware_detector.hpp>

#include <boost/program_options.hpp>
#include <fstream>

#ifdef ENABLE_GPU_ACCELERATION
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace CoGaDB {

#ifdef HAVE_CUDAMEMGETINFO

size_t getTotalGPUMemorySizeInByte() {
  // get the amount of free and total memory of the device
  size_t available_memory_in_bytes = 0, total_memory_in_bytes = 0;
  if (cudaSuccess !=
      cudaMemGetInfo(&available_memory_in_bytes, &total_memory_in_bytes)) {
    // std::cerr << "Error while trying to retrieve memory size information!
    // Aborting..." << std::endl;
    return 0;
  }
  return total_memory_in_bytes;
}

size_t getFreeGPUMemorySizeInByte() {
  // get the amount of free and total memory of the device
  size_t available_memory_in_bytes = 0, total_memory_in_bytes = 0;
  if (cudaSuccess !=
      cudaMemGetInfo(&available_memory_in_bytes, &total_memory_in_bytes)) {
    // std::cerr << "Error while trying to retrieve memory size information!
    // Aborting..." << std::endl;
    return 0;
  }

  return available_memory_in_bytes;
}

unsigned int getNumberofGPUMultiprocessors() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  return prop.multiProcessorCount;
}

void printGPUStatus() {
  std::cout << "Number of GPU Cores: " << getNumberofGPUMultiprocessors()
            << std::endl;
  std::cout << "Total GPU RAM: " << getTotalGPUMemorySizeInByte() << std::endl;
  std::cout << "Available GPU RAM: " << getFreeGPUMemorySizeInByte()
            << std::endl;
}

bool printGPUs(std::ostream& out) {
  cudaError_t err = cudaSuccess;
  int device_count = 0;
  err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess) {
    COGADB_ERROR("Could not get number of CUDA capable devices!", "");
    return false;
  }
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, i);
    if (err != cudaSuccess) {
      COGADB_ERROR("Could not get Properties of Device " << i << "!", "");
      return false;
    }
    out << "GPU " << i << ": " << prop.name << std::endl;
    out << "\t"
        << "Compute Capability: " << prop.major << "." << prop.minor
        << std::endl;
    out << "\t"
        << "Global Memory Size (GB): "
        << double(prop.totalGlobalMem) / (1024 * 1024 * 1024) << std::endl;
    out << "\t"
        << "#Multiprocessors: " << prop.multiProcessorCount << std::endl;
    out << "\t"
        << "Clockrate (Mhz): " << double(prop.clockRate) / (1000) << std::endl;
    out << "\t"
        << "Supports Concurrent Kernels: " << prop.concurrentKernels
        << std::endl;
    out << "\t"
        << "asyncEngineCount: " << prop.asyncEngineCount << std::endl;
    out << "\t"
        << "Kernel Timeout Enabled: " << prop.kernelExecTimeoutEnabled
        << std::endl;
    out << "\t"
        << "Overlap Copy and Computation: " << prop.deviceOverlap << std::endl;
    out << "\t"
        << "ECC Enabled: " << prop.ECCEnabled << std::endl;
  }

  return true;
}

int CUDA_device_check(int gpudevice) {
  int device_count = 0;
  int device;  // used with  cudaGetDevice() to verify cudaSetDevice()

  // get the number of non-emulation devices  detected
  cudaGetDeviceCount(&device_count);
  if (gpudevice > device_count) {
    printf("gpudevice >=  device_count ... exiting\n");
    exit(1);
  }
  cudaError_t cudareturn;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpudevice);

  if (deviceProp.major > 999) {
    std::cout << "No CUDA capable GPU found!" << std::endl;
    std::cout << "CUDA Device  Emulation (CPU) detected!" << std::endl;
  }

  // choose a cuda device for kernel  execution
  cudareturn = cudaSetDevice(gpudevice);
  if (cudareturn == cudaErrorInvalidDevice) {
    perror("cudaSetDevice returned  cudaErrorInvalidDevice");
  } else {
  }
  return 0;
}

int check_for_CUDA_devices() {
  int device_count = 0;

  // get the number of non-emulation devices  detected
  cudaGetDeviceCount(&device_count);
  if (device_count <= 0) {
    return 0;
  } else {
    // count number of CUDA capable devices
    int return_val = 0;
    for (int i = 0; i < device_count; i++) {
      if (CUDA_device_check(i) != -1) return_val++;
    }
    return return_val;
  }
}

#else /* !HAVE_CUDAMEMGETINFO */

size_t getTotalGPUMemorySizeInByte() { return 0; }

size_t getFreeGPUMemorySizeInByte() { return 0; }

unsigned int getNumberofGPUMultiprocessors() { return 0; }

void printGPUStatus() {
  std::cout << "Cannot determine status of GPUs! CoGaDB was compiled without "
               "GPU support!"
            << std::endl;
}

bool printGPUs(std::ostream& out) {
  out << "Cannot list GPUs! CoGaDB was compiled without GPU support!"
      << std::endl;
  return true;
}

int CUDA_device_check(int gpudevice) { return 0; }

int check_for_CUDA_devices() { return 0; }

#endif /* HAVE_CUDAMEMGETINFO */

HardwareDetector::HardwareDetector()
    : dev_specs_(), processing_device_id_(hype::PD0) {
  this->parseConfigFile();
}

HardwareDetector& HardwareDetector::instance() {
  static HardwareDetector hwd;
  return hwd;
}

const DeviceSpecifications& HardwareDetector::getDeviceSpecifications() {
  return dev_specs_;
}

size_t HardwareDetector::getFreeMemorySizeInByte(
    const hype::ProcessingDeviceMemoryID& mem_id) const {
  for (size_t i = 0; i < this->dev_specs_.size(); ++i) {
    if (dev_specs_[i].getMemoryID() == mem_id)
      return dev_specs_[i].getAvailableMemoryCapacity();
  }
  COGADB_ERROR(
      "Could not find Processing Device with memory ID " << (int)mem_id, "");
  return 0;
}
size_t HardwareDetector::getTotalMemorySizeInByte(
    const hype::ProcessingDeviceMemoryID& mem_id) const {
  for (size_t i = 0; i < this->dev_specs_.size(); ++i) {
    if (dev_specs_[i].getMemoryID() == mem_id)
      return dev_specs_[i].getTotalMemoryCapacity();
  }
  COGADB_ERROR(
      "Could not find Processing Device with memory ID " << (int)mem_id, "");
  return 0;
}

bool HardwareDetector::detectHardware() {
  bool ret = true;
  ret = ret && detectCPUs();
  ret = ret && detectGPUs();
  return ret;
}

bool HardwareDetector::detectCPUs() {
  // add one virtual CPU for each core
  return this->createCPUDevices(1);  // boost::thread::hardware_concurrency());
}

bool HardwareDetector::detectGPUs() {
  int nDevices = check_for_CUDA_devices();

  if (nDevices > 0) return createGPUDevices(1);  // nDevices);

  return true;
}

size_t HardwareDetector::getNumberOfProcessorsForDeviceType(
    hype::ProcessingDeviceType type) const {
  unsigned int counter = 0;
  for (unsigned int i = 0; i < dev_specs_.size(); ++i) {
    if (dev_specs_[i].getDeviceType() == type) counter++;
  }
  return counter;
}

size_t HardwareDetector::getNumberOfCPUs() const {
  return getNumberOfProcessorsForDeviceType(hype::CPU);
}

size_t HardwareDetector::getNumberOfGPUs() const {
  return getNumberOfProcessorsForDeviceType(hype::GPU);
}

bool HardwareDetector::createCPUDevices(unsigned int number_of_devices) {
  if (number_of_devices == 0) return false;
  for (unsigned int i = 0; i < number_of_devices; ++i) {
    assert(this->processing_device_id_ < hype::PD_DMA0);
    dev_specs_.push_back(
        DeviceSpecification((hype::ProcessingDeviceID)processing_device_id_++,
                            hype::CPU, hype::PD_Memory_0));  // one host CPU
  }
  return true;
}

// getFreeGPUMemorySizeInByte

bool HardwareDetector::createGPUDevices(unsigned int number_of_devices) {
  //        int memory_id=hype::PD_Memory_1;
  //        for(unsigned int i=0;i<nDevices;++i){
  //            assert(processing_device_id_<hype::PD_DMA0);
  //            dev_specs_.push_back(DeviceSpecification((hype::ProcessingDeviceID)processing_device_id_++,hype::GPU,
  //            (hype::ProcessingDeviceMemoryID)memory_id)); //one dedicated GPU
  //        }

  // we currently do not support multiple real GPUs anyway, so if we
  // add additional GPUs, we assume they have the same memory id,
  // which is expecially handy for varying the number of virtual GPUs for one
  // physical GPU
  for (unsigned int i = 0; i < number_of_devices; ++i) {
    assert(processing_device_id_ < hype::PD_DMA0);
    dev_specs_.push_back(DeviceSpecification(
        (hype::ProcessingDeviceID)processing_device_id_++, hype::GPU,
        hype::PD_Memory_1, &getFreeGPUMemorySizeInByte,
        getTotalGPUMemorySizeInByte()));  // one dedicated GPU
  }
  return true;
}

bool HardwareDetector::parseConfigFile() {
  unsigned int number_of_cpus = 0;
  unsigned int number_of_dedicated_gpus = 0;

  // Declare the supported options.
  boost::program_options::options_description desc("Allowed options");
  desc.add_options()("number_of_cpus",
                     boost::program_options::value<unsigned int>(),
                     "set the number of virtual cpu processing devices")(
      "number_of_dedicated_gpus", boost::program_options::value<unsigned int>(),
      "set the number of virtual dedicated gpu processing devices");

  boost::program_options::variables_map vm;
  std::fstream config_file("hardware_specification.conf");
  if (!config_file.good()) {
    return this->detectHardware();

    // if (!quiet) std::cout << "[HyPE]: No Configuration File 'hype.conf'
    // found! Parsing environment variables..." << std::endl;
    // boost::program_options::store(boost::program_options::parse_environment(desc,
    // map_environment_variable_name_to_option_name), vm);
    // return; //don't parse config file, if stream is not okay
  } else {
    if (!quiet)
      std::cout << "Parsing Configuration File 'hardware_specification.conf'..."
                << std::endl;
    boost::program_options::store(
        boost::program_options::parse_config_file(config_file, desc), vm);
  }

  boost::program_options::notify(vm);

  if (vm.count("number_of_cpus")) {
    if (!quiet)
      std::cout << "number_of_cpus: " << vm["number_of_cpus"].as<unsigned int>()
                << "\n";
    number_of_cpus = vm["number_of_cpus"].as<unsigned int>();
    if (!this->createCPUDevices(number_of_cpus)) {
      COGADB_FATAL_ERROR(
          "Invalid value for variable number_of_cpus: " << number_of_cpus, "");
    }
  } else {
    if (!quiet)
      std::cout << "number_of_cpus was not specified, perform automatic "
                   "detection of available CPUs...\n";
    this->detectCPUs();
  }

  if (vm.count("number_of_dedicated_gpus")) {
    if (!quiet)
      std::cout << "number_of_dedicated_gpus: "
                << vm["number_of_dedicated_gpus"].as<unsigned int>() << "\n";
    number_of_dedicated_gpus =
        vm["number_of_dedicated_gpus"].as<unsigned int>();
    if (!this->createGPUDevices(number_of_dedicated_gpus)) {
      COGADB_FATAL_ERROR("Invalid value for variable number_of_dedicated_gpus: "
                             << number_of_dedicated_gpus,
                         "");
    }
  } else {
    if (!quiet)
      std::cout << "number_of_dedicated_gpus was not specified, perform "
                   "automatic detection of available GPUs\n";
    this->detectGPUs();
  }

  return true;
}

hype::ProcessingDeviceMemoryID getMemoryIDForDeviceID(int gpu_id) {
  assert(gpu_id == 0);
  return hype::PD_Memory_1;
}

hype::ProcessingDeviceID getIDOfFirstGPU() { return hype::PD1; }

}  // end namespace CogaDB
