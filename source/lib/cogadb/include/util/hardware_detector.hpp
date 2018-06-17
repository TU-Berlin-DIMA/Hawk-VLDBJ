
#pragma once

#include <hype.hpp>

namespace CoGaDB {

  typedef hype::DeviceSpecification DeviceSpecification;

  class HardwareDetector {
   public:
    typedef std::vector<hype::DeviceSpecification> DeviceSpecifications;
    static HardwareDetector& instance();
    const DeviceSpecifications& getDeviceSpecifications();

    size_t getFreeMemorySizeInByte(
        const hype::ProcessingDeviceMemoryID& mem_id) const;
    size_t getTotalMemorySizeInByte(
        const hype::ProcessingDeviceMemoryID& mem_id) const;

    size_t getNumberOfCPUs() const;
    size_t getNumberOfGPUs() const;

   private:
    HardwareDetector();
    HardwareDetector(const HardwareDetector&);
    HardwareDetector& operator=(HardwareDetector&);
    bool detectHardware();
    bool detectCPUs();
    bool detectGPUs();
    size_t getNumberOfProcessorsForDeviceType(hype::ProcessingDeviceType) const;
    bool createCPUDevices(unsigned int number_of_devices);
    bool createGPUDevices(unsigned int number_of_devices);

    bool parseConfigFile();
    DeviceSpecifications dev_specs_;
    typedef std::map<hype::ProcessingDeviceMemoryID, int> MapMemIDToGPUDeviceID;
    int processing_device_id_;
  };

  typedef HardwareDetector::DeviceSpecifications DeviceSpecifications;

  hype::ProcessingDeviceMemoryID getMemoryIDForDeviceID(int gpu_id);

  hype::ProcessingDeviceID getIDOfFirstGPU();

  bool printGPUs(std::ostream& out);

  /* \brief convinience function, adds an algorithm specified by
   * AlgorithmSpecification to all available processing devices detected by the
   * HardwareDetector */
  // bool addAlgorithmSpecificationToHardware(const AlgorithmSpecification&);

}  // end namespace CogaDB
