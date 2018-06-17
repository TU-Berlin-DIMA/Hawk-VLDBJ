/*
 * File:   variant_measurement.hpp
 * Author: christoph
 *
 * Created on 3. June 2016, 09:54
 */

#ifndef VARIANT_MEASUREMENT_HPP
#define VARIANT_MEASUREMENT_HPP

#include <parser/client.hpp>
#include <util/variant_configurator.hpp>

#include <boost/algorithm/string.hpp>

#include <limits.h>
#include <unistd.h>

#ifndef HOST_NAME_MAX
#if defined(_POSIX_HOST_NAME_MAX)
#define HOST_NAME_MAX _POSIX_HOST_NAME_MAX
#endif
#endif

namespace CoGaDB {

  struct VariantMeasurement {
    VariantMeasurement(bool _success, double _exec_time, double _exec_pipe_time,
                       double _comp_time, double _kernel_comp_time,
                       double _host_comp_time, double _overhead_time)
        : success(_success),
          total_execution_time_in_s(_exec_time),
          total_pipeline_execution_time_in_s(_exec_pipe_time),
          total_compilation_time_in_s(_comp_time),
          total_kernel_compilation_time_in_s(_kernel_comp_time),
          total_host_compilation_time_in_s(_host_comp_time),
          total_overhead_time_in_s(_overhead_time) {}

    bool success;
    double total_execution_time_in_s;
    double total_pipeline_execution_time_in_s;
    double total_compilation_time_in_s;
    double total_kernel_compilation_time_in_s;
    double total_host_compilation_time_in_s;
    double total_overhead_time_in_s;
  };

  class QueryContext;
  typedef boost::shared_ptr<QueryContext> QueryContextPtr;

  const VariantMeasurement createVariantMeasurement(
      double total_elapsed_time_in_s, QueryContextPtr context);

  void print(ClientPtr client, const VariantMeasurement& vm);

  struct VariantExecutionStatistics {
    VariantExecutionStatistics(double _min, double _max, double _mean,
                               double _median, double _standard_deviation,
                               double _variance)
        : min(_min),
          max(_max),
          mean(_mean),
          median(_median),
          standard_deviation(_standard_deviation),
          variance(_variance) {}

    VariantExecutionStatistics()
        : min(0),
          max(0),
          mean(0),
          median(0),
          standard_deviation(0),
          variance(0) {}

    VariantExecutionStatistics(std::vector<VariantMeasurement> measurements);

    double min;
    double max;
    double mean;
    double median;
    double standard_deviation;
    double variance;
  };

  struct VariantPerformance {
    Variant first;
    VariantExecutionStatistics second;

    VariantPerformance(Variant first_, VariantExecutionStatistics second_)
        : first(first_), second(second_) {}

    VariantPerformance() : first(), second() {}
  };

  class DeviceLogger {
   public:
    DeviceLogger(ClientPtr client, cl_device_id device) : client(client) {
      // Temporary c string
      char cString[1024];

      // Host name
      char cHostName[HOST_NAME_MAX];
      gethostname(cHostName, HOST_NAME_MAX);
      this->hostName = cHostName;

      // Device
      this->device = getOpenCLGlobalDevice();

      // Device name
      if (this->device &&
          clGetDeviceInfo(this->device, CL_DEVICE_NAME, sizeof(cString),
                          cString, NULL) == CL_SUCCESS) {
        this->deviceName = cString;
        boost::trim(this->deviceName);
      }

      // Device type
      this->deviceType = VariableManager::instance().getVariableValueString(
          "code_gen.cl_device_type");

      // Device vendor
      if (this->device &&
          clGetDeviceInfo(this->device, CL_DEVICE_VENDOR, sizeof(cString),
                          cString, NULL) == CL_SUCCESS) {
        this->deviceVendor = cString;
        boost::trim(this->deviceVendor);
      }

      // Device version
      if (this->device &&
          clGetDeviceInfo(this->device, CL_DEVICE_VENDOR, sizeof(cString),
                          cString, NULL) == CL_SUCCESS) {
        this->deviceVersion = cString;
        boost::trim(this->deviceVersion);
      }

      // Driver version
      if (this->device &&
          clGetDeviceInfo(this->device, CL_DRIVER_VERSION, sizeof(cString),
                          cString, NULL) == CL_SUCCESS) {
        this->driverVersion = cString;
        boost::trim(this->driverVersion);
      }

      // Device platform
      if (this->device) {
        clGetDeviceInfo(this->device, CL_DEVICE_PLATFORM,
                        sizeof(this->platform), &this->platform, NULL);
      }

      // Platform name
      if (this->platform &&
          clGetPlatformInfo(this->platform, CL_PLATFORM_NAME, sizeof(cString),
                            cString, NULL) == CL_SUCCESS) {
        this->platformName = cString;
        boost::trim(this->platformName);
      }

      // Platform vendor
      if (this->platform &&
          clGetPlatformInfo(this->platform, CL_PLATFORM_VENDOR, sizeof(cString),
                            cString, NULL) == CL_SUCCESS) {
        this->platformVendor = cString;
        boost::trim(this->platformVendor);
      }

      // Platform version
      if (this->platform &&
          clGetPlatformInfo(this->platform, CL_PLATFORM_VERSION,
                            sizeof(cString), cString, NULL) == CL_SUCCESS) {
        this->platformVersion = cString;
        boost::trim(this->platformVersion);
      }

      // Initialize cache
      cache = build();
    }

    std::string getHostName() const { return hostName; }

    cl_device_id getDevice() const { return device; }

    std::string getDeviceName() const { return deviceName; }

    std::string getDeviceType() const { return deviceType; }

    std::string getDeviceVendor() const { return deviceVendor; }

    std::string getDeviceVersion() const { return deviceVersion; }

    std::string getDriverVersion() const { return driverVersion; }

    cl_platform_id getPlatform() const { return platform; }

    std::string getPlatformName() const { return platformName; }

    std::string getPlatformVendor() const { return platformVendor; }

    std::string getPlatformVersion() const { return platformVersion; }

    std::string build() const {
      std::stringstream out;
      out << hostName << ";" << deviceType << ";" << deviceName;
      return out.str();
    }

    std::string getHeader() const { return "Host;DeviceType;Device"; }

    std::string get() const {
      if (cache.length() != 0) {
        return cache;
      }
      return build();
    }

    void logHeader() { client->getOutputStream() << getHeader() << std::endl; }

    void log() { client->getOutputStream() << get() << std::endl; }

   private:
    ClientPtr client = nullptr;
    std::string hostName;
    cl_device_id device = nullptr;
    std::string deviceName;
    std::string deviceType;
    std::string deviceVendor;
    std::string deviceVersion;
    std::string driverVersion;
    cl_platform_id platform = nullptr;
    std::string platformName;
    std::string platformVendor;
    std::string platformVersion;
    std::string cache;
  };

  class VariantLogger {
   public:
    VariantLogger(ClientPtr client) : client(client) {}
    VariantLogger() : client(nullptr) {}

    std::string getHeader() const { return "Variant"; }

    std::string get(Variant const& variant) const {
      std::stringstream out;
      Variant::const_iterator cit = variant.begin();
      out << (*cit).second;
      while (++cit != variant.end()) {
        out << "-" << (*cit).second;
      }
      return out.str();
    }

    void logHeader() {
      if (client) {
        client->getOutputStream() << getHeader() << std::endl;
      }
    }

    void log(Variant const& variant) {
      if (client) {
        client->getOutputStream() << get(variant) << std::endl;
      }
    }

   private:
    ClientPtr client;
  };

  class VariantExecutionStatisticsLogger {
   public:
    VariantExecutionStatisticsLogger(ClientPtr client) : client(client) {}
    VariantExecutionStatisticsLogger() : client(nullptr) {}

    std::string getHeader() const { return "Min;Max;Median;Mean;Stdev;Var"; }

    std::string get(VariantExecutionStatistics const& ves) const {
      std::stringstream out;
      out << ves.min << ";" << ves.max << ";" << ves.median << ";" << ves.mean
          << ";" << ves.standard_deviation << ";" << ves.variance;
      return out.str();
    }

    void logHeader() {
      if (client) {
        client->getOutputStream() << getHeader() << std::endl;
      }
    }

    void log(VariantExecutionStatistics const& ves) {
      if (client) {
        client->getOutputStream() << get(ves) << std::endl;
      }
    }

   private:
    ClientPtr client;
  };

  class VariantPerformanceLogger {
   public:
    VariantPerformanceLogger(ClientPtr client)
        : client(client), varLogger(client), vesLogger(client) {}

    std::string getHeader() const {
      return varLogger.getHeader() + ";" + vesLogger.getHeader();
    }

    std::string get(VariantPerformance const& vp) const {
      return varLogger.get(vp.first) + ";" + vesLogger.get(vp.second);
    }

    void logHeader() { client->getOutputStream() << getHeader() << std::endl; }

    void log(VariantPerformance const& vp) {
      client->getOutputStream() << get(vp) << std::endl;
    }

   private:
    ClientPtr client;
    VariantLogger varLogger;
    VariantExecutionStatisticsLogger vesLogger;
  };

  class ExplorationLogger {
   public:
    ExplorationLogger(ClientPtr client, cl_device_id device,
                      std::string explorationMode, std::string logPrefix,
                      std::string logHeaderPrefix)
        : client(client),
          deviceLogger(client, device),
          variantPerformanceLogger(client),
          explorationMode(explorationMode),
          logPrefix(logPrefix),
          logHeaderPrefix(logHeaderPrefix) {}

    ExplorationLogger(ClientPtr client, cl_device_id device,
                      std::string explorationMode)
        : client(client),
          deviceLogger(client, device),
          variantPerformanceLogger(client),
          explorationMode(explorationMode),
          logPrefix(""),
          logHeaderPrefix("") {}

    DeviceLogger& getDeviceLogger() { return deviceLogger; }

    VariantPerformanceLogger& getVariantPerformanceLogger() {
      return variantPerformanceLogger;
    }

    std::string getExplorationMode() const { return explorationMode; }

    std::string getHeader() const {
      return deviceLogger.getHeader() + ";ExplorationMode;VariantTag;" +
             variantPerformanceLogger.getHeader();
    }

    std::string get(VariantPerformance const& vp,
                    std::string const& tag) const {
      return deviceLogger.get() + ';' + explorationMode + ';' + tag + ';' +
             variantPerformanceLogger.get(vp);
    }

    void logHeader() {
      client->getOutputStream() << logHeaderPrefix << getHeader() << std::endl;
    }

    void log(VariantPerformance const& vp, std::string const& tag) {
      client->getOutputStream() << logPrefix << get(vp, tag) << std::endl;
    }

   private:
    ClientPtr client;
    DeviceLogger deviceLogger;
    VariantPerformanceLogger variantPerformanceLogger;
    std::string explorationMode;
    std::string logPrefix;
    std::string logHeaderPrefix;
  };

}  // end namespace CogaDB

#endif /* VARIANT_MEASUREMENT_HPP */
