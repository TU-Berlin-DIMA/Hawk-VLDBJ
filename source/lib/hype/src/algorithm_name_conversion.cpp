

#include <boost/lexical_cast.hpp>
#include <util/algorithm_name_conversion.hpp>

namespace hype {
namespace util {

std::string toInternalAlgName(const std::string& external_alg_name,
                              const core::DeviceSpecification& dev_spec) {
  // add per algorithm a device specific number as sufix, so the algorithm name
  // is unique for each device, but still corresponds to the same "Algorithm
  // class"
  std::string internal_algorithm_name =
      external_alg_name + std::string("_") +
      boost::lexical_cast<std::string>(dev_spec.getProcessingDeviceID());
  return internal_algorithm_name;
}

std::string toExternallAlgName(const std::string& internal_alg_name) {
  // cut of the _DeviceIDnumber suffix
  unsigned int pos = internal_alg_name.find_last_of("_");
  return internal_alg_name.substr(0, pos);  // return the original algorithm
                                            // name, which is independent of the
                                            // device type
}

}  // end namespace util
}  // end namespace hype
