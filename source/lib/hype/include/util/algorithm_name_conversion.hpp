/*
 * File:   algorithm_name_conversion.hpp
 * Author: sebastian
 *
 * Created on 9. September 2013, 13:40
 */

#include <core/specification.hpp>

#ifndef ALGORITHM_NAME_CONVERSION_HPP
#define ALGORITHM_NAME_CONVERSION_HPP

namespace hype {
  namespace util {

    std::string toInternalAlgName(const std::string& external_alg_name,
                                  const core::DeviceSpecification& dev_spec);

    std::string toExternallAlgName(const std::string& internal_alg_name);

  }  // end namespace util
}  // end namespace hype

#endif /* ALGORITHM_NAME_CONVERSION_HPP */
