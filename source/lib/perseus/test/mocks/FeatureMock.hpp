#ifndef PERSEUS_FEATUREMOCK_HPP
#define PERSEUS_FEATUREMOCK_HPP

#include <gmock/gmock.h>
#include "Feature.hpp"

namespace perseus {
  class FeatureMock : public Feature {
   public:
    MOCK_CONST_METHOD0(name, const std::string());

    MOCK_METHOD0(randomize, void());

    virtual std::unique_ptr<Feature> clone() const {
      return std::unique_ptr<Feature>(cloneProxy());
    }
    MOCK_CONST_METHOD0(cloneProxy, Feature*());

    MOCK_METHOD1(copyValue, void(const Feature&));

    MOCK_CONST_METHOD0(toString, const std::string());

    MOCK_METHOD1(nextValue, bool(signed step));

    MOCK_CONST_METHOD0(count, const unsigned());
  };
}

#endif  // PERSEUS_FEATUREMOCK_HPP
