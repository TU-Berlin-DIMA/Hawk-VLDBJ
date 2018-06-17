#ifndef PERSEUS_VARIANTMOCK_HPP
#define PERSEUS_VARIANTMOCK_HPP

#include <gmock/gmock.h>
#include "Configuration.hpp"
#include "Variant.hpp"

using namespace ::testing;

namespace perseus {
  class VariantMock : public Variant {
   public:
    // Allow non-const access to configuration object. (configuration()
    // getter returns a const object.)
    Configuration configuration_;

    MOCK_METHOD1(invoke, void(ExecutionContext*));

    MOCK_METHOD0(waitForLastCall, void());

    MOCK_CONST_METHOD0(totalRuntime, const unsigned long long());

    MOCK_CONST_METHOD0(totalTuples, const unsigned long long());

    MOCK_CONST_METHOD0(totalCalls, const unsigned long());

    MOCK_CONST_METHOD0(currentRuntimePerTuple, const double());

    MOCK_METHOD1(setCurrentRuntimePerTuple, void(double));

    MOCK_CONST_METHOD0(name, const std::string());

    MOCK_CONST_METHOD0(configuration, const Configuration&());

    MOCK_METHOD0(reset, void());

    VariantMock() {
      ON_CALL(*this, configuration()).WillByDefault(ReturnRef(configuration_));
    }
  };
}

#endif  // PERSEUS_VARIANTMOCK_HPP
