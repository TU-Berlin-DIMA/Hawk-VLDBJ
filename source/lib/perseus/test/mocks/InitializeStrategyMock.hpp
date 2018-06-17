#ifndef PERSEUS_INITIALIZATIONSTRATEGYMOCK_HPP
#define PERSEUS_INITIALIZATIONSTRATEGYMOCK_HPP

#include "InitializationStrategy.hpp"

#include <gmock/gmock.h>

namespace perseus {

  class InitializationStrategyMock : public InitializationStrategy {
   public:
    virtual std::unique_ptr<Configuration> nextConfiguration(
        const std::vector<Feature*>& features,
        const VariantGenerator& generator) const {
      return std::unique_ptr<Configuration>(
          nextConfigurationProxy(features, generator));
    }

    MOCK_CONST_METHOD2(nextConfigurationProxy,
                       Configuration*(const std::vector<Feature*>&,
                                      const VariantGenerator&));

    MOCK_METHOD1(initializePool, void(VariantPool&));
  };
}

#endif  // PERSEUS_INITIALIZATIONSTRATEGYMOCK_HPP
