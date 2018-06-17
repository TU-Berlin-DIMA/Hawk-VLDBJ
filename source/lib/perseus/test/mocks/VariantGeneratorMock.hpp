#ifndef PERSEUS_VARIANTGENERATORMOCK_HPP
#define PERSEUS_VARIANTGENERATORMOCK_HPP

#include <gmock/gmock.h>
#include "Configuration.hpp"
#include "VariantGenerator.hpp"

namespace perseus {

  class VariantGeneratorMock : public VariantGenerator {
   public:
    virtual std::unique_ptr<Variant> createVariant(
        std::unique_ptr<Configuration> configuration) const {
      return std::unique_ptr<Variant>(createVariantProxy(configuration.get()));
    }

    MOCK_CONST_METHOD1(createVariantProxy, Variant*(const Configuration*));

    MOCK_CONST_METHOD0(features, const std::vector<Feature*>());

    MOCK_CONST_METHOD0(name, const std::string());

    MOCK_CONST_METHOD1(validateConfiguration, const bool(const Configuration&));

    MOCK_CONST_METHOD0(chunkSize, const size_t());
  };
}
#endif  // PERSEUS_VARIANTGENERATORMOCK_HPP
