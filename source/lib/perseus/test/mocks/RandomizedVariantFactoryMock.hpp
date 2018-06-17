#include <gmock/gmock.h>
#include "RandomizedVariantFactory.hpp"

namespace perseus {

  class RandomizedVariantFactoryMock : public RandomizedVariantFactory {
   public:
    virtual std::unique_ptr<Variant> createRandomVariant(
        const VariantGenerator& generator, const size_t chunkSize) const {
      return std::unique_ptr<Variant>(
          createRandomVariantProxy(generator, chunkSize));
    }

    MOCK_CONST_METHOD2(createRandomVariantProxy,
                       Variant*(const VariantGenerator&, const size_t));
  };
}