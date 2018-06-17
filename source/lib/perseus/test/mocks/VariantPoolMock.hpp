#ifndef PERSEUS_VARIANTPOOLMOCK_HPP
#define PERSEUS_VARIANTPOOLMOCK_HPP

#include <gmock/gmock.h>
#include "VariantGenerator.hpp"
#include "VariantGeneratorMock.hpp"
#include "VariantPool.hpp"

using namespace ::testing;

namespace perseus {

  class Variant;

  class VariantPoolMock : public VariantPool {
   private:
    VariantGeneratorMock generator_;

   public:
    VariantPoolMock(std::vector<Variant*> variants) {
      ON_CALL(*this, generator()).WillByDefault(ReturnRef(generator_));
      setVariants(variants);
    }

    void setVariants(std::vector<Variant*> variants) {
      ON_CALL(*this, poolSize()).WillByDefault(Return(variants.size()));
      ON_CALL(*this, variants()).WillByDefault(Return(variants));
      for (auto i = 0u; i < variants.size(); ++i) {
        ON_CALL(*this, getVariant(i)).WillByDefault(ReturnRef(*variants[i]));
      }
    }

    VariantPoolMock() : VariantPoolMock(std::vector<Variant*>{}) {}

    MOCK_CONST_METHOD0(poolSize, const size_t());

    MOCK_CONST_METHOD0(variants, const std::vector<Variant*>());

    MOCK_METHOD0(getVariant, Variant&());

    MOCK_CONST_METHOD1(getVariant, Variant&(unsigned));

    MOCK_METHOD1(removeVariant, void(unsigned));

    virtual void addVariant(std::unique_ptr<Variant> variant) {
      addVariantProxy(variant.get());
    }

    MOCK_METHOD1(addVariantProxy, void(Variant*));

    MOCK_METHOD2(swapVariants, void(const unsigned, const unsigned));

    virtual void updateVariant(const unsigned index,
                               std::unique_ptr<Variant> variant) {
      updateVariantProxy(index, variant.get());
    }

    MOCK_METHOD2(updateVariantProxy, void(const unsigned, Variant*));

    MOCK_CONST_METHOD0(name, const std::string());

    MOCK_CONST_METHOD0(generator, const VariantGenerator&());

    MOCK_CONST_METHOD0(chunkSize, const size_t());

    MOCK_METHOD0(finishQuery, void());

    MOCK_METHOD0(update, void());

    MOCK_CONST_METHOD0(reductionFactor, const double());

    MOCK_CONST_METHOD0(minimumSize, const size_t());

    MOCK_CONST_METHOD0(initialSize, const size_t());

    MOCK_METHOD0(initialize, void());

    MOCK_METHOD0(reset, void());
  };
}

#endif  // PERSEUS_VARIANTPOOLMOCK_HPP
