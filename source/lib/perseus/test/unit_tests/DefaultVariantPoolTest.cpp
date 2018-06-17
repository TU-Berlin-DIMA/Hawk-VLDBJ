#include "DefaultVariantPool.hpp"
#include <gmock/gmock.h>
#include "Configuration.hpp"
#include "InitializeStrategyMock.hpp"
#include "SelectionStrategyMock.hpp"
#include "UpdateStrategyMock.hpp"
#include "Variant.hpp"
#include "VariantGeneratorMock.hpp"
#include "VariantMock.hpp"

using namespace ::testing;

namespace perseus {

class ADefaultVariantPool : public testing::Test {
 private:
  std::unique_ptr<VariantGeneratorMock> generator_{new VariantGeneratorMock};
  std::unique_ptr<NiceMock<InitializationStrategyMock>> initializationStrategy_{
      new NiceMock<InitializationStrategyMock>};
  std::unique_ptr<NiceMock<SelectionStrategyMock>> selectionStrategy_{
      new NiceMock<SelectionStrategyMock>};
  std::unique_ptr<NiceMock<UpdateStrategyMock>> updateStrategy_{
      new NiceMock<UpdateStrategyMock>};

 public:
  size_t chunkSize = 1024;
  VariantMock variant;
  std::vector<Variant*> variants{&variant};
  VariantGeneratorMock& generator = *generator_;
  NiceMock<InitializationStrategyMock>& initializationStrategy =
      *initializationStrategy_;
  NiceMock<SelectionStrategyMock>& selectionStrategy = *selectionStrategy_;
  NiceMock<UpdateStrategyMock>& updateStrategy = *updateStrategy_;
  DefaultVariantPool pool{variants.size(),
                          chunkSize,
                          std::move(generator_),
                          std::move(initializationStrategy_),
                          std::move(selectionStrategy_),
                          std::move(updateStrategy_)};

  void SetUp() {
    ON_CALL(selectionStrategy, selectVariant())
        .WillByDefault(ReturnRef(variant));
  }
};

TEST_F(ADefaultVariantPool,
       DelegatesPoolInitializationAndResetsSelectionStrategy) {
  Sequence sequence;
  EXPECT_CALL(initializationStrategy, initializePool(Ref(pool)))
      .InSequence(sequence);
  EXPECT_CALL(selectionStrategy, reset(Ref(pool), _)).InSequence(sequence);
  pool.initialize();
}

TEST_F(ADefaultVariantPool, UpdatesPool) {
  Sequence sequence;
  EXPECT_CALL(selectionStrategy, finishQuery()).InSequence(sequence);
  EXPECT_CALL(updateStrategy, updatePool(Ref(pool))).InSequence(sequence);
  EXPECT_CALL(selectionStrategy, reset(Ref(pool), _)).InSequence(sequence);
  pool.update();
}

TEST_F(ADefaultVariantPool, JustFinishesCurrentVariant) {
  EXPECT_CALL(selectionStrategy, finishQuery());
  EXPECT_CALL(updateStrategy, updatePool(Ref(pool))).Times(0);
  EXPECT_CALL(selectionStrategy, reset(Ref(pool), _)).Times(0);
  pool.finishQuery();
}

TEST_F(ADefaultVariantPool, CanSwapVariants) {
  // given
  pool.addVariant(std::unique_ptr<Variant>(new VariantMock));
  pool.addVariant(std::unique_ptr<Variant>(new VariantMock));
  auto& variant1 = pool.getVariant(0);
  auto& variant2 = pool.getVariant(1);
  // when
  pool.swapVariants(0, 1);
  // then
  ASSERT_THAT(pool.getVariant(0), Ref(variant2));
  ASSERT_THAT(pool.getVariant(1), Ref(variant1));
}

TEST_F(ADefaultVariantPool, CanUpdateVariant) {
  // given
  pool.addVariant(std::unique_ptr<Variant>(new VariantMock));
  std::unique_ptr<Variant> variantPtr(new VariantMock);
  auto& variant = *variantPtr;
  // when
  pool.updateVariant(0, std::move(variantPtr));
  // then
  ASSERT_THAT(pool.getVariant(0), Ref(variant));
}
}