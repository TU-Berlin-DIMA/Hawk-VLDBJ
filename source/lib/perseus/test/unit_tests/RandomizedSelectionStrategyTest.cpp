#include "RandomizedSelectionStrategy.hpp"
#include <gmock/gmock.h>
#include "VariantMock.hpp"
#include "VariantPoolMock.hpp"
#include "test_utils.hpp"

using namespace ::testing;

namespace perseus {

TEST(RandomizedSelectionStrategy, throwsIfResetWasNeverCalled) {
  // given
  RandomizedSelectionStrategy strategy;
  // then
  ASSERT_THROW(strategy.selectVariant(), std::logic_error);
}

TEST(RandomizedSelectionStrategy, returnsRandomVariantFromPool) {
  // given
  unsigned int poolSize = unique<unsigned int>(1, 10);
  NiceMock<VariantPoolMock> pool;
  ON_CALL(pool, poolSize()).WillByDefault(Return(poolSize));
  unsigned int chosenIndex;
  VariantMock variant;
  EXPECT_CALL(pool, getVariant(_))
      .WillOnce(DoAll(SaveArg<0>(&chosenIndex), ReturnRef(variant)));
  // when
  RandomizedSelectionStrategy strategy;
  strategy.reset(pool);
  // then
  ASSERT_THAT(strategy.selectVariant(), Ref(variant));
  ASSERT_THAT(chosenIndex, AllOf(Ge(0u), Lt(poolSize)));
}
}
