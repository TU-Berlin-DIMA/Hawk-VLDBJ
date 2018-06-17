#include "RandomizedUpdateStrategy.hpp"
#include <gmock/gmock.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include "CurrentRuntimeVariantScorer.hpp"
#include "RandomizedVariantFactoryMock.hpp"
#include "VariantGeneratorMock.hpp"
#include "VariantMock.hpp"
#include "VariantPoolMock.hpp"
#include "test_utils.hpp"

using namespace ::testing;

#define FASTEST_RUNTIME 10.0

namespace perseus {

class ARandomizedUpdateStrategy : public Test {
 private:
  unsigned poolSize = 0;
  size_t chunkSize = unique<size_t>(1024, 1024 * 1024);
  std::vector<std::unique_ptr<NiceMock<VariantMock>>> variants;
  NiceMock<VariantGeneratorMock> generator;
  std::shared_ptr<VariantScorer> scorer =
      std::make_shared<CurrentRuntimeVariantScorer>();

 public:
  unsigned elitism = 0;
  NiceMock<VariantPoolMock> pool;
  std::unique_ptr<NiceMock<RandomizedVariantFactoryMock>> factory{
      new NiceMock<RandomizedVariantFactoryMock>};

 public:
  void SetUp() {
    ON_CALL(*factory, createRandomVariantProxy(_, _))
        .WillByDefault(ReturnNew<NiceMock<VariantMock>>());
    ON_CALL(pool, chunkSize()).WillByDefault(Return(chunkSize));
    ON_CALL(pool, reductionFactor()).WillByDefault(Return(1.0));
    ON_CALL(pool, generator()).WillByDefault(ReturnRef(generator));
  }

  void fastVariant() {
    elitism += 1;
    auto fastVariantRuntime = unique(1 / 10 * FASTEST_RUNTIME, FASTEST_RUNTIME);
    //            auto fastVariantRuntime = 1.0;
    newVariant(fastVariantRuntime);
  }

  void slowVariant() {
    auto slowVariantRuntime = unique(2 * FASTEST_RUNTIME, 10 * FASTEST_RUNTIME);
    //            auto slowVariantRuntime = 10.0;
    newVariant(slowVariantRuntime);
  }

  void untestedVariant() { newVariant(0.0); }

  void expectRemovalInOrder(std::vector<unsigned> indexes) {
    Sequence sequence;
    for (auto i = 0u; i < indexes.size(); ++i) {
      auto index = indexes[i];
      EXPECT_CALL(pool, removeVariant(index)).InSequence(sequence);
    }
  }

  void expectNewVariants(unsigned count) {
    Sequence sequence;
    for (auto i = 0u; i < count; ++i) {
      auto variant = new NiceMock<VariantMock>;
      EXPECT_CALL(*factory, createRandomVariantProxy(Ref(generator), chunkSize))
          .InSequence(sequence)
          .WillOnce(Return(variant));
      EXPECT_CALL(pool, addVariantProxy(variant));
    }
  }

  RandomizedUpdateStrategy createStrategy() {
    RandomizedUpdateStrategy strategy{scorer, elitism};
    strategy.setRandomizedVariantFactory(std::move(factory));
    return strategy;
  }

 private:
  void newVariant(double runtime) {
    ON_CALL(pool, minimumSize()).WillByDefault(Return(1));
    auto variant =
        std::unique_ptr<NiceMock<VariantMock>>(new NiceMock<VariantMock>);
    ON_CALL(*variant, currentRuntimePerTuple()).WillByDefault(Return(runtime));
    variants.push_back(std::move(variant));
    pool.setVariants(
        convertUniquePtrElementsToTypedRawPointers<Variant>(variants));
    poolSize += 1;
  }
};

TEST_F(ARandomizedUpdateStrategy, RemovesSlowestVariants) {
  // given
  fastVariant();
  slowVariant();
  fastVariant();
  slowVariant();
  slowVariant();
  fastVariant();
  // then
  expectRemovalInOrder({4, 3, 1});
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}

TEST_F(ARandomizedUpdateStrategy, AddsNewRandomVariants) {
  // given
  fastVariant();
  slowVariant();
  fastVariant();
  slowVariant();
  slowVariant();
  fastVariant();
  // then
  expectNewVariants(3);
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}

TEST_F(ARandomizedUpdateStrategy, DoesNotRemoveUntestedVariants) {
  // given
  fastVariant();
  slowVariant();
  fastVariant();
  slowVariant();
  untestedVariant();
  untestedVariant();
  // then
  expectRemovalInOrder({3, 1});
  expectNewVariants(2);
  auto strategy = createStrategy();
  // when
  strategy.updatePool(pool);
}

TEST_F(ARandomizedUpdateStrategy, DoesNotAddVariantsIfNoneWhereRemoved) {
  // given
  fastVariant();
  untestedVariant();
  untestedVariant();
  untestedVariant();
  elitism += 1;
  // then
  expectNewVariants(0);
  auto strategy = createStrategy();
  // when
  strategy.updatePool(pool);
}

TEST_F(ARandomizedUpdateStrategy,
       NewVariantsAreCreatedBeforeOldOnesAreDeleted) {
  // given
  fastVariant();
  slowVariant();
  // then
  {
    Sequence sequence;
    EXPECT_CALL(*factory, createRandomVariantProxy(_, _)).InSequence(sequence);
    EXPECT_CALL(pool, removeVariant(_)).InSequence(sequence);
  }
  auto strategy = createStrategy();
  // when
  strategy.updatePool(pool);
}

// The easiest way to implement this is by sorting the variant array in
// the pool by runtime. However, it doesn't really belong in the
// VariantPool interface but should be done in the UpdateStrategy.
// However, the update strategy does not have access to the pool. Maybe
// the the update strategy can be declared as a friend? Then I could also
// remove the addVariant and removeVariant methods. However, if I do that,
// how do I test for correct execution?
// Alternatively, vw-greedy could evaluate untested variants first.
TEST_F(ARandomizedUpdateStrategy, DISABLED_MovesUntestedVariantsToFront) {
  // given
  fastVariant();
  slowVariant();
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
  // then
  ASSERT_THAT(pool.getVariant(0).currentRuntimePerTuple(), 0.0);
}

TEST_F(ARandomizedUpdateStrategy, ReducesThePoolSize) {
  // given
  fastVariant();
  slowVariant();
  slowVariant();
  slowVariant();
  ON_CALL(pool, reductionFactor()).WillByDefault(Return(0.7));  // 1/sqrt(2)
  ON_CALL(pool, minimumSize()).WillByDefault(Return(pool.poolSize() * 0.5));
  // then
  expectNewVariants(2);
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}

TEST_F(ARandomizedUpdateStrategy, ReducesThePoolSizeButKeepsUntestedVariants) {
  // given
  fastVariant();
  slowVariant();
  slowVariant();
  untestedVariant();
  ON_CALL(pool, reductionFactor()).WillByDefault(Return(0.7));  // 1/sqrt(2)
  ON_CALL(pool, minimumSize()).WillByDefault(Return(pool.poolSize() * 0.5));
  // then
  expectNewVariants(1);
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}

TEST_F(ARandomizedUpdateStrategy, DoesNotReduceThePoolSizeBelowMinimum) {
  // given
  fastVariant();
  slowVariant();
  slowVariant();
  slowVariant();
  ON_CALL(pool, reductionFactor()).WillByDefault(Return(0.7));  // 1/sqrt(2)
  ON_CALL(pool, minimumSize()).WillByDefault(Return(pool.poolSize()));
  // then
  expectNewVariants(3);
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}
}
