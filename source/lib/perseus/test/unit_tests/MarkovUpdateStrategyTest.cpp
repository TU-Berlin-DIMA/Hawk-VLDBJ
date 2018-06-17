#include "MarkovUpdateStrategy.hpp"
#include <gmock/gmock.h>
#include <limits>
#include <vector>
#include "FeatureMock.hpp"
#include "VariantMock.hpp"
#include "VariantPoolMock.hpp"
#include "VariantScorerMock.hpp"
#include "test_utils.hpp"

#define HIGH_SCORE 0.999
#define LOW_SCORE 0.001

namespace perseus {

class AMarkovUpdateStrategy : public Test {
 private:
  unsigned poolSize = 0;
  std::vector<std::unique_ptr<NiceMock<VariantMock>>> variants;
  NiceMock<VariantGeneratorMock> generator;
  std::shared_ptr<VariantScorerMock> scorer =
      std::make_shared<NiceMock<VariantScorerMock>>();
  std::vector<std::tuple<double, unsigned>> scores;
  double totalScore = 0;

 public:
  NiceMock<VariantPoolMock> pool;

  void SetUp() {
    ON_CALL(pool, reductionFactor()).WillByDefault(Return(1.0));
    ON_CALL(pool, minimumSize()).WillByDefault(Return(2));
    ON_CALL(pool, generator()).WillByDefault(ReturnRef(generator));
    ON_CALL(generator, validateConfiguration(_)).WillByDefault(Return(true));
    ON_CALL(generator, createVariantProxy(_))
        .WillByDefault(ReturnNew<NiceMock<VariantMock>>());
  }

  void chain(double predecessor, double successor) {
    untestedChain();
    addScore(poolSize - 2, predecessor);
    addScore(poolSize - 1, successor);
  }

  void untestedChain() {
    newVariant();
    newVariant();
  }

  void expectUpdate(std::vector<unsigned> indexes) {
    Sequence sequence;
    expectUpdate(indexes, sequence);
  }

  void expectUpdate(std::vector<unsigned> indexes, Sequence sequence) {
    for (auto i = 0u; i < indexes.size(); ++i) {
      auto index = indexes[i];
      auto newVariant = new NiceMock<VariantMock>;
      EXPECT_CALL(generator, createVariantProxy(_))
          .InSequence(sequence)
          .WillOnce(Return(newVariant));
      EXPECT_CALL(pool, updateVariantProxy(index, newVariant))
          .InSequence(sequence);
    }
  }

  void expectSwap(std::vector<unsigned> indexes) {
    for (auto i = 0u; i < indexes.size(); ++i) {
      auto index = indexes[i];
      auto predecessor = 2 * index;
      auto successor = 2 * index + 1;
      EXPECT_CALL(pool, swapVariants(predecessor, successor));
    }
  }

  void expectRemovalInOrder(std::vector<unsigned> indexes, Sequence sequence) {
    for (auto i = 0u; i < indexes.size(); ++i) {
      auto index = indexes[i];
      EXPECT_CALL(pool, removeVariant(index)).InSequence(sequence);
    }
  }

  MarkovUpdateStrategy createStrategy() {
    for (auto& score : scores) {
      std::get<0>(score) /= totalScore;
    }
    ON_CALL(*scorer, scoreVariants(_)).WillByDefault(Return(scores));
    MarkovUpdateStrategy strategy{scorer};
    return strategy;
  }

 private:
  void newVariant() {
    auto variant =
        std::unique_ptr<NiceMock<VariantMock>>(new NiceMock<VariantMock>);
    auto feature = new NiceMock<FeatureMock>();
    ON_CALL(*feature, cloneProxy())
        .WillByDefault(ReturnNew<NiceMock<FeatureMock>>());
    variant->configuration_.addFeature(std::unique_ptr<Feature>(feature));
    ON_CALL(pool, getVariant(poolSize)).WillByDefault(ReturnRef(*variant));
    variants.push_back(std::move(variant));
    pool.setVariants(
        convertUniquePtrElementsToTypedRawPointers<Variant>(variants));
    poolSize += 1;
  }

  void addScore(unsigned index, double score) {
    scores.push_back(std::make_tuple(score, index));
    totalScore += score;
  }
};

TEST_F(AMarkovUpdateStrategy, UpdatesChains) {
  // given
  chain(HIGH_SCORE, LOW_SCORE);
  chain(HIGH_SCORE, LOW_SCORE);
  // then
  expectUpdate({1, 3});
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}

// Chance that test succeeds is about equal to HIGH_SCORE
TEST_F(AMarkovUpdateStrategy, SwapsVariants) {
  // given
  chain(LOW_SCORE, HIGH_SCORE);
  chain(HIGH_SCORE, LOW_SCORE);
  // then
  expectSwap({0});
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}

TEST_F(AMarkovUpdateStrategy, DoesNotUpdateUntestedChains) {
  // given
  chain(HIGH_SCORE, LOW_SCORE);
  untestedChain();
  expectUpdate({1});
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}

TEST_F(AMarkovUpdateStrategy, ReducesPoolSize) {
  // given
  chain(LOW_SCORE, LOW_SCORE);                                  // keep
  chain(HIGH_SCORE, HIGH_SCORE);                                // keep
  chain(LOW_SCORE / 2, LOW_SCORE / 2);                          // remove
  chain(HIGH_SCORE * 2, HIGH_SCORE * 2);                        // keep
  ON_CALL(pool, reductionFactor()).WillByDefault(Return(0.7));  // 1/sqrt(2)
  ON_CALL(pool, minimumSize()).WillByDefault(Return(pool.poolSize() * 0.5));
  // then
  Sequence sequence;
  expectRemovalInOrder({5, 4}, sequence);
  expectUpdate({1, 3, 5}, sequence);
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}

TEST_F(AMarkovUpdateStrategy, DoesNotReduceThePoolSizeBelowMinimum) {
  // given
  chain(LOW_SCORE, LOW_SCORE);
  chain(HIGH_SCORE, HIGH_SCORE);
  chain(LOW_SCORE / 2, LOW_SCORE / 2);
  chain(HIGH_SCORE * 2, HIGH_SCORE * 2);
  ON_CALL(pool, reductionFactor()).WillByDefault(Return(0.7));  // 1/sqrt(2)
  ON_CALL(pool, minimumSize()).WillByDefault(Return(pool.poolSize()));
  // then
  expectUpdate({1, 3, 5, 7});
  // when
  auto strategy = createStrategy();
  strategy.updatePool(pool);
}
}