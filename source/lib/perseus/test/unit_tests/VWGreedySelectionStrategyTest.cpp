#include "VWGreedySelectionStrategy.hpp"
#include <gmock/gmock.h>
#include <random>
#include <stdexcept>
#include <vector>
#include "VariantMock.hpp"
#include "VariantPoolMock.hpp"
#include "test_utils.hpp"
#include "utils.h"

using namespace ::testing;

namespace perseus {

class AVWGreedySelectionStrategy : public Test {
 public:
  unsigned int explorePeriod = 1024;
  unsigned int exploreLength = 4;
  unsigned int exploitPeriod = 256;
  unsigned int skipLength = 2;
  VWGreedySelectionStrategy strategy{explorePeriod, exploreLength,
                                     exploitPeriod, skipLength};
};

class AnInitializedVWGreedyStrategy : public AVWGreedySelectionStrategy {
 public:
  NiceMock<VariantPoolMock> pool;
  unsigned int poolSize = 4;
  std::vector<NiceMock<VariantMock>> variants{poolSize};
  unsigned int startOfFirstExploit = poolSize * (skipLength + exploreLength);

  void SetUp() {
    ON_CALL(pool, poolSize()).WillByDefault(Return(poolSize));
    std::vector<Variant*> variantPtrs(poolSize);
    for (auto i = 0u; i < poolSize; ++i) {
      variantPtrs[i] = &variants[i];
    }
    ON_CALL(pool, variants()).WillByDefault(Return(variantPtrs));
    for (unsigned int i = 0; i < poolSize; ++i) {
      ON_CALL(pool, getVariant(i)).WillByDefault(ReturnRef(variants[i]));
    }
    strategy.reset(pool);
  }

  std::pair<unsigned, VariantMock*> setupCurrentVariant() {
    unsigned int index = unique<unsigned int>(0, poolSize - 1);
    auto& variant = variants[index];
    strategy.currentVariant_ = &variant;
    strategy.currentIndex_ = index;
    return std::make_pair(index, &variant);
  }

  unsigned int atNextDecision() {
    return atNextDecision(unique<unsigned int>(1, 10));
  }

  unsigned int atNextDecision(unsigned int calls) {
    strategy.calls_ = strategy.nextDecision_ = calls;
    return calls;
  }

  unsigned int setupCurrentRuntime(VariantMock& variant,
                                   unsigned int currentRuntime) {
    {
      Sequence sequence;
      EXPECT_CALL(variant, waitForLastCall()).InSequence(sequence);
      EXPECT_CALL(variant, totalRuntime())
          .InSequence(sequence)
          .WillOnce(Return(currentRuntime));
    }
    return currentRuntime;
  }

  unsigned int setupCurrentTuples(VariantMock& variant,
                                  unsigned int currentTuples) {
    EXPECT_CALL(variant, totalTuples()).WillOnce(Return(currentTuples));
    return currentTuples;
  }

  unsigned int repeatableRandomIndex() {
    std::random_device rd;
    const std::mt19937::result_type SEED = rd();
    global_rnd.seed(SEED);
    std::uniform_int_distribution<unsigned int> distribution(0, poolSize - 1);
    unsigned int index = distribution(global_rnd);
    global_rnd.seed(SEED);
    return index;
  };
};

TEST_F(AVWGreedySelectionStrategy, ThrowsIfResetWasNeverCalled) {
  ASSERT_THROW(strategy.selectVariant(), std::logic_error);
}

TEST_F(AnInitializedVWGreedyStrategy, DoesNotThrowAfterReset) {
  // no throw
  strategy.selectVariant();
}

TEST_F(AnInitializedVWGreedyStrategy, IncreasesCallCount) {
  // given
  int calls = unique<unsigned int>(1, 10);
  strategy.calls_ = calls;
  // when
  strategy.selectVariant();
  // then
  ASSERT_THAT(strategy.calls_, calls + 1);
}

TEST_F(AnInitializedVWGreedyStrategy,
       SimplyReturnsCurrentVariantIfNoCalculationRequired) {
  // given
  strategy.calls_ = unique<unsigned int>(1, 10);
  strategy.nextDecision_ = strategy.calls_ + unique<unsigned int>(1, 10);
  auto currentVariant = setupCurrentVariant();
  auto& variant = *std::get<1>(currentVariant);
  // then
  EXPECT_CALL(pool, getVariant(_)).Times(0);
  // when
  ASSERT_THAT(strategy.selectVariant(), Ref(variant));
}

TEST_F(AnInitializedVWGreedyStrategy, ExploresFirstVariantInTheBeginning) {
  // when
  auto& variant = strategy.selectVariant();
  // then
  ASSERT_THAT(variant, Ref(variants[0]));
  ASSERT_THAT(strategy.nextDecision_, skipLength + exploreLength);
  ASSERT_THAT(strategy.currentVariant_, &variant);
  ASSERT_THAT(strategy.currentIndex_, 0);
}

TEST_F(AnInitializedVWGreedyStrategy, ExploresSecondVariantAfterFirst) {
  // given
  strategy.calls_ = skipLength + exploreLength;
  strategy.nextDecision_ = skipLength + exploreLength;
  // when
  auto& variant = strategy.selectVariant();
  // then
  ASSERT_THAT(variant, Ref(variants[1]));
  ASSERT_THAT(strategy.nextDecision_, 2 * (skipLength + exploreLength));
  ASSERT_THAT(strategy.currentVariant_, &variant);
  ASSERT_THAT(strategy.currentIndex_, 1);
}

TEST_F(AnInitializedVWGreedyStrategy, SetsCallToStartMeasuring) {
  // given
  unsigned int calls = atNextDecision();
  // when
  strategy.selectVariant();
  // then
  ASSERT_THAT(strategy.startMeasurement_, calls + skipLength);
}

TEST_F(AnInitializedVWGreedyStrategy,
       SavesCurrentVariantStatisticsAtMeasurementStart) {
  // given
  strategy.calls_ = strategy.startMeasurement_ = unique<unsigned int>(1, 10);
  auto currentVariant = setupCurrentVariant();
  auto& variant = *std::get<1>(currentVariant);
  unsigned int currentTuples = setupCurrentTuples(variant, 20);
  unsigned int currentRuntime = setupCurrentRuntime(variant, 10);
  // when
  strategy.selectVariant();
  // then
  ASSERT_THAT(strategy.previousTuples_, currentTuples);
  ASSERT_THAT(strategy.previousRuntime_, currentRuntime);
}

TEST_F(AnInitializedVWGreedyStrategy,
       MeasuresCurrentVariantPerformanceBeforeSwitchingToNextVariant) {
  // given
  atNextDecision();
  auto currentVariant = setupCurrentVariant();
  auto index = std::get<0>(currentVariant);
  auto& variant = *std::get<1>(currentVariant);
  unsigned int previousTuples = 20;
  unsigned int previousRuntime = 10;
  strategy.previousTuples_ = previousTuples;
  strategy.previousRuntime_ = previousRuntime;
  unsigned int currentTuples = setupCurrentTuples(variant, 120);
  unsigned int currentRuntime = setupCurrentRuntime(variant, 100);
  auto currentRuntimePerTuple = (double)(currentRuntime - previousRuntime) /
                                (currentTuples - previousTuples);
  // then
  EXPECT_CALL(variant, setCurrentRuntimePerTuple(currentRuntimePerTuple));
  // when
  strategy.selectVariant();
  // and
  ASSERT_THAT(strategy.latestPerformance_[index], currentRuntimePerTuple);
}

TEST_F(AnInitializedVWGreedyStrategy, ExploitsVariantAfterInitialExploration) {
  // given
  atNextDecision(startOfFirstExploit);
  unsigned int fastest = unique<unsigned int>(1, poolSize - 1);
  double slow = 5;
  double fast = 1;
  for (unsigned int i = 0; i < poolSize; ++i) {
    strategy.latestPerformance_[i] = i == fastest ? fast : slow;
  }
  // when
  auto& variant = strategy.selectVariant();
  // then
  ASSERT_THAT(variant, Ref(variants[fastest]));
  ASSERT_THAT(strategy.currentVariant_, &variant);
  ASSERT_THAT(strategy.currentIndex_, fastest);
  ASSERT_THAT(strategy.nextDecision_,
              startOfFirstExploit + exploitPeriod + skipLength);
}

TEST_F(AnInitializedVWGreedyStrategy,
       StartsNewExplorationPeriodAfterExplorePeriodCalls) {
  // given
  atNextDecision(startOfFirstExploit + explorePeriod);
  unsigned int index = repeatableRandomIndex();
  EXPECT_CALL(pool, getVariant(index));
  // when
  auto& variant = strategy.selectVariant();
  // then
  ASSERT_THAT(variant, Ref(variants[index]));
  ASSERT_THAT(strategy.currentVariant_, &variant);
  ASSERT_THAT(strategy.currentIndex_, index);
  ASSERT_THAT(strategy.nextDecision_,
              startOfFirstExploit + explorePeriod + exploreLength + skipLength);
  ASSERT_THAT(strategy.nextExplorePeriod_,
              startOfFirstExploit + 2 * explorePeriod);
}

TEST_F(AnInitializedVWGreedyStrategy, ResetsInternalStateWhenResetWithNewPool) {
  // given
  strategy.currentVariant_ = &variants[0];
  strategy.calls_ = unique<unsigned int>(1, 10);
  strategy.startMeasurement_ = unique<unsigned int>(1, 10);
  strategy.nextDecision_ = unique<unsigned int>(1, 10);
  strategy.nextExplorePeriod_ = unique<unsigned int>(1, 10);
  strategy.latestPerformance_[1] = unique<unsigned int>(1, 10);
  // then
  EXPECT_CALL(variants[0], reset());
  // when
  strategy.reset(pool);
  // then
  ASSERT_THAT(strategy.currentVariant_, nullptr);
  ASSERT_THAT(strategy.currentIndex_, -1);
  ASSERT_THAT(strategy.calls_, 0);
  ASSERT_THAT(strategy.startMeasurement_, 0);
  ASSERT_THAT(strategy.nextDecision_, 0);
  ASSERT_THAT(strategy.nextExplorePeriod_, startOfFirstExploit + explorePeriod);
  ASSERT_THAT(strategy.latestPerformance_.size(), poolSize);
}

// startMeasurement is called when calls_ = startMeasurement, then calls is
// incremented
TEST_F(AnInitializedVWGreedyStrategy, UpdatesMeasurementsWhenQueryIsFinished) {
  // given
  strategy.currentVariant_ = &variants[0];
  strategy.currentIndex_ = 0;
  strategy.calls_ = unique<unsigned>(2, 10);
  strategy.startMeasurement_ = strategy.calls_ - 1;
  // then
  EXPECT_CALL(variants[0], setCurrentRuntimePerTuple(_));
  // when
  strategy.finishQuery();
}

// startMeasurement is called when calls_ = startMeasurement, then calls is
// incremented
TEST_F(AnInitializedVWGreedyStrategy,
       DoesNotUpdateMeasurementsWhenQueryIsFinishedBeforeMeasurementStarted) {
  // given
  strategy.currentVariant_ = &variants[0];
  strategy.calls_ = unique<unsigned>(1, 10);
  strategy.startMeasurement_ = strategy.calls_;
  // then
  EXPECT_CALL(variants[0], setCurrentRuntimePerTuple(_)).Times(0);
  EXPECT_CALL(variants[0], waitForLastCall());
  // when
  strategy.finishQuery();
}

TEST_F(AnInitializedVWGreedyStrategy, SkipsInitialExploration) {
  // given
  strategy.initialExploration_ = false;
  strategy.reset(pool);
  atNextDecision(0);
  unsigned int index = repeatableRandomIndex();
  EXPECT_CALL(pool, getVariant(index));
  // when
  auto& variant = strategy.selectVariant();
  // then
  ASSERT_THAT(variant, Ref(variants[index]));
  ASSERT_THAT(strategy.currentVariant_, &variant);
  ASSERT_THAT(strategy.currentIndex_, index);
  ASSERT_THAT(strategy.nextDecision_, exploreLength + skipLength);
  ASSERT_THAT(strategy.nextExplorePeriod_, explorePeriod);
}

TEST_F(AnInitializedVWGreedyStrategy, DoesNotExploitUnexploredVariant) {
  // given
  atNextDecision(startOfFirstExploit);
  unsigned int fastest = unique<unsigned int>(1, poolSize - 1);
  double unexplored = 0;
  double fast = 1;
  for (unsigned int i = 0; i < poolSize; ++i) {
    strategy.latestPerformance_[i] = i == fastest ? fast : unexplored;
  }
  // when
  auto& variant = strategy.selectVariant();
  // then
  ASSERT_THAT(variant, Ref(variants[fastest]));
  ASSERT_THAT(strategy.currentVariant_, &variant);
  ASSERT_THAT(strategy.currentIndex_, fastest);
  ASSERT_THAT(strategy.nextDecision_,
              startOfFirstExploit + exploitPeriod + skipLength);
}
}
