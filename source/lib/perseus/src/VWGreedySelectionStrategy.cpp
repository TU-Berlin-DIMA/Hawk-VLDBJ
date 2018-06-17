#include "VWGreedySelectionStrategy.hpp"
#include <limits>
#include <random>
#include <stdexcept>
#include "Configuration.hpp"
#include "Variant.hpp"
#include "VariantPool.hpp"
#include "utils.h"

namespace perseus {

void VWGreedySelectionStrategy::reset(const VariantPool& variantPool,
                                      const bool clearPerformance) {
  pool_ = &variantPool;
  currentVariant_ = nullptr;
  calls_ = 0;
  startMeasurement_ = 0;
  nextDecision_ = 0;
  nextExplorePeriod_ =
      initialExploration_
          ? pool_->poolSize() * (skipLength_ + exploreLength_) + explorePeriod_
          : 0;
  // This should be more closely coupled with the update strategy. It works for
  // the current
  // setup because all update strategies except the vw-greedy baseline do an
  // initial
  // exploration and overwrite the contents of this array.
  if (clearPerformance) {
    latestPerformance_.clear();
  }
  latestPerformance_.resize(variantPool.poolSize());
  for (auto variant : variantPool.variants()) {
    variant->reset();
  }
}

Variant& VWGreedySelectionStrategy::selectVariant() {
  verifyInitialization();
  if (calls_ == nextDecision_) {
    updateMeasurement();
    if (isInitialExploration()) {
      selectVariantForInitialExploration();
    } else if (isStartOfExplorationPeriod()) {
      selectVariantForExploration();
    } else {
      selectVariantForExploitation();
    }
    startMeasurement_ = calls_ + skipLength_;
    nextDecision_ += startMeasurement_;
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "vw-greedy: nextDecision_=" << nextDecision_;
    };
  }
  if (calls_ == startMeasurement_) {
    startMeasurement();
  }
  ++calls_;
  return *currentVariant_;
}

bool VWGreedySelectionStrategy::isInitialExploration() const {
  return initialExploration_ &&
         calls_ < pool_->poolSize() * (skipLength_ + exploreLength_);
}

bool VWGreedySelectionStrategy::isStartOfExplorationPeriod() const {
  return calls_ >= nextExplorePeriod_;
}

void VWGreedySelectionStrategy::selectVariantForInitialExploration() {
  unsigned int index = calls_ / (skipLength_ + exploreLength_);
  currentVariant_ = &(pool_->getVariant(index));
  currentIndex_ = index;
  nextDecision_ = exploreLength_;
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "vw-greedy: Selected variant " << index << ": "
                             << currentVariant_ << " "
                             << currentVariant_->configuration()
                             << " for initial exploration";
  };
}

void VWGreedySelectionStrategy::selectVariantForExploration() {
  std::uniform_int_distribution<unsigned int> distribution(
      0, pool_->poolSize() - 1);
  unsigned int index = distribution(global_rnd);
  currentVariant_ = &(pool_->getVariant(index));
  currentIndex_ = index;
  nextDecision_ = exploreLength_;
  nextExplorePeriod_ += explorePeriod_;
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace)
        << "vw-greedy: Selected variant " << index << ": " << currentVariant_
        << " " << currentVariant_->configuration()
        << " for random exploration; nextExplorePeriod_=" << nextExplorePeriod_;
  };
}

void VWGreedySelectionStrategy::selectVariantForExploitation() {
  double bestPerformance = std::numeric_limits<double>::max();
  for (auto i = 0u; i < latestPerformance_.size(); ++i) {
    auto variantPerformance = latestPerformance_[i];
    if (variantPerformance > 0 && variantPerformance < bestPerformance) {
      currentIndex_ = i;
      currentVariant_ = &pool_->getVariant(currentIndex_);
      bestPerformance = variantPerformance;
    }
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "vw-greedy: Runtime of variant " << i << " "
                               << pool_->getVariant(i).configuration() << ": "
                               << variantPerformance;
    };
  }
  nextDecision_ = exploitPeriod_;
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "vw-greedy: Selected variant " << currentIndex_
                             << " " << currentVariant_->configuration()
                             << " for exploitation";
  };
}

void VWGreedySelectionStrategy::verifyInitialization() const {
  if (!pool_) {
    throw std::logic_error("vw-greedy selection strategy is not initialized.");
  }
}

void VWGreedySelectionStrategy::startMeasurement() {
  currentVariant_->waitForLastCall();
  previousRuntime_ = currentVariant_->totalRuntime();
  previousTuples_ = currentVariant_->totalTuples();
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace)
        << "vw-greedy: Starting measurement; previousRuntime_="
        << previousRuntime_ << "; previousTuples_=" << previousTuples_;
  };
}

void VWGreedySelectionStrategy::updateMeasurement() {
  if (currentVariant_) {
    currentVariant_->waitForLastCall();
    auto currentRuntime = currentVariant_->totalRuntime();
    auto currentTuples = currentVariant_->totalTuples();
    double runtimePerTuple = (double)(currentRuntime - previousRuntime_) /
                             (currentTuples - previousTuples_);
    latestPerformance_[currentIndex_] = runtimePerTuple;
    currentVariant_->setCurrentRuntimePerTuple(runtimePerTuple);
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "vw-greedy: Finished measurement of variant "
                               << currentIndex_ << " "
                               << currentVariant_->configuration()
                               << "; currentRuntime=" << currentRuntime
                               << "; currentTuples=" << currentTuples
                               << "; runtimePerTuple=" << runtimePerTuple;
    };
  }
}

void VWGreedySelectionStrategy::finishQuery() {
  if (calls_ > startMeasurement_) {
    updateMeasurement();
  } else if (currentVariant_) {
    currentVariant_->waitForLastCall();
  }
}
}