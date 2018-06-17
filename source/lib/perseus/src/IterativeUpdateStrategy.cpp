#include "IterativeUpdateStrategy.hpp"
#include <random>
#include "Configuration.hpp"

namespace perseus {

void FeatureTracker::advanceFeature(Configuration& configuration, signed step) {
  if (direction_ == BACKWARD) {
    step = -step;
  }
  auto feature = configuration.getFeature(name_);
  feature->nextValue(step);
}

std::unique_ptr<Variant> Chain::nextVariant(const VariantGenerator& generator,
                                            const double currentRuntime) {
  if (!firstIteration()) {
    updateRuntimeOfCurrentFeature(currentRuntime);
  }
  if (chainStarted_) {
    initializeRuntime(currentRuntime);
  }
  auto newVariant = std::unique_ptr<Variant>(nullptr);
  auto deadlocked = false;
  while (!newVariant) {
    if (chainIsFinished()) {
      deadlocked = startNewChain(deadlocked);
      newVariant = testConfigurationAndCreateVariant(generator);
    } else {
      if (firstIteration()) {
        startNewFeature();
      } else if (currentFeatureIsFinished()) {
        resetCurrentFeature();
        startNewFeature();
      }
      advanceCurrentFeature();
      newVariant = testConfigurationAndCreateVariant(generator);
    }
  }
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "iterative: New variant for chain " << index_
                             << ": " << configuration_;
  };
  return newVariant;
}

std::vector<std::string> Chain::initializeFeatures() const {
  auto features = configuration_.features();
  features.erase(std::remove_if(features.begin(), features.end(),
                                [](Feature* f) { return f->count() == 1; }),
                 features.end());
  std::vector<std::string> openFeatures;
  for (auto feature : features) {
    openFeatures.push_back(feature->name());
  }
  std::shuffle(openFeatures.begin(), openFeatures.end(), global_rnd);
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "iterative: order of features: "
                             << join(openFeatures, ", ");
  };
  return openFeatures;
}

const bool Chain::firstIteration() const { return processedFeatures_.empty(); }

const bool Chain::currentFeatureIsFinished() const {
  auto tracker = processedFeatures_.back();
  return tracker.current_ == tracker.max_;
}

const bool Chain::chainIsFinished() const {
  return openFeatures_.empty() && currentFeatureIsFinished();
}

const direction_t Chain::initializeDirection() {
  if (direction_ == RANDOM) {
    return directionDistribution_(global_rnd) < 0.5 ? FORWARD : BACKWARD;
  } else {
    return direction_;
  }
}

void Chain::startNewFeature() {
  auto name = openFeatures_.back();
  openFeatures_.pop_back();
  auto direction = initializeDirection();
  auto feature = configuration_.getFeature(name);
  FeatureTracker tracker(*feature, direction);
  processedFeatures_.push_back(tracker);
  bestRuntime_ = initialRuntime_;
}

void Chain::resetCurrentFeature() {
  auto& tracker = processedFeatures_.back();
  tracker.advanceFeature(configuration_);
}

void Chain::initializeRuntime(const double currentRuntime) {
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "iterative: initial runtime of chain " << index_
                             << ": " << currentRuntime;
  };
  bestRuntime_ = currentRuntime;
  initialRuntime_ = currentRuntime;
  chainStarted_ = false;
}

void Chain::updateRuntimeOfCurrentFeature(const double currentRuntime) {
  if (currentRuntime < bestRuntime_) {
    auto& tracker = processedFeatures_.back();
    auto feature = configuration_.getFeature(tracker.name_);
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "iterative: bestRuntime_=" << bestRuntime_
                               << "; currentRuntime=" << currentRuntime << "; "
                               << feature->toString();
    };
    tracker.fastest_ = tracker.current_;
    bestRuntime_ = currentRuntime;
  }
}

void Chain::advanceCurrentFeature() {
  auto& tracker = processedFeatures_.back();
  tracker.advanceFeature(configuration_);
  tracker.current_ += 1;
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "iterative: Modifying feature of chain "
                             << index_ << ": " << tracker.name_;
  }
}

std::unique_ptr<Variant> Chain::testConfigurationAndCreateVariant(
    const VariantGenerator& generator) const {
  if (generator.validateConfiguration(configuration_)) {
    return generator.createVariant(
        std::unique_ptr<Configuration>(new Configuration(configuration_)));
  } else {
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "iterative: Variant not viable: "
                               << configuration_;
    };
    return nullptr;
  }
}

const bool Chain::startNewChain(const bool deadlocked) {
  chainStarted_ = true;
  if (deadlocked) {
    revertToPreviousStartConfiguration();
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace)
          << "iterative: Chain " << index_
          << " is deadlocked; reverting to configuration: " << configuration_;
    };
    return false;
  } else {
    setNewStartConfiguration();
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace)
          << "iterative: Chain " << index_
          << " is finished; new starting configuration: " << configuration_;
    };
    return true;
  }
}

void Chain::resetFeatureTracker() {
  for (auto& tracker : processedFeatures_) {
    tracker.advanceFeature(configuration_, tracker.fastest_);
  }
  openFeatures_ = initializeFeatures();
  processedFeatures_.clear();
}

void Chain::setNewStartConfiguration() {
  advanceCurrentFeature();
  backtrackConfiguration_ = configuration_;
  resetFeatureTracker();
}

void Chain::revertToPreviousStartConfiguration() {
  configuration_ = backtrackConfiguration_;
  resetFeatureTracker();
}

void IterativeUpdateStrategy::reset() { chains_.clear(); }

std::vector<std::unique_ptr<Variant>>
IterativeUpdateStrategy::createNewVariants(
    const VariantPool& pool, size_t count,
    const std::vector<std::tuple<double, unsigned>> variantScores) {
  std::vector<std::unique_ptr<Variant>> newVariants;
  if (chains_.empty()) {
    for (auto i = 0u; i < count; ++i) {
      auto index = std::get<1>(variantScores[i]);
      auto& variant = pool.getVariant(index);
      auto runtime = (double)variant.totalRuntime() / variant.totalTuples();
      auto configuration = variant.configuration();
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace) << "iterative: Starting position of chain "
                                 << i << " is variant " << index << ", "
                                 << configuration;
      };
      Chain chain(i, configuration, runtime, direction_);
      auto newVariant = chain.nextVariant(pool.generator(), runtime);
      newVariants.push_back(std::move(newVariant));
      chains_.push_back(chain);
    }
  } else {
    for (auto i = 0u; i < count; ++i) {
      auto& chain = chains_[i];
      auto& variant = pool.getVariant(i + elitism());
      auto runtime = (double)variant.totalRuntime() / variant.totalTuples();
      auto newVariant = chain.nextVariant(pool.generator(), runtime);
      newVariants.push_back(std::move(newVariant));
    }
  }
  return newVariants;
}
}