#ifndef PERSEUS_ITERATIVEUPDATESTRATEGY_HPP
#define PERSEUS_ITERATIVEUPDATESTRATEGY_HPP

#include <algorithm>
#include "Configuration.hpp"
#include "ElitismUpdateStrategy.hpp"
#include "VariantGenerator.hpp"
#include "utils.h"

namespace perseus {

  typedef enum { FORWARD, BACKWARD, RANDOM } direction_t;

  class Chain;

  class FeatureTracker {
    friend class Chain;

   public:
    // expose state for easy testing
    std::string name_;
    unsigned current_ = 0;
    unsigned fastest_ = 0;

   private:
    const unsigned max_;
    const direction_t direction_;

   public:
    FeatureTracker(const Feature& feature, const direction_t direction)
        : name_(feature.name()),
          max_(feature.count() - 1),
          direction_(direction) {}

   private:
    void advanceFeature(Configuration& configuration, signed step = 1);
  };

  class Chain {
   public:
    // expose state for easy testing
    unsigned index_;
    Configuration configuration_;
    Configuration backtrackConfiguration_;
    std::vector<std::string> openFeatures_;
    double initialRuntime_;
    double bestRuntime_;
    std::vector<FeatureTracker> processedFeatures_;
    const direction_t direction_;
    std::uniform_real_distribution<double> directionDistribution_;
    bool chainStarted_ = false;

   public:
    Chain(const unsigned index, const Configuration& configuration,
          const double initialRuntime, const direction_t direction)
        : index_(index),
          configuration_(configuration),
          backtrackConfiguration_(configuration),
          openFeatures_(initializeFeatures()),
          initialRuntime_(initialRuntime),
          bestRuntime_(initialRuntime),
          direction_(direction) {}

    std::unique_ptr<Variant> nextVariant(const VariantGenerator& generator,
                                         const double currentRuntime);

   private:
    std::vector<std::string> initializeFeatures() const;

    const bool firstIteration() const;

    const bool currentFeatureIsFinished() const;

    const bool chainIsFinished() const;

    void initializeRuntime(const double currentRuntime);

    void updateRuntimeOfCurrentFeature(const double currentRuntime);

    const direction_t initializeDirection();

    void startNewFeature();

    void resetCurrentFeature();

    void advanceCurrentFeature();

    std::unique_ptr<Variant> testConfigurationAndCreateVariant(
        const VariantGenerator& generator) const;

    const bool startNewChain(const bool deadlocked);

    void resetFeatureTracker();

    void setNewStartConfiguration();

    void revertToPreviousStartConfiguration();
  };

  class IterativeUpdateStrategy : public ElitismUpdateStrategy {
   private:
    const direction_t direction_;
    std::vector<Chain> chains_;

   public:
    IterativeUpdateStrategy(std::shared_ptr<VariantScorer> variantScorer,
                            const unsigned elitism, direction_t direction)
        : ElitismUpdateStrategy(variantScorer, elitism),
          direction_(direction) {}

    virtual void reset() override;

   private:
    virtual std::vector<std::unique_ptr<Variant>> createNewVariants(
        const VariantPool& pool, size_t count,
        const std::vector<std::tuple<double, unsigned>> variantScores) override;
  };
}

#endif  // PERSEUS_ITERATIVEUPDATESTRATEGY_HPP
