#ifndef PERSEUS_VWGREEDYSELECTIONSTRATEGY_HPP
#define PERSEUS_VWGREEDYSELECTIONSTRATEGY_HPP

#include <vector>
#include "SelectionStrategy.hpp"

namespace perseus {

  class VWGreedySelectionStrategy : public SelectionStrategy {
   public:
    // expose state for easier testing
    // having both currentVariant_ and currentIndex_ is a source of bugs
    // currentVariant should be a function
    Variant* currentVariant_ = nullptr;
    int currentIndex_ = -1;
    unsigned int nextExplorePeriod_ = 0;
    unsigned int calls_ = 0;
    unsigned int startMeasurement_ = 0;
    unsigned int nextDecision_ = 0;
    unsigned long long previousTuples_ = 0;
    unsigned long long previousRuntime_ = 0;
    std::vector<double> latestPerformance_;

   private:
    // internal state not required for testing
    const VariantPool* pool_ = nullptr;
    const unsigned int explorePeriod_;
    const unsigned int exploreLength_;
    const unsigned int exploitPeriod_;
    const unsigned int skipLength_;

   public:
    // expose parameter for easier testing
    bool initialExploration_ = true;

   public:
    VWGreedySelectionStrategy(const unsigned int explorePeriod,
                              const unsigned int exploreLength,
                              const unsigned int exploitPeriod,
                              const unsigned int skipLength)
        : VWGreedySelectionStrategy(explorePeriod, exploreLength, exploitPeriod,
                                    skipLength, true) {}

    VWGreedySelectionStrategy(const unsigned int explorePeriod,
                              const unsigned int exploreLength,
                              const unsigned int exploitPeriod,
                              const unsigned int skipLength,
                              const bool initialExploration)
        : explorePeriod_(explorePeriod),
          exploreLength_(exploreLength),
          exploitPeriod_(exploitPeriod),
          skipLength_(skipLength),
          initialExploration_(initialExploration) {}

    virtual void reset(const VariantPool& variantPool,
                       const bool clearPerformance = true) override;

    virtual void finishQuery() override;

    virtual Variant& selectVariant() override;

   private:
    // helper functions
    void verifyInitialization() const;

    bool isInitialExploration() const;

    bool isStartOfExplorationPeriod() const;

    void selectVariantForInitialExploration();

    void selectVariantForExploitation();

    void selectVariantForExploration();

    void startMeasurement();

    void updateMeasurement();
  };
}

#endif  // PERSEUS_VWGREEDYSELECTIONSTRATEGY_HPP
