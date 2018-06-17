#include "IterativeUpdateStrategy.hpp"
#include <gmock/gmock.h>
#include <memory>
#include <string>
#include "DiscreteFeature.hpp"
#include "VariantGeneratorMock.hpp"
#include "VariantMock.hpp"

using namespace ::testing;

namespace perseus {

class AChainTest : public Test {};

TEST_F(AChainTest, UpdatesFeatureIfRuntimeIsBetter) {
  auto index = 0u;
  Configuration configuration;
  auto initialRuntime = 4.0;
  std::string name = "feature";
  auto feature = new DiscreteFeature<int>{name, {1, 2}};
  configuration.addFeature(std::unique_ptr<Feature>(feature));
  Chain chain(index, configuration, initialRuntime, FORWARD);
  chain.processedFeatures_.push_back({*feature, FORWARD});
  auto& tracker = chain.processedFeatures_[0];
  tracker.current_ = 10;
  tracker.name_ = name;
  NiceMock<VariantGeneratorMock> generator;
  auto newRuntime = 1.0;
  ON_CALL(generator, validateConfiguration(_)).WillByDefault(Return(true));
  ON_CALL(generator, createVariantProxy(_))
      .WillByDefault(ReturnNew<NiceMock<VariantMock>>());
  chain.nextVariant(generator, newRuntime);
  ASSERT_THAT(chain.bestRuntime_, newRuntime);
  ASSERT_THAT(tracker.fastest_, 10);
}
}