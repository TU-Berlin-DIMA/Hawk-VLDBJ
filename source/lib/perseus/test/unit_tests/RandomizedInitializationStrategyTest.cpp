#include "RandomizedInitializationStrategy.hpp"
#include <gmock/gmock.h>
#include <unordered_map>
#include "Configuration.hpp"
#include "FeatureMock.hpp"
#include "VariantGeneratorMock.hpp"

using namespace ::testing;

namespace perseus {

class ARandomizedInitializationStrategy : public Test {
 public:
  RandomizedInitializationStrategy randomizedInitializationStrategy;
  NiceMock<VariantGeneratorMock> variantGenerator;
  NiceMock<FeatureMock> feature;
  // owned by feature through cloneProxy
  NiceMock<FeatureMock>* clone = new NiceMock<FeatureMock>;
  std::vector<Feature*> features{&feature};

  void SetUp() {
    ON_CALL(feature, cloneProxy()).WillByDefault(Return(clone));
    ON_CALL(variantGenerator, validateConfiguration(_))
        .WillByDefault(Return(true));
  }
};

TEST_F(ARandomizedInitializationStrategy, ClonesTheFeaturesFromGenerator) {
  EXPECT_CALL(feature, cloneProxy()).WillRepeatedly(DoDefault());
  auto configuration = randomizedInitializationStrategy.nextConfiguration(
      features, variantGenerator);
  ASSERT_THAT(configuration->features(), ElementsAre(clone));
}

TEST_F(ARandomizedInitializationStrategy, RandomizesClone) {
  EXPECT_CALL(*clone, randomize());
  auto configuration = randomizedInitializationStrategy.nextConfiguration(
      features, variantGenerator);
}

TEST_F(ARandomizedInitializationStrategy, OnlyReturnsValidFeatures) {
  EXPECT_CALL(variantGenerator, validateConfiguration(_))
      .WillOnce(Return(false))
      .WillOnce(Return(true));
  EXPECT_CALL(*clone, randomize()).Times(2);
  randomizedInitializationStrategy.nextConfiguration(features,
                                                     variantGenerator);
}
}