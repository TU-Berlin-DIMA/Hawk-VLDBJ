#include "Configuration.hpp"
#include <gmock/gmock.h>
#include <stdexcept>
#include "DiscreteFeature.hpp"
#include "FeatureMock.hpp"

using namespace ::testing;

#define VALUES \
  { 1, 2, 3 }

namespace perseus {

class AConfigurationWithIntFeature : public Test {
 public:
  std::string featureName = "intFeature";
  Configuration configuration;
  DiscreteFeature<int>* intFeature =
      new DiscreteFeature<int>{featureName, VALUES};
  std::unique_ptr<Feature> intFeature_ = std::unique_ptr<Feature>(intFeature);

  void SetUp() { configuration.addFeature(std::move(intFeature_)); }
};

class AConfigurationWithTwoFeatures : public AConfigurationWithIntFeature {
 public:
  std::string featureName2 = "intFeature2";
  DiscreteFeature<int>* intFeature2 =
      new DiscreteFeature<int>{featureName2, VALUES};
  std::unique_ptr<Feature> intFeature2_ = std::unique_ptr<Feature>(intFeature2);

  void SetUp() {
    AConfigurationWithIntFeature::SetUp();
    configuration.addFeature(std::move(intFeature2_));
  }

  void testBothFeatures(const int value1, const int value2, const bool next) {
    ASSERT_THAT(configuration.getValue<int>(featureName), value1);
    ASSERT_THAT(configuration.getValue<int>(featureName2), value2);
    ASSERT_THAT(configuration.nextConfiguration(), next);
  }
};

TEST_F(AConfigurationWithIntFeature, HasListOfFeatures) {
  ASSERT_THAT(configuration.features(), ElementsAre(intFeature));
}

TEST_F(AConfigurationWithIntFeature, CanRetrieveFeature) {
  Feature* retrieved = configuration.getFeature(featureName);
  ASSERT_THAT(retrieved, NotNull());
  ASSERT_THAT(retrieved->name(), featureName);
}

TEST_F(AConfigurationWithIntFeature, CanRetrieveTypedFeature) {
  DiscreteFeature<int>* retrieved =
      configuration.getTypedFeature<int>(featureName);
  ASSERT_THAT(*retrieved, *intFeature);
}

TEST_F(AConfigurationWithIntFeature, ReturnsNullForWrongType) {
  ASSERT_THAT(configuration.getTypedFeature<double>(featureName), IsNull());
}

TEST_F(AConfigurationWithIntFeature, ReturnsNullForUnknownFeature) {
  ASSERT_THAT(configuration.getTypedFeature<int>("unknown_feature"), IsNull());
}

TEST_F(AConfigurationWithIntFeature, CanAccessValueDirectly) {
  intFeature->setValue(2);
  ASSERT_THAT(configuration.getValue<int>(featureName), 2);
}

TEST_F(AConfigurationWithIntFeature,
       ThrowsIfTryingToAccessValueOfUnknownFeature) {
  ASSERT_THROW(configuration.getValue<int>("unknown_feature"),
               std::logic_error);
}

TEST_F(AConfigurationWithIntFeature, CannotHaveFeatureOfSameName) {
  std::unique_ptr<Feature> anotherFeature(
      new DiscreteFeature<int>(featureName, {1}));
  ASSERT_THROW(configuration.addFeature(std::move(anotherFeature)),
               std::logic_error);
}

TEST_F(AConfigurationWithTwoFeatures, CanIterateOverFeatures) {
  testBothFeatures(1, 1, true);
  testBothFeatures(2, 1, true);
  testBothFeatures(3, 1, true);
  testBothFeatures(1, 2, true);
  testBothFeatures(2, 2, true);
  testBothFeatures(3, 2, true);
  testBothFeatures(1, 3, true);
  testBothFeatures(2, 3, true);
  testBothFeatures(3, 3, false);
}

TEST_F(AConfigurationWithIntFeature, AssignmentOperatorCopiesFeatures) {
  Configuration copy = configuration;
  ASSERT_THAT(copy.features().size(), configuration.features().size());
  ASSERT_THAT(copy.getValue<int>(featureName),
              configuration.getValue<int>(featureName));
}

TEST_F(AConfigurationWithIntFeature,
       AssignmentOperatorDoesNotDuplicateFeatures) {
  Configuration copy = configuration;
  copy = configuration;
  ASSERT_THAT(copy.features().size(), configuration.features().size());
}
}