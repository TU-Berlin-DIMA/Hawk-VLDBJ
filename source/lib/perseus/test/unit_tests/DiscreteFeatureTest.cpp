#include "DiscreteFeature.hpp"
#include <gmock/gmock.h>
#include "utils.h"

using namespace ::testing;

namespace perseus {

typedef enum { ONE, TWO, THREE, UNKNOWN } discrete_t;

class ADiscreteFeature : public Test {
 public:
  const std::string NAME{"name"};
  DiscreteFeature<discrete_t> feature{NAME, {ONE, TWO, THREE}};
};

TEST_F(ADiscreteFeature, HasAName) { ASSERT_THAT(feature.name(), StrEq(NAME)); }

TEST_F(ADiscreteFeature, HasValues) {
  ASSERT_THAT(feature.values(), ElementsAre(ONE, TWO, THREE));
}

TEST_F(ADiscreteFeature, CanBeSetToRandomValue) {
  // given
  std::random_device rd;
  discrete_t expected = ONE;
  while (expected == ONE) {
    const std::mt19937::result_type SEED = rd();
    global_rnd.seed(SEED);
    std::uniform_int_distribution<int> distribution(static_cast<int>(ONE),
                                                    static_cast<int>(THREE));
    expected = static_cast<discrete_t>(distribution(global_rnd));
    global_rnd.seed(SEED);
  }
  // when
  feature.randomize();
  // then
  ASSERT_THAT(feature.value(), expected);
}

TEST_F(ADiscreteFeature, AlwaysHasCorrectRandomValue) {
  for (int i = 0; i < 100; ++i) {
    feature.randomize();
    ASSERT_THAT(feature.value(), AllOf(Ge(ONE), Le(THREE)));
  }
}

TEST_F(ADiscreteFeature, CanBeSetToValue) {
  feature.setValue(TWO);
  ASSERT_THAT(feature.value(), TWO);
}

TEST_F(ADiscreteFeature, ThrowsIfSetToInvalidValue) {
  ASSERT_ANY_THROW(feature.setValue(UNKNOWN));
}

TEST_F(ADiscreteFeature, CanBeCloned) {
  // For some reason, inlining the feature.clone() call creates a
  // a failure. Specifically, the values_ array of the copy is empty.
  auto clone = feature.clone();
  DiscreteFeature<discrete_t>* copy =
      static_cast<DiscreteFeature<discrete_t>*>(clone.get());
  ASSERT_THAT(*copy, feature);
  ASSERT_THAT(copy, Not(&feature));
}

TEST_F(ADiscreteFeature, CanCopyValueFromOtherFeature) {
  auto clone = feature.clone();
  DiscreteFeature<discrete_t>* copy =
      static_cast<DiscreteFeature<discrete_t>*>(clone.get());
  ASSERT_THAT(copy->value(), ONE);
  feature.setValue(TWO);
  copy->copyValue(feature);
  ASSERT_THAT(copy->value(), TWO);
}

TEST_F(ADiscreteFeature, CanAdvanceToNextValue) {
  ASSERT_THAT(feature.nextValue(), true);
  ASSERT_THAT(feature.value(), TWO);
}

TEST_F(ADiscreteFeature, WrapsPastLastValue) {
  feature.setValue(THREE);
  ASSERT_THAT(feature.nextValue(), false);
  ASSERT_THAT(feature.value(), ONE);
}

TEST_F(ADiscreteFeature, CanAdvanceToNextValueWithStep) {
  ASSERT_THAT(feature.nextValue(2), true);
  ASSERT_THAT(feature.value(), THREE);
}

TEST_F(ADiscreteFeature, WrapsPastLastValueWithStep) {
  feature.setValue(TWO);
  ASSERT_THAT(feature.nextValue(2), false);
  ASSERT_THAT(feature.value(), ONE);
}

TEST_F(ADiscreteFeature, WrapsPastLastValueWithOverflow) {
  feature.setValue(THREE);
  ASSERT_THAT(feature.nextValue(2), false);
  ASSERT_THAT(feature.value(), TWO);
}

TEST_F(ADiscreteFeature, CanAdvanceToPreviousValue) {
  feature.setValue(THREE);
  ASSERT_THAT(feature.nextValue(-2), true);
  ASSERT_THAT(feature.value(), ONE);
}

TEST_F(ADiscreteFeature, WrapsPastFirstValue) {
  feature.setValue(TWO);
  ASSERT_THAT(feature.nextValue(-2), false);
  ASSERT_THAT(feature.value(), THREE);
}

TEST_F(ADiscreteFeature, WrapsPastFirstValueWithOverFlow) {
  ASSERT_THAT(feature.nextValue(-2), false);
  ASSERT_THAT(feature.value(), TWO);
}
}
