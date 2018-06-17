#include <util/variant_measurement.hpp>

#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

namespace CoGaDB {

TEST(StatisticsTests, StatisticsTest) {
  std::vector<VariantMeasurement> measurements;
  measurements.emplace_back(true, 1, 0, 0, 0, 0, 0);
  measurements.emplace_back(true, 2, 0, 0, 0, 0, 0);
  measurements.emplace_back(true, 3, 0, 0, 0, 0, 0);
  measurements.emplace_back(true, 4, 0, 0, 0, 0, 0);
  measurements.emplace_back(true, 5, 0, 0, 0, 0, 0);

  VariantExecutionStatistics ves(measurements);

  ASSERT_DOUBLE_EQ(ves.min, 1);
  ASSERT_DOUBLE_EQ(ves.max, 5);
  ASSERT_DOUBLE_EQ(ves.mean, 3);
  ASSERT_DOUBLE_EQ(ves.median, 3);
  ASSERT_DOUBLE_EQ(ves.standard_deviation, 1.4142135623730951);
  ASSERT_DOUBLE_EQ(ves.variance, 2);
}

TEST(StatisticsTests, StatisticsTest2) {
  std::vector<VariantMeasurement> measurements;
  measurements.emplace_back(true, 1, 0, 0, 0, 0, 0);
  measurements.emplace_back(true, 2, 0, 0, 0, 0, 0);
  measurements.emplace_back(true, 3, 0, 0, 0, 0, 0);
  measurements.emplace_back(true, 4, 0, 0, 0, 0, 0);

  VariantExecutionStatistics ves(measurements);

  ASSERT_DOUBLE_EQ(ves.median, 2.5);
}
}
