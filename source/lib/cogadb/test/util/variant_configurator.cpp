#include <util/variant_configurator.hpp>

#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

namespace CoGaDB {

TEST(DimensionTests, IterationTest) {
  std::vector<std::string> values = {"A", "B", "C"};
  const std::string dim_name = "test_dim";
  VariantIterator::Dimension dimension(dim_name, values, 0);
  ASSERT_EQ(std::distance(dimension.begin(), dimension.end()),
            int(values.size()));

  auto itr = dimension.begin();
  ASSERT_EQ(itr->size(), 1ul);
  ASSERT_EQ(itr->begin()->first, dim_name);
  ASSERT_EQ(itr->begin()->second, "A");

  std::advance(itr, 1);
  ASSERT_EQ(itr->size(), 1ul);
  ASSERT_EQ(itr->begin()->first, dim_name);
  ASSERT_EQ(itr->begin()->second, "B");

  std::advance(itr, 1);
  ASSERT_EQ(itr->size(), 1ul);
  ASSERT_EQ(itr->begin()->first, dim_name);
  ASSERT_EQ(itr->begin()->second, "C");
}

TEST(DimensionTests, IterationTest2) {
  std::vector<std::string> values = {"A"};
  const std::string dim_name = "test_dim";
  VariantIterator::Dimension dimension(dim_name, values, 0);
  ASSERT_EQ(std::distance(dimension.begin(), dimension.end()),
            int(values.size()));

  auto itr = dimension.begin();
  ASSERT_EQ(itr->size(), 1ul);
  ASSERT_EQ(itr->begin()->first, dim_name);
  ASSERT_EQ(itr->begin()->second, "A");
}

TEST(DimensionTests, IterationTest3) {
  std::vector<std::string> values = {"A", "B", "C"};
  std::vector<std::string> values_with_childs = {"A", "C"};
  std::vector<std::string> child_values = {"A"};
  const std::string child_dim_name = "test_dim_child";
  const std::string dim_name = "test_dim";
  VariantIterator::Dimension dimension(dim_name, values, 0);
  ASSERT_EQ(int(values.size()),
            std::distance(dimension.begin(), dimension.end()));

  dimension.addChilds(values_with_childs, child_dim_name, child_values);
  ASSERT_EQ(int(values_with_childs.size() * child_values.size() +
                values.size() - values_with_childs.size()),
            std::distance(dimension.begin(), dimension.end()));

  auto itr = dimension.begin();
  ASSERT_EQ(itr->size(), 2ul);
  ASSERT_NE(itr->find("test_dim"), itr->end());
  ASSERT_NE(itr->find("test_dim_child"), itr->end());

  std::advance(itr, 1);
  ASSERT_EQ(itr->size(), 1ul);
  ASSERT_EQ(itr->begin()->first, dim_name);
  ASSERT_EQ(itr->begin()->second, "B");

  std::advance(itr, 1);
  ASSERT_EQ(itr->size(), 2ul);
  ASSERT_NE(itr->find("test_dim"), itr->end());
  ASSERT_NE(itr->find("test_dim_child"), itr->end());
}

TEST(DimensionTests, MultipleChildValues) {
  std::vector<std::string> values = {"A", "B", "C"};
  std::vector<std::string> values_with_childs = {"A", "C"};
  std::vector<std::string> child_values = {"A", "B"};
  const std::string child_dim_name = "test_dim_child";
  const std::string dim_name = "test_dim";
  VariantIterator::Dimension dimension(dim_name, values, 0);
  ASSERT_EQ(std::distance(dimension.begin(), dimension.end()),
            int(values.size()));

  dimension.addChilds(values_with_childs, child_dim_name, child_values);
  ASSERT_EQ(std::distance(dimension.begin(), dimension.end()),
            int(values_with_childs.size() * child_values.size() +
                values.size() - values_with_childs.size()));
}

TEST(BlacklistTests, BlacklistTest) {
  VariantBlacklist blacklist;
  blacklist.add("abc", "def");
  ASSERT_EQ(blacklist.contains("abc", "def"), true);
  blacklist.add("abc", "ghi");
  ASSERT_EQ(blacklist.contains("abc", "def"), true);
  ASSERT_EQ(blacklist.contains("abc", "ghi"), true);
  ASSERT_EQ(blacklist.contains("jkl", "mno"), false);
}

TEST(BlacklistTests, BlacklistTest2) {
  VariantBlacklist blacklist;
  blacklist.add("1", "A");
  blacklist.add("2", "C");

  Variant v1 = {{"1", "A"}, {"2", "A"}};
  Variant v2 = {{"1", "B"}, {"2", "C"}};
  Variant v3 = {{"1", "A"}, {"2", "C"}};
  Variant v4 = {{"1", "D"}, {"2", "D"}};

  ASSERT_EQ(blacklist.is_blacklisted(v1), true);
  ASSERT_EQ(blacklist.is_blacklisted(v2), true);
  ASSERT_EQ(blacklist.is_blacklisted(v3), true);
  ASSERT_EQ(blacklist.is_blacklisted(v4), false);
}

TEST(VariantIteratorTests, DefaultCtrTest) {
  VariantIterator vit;
  ASSERT_EQ(std::distance(vit.begin(), vit.end()), 0);
}

TEST(VariantIteratorTests, IterationTest) {
  std::string dimension1 = "1";
  std::string dimension2 = "2";
  std::vector<std::string> values1 = {"A", "B"};
  std::vector<std::string> values2 = {"C"};
  VariantIterator vit;
  vit.add(dimension1, values1);
  vit.add(dimension2, values2);
  ASSERT_EQ(std::distance(vit.begin(), vit.end()), 2);
  Variant first;
  first.insert({"1", "A"});
  first.insert({"2", "C"});
  Variant second;
  second.insert({"1", "B"});
  second.insert({"2", "C"});

  auto itr = vit.begin();

  ASSERT_TRUE(itr != vit.end());
  ASSERT_EQ(*itr, first);

  std::advance(itr, 1);
  ASSERT_TRUE(itr != vit.end());
  ASSERT_EQ(*itr, second);

  std::advance(itr, 1);
  ASSERT_TRUE(itr == vit.end());
}

TEST(VariantIteratorTests, IterationTest2) {
  std::string dimension1 = "1";
  std::string dimension2 = "2";
  std::vector<std::string> values1 = {"A", "B"};
  std::vector<std::string> values2 = {"C"};

  std::string values2_child_dim_name = "3";
  std::vector<std::string> values2_childs = {"D", "E"};
  VariantIterator vit;
  vit.add(dimension1, values1);
  vit.add(dimension2, values2)
      .addChilds(values2, values2_child_dim_name, values2_childs);
  ASSERT_EQ(std::distance(vit.begin(), vit.end()), 4);
}

TEST(VariantIteratorTests, SortDimensionTest) {
  std::string dimension1 = "1";
  std::string dimension2 = "2";
  std::string dimension3 = "3";
  std::string dimension4 = "4";
  std::vector<std::string> values = {"A"};

  VariantIterator vit;
  vit.add(dimension1, values, 4);
  vit.add(dimension2, values, 3);
  vit.add(dimension3, values, 2);
  vit.add(dimension4, values, 1);

  auto dims = vit.getSortedDimensions();

  ASSERT_EQ(dims.size(), 4);

  ASSERT_EQ(dims[0].getName(), dimension1);
  ASSERT_EQ(dims[1].getName(), dimension2);
  ASSERT_EQ(dims[2].getName(), dimension3);
  ASSERT_EQ(dims[3].getName(), dimension4);
}

TEST(VariantIteratorTests, SortDimensionTest2) {
  std::string dimension1 = "1";
  std::string dimension2 = "2";
  std::string dimension3 = "3";
  std::string dimension4 = "4";
  std::string dimension5 = "5";
  std::vector<std::string> values = {"A"};

  VariantIterator vit;
  vit.add(dimension1, values, 4);
  vit.add(dimension2, values, 3);
  vit.add(dimension3, values, 2);
  vit.add(dimension4, values, 1);

  auto dims = vit.getSortedDimensions();

  ASSERT_EQ(dims.size(), 4);

  vit.add(dimension5, values);

  dims = vit.getSortedDimensions();

  ASSERT_EQ(dims.size(), 5);

  ASSERT_EQ(dims[0].getName(), dimension1);
  ASSERT_EQ(dims[1].getName(), dimension2);
  ASSERT_EQ(dims[2].getName(), dimension3);
  ASSERT_EQ(dims[3].getName(), dimension4);
  ASSERT_EQ(dims[4].getName(), dimension5);
}

TEST(VariantIteratorTests, FlattedDimensionTest) {
  std::string dimension1 = "1";
  std::string dimension2 = "2";
  std::string dimension3 = "3";
  std::string dimension4 = "4";
  std::vector<std::string> values = {"A"};

  VariantIterator vit;
  vit.add(dimension1, values, 4);
  vit.add(dimension2, values, 3);
  vit.add(dimension3, values, 2);
  vit.add(dimension4, values, 1);

  auto dims = vit.getFlattenedDimensions();

  ASSERT_EQ(dims.size(), 4);

  ASSERT_EQ(dims[0].first, dimension1);
  ASSERT_EQ(dims[1].first, dimension2);
  ASSERT_EQ(dims[2].first, dimension3);
  ASSERT_EQ(dims[3].first, dimension4);
}

TEST(VariantIteratorTests, FlattedDimensionTestWithNested) {
  std::string dimension1 = "1";
  std::string dimension2 = "2";
  std::string dimension3 = "3";
  std::string dimension4 = "4";
  std::string dimension5 = "5";
  std::string dimension6 = "6";
  std::vector<std::string> values = {"A"};

  VariantIterator vit;
  vit.add(dimension1, values, 4).addChilds({"A"}, dimension5, values);
  vit.add(dimension2, values, 3);
  vit.add(dimension3, values, 2).addChilds({"A"}, dimension6, values);
  ;
  vit.add(dimension4, values, 1);

  auto dims = vit.getFlattenedDimensions();

  ASSERT_EQ(dims.size(), 6);

  ASSERT_EQ(dims[0].first, dimension1);
  ASSERT_EQ(dims[1].first, dimension5);
  ASSERT_EQ(dims[2].first, dimension2);
  ASSERT_EQ(dims[3].first, dimension3);
  ASSERT_EQ(dims[4].first, dimension6);
  ASSERT_EQ(dims[5].first, dimension4);
}

TEST(VariantConfiguratorTests, ConfigurationTest) {
  VariantConfigurator vc;
  Variant first;
  first.insert({"1", "A"});
  first.insert({"2", "C"});
  Variant second;
  second.insert({"1", "B"});
  second.insert({"2", "C"});
  vc(first);
  vc(second);
}
}
