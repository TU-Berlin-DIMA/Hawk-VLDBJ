#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <core/global_definitions.hpp>

#include "compression/wah_bitmap_column.hpp"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

namespace CoGaDB {

const std::string TESTDATA_PATH =
    std::string(PATH_TO_COGADB_EXECUTABLE) + "/test/testdata/regression test/";

class WAHBitmapColumnTest : public testing::Test {
 public:
  WAHBitmapColumnTest() {
    wah_column_zero_one =
        WAHBitmapColumnPtr(new WAHBitmapColumn("ZERO_ON_ALTERNATING"));
    for (int i = 0; i < 100; i++) {
      if (i % 2 == 0) {
        wah_column_zero_one->unsetNextRow();
      } else {
        wah_column_zero_one->setNextRow();
      }
    }

    wah_column_zeroes = WAHBitmapColumnPtr(new WAHBitmapColumn("ONLY_ZEROES"));
    for (int i = 0; i < 100; i++) {
      wah_column_zeroes->unsetNextRow();
    }

    wah_column_ones = WAHBitmapColumnPtr(new WAHBitmapColumn("ONLY_ONES"));
    for (int i = 0; i < 100; i++) {
      wah_column_ones->setNextRow();
    }

    wah_column_mixed = WAHBitmapColumnPtr(new WAHBitmapColumn("MIXED"));
    for (int i = 0; i < 64; i++) {
      wah_column_mixed->unsetNextRow();
    }
    for (int i = 0; i < 60; i++) {
      wah_column_mixed->setNextRow();
    }
    for (int i = 0; i < 3; i++) {
      wah_column_mixed->unsetNextRow();
    }
    for (int i = 0; i < 3; i++) {
      wah_column_mixed->setNextRow();
    }
    for (int i = 0; i < 70; i++) {
      wah_column_mixed->unsetNextRow();
    }

    wah_column_empty = WAHBitmapColumnPtr(new WAHBitmapColumn("EMPTY"));
  }

  virtual ~WAHBitmapColumnTest() {}

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

 protected:
  WAHBitmapColumnPtr wah_column_zero_one;
  WAHBitmapColumnPtr wah_column_zeroes;
  WAHBitmapColumnPtr wah_column_ones;
  WAHBitmapColumnPtr wah_column_mixed;
  WAHBitmapColumnPtr wah_column_empty;

  unsigned long getSizeOfWordsVector(WAHBitmapColumnPtr bitmap_column) {
    return bitmap_column->words_.size();
  }

  uint32_t getVectorWord(WAHBitmapColumnPtr bitmap_column, unsigned int index) {
    return bitmap_column->words_[index];
  }

  uint32_t getTailingWord(WAHBitmapColumnPtr bitmap_column) {
    return bitmap_column->tailing_word_;
  }

  bool isFillWord(WAHBitmapColumnPtr bitmap_column, uint32_t word) {
    return bitmap_column->isFillWord(word);
  }

  uint32_t getFillWordValue(WAHBitmapColumnPtr bitmap_column, uint32_t word) {
    return bitmap_column->getFillWordValue(word);
  }

  uint32_t getFillWordCount(WAHBitmapColumnPtr bitmap_column, uint32_t word) {
    return bitmap_column->getFillWordCount(word);
  }
};

//@formatter:off
TEST_F(WAHBitmapColumnTest, NUMBER_OF_ROWS_ZERO_ONE_ALTERNATING) {
  ASSERT_EQ(100u, wah_column_zero_one->getNumberOfRows());
}

TEST_F(WAHBitmapColumnTest, NUMBER_OF_ROWS_100_ONES) {
  ASSERT_EQ(100u, wah_column_ones->getNumberOfRows());
}

TEST_F(WAHBitmapColumnTest, NUMBER_OF_ROWS_100_ZEROES) {
  ASSERT_EQ(100u, wah_column_zeroes->getNumberOfRows());
}

TEST_F(WAHBitmapColumnTest,
       NUMBER_OF_ROWS_64_ZEROES_60_ONES_3_ZEROES_3_ONES_70_ZEROES) {
  ASSERT_EQ(200u, wah_column_mixed->getNumberOfRows());
}

TEST_F(WAHBitmapColumnTest, NUMBER_OF_ROWS_EMPTY) {
  ASSERT_EQ(0u, wah_column_empty->getNumberOfRows());
}

TEST_F(WAHBitmapColumnTest, IS_FILL_WORD) {
  ASSERT_TRUE(isFillWord(wah_column_empty, 0x80000000));
  ASSERT_TRUE(isFillWord(wah_column_empty, 0x80000001));
  ASSERT_TRUE(isFillWord(wah_column_empty, 0xC0000001));
  ASSERT_FALSE(isFillWord(wah_column_empty, 0x70000001));
}

TEST_F(WAHBitmapColumnTest, GET_FILL_WORD_VALUE) {
  ASSERT_EQ(0u, getFillWordValue(wah_column_empty, 0x80000000));
  ASSERT_EQ(0u, getFillWordValue(wah_column_empty, 0xA0000001));
  ASSERT_EQ(1u, getFillWordValue(wah_column_empty, 0xC0000001));
  ASSERT_EQ(1u, getFillWordValue(wah_column_empty, 0xF0005001));
}

TEST_F(WAHBitmapColumnTest, GET_FILL_WORD_COUNT) {
  ASSERT_EQ(0x00000001u, getFillWordCount(wah_column_empty, 0x80000001));
  ASSERT_EQ(0x20000001u, getFillWordCount(wah_column_empty, 0xA0000001));
  ASSERT_EQ(0x20000001u, getFillWordCount(wah_column_empty, 0xE0000001));
  ASSERT_EQ(0x30005001u, getFillWordCount(wah_column_empty, 0xF0005001));
}

TEST_F(WAHBitmapColumnTest, INTERNAL_DATA_STRUCTURES_ZERO_ONE_ALTERNATING) {
  ASSERT_EQ(3u, getSizeOfWordsVector(wah_column_zero_one));

  ASSERT_EQ(715827882u, getVectorWord(wah_column_zero_one, 0));
  ASSERT_EQ(1431655765u, getVectorWord(wah_column_zero_one, 1));
  ASSERT_EQ(715827882u, getVectorWord(wah_column_zero_one, 2));

  ASSERT_EQ(85u, getTailingWord(wah_column_zero_one));
}

TEST_F(WAHBitmapColumnTest, INTERNAL_DATA_STRUCTURES_100_ONES) {
  ASSERT_EQ(1u, getSizeOfWordsVector(wah_column_ones));

  ASSERT_EQ(3221225475u, getVectorWord(wah_column_ones, 0));

  ASSERT_EQ(127u, getTailingWord(wah_column_ones));
}

TEST_F(WAHBitmapColumnTest, INTERNAL_DATA_STRUCTURES_100_ZEROES) {
  ASSERT_EQ(1u, getSizeOfWordsVector(wah_column_zeroes));

  ASSERT_EQ(2147483651u, getVectorWord(wah_column_zeroes, 0));

  ASSERT_EQ(0u, getTailingWord(wah_column_zeroes));
}

TEST_F(WAHBitmapColumnTest,
       INTERNAL_DATA_STRUCTURES_64_ZEROES_60_ONES_3_ZEROES_3_ONES_70_ZEROES) {
  ASSERT_EQ(5u, getSizeOfWordsVector(wah_column_mixed));

  ASSERT_EQ(2147483650u, getVectorWord(wah_column_mixed, 0));
  ASSERT_EQ(536870911u, getVectorWord(wah_column_mixed, 1));
  ASSERT_EQ(3221225473u, getVectorWord(wah_column_mixed, 2));
  ASSERT_EQ(234881024u, getVectorWord(wah_column_mixed, 3));
  ASSERT_EQ(2147483649u, getVectorWord(wah_column_mixed, 4));

  ASSERT_EQ(0u, getTailingWord(wah_column_mixed));
}

TEST_F(WAHBitmapColumnTest, PREFIXSUM_ZERO_ONE_ALTERNATING) {
  ASSERT_EQ(0u, wah_column_zero_one->getPrefixSum(0));
  ASSERT_EQ(0u, wah_column_zero_one->getPrefixSum(1));
  ASSERT_EQ(1u, wah_column_zero_one->getPrefixSum(2));
  ASSERT_EQ(15u, wah_column_zero_one->getPrefixSum(31));
  ASSERT_EQ(16u, wah_column_zero_one->getPrefixSum(32));
  ASSERT_EQ(25u, wah_column_zero_one->getPrefixSum(50));
  ASSERT_EQ(48u, wah_column_zero_one->getPrefixSum(97));
  // TODO create death test for this later on
  // EXPECT_FATAL_FAILURE(bitmap_column->getPrefixSum(100),
  // "INDEX_OUT_OF_BOUNDS");
}

TEST_F(WAHBitmapColumnTest, PREFIXSUM_100_ONES) {
  ASSERT_EQ(0u, wah_column_ones->getPrefixSum(0));
  ASSERT_EQ(1u, wah_column_ones->getPrefixSum(1));
  ASSERT_EQ(2u, wah_column_ones->getPrefixSum(2));
  ASSERT_EQ(31u, wah_column_ones->getPrefixSum(31));
  ASSERT_EQ(32u, wah_column_ones->getPrefixSum(32));
  ASSERT_EQ(50u, wah_column_ones->getPrefixSum(50));
  ASSERT_EQ(97u, wah_column_ones->getPrefixSum(97));
}

TEST_F(WAHBitmapColumnTest, PREFIXSUM_100_ZEROES) {
  ASSERT_EQ(0u, wah_column_zeroes->getPrefixSum(0));
  ASSERT_EQ(0u, wah_column_zeroes->getPrefixSum(1));
  ASSERT_EQ(0u, wah_column_zeroes->getPrefixSum(2));
  ASSERT_EQ(0u, wah_column_zeroes->getPrefixSum(31));
  ASSERT_EQ(0u, wah_column_zeroes->getPrefixSum(32));
  ASSERT_EQ(0u, wah_column_zeroes->getPrefixSum(50));
  ASSERT_EQ(0u, wah_column_zeroes->getPrefixSum(97));
}

TEST_F(WAHBitmapColumnTest,
       PREFIXSUM_64_ZEROES_60_ONES_3_ZEROES_3_ONES_70_ZEROES) {
  ASSERT_EQ(0u, wah_column_mixed->getPrefixSum(0));
  ASSERT_EQ(0u, wah_column_mixed->getPrefixSum(1));
  ASSERT_EQ(0u, wah_column_mixed->getPrefixSum(2));
  ASSERT_EQ(0u, wah_column_mixed->getPrefixSum(31));
  ASSERT_EQ(0u, wah_column_mixed->getPrefixSum(32));
  ASSERT_EQ(0u, wah_column_mixed->getPrefixSum(50));
  ASSERT_EQ(33u, wah_column_mixed->getPrefixSum(97));
  ASSERT_EQ(56u, wah_column_mixed->getPrefixSum(120));
  ASSERT_EQ(59u, wah_column_mixed->getPrefixSum(123));
  ASSERT_EQ(63u, wah_column_mixed->getPrefixSum(197));
}

TEST_F(WAHBitmapColumnTest, IS_ROW_SET_ZERO_ONE_ALTERNATING) {
  ASSERT_FALSE(wah_column_zero_one->isRowSet(0));
  ASSERT_TRUE(wah_column_zero_one->isRowSet(1));
  ASSERT_FALSE(wah_column_zero_one->isRowSet(2));
  ASSERT_TRUE(wah_column_zero_one->isRowSet(31));
  ASSERT_FALSE(wah_column_zero_one->isRowSet(32));
  ASSERT_FALSE(wah_column_zero_one->isRowSet(50));
  ASSERT_TRUE(wah_column_zero_one->isRowSet(97));
  // TODO create death test for this later on
  // EXPECT_FATAL_FAILURE(bitmap_column->getPrefixSum(100),
  // "INDEX_OUT_OF_BOUNDS");
}

TEST_F(WAHBitmapColumnTest, IS_ROW_SET_100_ONES) {
  ASSERT_TRUE(wah_column_ones->isRowSet(0));
  ASSERT_TRUE(wah_column_ones->isRowSet(1));
  ASSERT_TRUE(wah_column_ones->isRowSet(2));
  ASSERT_TRUE(wah_column_ones->isRowSet(31));
  ASSERT_TRUE(wah_column_ones->isRowSet(32));
  ASSERT_TRUE(wah_column_ones->isRowSet(50));
  ASSERT_TRUE(wah_column_ones->isRowSet(97));
}

TEST_F(WAHBitmapColumnTest, IS_ROW_SET_100_ZEROES) {
  ASSERT_FALSE(wah_column_zeroes->isRowSet(0));
  ASSERT_FALSE(wah_column_zeroes->isRowSet(1));
  ASSERT_FALSE(wah_column_zeroes->isRowSet(2));
  ASSERT_FALSE(wah_column_zeroes->isRowSet(31));
  ASSERT_FALSE(wah_column_zeroes->isRowSet(32));
  ASSERT_FALSE(wah_column_zeroes->isRowSet(50));
  ASSERT_FALSE(wah_column_zeroes->isRowSet(97));
}

TEST_F(WAHBitmapColumnTest,
       IS_ROW_SET_64_ZEROES_60_ONES_3_ZEROES_3_ONES_70_ZEROES) {
  ASSERT_FALSE(wah_column_mixed->isRowSet(0));
  ASSERT_FALSE(wah_column_mixed->isRowSet(1));
  ASSERT_FALSE(wah_column_mixed->isRowSet(2));
  ASSERT_FALSE(wah_column_mixed->isRowSet(31));
  ASSERT_FALSE(wah_column_mixed->isRowSet(32));
  ASSERT_FALSE(wah_column_mixed->isRowSet(50));
  ASSERT_TRUE(wah_column_mixed->isRowSet(97));
  ASSERT_TRUE(wah_column_mixed->isRowSet(120));
  ASSERT_TRUE(wah_column_mixed->isRowSet(123));
  ASSERT_FALSE(wah_column_mixed->isRowSet(197));
}

//@formatter:on

}  // end namespace
