#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <core/global_definitions.hpp>

#include "compression/dictionary_compressed_column.hpp"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

namespace CoGaDB {

const std::string TESTDATA_PATH = std::string(PATH_TO_COGADB_EXECUTABLE) +
                                  "/test/testdata/compression_tests/";

class DictionaryCompressedColumnTest : public testing::Test {
 public:
  typedef DictionaryCompressedColumn<std::string> Column;
  typedef boost::shared_ptr<Column> StringTypedDictionaryCompressedColumnPtr;

  DictionaryCompressedColumnTest() {
    region_column = StringTypedDictionaryCompressedColumnPtr(
        new DictionaryCompressedColumn<std::string>("C_REGION", VARCHAR));
    region_column->load(TESTDATA_PATH + "dictionary_compressed_column",
                        LOAD_ALL_DATA);
  }

  virtual ~DictionaryCompressedColumnTest() {}

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

 protected:
  uint32_t *getCompressedData(StringTypedDictionaryCompressedColumnPtr column) {
    return column->ids_->data();
  }

  Column::DictionaryPtr getDictionary(
      StringTypedDictionaryCompressedColumnPtr column) {
    return column->dictionary_;
  }

  Column::ReverseLookupVectorPtr getReverseLookupVector(
      StringTypedDictionaryCompressedColumnPtr column) {
    return column->reverse_lookup_vector_;
  }

  StringTypedDictionaryCompressedColumnPtr region_column;
};

//@formatter:off
TEST_F(DictionaryCompressedColumnTest, NUMBER_OF_ROWS_SSB_SF1) {
  ASSERT_EQ(30000u, region_column->getNumberOfRows());
}

TEST_F(DictionaryCompressedColumnTest, DICTIONARY_SIZE) {
  DictionaryCompressedColumnTest::Column::DictionaryPtr dictionary =
      getDictionary(region_column);
  DictionaryCompressedColumnTest::Column::ReverseLookupVectorPtr
      reverse_lookup_vector = getReverseLookupVector(region_column);
  ASSERT_EQ(5u, dictionary->size());
  ASSERT_EQ(5u, reverse_lookup_vector->size());
}

//@formatter:on

}  // end namespace
