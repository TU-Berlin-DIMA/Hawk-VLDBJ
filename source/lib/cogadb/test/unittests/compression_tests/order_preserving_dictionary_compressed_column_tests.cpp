#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <core/global_definitions.hpp>

#include "compression/order_preserving_dictionary_compressed_column.hpp"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"

namespace CoGaDB {

const std::string TESTDATA_PATH = std::string(PATH_TO_COGADB_EXECUTABLE) +
                                  "/test/testdata/compression_tests/";

class OrderPreservingDictColTest : public testing::Test {
 public:
  typedef boost::shared_ptr<
      OrderPreservingDictionaryCompressedColumn<std::string> >
      StringOrderPreservingDictCompColPtr;

  OrderPreservingDictColTest() {
    orderCompColumn = StringOrderPreservingDictCompColPtr(
        new OrderPreservingDictionaryCompressedColumn<std::string>("C_REGION",
                                                                   VARCHAR));

    // Serialisierte Daten sind in falscher Ordnung und daher nutzlos
    // orderCompColumn->load(TESTDATA_PATH + "dictionary_compressed_column");

    // create Random Column
    randomOrderCompColumn = StringOrderPreservingDictCompColPtr(
        new OrderPreservingDictionaryCompressedColumn<std::string>("RANDOM_COL",
                                                                   VARCHAR));

    for (int i = 1; i <= 1000; ++i) {
      // fill random comp column
      if (i % 7 == 0)
        randomOrderCompColumn->insert(std::string("AMERICA"));
      else if (i % 5 == 0)
        randomOrderCompColumn->insert(std::string("ASIA"));
      else
        randomOrderCompColumn->insert(std::string("EUROPE"));

      // fill seq ordered comp column
      if (i <= 300)
        orderCompColumn->insert(std::string("AMERICA"));
      else if (i > 300 && i <= 600)
        orderCompColumn->insert(std::string("ASIA"));
      else if (i > 600 && i <= 1000)
        orderCompColumn->insert(std::string("EUROPE"));
    }
  }

  virtual ~OrderPreservingDictColTest() {}

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  static void SetUpTestCase() {
    std::cout << "OrderPreservingTest: SetUpTestCase" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "OrderPreservingTest: TearDownTestCase" << std::endl;
  }

 protected:
  // OrderPreservingDictionaryCompressedColumn<std::string> testing;
  StringOrderPreservingDictCompColPtr orderCompColumn;

  StringOrderPreservingDictCompColPtr randomOrderCompColumn;

  uint32_t* getCompressedData(StringOrderPreservingDictCompColPtr column) {
    return column->ids_->data();
  }

  std::map<std::string, uint32_t>& getDictionary(
      StringOrderPreservingDictCompColPtr column) {
    return column->dictionary_;
  }

  std::vector<std::string>& getReverseLookupVector(
      StringOrderPreservingDictCompColPtr column) {
    return column->reverse_lookup_vector_;
  }
};

/* Test the correct size of the OrderComoColumn and if
* everything is inserted        */
TEST_F(OrderPreservingDictColTest, SIZE_TEST) {
  ASSERT_EQ(1000u, orderCompColumn->getNumberOfRows());
  ASSERT_EQ(1000u, randomOrderCompColumn->getNumberOfRows());
}

/** Tests for correctness of the reverse vector **/
TEST_F(OrderPreservingDictColTest, REVERSE_VECTOR_TEST) {
  std::vector<std::string> revVec =
      getReverseLookupVector(randomOrderCompColumn);

  ASSERT_STREQ("AMERICA", revVec[0].c_str());
  ASSERT_STREQ("ASIA", revVec[1].c_str());
  ASSERT_STREQ("EUROPE", revVec[2].c_str());

  revVec = getReverseLookupVector(orderCompColumn);

  ASSERT_STREQ("AMERICA", revVec[0].c_str());
  ASSERT_STREQ("ASIA", revVec[1].c_str());
  ASSERT_STREQ("EUROPE", revVec[2].c_str());
}

/** Tests for correctness of the dictionary **/
TEST_F(OrderPreservingDictColTest, DICTIONARY_TEST) {
  std::map<std::string, uint32_t> dict = getDictionary(randomOrderCompColumn);

  ASSERT_EQ(0u, dict["AMERICA"]);
  ASSERT_EQ(1u, dict["ASIA"]);
  ASSERT_EQ(2u, dict["EUROPE"]);

  dict = getDictionary(orderCompColumn);

  ASSERT_EQ(0u, dict["AMERICA"]);
  ASSERT_EQ(1u, dict["ASIA"]);
  ASSERT_EQ(2u, dict["EUROPE"]);
}

/** Tests inserting a new value which triggers a reordering **/
TEST_F(OrderPreservingDictColTest, INSERT_WITH_REORDERING_TEST) {
  randomOrderCompColumn->insert(std::string("AUSTRALIA"));
  ASSERT_EQ(1001u, randomOrderCompColumn->getNumberOfRows());

  std::vector<std::string> revVec =
      getReverseLookupVector(randomOrderCompColumn);

  ASSERT_STREQ("AMERICA", revVec[0].c_str());
  ASSERT_STREQ("ASIA", revVec[1].c_str());
  ASSERT_STREQ("AUSTRALIA", revVec[2].c_str());
  ASSERT_STREQ("EUROPE", revVec[3].c_str());

  std::map<std::string, uint32_t> dict = getDictionary(randomOrderCompColumn);

  ASSERT_EQ(0u, dict["AMERICA"]);
  ASSERT_EQ(1u, dict["ASIA"]);
  ASSERT_EQ(2u, dict["AUSTRALIA"]);
  ASSERT_EQ(3u, dict["EUROPE"]);
}

/** Tests inserting a new value which does not trigger a reordering **/
TEST_F(OrderPreservingDictColTest, INSERT_WITHOUT_REORDERING_TEST) {
  randomOrderCompColumn->insert(std::string("SOUTH_AMERICA"));
  ASSERT_EQ(1001u, randomOrderCompColumn->getNumberOfRows());

  std::vector<std::string> revVec =
      getReverseLookupVector(randomOrderCompColumn);

  ASSERT_STREQ("AMERICA", revVec[0].c_str());
  ASSERT_STREQ("ASIA", revVec[1].c_str());
  ASSERT_STREQ("EUROPE", revVec[2].c_str());
  ASSERT_STREQ("SOUTH_AMERICA", revVec[3].c_str());

  std::map<std::string, uint32_t> dict = getDictionary(randomOrderCompColumn);

  ASSERT_EQ(0u, dict["AMERICA"]);
  ASSERT_EQ(1u, dict["ASIA"]);
  ASSERT_EQ(2u, dict["EUROPE"]);
  ASSERT_EQ(3u, dict["SOUTH_AMERICA"]);
}

/** Tests updating a element at a specific index which does not
    trigger a reordering **/
TEST_F(OrderPreservingDictColTest, UPDATE_WITHOUT_REORDERING_TEST) {
  std::string valueBefore =
      boost::any_cast<std::string>(randomOrderCompColumn->get(554));
  ASSERT_STREQ("ASIA", valueBefore.c_str());

  randomOrderCompColumn->update(554, std::string("EUROPE"));

  std::string valueAfter =
      boost::any_cast<std::string>(randomOrderCompColumn->get(554));
  ASSERT_STRNE(valueBefore.c_str(), valueAfter.c_str());
}

/** Tests updating a element at a specific index which does
    trigger a reordering **/
TEST_F(OrderPreservingDictColTest, UPDATE_WITH_REORDERING_TEST) {
  std::string valueBefore =
      boost::any_cast<std::string>(randomOrderCompColumn->get(554));
  ASSERT_STREQ("ASIA", valueBefore.c_str());

  randomOrderCompColumn->update(554, std::string("AUSTRALIA"));

  std::string valueAfter =
      boost::any_cast<std::string>(randomOrderCompColumn->get(554));
  ASSERT_STRNE(valueBefore.c_str(), valueAfter.c_str());

  std::vector<std::string> revVec =
      getReverseLookupVector(randomOrderCompColumn);
  ASSERT_EQ(4u, revVec.size());
  ASSERT_STREQ("AMERICA", revVec[0].c_str());
  ASSERT_STREQ("ASIA", revVec[1].c_str());
  ASSERT_STREQ("AUSTRALIA", revVec[2].c_str());
  ASSERT_STREQ("EUROPE", revVec[3].c_str());
}

/** Tests removing a element -- deactivated because so far we do not delete
data from columns
TEST_F(OrderPreservingDictColTest, REMOVE_ELEMENT_TEST) {
  //removing a value leaves order untouched. hence it will not be tested here
  bool result = randomOrderCompColumn->remove(500);
  ASSERT_TRUE(result);
  ASSERT_EQ(999, randomOrderCompColumn->getNumberOfRows());
} **/
}
