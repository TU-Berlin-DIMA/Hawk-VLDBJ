/*
 * File:   standard_query_tests.cpp
 * Author: florian
 *
 * Created on 16. March 2016
 */

#ifndef GENERIC_TPCH_TESTS_HPP
#define GENERIC_TPCH_TESTS_HPP

#include <iostream>
#include <parser/commandline_interpreter.hpp>
#include <sql/server/sql_driver.hpp>
#include <util/init.hpp>
#include "gtest/gtest.h"
#include "parser/client.hpp"
#include "util/tests.hpp"

namespace CoGaDB {

class StandardQueryTests : public testing::Test {
 public:
  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  static void SetUpTestCase() {
    init();
    loadReferenceDatabaseTPCHScaleFactor1(client);
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  static ClientPtr client;
  CommandLineInterpreter cmd;

  static bool DEBUG;
};

bool StandardQueryTests::DEBUG = false;
ClientPtr StandardQueryTests::client = ClientPtr(new LocalClient());

TEST_F(StandardQueryTests, SumTest) {
  // std::string query = "select L_QUANTITY, L_DISCOUNT from lineitem where
  // L_QUANTITY<25 AND l_discount<3 AND l_discount>=1;";
  std::string query =
      "select sum(l_extendedprice * l_discount) as revenue from lineitem where "
      "l_quantity < 25;";
  TablePtr resultTable = SQL::executeSQL(query, client);

  // created a result
  ASSERT_TRUE(resultTable != 0);
  ASSERT_GT(resultTable->getNumberofRows(), uint64_t(0));

  if (DEBUG) {
    std::cout << "Result from SumTest: " << resultTable->toString()
              << std::endl;
  }
}

TEST_F(StandardQueryTests, DateTest) {
  std::string query =
      "select sum(l_extendedprice * l_discount) as revenue from lineitem where "
      "l_shipdate >= '1994-01-01' and l_shipdate < '1995-01-01';";
  TablePtr resultTable = SQL::executeSQL(query, client);

  // created a result
  ASSERT_TRUE(resultTable != 0);
  ASSERT_GT(resultTable->getNumberofRows(), uint64_t(0));

  if (DEBUG) {
    std::cout << "Result from DateTest: " << resultTable->toString()
              << std::endl;
  }
}

TEST_F(StandardQueryTests, BetweenTest) {
  std::string query =
      "select sum(l_extendedprice * l_discount) as revenue from lineitem where "
      "l_discount between 0.05 and 0.06;";
  TablePtr resultTable = SQL::executeSQL(query, client);

  // created a result
  ASSERT_TRUE(resultTable != 0);
  ASSERT_GT(resultTable->getNumberofRows(), uint64_t(0));

  if (DEBUG) {
    std::cout << "Result from BetweenTest: " << resultTable->toString()
              << std::endl;
  }
}

/*TEST_F(StandardQueryTests, OrderByTest) {
    std::string query =
    "select p_name, p_retailprice, p_brand from part where p_brand = 'Brand#13'
order by p_retailprice;";
    //"select l_returnflag, l_linestatus, l_quantity from lineitem where
l_shipdate <= '1998-12-01' order by l_linestatus;";
    TablePtr resultTable = SQL::executeSQL(query, client);

    //created a result
    ASSERT_TRUE(resultTable != 0);
    ASSERT_GT(resultTable->getNumberofRows(),0);

    if(DEBUG) {
        std::cout << "Result from OrderByTest: " <<
        resultTable->toString() << std::endl;
    }

} */

TEST_F(StandardQueryTests, GroupByTest) {
  std::string query =
      "select sum(p_retailprice) as brandprice, p_brand from part group by "
      "p_brand;";
  //"select l_returnflag, l_linestatus, l_quantity from lineitem where
  // l_shipdate <= '1998-12-01' order by l_linestatus;";
  TablePtr resultTable = SQL::executeSQL(query, client);

  // created a result
  ASSERT_TRUE(resultTable != 0);
  ASSERT_GT(resultTable->getNumberofRows(), uint64_t(0));

  if (DEBUG) {
    std::cout << "Result from GroupByTest: " << resultTable->toString()
              << std::endl;
  }
}

/*TEST_F(StandardQueryTests, SampleTest) {
    std::string query = "select L_QUANTITY, L_DISCOUNT from lineitem where
L_QUANTITY<25 AND l_discount<3 AND l_discount>=1;";
    TablePtr resultTable = SQL::executeSQL(query, client);
    ASSERT_TRUE(resultTable != 0);
    std::cout << resultTable->toString() << std::endl;
    ASSERT_TRUE(true);
} */

}  // end namespace

#endif
