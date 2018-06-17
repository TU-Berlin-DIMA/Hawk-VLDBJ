/*
 * File:   generic_ssb_tests.hpp
 * Author: florian
 *
 * Header file for the Star Schema Benchmark regression tests.
 * The tests will use a ssb database with scalefactor 1
 *
 * Created on 16. Juni 2015, 16:37
 */

#ifndef GENERIC_SSB_TESTS_HPP
#define GENERIC_SSB_TESTS_HPP

#include <util/init.hpp>
#include "generic_regression_tests.hpp"

namespace CoGaDB {

  class GenericSsbTest : public GenericRegressionTest {
   public:
    static const std::string SSB_DIR_NAME;

    static void SetUpTestCase() {
      std::cout << "GenericSSBTest: StartUpTestCase für SSB" << std::endl;
      init();
      loadReferenceDatabaseStarSchemaScaleFactor1(client);

      std::cout << "DONE WITH TESTCASE for SSB Setup" << std::endl;
    }
  };

  const std::string GenericSsbTest::SSB_DIR_NAME = std::string("ssb");
}

#endif /* GENERIC_SSB_TESTS_HPP */
