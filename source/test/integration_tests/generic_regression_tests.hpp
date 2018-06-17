/*
 * File:   generic_regression_tests.hpp
 * Author: florian
 *
 * Baseclass for the regression tests for both database benchmarks
 * TPC-H and SSB.
 *
 * For the tpc-h regression tests see "generic_tpch_tests"
 * For the ssb regression tests see "generic_ssb_tests"
 *
 * Created on 16. Juni 2015, 16:07
 */

#ifndef GENERIC_REGRESSION_TESTS_HPP
#define GENERIC_REGRESSION_TESTS_HPP

#include <iostream>
#include <parser/commandline_interpreter.hpp>
#include <sql/server/sql_driver.hpp>
#include <sql/server/sql_parsetree.hpp>

#include <fstream>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "gtest/gtest.h"

#include <core/variable_manager.hpp>
#include "util/filesystem.hpp"
#include "util/tests.hpp"

namespace CoGaDB {

  class GenericRegressionTest
      : public testing::TestWithParam<
            std::pair<hype::DeviceTypeConstraint, std::string> > {
   public:
    /*
    *  will be called before the testcase will be executed
    */
    static void SetUpTestCase() {
      std::cout << "GenericRegressionTest: StartUpTestCase" << std::endl;

      /* depending on join order, aggregation results can slighly change,
       * so we need to ensure that we use always the same optimizer pipeline */
      RuntimeConfiguration::instance().setOptimizer("default_optimizer");
      std::cout << "DONE" << std::endl;
    }

    /*
    *  will be called after the testcase is finished.
    */
    static void TearDownTestCase() {
      std::vector<TablePtr>& globalTables = CoGaDB::getGlobalTableList();
      globalTables.clear();
#ifndef __APPLE__
      quick_exit(0);
#endif
    }

   protected:
    const static ClientPtr client;

    void loadDatabase() { loadReferenceDatabaseStarSchemaScaleFactor1(client); }
  };

  const std::string getTestDataPath();
  const std::vector<std::string> getFilenamesFromDir(
      const std::string& dirPath);
  TablePtr createTableFromFile(const std::string& deviceDir,
                               const std::string& testname,
                               const std::string& pathToResults);
  std::string getFileContent(const std::string&);
  TablePtr assembleTableFromQuery(const std::string& deviceDir,
                                  const std::string&);

  //    void removeTableFromGlobalTableList(TablePtr table);s
}

#endif /* GENERIC_REGRESSION_TESTS_HPP */
