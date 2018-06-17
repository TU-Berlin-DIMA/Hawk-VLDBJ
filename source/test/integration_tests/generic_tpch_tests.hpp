/*
 * File:   generic_tpch_tests.hpp
 * Author: florian
 *
 * Header file for the TPC-H regression tests.
 * The tests will use a tpc-h database with scalefactor 1
 *
 * Created on 17. Juni 2015, 14:52
 */

#ifndef GENERIC_TPCH_TESTS_HPP
#define GENERIC_TPCH_TESTS_HPP

#include <iostream>
#include <parser/commandline_interpreter.hpp>
#include <sql/server/sql_driver.hpp>
#include <sql/server/sql_parsetree.hpp>

#include <fstream>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <util/init.hpp>
#include <util/tpch_benchmark.hpp>

#include "generic_regression_tests.hpp"
#include "gtest/gtest.h"
#include "util/tests.hpp"

namespace CoGaDB {

  class GenericTpchTest : public GenericRegressionTest {
   public:
    GenericTpchTest() {
      // some of the tpch-queries are hardcoded in CoGaDB.
      hardcodedTPCHQueryMap.insert(std::make_pair("tpch02", &getPlan_TPCH_Q2));
      hardcodedTPCHQueryMap.insert(std::make_pair("tpch04", &getPlan_TPCH_Q4));
      hardcodedTPCHQueryMap.insert(std::make_pair("tpch07", &getPlan_TPCH_Q7));
      hardcodedTPCHQueryMap.insert(std::make_pair("tpch09", &getPlan_TPCH_Q9));
      hardcodedTPCHQueryMap.insert(std::make_pair("tpch15", &getPlan_TPCH_Q15));
      hardcodedTPCHQueryMap.insert(std::make_pair("tpch18", &getPlan_TPCH_Q18));
      hardcodedTPCHQueryMap.insert(std::make_pair("tpch20", &getPlan_TPCH_Q20));
    }

    static const std::string TPCH_DIR_NAME;

    static void SetUpTestCase() {
      std::cout << "StartUpTestCase für TPCH" << std::endl;
      init();
      loadReferenceDatabaseTPCHScaleFactor1(client);
      // GenericRegressionTest::loadDatabase("startup.tpch.coga", "TPC-H");
    }

    virtual TablePtr assembleTableFromQuery(const std::string& deviceDir,
                                            const std::string&);

   private:
    TablePtr executeHardcodedTpchQuery(const std::string& tpchQuery);

    typedef query_processing::LogicalQueryPlanPtr (*HardcodedTPCHQueryPtr)(
        ClientPtr);
    typedef std::map<std::string, HardcodedTPCHQueryPtr> HardcodedTPCHQueryMap;

    HardcodedTPCHQueryMap hardcodedTPCHQueryMap;
  };

  const std::string GenericTpchTest::TPCH_DIR_NAME = std::string("tpch");
}

#endif /* GENERIC_TPCH_TESTS_HPP */
