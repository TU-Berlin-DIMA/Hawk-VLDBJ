//#include <core/table.hpp>
#include <iostream>
#include <parser/commandline_interpreter.hpp>
#include <sql/server/sql_driver.hpp>
#include <sql/server/sql_parsetree.hpp>

#include <persistence/storage_manager.hpp>

#include <fstream>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "generic_regression_tests.cpp"
#include "generic_regression_tests.hpp"
#include "generic_ssb_tests.hpp"
#include "generic_tpch_tests.hpp"
#include "gtest/gtest.h"

namespace CoGaDB {

const std::vector<std::pair<hype::DeviceTypeConstraint, std::string> >
readSsbTestcasesFromDir(hype::DeviceTypeConstraint device,
                        const std::string& dirname) {
  std::set<std::string> queryBlacklist = createBlacklistSet();

  std::vector<std::pair<hype::DeviceTypeConstraint, std::string> > testCases;
  std::string deviceDir = "/query_compiler/";

  std::string dirPath = getTestDataPath();
  dirPath.append(GenericSsbTest::SSB_DIR_NAME);
  dirPath.append(deviceDir);
  dirPath.append(dirname);

  std::vector<std::string> filenames = getFilenamesFromDir(dirPath);

  std::set<std::string>::iterator blacklistIt;
  for (std::vector<std::string>::const_iterator iter = filenames.begin();
       iter != filenames.end(); ++iter) {
    blacklistIt = queryBlacklist.find(*iter);
    if (blacklistIt == queryBlacklist.end()) {
      std::string testCase(*iter);
      testCases.push_back(
          std::pair<hype::DeviceTypeConstraint, std::string>(device, testCase));
    }
  }

  return testCases;
}

void testResultCorrect(const std::string& deviceDir,
                       const std::string& testname) {
  TablePtr newResultTable = assembleTableFromQuery(deviceDir, testname);
  TablePtr oldResultTable = createTableFromFile(deviceDir, testname, "results");

  if (!oldResultTable) {
    COGADB_FATAL_ERROR("Failed to load table from file " << testname, "");
  }

  if (newResultTable && oldResultTable) {
    if (!newResultTable->isMaterialized())
      newResultTable = newResultTable->materialize();
    newResultTable->setName(oldResultTable->getName());
    /* remove old qualified attribute names */
    renameFullyQualifiedNamesToUnqualifiedNames(newResultTable);
    /* add new qualified attribute names based on new table name */
    expandUnqualifiedColumnNamesToQualifiedColumnNames(newResultTable);
  }

  // at first check for different number of rows
  bool sameNumberOfRows =
      (newResultTable->getNumberofRows() == oldResultTable->getNumberofRows());
  if (!sameNumberOfRows) {
    std::cout << "Both tables have a different amount of rows: "
              << newResultTable->getNumberofRows() << " vs "
              << oldResultTable->getNumberofRows() << std::endl;
  }
  ASSERT_TRUE(sameNumberOfRows);

  bool sameSchema = isSameTableSchema(newResultTable->getSchema(),
                                      oldResultTable->getSchema());

  if (!sameSchema) {
    std::cout << "The schemas differ." << std::endl;

    TableSchema refSchema = oldResultTable->getSchema();
    TableSchema candidateSchema = newResultTable->getSchema();

    for (std::list<Attribut>::const_iterator it = refSchema.begin(),
                                             it_that = candidateSchema.begin();
         it != refSchema.end(); ++it, ++it_that) {
      std::cout << "[This] [Attribute] is " << util::getName((*it).first)
                << " [Name] is "
                << "\"" << (*it).second << "\"" << std::endl;
      std::cout << "[Candidate] [Attribute] is "
                << util::getName((*it_that).first) << " [Name] is "
                << "\"" << (*it_that).second << "\"" << std::endl;
      if ((*it) == (*it_that)) {
        std::cout << "Is Equal!" << std::endl;
      } else {
        std::cout << "Is Unequal!" << std::endl;
      }
    }
  }

  ASSERT_TRUE(sameSchema);

  // now that we know schema and row count matches we can call approx. equals
  // for the column-data
  bool equal = BaseTable::approximatelyEquals(oldResultTable, newResultTable);

  if (!equal) {
    COGADB_FATAL_ERROR("The data of the two tables do not match", "");
  }
  ASSERT_TRUE(equal);
}

// instantiate SSB Tests
INSTANTIATE_TEST_CASE_P(
    SSB_Query_Compiler_TestSuite, GenericSsbTest,
    ::testing::ValuesIn(readSsbTestcasesFromDir(hype::CPU_ONLY, "sql")));

TEST_P(GenericSsbTest, SSB_GenericTest) {
  std::cout << "Running test for file " << GetParam().second << std::endl;

  std::string deviceDir = "ssb/query_compiler/";

  VariableManager::instance().setVariableValue("query_execution_policy",
                                               "compiled");
  RuntimeConfiguration::instance().setOptimizer("no_join_order_optimizer");
  VariableManager::instance().setVariableValue("debug_code_generator", "false");
  VariableManager::instance().setVariableValue("print_query_result", "false");

  VariableManager::instance().setVariableValue("default_hash_table",
                                               "bucketchained");
  std::cout << "Testing C_CodeGenerator..." << std::endl;
  VariableManager::instance().setVariableValue("default_code_generator", "c");
  testResultCorrect(deviceDir, GetParam().second);

  std::cout << "Testing Multi_Staged CodeGenerator..." << std::endl;
  VariableManager::instance().setVariableValue("default_code_generator",
                                               "multi_staged");
  testResultCorrect(deviceDir, GetParam().second);
}
}  // end namespace
