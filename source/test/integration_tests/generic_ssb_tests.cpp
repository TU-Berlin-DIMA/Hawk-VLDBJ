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
  DIR* dirP;
  struct dirent* dir;
  struct stat filestatus;

  std::string deviceDir;
  if (device == hype::CPU_ONLY) {
    deviceDir = "/cpu/";

  } else if (device == hype::GPU_ONLY) {
    deviceDir = "/gpu/";
  } else {
    COGADB_ERROR("GlobalDeviceConstraint is whether CPU_ONLY nor GPU_ONLY.",
                 "");
    exit(-1);
  }

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

// instantiate SSB Tests
#ifdef ENABLE_GPU_ACCELERATION
INSTANTIATE_TEST_CASE_P(
    SSB_TestSuite_GPU, GenericSsbTest,
    ::testing::ValuesIn(readSsbTestcasesFromDir(hype::GPU_ONLY, "sql")));
#endif

INSTANTIATE_TEST_CASE_P(
    SSB_TestSuite_CPU, GenericSsbTest,
    testing::ValuesIn(readSsbTestcasesFromDir(hype::CPU_ONLY, "sql")), );

TEST_P(GenericSsbTest, SSB_GenericTest) {
  std::cout << "Running test for file " << GetParam().second << std::endl;

  std::cout << "List of global tables before test:" << std::endl;
  std::vector<TablePtr>& globalTables = CoGaDB::getGlobalTableList();
  for (size_t i = 0; i < globalTables.size(); i++) {
    std::cout << globalTables[i]->getName() << std::endl;
  }

  std::string deviceDir;
  if (GetParam().first == hype::CPU_ONLY) {
    RuntimeConfiguration::instance().setGlobalDeviceConstraint(hype::CPU_ONLY);
    deviceDir = "ssb/cpu";

  } else {
    RuntimeConfiguration::instance().setGlobalDeviceConstraint(hype::GPU_ONLY);
    deviceDir = "ssb/gpu";
  }

  TablePtr newResultTable =
      assembleTableFromQuery(deviceDir, GetParam().second);
  TablePtr oldResultTable =
      createTableFromFile(deviceDir, GetParam().second, "results");

  if (!oldResultTable) {
    COGADB_FATAL_ERROR("Failed to load table from file " << GetParam().second,
                       "");
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
    std::cout << "The data of the two tables do not match" << std::endl;
  }

  ASSERT_TRUE(equal);
}
}  // end namespace
