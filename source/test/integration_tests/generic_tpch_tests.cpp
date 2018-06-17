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
readTpchTestcasesFromDir(const std::string& dirname) {
  std::set<std::string> queryBlacklist = createBlacklistSet();

  std::vector<std::pair<hype::DeviceTypeConstraint, std::string> > testCases;
  DIR* dirP;
  struct dirent* dir;
  struct stat filestatus;

  std::string dirPath = getTestDataPath();
  dirPath.append(GenericTpchTest::TPCH_DIR_NAME);
  dirPath.append("/");
  dirPath.append(dirname);

  std::vector<std::string> filenames = getFilenamesFromDir(dirPath);

  std::set<std::string>::iterator blacklistIt;
  for (std::vector<std::string>::const_iterator iter = filenames.begin();
       iter != filenames.end(); ++iter) {
    std::cout << "query: " << *iter << std::endl;
    blacklistIt = queryBlacklist.find(*iter);
    if (blacklistIt == queryBlacklist.end())
      testCases.push_back(std::pair<hype::DeviceTypeConstraint, std::string>(
          hype::ANY_DEVICE, *iter));
  }

  return testCases;
}

TablePtr GenericTpchTest::assembleTableFromQuery(const std::string& deviceDir,
                                                 const std::string& testname) {
  std::cout << "TPCH assembleTableFromQuery" << std::endl;

  std::string filepath = getTestDataPath();
  filepath.append(deviceDir);
  filepath.append("/sql/").append(testname).append(".sql");

  std::string query = getFileContent(filepath);

  if (!quiet && debug) std::cout << "Query: " << query << std::endl;

  TablePtr resultTable;
  if (query.find("tpch", 0) == 0) {
    std::cout << "Exucting hardcoded query: " << query << std::endl;

    // query starts with tpch
    resultTable = executeHardcodedTpchQuery(query);

  } else {
    resultTable = SQL::executeSQL(query, client);
  }
  std::cout << "Table from query loaded with " << resultTable->getNumberofRows()
            << " rows " << std::endl;

  // materialize the table
  resultTable->materialize();

  return resultTable;
}

TablePtr GenericTpchTest::executeHardcodedTpchQuery(
    const std::string& tpchQuery) {
  std::cout << "Execute hardcoded TPCH Query " << tpchQuery << std::endl;

  HardcodedTPCHQueryMap::iterator query_it =
      hardcodedTPCHQueryMap.find(tpchQuery);

  if (query_it != hardcodedTPCHQueryMap.end()) {
    query_processing::LogicalQueryPlanPtr hardcodedQueryPlan =
        query_it->second(client);
    query_processing::PhysicalQueryPlanPtr plan =
        query_processing::optimize_and_execute(tpchQuery, *hardcodedQueryPlan,
                                               client);

    TablePtr hardcodedQueryTable = plan->getResult();

    return hardcodedQueryTable;

  } else {
    std::cout << "No entry in hardcoded queries for " << tpchQuery << std::endl;
    return TablePtr();
  }
}

// instantiate TPCH Tests
INSTANTIATE_TEST_CASE_P(TPCH_TestSuite, GenericTpchTest,
                        ::testing::ValuesIn(readTpchTestcasesFromDir("sql")));

TEST_P(GenericTpchTest, TPCH_GenericTest) {
  std::vector<TablePtr>& globalTables = CoGaDB::getGlobalTableList();

  // RuntimeConfiguration::instance().setGlobalDeviceConstraint(hype::GPU_ONLY);
  std::cout << "Starting TPCH Generic Test with " << GetParam().second
            << std::endl;
  TablePtr newResultTable = assembleTableFromQuery("tpch", GetParam().second);

  if (newResultTable == nullptr) {
    std::cout << "Table assembled from query returned a null pointer"
              << std::endl;
    FAIL();
  }

  TablePtr oldResultTableCpu =
      createTableFromFile("tpch", GetParam().second, "results/cpu");

  if (oldResultTableCpu == 0) {
    std::cout << "Table created from file returned a null pointer" << std::endl;
    FAIL();
  }

  removeTableFromGlobalTableList(oldResultTableCpu);

  std::cout << "Start with comparing the tables" << std::endl;

  oldResultTableCpu->materialize();
  newResultTable->materialize();

  if (newResultTable && oldResultTableCpu) {
    if (!newResultTable->isMaterialized())
      newResultTable = newResultTable->materialize();
    newResultTable->setName(oldResultTableCpu->getName());
    /* remove old qualified attribute names */
    renameFullyQualifiedNamesToUnqualifiedNames(newResultTable);
    /* add new qualified attribute names based on new table name */
    expandUnqualifiedColumnNamesToQualifiedColumnNames(newResultTable);
  }

  // at first check for different number of rows
  bool sameNumberOfRows = (newResultTable->getNumberofRows() ==
                           oldResultTableCpu->getNumberofRows());
  if (!sameNumberOfRows) {
    std::cout << "Both tables have a different amount of rows: "
              << newResultTable->getNumberofRows() << " vs "
              << oldResultTableCpu->getNumberofRows() << std::endl;
  }
  ASSERT_TRUE(sameNumberOfRows);

  bool sameSchema = isSameTableSchema(newResultTable->getSchema(),
                                      oldResultTableCpu->getSchema());

  if (!sameSchema | true) {
    std::cout << "The schemas differ." << std::endl;

    TableSchema refSchema = oldResultTableCpu->getSchema();
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
  bool equal =
      BaseTable::approximatelyEquals(oldResultTableCpu, newResultTable);

  if (!equal) {
    std::cout << "The data of the two tables do not match" << std::endl;
  }

  ASSERT_TRUE(equal);

#ifdef ENABLE_GPU_ACCELERATION
  if (!equal) {
    if (!quiet && debug)
      std::cout << "CPU result differs. Trying with GPU result." << std::endl;

    // try with GPU Results
    TablePtr oldResultTableGpu =
        createTableFromFile(GetParam().second, "gpu", "results/gpu");
    removeTableFromGlobalTableList(oldResultTableGpu);
    equal =
        BaseTable::approximatelyEquals(oldResultTableGpu, newResultTable, 20.0);
  }
#endif

  // delete the recently created table. this is a hack due to Issue #36
  //        globalTables.pop_back();

  ASSERT_TRUE(equal);
}

}  // end namespace
