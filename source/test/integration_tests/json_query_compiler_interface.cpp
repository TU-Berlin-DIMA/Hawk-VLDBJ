
#include <iostream>

#include <persistence/storage_manager.hpp>

#include <fstream>
#include <vector>

#include <core/runtime_configuration.hpp>
#include <util/tests.hpp>
#include "generic_regression_tests.hpp"

#include <gtest/gtest.h>

#include <boost/algorithm/string.hpp>
#include <parser/client.hpp>
#include <parser/json_parser.hpp>
#include "generic_regression_tests.cpp"

namespace CoGaDB {

struct TestParameter {
  TestParameter(std::string _json_query_file, TablePtr _reference_result)
      : json_query_file(_json_query_file),
        reference_result(_reference_result) {}
  std::string json_query_file;
  TablePtr reference_result;
};

class JSON_Interface_Test : public testing::TestWithParam<TestParameter> {
 public:
  /*
  *  will be called before the testcase will be executed
  */
  static void SetUpTestCase() {
    std::cout << "JSON_Interface_Test: StartUpTestCase" << std::endl;
    CoGaDB::init();
    ClientPtr my_client(new CoGaDB::LocalClient());
    loadReferenceDatabaseTPCHScaleFactor1(my_client);
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
};

const std::vector<TestParameter> getJSONTestQueries() {
  std::stringstream path;
  path << getTestDataPath() << "/tpch/"
       << "json_query_compiler_interface";
  std::stringstream path_json_plans;
  std::stringstream path_reference_results;
  path_json_plans << path.str() << "/json_query_plans";
  path_reference_results << path.str() << "/reference_results";
  std::vector<std::string> filenames =
      getFilenamesFromDir(path_json_plans.str());

  std::vector<TestParameter> test_parameters;
  for (auto file : filenames) {
    std::stringstream json_file_path;
    std::stringstream result_file_path;
    json_file_path << path_json_plans.str() << "/" << file << ".json";
    result_file_path << path_reference_results.str() << "/" << file << ".csv";

    TablePtr reference_result =
        loadTableFromSelfContainedCSV(result_file_path.str());
    assert(reference_result != NULL && "Reference table not found!");
    TestParameter t(json_file_path.str(), reference_result);
    test_parameters.push_back(t);
  }
  return test_parameters;
}

bool isEqual(TablePtr newResultTable, TablePtr oldResultTable) {
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
    return false;
  }

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
    return false;
  }
  // now that we know schema and row count matches we can call approx. equals
  // for the column-data
  bool equal = BaseTable::approximatelyEquals(oldResultTable, newResultTable);

  if (!equal) {
    std::cout << "The data of the two tables do not match" << std::endl;
    return false;
  }
  return true;
}

INSTANTIATE_TEST_CASE_P(TestSuite_JSON_Interface, JSON_Interface_Test,
                        ::testing::ValuesIn(getJSONTestQueries()));

TEST_P(JSON_Interface_Test, JSON_Interface_Test_instance) {
  std::cout << "Running test for file " << GetParam().json_query_file
            << std::endl;

  VariableManager::instance().setVariableValue("query_execution_policy",
                                               "compiled");
  VariableManager::instance().setVariableValue("default_code_generator",
                                               "multi_staged");
  VariableManager::instance().setVariableValue("code_gen.exec_strategy",
                                               "opencl");
  VariableManager::instance().setVariableValue("default_hash_table",
                                               "ocl_linear_probing");

  ClientPtr my_client(new CoGaDB::LocalClient());
  std::pair<bool, TablePtr> ret;
  ret = CoGaDB::load_and_execute_query_from_json(GetParam().json_query_file,
                                                 my_client);
  ASSERT_TRUE(ret.first);

  TablePtr newResultTable = ret.second;
  TablePtr oldResultTable = GetParam().reference_result;
  assert(newResultTable != NULL);
  assert(oldResultTable != NULL);

  bool equal = isEqual(newResultTable, oldResultTable);
  if (!equal) {
    COGADB_FATAL_ERROR("Incorrect result for query '"
                           << GetParam().json_query_file << "'",
                       "");
  }
  ASSERT_TRUE(equal);
}
}  // end namespace
