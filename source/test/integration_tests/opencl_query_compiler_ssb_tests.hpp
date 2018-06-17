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
#include "util/variant_configurator.hpp"

#include <array>

namespace CoGaDB {

  /* \brief Verify result table
   * \param newResultTable Computed table
   * \param oldResulTable Expected table
   */
  void verifyResultTable(TablePtr newResultTable, TablePtr oldResultTable) {
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
    bool sameNumberOfRows = (newResultTable->getNumberofRows() ==
                             oldResultTable->getNumberofRows());
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

      for (std::list<Attribut>::const_iterator
               it = refSchema.begin(),
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

  void testResultCorrect(const std::string &deviceDir,
                         const std::string &sqlSubDir,
                         const std::string &testname) {
    TablePtr computedResult =
        assembleTableFromQuery(deviceDir, sqlSubDir + testname);
    TablePtr expectedResult =
        createTableFromFile(deviceDir, testname, "results");

    if (!expectedResult) {
      COGADB_FATAL_ERROR("Failed to load table from file " << testname, "");
    }

    verifyResultTable(computedResult, expectedResult);
  }

  VariantIterator getProjectionDenseVariantSpace() {
    VariantIterator vit;

    auto &exec_dimension =
        vit.add("code_gen.pipe_exec_strategy",
                {"parallel_global_atomic_single_pass", "serial_single_pass",
                 "parallel_three_pass"});

    exec_dimension.addChilds(
        {"parallel_global_atomic_single_pass", "parallel_three_pass"},
        "code_gen.projection.global_size_multiplier",
        {"1", "8", "64", "256", "1024", "16384", "65536"});

    vit.add("code_gen.memory_access", {"coalesced", "sequential"});
    vit.add("code_gen.opt.enable_predication", {"false", "true"});

    return vit;
  }

  VariantIterator getProjectionSparseVariantSpace() {
    VariantIterator vit;

    auto &exec_dimension =
        vit.add("code_gen.pipe_exec_strategy",
                {"parallel_global_atomic_single_pass", "serial_single_pass",
                 "parallel_three_pass"});

    exec_dimension.addChilds(
        {"parallel_global_atomic_single_pass", "parallel_three_pass"},
        "code_gen.projection.global_size_multiplier", {"1", "256", "65536"});

    vit.add("code_gen.memory_access", {"coalesced", "sequential"});
    vit.add("code_gen.opt.enable_predication", {"false", "true"});

    return vit;
  }

  VariantIterator getAggregationVariantSpace() {
    VariantIterator vit;

    vit.add("code_gen.memory_access", {"coalesced", "sequential"});
    vit.add("code_gen.opt.enable_predication", {"false", "true"});

    return vit;
  }

  VariantIterator getGroupedAggregationDenseVariantSpace() {
    VariantIterator vit;

    auto &grouped_agg_dimension =
        vit.add("code_gen.opt.ocl_grouped_aggregation_strategy",
                {"atomic_workgroup", "atomic", "semaphore", "sequential"});

    grouped_agg_dimension.addChilds(
        {"atomic_workgroup"},
        "code_gen.opt.ocl_grouped_aggregation.atomic."
        "workgroup.local_size",
        {"32", "128", "512", "1024"});

    grouped_agg_dimension.addChilds(
        {"atomic_workgroup", "atomic", "semaphore"},
        "code_gen.opt.ocl_grouped_aggregation.global_size_multiplier",
        {"1", "8", "64", "256", "1024", "16384", "65536"});

    grouped_agg_dimension.addChilds(
        {"atomic_workgroup", "atomic", "semaphore", "sequential"},
        "code_gen.opt.ocl_grouped_aggregation_hashtable",
        {"cuckoo_hashing", "linear_probing", "quadratic_probing"});

    vit.add("code_gen.memory_access", {"coalesced", "sequential"});
    vit.add("code_gen.opt.enable_predication", {"false", "true"});

    return vit;
  }

  VariantIterator getGroupedAggregationSparseVariantSpace() {
    VariantIterator vit;

    auto &grouped_agg_dimension =
        vit.add("code_gen.opt.ocl_grouped_aggregation_strategy",
                {"atomic_workgroup", "atomic", "semaphore", "sequential"});

    grouped_agg_dimension.addChilds(
        {"atomic_workgroup"},
        "code_gen.opt.ocl_grouped_aggregation.atomic."
        "workgroup.local_size",
        {"32", "1024"});

    grouped_agg_dimension.addChilds(
        {"atomic_workgroup", "atomic", "semaphore"},
        "code_gen.opt.ocl_grouped_aggregation.global_size_multiplier",
        {"1", "256", "65536"});

    grouped_agg_dimension.addChilds(
        {"atomic_workgroup", "atomic", "semaphore", "sequential"},
        "code_gen.opt.ocl_grouped_aggregation_hashtable",
        {"cuckoo_hashing", "linear_probing", "quadratic_probing"});

    vit.add("code_gen.memory_access", {"coalesced", "sequential"});
    vit.add("code_gen.opt.enable_predication", {"false", "true"});

    return vit;
  }

  void runTest(const std::string &file, const std::string &sqlSubdir,
               const std::string &device, VariantIterator vit) {
    std::cout << "Running OpenCL Query Compiler SSB Test for query " << file
              << " in subdirectory " << sqlSubdir << " on device " << device
              << std::endl;

    auto const &available_devices =
        OCL_Runtime::instance().getAvailableDeviceTypes();
    if (std::find(available_devices.begin(), available_devices.end(), device) ==
        available_devices.end()) {
      std::cout << "Skipping test because no " << device << " available"
                << std::endl;
      return;
    }

    VariableManager::instance().setVariableValue("query_execution_policy",
                                                 "compiled");
    RuntimeConfiguration::instance().setOptimizer("no_join_order_optimizer");
    VariableManager::instance().setVariableValue("debug_code_generator",
                                                 "false");
    VariableManager::instance().setVariableValue("print_query_result", "false");

    std::cout << "Testing Multi_Staged CodeGenerator..." << std::endl;
    VariableManager::instance().setVariableValue("default_code_generator",
                                                 "multi_staged");
    VariableManager::instance().setVariableValue("code_gen.exec_strategy",
                                                 "opencl");
    VariableManager::instance().setVariableValue("default_hash_table",
                                                 "ocl_linear_probing");

    // Variants
    VariantConfigurator vc;

    vit.add("code_gen.cl_device_type", {device});

    std::stringstream num_threads;
    num_threads << boost::thread::hardware_concurrency();

    for (const auto &variant : vit) {
      vc(variant);

      if (VariableManager::instance().getVariableValueString(
              "code_gen.pipe_exec_strategy") ==
          "parallel_global_atomic_single_pass") {
        VariableManager::instance().setVariableValue("code_gen.num_threads",
                                                     "1");
      } else {
        VariableManager::instance().addVariable(
            "code_gen.num_threads",
            VariableState(num_threads.str(), INT, checkStringIsInteger));
      }

      testResultCorrect("ssb/opencl_query_compiler/", sqlSubdir, file);
    }
  }

  /* \brief OpenCL Query Compiler Test Suite with a single variant space.
   */
  class OpenCLQueryCompilerSSBSingleTest
      : public testing::TestWithParam<std::tuple<
            std::string, std::string, std::string, VariantIterator>> {
   public:
    static void SetUpTestCase() {
      /* depending on join order, aggregation results can slighly change,
       * so we need to ensure that we use always the same optimizer pipeline */
      RuntimeConfiguration::instance().setOptimizer("default_optimizer");
      init();
      ClientPtr client = boost::make_shared<LocalClient>();
      loadReferenceDatabaseStarSchemaScaleFactor1(client);
    }

    static void TearDownTestCase() {
      std::vector<TablePtr> &globalTables = CoGaDB::getGlobalTableList();
      globalTables.clear();
      CoGaDB::exit(0);
    }
  };

  TEST_P(OpenCLQueryCompilerSSBSingleTest, OpenCLQueryCompilerSSBTest) {
    // Return type of `GetParam()` is derived from
    // `OpenCLQueryCompilerSSBSingleTest`:
    //   ::std::tuple<std::string, std::string, std::string, VariantIterator>
    // Each element of the tuple is passed directly to `runTest()`:
    std::cout << "Running OpenCL Query Compiler SSB Test"
              << " for query " << std::get<0>(GetParam()) << " in subdirectory "
              << std::get<1>(GetParam()) << " on device "
              << std::get<2>(GetParam()) << std::endl;
    runTest(std::get<0>(GetParam()), std::get<1>(GetParam()),
            std::get<2>(GetParam()), std::get<3>(GetParam()));
  }

  /* \brief OpenCL Query Compiler Test Suite with a dense and a sparse variant
   * space.
   * Test suites being instantiated using this template are expected to have a
   * large
   * variant space. Such a large variant space results in long test runtimes
   * which is
   * why we split this space into a sparse and a dense part where the dense is
   * disabled
   * by default and only executed if the command line flag
   * `--gtest_also_run_disabled_tests`
   * is passed to the test binary.
   */
  class OpenCLQueryCompilerSSBTest
      : public testing::TestWithParam<
            ::std::tuple<std::string, std::string, std::string,
                         std::array<VariantIterator, 2>>> {
   public:
    static void SetUpTestCase() {
      /* depending on join order, aggregation results can slighly change,
       * so we need to ensure that we use always the same optimizer pipeline */
      RuntimeConfiguration::instance().setOptimizer("default_optimizer");
      init();
      ClientPtr client = boost::make_shared<LocalClient>();
      loadReferenceDatabaseStarSchemaScaleFactor1(client);
    }

    static void TearDownTestCase() {
      std::vector<TablePtr> &globalTables = CoGaDB::getGlobalTableList();
      globalTables.clear();
      CoGaDB::exit(0);
    }
  };

  TEST_P(OpenCLQueryCompilerSSBTest,
         DISABLED_OpenCLQueryCompilerSSBTest_Dense) {
    // Return type of `GetParam()` is derived from `OpenCLQueryCompilerSSBTest`:
    //   ::std::tuple<std::string, std::string, std::string,
    //                std::array<VariantIterator, 2>>
    // As fourth argument we do pass the first array element as we expect the
    // dense variant space here:
    std::cout << "Running OpenCL Query Compiler SSB Test with dense variant "
              << "space for query " << std::get<0>(GetParam())
              << " in subdirectory " << std::get<1>(GetParam()) << " on device "
              << std::get<2>(GetParam()) << std::endl;
    runTest(std::get<0>(GetParam()), std::get<1>(GetParam()),
            std::get<2>(GetParam()), std::get<3>(GetParam()).at(0));
  }

  //<<<<<<< local
  //  verifyResultTable(computedResult, expectedResult);
  //}

  // VariantIterator getProjectionDenseVariantSpaceCPU() {
  //  VariantIterator vit;

  //  auto &exec_dimension =
  //      vit.add("code_gen.pipe_exec_strategy",
  //              {"parallel_global_atomic_single_pass", "serial_single_pass",
  //               "parallel_three_pass"});

  //  exec_dimension.addChilds(
  //      {"parallel_global_atomic_single_pass", "parallel_three_pass"},
  //      "code_gen.projection.cpu.global_size",
  //      { "1", "2", "16", "32" });

  //  vit.add("code_gen.memory_access", {"coalesced", "sequential"});
  //  vit.add("code_gen.opt.enable_predication", {"false", "true"});

  //  return vit;
  //}

  // VariantIterator getProjectionSparseVariantSpaceCPU() {
  //  VariantIterator vit;

  //  auto &exec_dimension =
  //      vit.add("code_gen.pipe_exec_strategy",
  //              {"parallel_global_atomic_single_pass", "serial_single_pass",
  //               "parallel_three_pass"});

  //  exec_dimension.addChilds(
  //      {"parallel_global_atomic_single_pass", "parallel_three_pass"},
  //      "code_gen.projection.cpu.global_size",
  //      { "1", "32" });

  //  vit.add("code_gen.memory_access", {"coalesced", "sequential"});
  //  vit.add("code_gen.opt.enable_predication", {"false", "true"});

  //  return vit;
  //}

  // VariantIterator getProjectionDenseVariantSpaceGPU() {
  //  VariantIterator vit;

  //  auto &exec_dimension =
  //      vit.add("code_gen.pipe_exec_strategy",
  //              {"parallel_global_atomic_single_pass", "serial_single_pass",
  //               "parallel_three_pass"});

  //  exec_dimension.addChilds(
  //      {"parallel_global_atomic_single_pass", "parallel_three_pass"},
  //      "code_gen.projection.gpu.global_size",
  //      { "5000", "500000" });

  //  vit.add("code_gen.memory_access", {"coalesced", "sequential"});
  //  vit.add("code_gen.opt.enable_predication", {"false", "true"});

  //  return vit;
  //}

  // VariantIterator getProjectionSparseVariantSpaceGPU() {
  //  VariantIterator vit;

  //  auto &exec_dimension =
  //      vit.add("code_gen.pipe_exec_strategy",
  //              {"parallel_global_atomic_single_pass", "serial_single_pass",
  //               "parallel_three_pass"});

  //  exec_dimension.addChilds(
  //      {"parallel_global_atomic_single_pass", "parallel_three_pass"},
  //      "code_gen.projection.gpu.global_size",
  //      { "1000", "5000", "500000", "1000000" });

  //  vit.add("code_gen.memory_access", {"coalesced", "sequential"});
  //  vit.add("code_gen.opt.enable_predication", {"false", "true"});

  //  return vit;
  //}

  // VariantIterator getAggregationVariantSpace() {
  //  VariantIterator vit;

  //  vit.add("code_gen.memory_access", {"coalesced", "sequential"});
  //  vit.add("code_gen.opt.enable_predication", {"false", "true"});

  //  return vit;
  //}

  // VariantIterator getGroupedAggregationDenseVariantSpaceCPU() {
  //  VariantIterator vit;

  //  auto &grouped_agg_dimension =
  //      vit.add("code_gen.opt.ocl_grouped_aggregation_strategy",
  //              {"atomic_workgroup", "atomic", "semaphore", "sequential"});

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup"},
  //      "code_gen.opt.ocl_grouped_aggregation.atomic."
  //          "workgroup.local_size",
  //      { "1", "32", "512", "1024" });

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup", "atomic", "semaphore"},
  //      "code_gen.opt.ocl_grouped_aggregation.cpu.num_threads_per_kernel",
  //      { "1", "8", "16", "32" });

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup", "atomic", "semaphore", "sequential"},
  //      "code_gen.opt.ocl_grouped_aggregation_hashtable",
  //      {"cuckoo_hashing", "linear_probing", "quadratic_probing"});

  //  vit.add("code_gen.memory_access", {"coalesced", "sequential"});
  //  vit.add("code_gen.opt.enable_predication", {"false", "true"});

  //  return vit;
  //}

  // VariantIterator getGroupedAggregationSparseVariantSpaceCPU() {
  //  VariantIterator vit;

  //  auto &grouped_agg_dimension =
  //      vit.add("code_gen.opt.ocl_grouped_aggregation_strategy",
  //              {"atomic_workgroup", "atomic", "semaphore", "sequential"});

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup"},
  //      "code_gen.opt.ocl_grouped_aggregation.atomic."
  //          "workgroup.local_size",
  //      { "1", "1024" });

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup", "atomic", "semaphore"},
  //      "code_gen.opt.ocl_grouped_aggregation.cpu.num_threads_per_kernel",
  //      {"1", "32"});

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup", "atomic", "semaphore", "sequential"},
  //      "code_gen.opt.ocl_grouped_aggregation_hashtable",
  //      {"cuckoo_hashing", "linear_probing", "quadratic_probing"});

  //  vit.add("code_gen.memory_access", {"coalesced", "sequential"});
  //  vit.add("code_gen.opt.enable_predication", {"false", "true"});

  //  return vit;
  //}

  // VariantIterator getGroupedAggregationDenseVariantSpaceGPU() {
  //  VariantIterator vit;

  //  auto &grouped_agg_dimension =
  //      vit.add("code_gen.opt.ocl_grouped_aggregation_strategy",
  //              {"atomic_workgroup", "atomic", "semaphore", "sequential"});

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup"},
  //      "code_gen.opt.ocl_grouped_aggregation.atomic."
  //          "workgroup.local_size",
  //      { "32", "128", "512", "1024" });

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup", "atomic", "semaphore"},
  //      "code_gen.opt.ocl_grouped_aggregation.gpu.global_size_multiplier",
  //      { "2", "8", "64", "128" });

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup", "atomic", "semaphore", "sequential"},
  //      "code_gen.opt.ocl_grouped_aggregation_hashtable",
  //      {"cuckoo_hashing", "linear_probing", "quadratic_probing"});

  //  vit.add("code_gen.memory_access", {"coalesced", "sequential"});
  //  vit.add("code_gen.opt.enable_predication", {"false", "true"});

  //  return vit;
  //}

  // VariantIterator getGroupedAggregationSparseVariantSpaceGPU() {
  //  VariantIterator vit;

  //  auto &grouped_agg_dimension =
  //      vit.add("code_gen.opt.ocl_grouped_aggregation_strategy",
  //              { "atomic_workgroup", "atomic", "semaphore", "sequential" });

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup"},
  //      "code_gen.opt.ocl_grouped_aggregation.atomic."
  //          "workgroup.local_size",
  //      { "32", "1024" });

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup", "atomic", "semaphore"},
  //      "code_gen.opt.ocl_grouped_aggregation.gpu.global_size_multiplier",
  //      { "2", "128" });

  //  grouped_agg_dimension.addChilds(
  //      {"atomic_workgroup", "atomic", "semaphore", "sequential"},
  //      "code_gen.opt.ocl_grouped_aggregation_hashtable",
  //      {"cuckoo_hashing", "linear_probing"});

  //  vit.add("code_gen.memory_access", {"coalesced", "sequential"});
  //  vit.add("code_gen.opt.enable_predication", {"false", "true"});

  //  return vit;
  //}

  // void runTest(const std::string &file, const std::string &sqlSubdir,
  //             const std::string &device, VariantIterator vit) {
  //  std::cout << "Running OpenCL Query Compiler SSB Test for query " << file
  //            << " in subdirectory " << sqlSubdir
  //            << " on device " << device << std::endl;

  //  auto const &available_devices =
  //      OCL_Runtime::instance().getAvailableDeviceTypes();
  //  if (std::find(available_devices.begin(), available_devices.end(), device)
  //      == available_devices.end()) {
  //    std::cout << "Skipping test because no " << device << " available"
  //              << std::endl;
  //    return;
  //=======
  TEST_P(OpenCLQueryCompilerSSBTest, OpenCLQueryCompilerSSBTest_Sparse) {
    // return type of `GetParam()` is derived from `OpenCLQueryCompilerSSBTest`:
    //   ::std::tuple<std::string, std::string, std::string,
    //                std::array<VariantIterator, 2>>
    // As fourth argument we do pass the first array element as we expect the
    // sparse variant space here:
    std::cout << "Running OpenCL Query Compiler SSB Test with sparse variant "
              << "space for query " << std::get<0>(GetParam())
              << " in subdirectory " << std::get<1>(GetParam()) << " on device "
              << std::get<2>(GetParam()) << std::endl;
    runTest(std::get<0>(GetParam()), std::get<1>(GetParam()),
            std::get<2>(GetParam()), std::get<3>(GetParam()).at(1));
    //>>>>>>> other
  }

}  // end namespace
