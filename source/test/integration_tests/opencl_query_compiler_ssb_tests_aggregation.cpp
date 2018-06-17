#include "opencl_query_compiler_ssb_tests.hpp"

namespace CoGaDB {

INSTANTIATE_TEST_CASE_P(
    SSB_Query_Compiler_Aggregation_TestSuite, OpenCLQueryCompilerSSBSingleTest,
    ::testing::Combine(
        ::testing::ValuesIn(getFilenamesFromDir(
            getTestDataPath() + "/ssb/opencl_query_compiler/sql/aggregation/")),
        ::testing::Values(std::string("aggregation/")),
        ::testing::ValuesIn(std::vector<std::string>{"cpu", "dgpu", "igpu"}),
        ::testing::Values(getAggregationVariantSpace())));

}  // end namespace
