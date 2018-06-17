#include "opencl_query_compiler_ssb_tests.hpp"

namespace CoGaDB {

INSTANTIATE_TEST_CASE_P(
    SSB_Query_Compiler_CPU_GroupedAggregation_TestSuite,
    OpenCLQueryCompilerSSBTest,
    ::testing::Combine(
        ::testing::ValuesIn(getFilenamesFromDir(
            getTestDataPath() +
            "/ssb/opencl_query_compiler/sql/grouped_aggregation/")),
        ::testing::Values(std::string("grouped_aggregation/")),
        ::testing::ValuesIn(std::vector<std::string>{"cpu", "dgpu", "igpu"}),
        ::testing::Values(std::array<VariantIterator, 2>{
            {getGroupedAggregationDenseVariantSpace(),
             getGroupedAggregationSparseVariantSpace()}})));

}  // end namespace
