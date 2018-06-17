#include <sstream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "gtest/gtest.h"
#include "gtest/gtest-spi.h"

#include "parser/client.hpp"
#include "parser/json_parser.hpp"

namespace CoGaDB {

    /* \brief Test client */
    class TestClient : public Client{
    public:
        virtual std::string getNextCommand() {
            return "";
        }

        virtual std::ostream& getOutputStream() {
            return ss;
        };

        virtual std::istream& getInputStream() {
            return ss;
        }

        virtual std::stringstream& getStringStream() {
            return ss;
        }

    private:
        std::stringstream ss;
    };

   /* \brief Test fixture for JSON parser tests */
    class JsonParserTest : public ::testing::Test {
    public:
        typedef boost::shared_ptr<TestClient> TestClientPtr;
        TestClientPtr client;
        std::string data_path;

        JsonParserTest()
            : client(boost::make_shared<TestClient>()),
              data_path(std::string(PATH_TO_COGADB_EXECUTABLE)
                        .append("/test/testdata/example_json_plans/")) {
        }

        virtual ~JsonParserTest() {
        }

        virtual void SetUp() {
        }

        virtual void TearDown() {
        }

        /* \brief Assert that is contains s at the beginning */
        void assert_istream_starts_with(std::istream const& is, std::string s) {
            std::string message;
            auto const& l = s.length();
            message.resize(l);
            client->getInputStream().read(&message[0], l);
            ASSERT_EQ(message, s);
        }
    };

    //@formatter:off
    TEST_F(JsonParserTest, EmptyFilePathTest) {
        auto const& plan = import_query_from_json("", client);
        ASSERT_EQ(plan, nullptr);
        assert_istream_starts_with(client->getInputStream(),
                                   "Error: File");
    }

    TEST_F(JsonParserTest, GroupByPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("test_query_plan_groupby.json"));
    }

    TEST_F(JsonParserTest, JoinPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("test_query_plan_join.json"));
    }

    TEST_F(JsonParserTest, MapUdfPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("test_query_plan_map_udf.json"));
    }

    TEST_F(JsonParserTest, SelectionPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("test_query_plan_selection.json"));
    }

    TEST_F(JsonParserTest, SortCreateTablePlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("test_query_plan_sort_create_table.json"));
    }

    TEST_F(JsonParserTest, KMeansComputeChangeOfCentroidsPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_compute_change_of_centroids.json"));
    }

    TEST_F(JsonParserTest, KMeansComputeClusteringPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_compute_clustering.json"));
    }

    TEST_F(JsonParserTest, KMeansComputeNewCentroidsPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_compute_new_centroids.json"));
    }

    TEST_F(JsonParserTest, KMeansCreateCentroidsPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_create_centroids.json"));
    }

    TEST_F(JsonParserTest, KMeansDeleteClusteredPointsPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_delete_clustered_points.json"));
    }

    TEST_F(JsonParserTest, KMeansDeleteNewCentroidsPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_delete_new_centroids.json"));
    }

    TEST_F(JsonParserTest, KMeansDeleteOldCentroidsPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_delete_old_centroids.json"));
    }

    TEST_F(JsonParserTest, KMeansExportCentroidsPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_export_centroids.json"));
    }

    TEST_F(JsonParserTest, KMeansLoadPointsPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_load_points.json"));
    }

    TEST_F(JsonParserTest, KMeansRenameComputedCentroidsPlanTest) {
        auto const& document = load_json_from_file(
            data_path.append("k_means_plans/")
            .append("query_plan_rename_computed_centroids.json"));
    }
    //@formatter:on

} // end namespace
