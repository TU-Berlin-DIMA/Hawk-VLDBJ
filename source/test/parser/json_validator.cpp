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
    class JsonValidatorTest : public testing::Test {
    public:
        typedef boost::shared_ptr<TestClient> TestClientPtr;
        TestClientPtr client;
        std::string example_plans_path;
        std::string schema_tests_path;
        std::string schema_file;
        rapidjson::SchemaDocument schema;

        JsonValidatorTest()
            : client(boost::make_shared<TestClient>()),
              example_plans_path(std::string(PATH_TO_COGADB_EXECUTABLE)
                                 .append("/test/testdata/example_json_plans/")),
              schema_tests_path(std::string(PATH_TO_COGADB_EXECUTABLE)
                                .append("/test/testdata/json_schema_tests/")),
              schema_file(std::string(PATH_TO_COGADB_EXECUTABLE)
                          .append("/share/cogadb/parser/json/query_plan_schema.json")),
              schema(load_schema_from_file(schema_file)) {
        }

        virtual ~JsonValidatorTest() {
        }

        virtual void SetUp() {
        }

        virtual void TearDown() {
        }
    };

    //@formatter:off
    TEST_F(JsonValidatorTest, GroupByPlanValidationTest) {
        auto const& document = load_json_from_file(
            example_plans_path.append("test_query_plan_groupby.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, JoinPlanValidationTest) {
        auto const& document = load_json_from_file(
            example_plans_path.append("test_query_plan_join.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, MapUdfPlanValidationTest) {
        auto const& document = load_json_from_file(
            example_plans_path.append("test_query_plan_map_udf.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, SelectionPlanValidationTest) {
        auto const& document = load_json_from_file(
            example_plans_path.append("test_query_plan_selection.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, SortCreateTablePlanValidationTest) {
        auto const& document = load_json_from_file(
            example_plans_path.append("test_query_plan_sort_create_table.json"));
        validate_jsondocument(document, schema);
    }


    TEST_F(JsonValidatorTest, CreateTableSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("create_table.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, ExportIntoFileSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("export_into_file.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericGroupbyBothSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_groupby_both.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericGroupbySumSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_groupby_sum.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericGroupbyUdfSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_groupby_udf.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericJoinSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_join.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericSelectionSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_selection.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericSelection2SchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_selection2.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericSelection3SchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_selection3.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericSelectionBothSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_selection_both.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericSelectionColumnSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_selection_column.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, GenericSelectionConstantSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("generic_selection_constant.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, MapUdfSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("map_udf.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, ProjectionSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("projection.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, Projection2SchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("projection2.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, Projection3SchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("projection3.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, SortBySchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("sort_by.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, StoreTableSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("store_table.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, StoreTable2SchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("store_table2.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, StoreTable3SchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("store_table3.json"));
        validate_jsondocument(document, schema);
    }

    TEST_F(JsonValidatorTest, TableScanSchemaValidationTest) {
        auto const& document = load_json_from_file(
            schema_tests_path.append("table_scan.json"));
        validate_jsondocument(document, schema);
    }
    //@formatter:on

} // end namespace
