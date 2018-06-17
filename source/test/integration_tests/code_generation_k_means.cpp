
#include <fstream>
#include <iostream>

#include <dlfcn.h>
#include <stdlib.h>

#include <core/global_definitions.hpp>

#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generators/cpp_code_generator.hpp>

#include <core/selection_expression.hpp>
#include <parser/commandline_interpreter.hpp>
#include <util/time_measurement.hpp>

#include <boost/make_shared.hpp>
#include <iomanip>
#include <util/getname.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>

#include <limits>
#include <query_compilation/user_defined_code.hpp>
#include <util/tests.hpp>

#include <core/column.hpp>
#include <core/variable_manager.hpp>

#include <parser/json_parser.hpp>

using namespace CoGaDB;

std::string directory_containing_reference_results =
    PATH_TO_DATA_OF_TESTS "/unittests/code_generators";

size_t getNumberOfDimensionsFromFile(const std::string& path_to_data_csv,
                                     const std::string& separators) {
  std::ifstream fin(path_to_data_csv.c_str());  //"Makefile");
  std::string buffer;
  size_t number_of_dimensions = 0;

  if (!fin.is_open()) {
    std::cout << "Error: could not open file " << path_to_data_csv << std::endl;
    return false;
  }

  while (fin.good()) {
    getline(fin, buffer, '\n');
    /* ignore lines that start with '#' */
    if (!buffer.empty()) {
      if (buffer[0] == '#') continue;
    }
    std::vector<std::string> tokens;
    boost::split(tokens, buffer, boost::is_any_of(separators));
    number_of_dimensions = tokens.size();
    break;
  }
  fin.close();
  return number_of_dimensions;
}

const std::string getPointTableName(size_t iteration) {
  return std::string("POINTS_") + boost::lexical_cast<std::string>(iteration);
}

const std::string getCentroidTableName(size_t iteration) {
  return std::string("CENTROIDS_") +
         boost::lexical_cast<std::string>(iteration);
}

const std::string getPointAttributeName(size_t iteration, size_t dimension) {
  return getPointTableName(iteration) + std::string(".P_VALUE_") +
         boost::lexical_cast<std::string>(dimension);
}

const std::string getCentroidAttributeName(size_t iteration, size_t dimension) {
  return getCentroidTableName(iteration) + std::string(".C_VALUE_") +
         boost::lexical_cast<std::string>(dimension);
}

void create_initial_centroids(size_t k, size_t number_of_dimensions) {
  srand(0);

  TableSchema centroid_schema;
  centroid_schema.push_back(Attribut(INT, "CID"));
  for (size_t dimension = 0; dimension < number_of_dimensions; ++dimension) {
    centroid_schema.push_back(Attribut(
        DOUBLE,
        std::string("C_VALUE_") + boost::lexical_cast<std::string>(dimension)));
  }

  TablePtr centroids(new Table("CENTROIDS", centroid_schema));

  TablePtr points = getTablebyName("POINTS");
  assert(points != NULL);

  /* init centroids */
  for (size_t i = 0; i < k; ++i) {
    Tuple t;
    t.push_back(boost::any(int(i)));
    size_t random_point_id = i;  // % points->getNumberofRows();
    Tuple random_point = points->fetchTuple(random_point_id);
    for (size_t dimension = 0; dimension < number_of_dimensions; ++dimension) {
      t.push_back(random_point[dimension]);
    }
    centroids->insert(t);
  }
  std::cout << "Initial Centroids: " << std::endl;
  centroids->print();
  addToGlobalTableList(centroids);
}

const TablePtr k_means_json(CodeGeneratorType code_generator, size_t k,
                            double threshold,
                            const std::string& path_to_data_csv,
                            const std::string& separators,
                            CommandLineInterpreter& cmd) {
  ClientPtr client(new LocalClient());
  std::pair<bool, TablePtr> result;

  size_t iteration = 0;
  double change = std::numeric_limits<double>::max();

  std::string plan_dir_path =
      std::string(PATH_TO_DATA_OF_TESTS) + "/example_json_plans/k_means_plans/";

  VariableManager::instance().setVariableValue("query_execution_policy",
                                               "compiled");
  RuntimeConfiguration::instance().setOptimizer("no_join_order_optimizer");
  VariableManager::instance().setVariableValue("debug_code_generator", "true");

  std::cout << "Testing C_CodeGenerator..." << std::endl;
  //  VariableManager::instance().setVariableValue("default_code_generator",
  //  "c");
  VariableManager::instance().setVariableValue("default_code_generator",
                                               "multi_staged");

  /* replace the template path in the file with the real one */
  std::string plan_load_points =
      readFileContent(plan_dir_path + "query_plan_load_points.json");

  boost::replace_all(plan_load_points, "#PLACE_HOLDER_PATH_TO_DATA_FILE#",
                     path_to_data_csv);
  /* write updated json query plan to current working dir */
  std::ofstream file_load_points_json("query_plan_load_points.json",
                                      std::ofstream::out);
  file_load_points_json << plan_load_points;
  file_load_points_json.close();

  cmd.execute("drop_table CENTROIDS", client);
  cmd.execute("drop_table POINTS", client);
  cmd.execute("execute_query_from_json query_plan_load_points.json", client);
  /* 4 clusters, 2 dimensions */
  create_initial_centroids(4, 2);

  while (change > threshold) {
    cmd.execute(std::string("execute_query_from_json ") + plan_dir_path +
                    "query_plan_compute_clustering.json",
                client);
    cmd.execute(std::string("execute_query_from_json ") + plan_dir_path +
                    "query_plan_compute_new_centroids.json",
                client);
    cmd.execute(std::string("execute_query_from_json ") + plan_dir_path +
                    "query_plan_compute_change_of_centroids.json",
                client);
    cmd.execute("drop_table CENTROIDS", client);
    cmd.execute("drop_table CLUSTERED_POINTS", client);
    cmd.execute(std::string("execute_query_from_json ") + plan_dir_path +
                    "query_plan_rename_computed_centroids.json",
                client);
    cmd.execute(std::string("execute_query_from_json ") + plan_dir_path +
                    "query_plan_delete_new_centroids.json",
                client);

    std::string change_str = readFileContent("computed_change_tmp.csv");

    std::istringstream stream(change_str);
    std::string last_line;
    std::getline(stream, last_line);
    std::getline(stream, last_line);

    std::cout << "Computed change: '" << last_line << "'" << std::endl;
    change = boost::lexical_cast<double>(last_line);

    iteration++;
  }

  cmd.execute(std::string("execute_query_from_json ") + plan_dir_path +
                  "query_plan_export_centroids.json",
              client);
  std::string final_centroids = readFileContent("computed_centroids.csv");
  std::cout << "Final Centroids: " << std::endl;
  std::cout << final_centroids << std::endl;

  TablePtr centroids = getTablebyName("CENTROIDS");
  assert(centroids != NULL);

  return centroids;
}

int main(int argc, char* argv[]) {
  /* defines which code generator should be used */
  CodeGeneratorType code_generator =
      MULTI_STAGE_CODE_GENERATOR;  // C_CODE_GENERATOR;

  size_t k = 2;
  std::string path_to_data =
      std::string(PATH_TO_DATA_OF_TESTS) +
      "/example_json_plans/k_means_data/test_data_2_dimensions_4_clusters.csv";

  std::cout << "Unittests for K-Means" << std::endl;

  /* read instructions to load database from config file */
  ClientPtr client(new LocalClient());
  CommandLineInterpreter cmd(client);

  TablePtr newResultTable =
      k_means_json(code_generator, k, 0.05, path_to_data, "|", cmd);

  TableSchema centroid_schema;
  centroid_schema.push_back(Attribut(OID, "CID"));
  for (size_t dimension = 0; dimension < 2; ++dimension) {
    centroid_schema.push_back(Attribut(
        DOUBLE,
        std::string("C_VALUE_") + boost::lexical_cast<std::string>(dimension)));
  }

  /* hardcode reference result for now */
  TablePtr oldResultTable(new Table("CENTROIDS", centroid_schema));
  {
    Tuple t;
    t.push_back(int32_t(0));
    t.push_back(double(0.412));
    t.push_back(double(0.449));
    oldResultTable->insert(t);
  }
  {
    Tuple t;
    t.push_back(int32_t(1));
    t.push_back(double(0.735));
    t.push_back(double(0.727));
    oldResultTable->insert(t);
  }
  {
    Tuple t;
    t.push_back(int32_t(2));
    t.push_back(double(0.270));
    t.push_back(double(0.740));
    oldResultTable->insert(t);
  }
  {
    Tuple t;
    t.push_back(int32_t(3));
    t.push_back(double(0.605));
    t.push_back(double(0.489));
    oldResultTable->insert(t);
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
  std::list<SortAttribute> sort_attributes;
  sort_attributes.push_back(SortAttribute("CENTROIDS.CID", ASCENDING));
  newResultTable = BaseTable::sort(newResultTable, sort_attributes);

  bool equal = BaseTable::approximatelyEquals(oldResultTable, newResultTable);
  if (!equal) {
    std::cout << "Reference Result:" << std::endl;
    oldResultTable->print();
    std::cout << "Computed Result: " << std::endl;
    newResultTable->print();
    std::cout << "Error: The clustering result is NOT correct!" << std::endl;
#ifndef __APPLE__
    quick_exit(-1);
#endif
    return -1;
  } else {
    std::cout << "The clustering result is correct!" << std::endl;
#ifndef __APPLE__
    quick_exit(0);
#endif
    return 0;
  }
}
