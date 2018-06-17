/*
 * File:   tests.cpp
 * Author: sebastian
 *
 * Created on 18. November 2015, 15:51
 */

#include <iostream>
//#include <sql/server/sql_driver.hpp>
//#include <sql/server/sql_parsetree.hpp>
#include <parser/commandline_interpreter.hpp>

#include <fstream>
#include <vector>

#include <boost/filesystem.hpp>

#include <core/runtime_configuration.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>
#include <util/filesystem.hpp>
#include <util/tests.hpp>

namespace CoGaDB {

void SetupTestConfiguration(ClientPtr client) {
  const std::string homeConfigPath = getPathToHomeConfigDir();
  std::string path_to_config_file = homeConfigPath + "test_config.coga";
  // test if startup.coga exists
  std::ifstream iStartUpFile(path_to_config_file.c_str());
  if (!iStartUpFile.good()) {
    if (!boost::filesystem::create_directory(getPathToHomeConfigDir())) {
      // could not create directory check if it already exists and is a
      // directory
      if (!boost::filesystem::exists(homeConfigPath) ||
          !boost::filesystem::is_directory(homeConfigPath)) {
        COGADB_FATAL_ERROR("Failed to create directory '.cogadb' in home dir '"
                               << getPathToHomeConfigDir() << "'!",
                           "");
      }
    }
    std::ofstream oStartUpFile;
    oStartUpFile.open(path_to_config_file.c_str());
    if (oStartUpFile.is_open()) {
      std::cout << "Please enter the location of a star schema database with "
                   "scale factor 1: "
                << std::endl;

      std::string databaseLocation;
      std::getline(std::cin, databaseLocation);

      std::cout << "Your database is located at " << databaseLocation
                << std::endl;

      oStartUpFile << "set path_to_ssb_sf1_database=" << databaseLocation
                   << std::endl;

      std::cout << "Please enter the location of a TPC-H database with scale "
                   "factor 1: "
                << std::endl;

      std::getline(std::cin, databaseLocation);

      std::cout << "Your database is located at " << databaseLocation
                << std::endl;

      oStartUpFile << "set path_to_tpch_sf1_database=" << databaseLocation
                   << std::endl;

      oStartUpFile.close();
    }

    std::cout << "Successfully configured pathes to reference databases. "
              << "Stored config in '" << path_to_config_file << "'..."
              << std::endl;
  }
  /* execute config script */
  CommandLineInterpreter cmd;
  cmd.executeFromFile(path_to_config_file, client);
}

bool loadReferenceDatabaseStarSchemaScaleFactor1(ClientPtr client) {
  std::string path_to_config_file =
      getPathToHomeConfigDir() + "test_config.coga";
  SetupTestConfiguration(client);
  if (VariableManager::instance().getVariableValueString(
          "path_to_ssb_sf1_database") == "") {
    COGADB_FATAL_ERROR(
        "Could not find reference star schema database of scale factor 1. "
            << "Please specify the path to such a database in the variable "
               "'path_to_ssb_sf1_database'"
            << " in the file '" << path_to_config_file << "'",
        "");
  }
  loadTablesFromDirectory(VariableManager::instance().getVariableValueString(
                              "path_to_ssb_sf1_database"),
                          std::cout);
  if (!getTablebyName("LINEORDER")) {
    COGADB_FATAL_ERROR(
        "Did not find table 'LINEORDER', variable "
            << "'path_to_ssb_sf1_database' does not seem to point to star "
               "schema database!"
            << " Edit the file '" << path_to_config_file
            << "' and fix the path for variable 'path_to_ssb_sf1_database'",
        "");
  }
  if (getTablebyName("LINEORDER")->getNumberofRows() != 6001171) {
    COGADB_FATAL_ERROR(
        "Star schema benchmark queries require a scale factor 1 reference "
        "database!",
        "");
  }
  /* set this as path to database, required so that loadColumnFromDisk Works
   * properly  */
  RuntimeConfiguration::instance().setPathToDatabase(
      VariableManager::instance().getVariableValueString(
          "path_to_ssb_sf1_database"));
  return true;
}

bool loadReferenceDatabaseTPCHScaleFactor1(ClientPtr client) {
  std::string path_to_config_file =
      getPathToHomeConfigDir() + "test_config.coga";
  SetupTestConfiguration(client);
  if (VariableManager::instance().getVariableValueString(
          "path_to_tpch_sf1_database") == "") {
    COGADB_FATAL_ERROR(
        "Could not find reference TPC-H database of scale factor 1. "
            << "Please specify the path to such a database in the variable "
               "'path_to_tpch_sf1_database'"
            << " in the file '" << path_to_config_file << "'",
        "");
  }
  loadTablesFromDirectory(VariableManager::instance().getVariableValueString(
                              "path_to_tpch_sf1_database"),
                          std::cout);
  if (!getTablebyName("LINEITEM")) {
    COGADB_FATAL_ERROR(
        "Did not find table 'LINEITEM', variable "
            << "'path_to_tpch_sf1_database' does not seem to point to TPC-H "
               "database!"
            << " Edit the file '" << path_to_config_file
            << "' and fix the path for variable 'path_to_tpch_sf1_database'",
        "");
  }
  if (getTablebyName("LINEITEM")->getNumberofRows() != 6001215) {
    COGADB_FATAL_ERROR(
        "TPC-H benchmark queries require a scale factor 1 reference database!",
        "");
  }
  /* set this as path to database, required so that loadColumnFromDisk Works
   * properly  */
  RuntimeConfiguration::instance().setPathToDatabase(
      VariableManager::instance().getVariableValueString(
          "path_to_tpch_sf1_database"));
  return true;
}
}
