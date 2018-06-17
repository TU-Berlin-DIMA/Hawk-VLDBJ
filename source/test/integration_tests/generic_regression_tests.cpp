#include <boost/filesystem.hpp>
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

#include "generic_regression_tests.hpp"
#include "generic_ssb_tests.hpp"
#include "generic_tpch_tests.hpp"
#include "gtest/gtest.h"

namespace CoGaDB {

const std::string CLI_PARAM_TABLES_TO_CSV = "--write-table-schema";
bool writeTablesToCSV = false;
std::string writeTablesToCSVPath = "";

const ClientPtr GenericRegressionTest::client = ClientPtr(new LocalClient());

/**
* Creates a blacklist of queries which are currently not working correctly
**/
const std::set<std::string> createBlacklistSet() {
  std::set<std::string> blacklist;

  blacklist.insert("TPCH_Q01");
  blacklist.insert("TPCH_Q02");
  blacklist.insert("TPCH_Q07");
  blacklist.insert("TPCH_Q15");
  blacklist.insert("TPCH_Q18");
  blacklist.insert("TPCH_Q20");

  return blacklist;
}

const std::string getTestDataPath() {
  return std::string(PATH_TO_COGADB_EXECUTABLE) +
         "/test/testdata/regression test/";
}

// select all files inside the given dirs
const std::vector<std::string> getFilenamesFromDir(const std::string& dirPath) {
  std::cout << "getFilenamesFromDir: " << dirPath << std::endl;
  std::vector<std::string> files;
  DIR* dirP;
  struct dirent* dir;
  struct stat filestatus;

  dirP = opendir(dirPath.c_str());
  if (dirP == NULL) {
    COGADB_ERROR("Could not open the directory "
                     << dirPath << " Errno: " << errno << std::endl,
                 "");
    exit(-1);
  }

  while ((dir = readdir(dirP))) {
    std::string filepath = dirPath + "/" + dir->d_name;
    // check if file is valid and not a directory
    if (stat(filepath.c_str(), &filestatus)) continue;

    if (S_ISDIR(filestatus.st_mode)) continue;

    std::string fileNameRaw(dir->d_name);

    // ignore temp files
    if (fileNameRaw.find('~', fileNameRaw.length() - 1) != std::string::npos)
      continue;
    // ignore files with file extension ".ignored"
    std::string file_extension = boost::filesystem::extension(fileNameRaw);
    if (file_extension == ".ignored") continue;

    auto indexOfFileEnding = fileNameRaw.find_last_of(".");

    if (indexOfFileEnding == std::string::npos) {
      std::cout << "Testdata File " << fileNameRaw
                << " is not corretctly formatted. " << std::endl;
      continue;
    }

    if (indexOfFileEnding >= fileNameRaw.length()) {
      files.push_back(fileNameRaw);
      std::cout << "getFilenamesFromDir: Element " << fileNameRaw << std::endl;
    } else {
      std::string fileName(fileNameRaw.substr(0, indexOfFileEnding));
      files.push_back(fileName);
      std::cout << "getFilenamesFromDir: Element " << fileName << std::endl;
    }
  }

  closedir(dirP);
  return files;
}

void removeTableFromGlobalTableList(TablePtr table) {
  if (!table) return;
  std::vector<TablePtr>& tables = getGlobalTableList();
  std::vector<TablePtr>::iterator it;
  std::cout << "Delete intermediate table: " << std::endl;
  for (it = tables.begin(); it != tables.end(); ++it) {
    std::cout << (*it)->getName() << std::endl;
    if (table->getName() == (*it)->getName()) {
      tables.erase(it);
      std::cout << "Remove Table from Table List: " << table->getName()
                << std::endl;
      break;
    }
  }
}

std::string getFileContent(const std::string& filepath) {
  // returns the content of the file which is only the first line of it

  std::ifstream fin(filepath.c_str());

  if (!fin.is_open()) {
    std::cout << "Error: could not open file " << filepath << std::endl;
    return "";
  }

  std::string buffer;
  getline(fin, buffer, '\n');

  return buffer;
}

TablePtr createTableFromFile(const std::string& deviceDir,
                             const std::string& testname,
                             const std::string& pathToResults) {
  if (debug && !quiet) {
    std::cout << "Creating table from schema file" << std::endl;
  }

  std::string testSchemaFile = getTestDataPath();
  testSchemaFile.append(deviceDir);
  testSchemaFile.append("/schema/").append(testname).append(".sql");
  std::ifstream fin(testSchemaFile.c_str());

  if (!fin.is_open()) {
    std::cout << "Error: could not open file " << testSchemaFile << std::endl;
    return TablePtr();
  }

  std::string buffer;
  getline(fin, buffer, '\n');

  std::cout << "Before executing sql" << std::endl;
  ClientPtr client(new LocalClient());
  TablePtr resultTable = SQL::executeSQL(buffer, client);

  if (debug && !quiet) {
    std::cout << "Created table from schema file. Now loading old results"
              << std::endl;
  }

  if (!resultTable) return TablePtr();

  std::string testDataFile = getTestDataPath();
  testDataFile.append(deviceDir);
  testDataFile.append("/")
      .append(pathToResults)
      .append("/")
      .append(testname)
      .append(".csv");
  if (!resultTable->loadDatafromFile(testDataFile)) {
    COGADB_ERROR("Failed to load data from file: '" << testDataFile << "'", "");
    return TablePtr();
  }

  /* \fixme Fix: by creating the schema with the SQL command create table,
   the temporary table is added to the database table list.This is undesirable,
   as the temporary tables might contain attribute names of other tables,
   and will cause all kinds of errors, because the assumption that column_names
   are unique is broken. For now, we will remove the created table from the
   table list, but this is not the final solution! */
  removeTableFromGlobalTableList(resultTable);

  return resultTable;
}

TablePtr assembleTableFromQuery(const std::string& deviceDir,
                                const std::string& testname) {
  std::string filepath = getTestDataPath();
  filepath.append(deviceDir);
  filepath.append("/sql/").append(testname).append(".sql");

  std::cout << "Trying to assemble table from table from path: " << filepath
            << std::endl;

  std::string query = getFileContent(filepath);

  if (!quiet && debug) std::cout << "Query: " << query << std::endl;

  ClientPtr client(new LocalClient());
  TablePtr resultTable = SQL::executeSQL(query, client);

  if (!resultTable) {
    COGADB_ERROR("Failed to execute query: '" << query << "'", "");
    return resultTable;
  }

  /* \fixme Fix: by creating the schema with the SQL command create table,
   the temporary table is added to the database table list.This is undesirable,
   as the temporary tables might contain attribute names of other tables,
   and will cause all kinds of errors, because the assumption that column_names
   are unique is broken. For now, we will remove the created table from the
   table list, but this is not the final solution! */
  removeTableFromGlobalTableList(resultTable);

  std::cout << "Table from query loaded with " << resultTable->getNumberofRows()
            << " rows " << std::endl;
  std::cout << "Result: " << std::endl << resultTable->toString() << std::endl;

  return resultTable;
}

}  // end namespace

/*    int main(int argc, char **argv) {

        if(argc > 1) {
            std::cout << "Parameter wurde eingegeben: " << argv[1]
                    << " Speicherort lautet: " << argv[2] << std::endl;

            if(CoGaDB::CLI_PARAM_TABLES_TO_CSV.compare(argv[1]) != 0){
                std::cerr << argv[1] << " unknown parameter. " <<
                        "The input parameter for writing the table schemas is "
                        << CoGaDB::CLI_PARAM_TABLES_TO_CSV << " <path to store
   the csv files> "
                        << std::endl;

                return -1;
            }

            CoGaDB::writeTablesToCSV = true;

            if(argc != 3) {

                std::cerr << "ERROR. When using the parameter " <<
   CoGaDB::CLI_PARAM_TABLES_TO_CSV
                        << " you have to specify a location to store the csv
   files."
                        << std::endl;

                return -1;
            }

            CoGaDB::writeTablesToCSVPath = argv[2];

        }

        std::cout << "Write Table To CSV " << CoGaDB::writeTablesToCSV
                << " into " << CoGaDB::writeTablesToCSVPath << std::endl;

        //::testing::GTEST_FLAG(output) = "xml:hello.xml";
        testing::InitGoogleTest(&argc, argv);

        return RUN_ALL_TESTS();
    } */
