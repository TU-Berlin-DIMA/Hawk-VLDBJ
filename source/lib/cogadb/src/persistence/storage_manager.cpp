
//#include <persistance/storage_manager.hpp>
#include <core/runtime_configuration.hpp>
#include <core/variable_manager.hpp>
#include <lookup_table/join_index.hpp>
#include <persistence/storage_manager.hpp>
#include <util/filesystem.hpp>
#include <util/getname.hpp>
#include <util/time_measurement.hpp>
#include <util/types.hpp>

#include <fstream>
#include <iostream>

// serialization
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/utility.hpp>

namespace CoGaDB {

using namespace std;

const TablePtr loadTable(const std::string& table_name) {
  return loadTable(table_name,
                   RuntimeConfiguration::instance().getPathToDatabase());
}

const TablePtr loadTable(const std::string& table_name,
                         const std::string& path_to_database) {
  string path = path_to_database;
  if (path[path.size() - 1] != '/') {
    path.append("/");
  }
  if (!quiet && verbose && debug)
    cout << "Load table '" << table_name << "' ..." << endl;
  if (!quiet && verbose && debug) cout << "Searching for tables..." << endl;
  std::vector<std::string> v = getFilesinDirectory(path);
  for (unsigned int i = 0; i < v.size(); i++) {
    if (!quiet && verbose && debug) cout << "Found: " << v[i] << endl;
    if (table_name == v[i]) {
      path += "/";
      path += table_name;
      break;
    }
    if (i == v.size() - 1) {
      cout << "Could not find Table: '" << table_name << "' Aborting..."
           << endl;
      exit(-1);
    }
  }

  Timestamp begin = getTimestamp();
  // load table schema

  if (!quiet && verbose && debug)
    cout << "Loading Table Schema '" << table_name << "' from file '" << path
         << "'..." << endl;

  bool error_occured = false;
  TablePtr tab(new Table(table_name, path_to_database, error_occured));
  assert(error_occured == false);
  tab->load(RuntimeConfiguration::instance().getTableLoaderMode());

  Timestamp end = getTimestamp();
  assert(end >= begin);
  cout << "Needed " << end - begin << "ns ("
       << double(end - begin) / (1000 * 1000 * 1000) << "s) to load Table '"
       << table_name << "' ..." << endl;
  return tab;
}

bool storeTable(TablePtr table, bool persist_to_disk) {
  using namespace boost::filesystem;
  string dir_path(RuntimeConfiguration::instance().getPathToDatabase());

  if (!exists(dir_path)) create_directory(dir_path);
  /* create tables directory in database directory if is does not exist */
  string table_dir_path = dir_path + "/tables/";
  if (!exists(table_dir_path)) create_directory(table_dir_path);
  /* handle case where table is intermediate table */
  assert(table != NULL);
  if (!table->isMaterialized()) {
    table = table->materialize();
    assert(table != NULL);
  }
  if (getTablebyName(table->getName())) {
    COGADB_FATAL_ERROR("Table '" << table->getName() << "' already exists!",
                       "");
  }
  if (table == getTablebyName(table->getName())) {
    /* In case we want to copy a table in a database and use this store
     * routine, we need to copy the table. */
    table = table->materialize();
    assert(table != NULL);
  }
  /* remove old qualified attribute names */
  renameFullyQualifiedNamesToUnqualifiedNames(table);
  /* add new qualified attribute names based on new table name */
  expandUnqualifiedColumnNamesToQualifiedColumnNames(table);
  if (!addToGlobalTableList(table)) {
    COGADB_FATAL_ERROR("Failed to add table: '"
                           << table->getName() << "' to global table list! "
                           << "A table with the same name already exists!",
                       "");
  }
  bool error_code = true;
  if (persist_to_disk) {
    error_code =
        table->store(RuntimeConfiguration::instance().getPathToDatabase());
  }
  return error_code;
}

bool renameTable(const std::string& table_name,
                 const std::string& new_table_name) {
  using namespace boost::filesystem;
  TablePtr table = getTablebyName(table_name);
  TablePtr table_new_name = getTablebyName(new_table_name);
  if (!table) {
    return false;
  }
  if (table_new_name) {
    return false;
  }
  std::string dir_path(RuntimeConfiguration::instance().getPathToDatabase());
  std::string meta_data_file = dir_path;
  meta_data_file += "/";
  meta_data_file += table_name;
  std::string new_meta_data_file = dir_path;
  new_meta_data_file += "/";
  new_meta_data_file += new_table_name;
  dir_path += "/tables/";
  std::string new_dir_path = dir_path;
  dir_path += table_name;
  new_dir_path += new_table_name;
  if (exists(meta_data_file)) {
    std::cout << "Renaming: '" << meta_data_file << "' to '"
              << new_meta_data_file << "'" << std::endl;
    boost::filesystem::rename(meta_data_file, new_meta_data_file);
  }
  if (exists(dir_path)) {
    std::cout << "Renaming: '" << dir_path << "' to '" << new_dir_path << "'"
              << std::endl;
    boost::filesystem::rename(dir_path, new_dir_path);
  }
  table->setName(new_table_name);
  return true;
}

bool dropTable(const std::string& table_name) {
  using namespace boost::filesystem;

  TablePtr table = getTablebyName(table_name);
  if (table) {
    if (!removeFromGlobalTableList(table_name)) {
      return false;
    }
    std::string dir_path(RuntimeConfiguration::instance().getPathToDatabase());
    std::string meta_data_file = dir_path;
    meta_data_file += "/";
    meta_data_file += table_name;
    dir_path += "/tables/";
    dir_path += table_name;
    if (exists(meta_data_file)) {
      boost::filesystem::remove_all(meta_data_file);
    }
    if (exists(dir_path)) {
      boost::filesystem::remove_all(dir_path);
    }
    std::cout << "DROP TABLE: '" << table_name << "'" << std::endl;
    table = getTablebyName(table_name);
    assert(table == NULL && "TABLE was not deleted correctly!");
    return true;
  } else {
    COGADB_FATAL_ERROR("DROP TABLE: Table '" << table_name << "' not found!",
                       "");
    return false;
  }
}

bool loadColumnFromDisk(ColumnPtr col, const std::string& table_name) {
  using namespace boost::filesystem;
  string dir_path(RuntimeConfiguration::instance().getPathToDatabase());
  if (!exists(dir_path)) return false;
  dir_path += "/tables/";
  dir_path += table_name;
  if (!exists(dir_path)) {
    cout << "No directory '" << dir_path << "' Aborting..." << endl;
    return false;
  }

  col->load(dir_path);
  col->setStatusLoadedInMainMemory(true);

  return true;
}

bool loadColumnFromDisk(ColumnPtr col) {
  if (!col) return false;
  if (col->isLoadedInMainMemory()) return true;
  std::vector<TablePtr>& tables = getGlobalTableList();
  for (unsigned int i = 0; i < tables.size(); ++i) {
    if (tables[i]->hasColumn(col->getName())) {
      loadColumnFromDisk(col, tables[i]->getName());
    }
  }
  return true;
}

bool loadColumnsInMainMemory(const std::vector<ColumnPtr>& columns,
                             const std::string& table_name) {
  for (unsigned int i = 0; i < columns.size(); ++i) {
    if (!columns[i]) continue;
    if (!columns[i]->isLoadedInMainMemory()) {
      if (!loadColumnFromDisk(columns[i], table_name)) return false;
    }
  }

  return true;
}

bool loadColumnsInMainMemory(const std::vector<ColumnPtr>& columns) {
  for (unsigned int i = 0; i < columns.size(); ++i) {
    if (!columns[i]) continue;
    if (!columns[i]->isLoadedInMainMemory()) {
      if (!loadColumnFromDisk(columns[i])) return false;
    }
  }

  return true;
}

bool loadTablesFromDirectory(const std::string& path_to_database,
                             std::ostream& out, bool quiet) {
  vector<TablePtr>& tables = getGlobalTableList();
  string path = path_to_database;
  if (path[path.size() - 1] != '/') {
    path.append("/");
  }
  if (!quiet) out << "Searching for tables..." << endl;
  std::vector<std::string> v = getFilesinDirectory(path);
  for (size_t i = 0; i < v.size(); i++) {
    if (is_regular_file(path + v[i])) {
      if (!quiet) out << "Loading table '" << v[i] << "' ..." << endl;
      const TablePtr tab = loadTable(v[i], path);
      assert(tab != NULL);
      // ambiguous table name?
      for (size_t j = 0; j < tables.size(); ++j) {
        if (tables[j]->getName() == tab->getName()) {
          COGADB_FATAL_ERROR("Cannot load table '"
                                 << tab->getName() << "'"
                                 << ": Table with same name already exists! ",
                             "");
        }
      }
      tables.push_back(tab);
    }
  }
  if (!quiet) {
    out << "Tables:" << endl;
    for (size_t i = 0; i < tables.size(); i++) {
      out << tables[i]->getName() << endl;
    }
  }
  return true;
}

bool loadTables(ClientPtr client) {
  return loadTablesFromDirectory(
      RuntimeConfiguration::instance().getPathToDatabase(),
      client->getOutputStream(), false);
}

bool unloadTables(ClientPtr) {
  unloadGlobalTableList();
  return true;
}

std::vector<TablePtr>& getGlobalTableList() {
  static std::vector<TablePtr> tables;
  return tables;
}

void unloadGlobalTableList() { getGlobalTableList().clear(); }

const TablePtr getTablebyName(const std::string& name) {
  const vector<TablePtr>& tables = getGlobalTableList();
  vector<TablePtr>::const_iterator it;
  for (it = tables.begin(); it != tables.end(); it++) {
    if ((*it)->getName() == name) {
      return *it;
    }
  }

  typedef TablePtr (*SystemTableGenerator)();
  typedef std::map<std::string, SystemTableGenerator> SystemTables;

  SystemTables sys_tabs;
  sys_tabs.insert(std::make_pair(std::string("SYS_DATABASE_SCHEMA"),
                                 &getSystemTableDatabaseSchema));
  sys_tabs.insert(std::make_pair(std::string("SYS_JOIN_INDEXES"),
                                 &getSystemTableJoinIndexes));
  sys_tabs.insert(
      std::make_pair(std::string("SYS_VARIABLES"), &getSystemTableVariables));

  SystemTables::const_iterator cit = sys_tabs.find(name);
  if (cit != sys_tabs.end()) {
    TablePtr system_table = cit->second();
    return system_table;
  }

  // cout << "Table '" << name << "' not found..." << endl;
  return TablePtr();
}

bool addToGlobalTableList(TablePtr new_table) {
  if (!new_table) return false;
  TablePtr tab = getTablebyName(new_table->getName());
  // if table with the same name exist, do nothing
  if (tab) return false;
  // add to global table list
  getGlobalTableList().push_back(new_table);
  return true;
}

bool removeFromGlobalTableList(const std::string& table_name) {
  std::vector<TablePtr>& tables = getGlobalTableList();
  std::vector<TablePtr>::iterator it;
  for (it = tables.begin(); it != tables.end(); ++it) {
    if ((*it)->getName() == table_name) {
      tables.erase(it);
      return true;
    }
  }
  return false;
}

bool isPersistent(TablePtr table) {
  if (!table) return false;
  if (getTablebyName(table->getName()) == table) {
    return true;
  } else {
    return false;
  }
}

std::set<std::string> getColumnNamesOfSystemTables() {
  static std::set<std::string> column_names_of_sys_column;
  static bool initialized = false;

  if (!initialized) {
    std::vector<std::string> system_tables;
    system_tables.push_back("SYS_DATABASE_SCHEMA");
    system_tables.push_back("SYS_JOIN_INDEXES");
    system_tables.push_back("SYS_VARIABLES");
    for (size_t i = 0; i < system_tables.size(); ++i) {
      TablePtr tab = getTablebyName(system_tables[i]);
      assert(tab != NULL);
      TableSchema s = tab->getSchema();
      TableSchema::iterator it;
      for (it = s.begin(); it != s.end(); ++it) {
        column_names_of_sys_column.insert(it->second);
      }
    }
    initialized = true;
  }

  return column_names_of_sys_column;
}

TablePtr getSystemTableDatabaseSchema() {
  std::vector<TablePtr>& tables = getGlobalTableList();

  TableSchema result_schema;
  result_schema.push_back(Attribut(VARCHAR, "TABLE_NAME"));
  result_schema.push_back(Attribut(VARCHAR, "COLUMN_NAME"));
  result_schema.push_back(Attribut(VARCHAR, "TYPE"));
  result_schema.push_back(Attribut(VARCHAR, "COMPRESSION_METHOD"));
  result_schema.push_back(Attribut(VARCHAR, "ROWS"));
  result_schema.push_back(Attribut(INT, "IN_MEMORY"));
  result_schema.push_back(Attribut(VARCHAR, "MAIN_MEMORY_FOOTPRINT_IN_BYTES"));

  TablePtr result_tab(new Table("SYS_DATABASE_SCHEMA", result_schema));

  for (size_t i = 0; i < tables.size(); i++) {
    std::vector<ColumnProperties> col_props =
        tables[i]->getPropertiesOfColumns();
    for (size_t j = 0; j < col_props.size(); j++) {
      Tuple t;
      t.push_back(tables[i]->getName());
      t.push_back(col_props[j].name);
      t.push_back(util::getName(col_props[j].attribute_type));
      t.push_back(util::getName(col_props[j].column_type));
      t.push_back(
          boost::lexical_cast<std::string>(col_props[j].number_of_rows));
      t.push_back((int)col_props[j].is_in_main_memory);
      t.push_back(
          boost::lexical_cast<std::string>(col_props[j].size_in_main_memory));
      result_tab->insert(t);
    }
  }

  return result_tab;
}

bool storeTableAsSelfContainedCSV(const TablePtr table,
                                  const std::string& path_to_dir,
                                  const std::string& file_name) {
  if (!table) {
    return false;
  }

  std::string total_file_path = path_to_dir + std::string("/") + file_name;

  if (!boost::filesystem::exists(path_to_dir)) {
    if (!boost::filesystem::create_directory(path_to_dir)) {
      COGADB_ERROR("Path '" << path_to_dir << "' does not exist "
                            << "and could not be created!",
                   "");
      return false;
    }
  }
  std::ofstream file;
  std::string path_to_file = path_to_dir;
  path_to_file.append("/").append(file_name);
  file.open(path_to_file.c_str(), std::ofstream::out | std::ofstream::trunc);

  if (!file.is_open()) {
    COGADB_ERROR("Could not open file '" << total_file_path << "'for writing!",
                 "");
  }

  std::stringstream header;
  header << "#COGADB_CSV_TABLE\t";
  TableSchema schema = table->getSchema();
  TableSchema::const_iterator cit;
  for (cit = schema.begin(); cit != schema.end(); ++cit) {
    header << cit->second << ":" << util::getName(cit->first);
    if (boost::next(cit) != schema.end()) {
      header << "\t";
    }
  }
  header << std::endl;

  std::string data_rows = table->toString("csv", false);
  file << header.str();
  file << data_rows;
  file.close();

  if (file.good()) {
    return true;
  } else {
    return false;
  }
}

const TablePtr loadTableFromSelfContainedCSV(const std::string& path_to_file) {
  if (!boost::filesystem::exists(path_to_file)) {
    COGADB_ERROR("Could not load table from csv: file '" << path_to_file
                                                         << "' not found!",
                 "");
    return TablePtr();
  }

  std::ifstream file;

  file.open(path_to_file.c_str(), std::ifstream::in);

  if (!file.is_open()) {
    COGADB_ERROR("Could not load table from csv: failed to open file '"
                     << path_to_file << "'!",
                 "");
    return TablePtr();
  }

  std::string line;
  getline(file, line, '\n');

  vector<string> strs;
  boost::split(strs, line, boost::is_any_of("\t"));

  if (!strs.empty()) {
    if (strs.front() == "#COGADB_CSV_TABLE") {
      TableSchema schema;
      std::string table_name;
      for (size_t i = 1; i < strs.size(); ++i) {
        vector<string> tokens;
        boost::split(tokens, strs[i], boost::is_any_of(":"));
        assert(tokens.size() == 2);
        std::string column_name = tokens[0];
        std::string attribute_type_str = tokens[1];

        AttributeType attribute_type;
        if (!convertStringToAttributeType(attribute_type_str, attribute_type)) {
          COGADB_ERROR("'" << attribute_type_str << "' is not "
                           << "a valid attribute type name!",
                       "");
          return TablePtr();
        }
        schema.push_back(std::make_pair(attribute_type, column_name));
      }
      TablePtr table(new Table(table_name, schema));
      if (!table->loadDatafromFile(path_to_file)) {
        COGADB_ERROR("Failed to load data from file '" << path_to_file << "'!",
                     "");
        return TablePtr();
      }
      return table;
    } else {
      COGADB_ERROR("File '" << path_to_file << "'is not a CoGaDB "
                            << "self contained csv file!",
                   "");
      return TablePtr();
    }
  } else {
    COGADB_ERROR("File '" << path_to_file << "'is not a CoGaDB "
                          << "self contained csv file!",
                 "");
    return TablePtr();
  }
}

}  // end namespace CogaDB
