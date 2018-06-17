
#pragma once

#include <core/table.hpp>
#include <parser/client.hpp>
#include <set>

namespace CoGaDB {

  const TablePtr loadTable(const std::string& table_name);
  const TablePtr loadTable(const std::string& table_name,
                           const std::string& path_to_database);

  bool storeTable(CoGaDB::TablePtr table, bool persist_to_disk = true);
  bool renameTable(const std::string& table_name,
                   const std::string& new_table_name);
  bool dropTable(const std::string& table_name);

  // bool storeColumn(vector<int> v)

  bool loadColumnFromDisk(ColumnPtr col, const std::string& table_name);
  bool loadColumnFromDisk(ColumnPtr col);

  bool loadColumnsInMainMemory(const std::vector<ColumnPtr>& columns,
                               const std::string& table_name);
  bool loadColumnsInMainMemory(const std::vector<ColumnPtr>& columns);

  bool loadTablesFromDirectory(const std::string& path_to_database,
                               std::ostream& out, bool quiet = false);
  bool loadTables(ClientPtr client);
  bool unloadTables(ClientPtr client);

  std::vector<TablePtr>& getGlobalTableList();

  void unloadGlobalTableList();

  const TablePtr getTablebyName(const std::string& name);

  bool addToGlobalTableList(TablePtr new_table);
  bool removeFromGlobalTableList(const std::string& table_name);

  bool isPersistent(TablePtr table);

  std::set<std::string> getColumnNamesOfSystemTables();

  TablePtr getSystemTableDatabaseSchema();

  bool storeTableAsSelfContainedCSV(const TablePtr,
                                    const std::string& path_to_dir,
                                    const std::string& file_name);
  const TablePtr loadTableFromSelfContainedCSV(const std::string& path_to_file);

}  // end namespace CogaDB
