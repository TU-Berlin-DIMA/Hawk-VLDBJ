
#include <persistence/storage_manager.hpp>
#include <statistics/column_statistics.hpp>
#include <util/statistics.hpp>

namespace CoGaDB {

bool computeStatisticsOnTable(const std::string& table_name, ClientPtr client) {
  std::ostream& out = client->getOutputStream();

  TablePtr table = getTablebyName(table_name);
  if (!table) {
    out << "Table '" << table_name << "' not found!" << std::endl;
    return false;
  }
  TableSchema schema = table->getSchema();
  TableSchema::iterator it;
  for (it = schema.begin(); it != schema.end(); ++it) {
    ColumnPtr col = table->getColumnbyName(it->second);
    out << "Compute statistics of column: '" << it->second << "'" << std::endl;
    if (!col->computeColumnStatistics()) {
      out << "Failed to compute statistics for column: '" << it->second << "'!"
          << std::endl;
      return false;
    } else {
      out << col->getColumnStatistics().toString() << std::endl;
    }
  }

  return true;
}

}  // end namespace CoGaDB
