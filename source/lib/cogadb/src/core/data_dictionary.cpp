
#include <core/data_dictionary.hpp>
#include <persistence/storage_manager.hpp>
#include <sstream>

#include "core/foreign_key_constraint.hpp"

namespace CoGaDB {

DataDictionary::DataDictionary() {}

DataDictionary& DataDictionary::instance() {
  static DataDictionary dd;
  return dd;
}

std::list<std::pair<ColumnPtr, TablePtr> >
DataDictionary::getColumnsforColumnName(const std::string& column_name) {
  std::list<std::pair<ColumnPtr, TablePtr> > result;
  std::vector<TablePtr>& tables = CoGaDB::getGlobalTableList();
  for (unsigned int i = 0; i < tables.size(); ++i) {
    TableSchema schema = tables[i]->getSchema();
    for (TableSchema::iterator it = schema.begin(); it != schema.end(); ++it) {
      if (it->second == column_name) {
        ColumnPtr col = tables[i]->getColumnbyName(column_name);
        result.push_back(std::make_pair(col, tables[i]));
      }
    }
  }
  return result;
}

const TablePtr DataDictionary::getTableForColumnName(
    const std::string& column_name) {
  TablePtr result;
  std::vector<TablePtr>& tables = CoGaDB::getGlobalTableList();
  size_t counter = 0;
  for (unsigned int i = 0; i < tables.size(); ++i) {
    TableSchema schema = tables[i]->getSchema();
    for (TableSchema::iterator it = schema.begin(); it != schema.end(); ++it) {
      if (it->second == column_name) {
        counter++;
        result = tables[i];
      }
    }
  }
  if (counter == 1) {
    return result;
  } else if (counter == 0) {
    return TablePtr();
  } else {
    COGADB_FATAL_ERROR("Found multiple attribute that have the same name '"
                           << column_name << "'!",
                       "");
    return TablePtr();
  }
  return result;
}

const std::vector<AttributeReferencePtr> DataDictionary::getAttributesForName(
    const std::string& attr_name, uint32_t version) {
  std::vector<AttributeReferencePtr> result;
  std::vector<TablePtr>& tables = CoGaDB::getGlobalTableList();
  for (size_t i = 0; i < tables.size(); ++i) {
    TableSchema schema = tables[i]->getSchema();
    for (TableSchema::iterator it = schema.begin(); it != schema.end(); ++it) {
      if (it->second == attr_name) {
        AttributeReferencePtr attr = createInputAttribute(
            tables[i], attr_name,
            //                                createFullyQualifiedColumnIdentifier(tables[i]->getName(),
            //                                    attr_name, version),
            std::string(), /* autocompute reference result identifier */
            version);
        assert(attr != NULL);
        result.push_back(attr);
      }
    }
  }
  return result;
}

bool DataDictionary::findUniqueTableNameForSimpleAttributeName(
    const std::string& attr_name, std::string& table_name) {
  std::vector<std::string> result;
  std::vector<TablePtr>& tables = CoGaDB::getGlobalTableList();
  for (size_t i = 0; i < tables.size(); ++i) {
    TableSchema schema = tables[i]->getSchema();
    for (TableSchema::iterator it = schema.begin(); it != schema.end(); ++it) {
      //                    std::string plain_attribute_name = it->second;
      std::string original_table_name;
      std::string plain_attribute_name = it->second;
      uint32_t version = 1;
      if (!parseColumnIndentifierName(it->second, original_table_name,
                                      plain_attribute_name, version)) {
        COGADB_FATAL_ERROR(
            "Could not parse attribute identifier from attribute reference!",
            "");
      }
      if (plain_attribute_name == attr_name) {
        result.push_back(original_table_name);
      }
    }
  }
  if (result.size() == 1) {
    table_name = result.front();
    return true;
  } else {
    return false;
  }
}

size_t DataDictionary::countNumberOfOccurencesOfColumnName(
    const std::string& attr_name) {
  size_t result = 0;
  std::vector<TablePtr>& tables = CoGaDB::getGlobalTableList();
  for (size_t i = 0; i < tables.size(); ++i) {
    TableSchema schema = tables[i]->getSchema();
    for (TableSchema::iterator it = schema.begin(); it != schema.end(); ++it) {
      if (it->second == attr_name) {
        result++;
      }
    }
  }
  return result;
}

bool DataDictionary::isUniqueAttribueName(const std::string& attr_name) {
  return countNumberOfOccurencesOfColumnName(attr_name) == 1;
}

bool DataDictionary::existColumn(const std::string& column_name) {
  // is this a regular column from the database?
  if (countNumberOfOccurencesOfColumnName(column_name) > 0) return true;
  // is it a column from a system table?
  std::set<std::string> sys_columns = getColumnNamesOfSystemTables();
  if (sys_columns.find(column_name) != sys_columns.end()) {
    return true;
  }
  return false;
}

bool DataDictionary::getAttributeType(const std::string& table_name,
                                      const std::string& column_name,
                                      AttributeType& result_attribute_type) {
  std::vector<TablePtr>& tables = CoGaDB::getGlobalTableList();
  for (unsigned int i = 0; i < tables.size(); ++i) {
    TableSchema schema = tables[i]->getSchema();
    for (TableSchema::iterator it = schema.begin(); it != schema.end(); ++it) {
      if (it->second == column_name) {
        result_attribute_type = it->first;
        return true;
      }
    }
  }
  return false;
}

bool DataDictionary::hasPrimaryKeyConstraint(const std::string& column_name) {
  std::list<std::pair<ColumnPtr, TablePtr> > candidate_list =
      getColumnsforColumnName(column_name);
  if (candidate_list.empty()) {
    COGADB_FATAL_ERROR(
        "Data Dictionary: Cannot get PrimaryKeyConstraint Status of Column '"
            << column_name << "': Column not found in database!",
        "");
  }
  if (candidate_list.size() > 1) {
    COGADB_FATAL_ERROR(
        "Data Dictionary: Cannot get PrimaryKeyConstraint Status of Column '"
            << column_name << "': Ambiguous column name!",
        "");
  }
  return candidate_list.front().first->hasPrimaryKeyConstraint();
}

bool DataDictionary::hasForeignKeyConstraint(const std::string& column_name) {
  std::list<std::pair<ColumnPtr, TablePtr> > candidate_list =
      getColumnsforColumnName(column_name);
  if (candidate_list.empty()) {
    COGADB_FATAL_ERROR(
        "Data Dictionary: Cannot get ForeignKeyConstraint Status of Column '"
            << column_name << "': Column not found in database!",
        "");
  }
  if (candidate_list.size() > 1) {
    COGADB_FATAL_ERROR(
        "Data Dictionary: Cannot get ForeignKeyConstraint Status of Column '"
            << column_name << "': Ambiguous column name!",
        "");
  }
  return candidate_list.front().first->hasForeignKeyConstraint();
}

TablePtr DataDictionary::getTableWithIntegrityConstraints() {
  TableSchema result_schema;
  result_schema.push_back(Attribut(VARCHAR, "TABLE NAME"));
  result_schema.push_back(Attribut(VARCHAR, "COLUMN NAME"));
  result_schema.push_back(Attribut(VARCHAR, "INTEGRITY CONSTRAINT"));

  TablePtr result_tab(new Table("SYS_INTEGRITY_CONSTRAINTS", result_schema));
  std::vector<TablePtr>& tables = getGlobalTableList();
  for (unsigned int i = 0; i < tables.size(); ++i) {
    TableSchema schema = tables[i]->getSchema();
    TableSchema::iterator it;
    for (it = schema.begin(); it != schema.end(); ++it) {
      if (tables[i]->hasPrimaryKeyConstraint(it->second)) {
        Tuple t;
        t.push_back(tables[i]->getName());
        t.push_back(it->second);
        t.push_back(std::string("PRIMARY KEY"));
        result_tab->insert(t);
      }
      if (tables[i]->hasForeignKeyConstraint(it->second)) {
        Tuple t;
        t.push_back(tables[i]->getName());
        t.push_back(it->second);
        std::stringstream ss;
        const ForeignKeyConstraint* fk_constr =
            tables[i]->getForeignKeyConstraint(it->second);
        if (!fk_constr) continue;
        ss << "FOREIGN KEY referencing "
           << fk_constr->getNameOfPrimaryKeyTable() << "."
           << fk_constr->getNameOfPrimaryKeyColumn();
        t.push_back(ss.str());
        result_tab->insert(t);
      }
    }
  }
  return result_tab;
}

}  // end namespace CoGaDB
