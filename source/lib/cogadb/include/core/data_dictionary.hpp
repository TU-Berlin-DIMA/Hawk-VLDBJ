#pragma once
#include <core/base_table.hpp>
#include <core/global_definitions.hpp>
namespace CoGaDB {

  class DataDictionary {
   public:
    // returns a list of all columns with name column_name, toghether with their
    // corresponding tables
    std::list<std::pair<ColumnPtr, TablePtr> > getColumnsforColumnName(
        const std::string& column_name);
    const TablePtr getTableForColumnName(const std::string& column_name);
    const std::vector<AttributeReferencePtr> getAttributesForName(
        const std::string& attr_name, uint32_t version = 1);
    bool findUniqueTableNameForSimpleAttributeName(const std::string& attr_name,
                                                   std::string& table_name);
    size_t countNumberOfOccurencesOfColumnName(const std::string& attr_name);
    bool isUniqueAttribueName(const std::string& attr_name);
    bool existColumn(const std::string& column_name);
    /*\brief
      \result result stored in variable the reference type points to in case
      function returns true, if it return false the value the reference type to
      is not initialized!*/
    bool getAttributeType(const std::string& table_name,
                          const std::string& column_name, AttributeType& type);
    static DataDictionary& instance();
    bool hasPrimaryKeyConstraint(const std::string& column_name);
    bool hasForeignKeyConstraint(const std::string& column_name);
    TablePtr getTableWithIntegrityConstraints();

   private:
    // singleton
    DataDictionary();
    DataDictionary(const DataDictionary&);
    DataDictionary& operator=(DataDictionary&);
  };

  bool isPersistentColumn(ColumnPtr column_ptr);

  bool isIntermediateResultColumn(ColumnPtr column_ptr);

  /*
  DataDictionary& DataDictionary::instance(){
      static DataDictionary dd;
      return dd;
  }

  std::list<std::pair<ColumnPtr,TablePtr> >
  DataDictionary::getColumnsforColumnName(const std::string& column_name){
      std::list<std::pair<ColumnPtr,TablePtr> > result;
      std::vector<TablePtr>& tables = CoGaDB::getGlobalTableList();
      for(unsigned int i=0;i<tables.size();++i){
          TableSchema schema = tables[i]->getSchema();
          for(TableSchema::iterator it=schema.begin();schema.end();++it){
              if(it->second==column_name){
                  ColumnPtr col = tables[i]->getColumnbyName(column_name);
                  result.push_back(std::make_pair(col,tables[i]));

              }
          }
      }
      return result;
  }

  bool DataDictionary::getAttributeType(const std::string& table_name, const
  std::string& column_name, AttributeType& result_attribute_type){
      std::vector<TablePtr>& tables = CoGaDB::getGlobalTableList();
      for(unsigned int i=0;i<tables.size();++i){
          TableSchema schema = tables[i]->getSchema();
          for(TableSchema::iterator it=schema.begin();schema.end();++it){
              if(it->second==column_name){
                  result_attribute_type=it->first;
                  return true;
              }
          }
      }
      return false;
  }
  */
}  // end namespace CoGaDB
