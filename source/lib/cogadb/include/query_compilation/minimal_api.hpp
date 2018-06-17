/*
 * File:   minimal_api.hpp
 * Author: sebastian
 *
 * Created on 27. Juli 2015, 09:41
 */

#ifndef MINIMAL_API_HPP
#define MINIMAL_API_HPP

#include <stdint.h>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <core/attribute_reference.hpp>
#include <query_compilation/global_state.hpp>
#include <string>
extern "C" {
#include <hardware_optimizations/main_memory_joins/serial_hash_join/hashtable/cuckoo_hashtable.h>
#include <hardware_optimizations/main_memory_joins/serial_hash_join/hashtable/hashtable.h>
}

namespace CoGaDB {

  typedef std::string string;
  // forward declarations
  class ColumnBase;
  typedef boost::shared_ptr<ColumnBase> ColumnPtr;

  class BaseTable;
  typedef boost::shared_ptr<BaseTable> TablePtr;

  class HashTable;
  typedef boost::shared_ptr<HashTable> HashTablePtr;

  typedef void (*HashTableCleanupFunctionPtr)(void*);

  const TablePtr getTablebyName(const std::string& name);
  ColumnPtr getColumn(TablePtr, const std::string&);
  ColumnPtr getDictionaryCompressedColumn(TablePtr, const std::string&);
  HashTablePtr getHashTable(TablePtr, const std::string& column_name);

  int32_t* getArrayFromColumn_int32_t(ColumnPtr);
  uint32_t* getArrayFromColumn_uint32_t(ColumnPtr);
  uint64_t* getArrayFromColumn_uint64_t(ColumnPtr);
  float* getArrayFromColumn_float(ColumnPtr);
  double* getArrayFromColumn_double(ColumnPtr);
  std::string* getArrayFromColumn_string(ColumnPtr);
  char* getArrayFromColumn_char(ColumnPtr);

  uint32_t* getArrayCompressedKeysFromColumn_string(ColumnPtr);

  const TablePtr createTableFromColumns(const std::string& table_name,
                                        ColumnPtr* columns, size_t num_columns);
  const TablePtr createTableFromColumns(const std::string& table_name,
                                        const std::vector<ColumnPtr>& columns);
  bool addHashTable(TablePtr table, const std::string& column_name,
                    HashTablePtr hash_table);

  //    const HashTablePtr createSystemHashTable(hashtable_t*);
  //    hashtable_t* getHashTableFromSystemHashTable(HashTablePtr);

  const HashTablePtr createSystemHashTable(
      void*, HashTableCleanupFunctionPtr cleanup_handler, const std::string&);

  void* getHashTableFromSystemHashTable(HashTablePtr);

  size_t getNumberOfRows(TablePtr);

  ColumnPtr createResultArray_int32_t(const std::string& name,
                                      const int32_t* array,
                                      size_t num_elements);
  ColumnPtr createResultArray_uint32_t(const std::string& name,
                                       const uint32_t* array,
                                       size_t num_elements);
  ColumnPtr createResultArray_uint64_t(const std::string& name,
                                       const uint64_t* array,
                                       size_t num_elements);
  ColumnPtr createResultArray_float(const std::string& name, const float* array,
                                    size_t num_elements);
  ColumnPtr createResultArray_double(const std::string& name,
                                     const double* array, size_t num_elements);
  ColumnPtr createResultArray_string(const std::string& name,
                                     const std::string* array,
                                     size_t num_elements);
  ColumnPtr createResultArray_char(const std::string& name, const char* array,
                                   size_t num_elements);

  std::string* stringMalloc(size_t number_of_elements);
  void stringFree(std::string*& data);
  std::string* stringRealloc(std::string* data, size_t number_of_elements);

}  // end namespace CoGaDB

#endif /* MINIMAL_API_HPP */
