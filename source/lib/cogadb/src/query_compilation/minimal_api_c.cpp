#include <query_compilation/minimal_api_c.h>

#include <compression/dictionary_compressed_column.hpp>
#include <compression/order_preserving_dictionary_compressed_column.hpp>
#include <core/column.hpp>
#include <core/column.hpp>
#include <core/cstring_column.hpp>
#include <core/global_definitions.hpp>
#include <core/table.hpp>
#include <core/vector_typed.hpp>
#include <google/dense_hash_map>
#include <query_compilation/minimal_api.hpp>
#include <query_compilation/minimal_api_c_internal.hpp>
#include <util/dictionary_compression.hpp>
#include <util/getname.hpp>

#include <boost/make_shared.hpp>

#include <cstdarg>
#include <limits>

struct C_Table {
  CoGaDB::TablePtr table;
};

struct C_Column {
  CoGaDB::ColumnPtr column;
  CoGaDB::ColumnPtr decompressed_column;
};

struct C_HashTable {
  CoGaDB::HashTablePtr hash_table;
};

C_Table* CoGaDB::getCTableFromTablePtr(CoGaDB::TablePtr table) {
  C_Table* c_table = new C_Table;
  c_table->table = table;
  return c_table;
}

C_Table* getTableByName(const char* const column_name) {
  return CoGaDB::getCTableFromTablePtr(CoGaDB::getTablebyName(column_name));
}

#define CHECK_C_STRUCT(ptr, member, result) \
  if (ptr == NULL) {                        \
    return result;                          \
  }                                         \
  assert(ptr->member != NULL)

#define CHECK_C_TABLE(ctable) CHECK_C_STRUCT(ctable, table, NULL)

#define CHECK_C_TABLE_BOOL(ctable) CHECK_C_STRUCT(ctable, table, false)

#define CHECK_C_COLUMN(ccol) CHECK_C_STRUCT(ccol, column, NULL)

#define CHECK_C_COLUMN_BOOL(ccol) CHECK_C_STRUCT(ccol, column, false)

#define CHECK_C_HASHTABLE(chash) CHECK_C_STRUCT(chash, hash_table, NULL)

#define CHECK_C_HASHTABLE_BOOL(chash) CHECK_C_STRUCT(chash, hash_table, false)

CoGaDB::TablePtr CoGaDB::getTablePtrFromCTable(C_Table* c_table) {
  CHECK_C_TABLE(c_table);
  return c_table->table;
}

const CoGaDB::ColumnPtr convert_to_cstring_column_if_required(
    CoGaDB::ColumnPtr col) {
  using namespace CoGaDB;

  if (!col) return col;

  if (col->getType() != VARCHAR) return col;

  if (col->getColumnType() == DICTIONARY_COMPRESSED ||
      col->getColumnType() == DICTIONARY_COMPRESSED_ORDER_PRESERVING) {
    boost::shared_ptr<DictionaryCompressedColumn<std::string>>
        dict_compressed_col;
    boost::shared_ptr<OrderPreservingDictionaryCompressedColumn<std::string>>
        ordered_dict_compressed_col;
    dict_compressed_col =
        boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>>(
            col);
    ordered_dict_compressed_col = boost::dynamic_pointer_cast<
        OrderPreservingDictionaryCompressedColumn<std::string>>(col);
    ProcessorSpecification proc_spec(hype::PD0);
    if (dict_compressed_col) {
      return dict_compressed_col->getDecompressedColumn(proc_spec);
      //            uint32_t* compressed_values =
      //            dict_compressed_col->getIdData();
      //            const std::string* reverse_lookup_vector =
      //            dict_compressed_col->getReverseLookupVector();
      //            return  createPointerArrayToValues(compressed_values,
      //                dict_compressed_col->size(),
      //                reverse_lookup_vector);
    } else if (ordered_dict_compressed_col) {
      return ordered_dict_compressed_col->getDecompressedColumn(proc_spec);
      //            uint32_t* compressed_values =
      //            ordered_dict_compressed_col->getIdData();
      //            const std::string* reverse_lookup_vector =
      //            ordered_dict_compressed_col->getReverseLookupVector();
      //            return  createPointerArrayToValues(compressed_values,
      //                ordered_dict_compressed_col->size(),
      //                reverse_lookup_vector);
    } else {
      COGADB_FATAL_ERROR(
          "Could not retrieve vector for dictionary compressed columns key "
          "column!",
          "");
    }
  } else if (col->getColumnType() == PLAIN_MATERIALIZED) {
    if (col->type() == typeid(char*)) {
      return col;
    }
    COGADB_FATAL_ERROR(
        "Col: " << col->getName() << " Type " << util::getName(col->getType())
                << " ColType: " << util::getName(col->getColumnType()),
        "");
  }

  return nullptr;
}

C_Column* getColumnByName(C_Table* c_table, const char* const column_name) {
  CHECK_C_TABLE(c_table);

  return getColumnById(c_table,
                       c_table->table->getColumnIdbyColumnName(column_name));
}

C_Column* getColumnById(C_Table* c_table, const unsigned int id) {
  CHECK_C_TABLE(c_table);

  CoGaDB::ColumnPtr col = c_table->table->getColumnbyId(id);

  C_Column* c_col = new C_Column;
  c_col->column = col;
  return c_col;
}

C_HashTable* getHashTable(C_Table* c_table, const char* const column_name) {
  CHECK_C_TABLE(c_table);

  CoGaDB::HashTablePtr table =
      CoGaDB::getHashTable(c_table->table, column_name);

  C_HashTable* c_hashtable = new C_HashTable;

  c_hashtable->hash_table = table;
  return c_hashtable;
}

void releaseTable(C_Table* c_table) {
  if (!c_table) {
    return;
  }
  delete c_table;
}

void releaseColumn(C_Column* col) {
  if (!col) {
    return;
  }
  delete col;
}

void releaseHashTable(C_HashTable* table) {
  if (!table) {
    return;
  }
  delete table;
}

#define IMPLEMENT_GETARRAYFROMCOLUMN(type)                      \
  const type* getArrayFromColumn_##type(C_Column* c_column) {   \
    CHECK_C_COLUMN(c_column);                                   \
    return CoGaDB::getArrayFromColumn_##type(c_column->column); \
  }

IMPLEMENT_GETARRAYFROMCOLUMN(int32_t)
IMPLEMENT_GETARRAYFROMCOLUMN(uint32_t)
IMPLEMENT_GETARRAYFROMCOLUMN(uint64_t)
IMPLEMENT_GETARRAYFROMCOLUMN(float)
IMPLEMENT_GETARRAYFROMCOLUMN(double)
IMPLEMENT_GETARRAYFROMCOLUMN(char)

CoGaDB::ColumnPtr getDecompressedColumn(C_Column* c_column) {
  CoGaDB::ColumnPtr col = c_column->decompressed_column;

  if (!col) {
    col = c_column->column;

    if (!boost::dynamic_pointer_cast<CoGaDB::Vector>(col)) {
      col = convert_to_cstring_column_if_required(col);
    }

    if (!boost::dynamic_pointer_cast<CoGaDB::Vector>(col)) {
      col = decompress_if_required(col);
    }

    assert(col != NULL && "Column could not be decompressed!");

    c_column->decompressed_column = col;
  }

  return col;
}

const char* const* getArrayFromColumn_cstring(C_Column* c_column) {
  CHECK_C_COLUMN(c_column);

  CoGaDB::ColumnPtr col = getDecompressedColumn(c_column);

  assert(col->type() == typeid(char*));

  if ((col->type() == typeid(std::string))) {
    COGADB_FATAL_ERROR("", "");
  } else if (col->type() == typeid(char*)) {
    boost::shared_ptr<CoGaDB::Column<char*>> typed_col;
    typed_col = boost::dynamic_pointer_cast<CoGaDB::Column<char*>>(col);
    if (!typed_col) {
      return NULL;
    }
    return typed_col->data();
  }

  return NULL; /**/
}

uint32_t* getArrayCompressedKeysFromColumn_string(C_Column* c_column) {
  CHECK_C_COLUMN(c_column);
  return CoGaDB::getArrayCompressedKeysFromColumn_string(c_column->column);
}

C_Table* createTableFromColumns(const char* const table_name,
                                C_Column** columns, size_t num_columns) {
  std::vector<CoGaDB::ColumnPtr> result_columns;
  for (size_t i = 0; i < num_columns; ++i) {
    assert(columns[i] != NULL);
    assert(columns[i]->column != NULL);
    result_columns.push_back(columns[i]->column);
    releaseColumn(columns[i]);
  }

  CoGaDB::TablePtr table =
      CoGaDB::createTableFromColumns(table_name, result_columns);
  if (!table) {
    return NULL;
  }

  C_Table* c_table = new C_Table;
  c_table->table = table;

  return c_table;
}

bool addHashTable(C_Table* c_table, const char* const column_name,
                  C_HashTable* hash_table) {
  CHECK_C_TABLE_BOOL(c_table);
  CHECK_C_HASHTABLE_BOOL(hash_table);

  return CoGaDB::addHashTable(c_table->table, column_name,
                              hash_table->hash_table);
}

C_HashTable* createSystemHashTable(void* ht,
                                   HashTableCleanupFunctionPtr cleanup_handler,
                                   char* id) {
  if (!ht) {
    return NULL;
  }

  CoGaDB::HashTablePtr hash_table =
      CoGaDB::createSystemHashTable(ht, cleanup_handler, std::string(id));
  if (hash_table == NULL) {
    return NULL;
  }

  C_HashTable* c_hash_table = new C_HashTable;
  c_hash_table->hash_table = hash_table;

  return c_hash_table;
}

void* getHashTableFromSystemHashTable(C_HashTable* c_hash_table) {
  CHECK_C_HASHTABLE(c_hash_table);
  return CoGaDB::getHashTableFromSystemHashTable(c_hash_table->hash_table);
}

size_t getNumberOfRows(C_Table* c_table) {
  CHECK_C_STRUCT(c_table, table, 0);
  return CoGaDB::getNumberOfRows(c_table->table);
}

#define IMPLEMENT_CREATERESULTARRAY(type)                                      \
  C_Column* createResultArray_##type(const char* const name,                   \
                                     const type* array, size_t num_elements) { \
    CoGaDB::ColumnPtr col =                                                    \
        CoGaDB::createResultArray_##type(name, array, num_elements);           \
    if (!col) {                                                                \
      return NULL;                                                             \
    }                                                                          \
    free(const_cast<type*>(array));                                            \
    C_Column* c_column = new C_Column;                                         \
    c_column->column = col;                                                    \
                                                                               \
    return c_column;                                                           \
  }

IMPLEMENT_CREATERESULTARRAY(int32_t)
IMPLEMENT_CREATERESULTARRAY(uint32_t)
IMPLEMENT_CREATERESULTARRAY(uint64_t)
IMPLEMENT_CREATERESULTARRAY(float)
IMPLEMENT_CREATERESULTARRAY(double)
IMPLEMENT_CREATERESULTARRAY(char)

C_Column* createResultArray_cstring(const char* const name, const char** array,
                                    size_t num_elements) {
  boost::shared_ptr<CoGaDB::Column<char*>> c_string_col =
      boost::make_shared<CoGaDB::Column<char*>>(name, CoGaDB::VARCHAR);
  c_string_col->reserve(num_elements);

  for (unsigned int i = 0; i < num_elements; ++i) {
    c_string_col->push_back(const_cast<char*>(array[i]));
  }
  free(const_cast<char**>(array));
  C_Column* c_column = new C_Column;
  c_column->column = c_string_col;

  return c_column;
}

C_Column* createResultArray_cstring_compressed(C_Column* orig,
                                               const char* const name,
                                               const uint32_t* array,
                                               size_t num_elements) {
  CHECK_C_COLUMN(orig);

  typedef CoGaDB::DictionaryCompressedColumn<std::string> ResultColType;
  ResultColType::IDColumnPtr id_col =
      boost::make_shared<ResultColType::IDColumn>(name, CoGaDB::UINT32, array,
                                                  array + num_elements);
  free(const_cast<uint32_t*>(array));

  CoGaDB::ColumnPtr orig_generic_col = orig->column;
  typedef CoGaDB::VectorTyped<std::string> VectorType;
  boost::shared_ptr<VectorType> orig_vector;

  if ((orig_vector = boost::dynamic_pointer_cast<VectorType>(
           orig_generic_col)) != nullptr) {
    orig_generic_col = orig_vector->getSourceColumn();
  }

  boost::shared_ptr<ResultColType> orig_col =
      boost::dynamic_pointer_cast<ResultColType>(orig_generic_col);

  if (!orig_col) {
    return NULL;
  }

  boost::shared_ptr<ResultColType> result_col =
      boost::make_shared<ResultColType>(name, CoGaDB::VARCHAR, id_col,
                                        orig_col->getReverseLookupVectorPtr(),
                                        orig_col->getDictionary());

  C_Column* c_column = new C_Column;
  c_column->column = result_col;

  return c_column;
}

char** stringMalloc(size_t number_of_elements, size_t max_length) {
  char** array = new char*[number_of_elements];

  for (unsigned int i = 0; i < number_of_elements; ++i) {
    array[i] = new char[max_length];
  }

  return array;
}

void stringFree(char** data, size_t number_of_elements) {
  for (unsigned int i = 0; i < number_of_elements; ++i) {
    delete[] data[i];
  }
  delete[] data;
}

char** stringRealloc(char** data, size_t old_number_of_elements,
                     size_t number_of_elements, size_t max_length) {
  char** array = new char*[number_of_elements];

  for (unsigned int i = 0; i < old_number_of_elements; ++i) {
    array[i] = data[i];
    data[i] = NULL;
  }

  for (size_t i = old_number_of_elements; i < number_of_elements; ++i) {
    array[i] = new char[max_length];
  }

  delete[] data;
  return array;
}

size_t getMaxStringLengthForColumn(C_Column* c_column) {
  CHECK_C_STRUCT(c_column, column, 0);

  CoGaDB::ColumnPtr col = c_column->column;
  assert(col->type() == typeid(char*));

  boost::shared_ptr<CoGaDB::CStringColumn> typed_col;
  typed_col = boost::dynamic_pointer_cast<CoGaDB::CStringColumn>(col);

  if (!typed_col) {
    return 0;
  }

  return typed_col->getMaxStringLength();
}

uint64_t getGroupKey(unsigned int count...) {
  va_list args;
  va_start(args, count);
  std::size_t seed = 0;

  for (unsigned int i = 0; i < count; ++i) {
    CoGaDB::AttributeType type =
        static_cast<CoGaDB::AttributeType>(va_arg(args, int));

    switch (type) {
      case CoGaDB::INT:
      case CoGaDB::CHAR:
      case CoGaDB::BOOLEAN:
        boost::hash_combine(seed, va_arg(args, int));
        break;
      case CoGaDB::FLOAT:
      case CoGaDB::DOUBLE:
        boost::hash_combine(seed, va_arg(args, double));
        break;
      case CoGaDB::VARCHAR:
        boost::hash_combine(seed, va_arg(args, char*));
        break;
      case CoGaDB::UINT32:
      case CoGaDB::DATE:
        boost::hash_combine(seed, va_arg(args, uint32_t));
        break;
      case CoGaDB::OID:
        boost::hash_combine(seed, va_arg(args, uint64_t));
        break;
    }
  }
  return seed;
}

typedef boost::unordered_map<uint64_t, boost::shared_ptr<char[]>>
    AggregationHashTable;

struct C_AggregationHashTable {
  AggregationHashTable sHashTable;
  uint64_t sPayloadSize;

  C_AggregationHashTable(uint64_t payload_size) : sPayloadSize(payload_size) {}
};

C_AggregationHashTable* createAggregationHashTable(uint64_t payload_size) {
  return new C_AggregationHashTable(payload_size);
}

void freeAggregationHashTable(C_AggregationHashTable* table) { delete table; }

uint64_t getAggregationHashTableSize(C_AggregationHashTable* table) {
  return table->sHashTable.size();
}

void* getAggregationHashTablePayload(C_AggregationHashTable* table,
                                     uint64_t group_key) {
  if (table == NULL) {
    return NULL;
  }

  AggregationHashTable::iterator itr = table->sHashTable.find(group_key);

  if (itr != table->sHashTable.end()) {
    return itr->second.get();
  } else {
    return NULL;
  }
}

void* insertAggregationHashTable(C_AggregationHashTable* table,
                                 uint64_t group_key, void* payload) {
  if (table == NULL) {
    return NULL;
  }

  boost::shared_ptr<char[]> ptr(new char[table->sPayloadSize]());
  memcpy(ptr.get(), payload, table->sPayloadSize);
  typedef AggregationHashTable::iterator Iterator;
  Iterator it;
  std::pair<Iterator, bool> ret =
      table->sHashTable.insert(std::make_pair(group_key, ptr));
  if (ret.second == true) {
    return ptr.get();
  } else {
    COGADB_FATAL_ERROR("Failed to insert in HashTable!", "");
    return NULL;
  }
}

struct C_AggregationHashTableIterator {
  C_AggregationHashTable* sCHashTable;
  AggregationHashTable::iterator sIterator;

  C_AggregationHashTableIterator(C_AggregationHashTable* table)
      : sCHashTable(table), sIterator(table->sHashTable.begin()) {}
};

C_AggregationHashTableIterator* createAggregationHashTableIterator(
    C_AggregationHashTable* table) {
  return new C_AggregationHashTableIterator(table);
}

void freeAggregationHashTableIterator(C_AggregationHashTableIterator* itr) {
  delete itr;
}

void nextAggregationHashTableIterator(C_AggregationHashTableIterator* itr) {
  ++itr->sIterator;
}

char hasNextAggregationHashTableIterator(C_AggregationHashTableIterator* itr) {
  return itr->sIterator != itr->sCHashTable->sHashTable.end();
}

void* getAggregationHashTableIteratorPayload(
    C_AggregationHashTableIterator* itr) {
  if (!hasNextAggregationHashTableIterator(itr)) {
    return NULL;
  }

  return itr->sIterator->second.get();
}

struct C_UniqueKeyHashTable {
  google::dense_hash_map<TID, TID> sHashTable;

  C_UniqueKeyHashTable(uint64_t num_elements) : sHashTable(num_elements) {
    sHashTable.set_empty_key(std::numeric_limits<TID>::max());
  }
};

C_UniqueKeyHashTable* createUniqueKeyJoinHashTable(uint64_t num_elements) {
  return new C_UniqueKeyHashTable(num_elements);
}

void freeUniqueKeyJoinHashTable(C_UniqueKeyHashTable* ht) {
  if (ht) {
    delete ht;
  }
}

uint64_t getUniqueKeyJoinHashTableSize(C_UniqueKeyHashTable* ht) {
  assert(ht != NULL);
  return ht->sHashTable.size();
}

uint64_t* getUniqueKeyJoinHashTablePayload(C_UniqueKeyHashTable* ht,
                                           uint64_t key) {
  assert(ht != NULL);
  google::dense_hash_map<TID, TID>::iterator it = ht->sHashTable.find(key);
  if (it != ht->sHashTable.end()) {
    return &it->second;
  } else {
    return NULL;
  }
}

int insertUniqueKeyJoinHashTable(C_UniqueKeyHashTable* ht, uint64_t key,
                                 uint64_t payload) {
  assert(ht != NULL);
  std::pair<google::dense_hash_map<TID, TID>::iterator, bool> it;
  it = ht->sHashTable.insert(std::make_pair(key, payload));
  return it.second;
}
