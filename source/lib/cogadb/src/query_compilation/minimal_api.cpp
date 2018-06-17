
#include <compression/dictionary_compressed_column.hpp>
#include <core/column.hpp>
#include <core/hash_table.hpp>
#include <core/table.hpp>
#include <core/vector_typed.hpp>
#include <query_compilation/minimal_api.hpp>

#include "compression/order_preserving_dictionary_compressed_column.hpp"

namespace CoGaDB {

// forward declarations
class ColumnBase;
typedef boost::shared_ptr<ColumnBase> ColumnPtr;

class BaseTable;
typedef boost::shared_ptr<BaseTable> TablePtr;

//    struct UncompressedStringColumnCache{
//
//        static UncompressedStringColumnCache& instance();
//        const boost::shared_ptr<Column<std::string> > get(ColumnPtr col);
//    private:
//        UncompressedStringColumnCache();
//        typedef
//        std::map<boost::weak_ptr<ColumnBase>,boost::shared_ptr<Column<std::string>
//        > > Map;
//        Map map_;
//    };
//
//    UncompressedStringColumnCache::UncompressedStringColumnCache()
//    : map_()
//    {
//
//    }
//
//    const boost::shared_ptr<Column<std::string> >
//    UncompressedStringColumnCache::get(ColumnPtr col){
//        Map::iterator it;
//        assert(col->type()==typeid(std::string));
//        assert(col->getColumnType()==DICTIONARY_COMPRESSED);
//        boost::shared_ptr<DictionaryCompressedColumn<std::string> > typed_col;
//        typed_col =
//        boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>
//        >(col);
//        if(!typed_col){
//            COGADB_FATAL_ERROR("Input Array pointer is NULL!","");
//            return boost::shared_ptr<Column<std::string> >();
//        }
//        it=map_.find(boost::weak_ptr<ColumnBase>(col));
//        if(it==map_.end()){
//            ProcessorSpecification proc_spec(hype::PD0);
////            ColumnBaseTyped<std::string>::DenseValueColumnPtr
/// typed_col->copyIntoDenseValueColumn(proc_spec);
//            boost::shared_ptr<Column<std::string> > uncompressed =
//            typed_col->copyIntoDenseValueColumn(proc_spec);
//            assert(uncompressed!=NULL);
//            std::pair<Map::iterator, bool> ret=map_.insert(std::make_pair(
//                    boost::weak_ptr<ColumnBase>(col),
//                    uncompressed));
//            assert(ret.second==true);
//            it=ret.first;
//        }
//        return it->second;
//    }
//
//    UncompressedStringColumnCache& UncompressedStringColumnCache::instance(){
//        static UncompressedStringColumnCache cache;
//        return cache;
//    }

ColumnPtr getColumn(TablePtr table, const std::string& name) {
  if (!table) {
    return ColumnPtr();
  }
  ColumnPtr col = table->getColumnbyName(name);
  if (!col) {
    std::cout << name << " not in ";
    table->printSchema();
    //            table->print();
  }
  assert(col != NULL && "Column not found!");
  //        if(col->getType()==VARCHAR &&
  //        col->getColumnType()!=PLAIN_MATERIALIZED){
  //            col = UncompressedStringColumnCache::instance().get(col);
  //        }else{
  if (!boost::dynamic_pointer_cast<Vector>(col))
    col = decompress_if_required(col);
  //        }
  assert(col != NULL && "Column could not be decompressed!");
  return col;
}
ColumnPtr getDictionaryCompressedColumn(TablePtr table,
                                        const std::string& name) {
  if (!table) {
    return ColumnPtr();
  }
  ColumnPtr col = table->getColumnbyName(name);
  if (!col) {
    std::cout << name << " not in ";
    table->printSchema();
  }
  assert(col != NULL && "Column not found!");
  assert(col->getColumnType() == DICTIONARY_COMPRESSED ||
         col->getColumnType() == DICTIONARY_COMPRESSED_ORDER_PRESERVING);
  return col;
}

HashTablePtr getHashTable(TablePtr table, const std::string& column_name) {
  if (!table) {
    assert(table != NULL);
    return HashTablePtr();
  }
  HashTablePtr hash_table = table->getHashTablebyName(column_name);
  if (!hash_table) {
    std::cout << "No Hash Table found for attribute: '" << column_name
              << "' in table with schema ";
    table->printSchema();
  }
  assert(hash_table != NULL && "Hash table not found!");
  return hash_table;
}

int32_t* getArrayFromColumn_int32_t(ColumnPtr col) {
  if (!col) return NULL;
  assert(col->type() == typeid(int32_t));
  boost::shared_ptr<Column<int32_t> > typed_col;
  typed_col = boost::dynamic_pointer_cast<Column<int32_t> >(col);
  if (!typed_col) {
    boost::shared_ptr<VectorTyped<int32_t> > vector;
    vector = boost::dynamic_pointer_cast<VectorTyped<int32_t> >(col);
    assert(vector != NULL);
    return vector->data();
  }
  return typed_col->data();
}

uint32_t* getArrayFromColumn_uint32_t(ColumnPtr col) {
  if (!col) return NULL;
  assert(col->type() == typeid(uint32_t));
  boost::shared_ptr<Column<uint32_t> > typed_col;
  typed_col = boost::dynamic_pointer_cast<Column<uint32_t> >(col);
  if (!typed_col) {
    boost::shared_ptr<VectorTyped<uint32_t> > vector;
    vector = boost::dynamic_pointer_cast<VectorTyped<uint32_t> >(col);
    assert(vector != NULL);
    return vector->data();
  }
  return typed_col->data();
}

uint64_t* getArrayFromColumn_uint64_t(ColumnPtr col) {
  if (!col) return NULL;
  assert(col->type() == typeid(uint64_t));
  boost::shared_ptr<Column<uint64_t> > typed_col;
  typed_col = boost::dynamic_pointer_cast<Column<uint64_t> >(col);
  if (!typed_col) {
    boost::shared_ptr<VectorTyped<uint64_t> > vector;
    vector = boost::dynamic_pointer_cast<VectorTyped<uint64_t> >(col);
    assert(vector != NULL);
    return vector->data();
  }
  return typed_col->data();
}

float* getArrayFromColumn_float(ColumnPtr col) {
  if (!col) return NULL;
  assert(col->type() == typeid(float));
  boost::shared_ptr<Column<float> > typed_col;
  typed_col = boost::dynamic_pointer_cast<Column<float> >(col);
  if (!typed_col) {
    boost::shared_ptr<VectorTyped<float> > vector;
    vector = boost::dynamic_pointer_cast<VectorTyped<float> >(col);
    assert(vector != NULL);
    return vector->data();
  }
  return typed_col->data();
}

double* getArrayFromColumn_double(ColumnPtr col) {
  if (!col) return NULL;
  assert(col->type() == typeid(double));
  boost::shared_ptr<Column<double> > typed_col;
  typed_col = boost::dynamic_pointer_cast<Column<double> >(col);
  if (!typed_col) {
    boost::shared_ptr<VectorTyped<double> > vector;
    vector = boost::dynamic_pointer_cast<VectorTyped<double> >(col);
    assert(vector != NULL);
    return vector->data();
  }
  return typed_col->data();
}

std::string* getArrayFromColumn_string(ColumnPtr col) {
  if (!col) return NULL;
  assert(col->type() == typeid(std::string));
  boost::shared_ptr<Column<std::string> > typed_col;
  typed_col = boost::dynamic_pointer_cast<Column<std::string> >(col);
  if (!typed_col) {
    boost::shared_ptr<VectorTyped<std::string> > vector;
    vector = boost::dynamic_pointer_cast<VectorTyped<std::string> >(col);
    assert(vector != NULL);
    assert(vector->data() != NULL);
    return vector->data();
  }
  return typed_col->data();
}

char* getArrayFromColumn_char(ColumnPtr col) {
  if (!col) return NULL;
  assert(col->type() == typeid(char));
  boost::shared_ptr<Column<char> > typed_col;
  typed_col = boost::dynamic_pointer_cast<Column<char> >(col);
  if (!typed_col) {
    boost::shared_ptr<VectorTyped<char> > vector;
    vector = boost::dynamic_pointer_cast<VectorTyped<char> >(col);
    assert(vector != NULL);
    assert(vector->data() != NULL);
    return vector->data();
  }
  return typed_col->data();
}

uint32_t* getArrayCompressedKeysFromColumn_string(ColumnPtr col) {
  if (!col) return NULL;
  assert(col->type() == typeid(std::string));
  assert(col->getColumnType() == DICTIONARY_COMPRESSED ||
         col->getColumnType() == DICTIONARY_COMPRESSED_ORDER_PRESERVING);
  boost::shared_ptr<DictionaryCompressedColumn<std::string> >
      dict_compressed_col;
  boost::shared_ptr<OrderPreservingDictionaryCompressedColumn<std::string> >
      ordered_dict_compressed_col;
  dict_compressed_col =
      boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string> >(
          col);
  ordered_dict_compressed_col = boost::dynamic_pointer_cast<
      OrderPreservingDictionaryCompressedColumn<std::string> >(col);
  if (dict_compressed_col) {
    return dict_compressed_col->getIdData();
  } else if (ordered_dict_compressed_col) {
    return ordered_dict_compressed_col->getIdData();
  } else {
    boost::shared_ptr<VectorTyped<std::string> > vector;
    vector = boost::dynamic_pointer_cast<VectorTyped<std::string> >(col);
    assert(vector != NULL);
    assert(vector->getDictionaryCompressedKeys() != NULL);
    return vector->getDictionaryCompressedKeys();
  }
}

const TablePtr createTableFromColumns(const std::string& table_name,
                                      ColumnPtr* columns, size_t num_columns) {
  std::vector<ColumnPtr> result_columns;
  for (size_t i = 0; i < num_columns; ++i) {
    assert(columns[i] != NULL);
    result_columns.push_back(columns[i]);
  }

  return createTableFromColumns(table_name, result_columns);
}

const TablePtr createTableFromColumns(const std::string& table_name,
                                      const std::vector<ColumnPtr>& columns) {
  TablePtr result_table(new Table(table_name, columns));
  return result_table;
}

bool addHashTable(TablePtr table, const std::string& column_name,
                  HashTablePtr hash_table) {
  if (!table) return false;
  return table->addHashTable(column_name, hash_table);
}

const HashTablePtr createSystemHashTable(
    void* ht, HashTableCleanupFunctionPtr cleanup_handler,
    const std::string& identifier) {
  HashTablePtr hash_table(new HashTable(ht, cleanup_handler, identifier));
  return hash_table;
}

void* getHashTableFromSystemHashTable(HashTablePtr hash_table) {
  return hash_table->ptr;
}

size_t getNumberOfRows(TablePtr table) {
  if (table) {
    return table->getNumberofRows();
  } else {
    COGADB_FATAL_ERROR("Invalid table pointer!", "");
  }
}

ColumnPtr createResultArray_int32_t(const std::string& name,
                                    const int32_t* array, size_t num_elements) {
  return ColumnPtr(new Column<int32_t>(name, INT, array, array + num_elements));
}
ColumnPtr createResultArray_uint32_t(const std::string& name,
                                     const uint32_t* array,
                                     size_t num_elements) {
  return ColumnPtr(
      new Column<uint32_t>(name, UINT32, array, array + num_elements));
}
ColumnPtr createResultArray_uint64_t(const std::string& name,
                                     const uint64_t* array,
                                     size_t num_elements) {
  return ColumnPtr(
      new Column<uint64_t>(name, OID, array, array + num_elements));
}
ColumnPtr createResultArray_float(const std::string& name, const float* array,
                                  size_t num_elements) {
  return ColumnPtr(new Column<float>(name, FLOAT, array, array + num_elements));
}
ColumnPtr createResultArray_double(const std::string& name, const double* array,
                                   size_t num_elements) {
  return ColumnPtr(
      new Column<double>(name, DOUBLE, array, array + num_elements));
}
ColumnPtr createResultArray_string(const std::string& name,
                                   const std::string* array,
                                   size_t num_elements) {
  return ColumnPtr(
      new Column<std::string>(name, VARCHAR, array, array + num_elements));
}
ColumnPtr createResultArray_char(const std::string& name, const char* array,
                                 size_t num_elements) {
  return ColumnPtr(new Column<char>(name, CHAR, array, array + num_elements));
}

boost::mutex string_malloc_mutex;

std::string* stringMalloc(size_t number_of_elements) {
  boost::lock_guard<boost::mutex> lock(string_malloc_mutex);
  return AllocationManager<std::string>::malloc(hype::PD_Memory_0,
                                                number_of_elements);
}

void stringFree(std::string*& data) {
  boost::lock_guard<boost::mutex> lock(string_malloc_mutex);
  AllocationManager<std::string>::free(hype::PD_Memory_0, data);
}

std::string* stringRealloc(std::string* data, size_t number_of_elements) {
  boost::lock_guard<boost::mutex> lock(string_malloc_mutex);
  return AllocationManager<std::string>::realloc(hype::PD_Memory_0, data,
                                                 number_of_elements);
}

}  // end namespace CoGaDB
