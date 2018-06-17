
#include <compression/dictionary_compressed_column.hpp>
#include <core/base_column.hpp>
#include <core/column.hpp>
#include <core/processor_data_cache.hpp>
#include <iostream>

#include <compression/bit_vector_compressed_column.hpp>
#include <compression/bitpacked_dictionary_compressed_column.hpp>
#include <compression/delta_compressed_column.hpp>
#include <compression/order_preserving_dictionary_compressed_column.hpp>
#include <compression/reference_based_compressed_column.hpp>
#include <compression/rle_compressed_column.hpp>
#include <compression/rle_compressed_column_with_prefix_counts.hpp>
#include <compression/rle_delta_one_compressed_column_int.hpp>
#include <compression/rle_delta_one_compressed_column_int_with_prefix_counts.hpp>
#include <compression/void_compressed_column_int.hpp>
#include <lookup_table/join_index.hpp>

#include <backends/processor_backend.hpp>
#include <util/utility_functions.hpp>

using namespace std;

namespace CoGaDB {

ColumnBase::ColumnBase(const std::string& name, AttributeType db_type,
                       ColumnType column_type)
    : name_(name),
      db_type_(db_type),
      column_type_(column_type),
      is_loaded_(true) {}

ColumnBase::~ColumnBase() {}

AttributeType ColumnBase::getType() const throw() { return db_type_; }

void ColumnBase::setType(const AttributeType& type) { this->db_type_ = type; }

ColumnType ColumnBase::getColumnType() const throw() { return column_type_; }

bool ColumnBase::isLoadedInMainMemory() const throw() { return is_loaded_; }

void ColumnBase::setStatusLoadedInMainMemory(bool is_loaded) throw() {
  is_loaded_ = is_loaded;
}

const string ColumnBase::getName() const throw() { return name_; }

void ColumnBase::setName(const std::string& value) throw() {
  this->name_ = value;
}

const ColumnPtr createColumn(AttributeType type, const std::string& name) {
  typedef std::map<AttributeType, ColumnType> DefaultCompressionMethods;
  DefaultCompressionMethods default_compression;
  default_compression[VARCHAR] = DICTIONARY_COMPRESSED;  // PLAIN_MATERIALIZED;

  ColumnType col_type = PLAIN_MATERIALIZED;

  // if we find a default compression method in the map, we use it,
  // if not, just use uncompressed columns
  DefaultCompressionMethods::const_iterator cit =
      default_compression.find(type);
  if (cit != default_compression.end()) {
    col_type = cit->second;
  }
  return createColumn(type, name, col_type);
}

template <typename T>
const ColumnPtr CreateColumnPtr(AttributeType type, const std::string& name,
                                ColumnType column_type) {
  switch (column_type) {
    case PLAIN_MATERIALIZED:
      return ColumnPtr(new Column<T>(name, type));
    case DICTIONARY_COMPRESSED:
      return ColumnPtr(new DictionaryCompressedColumn<T>(name, type));
    case DICTIONARY_COMPRESSED_ORDER_PRESERVING:
      return ColumnPtr(
          new OrderPreservingDictionaryCompressedColumn<T>(name, type));
    case RUN_LENGTH_COMPRESSED:
      return ColumnPtr(new RLECompressedColumn<T>(name, type));
    case DELTA_COMPRESSED:
      return ColumnPtr(new DeltaCompressedColumn<T>(name, type));
    case BIT_VECTOR_COMPRESSED:
      return ColumnPtr(new BitVectorCompressedColumn<T>(name, type));
    case BITPACKED_DICTIONARY_COMPRESSED:
      return ColumnPtr(new BitPackedDictionaryCompressedColumn<T>(name, type));
    case RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER:
      return ColumnPtr(new RLEDeltaOneCompressedColumnNumber<T>(name, type));
    case VOID_COMPRESSED_NUMBER:
      return ColumnPtr(new VoidCompressedColumnNumber<T>(name, type));
    case REFERENCE_BASED_COMPRESSED:
      return ColumnPtr(new ReferenceBasedCompressedColumn<T>(name, type));
    case RUN_LENGTH_COMPRESSED_PREFIX:
      return ColumnPtr(new RLECompressedColumnWithPrefixCounts<T>(name, type));
    case RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER_PREFIX:
      return ColumnPtr(
          new RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>(name, type));
    case LOOKUP_ARRAY:
      COGADB_ERROR(
          "Lookup Array may not be created function CreateColumnPtr()!", "");
      return ColumnPtr();
  }
  return ColumnPtr();
}

const ColumnPtr createColumn(AttributeType type, const std::string& name,
                             ColumnType column_type) {
  ColumnPtr ptr;
  if (type == INT) {
    // TypedColumnFactory factory;
    ptr = CreateColumnPtr<int>(type, name, column_type);
  } else if (type == UINT32) {
    // TypedColumnFactory factory;
    ptr = CreateColumnPtr<uint32_t>(type, name, column_type);
  } else if (type == DATE) {
    // TypedColumnFactory factory;
    ptr = CreateColumnPtr<uint32_t>(type, name, column_type);
  } else if (type == CHAR) {
    // TypedColumnFactory factory;
    ptr = CreateColumnPtr<char>(type, name, column_type);
  } else if (type == OID) {
    // TypedColumnFactory factory;
    ptr = CreateColumnPtr<TID>(type, name, column_type);
  } else if (type == FLOAT) {
    ptr = CreateColumnPtr<float>(type, name, column_type);
  } else if (type == DOUBLE) {
    ptr = CreateColumnPtr<double>(type, name, column_type);
  } else if (type == VARCHAR) {
    // ptr=ColumnPtr(new Column<string>(name,VARCHAR));
    ptr = CreateColumnPtr<std::string>(type, name, column_type);
  } else if (type == BOOLEAN) {
    // ptr=ColumnPtr(new Column<bool>(name,BOOLEAN));
    cout << "Fatal Error! invalid AttributeType: " << type
         << " for Column: " << name
         << " Note: bool is currently not supported, will be added again in "
            "the future!"
         << endl;
  } else {
    cout << "Fatal Error! invalid AttributeType: " << type
         << " for Column: " << name << endl;
  }
  return ptr;
}

bool operator!=(const ColumnBase& lhs, const ColumnBase& rhs) {
  return !(const_cast<ColumnBase&>(lhs) == const_cast<ColumnBase&>(rhs));
}

const PositionListPtr createPositionList(
    size_t num_of_elements, const hype::ProcessingDeviceMemoryID& mem_id) {
  PositionListPtr tids;
  try {
    tids = PositionListPtr(new PositionList("", OID, mem_id));
    if (num_of_elements > 0) {
      tids->resize(num_of_elements);
    }
  } catch (std::bad_alloc& e) {
    return PositionListPtr();
  }
  return tids;
}

const PositionListPtr createPositionList(
    size_t num_of_elements, const ProcessorSpecification& proc_spec) {
  return createPositionList(num_of_elements, getMemoryID(proc_spec));
}

const PositionListPtr createAscendingPositionList(
    size_t num_of_elements, const ProcessorSpecification& proc_spec,
    const TID start_value) {
  PositionListPtr tids = createPositionList(num_of_elements, proc_spec);
  if (!tids) return PositionListPtr();
  ProcessorBackend<TID>* backend =
      ProcessorBackend<TID>::get(proc_spec.proc_id);
  bool ret = backend->generateAscendingSequence(tids->data(), num_of_elements,
                                                start_value, proc_spec);
  if (ret) {
    return tids;
  } else {
    return PositionListPtr();
  }
}

TID* getPointer(PositionList& tids) {
  // if(!tids) return NULL;
  if (tids.empty()) return NULL;
  return tids.data();
}

size_t getSize(const PositionList& tids) {
  // if(!tids) return 0;
  return tids.size();
}

bool resize(PositionList& tids, size_t new_num_elements) {
  try {
    tids.resize(new_num_elements);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory in memory " << (int)tids.getMemoryID()
              << "!" << std::endl;
    return false;
  }
  return true;
}

hype::ProcessingDeviceMemoryID getMemoryID(const PositionList& tids) {
  return tids.getMemoryID();
}

ColumnPtr copy(ColumnPtr col, const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!col) return ColumnPtr();
  return col->copy(mem_id);
}

PositionListPtr copy(PositionListPtr tids,
                     const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!tids) return PositionListPtr();
  ColumnPtr result = tids->copy(mem_id);
  return boost::dynamic_pointer_cast<PositionList>(result);
}

PositionListPairPtr copy(PositionListPairPtr pair_tids,
                         const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!pair_tids) return PositionListPairPtr();
  if (!pair_tids->first) return PositionListPairPtr();
  if (!pair_tids->second) return PositionListPairPtr();

  // at least one positionlist needs to be copied
  PositionListPairPtr new_pair_tids(new PositionListPair());
  new_pair_tids->first = copy(pair_tids->first, mem_id);
  new_pair_tids->second = copy(pair_tids->second, mem_id);
  // if at least one copy operation failed, this operation fails as well
  if (!new_pair_tids->first || !new_pair_tids->second)
    return PositionListPairPtr();

  return new_pair_tids;
}

LookupColumnPtr copy(LookupColumnPtr lookup_column,
                     const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!lookup_column) return LookupColumnPtr();
  return lookup_column->copy(mem_id);
}

JoinIndexPtr copy(JoinIndexPtr join_index,
                  const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!join_index) return JoinIndexPtr();
  if (!join_index->first) return JoinIndexPtr();
  if (!join_index->second) return JoinIndexPtr();

  // at least one LookupColumn needs to be copied
  JoinIndexPtr new_join_index(new JoinIndex());
  new_join_index->first = copy(join_index->first, mem_id);
  new_join_index->second = copy(join_index->second, mem_id);
  // if at least one copy operation failed, this operation fails as well
  if (!new_join_index->first || !new_join_index->second) return JoinIndexPtr();

  return new_join_index;
}

BitmapPtr copy(BitmapPtr bitmap, const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!bitmap) BitmapPtr();
  return bitmap->copy(mem_id);
}

std::vector<ColumnPtr> copy(const std::vector<ColumnPtr>& col_vec,
                            const hype::ProcessingDeviceMemoryID& mem_id) {
  std::vector<ColumnPtr> placed_col_vec;
  for (size_t i = 0; i < col_vec.size(); ++i) {
    ColumnPtr tmp = copy(col_vec[i], mem_id);
    if (!tmp) {
      return std::vector<ColumnPtr>();
    } else {
      placed_col_vec.push_back(tmp);
    }
  }
  return placed_col_vec;
}

/*! create a copy only of data is not dorment in memory with ID mem_id,
    this copies data between processors on demand*/
ColumnPtr copy_if_required(ColumnPtr col,
                           const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!col) return ColumnPtr();
  if (col->getMemoryID() == mem_id) {
    return col;
  } else {
    if (!isCPUMemory(mem_id)) {
      return DataCacheManager::instance().getDataCache(mem_id).getColumn(col);
    } else {
      ColumnPtr cached = DataCacheManager::instance()
                             .getDataCache(col->getMemoryID())
                             .getHostColumn(col);
      if (cached) return cached;
    }
    return copy(col, mem_id);
  }
}

std::vector<ColumnPtr> copy_if_required(
    const std::vector<ColumnPtr>& col_vec,
    const hype::ProcessingDeviceMemoryID& mem_id) {
  std::vector<ColumnPtr> placed_col_vec;
  for (size_t i = 0; i < col_vec.size(); ++i) {
    ColumnPtr tmp = copy_if_required(col_vec[i], mem_id);
    if (!tmp) {
      return std::vector<ColumnPtr>();
    } else {
      placed_col_vec.push_back(tmp);
    }
  }
  return placed_col_vec;
}

PositionListPtr copy_if_required(PositionListPtr tids,
                                 const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!tids) return PositionListPtr();
  if (tids->getMemoryID() == mem_id) {
    return tids;
  } else {
    if (!isCPUMemory(mem_id) &&
        JoinIndexes::instance().isReverseJoinIndex(tids)) {
      ColumnPtr placed_tids =
          DataCacheManager::instance().getDataCache(mem_id).getColumn(tids);
      if (!placed_tids) return PositionListPtr();
      return boost::dynamic_pointer_cast<PositionList>(placed_tids);
    }
    return copy(tids, mem_id);
  }
}

PositionListPairPtr copy_if_required(
    PositionListPairPtr pair_tids,
    const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!pair_tids) return PositionListPairPtr();
  if (!pair_tids->first) return PositionListPairPtr();
  if (!pair_tids->second) return PositionListPairPtr();

  if (pair_tids->first->getMemoryID() == mem_id &&
      pair_tids->second->getMemoryID() == mem_id) {
    return pair_tids;
  }
  // at least one positionlist needs to be copied
  PositionListPairPtr new_pair_tids(new PositionListPair());
  new_pair_tids->first = copy_if_required(pair_tids->first, mem_id);
  new_pair_tids->second = copy_if_required(pair_tids->second, mem_id);
  // if at least one copy operation failed, this operation fails as well
  if (!new_pair_tids->first || !new_pair_tids->second)
    return PositionListPairPtr();

  return new_pair_tids;
}

LookupColumnPtr copy_if_required(LookupColumnPtr lookup_column,
                                 const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!lookup_column) return LookupColumnPtr();
  if (lookup_column->getMemoryID() == mem_id) {
    return lookup_column;
  } else {
    return copy(lookup_column, mem_id);
  }
}

JoinIndexPtr copy_if_required(JoinIndexPtr join_index,
                              const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!join_index) return JoinIndexPtr();
  if (!join_index->first) return JoinIndexPtr();
  if (!join_index->second) return JoinIndexPtr();

  // is already in correct memory?
  if (join_index->first->getMemoryID() == mem_id &&
      join_index->second->getMemoryID() == mem_id) {
    return join_index;
  }

  if (!isCPUMemory(mem_id)) {
    return DataCacheManager::instance().getDataCache(mem_id).getJoinIndex(
        join_index);
  }

  // at least one LookupColumn needs to be copied
  JoinIndexPtr new_join_index(new JoinIndex());
  new_join_index->first = copy_if_required(join_index->first, mem_id);
  new_join_index->second = copy_if_required(join_index->second, mem_id);
  // if at least one copy operation failed, this operation fails as well
  if (!new_join_index->first || !new_join_index->second) return JoinIndexPtr();

  return new_join_index;
}

BitmapPtr copy_if_required(BitmapPtr bitmap,
                           const hype::ProcessingDeviceMemoryID& mem_id) {
  if (!bitmap) return BitmapPtr();
  if (bitmap->getMemoryID() == mem_id) {
    return bitmap;
  } else {
    return copy(bitmap, mem_id);
  }
}

ColumnPtr copy_if_required(ColumnPtr col,
                           const ProcessorSpecification& proc_spec) {
  if (!col) return ColumnPtr();
  return copy_if_required(col, hype::util::getMemoryID(proc_spec.proc_id));
}

std::vector<ColumnPtr> copy_if_required(
    const std::vector<ColumnPtr>& col_vec,
    const ProcessorSpecification& proc_spec) {
  return copy_if_required(col_vec, hype::util::getMemoryID(proc_spec.proc_id));
}

PositionListPtr copy_if_required(PositionListPtr tids,
                                 const ProcessorSpecification& proc_spec) {
  return copy_if_required(tids, hype::util::getMemoryID(proc_spec.proc_id));
}

PositionListPairPtr copy_if_required(PositionListPairPtr pair_tids,
                                     const ProcessorSpecification& proc_spec) {
  return copy_if_required(pair_tids,
                          hype::util::getMemoryID(proc_spec.proc_id));
}

LookupColumnPtr copy_if_required(LookupColumnPtr lookup_column,
                                 const ProcessorSpecification& proc_spec) {
  return copy_if_required(lookup_column,
                          hype::util::getMemoryID(proc_spec.proc_id));
}

JoinIndexPtr copy_if_required(JoinIndexPtr join_index,
                              const ProcessorSpecification& proc_spec) {
  return copy_if_required(join_index,
                          hype::util::getMemoryID(proc_spec.proc_id));
}

BitmapPtr copy_if_required(BitmapPtr bitmap,
                           const ProcessorSpecification& proc_spec) {
  return copy_if_required(bitmap, hype::util::getMemoryID(proc_spec.proc_id));
}

ColumnPtr decompress_if_required(ColumnPtr col) {
  if (!col) return ColumnPtr();
  if (col->getColumnType() != PLAIN_MATERIALIZED) {
    ProcessorSpecification proc_spec(hype::PD0);
    return col->decompress(proc_spec);
  } else {
    return col;
  }
}

PositionListPtr computePositionListUnion(PositionListPtr tids1,
                                         PositionListPtr tids2) {
  PositionListPtr tmp_tids(createPositionList(tids1->size() + tids2->size()));
  PositionList::iterator it;

  it = std::set_union(tids1->begin(), tids1->end(), tids2->begin(),
                      tids2->end(), tmp_tids->begin());
  // set size to actual result size (union eliminates duplicates)
  tmp_tids->resize(it - tmp_tids->begin());
  return tmp_tids;
}

PositionListPtr computePositionListIntersection(PositionListPtr tids1,
                                                PositionListPtr tids2) {
  PositionListPtr tmp_tids(createPositionList(tids1->size() + tids2->size()));
  PositionList::iterator it;

  it = std::set_intersection(tids1->begin(), tids1->end(), tids2->begin(),
                             tids2->end(), tmp_tids->begin());
  // set size to actual result size (union eliminates duplicates)
  tmp_tids->resize(it - tmp_tids->begin());
  return tmp_tids;
}

}  // end namespace CogaDB
