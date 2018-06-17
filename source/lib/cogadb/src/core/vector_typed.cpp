
#include <assert.h>
#include <boost/lexical_cast.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <compression/order_preserving_dictionary_compressed_column.hpp>
#include <core/block_iterator.hpp>
#include <core/column.hpp>
#include <core/vector_typed.hpp>
#include <util/getname.hpp>

namespace CoGaDB {

template <typename T>
VectorTyped<T>::VectorTyped(TID begin_index, size_t num_elements,
                            ColumnTypedPtr source_column)
    : Vector(source_column->getName(), source_column->getType(),
             source_column->getColumnType(), begin_index, num_elements),
      source_column_(source_column) {
  assert(source_column_ != NULL);
}

template <typename T>
VectorTyped<T>::~VectorTyped() {}

template <typename T>
bool VectorTyped<T>::insert(const boost::any& new_Value) {
  return false;
}

template <typename T>
bool VectorTyped<T>::insert(const T& new_Value) {
  return false;
}

template <typename T>
bool VectorTyped<T>::update(TID tid, const boost::any& new_value) {
  return false;
}

template <typename T>
bool VectorTyped<T>::update(PositionListPtr tid, const boost::any& new_value) {
  return false;
}

template <typename T>
bool VectorTyped<T>::append(ColumnPtr col) {
  return false;
}

template <typename T>
bool VectorTyped<T>::remove(TID tid) {
  return false;
}

template <typename T>
bool VectorTyped<T>::remove(PositionListPtr tid) {
  return false;
}

template <typename T>
bool VectorTyped<T>::clearContent() {
  return false;
}

template <typename T>
const boost::any VectorTyped<T>::get(TID tid) {
  T* array = this->data();
  assert(tid < num_elements_);
  return boost::any(array[tid]);
}

template <typename T>
std::string VectorTyped<T>::getStringValue(TID tid) {
  T* array = this->data();
  assert(tid < num_elements_);
  return boost::lexical_cast<std::string>(array[tid]);
}

template <typename T>
void VectorTyped<T>::print() const throw() {
  std::cout << "Vector: " << std::endl;
  std::cout << "Offset: " << begin_index_ << std::endl;
  std::cout << "Size: " << num_elements_ << std::endl;
  T* array = this->data();
  for (size_t i = 0; i < num_elements_; ++i) {
    std::cout << array[i] << std::endl;
  }
}

template <typename T>
size_t VectorTyped<T>::size() const throw() {
  return this->num_elements_;
}

template <typename T>
size_t VectorTyped<T>::getSizeinBytes() const throw() {
  return 0;
}

template <typename T>
const ColumnPtr VectorTyped<T>::changeCompression(
    const ColumnType& col_type) const {
  return ColumnPtr();
}

template <typename T>
const ColumnPtr VectorTyped<T>::copy() const {
  return ColumnPtr();
}

template <typename T>
const ColumnPtr VectorTyped<T>::copy(
    const hype::ProcessingDeviceMemoryID&) const {
  return ColumnPtr();
}

template <typename T>
const typename VectorTyped<T>::DenseValueColumnPtr
VectorTyped<T>::copyIntoDenseValueColumn(
    const ProcessorSpecification& proc_spec) const {
  return DenseValueColumnPtr();
}

template <typename T>
const DoubleDenseValueColumnPtr VectorTyped<T>::convertToDenseValueDoubleColumn(
    const ProcessorSpecification& proc_spec) const {
  return DoubleDenseValueColumnPtr();
}

template <typename T>
const StringDenseValueColumnPtr
VectorTyped<T>::convertToDenseValueStringColumn() const {
  return StringDenseValueColumnPtr();
}

template <typename T>
T* VectorTyped<T>::data() const {
  T* array = NULL;
  if (source_column_->getColumnType() == PLAIN_MATERIALIZED) {
    boost::shared_ptr<Column<T> > col;
    col = boost::dynamic_pointer_cast<Column<T> >(source_column_);
    assert(col != NULL);
    T* tmp = col->data();
    array = &tmp[begin_index_];
  } else if (source_column_->getColumnType() == DICTIONARY_COMPRESSED) {
    boost::shared_ptr<Column<T> > col;
    col = UncompressedStringColumnCache::instance().get(source_column_);
    assert(col != NULL);
    T* tmp = col->data();
    array = &tmp[begin_index_];
  } else if (source_column_->getColumnType() ==
             DICTIONARY_COMPRESSED_ORDER_PRESERVING) {
    boost::shared_ptr<Column<T> > col;
    col = UncompressedStringColumnCache::instance().get(source_column_);
    assert(col != NULL);
    T* tmp = col->data();
    array = &tmp[begin_index_];
  } else {
    COGADB_FATAL_ERROR("" << util::getName(source_column_->getColumnType()),
                       "");
  }
  assert(array != NULL);
  return array;
}

template <typename T>
uint32_t* VectorTyped<T>::getDictionaryCompressedKeys() const {
  uint32_t* array = NULL;
  assert(source_column_->getColumnType() == DICTIONARY_COMPRESSED ||
         source_column_->getColumnType() ==
             DICTIONARY_COMPRESSED_ORDER_PRESERVING);
  boost::shared_ptr<DictionaryCompressedColumn<T> > dict_compressed_col;
  boost::shared_ptr<OrderPreservingDictionaryCompressedColumn<T> >
      ordered_dict_compressed_col;
  dict_compressed_col =
      boost::dynamic_pointer_cast<DictionaryCompressedColumn<T> >(
          source_column_);
  ordered_dict_compressed_col = boost::dynamic_pointer_cast<
      OrderPreservingDictionaryCompressedColumn<T> >(source_column_);
  if (dict_compressed_col) {
    uint32_t* tmp = dict_compressed_col->getIdData();
    array = &tmp[begin_index_];
  } else if (ordered_dict_compressed_col) {
    uint32_t* tmp = ordered_dict_compressed_col->getIdData();
    array = &tmp[begin_index_];
  } else {
    COGADB_FATAL_ERROR(
        "Could not retrieve vector for dictionary compressed columns key "
        "column!",
        "");
  }
  return array;
}

template <typename T>
const ColumnPtr VectorTyped<T>::decompress(
    const ProcessorSpecification& proc_spec) const {
  return ColumnPtr();
}

template <typename T>
hype::ProcessingDeviceMemoryID VectorTyped<T>::getMemoryID() const {
  return hype::PD_Memory_0;
}

template <typename T>
const ColumnPtr VectorTyped<T>::gather(PositionListPtr tid_list,
                                       const GatherParam&) {
  return ColumnPtr();
}

template <typename T>
const PositionListPtr VectorTyped<T>::sort(const SortParam& param) {
  return PositionListPtr();
}

template <typename T>
const PositionListPtr VectorTyped<T>::selection(const SelectionParam& param) {
  return PositionListPtr();
}

template <typename T>
const BitmapPtr VectorTyped<T>::bitmap_selection(const SelectionParam& param) {
  return BitmapPtr();
}

template <typename T>
const PositionListPtr VectorTyped<T>::parallel_selection(
    const boost::any& value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  return PositionListPtr();
}

template <typename T>
const PositionListPtr VectorTyped<T>::lock_free_parallel_selection(
    const boost::any& value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  return PositionListPtr();
}

template <typename T>
const PositionListPairPtr VectorTyped<T>::hash_join(ColumnPtr join_column) {
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr VectorTyped<T>::parallel_hash_join(
    ColumnPtr join_column, unsigned int number_of_threads) {
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr VectorTyped<T>::sort_merge_join(
    ColumnPtr join_column) {
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr VectorTyped<T>::nested_loop_join(
    ColumnPtr join_column) {
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr VectorTyped<T>::radix_join(ColumnPtr join_column) {
  return PositionListPairPtr();
}

template <typename T>
const PositionListPairPtr VectorTyped<T>::join(ColumnPtr join_column,
                                               const JoinParam&) {
  return PositionListPairPtr();
}

template <typename T>
const PositionListPtr VectorTyped<T>::tid_semi_join(ColumnPtr join_column,
                                                    const JoinParam&) {
  return PositionListPtr();
}

template <typename T>
const BitmapPtr VectorTyped<T>::bitmap_semi_join(ColumnPtr join_column,
                                                 const JoinParam&) {
  return BitmapPtr();
}

template <typename T>
const AggregationResult VectorTyped<T>::aggregate(const AggregationParam&) {
  return AggregationResult();
}

template <typename T>
const ColumnPtr VectorTyped<T>::column_algebra_operation(
    ColumnPtr source_column, const AlgebraOperationParam&) {
  return ColumnPtr();
}

template <typename T>
const ColumnPtr VectorTyped<T>::column_algebra_operation(
    const boost::any& value, const AlgebraOperationParam&) {
  return ColumnPtr();
}

template <typename T>
const ColumnGroupingKeysPtr VectorTyped<T>::createColumnGroupingKeys(
    const ProcessorSpecification& proc_spec) const {
  return ColumnGroupingKeysPtr();
}

template <typename T>
size_t VectorTyped<T>::getNumberOfRequiredBits() const {
  return 65;
}

template <typename T>
const AggregationResult VectorTyped<T>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, const AggregationParam&) {
  return AggregationResult();
}

template <typename T>
bool VectorTyped<T>::store(const std::string& path) {
  return false;
}

template <typename T>
bool VectorTyped<T>::load(const std::string& path,
                          ColumnLoaderMode column_loader_mode) {
  return false;
}

template <typename T>
bool VectorTyped<T>::isMaterialized() const throw() {
  return false;
}

template <typename T>
bool VectorTyped<T>::isCompressed() const throw() {
  return false;
}

template <typename T>
const ColumnPtr VectorTyped<T>::materialize() throw() {
  return ColumnPtr();
}

template <typename T>
bool VectorTyped<T>::is_equal(ColumnPtr column) {
  return false;
}

template <typename T>
bool VectorTyped<T>::isApproximatelyEqual(ColumnPtr column) {
  return false;
}

template <typename T>
bool VectorTyped<T>::operator==(ColumnBase& col) {
  return false;
}

template <typename T>
int VectorTyped<T>::compareValuesAtIndexes(TID id1, TID id2) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  return 0;
}

template <typename T>
bool VectorTyped<T>::setPrimaryKeyConstraint() {
  return false;
}

template <typename T>
bool VectorTyped<T>::hasPrimaryKeyConstraint() const throw() {
  return source_column_->hasPrimaryKeyConstraint();
}

template <typename T>
bool VectorTyped<T>::hasForeignKeyConstraint() const throw() {
  return source_column_->hasForeignKeyConstraint();
}

template <typename T>
bool VectorTyped<T>::setForeignKeyConstraint(
    const ForeignKeyConstraint& prim_foreign_key_reference) {
  return false;
}

template <typename T>
const ForeignKeyConstraint& VectorTyped<T>::getForeignKeyConstraint() {
  return source_column_->getForeignKeyConstraint();
}

template <typename T>
const ColumnStatistics& VectorTyped<T>::getColumnStatistics() const {
  return source_column_->getColumnStatistics();
}

template <typename T>
const ExtendedColumnStatistics<T>& VectorTyped<T>::getExtendedColumnStatistics()
    const {
  return source_column_->getExtendedColumnStatistics();
}

template <typename T>
bool VectorTyped<T>::computeColumnStatistics() {
  return false;
}

template <typename T>
const std::type_info& VectorTyped<T>::type() const throw() {
  return source_column_->type();
}

template <typename T>
T& VectorTyped<T>::operator[](const TID index) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  static T t;
  return t;
}

template <typename T>
typename VectorTyped<T>::ColumnTypedPtr VectorTyped<T>::getSourceColumn()
    const {
  return source_column_;
}

const ColumnPtr createVector(ColumnPtr col, BlockIterator& it) {
  if (!col) return ColumnPtr();

  ColumnPtr result;
  //
  ////        FLOAT, VARCHAR, BOOLEAN, UINT32, OID, DOUBLE, CHAR, DATE
  //        if(col->getType()==INT){
  //            boost::shared_ptr<ColumnBaseTyped<int32_t> > col;
  //
  //        }else if(col->getType()==FLOAT){
  //
  //        }else if(col->getType()==BOOLEAN){
  //
  //        }else if(col->getType()==UINT32){
  //
  //        }else if(col->getType()==OID){
  //
  //        }else if(col->getType()==DOUBLE){
  //
  //        }else if(col->getType()==CHAR){
  //
  //        }else if(col->getType()==DATE){
  //
  //        }
  //
  return result;
}

template <typename T>
VectorTyped<T>::UncompressedStringColumnCache::UncompressedStringColumnCache()
    : map_(), mutex_() {}

template <typename T>
const boost::shared_ptr<Column<T> >
VectorTyped<T>::UncompressedStringColumnCache::get(ColumnPtr col) {
  boost::lock_guard<boost::mutex> lock(mutex_);
  typename Map::iterator it;
  assert(col->type() == typeid(T));
  assert(col->getColumnType() == DICTIONARY_COMPRESSED ||
         col->getColumnType() == DICTIONARY_COMPRESSED_ORDER_PRESERVING);
  boost::shared_ptr<ColumnBaseTyped<T> > typed_col;
  typed_col = boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(col);
  if (!typed_col) {
    COGADB_FATAL_ERROR("Input Array pointer is NULL!", "");
    return boost::shared_ptr<Column<T> >();
  }
  it = map_.find(boost::weak_ptr<ColumnBase>(col));
  if (it == map_.end()) {
    ProcessorSpecification proc_spec(hype::PD0);
    //            ColumnBaseTyped<std::string>::DenseValueColumnPtr
    //            typed_col->copyIntoDenseValueColumn(proc_spec);
    boost::shared_ptr<Column<T> > uncompressed =
        typed_col->copyIntoDenseValueColumn(proc_spec);
    assert(uncompressed != NULL);
    std::pair<typename Map::iterator, bool> ret = map_.insert(
        std::make_pair(boost::weak_ptr<ColumnBase>(col), uncompressed));
    //            assert(ret.second==true);
    it = ret.first;
  }
  return it->second;
}

boost::mutex init_mutex_uncompressed_cache;
template <typename T>
typename VectorTyped<T>::UncompressedStringColumnCache&
VectorTyped<T>::UncompressedStringColumnCache::instance() {
  boost::lock_guard<boost::mutex> lock(init_mutex_uncompressed_cache);
  static UncompressedStringColumnCache cache;
  return cache;
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(VectorTyped)

}  // end namespace CoGaDB
