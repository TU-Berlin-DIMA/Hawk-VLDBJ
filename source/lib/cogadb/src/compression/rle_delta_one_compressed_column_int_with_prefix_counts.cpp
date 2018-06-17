#include <backends/gpu/util.hpp>
#include <compression/rle_delta_one_compressed_column_int_with_prefix_counts.hpp>

namespace CoGaDB {

/***************** Start of Implementation Section ******************/

template <class T>
RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::
    RLEDeltaOneCompressedColumnNumberWithPrefixCounts(const std::string &name,
                                                      AttributeType db_type)
    : CompressedColumn<T>(name, db_type,
                          RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER_PREFIX),
      values_(),
      count_(),
      hack_last_uncompressed(),
      _last_lookup_index(0),
      _last_index_position(0),
      _last_row_sum(0) {
  // TODO Throw error if T not integer type (int, long, ...))
}

template <class T>
RLEDeltaOneCompressedColumnNumberWithPrefixCounts<
    T>::~RLEDeltaOneCompressedColumnNumberWithPrefixCounts() {}

template <class T>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::insert(
    const boost::any &new_value) {
  if (new_value.empty()) return false;
  if (typeid(T) == new_value.type()) {
    return insert(boost::any_cast<T>(new_value));
  }
  return false;
}

template <class T>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::insert(
    const T &new_value) {
  // row
  if (!values_.empty()) {
    if (values_.size() == 1) {
      // value must be exactly bigger by 1 than the value before
      if (new_value == (values_.back() + static_cast<T>(count_.back()))) {
        count_[count_.size() - 1] = count_.back() + 1;
        return true;
      }
    } else {
      // value must be exactly bigger by 1 than the value before
      if (new_value ==
          (values_.back() +
           static_cast<T>(count_.back() - count_[count_.size() - 2]))) {
        count_[count_.size() - 1] = count_.back() + 1;
        return true;
      }
    }
  }
  values_.push_back(new_value);
  // increment count value from last run or init first
  if (count_.empty()) {
    count_.push_back(1);
  } else {
    count_.push_back(count_.back() + 1);
  }
  return true;
}

template <>
inline bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<char *>::insert(
    char *const &new_value) {
  COGADB_FATAL_ERROR("Not implemented for other types than INT.", "");
}

template <>
inline bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<
    std::string>::insert(const std::string &new_value) {
  COGADB_FATAL_ERROR("Not implemented for other types than INT.", "");
}

template <class T>
const boost::any RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::get(
    TID tid) {
  return boost::any(operator[](tid));
}

template <class T>
void RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::print() const
    throw() {
  std::cout << "| " << this->name_ << " |" << std::endl;
  std::cout << "________________________" << std::endl;
  for (unsigned int i = 0; i < values_.size() && i < count_.size(); i++) {
    std::cout << "| " << values_[i] << " * " << (int)count_[i] << " | "
              << std::endl;
  }
}

template <class T>
size_t RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::size() const
    throw() {
  size_t size_sum = 0;
  for (size_t i = 0; i < count_.size(); i++) {
    size_sum += count_[i] + 1;
  }
  return size_sum;
}

template <class T>
const ColumnPtr RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::copy()
    const {
  return ColumnPtr(
      new RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>(*this));
}

template <class T>
const typename ColumnBaseTyped<T>::DenseValueColumnPtr
RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::copyIntoDenseValueColumn(
    const ProcessorSpecification &proc_spec) const {
  assert(proc_spec.proc_id == hype::PD0);

  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  DenseValueColumnPtr result(
      new DenseValueColumn(this->getName(), this->getType()));

  if (count_.size() == 0) {
    COGADB_FATAL_ERROR(
        "Count_ vector has no values. Column seems not to be in-memory.", "");
  }
  TID i = 0;
  result->resize(count_.back());
  T *col_array = result->data();

  // write first bucket serial
  T value = values_[0];
  for (; i < count_[0]; ++i) {
    col_array[i] = value++;
  }

  size_t bucket_number;
#pragma omp parallel for
  for (bucket_number = 1; bucket_number < count_.size(); ++bucket_number) {
    T value = values_[bucket_number];
    TID i = count_[bucket_number - 1];
    for (; i < count_[bucket_number]; ++i) {
      col_array[i] = value++;
    }
  }
  /*
          for (size_t bucket_number = 0; bucket_number < count_.size();
     ++bucket_number) {
              T value = values_[bucket_number];
              for ( ; i < count_[bucket_number]; ++i) {
                  result->insert(value++);
              }
          }
  */
  return result;
}

template <>
const typename ColumnBaseTyped<std::string>::DenseValueColumnPtr
RLEDeltaOneCompressedColumnNumberWithPrefixCounts<std::string>::
    copyIntoDenseValueColumn(const ProcessorSpecification &proc_spec) const {
  typedef typename ColumnBaseTyped<std::string>::DenseValueColumnPtr
      DenseValueColumnPtr;
  COGADB_FATAL_ERROR(
      "Called Unimplemented Method! For Type: " << typeid(std::string).name(),
      "");
  return DenseValueColumnPtr();
}

template <class T>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::update(
    TID tid, const boost::any &new_value) {
  if (new_value.empty()) {
    std::cout << "E: Update: new value is empty" << std::endl;
    return false;
  }
  unsigned int len = size();
  if (len <= tid) {
    return false;
  }
  if (typeid(T) == new_value.type()) {
    // cast value to T
    T new_T_value = boost::any_cast<T>(new_value);

    // decode value
    std::vector<T> decoded_values_;
    for (unsigned int i = 0; i < len; i++) {
      T value = operator[](i);
      decoded_values_.push_back(value);
    }

    // update decoded_values_
    decoded_values_[tid] = new_T_value;

    // update values_ -> clearContent + insert(decoded) -> encoded values_
    clearContent();
    for (unsigned int i = 0; i < decoded_values_.size(); i++) {
      T v = decoded_values_[i];
      insert(v);
    }

    return true;
  } else {
    std::cout << "Fatal Error!!! Typemismatch for column " << this->name_
              << std::endl;
  }
  return false;
}

template <class T>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::update(
    PositionListPtr tids, const boost::any &new_value) {
  if (!tids) return false;
  if (new_value.empty()) return false;
  TID tid;
  for (unsigned int i = 0; i < tids->size(); i++) {
    tid = (*tids)[i];
    update(tid, new_value);
  }
  return true;
}

template <class T>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::remove(TID tid) {
  unsigned int len = size();
  if (len <= tid) return false;

  // decode value
  std::vector<T> decoded_values_;
  for (unsigned int i = 0; i < len; i++) {
    T value = operator[](i);
    decoded_values_.push_back(value);
  }

  // erase value at tid
  decoded_values_.erase(decoded_values_.begin() + tid);

  // update values_ -> clearContent + insert(decoded) -> encoded values_
  clearContent();
  for (unsigned int i = 0; i < decoded_values_.size(); i++) {
    T v = decoded_values_[i];
    insert(v);
  }

  return true;
}

template <class T>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::remove(
    PositionListPtr tids) {
  if (!tids) return false;
  if (tids->empty()) return false;

  unsigned int loop_counter = tids->size();
  while (loop_counter > 0) {
    loop_counter--;
    remove((*tids)[loop_counter]);
  }

  return true;
}

template <class T>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::clearContent() {
  values_.clear();
  count_.clear();
  return true;
}

template <class T>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::store_impl(
    const std::string &path, boost::archive::binary_oarchive &oa) {
  oa << values_;
  oa << count_;
  return true;
}

template <class T>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::load_impl(
    const std::string &path, boost::archive::binary_iarchive &ia) {
  ia >> values_;
  ia >> count_;
  return true;
}

template <>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<char *>::store_impl(
    const std::string &path, boost::archive::binary_oarchive &oa) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  return false;
}

template <>
bool RLEDeltaOneCompressedColumnNumberWithPrefixCounts<char *>::load_impl(
    const std::string &path, boost::archive::binary_iarchive &ia) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  return false;
}

template <class T>
T &RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::operator[](
    const TID index) {
  if (count_.size() == 0) {
    COGADB_FATAL_ERROR(
        "Count_ vector has no values. Column seems not to be in-memory.", "");
  }

  if (index >= count_.back()) {
    COGADB_FATAL_ERROR("Column " << this->getName() << " TID out of bounds. "
                                 << "TID: " << index
                                 << " Max valid TID: " << count_.back(),
                       "");
  }
  TID i = 0;
  if (_last_lookup_index == 0 || index - 1 != _last_lookup_index) {
    i = binary_search_find_nearest_greater(count_.data(), count_.size(), index);
    _last_index_position = i;
    _last_lookup_index = index;
  } else {
    i = fast_sequential_lookup(index);
  }
  if (i == 0) {
    hack_last_uncompressed = values_[i] + index;
  } else {
    hack_last_uncompressed = values_[i] + index - count_[i - 1];
  }
  return hack_last_uncompressed;
}

template <class T>
TID &RLEDeltaOneCompressedColumnNumberWithPrefixCounts<
    T>::fast_sequential_lookup(const TID index) {
  assert(index - 1 == _last_lookup_index);
  if (index >= count_[_last_index_position]) {
    _last_index_position++;
  }
  _last_lookup_index = index;
  return _last_index_position;
}

template <>
inline std::string &RLEDeltaOneCompressedColumnNumberWithPrefixCounts<
    std::string>::operator[](const TID index) {
  COGADB_FATAL_ERROR("Not implemented for any type but int.", "");
}

template <class T>
size_t RLEDeltaOneCompressedColumnNumberWithPrefixCounts<T>::getSizeinBytes()
    const throw() {
  // std::vector<T> values_
  return values_.capacity() * sizeof(T)
         // std::vector<unsigned char> count_
         + count_.capacity() * sizeof(TID)
         // this->has_primary_key_constraint_, this->has_foreign_key_constraint_
         + 2 * sizeof(bool)
         // this->fk_constr_
         + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
         this->fk_constr_.getNameOfForeignKeyTable().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(
    RLEDeltaOneCompressedColumnNumberWithPrefixCounts)
}
