#include <compression/rle_delta_one_compressed_column_int.hpp>

namespace CoGaDB {

/***************** Start of Implementation Section ******************/

template <class T>
RLEDeltaOneCompressedColumnNumber<T>::RLEDeltaOneCompressedColumnNumber(
    const std::string &name, AttributeType db_type)
    : CompressedColumn<T>(name, db_type,
                          RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER),
      values_(),
      count_(),
      hack_last_uncompressed() {
  // TODO Throw error if T not integer type (int, long, ...))
}

template <class T>
RLEDeltaOneCompressedColumnNumber<T>::~RLEDeltaOneCompressedColumnNumber() {}

template <class T>
bool RLEDeltaOneCompressedColumnNumber<T>::insert(const boost::any &new_value) {
  if (new_value.empty()) return false;
  if (typeid(T) == new_value.type()) {
    return insert(boost::any_cast<T>(new_value));
  }
  return false;
}

template <typename T>
bool RLEDeltaOneCompressedColumnNumber<T>::insert(const T &new_value) {
  // row
  if (!values_.empty() &&
      // value must be exactly bigger by 1 than the value before
      (new_value - 1) == (values_.back() + count_.back()) &&
      count_.back() < 255) {
    count_[count_.size() - 1] = count_.back() + 1;
  } else {
    values_.push_back(new_value);
    count_.push_back(0);
  }

  return true;
}

template <>
inline bool RLEDeltaOneCompressedColumnNumber<std::string>::insert(
    const std::string &new_value) {
  COGADB_FATAL_ERROR("Not implemented for other types than INT.", "");
  return false;
}

template <class T>
const boost::any RLEDeltaOneCompressedColumnNumber<T>::get(TID tid) {
  return boost::any(operator[](tid));
}

template <class T>
void RLEDeltaOneCompressedColumnNumber<T>::print() const throw() {
  std::cout << "| " << this->name_ << " |" << std::endl;
  std::cout << "________________________" << std::endl;
  for (unsigned int i = 0; i < values_.size() && i < count_.size(); i++) {
    std::cout << "| " << values_[i] << " * " << (int)count_[i] + 1 << " | "
              << std::endl;
  }
}

template <class T>
size_t RLEDeltaOneCompressedColumnNumber<T>::size() const throw() {
  int size_sum = 0;
  for (unsigned int i = 0; i < count_.size(); i++) {
    size_sum += count_[i] + 1;
  }
  return size_sum;
}

template <class T>
const ColumnPtr RLEDeltaOneCompressedColumnNumber<T>::copy() const {
  return ColumnPtr(new RLEDeltaOneCompressedColumnNumber<T>(*this));
}

template <class T>
const typename ColumnBaseTyped<T>::DenseValueColumnPtr
RLEDeltaOneCompressedColumnNumber<T>::copyIntoDenseValueColumn(
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
  for (size_t bucket_number = 0; bucket_number < count_.size();
       ++bucket_number) {
    T value = values_[bucket_number];
    for (size_t i = 0; i < count_[bucket_number] + 1u; ++i) {
      result->insert(value++);
    }
  }

  return result;
}

template <>
const typename ColumnBaseTyped<std::string>::DenseValueColumnPtr
RLEDeltaOneCompressedColumnNumber<std::string>::copyIntoDenseValueColumn(
    const ProcessorSpecification &proc_spec) const {
  typedef typename ColumnBaseTyped<std::string>::DenseValueColumnPtr
      DenseValueColumnPtr;
  COGADB_FATAL_ERROR(
      "Called Unimplemented Method! For Type: " << typeid(std::string).name(),
      "");
  return DenseValueColumnPtr();
}

template <class T>
bool RLEDeltaOneCompressedColumnNumber<T>::update(TID tid,
                                                  const boost::any &new_value) {
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
bool RLEDeltaOneCompressedColumnNumber<T>::update(PositionListPtr tids,
                                                  const boost::any &new_value) {
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
bool RLEDeltaOneCompressedColumnNumber<T>::remove(TID tid) {
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
bool RLEDeltaOneCompressedColumnNumber<T>::remove(PositionListPtr tids) {
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
bool RLEDeltaOneCompressedColumnNumber<T>::clearContent() {
  values_.clear();
  count_.clear();
  return true;
}

template <class T>
bool RLEDeltaOneCompressedColumnNumber<T>::store_impl(
    const std::string &path, boost::archive::binary_oarchive &oa) {
  oa << values_;
  oa << count_;
  return true;
}

template <class T>
bool RLEDeltaOneCompressedColumnNumber<T>::load_impl(
    const std::string &path, boost::archive::binary_iarchive &ia) {
  ia >> values_;
  ia >> count_;
  return true;
}

template <>
bool RLEDeltaOneCompressedColumnNumber<char *>::store_impl(
    const std::string &path, boost::archive::binary_oarchive &oa) {
  return false;
}

template <>
bool RLEDeltaOneCompressedColumnNumber<char *>::load_impl(
    const std::string &path, boost::archive::binary_iarchive &ia) {
  return false;
}

template <class T>
T &RLEDeltaOneCompressedColumnNumber<T>::operator[](const TID index) {
  uint32_t row_sum = 0;
  uint32_t i = 0;
  if (count_.size() == 0) {
    COGADB_FATAL_ERROR(
        "Count_ vector has no values. Column seems not to be in-memory.", "");
  }
  for (; row_sum + count_[i] < index; i++) {
    row_sum += ((count_[i]) + 1);
  }
  hack_last_uncompressed = values_[i] + index - row_sum;
  return hack_last_uncompressed;
}

template <>
inline std::string &RLEDeltaOneCompressedColumnNumber<std::string>::operator[](
    const TID index) {
  COGADB_FATAL_ERROR("Not implemented for any type but int.", "");
}

template <class T>
size_t RLEDeltaOneCompressedColumnNumber<T>::getSizeinBytes() const throw() {
  // std::vector<T> values_
  return values_.capacity() * sizeof(T)
         // std::vector<unsigned char> count_
         + count_.capacity() * sizeof(char)
         // this->has_primary_key_constraint_, this->has_foreign_key_constraint_
         + 2 * sizeof(bool)
         // this->fk_constr_
         + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
         this->fk_constr_.getNameOfForeignKeyTable().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(
    RLEDeltaOneCompressedColumnNumber)
}
