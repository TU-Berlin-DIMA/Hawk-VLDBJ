#include <omp.h>
#include <compression/rle_compressed_column.hpp>
#include <util/types.hpp>

#pragma GCC diagnostic ignored "-Wunused-value"

namespace CoGaDB {

/***************** Start of Implementation Section ******************/

template <class T>
RLECompressedColumn<T>::RLECompressedColumn(const std::string &name,
                                            AttributeType db_type)
    : CompressedColumn<T>(name, db_type, RUN_LENGTH_COMPRESSED),
      values_(),
      count_(),
      last_compressed_value(),
      _last_lookup_index(0),
      _last_index_position(0),
      _last_row_sum(0) {}

template <class T>
RLECompressedColumn<T>::~RLECompressedColumn() {}

template <class T>
bool RLECompressedColumn<T>::insert(const boost::any &new_value) {
  if (new_value.empty()) return false;
  if (typeid(T) == new_value.type()) {
    return insert(boost::any_cast<T>(new_value));
  }
  return false;
}

template <class T>
bool RLECompressedColumn<T>::insert(const T &new_value) {
  // row
  if (!values_.empty() && new_value == values_.back() && count_.back() < 255) {
    count_[count_.size() - 1] = count_.back() + 1;
  } else {
    values_.push_back(new_value);
    count_.push_back(0);
  }

  return true;
}

inline std::string rle_encode_string(const std::string s) {
  int count = 0;
  std::stringstream result_str;
  char found = '\0';
  for (unsigned int i = 0; i < s.length(); i++) {
    const char c = s[i];

    //		for(const char c : s) {
    if (i == 0) {  // first char
      found = c;
    } else if (found == c) {  // following char
      if (count >= 255) {     // if count is bigger than max char write new char
        result_str << (char)255 << (char)found;
        count = count - 255;
      } else
        count++;
    } else {  // new char, wirte last found, reset count
      result_str << (char)count << (char)found;
      found = c;
      count = 0;
    }
  }
  result_str << (char)count << (char)found;  // write last char
  return result_str.str();
}

template <>
inline bool RLECompressedColumn<std::string>::insert(
    const std::string &new_value) {
  std::string result = rle_encode_string(new_value);
  if ((!values_.empty()) && (result == values_.back()) &&
      (count_.back() < 255)) {
    count_[count_.size() - 1] = count_.back() + 1;
  } else {
    values_.push_back(result);
    count_.push_back(0);
  }
  return true;
}

template <class T>
const boost::any RLECompressedColumn<T>::get(TID tid) {
  return boost::any(operator[](tid));
}

template <class T>
void RLECompressedColumn<T>::print() const throw() {
  std::cout << "| " << this->name_ << " |" << std::endl;
  std::cout << "________________________" << std::endl;
  for (unsigned int i = 0; i < values_.size() && i < count_.size(); i++) {
    std::cout << "| " << values_[i] << " * " << (int)count_[i] + 1 << " | "
              << std::endl;
  }
}

template <class T>
size_t RLECompressedColumn<T>::size() const throw() {
  int size_sum = 0;
  for (unsigned int i = 0; i < count_.size(); i++) {
    size_sum += count_[i] + 1;
  }
  return size_sum;
}

template <class T>
const ColumnPtr RLECompressedColumn<T>::copy() const {
  return ColumnPtr(new RLECompressedColumn<T>(*this));
}

template <class T>
const typename ColumnBaseTyped<T>::DenseValueColumnPtr
RLECompressedColumn<T>::copyIntoDenseValueColumn(
    const ProcessorSpecification &proc_spec) const {
  assert(proc_spec.proc_id == hype::PD0);

  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  // DenseValueColumnPtr result(new DenseValueColumn(this->getName(),
  // this->getType()));

  if (count_.size() == 0) {
    COGADB_FATAL_ERROR(
        "Count_ vector has no values. Column seems not to be in-memory.", "");
  }
  //        for (size_t bucket_number = 0; bucket_number < count_.size();
  //        ++bucket_number) {
  //            for (size_t i = 0; i < count_[bucket_number] + 1; ++i) {
  //                result->insert(values_[bucket_number]);
  //            }
  //        }

  std::vector<DenseValueColumnPtr> partial_results;
  const size_t number_of_threads = omp_get_max_threads();
  for (size_t i = 0; i < number_of_threads; ++i) {
    partial_results.push_back(DenseValueColumnPtr(
        new DenseValueColumn(this->getName(), this->getType())));
  }

  size_t bucket_number;
#pragma omp parallel for schedule(static)
  for (bucket_number = 0; bucket_number < count_.size(); ++bucket_number) {
    size_t thread_id = omp_get_thread_num();
    for (size_t i = 0; i < count_[bucket_number] + 1u; ++i) {
      partial_results[thread_id]->insert(values_[bucket_number]);
    }
  }

  size_t total_result_size = 0;
  std::vector<size_t> write_indexes(number_of_threads);
  std::vector<size_t> thread_result_sizes(number_of_threads);
  for (size_t i = 0; i < number_of_threads; i++) {
    thread_result_sizes[i] = partial_results[i]->size();
    // compute exclusive prefix sum!
    write_indexes[i] = total_result_size;
    total_result_size += thread_result_sizes[i];
  }

  DenseValueColumnPtr result(
      new DenseValueColumn(this->getName(), this->getType()));
  // TODO size can be determined faster by just looking up the very last count_
  // TID value
  // result->reserve(this->size());
  result->resize(total_result_size);
  T *col_array = result->data();

// copy partial tids into final array
#pragma omp parallel for
  for (size_t i = 0; i < number_of_threads; i++) {
    if (thread_result_sizes[i] > 0) {
      size_t j, k;
      for (k = 0, j = write_indexes[i]; k < thread_result_sizes[i],
          j < write_indexes[i] + thread_result_sizes[i];
           ++j, ++k) {
        col_array[j] = (*partial_results[i])[k];
      }
    }
  }
  return result;
}

template <class T>
const ColumnPtr RLECompressedColumn<T>::gather(PositionListPtr tid_list,
                                               const GatherParam &param) {
  assert(param.proc_spec.proc_id == hype::PD0);

  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr materialized_column =
      RLECompressedColumn<T>::copyIntoDenseValueColumn(param.proc_spec);
  if (!materialized_column) return ColumnPtr();
  return materialized_column->gather(tid_list, param);
}

template <class T>
bool RLECompressedColumn<T>::update(TID tid, const boost::any &new_value) {
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
bool RLECompressedColumn<T>::update(PositionListPtr tids,
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
bool RLECompressedColumn<T>::remove(TID tid) {
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
bool RLECompressedColumn<T>::remove(PositionListPtr tids) {
  if (!tids) return false;
  if (tids->empty()) return false;

  //		typename PositionList::reverse_iterator rit;
  //
  //		for (rit = tids->rbegin(); rit!=tids->rend(); ++rit)
  //		{
  //			remove(*rit);
  //		}

  unsigned int loop_counter = tids->size();
  while (loop_counter > 0) {
    loop_counter--;
    remove((*tids)[loop_counter]);
  }

  return true;
}

template <class T>
void RLECompressedColumn<T>::reset_sequential_lookup_variables() {
  _last_index_position = 0;
  _last_lookup_index = 0;
  _last_row_sum = 0;
}

template <class T>
bool RLECompressedColumn<T>::clearContent() {
  values_.clear();
  count_.clear();
  reset_sequential_lookup_variables();
  return true;
}

template <class T>
const PositionListPtr RLECompressedColumn<T>::selection(
    const SelectionParam &param) {
  PositionListPtr result_tids = createPositionList();

  assert(param.pred_type == ValueConstantPredicate);

  T value = T();
  bool ret_success =
      getValueFromAny(this->name_, param.value, value, this->db_type_);
  if (!ret_success) PositionListPtr();

  const ValueComparator comp = param.comp;

  T *values_array = hype::util::begin_ptr(values_);

  size_t current_pos = 0;
  size_t number_of_matching_tids = 0;
  if (comp == EQUAL) {
    for (size_t i = 0; i < values_.size(); ++i) {
      number_of_matching_tids = (values_array[i] == value) * (count_[i] + 1);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos += (count_[i] + 1);
    }

  } else if (comp == LESSER) {
    for (size_t i = 0; i < values_.size(); ++i) {
      number_of_matching_tids = (values_array[i] < value) * (count_[i] + 1);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos += (count_[i] + 1);
    }
  } else if (comp == LESSER_EQUAL) {
    for (size_t i = 0; i < values_.size(); ++i) {
      number_of_matching_tids = (values_array[i] <= value) * (count_[i] + 1);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos += (count_[i] + 1);
    }
  } else if (comp == GREATER) {
    for (size_t i = 0; i < values_.size(); ++i) {
      number_of_matching_tids = (values_array[i] > value) * (count_[i] + 1);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos += (count_[i] + 1);
    }
  } else if (comp == GREATER_EQUAL) {
    for (size_t i = 0; i < values_.size(); ++i) {
      number_of_matching_tids = (values_array[i] >= value) * (count_[i] + 1);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos += (count_[i] + 1);
    }
  }

  return result_tids;
}

template <class T>
bool RLECompressedColumn<T>::store_impl(const std::string &path,
                                        boost::archive::binary_oarchive &oa) {
  oa << values_;
  oa << count_;
  return true;
}

template <class T>
bool RLECompressedColumn<T>::load_impl(const std::string &path,
                                       boost::archive::binary_iarchive &ia) {
  ia >> values_;
  ia >> count_;
  return true;
}

template <>
bool RLECompressedColumn<char *>::store_impl(
    const std::string &path, boost::archive::binary_oarchive &oa) {
  return false;
}

template <>
bool RLECompressedColumn<char *>::load_impl(
    const std::string &path, boost::archive::binary_iarchive &ia) {
  return false;
}

template <class T>
T &RLECompressedColumn<T>::operator[](const TID index) {
  if (_last_lookup_index == 0 || index - 1 != _last_lookup_index) {
    TID row_sum = 0;
    TID i = 0;
    if (count_.size() == 0) {
      COGADB_FATAL_ERROR(
          "Count_ vector has no values. Column seems not to be in-memory.", "");
    }
    for (; row_sum + count_[i] < index; i++) {
      row_sum += ((count_[i]) + 1);
    }
    _last_row_sum = row_sum;
    _last_index_position = i;
    _last_lookup_index = index;
    return values_[i];
  } else {
    return fast_sequential_lookup(index);
  }
}

template <class T>
T &RLECompressedColumn<T>::fast_sequential_lookup(const TID index) {
  TID row_sum = _last_row_sum;
  TID i = _last_index_position;
  for (; row_sum + count_[i] < index; i++) {
    row_sum += ((count_[i]) + 1);
  }
  _last_row_sum = row_sum;
  _last_index_position = i;
  _last_lookup_index = index;
  return values_[i];
}

template <>
inline std::string &RLECompressedColumn<std::string>::operator[](
    const TID index) {
  // find value
  unsigned int row_sum = 0;
  unsigned int i = 0;
  if (count_.size() == 0) {
    COGADB_FATAL_ERROR(
        "Count_ vector has no values. Column seems not to be in-memory.", "");
  }
  for (; row_sum + count_[i] < index; i++) {
    row_sum += ((count_[i]) + 1);
  }
  std::string encoded = values_[i];

  // decode string
  std::string decoded;
  for (unsigned int j = 0; j < encoded.size(); j += 2) {
    unsigned char c = encoded[j];
    for (unsigned int k = 0; k <= c; k++) {
      decoded.push_back(encoded[j + 1]);
    }
  }

  last_compressed_value = decoded;
  return last_compressed_value;
}

template <class T>
size_t RLECompressedColumn<T>::getSizeinBytes() const throw() {
  // std::vector<T> values_
  return values_.capacity() * sizeof(T)
         // std::vector<unsigned char> count_
         + count_.capacity() * sizeof(char)
         // std::string last_compressed_value
         + last_compressed_value.capacity()
         // this->has_primary_key_constraint_, this->has_foreign_key_constraint_
         + 2 * sizeof(bool)
         // this->fk_constr_
         + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
         this->fk_constr_.getNameOfForeignKeyTable().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
}

template <>
inline size_t RLECompressedColumn<std::string>::getSizeinBytes() const throw() {
  size_t size_in_bytes = 0;
  // std::vector<T> values_
  for (unsigned int i = 0; i < values_.size(); ++i) {
    size_in_bytes += values_[i].capacity();
  }
  return size_in_bytes
         // std::vector<unsigned char> count_
         + count_.capacity() * sizeof(char)
         // std::string last_compressed_value
         + last_compressed_value.capacity()
         // this->has_primary_key_constraint_, this->has_foreign_key_constraint_
         + 2 * sizeof(bool)
         // this->fk_constr_
         + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
         this->fk_constr_.getNameOfForeignKeyTable().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(RLECompressedColumn)
}
