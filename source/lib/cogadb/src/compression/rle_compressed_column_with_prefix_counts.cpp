#include <backends/gpu/util.hpp>
#include <compression/rle_compressed_column_with_prefix_counts.hpp>

namespace CoGaDB {

/***************** Start of Implementation Section ******************/

template <class T>
RLECompressedColumnWithPrefixCounts<T>::RLECompressedColumnWithPrefixCounts(
    const std::string &name, AttributeType db_type)
    : CompressedColumn<T>(name, db_type, RUN_LENGTH_COMPRESSED_PREFIX),
      values_(),
      count_(),
      last_compressed_value(),
      _last_lookup_index(0),
      _last_index_position(0),
      _last_row_sum(0) {}

template <class T>
RLECompressedColumnWithPrefixCounts<T>::~RLECompressedColumnWithPrefixCounts() {
}

template <class T>
bool RLECompressedColumnWithPrefixCounts<T>::insert(
    const boost::any &new_value) {
  if (new_value.empty()) return false;
  if (typeid(T) == new_value.type()) {
    return insert(boost::any_cast<T>(new_value));
  }
  return false;
}

template <class T>
bool RLECompressedColumnWithPrefixCounts<T>::insert(const T &new_value) {
  // row
  if (!values_.empty() && new_value == values_.back()) {
    count_[count_.size() - 1] = count_.back() + 1;
  } else {
    values_.push_back(new_value);
    // increment count value from last run or init first
    if (count_.empty()) {
      count_.push_back(1);
    } else {
      count_.push_back(count_.back() + 1);
    }
  }

  return true;
}

template <class T>
const boost::any RLECompressedColumnWithPrefixCounts<T>::get(TID tid) {
  return boost::any(operator[](tid));
}

template <class T>
void RLECompressedColumnWithPrefixCounts<T>::print() const throw() {
  std::cout << "| " << this->name_ << " |" << std::endl;
  std::cout << "________________________" << std::endl;
  for (unsigned int i = 0; i < values_.size() && i < count_.size(); i++) {
    std::cout << "| " << values_[i] << " * " << (int)count_[i] << " | "
              << std::endl;
  }
}

template <class T>
size_t RLECompressedColumnWithPrefixCounts<T>::size() const throw() {
  if (count_.empty()) {
    return 0;
  }
  return count_.back();
}

template <class T>
const ColumnPtr RLECompressedColumnWithPrefixCounts<T>::copy() const {
  return ColumnPtr(new RLECompressedColumnWithPrefixCounts<T>(*this));
}

template <class T>
const typename ColumnBaseTyped<T>::DenseValueColumnPtr
RLECompressedColumnWithPrefixCounts<T>::copyIntoDenseValueColumn(
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
    col_array[i] = value;
  }

  size_t bucket_number;
#pragma omp parallel for
  for (bucket_number = 1; bucket_number < count_.size(); ++bucket_number) {
    T value = values_[bucket_number];
    TID i = count_[bucket_number - 1];
    for (; i < count_[bucket_number]; ++i) {
      col_array[i] = value;
    }
  }
  /*
          for (size_t bucket_number = 0; bucket_number < count_.size();
     ++bucket_number) {
              for ( ; i < count_[bucket_number]; ++i) {
                  result->insert(values_[bucket_number]);
              }
          }*/
  return result;
}

template <class T>
const ColumnPtr RLECompressedColumnWithPrefixCounts<T>::gather(
    PositionListPtr tid_list, const GatherParam &param) {
  assert(param.proc_spec.proc_id == hype::PD0);

  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr materialized_column =
      RLECompressedColumnWithPrefixCounts<T>::copyIntoDenseValueColumn(
          param.proc_spec);
  if (!materialized_column) return ColumnPtr();
  return materialized_column->gather(tid_list, param);
}

template <class T>
bool RLECompressedColumnWithPrefixCounts<T>::update(
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
bool RLECompressedColumnWithPrefixCounts<T>::update(
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
bool RLECompressedColumnWithPrefixCounts<T>::remove(TID tid) {
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
bool RLECompressedColumnWithPrefixCounts<T>::remove(PositionListPtr tids) {
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
void RLECompressedColumnWithPrefixCounts<
    T>::reset_sequential_lookup_variables() {
  _last_index_position = 0;
  _last_lookup_index = 0;
  _last_row_sum = 0;
}

template <class T>
bool RLECompressedColumnWithPrefixCounts<T>::clearContent() {
  values_.clear();
  count_.clear();
  reset_sequential_lookup_variables();
  return true;
}

template <class T>
const PositionListPtr RLECompressedColumnWithPrefixCounts<T>::selection(
    const SelectionParam &param) {
  PositionListPtr result_tids = createPositionList();

  assert(param.pred_type == ValueConstantPredicate);
  boost::any value_for_comparison = param.value;
  const ValueComparator comp = param.comp;

  if (value_for_comparison.type() != typeid(T)) {
    COGADB_FATAL_ERROR("Typemismatch for column!", "");
  }
  T value = boost::any_cast<T>(value_for_comparison);

  T *values_array = hype::util::begin_ptr(values_);

  TID current_pos = 0;
  TID number_of_matching_tids = 0;
  if (comp == EQUAL) {
    TID i = 0;
    // special handling for i = 0 as the number of values in the run is computed
    // differently
    number_of_matching_tids = (values_array[i] == value) * count_[i];
    for (size_t j = 0; j < number_of_matching_tids; ++j) {
      result_tids->push_back(current_pos + j);
    }
    current_pos = count_[i];
    i++;
    for (; i < values_.size(); ++i) {
      number_of_matching_tids =
          (values_array[i] == value) * (count_[i] - count_[i - 1]);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos = count_[i];
    }
  } else if (comp == LESSER) {
    TID i = 0;
    // special handling for i = 0 as the number of values in the run is computed
    // differently
    number_of_matching_tids = (values_array[i] < value) * count_[i];
    for (size_t j = 0; j < number_of_matching_tids; ++j) {
      result_tids->push_back(current_pos + j);
    }
    current_pos = count_[i];
    i++;
    for (; i < values_.size(); ++i) {
      number_of_matching_tids =
          (values_array[i] < value) * (count_[i] - count_[i - 1]);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos = count_[i];
    }
  } else if (comp == LESSER_EQUAL) {
    TID i = 0;
    // special handling for i = 0 as the number of values in the run is computed
    // differently
    number_of_matching_tids = (values_array[i] <= value) * count_[i];
    for (size_t j = 0; j < number_of_matching_tids; ++j) {
      result_tids->push_back(current_pos + j);
    }
    current_pos = count_[i];
    i++;
    for (; i < values_.size(); ++i) {
      number_of_matching_tids =
          (values_array[i] <= value) * (count_[i] - count_[i - 1]);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos = count_[i];
    }
  } else if (comp == GREATER) {
    TID i = 0;
    // special handling for i = 0 as the number of values in the run is computed
    // differently
    number_of_matching_tids = (values_array[i] > value) * count_[i];
    for (size_t j = 0; j < number_of_matching_tids; ++j) {
      result_tids->push_back(current_pos + j);
    }
    current_pos = count_[i];
    i++;
    for (; i < values_.size(); ++i) {
      number_of_matching_tids =
          (values_array[i] > value) * (count_[i] - count_[i - 1]);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos = count_[i];
    }
  } else if (comp == GREATER_EQUAL) {
    TID i = 0;
    // special handling for i = 0 as the number of values in the run is computed
    // differently
    number_of_matching_tids = (values_array[i] > value) * count_[i];
    for (size_t j = 0; j < number_of_matching_tids; ++j) {
      result_tids->push_back(current_pos + j);
    }
    current_pos = count_[i];
    i++;
    for (; i < values_.size(); ++i) {
      number_of_matching_tids =
          (values_array[i] > value) * (count_[i] - count_[i - 1]);
      for (size_t j = 0; j < number_of_matching_tids; ++j) {
        result_tids->push_back(current_pos + j);
      }
      current_pos = count_[i];
    }
  }

  return result_tids;
}

template <class T>
bool RLECompressedColumnWithPrefixCounts<T>::store_impl(
    const std::string &path, boost::archive::binary_oarchive &oa) {
  oa << values_;
  oa << count_;
  return true;
}

template <class T>
bool RLECompressedColumnWithPrefixCounts<T>::load_impl(
    const std::string &path, boost::archive::binary_iarchive &ia) {
  ia >> values_;
  ia >> count_;
  return true;
}

template <>
bool RLECompressedColumnWithPrefixCounts<char *>::store_impl(
    const std::string &path, boost::archive::binary_oarchive &oa) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  return false;
}

template <>
bool RLECompressedColumnWithPrefixCounts<char *>::load_impl(
    const std::string &path, boost::archive::binary_iarchive &ia) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
  return false;
}

template <class T>
T &RLECompressedColumnWithPrefixCounts<T>::operator[](const TID index) {
  if (count_.size() == 0) {
    COGADB_FATAL_ERROR(
        "Count_ vector has no values. Column seems not to be in-memory.", "");
  }

  if (index >= count_.back()) {
    COGADB_FATAL_ERROR("Index out of bounds.", "");
  }

  if (_last_lookup_index == 0 || index - 1 != _last_lookup_index) {
    TID i =
        binary_search_find_nearest_greater(count_.data(), count_.size(), index);
    _last_index_position = i;
    _last_lookup_index = index;
    return values_[i];
  } else {
    return fast_sequential_lookup(index);
  }
}

template <class T>
T &RLECompressedColumnWithPrefixCounts<T>::fast_sequential_lookup(
    const TID index) {
  assert(index - 1 == _last_lookup_index);
  TID i = _last_index_position;
  if (index >= count_[i]) {
    i++;
  }
  _last_index_position = i;
  _last_lookup_index = index;
  return values_[i];
}

template <class T>
size_t RLECompressedColumnWithPrefixCounts<T>::getSizeinBytes() const throw() {
  // std::vector<T> values_
  return values_.capacity() * sizeof(T)
         // std::vector<unsigned char> count_
         + count_.capacity() * sizeof(TID)
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
inline size_t RLECompressedColumnWithPrefixCounts<std::string>::getSizeinBytes()
    const throw() {
  size_t size_in_bytes = 0;
  // std::vector<T> values_
  for (unsigned int i = 0; i < values_.size(); ++i) {
    size_in_bytes += values_[i].capacity();
  }
  return size_in_bytes
         // std::vector<unsigned char> count_
         + count_.capacity() * sizeof(TID)
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

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(
    RLECompressedColumnWithPrefixCounts)
}
