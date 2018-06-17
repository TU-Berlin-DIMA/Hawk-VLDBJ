#include <compression/void_compressed_column_int.hpp>
#include <util/types.hpp>

namespace CoGaDB {

/***************** Start of Implementation Section ******************/

template <class T>
VoidCompressedColumnNumber<T>::VoidCompressedColumnNumber(
    const std::string &name, AttributeType db_type)
    : CompressedColumn<T>(name, db_type, VOID_COMPRESSED_NUMBER),
      number_of_rows_(0) {
  // TODO Throw error if T not integer type (int, long, ...))
}

template <class T>
VoidCompressedColumnNumber<T>::~VoidCompressedColumnNumber() {}

template <class T>
bool VoidCompressedColumnNumber<T>::insert(const boost::any &new_value) {
  if (new_value.empty()) return false;
  if (typeid(T) == new_value.type()) {
    return insert(boost::any_cast<T>(new_value));
  }
  return false;
}

template <typename T>
bool VoidCompressedColumnNumber<T>::insert(const T &new_value) {
  COGADB_FATAL_ERROR("Not implemented for other types than INT.", "");
}

template <>
inline bool VoidCompressedColumnNumber<int>::insert(const int &new_value) {
  number_of_rows_++;
  return true;
}

template <>
inline bool VoidCompressedColumnNumber<TID>::insert(const TID &new_value) {
  number_of_rows_++;
  return true;
}

template <typename T>
template <typename InputIterator>
bool VoidCompressedColumnNumber<T>::insert(InputIterator first,
                                           InputIterator last) {
  for (InputIterator it = first; it != last; it++) {
    insert(*it);
  }
  return true;
}

template <class T>
const boost::any VoidCompressedColumnNumber<T>::get(TID tid) {
  return boost::any(operator[](tid));
}

template <class T>
void VoidCompressedColumnNumber<T>::print() const throw() {
  std::cout << "| " << this->name_ << " |" << std::endl;
  std::cout << "________________________" << std::endl;
  for (unsigned int i = 0; i < number_of_rows_; i++) {
    std::cout << "| " << i << " | " << std::endl;
  }
}

template <class T>
size_t VoidCompressedColumnNumber<T>::size() const throw() {
  return number_of_rows_;
}

template <class T>
const ColumnPtr VoidCompressedColumnNumber<T>::copy() const {
  return ColumnPtr(new VoidCompressedColumnNumber<T>(*this));
}

template <class T>
const typename ColumnBaseTyped<T>::DenseValueColumnPtr
VoidCompressedColumnNumber<T>::copyIntoDenseValueColumn(
    const ProcessorSpecification &proc_spec) const {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  return DenseValueColumnPtr();
}

template <>
const typename ColumnBaseTyped<int32_t>::DenseValueColumnPtr
VoidCompressedColumnNumber<int32_t>::copyIntoDenseValueColumn(
    const ProcessorSpecification &proc_spec) const {
  assert(proc_spec.proc_id == hype::PD0);

  typedef typename ColumnBaseTyped<int32_t>::DenseValueColumnPtr
      DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<int32_t>::DenseValueColumn DenseValueColumn;
  DenseValueColumnPtr result(
      new DenseValueColumn(this->getName(), this->getType()));
  size_t num_elements = this->size();
  result->resize(num_elements);
  int32_t *col_array = result->data();

#pragma omp parallel for
  for (size_t i = 0; i < num_elements; ++i) {
    col_array[i] = (int32_t)i;
  }
  return result;
}

template <>
const typename ColumnBaseTyped<TID>::DenseValueColumnPtr
VoidCompressedColumnNumber<TID>::copyIntoDenseValueColumn(
    const ProcessorSpecification &proc_spec) const {
  assert(proc_spec.proc_id == hype::PD0);

  typedef
      typename ColumnBaseTyped<TID>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<TID>::DenseValueColumn DenseValueColumn;
  DenseValueColumnPtr result(
      new DenseValueColumn(this->getName(), this->getType()));
  size_t num_elements = this->size();
  result->resize(num_elements);
  TID *col_array = result->data();

#pragma omp parallel for
  for (size_t i = 0; i < num_elements; ++i) {
    col_array[i] = (TID)i;
  }
  return result;
}

//    const ColumnPtr VoidCompressedColumnNumber<T>::gather(PositionListPtr
//    tid_list, const GatherParam&){
//
//    }

template <class T>
const PositionListPtr VoidCompressedColumnNumber<T>::selection(
    const SelectionParam &param) {
  COGADB_FATAL_ERROR("Not implemented for any type but int.", "");
}

template <>
const PositionListPtr VoidCompressedColumnNumber<TID>::selection(
    const SelectionParam &param) {
  PositionListPtr result_tids = createPositionList();

  assert(param.pred_type == ValueConstantPredicate);

  TID value;
  bool ret_success =
      getValueFromAny(this->name_, param.value, value, this->db_type_);
  if (!ret_success) PositionListPtr();

  const ValueComparator comp = param.comp;

  if (comp == EQUAL) {
    result_tids->push_back(value);
  } else if (comp == LESSER) {
    for (TID i = 0; i < value; ++i) {
      result_tids->push_back(i);
    }
  } else if (comp == LESSER_EQUAL) {
    for (TID i = 0; i <= value; ++i) {
      result_tids->push_back(i);
    }
  } else if (comp == GREATER) {
    for (TID i = (value + 1); i < size(); ++i) {
      result_tids->push_back(i);
    }
  } else if (comp == GREATER_EQUAL) {
    for (TID i = value; i < size(); ++i) {
      result_tids->push_back(i);
    }
  }
  return result_tids;
}

template <class T>
bool VoidCompressedColumnNumber<T>::update(TID tid,
                                           const boost::any &new_value) {
  return false;
}

template <class T>
bool VoidCompressedColumnNumber<T>::update(PositionListPtr tids,
                                           const boost::any &new_value) {
  return false;
}

template <class T>
bool VoidCompressedColumnNumber<T>::remove(TID tid) {
  return false;
}

template <class T>
bool VoidCompressedColumnNumber<T>::remove(PositionListPtr tids) {
  if (!tids) return false;
  if (tids->empty()) return false;
  return false;
}

template <class T>
bool VoidCompressedColumnNumber<T>::clearContent() {
  number_of_rows_ = 0;
  return true;
}

template <class T>
bool VoidCompressedColumnNumber<T>::store_impl(
    const std::string &path, boost::archive::binary_oarchive &oa) {
  oa << number_of_rows_;
  return true;
}

template <class T>
bool VoidCompressedColumnNumber<T>::load_impl(
    const std::string &path, boost::archive::binary_iarchive &ia) {
  ia >> number_of_rows_;
  return true;
}

template <class T>
T &VoidCompressedColumnNumber<T>::operator[](const TID index) {
  COGADB_FATAL_ERROR("Not implemented for any type but int.", "");
  // return NULL;
}

template <>
inline int &VoidCompressedColumnNumber<int>::operator[](const TID index) {
  hack_last_returned_ = index;
  return hack_last_returned_;
}

template <>
inline TID &VoidCompressedColumnNumber<TID>::operator[](const TID index) {
  hack_last_returned_ = index;
  return hack_last_returned_;
}

template <class T>
size_t VoidCompressedColumnNumber<T>::getSizeinBytes() const throw() {
  // std::vector<T> values_
  return sizeof(size_t)
         // this->fk_constr_
         + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
         this->fk_constr_.getNameOfForeignKeyTable().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(
    VoidCompressedColumnNumber)
}
