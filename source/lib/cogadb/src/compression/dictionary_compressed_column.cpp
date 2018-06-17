
#include <compression/dictionary_compressed_column.hpp>
#include <limits>
#include <util/dictionary_compression.hpp>
#include <util/regular_expression.hpp>

namespace CoGaDB {

template <class T>
const typename DictionaryCompressedColumn<T>::DictionaryPtr&
DictionaryCompressedColumn<T>::getDictionary() const throw() {
  return dictionary_;
}

template <class T>
uint32_t* DictionaryCompressedColumn<T>::getIdData() {
  return ids_->data();
}

template <class T>
const T* const DictionaryCompressedColumn<T>::getReverseLookupVector() const {
  if (reverse_lookup_vector_->empty()) {
    return NULL;
  } else {
    return reverse_lookup_vector_->data();
  }
}

template <class T>
const typename DictionaryCompressedColumn<T>::ReverseLookupVectorPtr
DictionaryCompressedColumn<T>::getReverseLookupVectorPtr() const {
  if (reverse_lookup_vector_->empty()) {
    return NULL;
  } else {
    return reverse_lookup_vector_;
  }
}

template <class T>
T& DictionaryCompressedColumn<T>::reverseLookup(uint32_t id) {
  return (*reverse_lookup_vector_)[id];
}

template <class T>
std::pair<bool, uint32_t> DictionaryCompressedColumn<T>::getDictionaryID(
    const T& value) const {
  //                    std::string filter_val =
  //                    boost::any_cast<std::string>(comp_val);
  //                    const typename
  //                    DictionaryCompressedColumn<std::string>::Dictionary&
  //                    dictionary = host_col->getDictionary();
  //                    typename
  //                    DictionaryCompressedColumn<std::string>::Dictionary::const_iterator
  //                    it;
  typename Dictionary::const_iterator it;
  uint32_t id = 0;
  it = dictionary_->find(value);
  // translate the string value for comparison to internal integer id, which
  // represen the strings in the column
  if (it != dictionary_->end()) {
    id = it->second;
    return std::make_pair(true, id);
  } else {
    // return pair with flag false, for not found and
    // an id that is not included in the dictionary,
    // so fitlering using this id will result in empty result
    return std::make_pair(false, maximal_id_ + 1);
  }
}

template <class T>
unsigned int DictionaryCompressedColumn<T>::getNumberofDistinctValues() const {
  return this->dictionary_->size();
}

template <class T>
uint32_t DictionaryCompressedColumn<T>::getLargestID() const {
  return this->maximal_id_;
}

template <class T>
size_t DictionaryCompressedColumn<T>::getNumberOfRows() const {
  return this->size();
}

/***************** Start of Implementation Section ******************/

template <class T>
DictionaryCompressedColumn<T>::DictionaryCompressedColumn(
    const std::string& name, AttributeType db_type,
    const hype::ProcessingDeviceMemoryID& mem_id)
    : CompressedColumn<T>(name, db_type, DICTIONARY_COMPRESSED),
      ids_(new IDColumn(name + "_ENCODING_IDS", UINT32, mem_id)),
      dictionary_(boost::make_shared<Dictionary>()),
      reverse_lookup_vector_(boost::make_shared<ReverseLookupVector>()),
      maximal_id_(0) {}

template <class T>
DictionaryCompressedColumn<T>::DictionaryCompressedColumn(
    const std::string& name, AttributeType db_type, const IDColumnPtr& id_col,
    const ReverseLookupVectorPtr& reverse_lookup, const DictionaryPtr& dict)
    : CompressedColumn<T>(name, db_type, DICTIONARY_COMPRESSED),
      ids_(id_col),
      dictionary_(dict),
      reverse_lookup_vector_(reverse_lookup),
      maximal_id_(0) {
  if (id_col->size() > 0) {
    maximal_id_ = *std::max_element(id_col->begin(), id_col->end());
  }
  id_col->setName(id_col->getName() + "_ENCODING_IDS");
}

template <class T>
DictionaryCompressedColumn<T>::DictionaryCompressedColumn(
    const DictionaryCompressedColumn<T>& other)
    : CompressedColumn<T>(other.getName(), other.getType(),
                          other.getColumnType()),
      ids_(boost::dynamic_pointer_cast<IDColumn>(other.ids_->copy())),
      dictionary_(other.dictionary_),
      reverse_lookup_vector_(other.reverse_lookup_vector_),
      maximal_id_(other.maximal_id_) {
  assert(ids_ != NULL);
}

template <class T>
DictionaryCompressedColumn<T>& DictionaryCompressedColumn<T>::operator=(
    const DictionaryCompressedColumn<T>& other) {
  if (this != &other)  // protect against invalid self-assignment
  {
    this->name_ = other.name_;
    this->db_type_ = other.db_type_;
    this->column_type_ = other.column_type_;
    this->ids_ = boost::dynamic_pointer_cast<IDColumn>(other.ids_->copy());
    assert(this->ids_ != NULL);
    this->dictionary_ = other.dictionary_;
    this->reverse_lookup_vector_ = other.reverse_lookup_vector_;
    this->maximal_id_ = other.maximal_id_;
  }
  return *this;
}

template <class T>
DictionaryCompressedColumn<T>::~DictionaryCompressedColumn() {}

template <class T>
bool DictionaryCompressedColumn<T>::insert(const boost::any& new_Value) {
  T value = boost::any_cast<T>(new_Value);

  return this->insert(value);
}

template <class T>
bool DictionaryCompressedColumn<T>::insert(const T& value) {
  typename Dictionary::iterator it = dictionary_->find(value);
  if (it != dictionary_->end()) {
    ids_->push_back(it->second);
  } else {
    ids_->push_back(maximal_id_);
    dictionary_->insert(std::make_pair(value, maximal_id_));
    // element id is position in reverse lookup vector to get the real value in
    // O(1) time
    reverse_lookup_vector_->push_back(value);
    maximal_id_++;
  }

  return true;
}

template <class T>
const boost::any DictionaryCompressedColumn<T>::get(TID tid) {
  return boost::any(this->operator[](tid));
}

template <class T>
void DictionaryCompressedColumn<T>::print() const throw() {
  std::cout << "| " << this->name_ << " (Dictionary Compressed) |" << std::endl;
  std::cout << "________________________" << std::endl;
  for (unsigned int i = 0; i < this->size(); i++) {
    std::cout << "| " << (*reverse_lookup_vector_)[(*ids_)[i]] << " |"
              << std::endl;
  }
}

template <class T>
size_t DictionaryCompressedColumn<T>::size() const throw() {
  return ids_->size();
}

template <class T>
const ColumnPtr DictionaryCompressedColumn<T>::copy() const {
  return ColumnPtr(new DictionaryCompressedColumn<T>(*this));
}

template <class T>
const ColumnPtr DictionaryCompressedColumn<T>::copy(
    const hype::ProcessingDeviceMemoryID& mem_id) const {
  if (this->getMemoryID() == mem_id) return this->copy();

  ColumnPtr new_id_col = this->ids_->copy(mem_id);
  if (!new_id_col) return ColumnPtr();

  IDColumnPtr new_typed_id_col =
      boost::dynamic_pointer_cast<IDColumn>(new_id_col);

  if (!new_typed_id_col) return ColumnPtr();

  DictionaryCompressedColumn<T>* new_col = new DictionaryCompressedColumn<T>(
      this->getName(), this->getType(), mem_id);
  new_col->dictionary_ = this->dictionary_;
  new_col->reverse_lookup_vector_ = this->reverse_lookup_vector_;
  new_col->maximal_id_ = this->maximal_id_;
  new_col->ids_ = new_typed_id_col;

  return ColumnPtr(new_col);
}

template <class T>
const typename ColumnBaseTyped<T>::DenseValueColumnPtr
DictionaryCompressedColumn<T>::copyIntoDenseValueColumn(
    const ProcessorSpecification& proc_spec) const {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  //        assert(proc_spec.proc_id == hype::PD0);
  if (proc_spec.proc_id != hype::PD0) {
    COGADB_FATAL_ERROR(
        "Function copyIntoDenseValueColumn() can only be called on CPUs!", "");
  }
  DenseValueColumnPtr result(
      new DenseValueColumn(this->getName(), this->getType()));
  size_t num_elements = this->size();
  DictionaryCompressedColumn<T>* this_col =
      const_cast<DictionaryCompressedColumn<T>*>(this);
  for (size_t i = 0; i < num_elements; ++i) {
    result->insert((*this_col)[i]);
  }
  return result;
}

template <class T>
const ColumnPtr DictionaryCompressedColumn<T>::gather(
    PositionListPtr tid_list, const GatherParam& param) {
  DictionaryCompressedColumn<T>* result =
      new DictionaryCompressedColumn<T>(this->name_, this->db_type_);

  //        PositionListPtr copied_tids = copy_if_required(tid_list,
  //        hype::PD_Memory_0);//this->mem_alloc->getMemoryID());
  //        if(!copied_tids) return ColumnPtr();

  ColumnPtr result_ids = this->ids_->gather(tid_list, param);
  if (!result_ids) return ColumnPtr();
  IDColumnPtr typed_result_ids =
      boost::dynamic_pointer_cast<IDColumn>(result_ids);
  assert(typed_result_ids != NULL);

  result->ids_ = typed_result_ids;
  result->dictionary_ = dictionary_;
  result->maximal_id_ = maximal_id_;
  result->reverse_lookup_vector_ = reverse_lookup_vector_;
  result->fk_constr_ = this->fk_constr_;

  return ColumnPtr(result);
}

template <class T>
const ColumnGroupingKeysPtr
DictionaryCompressedColumn<T>::createColumnGroupingKeys(
    const ProcessorSpecification& proc_spec) const {
  return this->ids_->createColumnGroupingKeys(proc_spec);
}

template <class T>
size_t DictionaryCompressedColumn<T>::getNumberOfRequiredBits() const {
  return this->ids_->getNumberOfRequiredBits();
}

template <class T>
bool DictionaryCompressedColumn<T>::update(TID index,
                                           const boost::any& new_Value) {
  T value = boost::any_cast<T>(new_Value);
  if (index >= this->size()) return false;

  typename Dictionary::iterator it = dictionary_->find(value);
  if (it != dictionary_->end()) {
    // ids_->push_back(it->second);
    (*ids_)[index] = it->second;
    // reverse_lookup_vector_[ids[index]]=it->first;
  } else {
    // ids_->push_back(maximal_id_);
    (*ids_)[index] = maximal_id_;
    dictionary_->insert(std::make_pair(value, maximal_id_));
    // element id is position in reverse lookup vector to get the real value in
    // O(1) time
    reverse_lookup_vector_->push_back(value);
    maximal_id_++;
  }
  return true;
}

template <class T>
bool DictionaryCompressedColumn<T>::update(PositionListPtr tids,
                                           const boost::any& new_Value) {
  if (!tids) return false;
  // test whether tid list has at least one element, if not, return with error
  if (tids->empty()) return false;

  bool result = true;
  for (unsigned int i = 0; i < tids->size(); i++) {
    result = result && this->update((*tids)[i], new_Value);
  }
  return result;
}

template <class T>
bool DictionaryCompressedColumn<T>::remove(TID tid) {
  ids_->remove(tid);  // ids_->begin() + tid);
  return true;
}

template <class T>
bool DictionaryCompressedColumn<T>::remove(PositionListPtr tids) {
  return ids_->remove(tids);
}

template <class T>
bool DictionaryCompressedColumn<T>::clearContent() {
  this->ids_->clear();
  this->dictionary_->clear();
  this->reverse_lookup_vector_->clear();
  this->maximal_id_ = 0;
  return true;
}

template <class T>
bool DictionaryCompressedColumn<T>::store_impl(
    const std::string& path_to_table_dir, boost::archive::binary_oarchive& oa) {
  oa << *dictionary_;
  oa << *reverse_lookup_vector_;
  oa << this->maximal_id_;
  return ids_->store(path_to_table_dir);
}

template <>
bool DictionaryCompressedColumn<char*>::store_impl(
    const std::string& path_to_table_dir, boost::archive::binary_oarchive& oa) {
  COGADB_FATAL_ERROR("", "");
  return false;
}

template <class T>
const ColumnPtr DictionaryCompressedColumn<T>::getDecompressedColumn_impl(
    const ProcessorSpecification& proc_spec) {
  return this->copyIntoDenseValueColumn(proc_spec);
}

template <>
const ColumnPtr
DictionaryCompressedColumn<std::string>::getDecompressedColumn_impl(
    const ProcessorSpecification& proc_spec) {
  boost::shared_ptr<Column<char*> > col = createPointerArrayToValues(
      ids_->data(), ids_->size(), this->reverse_lookup_vector_->data());
  return col;
}

template <class T>
bool DictionaryCompressedColumn<T>::load_impl(
    const std::string& path_to_table_dir, boost::archive::binary_iarchive& ia) {
  ia >> *dictionary_;
  ia >> *reverse_lookup_vector_;
  ia >> maximal_id_;
  return ids_->load(path_to_table_dir, LOAD_ALL_DATA);
}

template <>
bool DictionaryCompressedColumn<char*>::load_impl(
    const std::string& path_to_table_dir, boost::archive::binary_iarchive& ia) {
  COGADB_FATAL_ERROR("", "");
  return false;
}

template <class T>
bool DictionaryCompressedColumn<T>::isMaterialized() const throw() {
  return false;
}

template <class T>
const ColumnPtr DictionaryCompressedColumn<T>::materialize() throw() {
  return this->copy();
}

template <class T>
hype::ProcessingDeviceMemoryID DictionaryCompressedColumn<T>::getMemoryID()
    const {
  // column is dormant in memory where ever our ids_ column is
  return ids_->getMemoryID();
}

template <class T>
const PositionListPtr DictionaryCompressedColumn<T>::selection(
    const SelectionParam& param) {
  if (param.pred_type == ValueConstantPredicate &&
      (param.comp == EQUAL || param.comp == UNEQUAL)) {
    T value = boost::any_cast<T>(param.value);
    typename Dictionary::iterator it = this->dictionary_->find(value);
    if (it == dictionary_->end()) {
      // result is empty, because value does not exist
      return createPositionList(0);
    } else {
      SelectionParam new_param(param);
      new_param.value = boost::any(it->second);
      return ids_->selection(new_param);
    }
  } else if (param.pred_type == ValueRegularExpressionPredicate) {
    assert(param.comp == EQUAL || param.comp == UNEQUAL);
    ColumnPtr matching_ids = getMatchingIDsFromDictionary<T>(
        boost::any_cast<std::string>(param.value), *dictionary_);
    JoinType join_type = RIGHT_SEMI_JOIN;
    if (param.comp == UNEQUAL) {
      join_type = RIGHT_ANTI_SEMI_JOIN;
    }

    JoinParam join_param(param.proc_spec, HASH_JOIN, join_type);
    return matching_ids->tid_semi_join(this->ids_, join_param);
  } else {
    return ColumnBaseTyped<T>::selection(param);
  }
}

template <class T>
const BitmapPtr DictionaryCompressedColumn<T>::bitmap_selection(
    const SelectionParam& param) {
  if (param.pred_type == ValueConstantPredicate && param.comp == EQUAL) {
    T value = boost::any_cast<T>(param.value);
    typename Dictionary::iterator it = dictionary_->find(value);
    if (it == dictionary_->end()) {
      // result is empty, because value does not exist
      return BitmapPtr(new Bitmap(this->size(), false, true));
    } else {
      SelectionParam new_param(param);
      new_param.value = boost::any(it->second);
      return ids_->bitmap_selection(new_param);
    }
  } else {
    return ColumnBaseTyped<T>::bitmap_selection(param);
  }
}

template <class T>
bool DictionaryCompressedColumn<T>::isCompressed() const throw() {
  return true;
}

template <class T>
T& DictionaryCompressedColumn<T>::operator[](const TID index) {
  return (*reverse_lookup_vector_)[(*ids_)[index]];
}

template <class T>
size_t DictionaryCompressedColumn<T>::getSizeinBytes() const throw() {
  // std::vector<uint32_t> ids_
  return ids_->capacity() * sizeof(uint32_t)
         // Dictionary dictionary_ values
         + dictionary_->size() * sizeof(uint32_t)
         // Dictionary dictionary_ keys
         + dictionary_->size() * sizeof(T)
         // std::vector<T> reverse_lookup_vector_
         + reverse_lookup_vector_->capacity() * sizeof(T)
         // uint32_t maximal_id_
         + sizeof(uint32_t)
         // this->has_primary_key_constraint_, this->has_foreign_key_constraint_
         + 2 * sizeof(bool)
         // this->fk_constr_
         + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
         this->fk_constr_.getNameOfForeignKeyTable().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
}

template <>
inline size_t DictionaryCompressedColumn<std::string>::getSizeinBytes() const
    throw() {
  size_t size_in_bytes = 0;
  // Dictionary dictionary_ keys
  for (Dictionary::const_iterator iter(dictionary_->begin()),
       end(dictionary_->end());
       iter != end; ++iter) {
    size_in_bytes += iter->first.capacity();
  }
  // std::vector<T> reverse_lookup_vector_
  for (ReverseLookupVector::const_iterator itr(reverse_lookup_vector_->begin()),
       end(reverse_lookup_vector_->end());
       itr != end; ++itr) {
    size_in_bytes += itr->capacity();
  }
  // std::vector<uint32_t> ids_
  return size_in_bytes + ids_->capacity() * sizeof(uint32_t)
         // Dictionary dictionary_ values
         + dictionary_->size() * sizeof(uint32_t)
         // uint32_t maximal_id_
         + sizeof(uint32_t)
         // this->has_primary_key_constraint_, this->has_foreign_key_constraint_
         + 2 * sizeof(bool)
         // this->fk_constr_
         + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
         this->fk_constr_.getNameOfForeignKeyTable().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
}

/***************** End of Implementation Section ******************/

//    template class DictionaryCompressedColumn<std::string>;
COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(
    DictionaryCompressedColumn)

}  // end namespace CoGaDB
