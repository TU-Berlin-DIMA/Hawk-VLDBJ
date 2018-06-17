
#include <compression/order_preserving_dictionary_compressed_column.hpp>
#include <util/dictionary_compression.hpp>
#include <util/regular_expression.hpp>

namespace CoGaDB {

template <class T>
const typename OrderPreservingDictionaryCompressedColumn<T>::Dictionary&
OrderPreservingDictionaryCompressedColumn<T>::getDictionary() const throw() {
  return dictionary_;
}

template <class T>
uint32_t* OrderPreservingDictionaryCompressedColumn<T>::getIdData() {
  return ids_->data();
}

template <class T>
const T* const
OrderPreservingDictionaryCompressedColumn<T>::getReverseLookupVector() const {
  if (reverse_lookup_vector_.empty()) {
    return NULL;
  } else {
    return &reverse_lookup_vector_[0];
  }
}

template <class T>
T& OrderPreservingDictionaryCompressedColumn<T>::reverseLookup(uint32_t id) {
  return reverse_lookup_vector_[id];
}

template <class T>
std::pair<bool, uint32_t> OrderPreservingDictionaryCompressedColumn<
    T>::getDictionaryID(const T& value) const {
  //                    std::string filter_val =
  //                    boost::any_cast<std::string>(comp_val);
  //                    const typename
  //                    OrderPreservingDictionaryCompressedColumn<std::string>::Dictionary&
  //                    dictionary = host_col->getDictionary();
  //                    typename
  //                    OrderPreservingDictionaryCompressedColumn<std::string>::Dictionary::const_iterator
  //                    it;
  typename Dictionary::const_iterator it;
  uint32_t id = 0;
  it = dictionary_.find(value);
  // translate the string value for comparison to internal integer id, which
  // represen the strings in the column
  if (it != dictionary_.end()) {
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
std::pair<bool, uint32_t> OrderPreservingDictionaryCompressedColumn<T>::
    getClosestDictionaryIDForPredicate(
        const T& value, const ValueComparator& comp,
        ValueComparator& rewritten_value_comparator) const {
  ValueComparator tmp_rewritten_value_comparator = comp;
  std::pair<bool, uint32_t> ret = this->getDictionaryID(value);
  if (ret.first) {
    rewritten_value_comparator = tmp_rewritten_value_comparator;
    return ret;
  } else {
    if (comp == EQUAL || comp == UNEQUAL) {
      /* ok, we have not found an entry for "value" in our dictionary,
       * so the value does not exist. Return a non existing key,
       * in this case we can still exploit the dictionary compression.
       * \todo: make optimizer detect this, we can optimize this
       * predicate away (possibly do not even perform the query when
       * we know the result will be empty anyway)
       */
      rewritten_value_comparator = tmp_rewritten_value_comparator;
      return std::make_pair(true, maximal_id_ + 1);
    }
  }

  typename Dictionary::const_iterator it;
  uint32_t id = 0;
  if (comp == GREATER) {
    it = dictionary_.lower_bound(value);
    if (it != dictionary_.begin() && it != dictionary_.end()) {
      --it;
    } else if (it == dictionary_.begin()) {
      /* greater than a value that does not exist
       -> this requires a rewrite of the predicate! */
      tmp_rewritten_value_comparator = GREATER_EQUAL;
    } else if (it == dictionary_.end()) {
      rewritten_value_comparator = tmp_rewritten_value_comparator;
      return std::make_pair(true, maximal_id_ + 1);
    }
  } else if (comp == GREATER_EQUAL) {
    it = dictionary_.upper_bound(value);
  } else if (comp == LESSER) {
    it = dictionary_.upper_bound(value);
    //            if(it==dictionary_.begin()){
    //                /* handle case were we create a less than comparison of an
    //                unsigned integer
    //                 with 0: rewrite to lesser_equal*/
    //            }
  } else if (comp == LESSER_EQUAL) {
    it = dictionary_.lower_bound(value);
  } else {
    COGADB_FATAL_ERROR("Invalid value comparator!", "");
  }
  rewritten_value_comparator = tmp_rewritten_value_comparator;
  // translate the string value for comparison to internal integer id,
  // which represents the string in the column
  if (it != dictionary_.end()) {
    id = it->second;
    return std::make_pair(true, id);
  } else {
    return std::make_pair(false, maximal_id_ + 1);
  }
}

template <class T>
unsigned int OrderPreservingDictionaryCompressedColumn<
    T>::getNumberofDistinctValues() const {
  return this->dictionary_.size();
}

template <class T>
uint32_t OrderPreservingDictionaryCompressedColumn<T>::getLargestID() const {
  return this->maximal_id_;
}

template <class T>
size_t OrderPreservingDictionaryCompressedColumn<T>::getNumberOfRows() const {
  return this->size();
}

/***************** Start of Implementation Section ******************/

template <class T>
OrderPreservingDictionaryCompressedColumn<T>::
    OrderPreservingDictionaryCompressedColumn(
        const std::string& name, AttributeType db_type,
        const hype::ProcessingDeviceMemoryID& mem_id)
    : CompressedColumn<T>(name, db_type,
                          DICTIONARY_COMPRESSED_ORDER_PRESERVING),
      ids_(new IDColumn(name + "_ENCODING_IDS", UINT32, mem_id)),
      dictionary_(),
      reverse_lookup_vector_(),
      maximal_id_(0) {}

template <class T>
OrderPreservingDictionaryCompressedColumn<T>::
    OrderPreservingDictionaryCompressedColumn(
        const OrderPreservingDictionaryCompressedColumn<T>& other)
    : CompressedColumn<T>(other.getName(), other.getType(),
                          other.getColumnType()),
      ids_(boost::dynamic_pointer_cast<IDColumn>(other.ids_->copy())),
      dictionary_(other.dictionary_),
      reverse_lookup_vector_(other.reverse_lookup_vector_),
      maximal_id_(other.maximal_id_) {
  assert(ids_ != NULL);
}

template <class T>
OrderPreservingDictionaryCompressedColumn<T>&
OrderPreservingDictionaryCompressedColumn<T>::operator=(
    const OrderPreservingDictionaryCompressedColumn<T>& other) {
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
OrderPreservingDictionaryCompressedColumn<
    T>::~OrderPreservingDictionaryCompressedColumn() {}

template <class T>
bool OrderPreservingDictionaryCompressedColumn<T>::insert(
    const boost::any& new_Value) {
  T value = boost::any_cast<T>(new_Value);

  return this->insert(value);
}

template <class T>
bool OrderPreservingDictionaryCompressedColumn<T>::insert(const T& value) {
  typename Dictionary::iterator it = dictionary_.find(value);
  if (it != dictionary_.end()) {
    ids_->push_back(it->second);
  } else {
    // not in Dictionary --> new value
    updateColumnPreserveringOrder(ids_->size(), value);
  }

  return true;
}

template <class T>
void OrderPreservingDictionaryCompressedColumn<
    T>::updateColumnPreserveringOrder(const u_int64_t index, const T& value) {
  // check if new value is not bigger than old values
  uint32_t idOfNewValue = 0;
  uint32_t currentIndex = 0;
  bool not_found_index = true;

  for (typename std::vector<T>::iterator it = reverse_lookup_vector_.begin();
       it != reverse_lookup_vector_.end(); ++it, ++currentIndex) {
    // compare
    if (value < *it) {
      idOfNewValue = currentIndex;
      not_found_index = false;
      break;
    }
  }

  if (!quiet && debug)
    std::cout << "Einfüge-Index: " << index
              << " Size von ids_: " << ids_->size() << std::endl;

  if (not_found_index) {
    // new value is added at the end --> column does not have to be updated
    if (index == ids_->size())
      ids_->push_back(maximal_id_);  // an index ablegen
    else
      (*ids_)[index] = maximal_id_;

    dictionary_.insert(std::make_pair(value, maximal_id_));
    // element id is position in reverse lookup vector to get the real value in
    // O(1) time
    reverse_lookup_vector_.push_back(value);
    ++maximal_id_;

  } else {
    // insert new value into reverse_lookup_vector
    typename std::vector<T>::iterator it = reverse_lookup_vector_.begin();

    if (!quiet && debug)
      std::cout << "Adding new value into Reverse Lookup Vector at position "
                << idOfNewValue << std::endl;

    reverse_lookup_vector_.insert(it + idOfNewValue, value);

    // clear current Dictionary and fill with content from the given vector
    dictionary_.clear();
    int encodedId = 0;

    for (typename std::vector<T>::vector::iterator it =
             reverse_lookup_vector_.begin();
         it != reverse_lookup_vector_.end(); ++it) {
      dictionary_.insert(std::make_pair(*it, encodedId));
      encodedId++;
    }

    // we only have to update (increment) ids which are >= idOfNewValue
    for (IDColumn::iterator it = ids_->begin(); it != ids_->end(); ++it) {
      uint32_t currentValue = *it;

      if (currentValue >= idOfNewValue) {
        currentValue++;
        *it = currentValue;
      }
    }
    // finally add new Value to ids
    if (index == ids_->size())
      ids_->push_back(idOfNewValue);
    else
      (*ids_)[index] = idOfNewValue;

    ++maximal_id_;
  }
}

template <class T>
const boost::any OrderPreservingDictionaryCompressedColumn<T>::get(TID tid) {
  return boost::any(this->operator[](tid));
}

template <class T>
void OrderPreservingDictionaryCompressedColumn<T>::print() const throw() {
  std::cout << "| " << this->name_ << " (Dictionary Compressed) |" << std::endl;
  std::cout << "________________________" << std::endl;
  for (unsigned int i = 0; i < this->size(); i++) {
    std::cout << "| " << reverse_lookup_vector_[(*ids_)[i]] << " |"
              << std::endl;
  }
}

template <class T>
size_t OrderPreservingDictionaryCompressedColumn<T>::size() const throw() {
  return ids_->size();
}

template <class T>
const ColumnPtr OrderPreservingDictionaryCompressedColumn<T>::copy() const {
  return ColumnPtr(new OrderPreservingDictionaryCompressedColumn<T>(*this));
}

template <class T>
const ColumnPtr OrderPreservingDictionaryCompressedColumn<T>::copy(
    const hype::ProcessingDeviceMemoryID& mem_id) const {
  if (this->getMemoryID() == mem_id) return this->copy();

  ColumnPtr new_id_col = this->ids_->copy(mem_id);
  if (!new_id_col) return ColumnPtr();

  IDColumnPtr new_typed_id_col =
      boost::dynamic_pointer_cast<IDColumn>(new_id_col);

  if (!new_typed_id_col) return ColumnPtr();

  OrderPreservingDictionaryCompressedColumn<T>* new_col =
      new OrderPreservingDictionaryCompressedColumn<T>(this->getName(),
                                                       this->getType(), mem_id);
  new_col->dictionary_ = this->dictionary_;
  new_col->reverse_lookup_vector_ = this->reverse_lookup_vector_;
  new_col->maximal_id_ = this->maximal_id_;
  new_col->ids_ = new_typed_id_col;

  return ColumnPtr(new_col);
}

template <class T>
const typename ColumnBaseTyped<T>::DenseValueColumnPtr
OrderPreservingDictionaryCompressedColumn<T>::copyIntoDenseValueColumn(
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
  OrderPreservingDictionaryCompressedColumn<T>* this_col =
      const_cast<OrderPreservingDictionaryCompressedColumn<T>*>(this);
  for (size_t i = 0; i < num_elements; ++i) {
    result->insert((*this_col)[i]);
  }
  return result;
}

template <class T>
const ColumnPtr OrderPreservingDictionaryCompressedColumn<T>::gather(
    PositionListPtr tid_list, const GatherParam& param) {
  OrderPreservingDictionaryCompressedColumn<T>* result =
      new OrderPreservingDictionaryCompressedColumn<T>(this->name_,
                                                       this->db_type_);

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
OrderPreservingDictionaryCompressedColumn<T>::createColumnGroupingKeys(
    const ProcessorSpecification& proc_spec) const {
  return this->ids_->createColumnGroupingKeys(proc_spec);
}

template <class T>
size_t OrderPreservingDictionaryCompressedColumn<T>::getNumberOfRequiredBits()
    const {
  return this->ids_->getNumberOfRequiredBits();
}

template <class T>
bool OrderPreservingDictionaryCompressedColumn<T>::update(
    TID index, const boost::any& new_Value) {
  T value = boost::any_cast<T>(new_Value);
  if (index >= this->size()) return false;

  std::cout << "Updating index " << index << std::endl;

  typename Dictionary::iterator it = dictionary_.find(value);
  if (it != dictionary_.end()) {
    // ids_->push_back(it->second);
    (*ids_)[index] = it->second;
    // reverse_lookup_vector_[ids[index]]=it->first;
  } else {
    // ids_->push_back(maximal_id_);
    std::cout << "New Value does not exists yet -> will be added now"
              << std::endl;
    updateColumnPreserveringOrder(index, value);

    /**(*ids_)[index] = maximal_id_;
    dictionary_.insert(std::make_pair(value, maximal_id_));
    //element id is position in reverse lookup vector to get the real value in
    O(1) time
    reverse_lookup_vector_.push_back(value);
    maximal_id_++; **/
  }
  return true;
}

template <class T>
bool OrderPreservingDictionaryCompressedColumn<T>::update(
    PositionListPtr tids, const boost::any& new_Value) {
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
bool OrderPreservingDictionaryCompressedColumn<T>::remove(TID tid) {
  ids_->remove(tid);  // ids_->begin() + tid);
  return true;
}

template <class T>
bool OrderPreservingDictionaryCompressedColumn<T>::remove(
    PositionListPtr tids) {
  return ids_->remove(tids);
}

template <class T>
bool OrderPreservingDictionaryCompressedColumn<T>::clearContent() {
  this->ids_->clear();
  this->dictionary_.clear();
  this->reverse_lookup_vector_.clear();
  this->maximal_id_ = 0;
  return true;
}

template <class T>
bool OrderPreservingDictionaryCompressedColumn<T>::store_impl(
    const std::string& path_to_table_dir, boost::archive::binary_oarchive& oa) {
  oa << dictionary_;
  oa << reverse_lookup_vector_;
  oa << this->maximal_id_;
  return ids_->store(path_to_table_dir);
}

template <>
bool OrderPreservingDictionaryCompressedColumn<char*>::store_impl(
    const std::string& path_to_table_dir, boost::archive::binary_oarchive& oa) {
  COGADB_FATAL_ERROR("", "");
  return false;
}

template <class T>
const ColumnPtr OrderPreservingDictionaryCompressedColumn<
    T>::getDecompressedColumn_impl(const ProcessorSpecification& proc_spec) {
  return this->copyIntoDenseValueColumn(proc_spec);
}

template <>
const ColumnPtr OrderPreservingDictionaryCompressedColumn<std::string>::
    getDecompressedColumn_impl(const ProcessorSpecification& proc_spec) {
  boost::shared_ptr<Column<char*> > col = createPointerArrayToValues(
      ids_->data(), ids_->size(), this->reverse_lookup_vector_.data());
  return col;
}

template <class T>
bool OrderPreservingDictionaryCompressedColumn<T>::load_impl(
    const std::string& path_to_table_dir, boost::archive::binary_iarchive& ia) {
  ia >> dictionary_;
  ia >> reverse_lookup_vector_;
  ia >> this->maximal_id_;
  return ids_->load(path_to_table_dir, LOAD_ALL_DATA);
}

template <>
bool OrderPreservingDictionaryCompressedColumn<char*>::load_impl(
    const std::string& path_to_table_dir, boost::archive::binary_iarchive& ia) {
  COGADB_FATAL_ERROR("", "");
  return false;
}

template <class T>
bool OrderPreservingDictionaryCompressedColumn<T>::isMaterialized() const
    throw() {
  return false;
}

template <class T>
const ColumnPtr
OrderPreservingDictionaryCompressedColumn<T>::materialize() throw() {
  return this->copy();
}

template <class T>
hype::ProcessingDeviceMemoryID
OrderPreservingDictionaryCompressedColumn<T>::getMemoryID() const {
  // column is dormant in memory where ever our ids_ column is
  return ids_->getMemoryID();
}

template <class T>
const PositionListPtr OrderPreservingDictionaryCompressedColumn<T>::selection(
    const SelectionParam& param) {
  if (param.pred_type == ValueConstantPredicate &&
      (param.comp == EQUAL || param.comp == UNEQUAL)) {
    T value = boost::any_cast<T>(param.value);
    typename Dictionary::iterator it = this->dictionary_.find(value);
    if (it == dictionary_.end()) {
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
        boost::any_cast<std::string>(param.value), dictionary_);
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
const BitmapPtr OrderPreservingDictionaryCompressedColumn<T>::bitmap_selection(
    const SelectionParam& param) {
  if (param.pred_type == ValueConstantPredicate && param.comp == EQUAL) {
    T value = boost::any_cast<T>(param.value);
    typename Dictionary::iterator it = this->dictionary_.find(value);
    if (it == dictionary_.end()) {
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
bool OrderPreservingDictionaryCompressedColumn<T>::isCompressed() const
    throw() {
  return true;
}

template <class T>
T& OrderPreservingDictionaryCompressedColumn<T>::operator[](const TID index) {
  return this->reverse_lookup_vector_[(*ids_)[index]];
}

template <class T>
size_t OrderPreservingDictionaryCompressedColumn<T>::getSizeinBytes() const
    throw() {
  // std::vector<uint32_t> ids_
  return ids_->capacity() * sizeof(uint32_t)
         // Dictionary dictionary_ values
         + dictionary_.size() * sizeof(uint32_t)
         // Dictionary dictionary_ keys
         + dictionary_.size() * sizeof(T)
         // std::vector<T> reverse_lookup_vector_
         + reverse_lookup_vector_.capacity() * sizeof(T)
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
inline size_t
OrderPreservingDictionaryCompressedColumn<std::string>::getSizeinBytes() const
    throw() {
  size_t size_in_bytes = 0;
  // Dictionary dictionary_ keys
  typename Dictionary::const_iterator iter;
  for (iter = dictionary_.begin(); iter != dictionary_.end(); ++iter) {
    size_in_bytes += iter->first.capacity();
  }
  // std::vector<T> reverse_lookup_vector_
  for (size_t i = 0; i < reverse_lookup_vector_.size(); i++) {
    size_in_bytes += reverse_lookup_vector_[i].capacity();
  }
  // std::vector<uint32_t> ids_
  return size_in_bytes + ids_->capacity() * sizeof(uint32_t)
         // Dictionary dictionary_ values
         + dictionary_.size() * sizeof(uint32_t)
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

//    template class OrderPreservingDictionaryCompressedColumn<std::string>;
COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(
    OrderPreservingDictionaryCompressedColumn)

}  // end namespace CoGaDB
