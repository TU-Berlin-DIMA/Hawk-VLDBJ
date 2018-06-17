#pragma once

#include <boost/serialization/map.hpp>
#include <core/compressed_column.hpp>

#include <core/column.hpp>

namespace CoGaDB {

  class DictionaryCompressedColForBaseValues {
   public:
    virtual const std::vector<uint32_t> &getCompressedValues() const
        throw() = 0;

    virtual unsigned int getNumberofDistinctValues() const = 0;

    virtual uint32_t getLargestID() const = 0;

    virtual size_t getNumberOfRows() const = 0;
  };

  /*!
   *  \brief     This class represents a dictionary compressed column with type
   * T, is the base class for all compressed typed column classes.
   *  \author    Sebastian Breß
   *  \version   0.2
   *  \date      2013
   *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
   * http://www.gnu.org/licenses/lgpl-3.0.txt
   */
  template <class T>
  class DictionaryCompressedColumnForBaseValues
      : public CompressedColumn<T>,
        public DictionaryCompressedColForBaseValues {
   public:
    typedef std::map<T, uint32_t> Dictionary;
    typedef Column<uint32_t> IDColumn;
    typedef boost::shared_ptr<IDColumn> IDColumnPtr;

    /***************** constructors and destructor *****************/
    DictionaryCompressedColumnForBaseValues(
        const std::string &name, AttributeType db_type,
        const hype::ProcessingDeviceMemoryID &mem_id = hype::PD_Memory_0);

    DictionaryCompressedColumnForBaseValues(
        const std::string &name, AttributeType db_type, Dictionary dictionary,
        std::vector<T> reverse_lookup_vector,
        const hype::ProcessingDeviceMemoryID &mem_id = hype::PD_Memory_0);

    DictionaryCompressedColumnForBaseValues(
        const DictionaryCompressedColumnForBaseValues &);

    DictionaryCompressedColumnForBaseValues &operator=(
        const DictionaryCompressedColumnForBaseValues &);

    virtual ~DictionaryCompressedColumnForBaseValues();

    bool insert(const boost::any &new_Value);

    bool insert(const T &new_value);

    template <typename InputIterator>
    bool insert(InputIterator first, InputIterator last);

    virtual bool update(TID tid, const boost::any &new_value);

    virtual bool update(PositionListPtr tid, const boost::any &new_value);

    virtual bool remove(TID tid);

    // assumes tid list is sorted ascending
    virtual bool remove(PositionListPtr tid);

    bool clearContent();

    virtual const boost::any get(TID tid);

    // virtual const boost::any* const getRawData()=0;
    virtual void print() const throw();

    virtual size_t size() const throw();

    virtual size_t getSizeinBytes() const throw();

    virtual const ColumnPtr copy() const;

    virtual const ColumnPtr copy(
        const hype::ProcessingDeviceMemoryID &mem_id) const;

    virtual const typename ColumnBaseTyped<T>::DenseValueColumnPtr
    copyIntoDenseValueColumn(const ProcessorSpecification &proc_spec) const;

    const ColumnPtr gather(PositionListPtr tid_list, const GatherParam &);

    virtual const ColumnGroupingKeysPtr createColumnGroupingKeys(
        const ProcessorSpecification &proc_spec) const;

    virtual size_t getNumberOfRequiredBits() const;

    virtual bool isMaterialized() const throw();

    virtual const ColumnPtr materialize() throw();

    virtual hype::ProcessingDeviceMemoryID getMemoryID() const;

    virtual const PositionListPtr selection(const SelectionParam &param);

    virtual const BitmapPtr bitmap_selection(const SelectionParam &param);

    virtual bool isCompressed() const throw();

    const Dictionary &getDictionary() const throw();

    const std::vector<uint32_t> &getCompressedValues() const throw();

    std::pair<bool, uint32_t> getDictionaryID(const T &value) const;

    unsigned int getNumberofDistinctValues() const;

    uint32_t getLargestID() const;

    size_t getNumberOfRows() const;

    virtual T &operator[](const TID index);

   private:
    IDColumnPtr ids_;
    Dictionary dictionary_;
    std::vector<T> reverse_lookup_vector_;
    uint32_t maximal_id_;

    bool load_impl(const std::string &path,
                   boost::archive::binary_iarchive &ia);
    bool store_impl(const std::string &path,
                    boost::archive::binary_oarchive &oa);
  };

  template <class T>
  const typename DictionaryCompressedColumnForBaseValues<T>::Dictionary &
  DictionaryCompressedColumnForBaseValues<T>::getDictionary() const throw() {
    return dictionary_;
  }

  template <class T>
  const std::vector<uint32_t>
      &DictionaryCompressedColumnForBaseValues<T>::getCompressedValues() const
      throw() {
    // workaround, implement proper groupby which does not need something like
    // this!
    static std::vector<uint32_t> v(1, 1);
    return v;
    //        return ids_;
  }

  template <class T>
  std::pair<bool, uint32_t> DictionaryCompressedColumnForBaseValues<
      T>::getDictionaryID(const T &value) const {
    //                    std::string filter_val =
    //                    boost::any_cast<std::string>(comp_val);
    //                    const typename
    //                    DictionaryCompressedColumnForBaseValues<std::string>::Dictionary&
    //                    dictionary = host_col->getDictionary();
    //                    typename
    //                    DictionaryCompressedColumnForBaseValues<std::string>::Dictionary::const_iterator
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
  unsigned int DictionaryCompressedColumnForBaseValues<
      T>::getNumberofDistinctValues() const {
    return this->dictionary_.size();
  }

  template <class T>
  uint32_t DictionaryCompressedColumnForBaseValues<T>::getLargestID() const {
    return this->maximal_id_;
  }

  template <class T>
  size_t DictionaryCompressedColumnForBaseValues<T>::getNumberOfRows() const {
    return this->size();
  }

  /***************** Start of Implementation Section ******************/

  template <class T>
  DictionaryCompressedColumnForBaseValues<T>::
      DictionaryCompressedColumnForBaseValues(
          const std::string &name, AttributeType db_type,
          const hype::ProcessingDeviceMemoryID &mem_id)
      : CompressedColumn<T>(name, db_type, DICTIONARY_COMPRESSED),
        ids_(new IDColumn(name + "_ENCODING_IDS", UINT32, mem_id)),
        maximal_id_(0) {}

  template <class T>
  DictionaryCompressedColumnForBaseValues<T>::
      DictionaryCompressedColumnForBaseValues(
          const std::string &name, AttributeType db_type, Dictionary dictionary,
          std::vector<T> reverse_lookup_vector,
          const hype::ProcessingDeviceMemoryID &mem_id)
      : CompressedColumn<T>(name, db_type, DICTIONARY_COMPRESSED),
        ids_(new IDColumn(name + "_ENCODING_IDS", UINT32, mem_id)),
        dictionary_(dictionary),
        reverse_lookup_vector_(reverse_lookup_vector),
        maximal_id_(0) {}

  template <class T>
  DictionaryCompressedColumnForBaseValues<T>::
      DictionaryCompressedColumnForBaseValues(
          const DictionaryCompressedColumnForBaseValues<T> &other)
      : CompressedColumn<T>(other.getName(), other.getType(),
                            other.getColumnType()),
        ids_(boost::dynamic_pointer_cast<IDColumn>(other.ids_->copy())),
        dictionary_(other.dictionary_),
        reverse_lookup_vector_(other.reverse_lookup_vector_),
        maximal_id_(other.maximal_id_) {
    assert(ids_ != NULL);
  }

  template <class T>
  DictionaryCompressedColumnForBaseValues<T>
      &DictionaryCompressedColumnForBaseValues<T>::operator=(
          const DictionaryCompressedColumnForBaseValues<T> &other) {
    if (this != &other)  // protect against invalid self-assignment
    {
      this->name_ = other.name_;
      this->db_type_ = other.db_type_;
      this->column_type_ = other.column_type_;
      this->ids_ = other.ids_->copy();
      this->dictionary_ = other.dictionary_;
      this->reverse_lookup_vector_ = other.reverse_lookup_vector_;
      this->maximal_id_ = other.maximal_id_;
    }
    return *this;
  }

  //	template<class T>
  //	DictionaryCompressedColumnForBaseValues<T>::DictionaryCompressedColumnForBaseValues(DictionaryCompressedColumnForBaseValues&
  // other)
  //        : ids_(other.ids_), dictionary_(other.dictionary_),
  //        reverse_lookup_vector_(other.reverse_lookup_vector_),
  //        maximal_id_(other.maximal_id_)
  //        {
  //
  //        }

  template <class T>
  DictionaryCompressedColumnForBaseValues<
      T>::~DictionaryCompressedColumnForBaseValues() {}

  template <class T>
  bool DictionaryCompressedColumnForBaseValues<T>::insert(
      const boost::any &new_Value) {
    T value = boost::any_cast<T>(new_Value);

    return this->insert(value);
  }

  template <class T>
  bool DictionaryCompressedColumnForBaseValues<T>::insert(const T &value) {
    typename Dictionary::iterator it = dictionary_.find(value);
    if (it != dictionary_.end()) {
      ids_->push_back(it->second);
    } else {
      ids_->push_back(maximal_id_);
      dictionary_.insert(std::make_pair(value, maximal_id_));
      // element id is position in reverse lookup vector to get the real value
      // in O(1) time
      reverse_lookup_vector_.push_back(value);
      maximal_id_++;
    }

    return true;
  }

  template <>
  inline bool DictionaryCompressedColumnForBaseValues<std::string>::insert(
      const std::string &value) {
    ids_->push_back((int)*value.c_str());
    return true;
  }

  template <typename T>
  template <typename InputIterator>
  bool DictionaryCompressedColumnForBaseValues<T>::insert(InputIterator first,
                                                          InputIterator last) {
    for (; first != last; ++first) {
      if (!this->insert(*first)) {
        return false;
      }
    }
    return true;
  }

  template <class T>
  const boost::any DictionaryCompressedColumnForBaseValues<T>::get(TID tid) {
    return boost::any(this->operator[](tid));
  }

  template <class T>
  void DictionaryCompressedColumnForBaseValues<T>::print() const throw() {
    std::cout << "| " << this->name_ << " (Dictionary Compressed) |"
              << std::endl;
    std::cout << "________________________" << std::endl;
    for (unsigned int i = 0; i < this->size(); i++) {
      std::cout << "| " << reverse_lookup_vector_[(*ids_)[i]] << " |"
                << std::endl;
    }
  }

  template <class T>
  size_t DictionaryCompressedColumnForBaseValues<T>::size() const throw() {
    return ids_->size();
  }

  template <class T>
  const ColumnPtr DictionaryCompressedColumnForBaseValues<T>::copy() const {
    return ColumnPtr(new DictionaryCompressedColumnForBaseValues<T>(*this));
  }

  template <class T>
  const ColumnPtr DictionaryCompressedColumnForBaseValues<T>::copy(
      const hype::ProcessingDeviceMemoryID &mem_id) const {
    if (this->getMemoryID() == mem_id) return this->copy();

    ColumnPtr new_id_col = this->ids_->copy(mem_id);
    if (!new_id_col) return ColumnPtr();

    IDColumnPtr new_typed_id_col =
        boost::dynamic_pointer_cast<IDColumn>(new_id_col);

    if (!new_typed_id_col) return ColumnPtr();

    DictionaryCompressedColumnForBaseValues<T> *new_col =
        new DictionaryCompressedColumnForBaseValues<T>(this->getName(),
                                                       this->getType(), mem_id);
    new_col->dictionary_ = this->dictionary_;
    new_col->reverse_lookup_vector_ = this->reverse_lookup_vector_;
    new_col->maximal_id_ = this->maximal_id_;
    new_col->ids_ = new_typed_id_col;

    return ColumnPtr(new_col);
  }

  template <class T>
  const typename ColumnBaseTyped<T>::DenseValueColumnPtr
  DictionaryCompressedColumnForBaseValues<T>::copyIntoDenseValueColumn(
      const ProcessorSpecification &proc_spec) const {
    typedef
        typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
    typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
    assert(proc_spec.proc_id == hype::PD0);
    DenseValueColumnPtr result(
        new DenseValueColumn(this->getName(), this->getType()));
    size_t num_elements = this->size();
    DictionaryCompressedColumnForBaseValues<T> *this_col =
        const_cast<DictionaryCompressedColumnForBaseValues<T> *>(this);
    for (size_t i = 0; i < num_elements; ++i) {
      result->insert((*this_col)[i]);
    }
    return result;
  }

  template <class T>
  const ColumnPtr DictionaryCompressedColumnForBaseValues<T>::gather(
      PositionListPtr tid_list, const GatherParam &param) {
    DictionaryCompressedColumnForBaseValues<T> *result =
        new DictionaryCompressedColumnForBaseValues<T>(this->name_,
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

    //        tid_list=copied_tids;
    //
    //        //std::vector<T>& data = result->getContent();
    //        //data.resize(tid_list->size());
    //        for (size_t i = 0; i < tid_list->size(); i++) {
    //            result->insert((*this)[(*tid_list)[i]]);
    //        }
    //        return ColumnPtr(result);
  }

  template <class T>
  const ColumnGroupingKeysPtr
  DictionaryCompressedColumnForBaseValues<T>::createColumnGroupingKeys(
      const ProcessorSpecification &proc_spec) const {
    return this->ids_->createColumnGroupingKeys(proc_spec);
  }

  template <class T>
  size_t DictionaryCompressedColumnForBaseValues<T>::getNumberOfRequiredBits()
      const {
    return this->ids_->getNumberOfRequiredBits();
  }

  template <class T>
  bool DictionaryCompressedColumnForBaseValues<T>::update(
      TID index, const boost::any &new_Value) {
    T value = boost::any_cast<T>(new_Value);
    if (index >= this->size()) return false;

    typename Dictionary::iterator it = dictionary_.find(value);
    if (it != dictionary_.end()) {
      // ids_->push_back(it->second);
      (*ids_)[index] = it->second;
      // reverse_lookup_vector_[ids[index]]=it->first;
    } else {
      // ids_->push_back(maximal_id_);
      (*ids_)[index] = maximal_id_;
      dictionary_.insert(std::make_pair(value, maximal_id_));
      // element id is position in reverse lookup vector to get the real value
      // in O(1) time
      reverse_lookup_vector_.push_back(value);
      maximal_id_++;
    }
    return true;
  }

  template <class T>
  bool DictionaryCompressedColumnForBaseValues<T>::update(
      PositionListPtr tids, const boost::any &new_Value) {
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
  bool DictionaryCompressedColumnForBaseValues<T>::remove(TID tid) {
    ids_->remove(tid);  // ids_->begin() + tid);
    return true;
  }

  template <class T>
  bool DictionaryCompressedColumnForBaseValues<T>::remove(
      PositionListPtr tids) {
    return ids_->remove(tids);
    //        if (!tids)
    //            return false;
    //        //test whether tid list has at least one element, if not, return
    //        with error
    //        if (tids->empty())
    //            return false;
    //
    //        //		typename PositionList::reverse_iterator rit;
    //        //                //delete tuples in reverse order, otherwise the
    //        first deletion would invalidate all other tids
    //        //		for (rit = tids->rbegin(); rit!=tids->rend();
    //        ++rit){
    //        //			ids_->erase(ids_->begin()+(*rit));
    //        //                }
    //
    //        unsigned int loop_counter = tids->size();
    //        while (loop_counter > 0) {
    //            loop_counter--;
    //            ids_->erase(ids_->begin()+(*tids)[loop_counter]);
    //        }
    //
    //        return true;
  }

  template <class T>
  bool DictionaryCompressedColumnForBaseValues<T>::clearContent() {
    this->ids_->clear();
    this->dictionary_.clear();
    this->reverse_lookup_vector_.clear();
    this->maximal_id_ = 0;
    return true;
  }

  template <class T>
  bool DictionaryCompressedColumnForBaseValues<T>::store_impl(
      const std::string &path, boost::archive::binary_oarchive &oa) {
    oa << dictionary_;
    oa << reverse_lookup_vector_;
    oa << this->maximal_id_;
    return ids_->store(path);
  }

  template <class T>
  bool DictionaryCompressedColumnForBaseValues<T>::load_impl(
      const std::string &path, boost::archive::binary_iarchive &ia) {
    ia >> dictionary_;
    ia >> reverse_lookup_vector_;
    ia >> this->maximal_id_;
    return ids_->load(path, LOAD_ALL_DATA);
  }

  template <class T>
  bool DictionaryCompressedColumnForBaseValues<T>::isMaterialized() const
      throw() {
    return false;
  }

  template <class T>
  const ColumnPtr
  DictionaryCompressedColumnForBaseValues<T>::materialize() throw() {
    return this->copy();
  }

  template <class T>
  hype::ProcessingDeviceMemoryID
  DictionaryCompressedColumnForBaseValues<T>::getMemoryID() const {
    // column is dormant in memory where ever our ids_ column is
    return ids_->getMemoryID();
  }

  template <class T>
  const PositionListPtr DictionaryCompressedColumnForBaseValues<T>::selection(
      const SelectionParam &param) {
    if (param.pred_type == ValueConstantPredicate && param.comp == EQUAL) {
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
    } else {
      return ColumnBaseTyped<T>::selection(param);
    }
  }

  template <class T>
  const BitmapPtr DictionaryCompressedColumnForBaseValues<T>::bitmap_selection(
      const SelectionParam &param) {
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
  bool DictionaryCompressedColumnForBaseValues<T>::isCompressed() const
      throw() {
    return true;
  }

  template <class T>
  T &DictionaryCompressedColumnForBaseValues<T>::operator[](const TID index) {
    return this->reverse_lookup_vector_[(*ids_)[index]];
  }

  template <class T>
  size_t DictionaryCompressedColumnForBaseValues<T>::getSizeinBytes() const
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
           // this->has_primary_key_constraint_,
           // this->has_foreign_key_constraint_
           + 2 * sizeof(bool)
           // this->fk_constr_
           + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
           this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
           this->fk_constr_.getNameOfForeignKeyTable().capacity() +
           this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
  }

  template <>
  inline size_t
  DictionaryCompressedColumnForBaseValues<std::string>::getSizeinBytes() const
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
           // this->has_primary_key_constraint_,
           // this->has_foreign_key_constraint_
           + 2 * sizeof(bool)
           // this->fk_constr_
           + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
           this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
           this->fk_constr_.getNameOfForeignKeyTable().capacity() +
           this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
  }

  /***************** End of Implementation Section ******************/

}  // end namespace CogaDB
