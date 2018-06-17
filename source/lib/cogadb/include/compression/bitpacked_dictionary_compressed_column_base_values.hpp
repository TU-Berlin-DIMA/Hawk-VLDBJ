#pragma once

#include <boost/serialization/map.hpp>
#include <core/compressed_column.hpp>
#include <util/column_grouping_keys.hpp>
#include <util/utility_functions.hpp>

namespace CoGaDB {

  class BitPackedDictionaryCompressedColForBaseValues {
   public:
    virtual const std::vector<uint32_t> &getCompressedValues() const
        throw() = 0;

    virtual unsigned int getNumberofDistinctValues() const = 0;

    virtual uint32_t getLargestID() const = 0;

    virtual size_t getNumberOfRows() const = 0;
  };

  /*!
   *  \brief     This class represents a bit packed dictionary compressed column
   * with type T.
   *  \author    Sebastian Dorok
   *  \version   0.1
   *  \date      2014
   *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
   * http://www.gnu.org/licenses/lgpl-3.0.txt
   */
  template <typename ValueType>
  class BitPackedDictionaryCompressedColumnForBaseValues
      : public CompressedColumn<ValueType>,
        public BitPackedDictionaryCompressedColForBaseValues {
   public:
    typedef std::map<ValueType, uint32_t> Dictionary;

    /***************** constructors and destructor *****************/
    BitPackedDictionaryCompressedColumnForBaseValues(
        const std::string &name, AttributeType db_type,
        std::size_t number_of_bits_per_value = 32);

    BitPackedDictionaryCompressedColumnForBaseValues(
        const std::string &name, AttributeType db_type, Dictionary dictionary,
        std::vector<ValueType> reverse_lookup_vector,
        std::size_t number_of_bits_per_value = 32);

    // DictionaryCompressedColumn(DictionaryCompressedColumn&);
    virtual ~BitPackedDictionaryCompressedColumnForBaseValues();

    bool insert(const boost::any &new_Value);

    bool insert(const ValueType &new_value);

    template <typename InputIterator>
    bool insert(InputIterator first, InputIterator last);

    const ColumnGroupingKeysPtr createColumnGroupingKeys(
        const ProcessorSpecification &proc_spec) const;

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

    const ColumnPtr gather(PositionListPtr tid_list, const GatherParam &);

    virtual bool isMaterialized() const throw();

    virtual const ColumnPtr materialize() throw();

    virtual bool isCompressed() const throw();

    const Dictionary &getDictionary() const throw();

    const std::vector<uint32_t> &getCompressedValues() const throw();

    std::pair<bool, uint32_t> getDictionaryID(const ValueType &value) const;

    unsigned int getNumberofDistinctValues() const;

    uint32_t getLargestID() const;

    size_t getNumberOfRows() const;

    virtual ValueType &operator[](const TID index);

   private:
    std::vector<uint32_t> data_items_;
    Dictionary dictionary_;
    std::vector<ValueType> reverse_lookup_vector_;
    uint32_t maximal_id_;
    size_t number_of_rows_;
    uint32_t number_of_bits_per_value_;
    uint32_t number_of_values_per_word_;
    uint32_t padding_;

    bool load_impl(const std::string &path,
                   boost::archive::binary_iarchive &ia);

    bool store_impl(const std::string &path,
                    boost::archive::binary_oarchive &oa);
  };

  template <typename ValueType>
  const typename BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::Dictionary &
  BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::getDictionary()
      const throw() {
    return dictionary_;
  }

  template <typename ValueType>
  const std::vector<uint32_t> &BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::getCompressedValues() const throw() {
    return data_items_;
  }

  /*
   * Returns a pair of boolean value and a dictionary id associated for a given
   * value.
   * If the boolean is true the value already exists in the dictionary,
   * otherwise the boolean is false.
   */
  template <typename ValueType>
  std::pair<bool, uint32_t> BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::getDictionaryID(const ValueType &value) const {
    // std::string filter_val = boost::any_cast<std::string>(comp_val);
    // const typename DictionaryCompressedColumn<std::string>::Dictionary&
    // dictionary = host_col->getDictionary();
    // typename
    // DictionaryCompressedColumn<std::string>::Dictionary::const_iterator it;
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

  template <typename ValueType>
  unsigned int BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::getNumberofDistinctValues() const {
    return this->dictionary_.size();
  }

  template <typename ValueType>
  uint32_t BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::getLargestID() const {
    return this->maximal_id_;
  }

  template <typename ValueType>
  size_t BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::getNumberOfRows() const {
    return this->number_of_rows_;
  }

  /***************** Start of Implementation Section ******************/

  template <typename ValueType>
  BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::
      BitPackedDictionaryCompressedColumnForBaseValues(
          const std::string &name, AttributeType db_type,
          std::size_t number_of_bits_per_value)
      : CompressedColumn<ValueType>(name, db_type,
                                    BITPACKED_DICTIONARY_COMPRESSED),
        data_items_(),
        dictionary_(),
        reverse_lookup_vector_(),
        maximal_id_(0),
        number_of_rows_(0),
        number_of_bits_per_value_(number_of_bits_per_value) {
    // TODO Currently we use a fixed word size of 32 bits
    this->number_of_values_per_word_ = 32 / number_of_bits_per_value_;
    this->padding_ = 32 % number_of_bits_per_value_;
  }

  template <typename ValueType>
  BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::
      BitPackedDictionaryCompressedColumnForBaseValues(
          const std::string &name, AttributeType db_type, Dictionary dictionary,
          std::vector<ValueType> reverse_lookup_vector,
          std::size_t number_of_bits_per_value)
      : CompressedColumn<ValueType>(name, db_type,
                                    BITPACKED_DICTIONARY_COMPRESSED),
        data_items_(),
        dictionary_(dictionary),
        reverse_lookup_vector_(reverse_lookup_vector),
        maximal_id_(0),
        number_of_rows_(0),
        number_of_bits_per_value_(number_of_bits_per_value) {
    // TODO Currently we use a fixed word size of 32 bits
    this->number_of_values_per_word_ = 32 / number_of_bits_per_value_;
    this->padding_ = 32 % number_of_bits_per_value_;
  }

  //	template<typename ValueType>
  //	DictionaryCompressedColumn<ValueType>::DictionaryCompressedColumn(DictionaryCompressedColumn&
  // other)
  //        : data_items_(other.data_items_), dictionary_(other.dictionary_),
  //        reverse_lookup_vector_(other.reverse_lookup_vector_),
  //        maximal_id_(other.maximal_id_)
  //        {
  //
  //        }

  template <typename ValueType>
  BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::~BitPackedDictionaryCompressedColumnForBaseValues() {}

  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::insert(
      const boost::any &new_Value) {
    ValueType value = boost::any_cast<ValueType>(new_Value);

    return this->insert(value);
  }

  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::insert(
      const ValueType &value) {
    COGADB_ERROR("Not implemented for other types than string -> base values!",
                 "");

    return true;
  }

  template <>
  inline bool BitPackedDictionaryCompressedColumnForBaseValues<
      std::string>::insert(const std::string &value) {
    assert(value.length() == 1);
    // TODO only works for characters and for 3 bit values currently - use pow
    // to fix
    int id = (int)*value.c_str() & 7;

    // index as key
    if (this->number_of_rows_ % this->number_of_values_per_word_ == 0) {
      // e.g., every 10th value requires a new 32 bit word when encoded with 3
      // bit (2 bit padding))
      this->data_items_.push_back(id);
    } else {
      uint32_t x = this->data_items_[this->number_of_rows_ /
                                     this->number_of_values_per_word_];
      // TODO Currently, the lowest index is left most -> maybe retrieval
      // performance is better if it is right-most
      uint32_t y = x << number_of_bits_per_value_ | id;
      this->data_items_[this->number_of_rows_ /
                        this->number_of_values_per_word_] = y;
    }

    // we just inserted a new word
    this->number_of_rows_++;

    return true;
  }

  template <typename ValueType>
  template <typename InputIterator>
  bool BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::insert(
      InputIterator first, InputIterator last) {
    for (; first != last; ++first) {
      if (!this->insert(*first)) {
        return false;
      }
    }
    return true;
  }

  template <typename ValueType>
  const boost::any
  BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::get(TID tid) {
    return boost::any(this->operator[](tid));
  }

  template <typename ValueType>
  void BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::print()
      const throw() {
    std::cout << "| " << this->name_ << " (Bit Packed Dictionary Compressed) |"
              << std::endl;
    std::cout << "________________________" << std::endl;
    for (unsigned int i = 0; i < this->number_of_rows_; i++) {
      int x = this->data_items_[i / this->number_of_values_per_word_];
      int y = x >> (((std::min((size_t) this->number_of_values_per_word_,
                               (this->number_of_rows_ -
                                (i / this->number_of_values_per_word_) *
                                    this->number_of_values_per_word_)) -
                      1) -
                     (i % this->number_of_values_per_word_)) *
                    number_of_bits_per_value_) &
              ((1 << number_of_bits_per_value_) - 1);
      std::cout << "Element " << i << ": " << this->reverse_lookup_vector_[y]
                << "(" << y << ") ";
    }
    std::cout << std::endl;
  }

  template <typename ValueType>
  size_t BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::size()
      const throw() {
    return getNumberOfRows();  // data_items_.size();
  }

  template <typename ValueType>
  const ColumnPtr
  BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::copy() const {
    return ColumnPtr(
        new BitPackedDictionaryCompressedColumnForBaseValues<ValueType>(*this));
  }

  template <typename ValueType>
  const ColumnPtr BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::gather(PositionListPtr tid_list, const GatherParam &param) {
    assert(param.proc_spec.proc_id == hype::PD0);
    // TODO look at implementation of dictionary compressed column
    BitPackedDictionaryCompressedColumnForBaseValues<ValueType> *result =
        new BitPackedDictionaryCompressedColumnForBaseValues<ValueType>(
            this->name_, this->db_type_, this->dictionary_,
            this->reverse_lookup_vector_, this->number_of_bits_per_value_);
    for (unsigned int i = 0; i < tid_list->size(); i++) {
      result->insert((*this)[(*tid_list)[i]]);
    }
    return ColumnPtr(result);
  }

  template <class T>
  const ColumnGroupingKeysPtr
  BitPackedDictionaryCompressedColumnForBaseValues<T>::createColumnGroupingKeys(
      const ProcessorSpecification &proc_spec) const {
    assert(proc_spec.proc_id == hype::PD0);

    ColumnGroupingKeysPtr result(
        new ColumnGroupingKeys(hype::util::getMemoryID(proc_spec.proc_id)));
    result->keys->reserve(this->number_of_rows_);
    result->required_number_of_bits = this->number_of_bits_per_value_;
    for (size_t i = 0; i < this->number_of_rows_; ++i) {
      int32_t v = getUncompressedID(
          i, this->data_items_, this->number_of_values_per_word_,
          this->number_of_bits_per_value_, this->number_of_rows_);
      result->keys->insert(v);
    }
    return result;
  }

  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::update(
      TID index, const boost::any &new_Value) {
    ValueType value = boost::any_cast<ValueType>(new_Value);
    // check whether value is in dictionary - TODO helper method?
    typename Dictionary::const_iterator it;
    uint32_t id = maximal_id_;
    it = dictionary_.find(value);
    // translate the string value for comparison to internal integer id, which
    // represen the strings in the column
    if (it != dictionary_.end()) {
      id = it->second;
    } else {
      dictionary_.insert(std::make_pair(value, id));
      // element id is position in reverse lookup vector to get the real value
      // in O(1) time
      reverse_lookup_vector_.push_back(value);
      maximal_id_++;
    }

    // retrieve word containing compressed data
    uint32_t x = this->data_items_[index / this->number_of_values_per_word_];
    // which entry is it in the word
    int offset = index % this->number_of_values_per_word_;
    // create bit mask to retrieve compressed value
    int bitmask = (1 << number_of_bits_per_value_) - 1;
    // determine highest offset within in word
    int highestOffsetInWord =
        std::min((size_t) this->number_of_values_per_word_,
                 (this->number_of_rows_ -
                  (index / this->number_of_values_per_word_) *
                      this->number_of_values_per_word_)) -
        1;
    // shift bitmask to correct position
    bitmask =
        bitmask << (highestOffsetInWord - offset) * number_of_bits_per_value_;
    // and invert it
    bitmask = ~bitmask;
    // shift new id to correct value
    id = id << (highestOffsetInWord - offset) * number_of_bits_per_value_;
    // delete old value
    x &= bitmask;
    // override old value with new id
    x |= id;
    // write word back
    this->data_items_[index / this->number_of_values_per_word_] = x;
    return true;
  }

  // TODO Adjust!

  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::update(
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

  /*
   * TODO Isn't there a far better way!!
   * TODO delete value from dictionary?
   */
  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::remove(
      TID tid) {
    int index = tid;
    // retrieve word containing compressed data
    uint32_t x = this->data_items_[index / this->number_of_values_per_word_];
    // which entry is it in the word
    int offset = index % this->number_of_values_per_word_;
    // create bit mask to retrieve compressed value
    int bitmask = (1 << number_of_bits_per_value_) - 1;
    // determine highest offset within in word
    int highestOffsetInWord =
        std::min((size_t) this->number_of_values_per_word_,
                 (this->number_of_rows_ -
                  (index / this->number_of_values_per_word_) *
                      this->number_of_values_per_word_)) -
        1;
    // shift bitmask to correct position
    bitmask =
        bitmask << (highestOffsetInWord - offset) * number_of_bits_per_value_;
    // erase entry
    int bitmask2 =
        (1 << ((highestOffsetInWord - offset) * number_of_bits_per_value_)) - 1;
    int bitmask3 = ~bitmask2 << number_of_bits_per_value_;

    uint32_t tmp = x & bitmask3;
    uint32_t tmp2 = x & bitmask2;
    tmp2 = tmp2 << number_of_bits_per_value_;

    this->data_items_[index / this->number_of_values_per_word_] = tmp | tmp2;

    // shift data behind
    for (uint32_t i = index / this->number_of_values_per_word_;
         i < (this->number_of_rows_ / this->number_of_values_per_word_) - 1;
         i++) {
      uint32_t x = this->data_items_[i];
      uint32_t x2 = this->data_items_[i + 1];

      // create bit mask to retrieve compressed value
      int bitmask = (1 << number_of_bits_per_value_) - 1;
      // determine highest offset within in word
      int highestOffsetInWord =
          std::min((size_t) this->number_of_values_per_word_,
                   (this->number_of_rows_ -
                    (index / this->number_of_values_per_word_) *
                        this->number_of_values_per_word_)) -
          1;
      // extract compressed value
      int y = x2 >> highestOffsetInWord * number_of_bits_per_value_ & bitmask;

      this->data_items_[i] = x | y;
      this->data_items_[i + 1] = x2 << number_of_bits_per_value_;
    }
    // TODO handling of last
    this->number_of_rows_--;
    if ((this->number_of_rows_ % this->number_of_values_per_word_) > 0) {
      this->data_items_[this->number_of_rows_ /
                        this->number_of_values_per_word_] >>=
          number_of_bits_per_value_;
    } else {
      this->data_items_.pop_back();
    }
    return true;
  }

  // TODO Adjust!

  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::remove(
      PositionListPtr tids) {
    if (!tids) return false;
    // test whether tid list has at least one element, if not, return with error
    if (tids->empty()) return false;

    //		typename PositionList::reverse_iterator rit;
    //                //delete tuples in reverse order, otherwise the first
    //                deletion would invalidate all other tids
    //		for (rit = tids->rbegin(); rit!=tids->rend(); ++rit){
    //			data_items_.erase(data_items_.begin()+(*rit));
    //                }

    unsigned int loop_counter = tids->size();
    while (loop_counter > 0) {
      loop_counter--;
      // TODO use normal method
      data_items_.erase(data_items_.begin() + (*tids)[loop_counter]);
    }

    return true;
  }

  template <typename ValueType>
  bool
  BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::clearContent() {
    this->data_items_.clear();
    this->dictionary_.clear();
    this->reverse_lookup_vector_.clear();
    this->maximal_id_ = 0;
    this->number_of_rows_ = 0;
    return true;
  }

  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::store_impl(
      const std::string &path, boost::archive::binary_oarchive &oa) {
    oa << data_items_;
    oa << dictionary_;
    oa << reverse_lookup_vector_;
    oa << this->maximal_id_;
    oa << this->number_of_rows_;
    oa << this->number_of_bits_per_value_;

    return true;
  }

  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::load_impl(
      const std::string &path, boost::archive::binary_iarchive &ia) {
    ia >> data_items_;
    ia >> dictionary_;
    ia >> reverse_lookup_vector_;
    ia >> this->maximal_id_;
    ia >> this->number_of_rows_;
    ia >> this->number_of_bits_per_value_;
    std::cout << "Number of rows: " << this->number_of_rows_ << std::endl;
    std::cout << "Computed number of rows: "
              << data_items_.size() * this->number_of_values_per_word_
              << std::endl;

    return true;
  }

  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::isMaterialized() const throw() {
    return false;
  }

  template <typename ValueType>
  const ColumnPtr BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::materialize() throw() {
    return this->copy();
  }

  template <typename ValueType>
  bool BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::isCompressed() const throw() {
    return true;
  }

  template <typename ValueType>
  ValueType &BitPackedDictionaryCompressedColumnForBaseValues<ValueType>::
  operator[](const TID index) {
    // retrieve word containing compressed data
    uint32_t x = this->data_items_[index / this->number_of_values_per_word_];
    // which entry is it in the word
    int offset = index % this->number_of_values_per_word_;
    // create bit mask to retrieve compressed value
    int bitmask = (1 << number_of_bits_per_value_) - 1;
    // determine highest offset within word
    int highestOffsetInWord =
        std::min((size_t) this->number_of_values_per_word_,
                 (this->number_of_rows_ -
                  (index / this->number_of_values_per_word_) *
                      this->number_of_values_per_word_)) -
        1;
    // extract compressed value
    int y = x >> (highestOffsetInWord - offset) * number_of_bits_per_value_ &
            bitmask;
    // return uncompressed value
    return this->reverse_lookup_vector_[y];
  }

  template <typename ValueType>
  size_t BitPackedDictionaryCompressedColumnForBaseValues<
      ValueType>::getSizeinBytes() const throw() {
    // std::vector<uint32_t> data_items_
    return data_items_.capacity() * sizeof(uint32_t)
           // Dictionary dictionary_ values
           + dictionary_.size() * sizeof(uint32_t)
           // Dictionary dictionary_ keys
           + dictionary_.size() * sizeof(ValueType)
           // std::vector<T> reverse_lookup_vector_
           + reverse_lookup_vector_.capacity() * sizeof(ValueType)
           // uint32_t maximal_id_, uint32_t number_of_rows_,
           // uint32_t number_of_bits_per_value_, uint32_t
           // number_of_values_per_word_,
           // uint32_t padding_
           + 5 * sizeof(uint32_t)
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
  inline size_t BitPackedDictionaryCompressedColumnForBaseValues<
      std::string>::getSizeinBytes() const throw() {
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
    // std::vector<uint32_t> data_items_
    return data_items_.capacity() * sizeof(uint32_t)
           // Dictionary dictionary_ values
           + dictionary_.size() * sizeof(uint32_t)
           // uint32_t maximal_id_, uint32_t number_of_rows_,
           // uint32_t number_of_bits_per_value_, uint32_t
           // number_of_values_per_word_,
           // uint32_t padding_
           + 5 * sizeof(uint32_t)
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
