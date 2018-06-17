#pragma once
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <core/compressed_column.hpp>
#include <iostream>
#include <map>
#include <string>

namespace CoGaDB {

  const char one = '1';
  const char zero = '0';

  typedef std::string sstring;

  /*!
   *  \brief     This class represents a bit vector compressed column with type
   * T, is the base class for all compressed typed column classes.
   */
  template <class T>
  class BitVectorCompressedColumn : public CompressedColumn<T> {
   public:
    /***************** constructors and destructor *****************/
    BitVectorCompressedColumn(const sstring& name, AttributeType db_type);
    ~BitVectorCompressedColumn();

    bool insert(const boost::any& new_value);
    bool insert(const T& new_value);
    template <typename InputIterator>
    bool insert(InputIterator first, InputIterator last);

    bool update(TID tid, const boost::any& new_value);
    bool update(PositionListPtr tids, const boost::any& new_value);

    bool remove(TID tid);
    // assumes tid list is sorted ascending
    bool remove(PositionListPtr tids);
    bool clearContent();

    const boost::any get(TID tid);
    void print() const throw();
    size_t size() const throw();
    size_t getSizeinBytes() const throw();

    const ColumnPtr copy() const;

    //        bool store(const sstring& path_);
    //        bool load(const sstring& path_);

    T& operator[](const TID index);

   private:
    bool load_impl(const std::string& path,
                   boost::archive::binary_iarchive& ia);
    bool store_impl(const std::string& path,
                    boost::archive::binary_oarchive& oa);

    // Methods
    bool reorganizeAfterInsert(const T& new_Value);
    void reorganizeAfterRemove(TID tid);
    sstring buildNewKey();
    sstring buildNewKey(TID tid);
    sstring getPath(const sstring& path_);
    int countValueForKey(const sstring key);
    sstring getKey(const TID index);

    // Attributes
    std::map<sstring, T> bitVectorMap;
  };

  /***************** Start of Implementation Section ******************/

  template <class T>
  BitVectorCompressedColumn<T>::BitVectorCompressedColumn(const sstring& name,
                                                          AttributeType db_type)
      : CompressedColumn<T>(name, db_type, BIT_VECTOR_COMPRESSED),
        bitVectorMap() {}

  template <class T>
  BitVectorCompressedColumn<T>::~BitVectorCompressedColumn() {}

  template <class T>
  bool BitVectorCompressedColumn<T>::insert(const boost::any& new_value) {
    if (new_value.empty()) return false;

    if (typeid(T) == new_value.type()) {
      T value = boost::any_cast<T>(new_value);
      return this->insert(value);
    }
    return false;
  }

  template <class T>
  bool BitVectorCompressedColumn<T>::insert(const T& new_value) {
    bool found = reorganizeAfterInsert(new_value);

    if (found) return true;

    sstring new_key = buildNewKey();

    bitVectorMap[new_key] = new_value;

    return true;
  }

  template <typename T>
  sstring BitVectorCompressedColumn<T>::buildNewKey(TID tid) {
    sstring key = "";
    sstring firstEntry = bitVectorMap.begin()->first;
    for (unsigned int i = 0; i < firstEntry.length(); ++i) {
      if (i == tid)
        key += one;
      else
        key += zero;
    }

    return key;
  }

  template <typename T>
  sstring BitVectorCompressedColumn<T>::buildNewKey() {
    if (bitVectorMap.empty()) {
      return "1";
    }

    return buildNewKey((bitVectorMap.begin()->first.length()) - 1);
  }

  template <typename T>
  bool BitVectorCompressedColumn<T>::reorganizeAfterInsert(const T& new_Value) {
    std::map<sstring, T> temp;
    bool found = false;
    sstring key;
    T value;

    while (!bitVectorMap.empty()) {
      key = bitVectorMap.begin()->first;
      value = bitVectorMap.begin()->second;

      bitVectorMap.erase(key);

      if (value == new_Value) {
        key += one;
        found = true;
      } else {
        key += zero;
      }

      temp[key] = value;
    }

    bitVectorMap = temp;

    return found;
  }

  template <typename T>
  template <typename InputIterator>
  bool BitVectorCompressedColumn<T>::insert(InputIterator first,
                                            InputIterator last) {
    for (InputIterator it = first; it != last; it++) {
      insert(*it);
    }

    return true;
  }

  template <class T>
  const boost::any BitVectorCompressedColumn<T>::get(TID tid) {
    return boost::any(operator[](tid));
  }

  template <class T>
  void BitVectorCompressedColumn<T>::print() const throw() {
    typename std::map<sstring, T>::const_iterator iter;
    for (iter = bitVectorMap.begin(); iter != bitVectorMap.end(); iter++) {
      std::cout << "key: " << iter->first << " value: " << iter->second
                << std::endl;
    }
  }

  template <class T>
  size_t BitVectorCompressedColumn<T>::size() const throw() {
    if (!bitVectorMap.empty()) {
      sstring firstEntry = bitVectorMap.begin()->first;
      return firstEntry.length();
    }

    return 0;
  }

  template <class T>
  const ColumnPtr BitVectorCompressedColumn<T>::copy() const {
    return ColumnPtr(new BitVectorCompressedColumn<T>(*this));
  }

  template <class T>
  int BitVectorCompressedColumn<T>::countValueForKey(const sstring key) {
    int result = 0;

    for (unsigned int i = 0; i < key.length(); i++) {
      if (key[i] == one) result++;
    }

    return result;
  }

  template <class T>
  bool BitVectorCompressedColumn<T>::update(TID tid,
                                            const boost::any& new_value) {
    if (new_value.empty()) return false;

    if (typeid(T) != new_value.type()) {
      std::cout << "Fatal Error!!! Typemismatch for column " << this->name_
                << std::endl;
      return false;
    }

    sstring old_key = getKey(tid);

    if (old_key.empty()) {
      std::cout << "Error!!! TID not found " << tid << std::endl;
      return false;
    }

    T old_value = bitVectorMap[old_key];
    T value = boost::any_cast<T>(new_value);

    int sumElements = countValueForKey(old_key);

    if (sumElements == 1) {
      bitVectorMap[old_key] = value;
    } else if (sumElements > 1) {
      old_key[tid] = zero;
      bitVectorMap[old_key] = old_value;

      sstring new_Key = buildNewKey(tid);
      bitVectorMap[new_Key] = value;
    }

    return true;
  }

  template <class T>
  bool BitVectorCompressedColumn<T>::update(PositionListPtr tids,
                                            const boost::any& new_value) {
    if (!tids || tids->empty()) return false;

    for (unsigned int i = 0; i < tids->size(); i++) {
      update((*tids)[i], new_value);
    }
    return true;
  }

  template <class T>
  bool BitVectorCompressedColumn<T>::remove(TID tid) {
    sstring key = getKey(tid);

    if (key.empty()) {
      std::cout << "Error!!! TID not found " << tid << std::endl;
      return false;
    }

    if (countValueForKey(key) == 1) bitVectorMap.erase(key);

    reorganizeAfterRemove(tid);

    return true;
  }

  template <class T>
  void BitVectorCompressedColumn<T>::reorganizeAfterRemove(TID tid) {
    std::map<sstring, T> temp;
    sstring key;
    T value;

    while (!bitVectorMap.empty()) {
      key = bitVectorMap.begin()->first;
      value = bitVectorMap.begin()->second;

      bitVectorMap.erase(key);

      temp[key.erase(tid, 1)] = value;
    }

    bitVectorMap = temp;
  }

  template <class T>
  bool BitVectorCompressedColumn<T>::remove(PositionListPtr tids) {
    if (!tids || tids->empty()) return false;

    //        typename PositionList::reverse_iterator rit;
    //
    //		for (rit = tids->rbegin(); rit!=tids->rend(); ++rit)
    //        {
    //            remove(*rit);
    //        }

    unsigned int loop_counter = tids->size();
    while (loop_counter > 0) {
      loop_counter--;
      remove((*tids)[loop_counter]);
    }

    return true;
  }

  template <class T>
  bool BitVectorCompressedColumn<T>::clearContent() {
    bitVectorMap.clear();
    return true;
  }

  template <class T>
  sstring BitVectorCompressedColumn<T>::getPath(const sstring& path_) {
    sstring path(path_);
    path += "/";
    path += this->getName();

    return path;
  }

  template <class T>
  bool BitVectorCompressedColumn<T>::store_impl(
      const std::string&, boost::archive::binary_oarchive& oa) {
    oa << bitVectorMap;
    return true;
  }

  template <class T>
  bool BitVectorCompressedColumn<T>::load_impl(
      const std::string&, boost::archive::binary_iarchive& ia) {
    ia >> bitVectorMap;
    return true;
  }

  template <class T>
  T& BitVectorCompressedColumn<T>::operator[](const TID index) {
    return bitVectorMap[getKey(index)];
  }

  template <class T>
  sstring BitVectorCompressedColumn<T>::getKey(const TID index) {
    sstring key;
    typename std::map<sstring, T>::iterator iter;
    for (iter = bitVectorMap.begin(); iter != bitVectorMap.end(); iter++) {
      key = iter->first;
      if (key[index] == one) return key;
    }

    return "";
  }

  template <class T>
  size_t BitVectorCompressedColumn<T>::getSizeinBytes() const throw() {
    return bitVectorMap.size() * (sizeof(T) + sizeof(sstring));
  }
}
