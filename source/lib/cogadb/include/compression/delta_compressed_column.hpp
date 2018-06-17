#pragma once
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <core/compressed_column.hpp>
#include <fstream>

namespace CoGaDB {
  typedef unsigned int uint;
  typedef unsigned char uchar;

  template <class T>
  class DeltaCompressedColumn : public CompressedColumn<T> {
   public:
    DeltaCompressedColumn(const std::string& name, AttributeType db_type);
    ~DeltaCompressedColumn();

    bool insert(const boost::any& new_Value);
    bool insert(const T& new_value);

    template <typename InputIterator>
    bool insert(InputIterator first, InputIterator last);

    bool update(TID tid, const boost::any& new_value);
    bool update(PositionListPtr tid, const boost::any& new_value);

    bool remove(TID tid);
    // assumes tid list is sorted ascending
    bool remove(PositionListPtr tid);
    bool clearContent();

    const boost::any get(TID tid);
    void print() const throw();
    size_t size() const throw();
    size_t getSizeinBytes() const throw();

    const ColumnPtr copy() const;

    T& operator[](const TID index);

   private:
    bool load_impl(const std::string& path,
                   boost::archive::binary_iarchive& ia);
    bool store_impl(const std::string& path,
                    boost::archive::binary_oarchive& oa);
    void compress_insert(uint new_value);
    void compress_update(TID tid, const uint& new_value_);
    void compress_update_part(uint& i, uint old_delta, uint new_delta);
    void compress_delete(TID tid);

    uint uncompress(const uint index);
    uint decode(uint& index);
    const uint encoded_length(const uint& cvalue);

    std::vector<uchar> cvalues_;
    uint size_;
    T hack_last_uncompressed;
  };

  template <>
  class DeltaCompressedColumn<std::string>
      : public CompressedColumn<std::string> {
   public:
    DeltaCompressedColumn(const std::string& name, AttributeType db_type);
    ~DeltaCompressedColumn();

    bool insert(const boost::any& new_Value);
    bool insert(const std::string& new_value);

    template <typename InputIterator>
    bool insert(InputIterator first, InputIterator last);

    bool update(TID tid, const boost::any& new_value);
    bool update(PositionListPtr tid, const boost::any& new_value);

    bool remove(TID tid);
    // assumes tid list is sorted ascending
    bool remove(PositionListPtr tid);
    bool clearContent();

    const boost::any get(TID tid);
    void print() const throw();
    size_t size() const throw();
    size_t getSizeinBytes() const throw();

    const ColumnPtr copy() const;

    std::string& operator[](const TID index);

   private:
    bool load_impl(const std::string& path,
                   boost::archive::binary_iarchive& ia);
    bool store_impl(const std::string& path,
                    boost::archive::binary_oarchive& oa);
    std::string uncompress(TID tid);
    void compress_update(TID tid, std::string value);

    std::vector<std::string> cvalues_;
    std::string hack_last_uncompressed;
  };

  // ------------------------------------------------------ Implementation
  // -------------------------------------------------------

  //-------------------------------------------------- String
  //--------------------------------------------------------------------

  inline DeltaCompressedColumn<std::string>::DeltaCompressedColumn(
      const std::string& name, AttributeType db_type)
      : CompressedColumn<std::string>(name, db_type, DELTA_COMPRESSED) {}

  inline DeltaCompressedColumn<std::string>::~DeltaCompressedColumn() {}

  inline bool DeltaCompressedColumn<std::string>::insert(
      const boost::any& new_value) {
    if (new_value.empty()) {
      return false;
    }
    if (typeid(std::string) == new_value.type()) {
      std::string value = boost::any_cast<std::string>(new_value);
      return insert(value);
    }
    return false;
  }

  inline bool DeltaCompressedColumn<std::string>::insert(
      const std::string& new_value) {
    compress_update(cvalues_.size(), new_value);
    return true;
  }

  inline void DeltaCompressedColumn<std::string>::compress_update(
      TID tid, std::string value) {
    std::string n_string = value;
    uchar count = 0;
    if (tid > 0) {
      std::string prev_string = uncompress(tid - 1);

      for (uint i = 0; i < value.size() && i < prev_string.size(); i++) {
        if (value[i] == prev_string[i]) {
          count++;
        } else {
          break;
        }
      }
      n_string.erase(0, count);
    }
    n_string.push_back(count);
    // n_string.shrink_to_fit();
    if (tid == cvalues_.size()) {
      cvalues_.push_back(n_string);
    } else {
      cvalues_[tid] = n_string;
    }
  }

  inline std::string DeltaCompressedColumn<std::string>::uncompress(TID tid) {
    std::string complete_string = cvalues_[tid];
    // uchar last_pre_length = complete_string.back();
    uchar last_pre_length = *complete_string.rbegin();
    // complete_string.pop_back();
    complete_string.erase(complete_string.end() - 1);
    for (int i = tid - 1; i >= 0; i--) {
      std::string last_string = cvalues_[i];
      // uchar pre_length = last_string.back();
      uchar pre_length = *last_string.rbegin();
      if (last_pre_length > pre_length) {
        uchar diff = last_pre_length - pre_length;
        complete_string.insert(0, last_string.substr(0, diff));
        last_pre_length = pre_length;
      }
      if (pre_length == 0) {
        break;
      }
    }
    return complete_string;
  }

  template <typename InputIterator>
  inline bool DeltaCompressedColumn<std::string>::insert(InputIterator first,
                                                         InputIterator last) {
    for (; first != last; ++first) {
      if (!insert(*first)) {
        return false;
      }
    }
    return true;
  }

  inline const boost::any DeltaCompressedColumn<std::string>::get(TID tid) {
    if (tid < cvalues_.size())
      return boost::any((*this)[tid]);
    else {
      std::cout << "fatal Error!!! Invalid TID!!! Attribute: " << this->name_
                << " TID: " << tid << std::endl;
      return boost::any();
    }
  }

  inline void DeltaCompressedColumn<std::string>::print() const throw() {
    std::cout << "| " << this->name_ << " |" << std::endl;
    std::cout << "________________________" << std::endl;
  }

  inline size_t DeltaCompressedColumn<std::string>::size() const throw() {
    return cvalues_.size();
  }

  inline const ColumnPtr DeltaCompressedColumn<std::string>::copy() const {
    return ColumnPtr(new DeltaCompressedColumn<std::string>(*this));
  }

  inline bool DeltaCompressedColumn<std::string>::update(
      TID tid, const boost::any& new_value) {
    if (new_value.empty()) {
      return false;
    }
    if (typeid(std::string) == new_value.type()) {
      std::string value = boost::any_cast<std::string>(new_value);
      if (tid < cvalues_.size() - 1) {
        std::string next_string = uncompress(tid + 1);
        compress_update(tid, value);
        compress_update(tid + 1, next_string);
      } else {
        compress_update(tid, value);
      }
      return true;
    } else {
      std::cout << "Fatal Error!!! Typemismatch for column " << this->name_
                << std::endl;
      return false;
    }
  }

  inline bool DeltaCompressedColumn<std::string>::update(
      PositionListPtr tids, const boost::any& new_value) {
    if (!tids || new_value.empty()) {
      return false;
    }
    if (typeid(std::string) == new_value.type()) {
      for (unsigned int i = 0; i < tids->size(); i++) {
        TID tid = (*tids)[i];
        if (!update(tid, new_value)) {
          return false;
        }
      }
      return true;
    } else {
      std::cout << "Fatal Error!!! Typemismatch for column " << this->name_
                << std::endl;
    }
    return false;
  }

  inline bool DeltaCompressedColumn<std::string>::remove(TID tid) {
    if (tid + 1 < cvalues_.size()) {
      std::string next_string = uncompress(tid + 1);
      cvalues_.erase(cvalues_.begin() + tid);
      compress_update(tid, next_string);
    } else if (tid == cvalues_.size() - 1) {
      cvalues_.pop_back();
    }
    return true;
  }

  inline bool DeltaCompressedColumn<std::string>::remove(PositionListPtr tids) {
    if (!tids || tids->empty()) {
      return false;
    }

    // std::sort(tids->begin(), tids->end());

    for (uint i = tids->size(); i > 0; --i) {
      if (!remove((*tids)[i - 1])) {
        return false;
      }
    }

    return true;
  }

  inline bool DeltaCompressedColumn<std::string>::clearContent() {
    cvalues_.clear();
    return true;
  }

  inline bool DeltaCompressedColumn<std::string>::store_impl(
      const std::string& path, boost::archive::binary_oarchive& oa) {
    oa << cvalues_;
    return true;
  }

  inline bool DeltaCompressedColumn<std::string>::load_impl(
      const std::string& path, boost::archive::binary_iarchive& ia) {
    ia >> cvalues_;
    return true;
  }

  inline std::string& DeltaCompressedColumn<std::string>::operator[](
      const TID index) {
    hack_last_uncompressed = uncompress(index);
    return hack_last_uncompressed;
  }

  inline size_t DeltaCompressedColumn<std::string>::getSizeinBytes() const
      throw() {
    size_t size_in_bytes = 0;
    for (uint i = 0; i < cvalues_.size(); ++i) {
      size_in_bytes += cvalues_[i].capacity();
    }
    return size_in_bytes;
  }

  //----------------------------------------------- INT, FLOAT
  //----------------------------------------------------------------

  template <class T>
  DeltaCompressedColumn<T>::DeltaCompressedColumn(const std::string& name,
                                                  AttributeType db_type)
      : CompressedColumn<T>(name, db_type, DELTA_COMPRESSED) {
    size_ = 0;
  }

  template <class T>
  DeltaCompressedColumn<T>::~DeltaCompressedColumn() {}

  template <class T>
  bool DeltaCompressedColumn<T>::insert(const boost::any& new_value) {
    if (new_value.empty()) {
      return false;
    }
    if (typeid(T) == new_value.type()) {
      T value = boost::any_cast<T>(new_value);

      uint* x = reinterpret_cast<uint*>(&value);
      compress_insert(*x);

      return true;
    }
    return false;
  }

  template <class T>
  bool DeltaCompressedColumn<T>::insert(const T& new_value) {
    T value = new_value;

    uint* x = reinterpret_cast<uint*>(&value);
    compress_insert(*x);

    return true;
  }

  template <typename T>
  template <typename InputIterator>
  bool DeltaCompressedColumn<T>::insert(InputIterator first,
                                        InputIterator last) {
    for (; first != last; ++first) {
      if (!insert(*first)) {
        return false;
      }
    }
    return true;
  }

  template <class T>
  const boost::any DeltaCompressedColumn<T>::get(TID tid) {
    if (tid < size_) {
      T uncompressed = uncompress(tid);
      T ret = *reinterpret_cast<T*>(&uncompressed);

      return boost::any(ret);
    } else {
      std::cout << "fatal Error!!! Invalid TID!!! Attribute: " << this->name_
                << " TID: " << tid << std::endl;
      return boost::any();
    }
  }

  template <class T>
  void DeltaCompressedColumn<T>::print() const throw() {
    std::cout << "| " << this->name_ << " |" << std::endl;
    std::cout << "________________________" << std::endl;
  }

  template <class T>
  size_t DeltaCompressedColumn<T>::size() const throw() {
    return size_;
  }

  template <class T>
  const ColumnPtr DeltaCompressedColumn<T>::copy() const {
    return ColumnPtr(new DeltaCompressedColumn<T>(*this));
  }

  template <class T>
  bool DeltaCompressedColumn<T>::update(TID tid, const boost::any& new_value) {
    if (new_value.empty()) {
      return false;
    }
    if (typeid(T) == new_value.type()) {
      T value = boost::any_cast<T>(new_value);

      unsigned int* x = reinterpret_cast<unsigned int*>(&value);
      compress_update(tid, *x);

      return true;
    } else {
      std::cout << "Fatal Error!!! Typemismatch for column " << this->name_
                << std::endl;
      return false;
    }
  }

  template <class T>
  bool DeltaCompressedColumn<T>::update(PositionListPtr tids,
                                        const boost::any& new_value) {
    if (!tids || new_value.empty()) {
      return false;
    }
    if (typeid(T) == new_value.type()) {
      T value = boost::any_cast<T>(new_value);
      uint* x = reinterpret_cast<uint*>(&value);

      for (uint i = 0; i < tids->size(); i++) {
        TID tid = (*tids)[i];

        compress_update(tid, *x);
      }
      return true;
    } else {
      std::cout << "Fatal Error!!! Typemismatch for column " << this->name_
                << std::endl;
    }
    return false;
  }

  template <class T>
  bool DeltaCompressedColumn<T>::remove(TID tid) {
    compress_delete(tid);
    return true;
  }

  template <class T>
  bool DeltaCompressedColumn<T>::remove(PositionListPtr tids) {
    if (!tids || tids->empty()) {
      return false;
    }

    // std::sort(tids->begin(), tids->end());

    for (uint i = tids->size(); i > 0; --i) {
      compress_delete((*tids)[i - 1]);
    }
    return true;
  }

  template <class T>
  bool DeltaCompressedColumn<T>::clearContent() {
    cvalues_.clear();
    size_ = 0;
    return true;
  }

  template <class T>
  bool DeltaCompressedColumn<T>::store_impl(
      const std::string& path_to_table_dir,
      boost::archive::binary_oarchive& oa) {
    oa << cvalues_;
    return true;
  }

  template <class T>
  bool DeltaCompressedColumn<T>::load_impl(
      const std::string& path_to_table_dir,
      boost::archive::binary_iarchive& ia) {
    ia >> cvalues_;
    size_ = 0;
    for (uint i = 0; i < cvalues_.size();) {
      decode(i);
      size_++;
    }
    return true;
  }

  template <class T>
  T& DeltaCompressedColumn<T>::operator[](const TID index) {
    T uncompressed = uncompress(index);
    hack_last_uncompressed = *reinterpret_cast<T*>(&uncompressed);
    return hack_last_uncompressed;
  }

  template <class T>
  size_t DeltaCompressedColumn<T>::getSizeinBytes() const throw() {
    // vector content + size_ variable
    return (cvalues_.capacity() * sizeof(unsigned char)) + sizeof(uint);
  }

  template <class T>
  void DeltaCompressedColumn<T>::compress_insert(uint new_value) {
    uint delta;
    if (cvalues_.empty()) {
      delta = new_value;
    } else {
      // determine delta
      delta = new_value ^ uncompress(size_ - 1);
    }
    // pass delta to char vector
    while (delta > 127) {
      cvalues_.push_back(128 | (delta & 127));
      delta >>= 7;
    }
    cvalues_.push_back(255 & delta);
    size_++;
  }

  template <class T>
  void DeltaCompressedColumn<T>::compress_update(TID tid,
                                                 const uint& new_value_) {
    if (tid < 0 || tid >= size_) {
      throw std::out_of_range("error");
    }
    uint i = 0;

    uint old_value = decode(i);
    uint old_delta;
    uint new_delta;

    if (tid > 0) {
      uint prev_value = old_value;
      for (uint j = 1; j < tid; j++) {
        old_delta = decode(i);
        prev_value ^= old_delta;
      }
      uint tempi = i;
      old_delta = decode(tempi);

      old_value = prev_value ^ old_delta;
      new_delta = prev_value ^ new_value_;

      compress_update_part(i, old_delta, new_delta);
    } else {
      uint tempi = 0;
      compress_update_part(tempi, old_value, new_value_);
    }

    if (i < cvalues_.size()) {
      uint tempi = i;
      old_delta = decode(tempi);
      new_delta = new_value_ ^ (old_value ^ old_delta);

      compress_update_part(i, old_delta, new_delta);
    }
  }

  template <class T>
  void DeltaCompressedColumn<T>::compress_delete(TID tid) {
    if (tid < 0 || tid >= size_) {
      throw std::out_of_range("error");
    }

    if (tid == size_ - 1) {
      cvalues_.pop_back();
      if (cvalues_.size() > 0) {
        while (cvalues_.back() >= 128) {
          cvalues_.pop_back();
        }
      }
    } else {
      uint i = 0;
      for (uint j = 0; j < tid; j++) {
        decode(i);
      }
      uint tempi = i;
      uint old_delta = decode(tempi);
      uint old_delta2 = decode(tempi);

      uint new_delta = old_delta ^ old_delta2;

      uint old_length = encoded_length(old_delta);
      cvalues_.erase(cvalues_.begin() + i, cvalues_.begin() + i + old_length);

      if (i < cvalues_.size()) {
        compress_update_part(i, old_delta2, new_delta);
      }
    }
    size_--;
  }

  template <class T>
  void DeltaCompressedColumn<T>::compress_update_part(uint& i, uint old_delta,
                                                      uint new_delta) {
    uint old_length = encoded_length(old_delta);
    uint new_length = encoded_length(new_delta);

    int delta_length = new_length - old_length;
    if (delta_length != 0) {
      if (delta_length < 0) {
        cvalues_.erase(cvalues_.begin() + i,
                       cvalues_.begin() + i - delta_length);
      } else {
        cvalues_.insert(cvalues_.begin() + i, delta_length, 0);
      }
    }

    while (new_delta > 127) {
      cvalues_[i++] = (128 | (new_delta & 127));
      new_delta >>= 7;
    }
    cvalues_[i++] = (255 & new_delta);
  }

  template <class T>
  uint DeltaCompressedColumn<T>::uncompress(const uint index) {
    if (index < 0 || index >= size_) {
      throw std::out_of_range("error");
    }

    uint i = 0;
    uint result = decode(i);
    for (uint j = 1; j <= index; j++) {
      uint delta = decode(i);
      result ^= delta;
    }

    return result;
  }

  template <class T>
  uint DeltaCompressedColumn<T>::decode(uint& index) {
    uint result = 0;
    int shift = 0;

    uchar temp;
    do {
      temp = cvalues_[index++];
      result += (temp & 127) << shift;
      shift += 7;
    } while (temp > 127);

    return result;
  }

  template <class T>
  const uint DeltaCompressedColumn<T>::encoded_length(const uint& cvalue) {
    uint val = cvalue;
    uint length = 0;
    do {
      val >>= 7;
      length++;
    } while (val > 0);
    return length;
  }
}
