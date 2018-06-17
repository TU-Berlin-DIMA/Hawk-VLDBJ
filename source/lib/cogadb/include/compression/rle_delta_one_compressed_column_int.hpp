#pragma once

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <core/compressed_column.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <util/begin_ptr.hpp>

namespace CoGaDB {

  template <class T>
  class RLEDeltaOneCompressedColumnNumber : public CompressedColumn<T> {
   public:
    RLEDeltaOneCompressedColumnNumber(const std::string& name,
                                      AttributeType db_type);
    ~RLEDeltaOneCompressedColumnNumber();

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

    //        virtual const PositionListPtr selection(const boost::any&
    //        value_for_comparison, const ValueComparator comp);
    //        virtual const PositionListPtr selection(ColumnPtr
    //        comparison_column, const ValueComparator comp);
    //        virtual const PositionListPtr selection(const SelectionParam&
    //        param);

    const boost::any get(TID tid);
    void print() const throw();
    size_t size() const throw();
    size_t getSizeinBytes() const throw();

    const ColumnPtr copy() const;

    const typename ColumnBaseTyped<T>::DenseValueColumnPtr
    copyIntoDenseValueColumn(const ProcessorSpecification& proc_spec) const;

    T& operator[](const TID index);

   private:
    bool load_impl(const std::string& path,
                   boost::archive::binary_iarchive& ia);
    bool store_impl(const std::string& path,
                    boost::archive::binary_oarchive& oa);
    std::vector<T> values_;
    std::vector<unsigned char> count_;
    T hack_last_uncompressed;
  };

  /***************** Start of Implementation Section ******************/

  template <typename T>
  template <typename InputIterator>
  bool RLEDeltaOneCompressedColumnNumber<T>::insert(InputIterator first,
                                                    InputIterator last) {
    for (InputIterator it = first; it != last; it++) {
      insert(*it);
    }
    return true;
  }
}
