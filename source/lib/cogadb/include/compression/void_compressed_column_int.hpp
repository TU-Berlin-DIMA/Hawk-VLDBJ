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
  class VoidCompressedColumnNumber : public CompressedColumn<T> {
   public:
    VoidCompressedColumnNumber(const std::string& name, AttributeType db_type);
    ~VoidCompressedColumnNumber();

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

    const PositionListPtr selection(const SelectionParam& param);

    const ColumnPtr copy() const;

    const typename ColumnBaseTyped<T>::DenseValueColumnPtr
    copyIntoDenseValueColumn(const ProcessorSpecification& proc_spec) const;

    //        const ColumnPtr gather(PositionListPtr tid_list, const
    //        GatherParam&);

    //        bool store(const std::string& path);
    //        bool load(const std::string& path);

    T& operator[](const TID index);

   private:
    bool load_impl(const std::string& path,
                   boost::archive::binary_iarchive& ia);
    bool store_impl(const std::string& path,
                    boost::archive::binary_oarchive& oa);
    size_t number_of_rows_;
    T hack_last_returned_;
  };
}
