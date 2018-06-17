
#pragma once

#include <core/column.hpp>
#include <core/column_base_typed.hpp>

namespace CoGaDB {

  /*!
   *
   *
   *  \brief     This class represents a compressed column with type T, is the
   *base class for all compressed typed column classes and allows a uniform
   *handling of compressed columns of a certain type T.
   *  \details   This class is indentended to be a base class, so it has a
   *virtual destruktor and pure virtual methods, which need to be implemented in
   *a derived class.
   *  \author    Sebastian Breß
   *  \version   0.2
   *  \date      2013
   *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
   *http://www.gnu.org/licenses/lgpl-3.0.txt
   */
  template <class T>
  class CompressedColumn : public ColumnBaseTyped<T> {
   public:
    /***************** constructors and destructor *****************/
    CompressedColumn(const std::string& name, AttributeType db_type,
                     ColumnType);
    CompressedColumn(const CompressedColumn&);
    CompressedColumn& operator=(const CompressedColumn&);
    virtual ~CompressedColumn();

    virtual bool insert(const boost::any& new_Value) = 0;
    virtual bool insert(const T& new_value) = 0;

    virtual bool update(TID tid, const boost::any& new_value) = 0;
    virtual bool update(PositionListPtr tid, const boost::any& new_value) = 0;

    virtual bool remove(TID tid) = 0;
    // assumes tid list is sorted ascendingd
    virtual bool remove(PositionListPtr tid) = 0;
    virtual bool clearContent() = 0;

    virtual const boost::any get(TID tid) = 0;
    // virtual const boost::any* const getRawData()=0;
    virtual void print() const throw() = 0;
    virtual size_t size() const throw() = 0;
    virtual size_t getSizeinBytes() const throw() = 0;

    virtual const ColumnPtr copy() const = 0;
    virtual const ColumnPtr copy(const hype::ProcessingDeviceMemoryID&) const;
    virtual const typename ColumnBaseTyped<T>::DenseValueColumnPtr
    copyIntoDenseValueColumn(const ProcessorSpecification& proc_spec) const;
    const ColumnPtr gather(PositionListPtr tid_list, const GatherParam&);

    const ColumnPtr getDecompressedColumn(
        const ProcessorSpecification& proc_spec);

    virtual bool isMaterialized() const throw();

    virtual bool isCompressed() const throw();
    const ColumnPtr materialize() throw();

    uint32_t* getGroupingIds() throw();

    virtual T& operator[](const TID index) = 0;

   private:
    virtual const ColumnPtr getDecompressedColumn_impl(
        const ProcessorSpecification& proc_spec);
    boost::mutex decompressed_column_access_mutex_;
    ColumnPtr decompressed_;
  };

  typedef CompressedColumn<int> CompressedIntegerColumn;
  typedef CompressedColumn<float> CompressedFloatColumn;
  typedef CompressedColumn<double> CompressedDoubleColumn;
  typedef CompressedColumn<std::string> CompressedStringColumn;

  /***************** Start of Implementation Section ******************/

  template <class T>
  CompressedColumn<T>::CompressedColumn(const std::string& name,
                                        AttributeType db_type,
                                        ColumnType column_type)
      : ColumnBaseTyped<T>(name, db_type, column_type),
        decompressed_column_access_mutex_(),
        decompressed_() {}

  template <class T>
  CompressedColumn<T>::CompressedColumn(const CompressedColumn& col)
      : ColumnBaseTyped<T>(col),
        decompressed_column_access_mutex_(),
        decompressed_(col.decompressed_) {}

  template <class T>
  typename CompressedColumn<T>::CompressedColumn& CompressedColumn<T>::
  operator=(const CompressedColumn& other) {
    ColumnBaseTyped<T>::operator=(other);
    if (this != &other) {
      this->decompressed_ = other.decompressed_;
      /* we do not assign the value of the mutex! */
      // this->decompressed_column_access_mutex_;
    }
    return *this;
  }

  template <class T>
  CompressedColumn<T>::~CompressedColumn() {}

  template <class T>
  const ColumnPtr CompressedColumn<T>::getDecompressedColumn(
      const ProcessorSpecification& proc_spec) {
    boost::lock_guard<boost::mutex> lock(decompressed_column_access_mutex_);

    if (decompressed_) {
      return decompressed_;
    } else {
      decompressed_ = this->getDecompressedColumn_impl(proc_spec);
      return decompressed_;
    }
  }

  template <class T>
  const ColumnPtr CompressedColumn<T>::getDecompressedColumn_impl(
      const ProcessorSpecification& proc_spec) {
    return this->copyIntoDenseValueColumn(proc_spec);
  }

  template <class T>
  bool CompressedColumn<T>::isMaterialized() const throw() {
    return false;
  }

  template <class T>
  bool CompressedColumn<T>::isCompressed() const throw() {
    return true;
  }

  template <class T>
  const ColumnPtr CompressedColumn<T>::materialize() throw() {
    Column<T>* result = new Column<T>(this->name_, this->db_type_);
    //            std::vector<T>& data = result->getContent();
    //            data.resize(this->size());
    result->resize(this->size());
    T* data = result->data();
    for (unsigned int i = 0; i < this->size(); i++) {
      data[i] = (*this)[i];
    }
    return ColumnPtr(result);
  }

  template <class T>
  const ColumnPtr CompressedColumn<T>::copy(
      const hype::ProcessingDeviceMemoryID& mem_id) const {
    return ColumnBaseTyped<T>::copy(mem_id);
  }

  template <class T>
  const typename ColumnBaseTyped<T>::DenseValueColumnPtr
  CompressedColumn<T>::copyIntoDenseValueColumn(
      const ProcessorSpecification& proc_spec) const {
    assert(proc_spec.proc_id == hype::PD0);

    typedef
        typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
    typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
    DenseValueColumnPtr result(
        new DenseValueColumn(this->getName(), this->getType()));
    size_t num_elements = this->size();
    CompressedColumn<T>* this_col = const_cast<CompressedColumn<T>*>(this);
    for (size_t i = 0; i < num_elements; ++i) {
      result->insert((*this_col)[i]);
    }
    return result;
  }

  template <class T>
  const ColumnPtr CompressedColumn<T>::gather(PositionListPtr tid_list,
                                              const GatherParam& param) {
    typedef
        typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

    DenseValueColumnPtr dense_column =
        this->copyIntoDenseValueColumn(param.proc_spec);
    assert(dense_column != NULL);

    return dense_column->gather(tid_list, param);
  }

  /***************** End of Implementation Section ******************/

}  // end namespace CogaDB
