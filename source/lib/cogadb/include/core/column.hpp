#pragma once

#include <core/column_base_typed.hpp>
#include <fstream>
#include <iostream>

#include <stdint.h>

#include <util/begin_ptr.hpp>

#include <hardware_optimizations/simd_acceleration.hpp>

#include <core/memory_allocator.hpp>

namespace CoGaDB {

  const size_t DEFAULT_COLUMN_SIZE = 1000;

  bool isCPUMemory(const hype::ProcessingDeviceMemoryID& mem_id);
  bool isGPUMemory(const hype::ProcessingDeviceMemoryID& mem_id);

  template <typename T>
  class Column : public ColumnBaseTyped<T> {
    typedef typename MemoryAllocator<T>::MemoryAllocatorPtr MemoryAllocatorPtr;

   public:
    typedef T value_type;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef T* iterator;
    typedef T* const const_iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;

    /***************** constructors and destructor *****************/
    Column(const std::string& name, AttributeType db_type,
           hype::ProcessingDeviceMemoryID mem_id = hype::PD_Memory_0);

    // fill constructor
    Column(const std::string& name, AttributeType db_type,
           size_t number_of_elements, const T& val = T(),
           hype::ProcessingDeviceMemoryID mem_id = hype::PD_Memory_0);

    // range	constructor
    template <class InputIterator>
    Column(const std::string& name, AttributeType db_type, InputIterator first,
           InputIterator last)
        : ColumnBaseTyped<T>(name, db_type, PLAIN_MATERIALIZED),
          type_tid_comparator(),
          values_(0),
          num_elements_(0),
          buffer_size_(0),
          mem_alloc(MemoryAllocator<T>::getMemoryAllocator(hype::PD_Memory_0)) {
      assert(last >= first);
      assert(this->mem_alloc->getMemoryID() == hype::PD_Memory_0);
      size_t new_size = last - first;
      this->num_elements_ = new_size;
      if (new_size == 0) {
        new_size = 1;
      }
      if (this->capacity() < new_size) {
        this->reserve(new_size);
      }
      assert(values_ != NULL);
      std::copy(first, last, values_);
    }

    Column(const Column<T>& x);

    Column<T>& operator=(const Column<T>& other);

    ~Column();

    virtual bool insert(const boost::any& new_value);
    bool insert(const T& new_value);
    template <typename InputIterator>
    bool insert(InputIterator first, InputIterator last);

    virtual bool update(TID tid, const boost::any& new_value);
    virtual bool update(PositionListPtr tid, const boost::any& new_value);

    virtual bool remove(TID tid);
    // assumes tid list is sorted ascending
    virtual bool remove(PositionListPtr tid);

    virtual bool append(boost::shared_ptr<ColumnBaseTyped<T> > typed_col);

    virtual bool clearContent();

    virtual const boost::any get(TID tid);
    // virtual const boost::any* const getRawData();
    virtual void print() const throw();
    virtual size_t size() const throw();
    virtual size_t getSizeinBytes() const throw();

    virtual const ColumnPtr copy() const;
    virtual const ColumnPtr copy(
        const hype::ProcessingDeviceMemoryID& mem_id) const;
    virtual const typename ColumnBaseTyped<T>::DenseValueColumnPtr
    copyIntoDenseValueColumn(const ProcessorSpecification& proc_spec) const;
    virtual const DoubleDenseValueColumnPtr convertToDenseValueDoubleColumn(
        const ProcessorSpecification& proc_spec) const;
    virtual const StringDenseValueColumnPtr convertToDenseValueStringColumn()
        const;
    virtual const ColumnPtr materialize() throw();
    virtual hype::ProcessingDeviceMemoryID getMemoryID() const;
    virtual const ColumnPtr gather(PositionListPtr tid_list,
                                   const GatherParam&);

    virtual const ColumnGroupingKeysPtr createColumnGroupingKeys(
        const ProcessorSpecification& proc_spec) const;
    virtual size_t getNumberOfRequiredBits() const;

    virtual const AggregationResult aggregateByGroupingKeys(
        ColumnGroupingKeysPtr grouping_keys, const AggregationParam&);
    virtual const AggregationResult aggregate(const AggregationParam&);

    //        virtual bool store(const std::string& path);
    //        virtual bool load(const std::string& path);

    virtual bool isMaterialized() const throw();
    virtual bool isCompressed() const throw();
    virtual T* data() throw();

    virtual const PositionListPtr selection(const SelectionParam& param);
    virtual const BitmapPtr bitmap_selection(const SelectionParam& param);

#ifdef ENABLE_CDK_USAGE
    virtual const PositionListPairPtr hash_join(ColumnPtr join_column);
    virtual const PositionListPairPtr radix_join(ColumnPtr join_column);
#endif

    virtual const PositionListPairPtr join(ColumnPtr join_column,
                                           const JoinParam&);
    virtual const PositionListPairPtr join(Column<T>& join_column,
                                           const JoinParam& param);

    virtual const PositionListPtr tid_semi_join(ColumnPtr join_column,
                                                const JoinParam&);
    virtual const PositionListPtr tid_semi_join(Column<T>& join_column,
                                                const JoinParam&);

    virtual const BitmapPtr bitmap_semi_join(ColumnPtr join_column,
                                             const JoinParam&);
    virtual const BitmapPtr bitmap_semi_join(Column<T>& join_column,
                                             const JoinParam&);

    virtual const ColumnPtr column_algebra_operation(
        ColumnPtr source_column, const AlgebraOperationParam&);
    virtual const ColumnPtr column_algebra_operation(
        Column<T>& source_column, const AlgebraOperationParam&);
    virtual const ColumnPtr column_algebra_operation(
        const boost::any& value, const AlgebraOperationParam&);

    virtual const PositionListPtr sort(const SortParam& param);
    virtual const PositionListPtr sort(const SortParam& param, bool no_copy);

    /* BEGIN CONTAINER OPERATIONS*/
    virtual size_t capacity() const { return buffer_size_; }

    virtual inline T& operator[](const TID index) { return values_[index]; }

    virtual inline const T& operator[](const TID index) const {
      return values_[index];
    }

    virtual void reserve(size_t new_capacity);
    virtual void resize(size_t new_size);

    virtual void push_back(const T& val) __attribute__((always_inline)) {
      if (num_elements_ < buffer_size_ - 1) {
        // insert element
        values_[num_elements_] = val;
        num_elements_++;
      } else {
        // increase memory by 70%
        size_t new_size = 1.7 * buffer_size_;
        // handle case where buffer_size is 1,
        // then the new_size is equal to the buffer_size
        if (new_size == buffer_size_) {
          new_size = 2 * buffer_size_;
        }
        // tid_ptr is automatically updated by reserve
        Column<T>::reserve(new_size);
        assert(num_elements_ < buffer_size_);
        assert(capacity() == new_size);

        values_[num_elements_] = val;
        num_elements_++;
      }
    }

    virtual T* begin();
    virtual const T* begin() const;

    virtual T* end();
    virtual const T* end() const;

    virtual bool empty() const;
    virtual void clear();

    /* END CONTAINER OPERATIONS*/

   protected:
    virtual bool load_impl(const std::string& path,
                           boost::archive::binary_iarchive& ia);
    virtual bool store_impl(const std::string& path,
                            boost::archive::binary_oarchive& oa);

   private:
    struct Type_TID_Comparator {
      inline bool operator()(std::pair<T, TID> i, std::pair<T, TID> j) {
        return (i.first < j.first);
      }
    } type_tid_comparator;
    /*! values*/
    T* values_;
    size_t num_elements_;
    size_t buffer_size_;
    MemoryAllocatorPtr mem_alloc;
  };

  template <typename T>
  template <typename InputIterator>
  bool Column<T>::insert(InputIterator first, InputIterator last) {
    InputIterator current = first;
    while (current != last) {
      this->insert(*current);
      ++current;
    }
    return true;
  }

}  // end namespace CogaDB
