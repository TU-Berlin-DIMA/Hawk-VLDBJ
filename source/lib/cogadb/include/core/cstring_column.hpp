/*
 * File:   cstring_column.hpp
 * Author: sebastian
 *
 * Created on 23. August 2015, 12:43
 */

#ifndef CSTRING_COLUMN_HPP
#define CSTRING_COLUMN_HPP

#include <core/column.hpp>
#include <core/memory_allocator.hpp>

namespace CoGaDB {

  class StringHeap;
  typedef boost::shared_ptr<StringHeap> StringHeapPtr;

  enum StringHeapType { FIXED_LENGTH_STRING_HEAP, VARIABLE_LENGTH_STRING_HEAP };

  class StringHeap {
   public:
    /* \brief initialize basic members */
    StringHeap(const StringHeapType heap_type, const size_t max_string_length,
               const hype::ProcessingDeviceMemoryID mem_id = hype::PD_Memory_0);
    /* \brief returns an array of char* that point to a string in the heap
     The order of pointers is the same as the order the pointers where inserted
     into the heap */
    virtual char* getPointerArray() = 0;
    /* \brief initializes an array such that each pointer in the array
     * points to an element on the heap. The order will be the same as
     the order strings were inserted into the heap. */
    virtual bool initStringArray(
        char** c_string_ptr_array, size_t num_elements,
        const hype::ProcessingDeviceMemoryID& mem_id) = 0;
    /* \brief inserts value \param val at the end of the heap and returns a
     pointer to the string on the heap on success and NULL when an error occured
     */
    virtual char* push_back(const char* const val, bool& heap_memory_moved) = 0;

    virtual void clear() = 0;

    virtual const StringHeapPtr copy() const = 0;

    virtual const StringHeapPtr copy(
        const hype::ProcessingDeviceMemoryID& mem_id) const = 0;

    virtual void reserve(size_t new_capacity, bool& heap_memory_moved) = 0;

    virtual void resize(size_t new_size, bool& heap_memory_moved) = 0;

    /* \brief loads heap from disk */
    bool load(const std::string& base_path, const std::string& column_name,
              boost::archive::binary_iarchive& ia);
    /* \brief stores heap on disk */
    bool store(const std::string& base_path, const std::string& column_name,
               boost::archive::binary_oarchive& oa);

    virtual size_t getSizeinBytes() const throw() = 0;

    StringHeapType getStringHeapType() const;

    size_t getMaxStringLength() const;

    hype::ProcessingDeviceMemoryID getMemoryID() const;

    size_t size() const;

    virtual void print() const = 0;

   private:
    virtual bool load_impl(const std::string& base_path,
                           const std::string& column_name,
                           boost::archive::binary_iarchive& ia) = 0;
    virtual bool store_impl(const std::string& base_path,
                            const std::string& column_name,
                            boost::archive::binary_oarchive& oa) = 0;

   protected:
    //        Column<char> data_;
    size_t num_elements_;

   private:
    /* a string heap may only be copied by the virtual copy constructor
     * method "copy" */
    //        StringHeap(const StringHeap&);
    //        StringHeap& operator=(const StringHeap&);
    size_t max_string_length_;
    //        size_t number_of_elements_;
    hype::ProcessingDeviceMemoryID mem_id_;
    StringHeapType heap_type_;
  };

  class FixedLengthStringHeap : public StringHeap {
   public:
    FixedLengthStringHeap(
        const size_t max_string_length,
        const hype::ProcessingDeviceMemoryID mem_id = hype::PD_Memory_0);
    FixedLengthStringHeap(
        const size_t max_string_length, const size_t num_elements,
        const hype::ProcessingDeviceMemoryID mem_id = hype::PD_Memory_0);
    char* getPointerArray();
    bool initStringArray(char** c_string_ptr_array, size_t num_elements,
                         const hype::ProcessingDeviceMemoryID& mem_id);
    char* push_back(const char* const val, bool& heap_memory_moved);

    void clear();

    const StringHeapPtr copy() const;

    const StringHeapPtr copy(
        const hype::ProcessingDeviceMemoryID& mem_id) const;

    void reserve(size_t new_capacity, bool& heap_memory_moved);

    void resize(size_t new_size, bool& heap_memory_moved);

    bool load_impl(const std::string& base_path, const std::string& column_name,
                   boost::archive::binary_iarchive& ia);
    bool store_impl(const std::string& base_path,
                    const std::string& column_name,
                    boost::archive::binary_oarchive& oa);
    size_t getSizeinBytes() const throw();
    virtual void print() const;

    boost::shared_ptr<Column<char> > data_;
  };

  const StringHeapPtr createStringHeap(
      StringHeapType type, size_t max_string_length,
      const hype::ProcessingDeviceMemoryID mem_id);

  class CStringColumn : public Column<char*> {
    typedef MemoryAllocator<char*>::MemoryAllocatorPtr MemoryAllocatorPtr;

   public:
    typedef char* value_type;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef char** iterator;
    typedef char** const const_iterator;
    typedef ptrdiff_t difference_type;
    typedef size_t size_type;

    /***************** constructors and destructor *****************/
    CStringColumn(const std::string& name, size_t maximal_string_length,
                  hype::ProcessingDeviceMemoryID mem_id = hype::PD_Memory_0);

    CStringColumn(const CStringColumn& x);

    CStringColumn& operator=(const CStringColumn& other);

    ~CStringColumn();

    virtual bool insert(const boost::any& new_value);
    bool insert(const value_type& new_value);
    //        template <typename InputIterator>
    //        bool insert(InputIterator first, InputIterator last);

    virtual bool update(TID tid, const boost::any& new_value);
    virtual bool update(PositionListPtr tid, const boost::any& new_value);

    virtual bool remove(TID tid);
    // assumes tid list is sorted ascending
    virtual bool remove(PositionListPtr tid);
    virtual bool clearContent();

    virtual const boost::any get(TID tid);
    //        //virtual const boost::any* const getRawData();
    //        virtual void print() const throw ();
    virtual size_t size() const throw();
    virtual size_t getSizeinBytes() const throw();
    size_t getMaxStringLength() const;

    virtual const ColumnPtr copy() const;
    virtual const ColumnPtr copy(
        const hype::ProcessingDeviceMemoryID& mem_id) const;
    virtual const typename Column<char*>::DenseValueColumnPtr
    copyIntoDenseValueColumn(const ProcessorSpecification& proc_spec) const;
    virtual const DoubleDenseValueColumnPtr convertToDenseValueDoubleColumn(
        const ProcessorSpecification& proc_spec) const;
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

    virtual const PositionListPtr selection(const SelectionParam& param);
    virtual const BitmapPtr bitmap_selection(const SelectionParam& param);

#ifdef ENABLE_CDK_USAGE
    virtual const PositionListPairPtr hash_join(ColumnPtr join_column);
    virtual const PositionListPairPtr radix_join(ColumnPtr join_column);
#endif

    virtual const PositionListPairPtr join(ColumnPtr join_column,
                                           const JoinParam&);
    virtual const PositionListPairPtr join(Column<value_type>& join_column,
                                           const JoinParam& param);

    virtual const PositionListPtr tid_semi_join(ColumnPtr join_column,
                                                const JoinParam&);
    virtual const PositionListPtr tid_semi_join(Column<value_type>& join_column,
                                                const JoinParam&);

    virtual const BitmapPtr bitmap_semi_join(ColumnPtr join_column,
                                             const JoinParam&);
    virtual const BitmapPtr bitmap_semi_join(Column<value_type>& join_column,
                                             const JoinParam&);

    virtual const ColumnPtr column_algebra_operation(
        ColumnPtr source_column, const AlgebraOperationParam&);
    virtual const ColumnPtr column_algebra_operation(
        Column<value_type>& source_column, const AlgebraOperationParam&);
    virtual const ColumnPtr column_algebra_operation(
        const boost::any& value, const AlgebraOperationParam&);

    virtual const PositionListPtr sort(const SortParam& param);
    virtual const PositionListPtr sort(const SortParam& param, bool no_copy);

    /* BEGIN CONTAINER OPERATIONS*/
    virtual void reserve(size_t new_capacity);
    virtual void resize(size_t new_size);

    virtual void push_back(
        const value_type& val);  // __attribute__((always_inline)){
    void push_back(const char* val);

    virtual void clear();

    /* END CONTAINER OPERATIONS*/

    void printHeap();

   private:
    virtual bool load_impl(const std::string& path,
                           boost::archive::binary_iarchive& ia);
    virtual bool store_impl(const std::string& path,
                            boost::archive::binary_oarchive& oa);

    /*! values*/
    StringHeapPtr heap_;
  };

  //    template <typename T>
  //    template <typename InputIterator>
  //    bool CStringColumn<T>::insert(InputIterator first, InputIterator last) {
  //
  //        InputIterator current=first;
  //        while(current!=last){
  //            this->insert(*current);
  //            ++current;
  //        }
  //        return true;
  //    }

}  // end namespace CogaDB

#endif /* CSTRING_COLUMN_HPP */
