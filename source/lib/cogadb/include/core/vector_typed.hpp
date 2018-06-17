/*
 * File:   vector_typed.hpp
 * Author: sebastian
 *
 * Created on 27. Dezember 2015, 15:01
 */

#ifndef VECTOR_TYPED_HPP
#define VECTOR_TYPED_HPP

#include <core/column_base_typed.hpp>
#include <core/vector.hpp>

namespace CoGaDB {

  template <typename T>
  class Column;

  template <typename T>
  class VectorTyped : public Vector {
   public:
    typedef ColumnBaseTyped<T> ColumnTyped;
    typedef boost::shared_ptr<ColumnTyped> ColumnTypedPtr;
    typedef Column<T> DenseValueColumn;
    typedef boost::shared_ptr<DenseValueColumn> DenseValueColumnPtr;
    /***************** constructors and destructor *****************/
    VectorTyped(TID begin_index, size_t num_elements,
                ColumnTypedPtr source_column_);
    virtual ~VectorTyped();

    virtual bool insert(const boost::any& new_Value);
    virtual bool insert(const T& new_Value);
    virtual bool update(TID tid, const boost::any& new_value);
    virtual bool update(PositionListPtr tid, const boost::any& new_value);
    virtual bool append(ColumnPtr col);
    virtual bool remove(TID tid);
    // assumes tid list is sorted ascending
    virtual bool remove(PositionListPtr tid);
    virtual bool clearContent();

    virtual const boost::any get(TID tid);
    virtual std::string getStringValue(TID tid);

    virtual void print() const throw();
    virtual size_t size() const throw();
    virtual size_t getSizeinBytes() const throw();

    virtual const ColumnPtr changeCompression(const ColumnType& col_type) const;
    virtual const ColumnPtr copy() const;
    virtual const ColumnPtr copy(const hype::ProcessingDeviceMemoryID&) const;

    virtual const DenseValueColumnPtr copyIntoDenseValueColumn(
        const ProcessorSpecification& proc_spec) const;

    virtual const DoubleDenseValueColumnPtr convertToDenseValueDoubleColumn(
        const ProcessorSpecification& proc_spec) const;

    virtual const StringDenseValueColumnPtr convertToDenseValueStringColumn()
        const;

    virtual T* data() const;
    virtual uint32_t* getDictionaryCompressedKeys() const;

    virtual const ColumnPtr decompress(
        const ProcessorSpecification& proc_spec) const;

    virtual hype::ProcessingDeviceMemoryID getMemoryID() const;
    virtual const ColumnPtr gather(PositionListPtr tid_list,
                                   const GatherParam&);
    /***************** relational operations on Columns which return lookup
     * tables *****************/
    virtual const PositionListPtr sort(const SortParam& param);
    virtual const PositionListPtr selection(const SelectionParam& param);
    virtual const BitmapPtr bitmap_selection(const SelectionParam& param);
    virtual const PositionListPtr parallel_selection(
        const boost::any& value_for_comparison, const ValueComparator comp,
        unsigned int number_of_threads);
    virtual const PositionListPtr lock_free_parallel_selection(
        const boost::any& value_for_comparison, const ValueComparator comp,
        unsigned int number_of_threads);
    // join algorithms
    virtual const PositionListPairPtr hash_join(ColumnPtr join_column);
    virtual const PositionListPairPtr parallel_hash_join(
        ColumnPtr join_column, unsigned int number_of_threads);
    //        static void hash_join_pruning_thread(ColumnBaseTyped<T>*
    //        join_column, HashTable* hashtable, TID* join_tids_table1, TID*
    //        join_tids_table2, size_t thread_id, size_t number_of_threads,
    //        size_t* result_size);
    //        static void join_write_result_chunk_thread(ColumnBaseTyped<T>*
    //        join_column, TID* join_tids_table1, TID* join_tids_table2, TID*
    //        join_tids_result_table1, TID* join_tids_result_table2, size_t
    //        thread_id, size_t number_of_threads, TID begin_index_result, TID
    //        end_index_result);
    virtual const PositionListPairPtr sort_merge_join(ColumnPtr join_column);
    virtual const PositionListPairPtr nested_loop_join(ColumnPtr join_column);
    virtual const PositionListPairPtr radix_join(ColumnPtr join_column);

    virtual const PositionListPairPtr join(ColumnPtr join_column,
                                           const JoinParam&);
    virtual const PositionListPtr tid_semi_join(ColumnPtr join_column,
                                                const JoinParam&);
    virtual const BitmapPtr bitmap_semi_join(ColumnPtr join_column,
                                             const JoinParam&);
    virtual const AggregationResult aggregate(const AggregationParam&);

    virtual const ColumnPtr column_algebra_operation(
        ColumnPtr source_column, const AlgebraOperationParam&);
    virtual const ColumnPtr column_algebra_operation(
        const boost::any& value, const AlgebraOperationParam&);

    virtual const ColumnGroupingKeysPtr createColumnGroupingKeys(
        const ProcessorSpecification& proc_spec) const;
    virtual size_t getNumberOfRequiredBits() const;

    virtual const AggregationResult aggregateByGroupingKeys(
        ColumnGroupingKeysPtr grouping_keys, const AggregationParam&);

    virtual bool store(const std::string& path);
    virtual bool load(const std::string& path,
                      ColumnLoaderMode column_loader_mode);

    virtual bool isMaterialized() const throw();
    virtual bool isCompressed() const throw();
    virtual const ColumnPtr materialize() throw();
    virtual bool is_equal(ColumnPtr column);
    virtual bool isApproximatelyEqual(ColumnPtr column);
    virtual bool operator==(ColumnBase& col);
    virtual int compareValuesAtIndexes(TID id1, TID id2);

    bool setPrimaryKeyConstraint();
    bool hasPrimaryKeyConstraint() const throw();

    bool hasForeignKeyConstraint() const throw();
    bool setForeignKeyConstraint(
        const ForeignKeyConstraint& prim_foreign_key_reference);
    const ForeignKeyConstraint& getForeignKeyConstraint();

    virtual const ColumnStatistics& getColumnStatistics() const;
    virtual const ExtendedColumnStatistics<T>& getExtendedColumnStatistics()
        const;
    virtual bool computeColumnStatistics();

    /*! \brief returns type information of internal values*/
    virtual const std::type_info& type() const throw();
    /*! \brief defines operator[] for this class, which enables the user to
     * thread all typed columns as arrays.
     * \details Note that this method is pure virtual, so it has to be defined
     * in a derived class.
     * \return a reference to the value at position index
     * */
    virtual T& operator[](const TID index);
    //	virtual bool operator==(ColumnBaseTyped<T>& column);
    //        virtual bool operator==(ColumnBase& column);

    ColumnTypedPtr getSourceColumn() const;

   private:
    ColumnTypedPtr source_column_;

    struct UncompressedStringColumnCache {
      static UncompressedStringColumnCache& instance();
      const boost::shared_ptr<Column<T> > get(ColumnPtr col);

     private:
      UncompressedStringColumnCache();
      typedef std::map<boost::weak_ptr<ColumnBase>,
                       boost::shared_ptr<Column<T> > >
          Map;
      Map map_;
      boost::mutex mutex_;
    };
  };

}  // end namespace CoGaDB

#endif /* VECTOR_TYPED_HPP */
