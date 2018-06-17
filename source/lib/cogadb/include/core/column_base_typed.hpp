
#pragma once

#include <core/base_column.hpp>
#include <iostream>
#include <vector>

#include <algorithm>
#include <functional>
#include <utility>
//#include <vector>

//#include <unordered_map>
#include <boost/any.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/unordered_map.hpp>
// TBB includes
#include <tbb/parallel_sort.h>
#include <tbb/task_scheduler_init.h>
#include <core/foreign_key_constraint.hpp>
#include "base_table.hpp"
#include "tbb/parallel_scan.h"

#include <util/begin_ptr.hpp>
#include <util/time_measurement.hpp>

#include <statistics/column_statistics.hpp>

//#include <core/column.hpp>

namespace CoGaDB {
  /*!
   *
   *
   *  \brief     This class represents a column with type T, is the base class
   *for all typed column classes and allows a uniform handling of columns of a
   *certain type T.
   *  \details   This class is indentended to be a base class, so it has a
   *virtual destruktor and pure virtual methods, which need to be implemented in
   *a derived class.
   * 				Furthermore, it declares pure virtual methods to
   *allow
   *a
   *generic handling of typed columns, e.g., operator[]. All algorithms can be
   *applied to a typed
   * 				column, because of this operator. This abstracts
   *from
   *a
   *columns implementation detail, e.g., whether they are compressed or not.
   *  \author    Sebastian Breß
   *  \version   0.2
   *  \date      2013
   *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
   *http://www.gnu.org/licenses/lgpl-3.0.txt
   */

  template <typename T>
  class Column;

  template <class T>
  class ColumnBaseTyped : public ColumnBase {
   public:
    typedef boost::unordered_multimap<T, TID, boost::hash<T>, std::equal_to<T> >
        HashTable;
    typedef boost::shared_ptr<std::vector<T> > TypedVectorPtr;
    typedef Column<T> DenseValueColumn;
    typedef boost::shared_ptr<DenseValueColumn> DenseValueColumnPtr;
    // typedef ExtendedColumnStatistics<T> ExtendedColumnStatistics;
    // typedef boost::shared_ptr<ColumnBaseTyped> ColumnPtr;
    /***************** constructors and destructor *****************/
    ColumnBaseTyped(const std::string& name, AttributeType db_type, ColumnType);
    virtual ~ColumnBaseTyped();

    virtual bool insert(const boost::any& new_Value) = 0;
    virtual bool insert(const T& new_Value) = 0;
    virtual bool update(TID tid, const boost::any& new_value) = 0;
    virtual bool update(PositionListPtr tid, const boost::any& new_value) = 0;

    virtual bool remove(TID tid) = 0;
    // assumes tid list is sorted ascending
    virtual bool remove(PositionListPtr tid) = 0;

    // this function should not be overwritten by a derived class!
    bool append(ColumnPtr col);
    virtual bool append(boost::shared_ptr<ColumnBaseTyped<T> > typed_col);
    virtual bool clearContent() = 0;

    virtual const boost::any get(TID tid) = 0;
    virtual std::string getStringValue(TID tid);
    // virtual const boost::any* const getRawData()=0;
    virtual void print() const throw() = 0;
    virtual size_t size() const throw() = 0;
    virtual size_t getSizeinBytes() const throw() = 0;
    const VectorPtr getVector(ColumnPtr col, BlockIteratorPtr it) const;

    virtual const ColumnPtr changeCompression(const ColumnType& col_type) const;
    virtual const ColumnPtr copy() const = 0;
    virtual const ColumnPtr copy(const hype::ProcessingDeviceMemoryID&) const;
    /*! \brief decompresses column if neccessary or fetch values using gather
     *  if required and copy it into a dense value column*/
    virtual const DenseValueColumnPtr copyIntoDenseValueColumn(
        const ProcessorSpecification& proc_spec) const = 0;

    virtual const DoubleDenseValueColumnPtr convertToDenseValueDoubleColumn(
        const ProcessorSpecification& proc_spec) const;

    virtual const StringDenseValueColumnPtr convertToDenseValueStringColumn()
        const;

    virtual const ColumnPtr decompress(
        const ProcessorSpecification& proc_spec) const;

    virtual hype::ProcessingDeviceMemoryID getMemoryID() const;
    virtual const ColumnPtr gather(PositionListPtr tid_list,
                                   const GatherParam&) = 0;
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
    static void hash_join_pruning_thread(
        ColumnBaseTyped<T>* join_column, HashTable* hashtable,
        TID* join_tids_table1, TID* join_tids_table2, size_t thread_id,
        size_t number_of_threads, size_t* result_size);
    static void join_write_result_chunk_thread(
        ColumnBaseTyped<T>* join_column, TID* join_tids_table1,
        TID* join_tids_table2, TID* join_tids_result_table1,
        TID* join_tids_result_table2, size_t thread_id,
        size_t number_of_threads, TID begin_index_result, TID end_index_result);
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

    virtual bool isMaterialized() const throw() = 0;
    virtual bool isCompressed() const throw() = 0;
    virtual const ColumnPtr materialize() throw() = 0;
    virtual bool isApproximatelyEqual(ColumnPtr column);
    virtual bool is_equal(ColumnPtr column);
    virtual int compareValuesAtIndexes(TID id1, TID id2);

    TypedVectorPtr copyIntoPlainVector();

    /* ATTENTION: after setting integrity constraints, the result of a write
     * operation is undefined! This follows the typical bulk load and
     * analyize workflow of OLAP. Hence, we do currently not implement
     * integrity checks when inserting new data. First import your data and then
     * set the constraints! */
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
    virtual T& operator[](const TID index) = 0;
    virtual bool operator==(ColumnBaseTyped<T>& column);
    virtual bool operator==(ColumnBase& column);

   protected:
    bool checkUniqueness();
    bool checkReferentialIntegrity(ColumnPtr primary_key_col);

    bool has_primary_key_constraint_;
    bool has_foreign_key_constraint_;
    ForeignKeyConstraint fk_constr_;
    ExtendedColumnStatistics<T> statistics_;
    /* The following fields do not to be loaded or stored,
     * but are passed by the constructor or the load function.
     * We no not save the path to tables or columns, because
     * this would make it difficult to move databases around
     * in the file system or to use them on other machines. */
    std::string path_to_table_dir_;
    std::string path_to_column_;

   private:
    virtual bool load_impl(const std::string& path,
                           boost::archive::binary_iarchive& ia) = 0;
    virtual bool store_impl(const std::string& path,
                            boost::archive::binary_oarchive& oa) = 0;
  };

}  // end namespace CogaDB
