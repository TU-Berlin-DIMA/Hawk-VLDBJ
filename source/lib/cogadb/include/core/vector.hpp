/*
 * File:   vector.hpp
 * Author: sebastian
 *
 * Created on 27. Dezember 2015, 14:37
 */

#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <core/base_column.hpp>
namespace CoGaDB {

  class Vector;
  typedef boost::shared_ptr<Vector> VectorPtr;

  class BlockIterator;

  const ColumnPtr createVector(ColumnPtr col, BlockIterator& it);

  class Vector : public ColumnBase {
   public:
    Vector(const std::string& name, AttributeType db_type,
           ColumnType column_type, TID begin_index, size_t num_elements);
    virtual ~Vector();

    virtual bool insert(const boost::any& new_Value) = 0;

    virtual bool update(TID tid, const boost::any& new_Value) = 0;

    virtual bool update(PositionListPtr tids, const boost::any& new_value) = 0;

    virtual bool append(ColumnPtr col) = 0;

    virtual bool remove(TID tid) = 0;

    virtual bool remove(PositionListPtr tid) = 0;

    virtual const boost::any get(TID tid) = 0;  // not const, because operator
                                                // [] does not provide const
                                                // return type and the child
                                                // classes rely on []

    virtual std::string getStringValue(TID tid) = 0;

    virtual void print() const throw() = 0;

    virtual size_t size() const throw() = 0;

    virtual size_t getSizeinBytes() const throw() = 0;

    const VectorPtr getVector(ColumnPtr col, BlockIteratorPtr it) const;

    virtual const ColumnPtr changeCompression(
        const ColumnType& col_type) const = 0;

    virtual const ColumnPtr copy() const = 0;

    virtual const ColumnPtr copy(
        const hype::ProcessingDeviceMemoryID&) const = 0;

    virtual const DoubleDenseValueColumnPtr convertToDenseValueDoubleColumn(
        const ProcessorSpecification& proc_spec) const = 0;

    virtual const StringDenseValueColumnPtr convertToDenseValueStringColumn()
        const = 0;

    virtual const ColumnPtr decompress(
        const ProcessorSpecification& proc_spec) const = 0;

    virtual hype::ProcessingDeviceMemoryID getMemoryID() const = 0;

    virtual const ColumnPtr gather(PositionListPtr tid_list,
                                   const GatherParam&) = 0;

    virtual const ColumnPtr materialize() throw() = 0;

    virtual const PositionListPtr sort(const SortParam& param) = 0;

    virtual const PositionListPtr selection(const SelectionParam& param) = 0;

    virtual const BitmapPtr bitmap_selection(const SelectionParam& param) = 0;

    virtual const PositionListPtr parallel_selection(
        const boost::any& value_for_comparison, const ValueComparator comp,
        unsigned int number_of_threads) = 0;

    virtual const PositionListPairPtr hash_join(ColumnPtr join_column) = 0;

    virtual const PositionListPairPtr parallel_hash_join(
        ColumnPtr join_column, unsigned int number_of_threads) = 0;

    virtual const PositionListPairPtr sort_merge_join(
        ColumnPtr join_column) = 0;

    virtual const PositionListPairPtr nested_loop_join(
        ColumnPtr join_column) = 0;

    virtual const PositionListPairPtr radix_join(ColumnPtr join_column) = 0;

    virtual const PositionListPairPtr join(ColumnPtr join_column,
                                           const JoinParam&) = 0;
    virtual const PositionListPtr tid_semi_join(ColumnPtr join_column,
                                                const JoinParam&) = 0;
    virtual const BitmapPtr bitmap_semi_join(ColumnPtr join_column,
                                             const JoinParam&) = 0;

    virtual const ColumnGroupingKeysPtr createColumnGroupingKeys(
        const ProcessorSpecification& proc_spec) const = 0;
    virtual size_t getNumberOfRequiredBits() const = 0;

    virtual const AggregationResult aggregateByGroupingKeys(
        ColumnGroupingKeysPtr grouping_keys, const AggregationParam&) = 0;
    virtual const AggregationResult aggregate(const AggregationParam&) = 0;

    virtual const ColumnPtr column_algebra_operation(
        ColumnPtr source_column, const AlgebraOperationParam&) = 0;
    virtual const ColumnPtr column_algebra_operation(
        const boost::any& value, const AlgebraOperationParam&) = 0;

    virtual bool store(const std::string& path) = 0;

    virtual bool load(const std::string& path,
                      ColumnLoaderMode column_loader_mode) = 0;

    virtual bool isMaterialized() const throw() = 0;

    virtual bool isCompressed() const throw() = 0;

    virtual const std::type_info& type() const throw() = 0;

    AttributeType getType() const throw();

    void setType(const AttributeType&);

    ColumnType getColumnType() const throw();

    bool isLoadedInMainMemory() const throw();

    void setStatusLoadedInMainMemory(bool is_loaded) throw();

    const std::string getName() const throw();

    void setName(const std::string& value) throw();

    virtual bool is_equal(ColumnPtr column) = 0;
    virtual bool operator==(ColumnBase& col) = 0;

    virtual int compareValuesAtIndexes(TID id1, TID id2) = 0;

    virtual bool setPrimaryKeyConstraint() = 0;
    virtual bool hasPrimaryKeyConstraint() const throw() = 0;

    virtual bool hasForeignKeyConstraint() const throw() = 0;
    virtual bool setForeignKeyConstraint(
        const ForeignKeyConstraint& prim_foreign_key_reference) = 0;
    virtual const ForeignKeyConstraint& getForeignKeyConstraint() = 0;

    virtual const ColumnStatistics& getColumnStatistics() const = 0;
    virtual bool computeColumnStatistics() = 0;

   protected:
    TID begin_index_;
    size_t num_elements_;
  };

  //	Vector::Vector(const std::string& name,
  //                AttributeType db_type,
  //                ColumnType column_type,
  //                TID begin_index,
  //                size_t num_elements) :
  //        ColumnBase(name, db_type, column_type),
  //        begin_index_(begin_index), num_elements_(num_elements)
  //        {
  //
  //        }

}  // end namespace CoGaDB

#endif /* VECTOR_HPP */
