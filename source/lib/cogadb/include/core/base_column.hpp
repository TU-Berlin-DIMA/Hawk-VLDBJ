
#pragma once

//#include <core/info_column.hpp>

// STL includes
#include <typeinfo>
// boost includes
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/vector.hpp>
// CoGaDB includes
#include <core/global_definitions.hpp>
//#include <core/positionlist.hpp>
#include <core/bitmap.hpp>
//#include <core/foreign_key_constraint.hpp>
#include <config/global_definitions.hpp>
#include <core/operator_parameter_types.hpp>

namespace CoGaDB {
  /////* \brief a PositionList is an STL vector of TID values*/
  ////typedef std::vector<TID> PositionList;
  ////    typedef test::PositionList PositionList;
  ///* \brief a PositionListPtr is a a references counted smart pointer to a
  /// PositionList object*/
  // typedef shared_pointer_namespace::shared_ptr<PositionList> PositionListPtr;
  ///* \brief a PositionListPair is an STL pair consisting of two
  /// PositionListPtr objects
  // *  \details This type is returned by binary operators, e.g., joins*/
  // typedef std::pair<PositionListPtr,PositionListPtr> PositionListPair;
  ///* \brief a PositionListPairPtr is a a references counted smart pointer to a
  /// PositionListPair object*/
  // typedef shared_pointer_namespace::shared_ptr<PositionListPair>
  // PositionListPairPtr;

  class Table;                 // forward declaration
  class ForeignKeyConstraint;  // forward declaration
  struct ColumnStatistics;     // forward declaration
  class ColumnGroupingKeys;    // forward declaration
  typedef boost::shared_ptr<ColumnGroupingKeys> ColumnGroupingKeysPtr;
  // class PositionList;
  template <typename T>
  class Column;
  typedef Column<TID> PositionList;
  typedef Column<double> DoubleDenseValueColumn;
  typedef Column<std::string> StringDenseValueColumn;

  typedef boost::shared_ptr<PositionList> PositionListPtr;
  typedef boost::shared_ptr<DoubleDenseValueColumn> DoubleDenseValueColumnPtr;
  typedef boost::shared_ptr<StringDenseValueColumn> StringDenseValueColumnPtr;

  /*  \details This type is returned by binary operators, e.g., joins*/
  typedef std::pair<PositionListPtr, PositionListPtr> PositionListPair;
  /* \brief a PositionListPairPtr is a a references counted smart pointer to a
   * PositionListPair object*/
  typedef shared_pointer_namespace::shared_ptr<PositionListPair>
      PositionListPairPtr;

  class LookupColumn;
  typedef boost::shared_ptr<LookupColumn> LookupColumnPtr;
  typedef std::pair<LookupColumnPtr, LookupColumnPtr> JoinIndex;
  typedef boost::shared_ptr<JoinIndex> JoinIndexPtr;

  class Vector;
  typedef boost::shared_ptr<Vector> VectorPtr;

  class BlockIterator;
  typedef boost::shared_ptr<BlockIterator> BlockIteratorPtr;

  // PositionListPtr createPositionList();

  /*!
   *
   *
   *  \brief     This class represents a generic column, is the base class for
   *all column classes and allows a uniform handling of columns.
   *  \details   This class is indentended to be a base class, so it has a
   *virtual destruktor and pure virtual methods, which need to be implemented in
   *a derived class.
   *  \author    Sebastian Breß
   *  \version   0.2
   *  \date      2013
   *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
   *http://www.gnu.org/licenses/lgpl-3.0.txt
   */

  class ColumnBase {  // : public InfoColumn{
   public:
    /*! \brief defines a smart pointer to a ColumnBase Object*/
    //	typedef shared_pointer_namespace::shared_ptr<ColumnBase> ColumnPtr;
    typedef std::pair<ColumnPtr, ColumnPtr> AggregationResult;
    /***************** constructors and destructor *****************/
    ColumnBase(const std::string& name, AttributeType db_type,
               ColumnType column_type = PLAIN_MATERIALIZED);
    virtual ~ColumnBase();
    /***************** methods *****************/
    /*! \brief appends a value new_Value to end of column
     *  \return true for sucess and false in case an error occured*/
    virtual bool insert(const boost::any& new_Value) = 0;
    /*! \brief updates the value on position tid with a value new_Value
     *  \return true for sucess and false in case an error occured*/
    virtual bool update(TID tid, const boost::any& new_Value) = 0;
    /*! \brief updates the values specified by the position list with a value
     * new_Value
     *  \return true for sucess and false in case an error occured*/
    virtual bool update(PositionListPtr tids, const boost::any& new_value) = 0;
    /*! \brief deletes the value on position tid
     *  \return true for sucess and false in case an error occured*/
    virtual bool remove(TID tid) = 0;
    /*! \brief appends the column \ref col to the current columns
     *  \return true for success and false in case an error occured (e.g., if
     * columns are not of same type)*/
    virtual bool append(ColumnPtr col) = 0;
    /*! \brief deletes the values defined in the position list
     *  \details assumes tid list is sorted ascending
     *  \return true for sucess and false in case an error occured*/
    virtual bool remove(PositionListPtr tid) = 0;
    /*! \brief generic function for fetching a value form a column (slow)
     *  \details check whether the object is valid (e.g., when a tid is not
     * valid, then the returned object is invalid as well)
     *  \return object of type boost::any containing the value on position tid*/
    virtual const boost::any get(TID tid) = 0;  // not const, because operator
                                                // [] does not provide const
                                                // return type and the child
                                                // classes rely on []
    /*! \brief generic function for fetching a string representation for a value
     * form a column
     *  \return string representing the value on position tid*/
    virtual std::string getStringValue(TID tid) = 0;  // not const, because
                                                      // operator [] does not
                                                      // provide const return
                                                      // type and the child
                                                      // classes rely on []
    /*! \brief prints the content of a column*/
    virtual void print() const throw() = 0;
    /*! \brief returns the number of values (rows) in a column*/
    virtual size_t size() const throw() = 0;
    /*! \brief returns the size in bytes the column consumes in main memory*/
    virtual size_t getSizeinBytes() const throw() = 0;

    virtual const VectorPtr getVector(ColumnPtr col,
                                      BlockIteratorPtr it) const = 0;
    /*! \brief creates a copy of this column that is compressed using
     *    compression methid specified in col_type
     * \return a ColumnPtr to an copy of the current column which is compressed
     * according to col_type*/
    virtual const ColumnPtr changeCompression(
        const ColumnType& col_type) const = 0;
    /*! \brief virtual copy constructor
     * \return a ColumnPtr to an exact copy of the current column*/
    virtual const ColumnPtr copy() const = 0;
    /*! \brief virtual copy constructor, copies column to memory of
     *         different processor, if neccessary
     * \return a ColumnPtr to an exact copy of the current column*/
    virtual const ColumnPtr copy(
        const hype::ProcessingDeviceMemoryID&) const = 0;
    /*! \brief convert values in column to double values and
     *  return a dense value DOUBLE column
     *  \param proc_spec execute this operation on a certain processor
     */
    virtual const DoubleDenseValueColumnPtr convertToDenseValueDoubleColumn(
        const ProcessorSpecification& proc_spec) const = 0;
    /*! \brief convert values in column to std::string values and
     *  return a dense value VARCHAR column
     *  \detail This operation can be executed on CPU only.
     */
    virtual const StringDenseValueColumnPtr convertToDenseValueStringColumn()
        const = 0;
    /*! \brief decompress compressed column
     *  \detail This operation can be executed on CPU only. When called on a
     * decompressed column, a copy is created.
     */
    virtual const ColumnPtr decompress(
        const ProcessorSpecification& proc_spec) const = 0;
    /*! \brief returns the ID of the memory the column is stored, e.g., in main
     memory
     or in GPU memory*/
    virtual hype::ProcessingDeviceMemoryID getMemoryID() const = 0;
    /*! \brief creates a new column by fetching all values identified by the
     * tid_list
     * \return a ColumnPtr that contains only values from the tid_list*/
    virtual const ColumnPtr gather(PositionListPtr tid_list,
                                   const GatherParam&) = 0;
    /*! \brief materializes a column to a normal uncompressed column with dense
     * values
     * \return a ColumnPtr to an materialized column*/
    virtual const ColumnPtr materialize() throw() = 0;
    /***************** relational operations on Columns which return a
     * PositionListPtr/PositionListPairPtr *****************/
    /*! \brief sorts a column w.r.t. a SortOrder
     * \return PositionListPtr to a PositionList, which represents the result*/
    virtual const PositionListPtr sort(const SortParam& param) = 0;
    /*! \brief filters the values of a column according to a filter condition
     * consisting of a comparison value and a ValueComparator (=,<,>)
     * \return PositionListPtr to a PositionList, which represents the result*/
    virtual const PositionListPtr selection(const SelectionParam& param) = 0;
    /*! \brief filters the values of a column according to a filter condition
     * consisting of a comparison value and a ValueComparator (=,<,>)
     * \return BitmapPtr to a Bitmap, which represents the result, by having one
     * Bit per per row, indicating whether the row matches the predicate or
     * not*/
    virtual const BitmapPtr bitmap_selection(const SelectionParam& param) = 0;
    /*! \brief filters the values of a column according to a filter condition
     * consisting of a comparison column and a ValueComparator (=,<,>).
     * This implements the comparison of two values from two columns.
     * \return PositionListPtr to a PositionList, which represents the result*/
    //	virtual const PositionListPtr selection(ColumnPtr, const ValueComparator
    // comp)= 0;
    /*! \brief filters the values of a column in parallel according to a filter
     * condition consisting of a comparison value and a ValueComparator (=,<,>)
     * \details the additional parameter specifies the number of threads that
     * may be used to perform the operation
     * \return PositionListPtr to a PositionList, which represents the result*/
    virtual const PositionListPtr parallel_selection(
        const boost::any& value_for_comparison, const ValueComparator comp,
        unsigned int number_of_threads) = 0;
    /*! \brief joins two columns using the hash join algorithm
     * \return PositionListPairPtr to a PositionListPair, which represents the
     * result*/
    virtual const PositionListPairPtr hash_join(ColumnPtr join_column) = 0;
    /*! \brief joins two columns using the hash join algorithm with a parallel
     * pruning phase
     * \return PositionListPairPtr to a PositionListPair, which represents the
     * result*/
    virtual const PositionListPairPtr parallel_hash_join(
        ColumnPtr join_column, unsigned int number_of_threads) = 0;
    /*! \brief joins two columns using the sort merge join algorithm
     * \return PositionListPairPtr to a PositionListPair, which represents the
     * result*/
    virtual const PositionListPairPtr sort_merge_join(
        ColumnPtr join_column) = 0;
    /*! \brief joins two columns using the nested loop join algorithm
     * \return PositionListPairPtr to a PositionListPair, which represents the
     * result*/
    virtual const PositionListPairPtr nested_loop_join(
        ColumnPtr join_column) = 0;
    /*! \brief joins two columns using the radix join algorithm
     * \return PositionListPairPtr to a PositionListPair, which represents the
     * result*/
    virtual const PositionListPairPtr radix_join(ColumnPtr join_column) = 0;

    /*! \brief joins two columns
     * \return PositionListPairPtr to a PositionListPair, which represents the
     * result*/
    virtual const PositionListPairPtr join(ColumnPtr join_column,
                                           const JoinParam&) = 0;
    virtual const PositionListPtr tid_semi_join(ColumnPtr join_column,
                                                const JoinParam&) = 0;
    virtual const BitmapPtr bitmap_semi_join(ColumnPtr join_column,
                                             const JoinParam&) = 0;

    /*! creates a compact version of the values in the column, which is used by
     * groupby*/
    virtual const ColumnGroupingKeysPtr createColumnGroupingKeys(
        const ProcessorSpecification& proc_spec) const = 0;
    /*! returns the number of bits we need to represent the values stored in the
     * column*/
    virtual size_t getNumberOfRequiredBits() const = 0;
    /*! \brief aggregates the column values according to grouping keys
     *         and AggregationMethod agg_meth using aggregation algorithm
     * agg_alg
     *  \details using aggregation algorithm sort based requires prior sorting
     * of column!
     */
    virtual const AggregationResult aggregateByGroupingKeys(
        ColumnGroupingKeysPtr grouping_keys, const AggregationParam&) = 0;
    /*! \brief aggregation without group by clause*/
    virtual const AggregationResult aggregate(const AggregationParam&) = 0;
    /***************** column algebra operations *****************/
    virtual const ColumnPtr column_algebra_operation(
        ColumnPtr source_column, const AlgebraOperationParam&) = 0;
    virtual const ColumnPtr column_algebra_operation(
        const boost::any& value, const AlgebraOperationParam&) = 0;

    /***************** persistency operations *****************/
    /*! \brief store a column on the disk
     *  \return true for sucess and false in case an error occured*/
    virtual bool store(const std::string& path) = 0;
    /*! \brief load column from disk
     *  \details calling load on a column that is not empty yields undefined
     * behaviour
     *  \param path the path to the table directory the column resides in
     *  \param column_loader_mode load only statistics and integrity constraints
     * from disk or load also the actual data
     *  \return true for sucess and false in case an error occured*/
    virtual bool load(const std::string& path,
                      ColumnLoaderMode column_loader_mode = LOAD_ALL_DATA) = 0;
    /*! \brief use this method to determine whether the column is materialized
     * or a Lookup Column
     * \return true in case the column is storing the plain values (without
     * compression) and false in case the column is a LookupColumn.*/
    /***************** misc operations *****************/
    virtual bool isMaterialized() const throw() = 0;
    /*! \brief use this method to determine whether the column is materialized
     * or a Lookup Column
     * \return true in case the column is storing the compressed values and
     * false otherwise.*/
    virtual bool isCompressed() const throw() = 0;
    /*! \brief returns type information of internal values*/
    virtual const std::type_info& type() const throw() = 0;
    /*! \brief returns database type of column (as defined in "SQL" statement)*/
    // TODO someone should change this method to "getAttributeType"!!
    AttributeType getType() const throw();
    /*! \brief set database type of column (as defined in "SQL" statement)*/
    void setType(const AttributeType&);
    /*! \brief returns type of column (e.g., plain (uncompressed) or dictionary
     * compressed)*/
    ColumnType getColumnType() const throw();

    /*! \brief returns true if column is loaded in memory, otherwise we return
       false,
         because the column resides on disk*/
    bool isLoadedInMainMemory() const throw();
    /*! \brief sets the status whether this column is currently memory resident
     * or on disk*/
    void setStatusLoadedInMainMemory(bool is_loaded) throw();
    /*! \brief returns attribute name of column
            \return attribute name of column*/
    const std::string getName() const throw();
    /*! \brief sets the attribute name of column*/
    void setName(const std::string& value) throw();
    /*! \brief test this column and column for equality
            \return returns true if columns are equal and false otherwise*/
    virtual bool is_equal(ColumnPtr column) = 0;
    virtual bool isApproximatelyEqual(ColumnPtr column) = 0;
    virtual bool operator==(ColumnBase& col) = 0;
    /*! \brief compares the values of this column on position id1 with value at
     * position id2 */
    virtual int compareValuesAtIndexes(TID id1, TID id2) = 0;
    /* ATTENTION: after setting integrity constraints, the result of a write
     * operation is undefined! This follows the typical bulk load and
     * analyize workflow of OLAP. Hence, we do currently not implement
     * integrity checks when inserting new data. First import your data and then
     * set the constraints! */
    virtual bool setPrimaryKeyConstraint() = 0;
    virtual bool hasPrimaryKeyConstraint() const throw() = 0;

    virtual bool hasForeignKeyConstraint() const throw() = 0;
    virtual bool setForeignKeyConstraint(
        const ForeignKeyConstraint& prim_foreign_key_reference) = 0;
    virtual const ForeignKeyConstraint& getForeignKeyConstraint() = 0;

    virtual const ColumnStatistics& getColumnStatistics() const = 0;
    virtual bool computeColumnStatistics() = 0;

   protected:
    /*! \brief attribute name of the column*/
    std::string name_;
    /*! \brief database type of the column*/
    AttributeType db_type_;
    /*! \brief compression type of the column*/
    ColumnType column_type_;
    /*! \brief indicates whether column is still on disk (false)
     or loaded in-memory*/
    bool is_loaded_;
    //	Table& table_;
  };

  /*! \brief makes a smart pointer to a ColumnBase Object visible in the
   * namespace*/
  // typedef ColumnBase::ColumnPtr ColumnPtr;
  typedef ColumnBase::AggregationResult AggregationResult;

  typedef std::vector<ColumnPtr> ColumnVector;
  typedef shared_pointer_namespace::shared_ptr<ColumnVector> ColumnVectorPtr;

  /*! \brief Column factory function, creates an empty materialized column*/
  const ColumnPtr createColumn(AttributeType type, const std::string& name);
  const ColumnPtr createColumn(AttributeType type, const std::string& name,
                               ColumnType column_type);

  // bool operator== (const ColumnBase& lhs, const ColumnBase& rhs);
  bool operator!=(const ColumnBase& lhs, const ColumnBase& rhs);

  const PositionListPtr createPositionList(
      size_t num_of_elements = 0,
      const hype::ProcessingDeviceMemoryID& mem_id = hype::PD_Memory_0);
  const PositionListPtr createPositionList(
      size_t num_of_elements, const ProcessorSpecification& proc_spec);
  const PositionListPtr createAscendingPositionList(
      size_t num_of_elements, const ProcessorSpecification& proc_spec,
      const TID start_value = TID(0));

  TID* getPointer(PositionList&);
  size_t getSize(const PositionList&);
  bool resize(PositionList&, size_t new_num_elements);
  hype::ProcessingDeviceMemoryID getMemoryID(const PositionList&);

  /*! create a copy of a data object and transfer data between processors if
   * required*/
  ColumnPtr copy(ColumnPtr col, const hype::ProcessingDeviceMemoryID& mem_id);
  std::vector<ColumnPtr> copy(const std::vector<ColumnPtr>& col_vec,
                              const hype::ProcessingDeviceMemoryID& mem_id);
  PositionListPtr copy(PositionListPtr tids,
                       const hype::ProcessingDeviceMemoryID& mem_id);
  PositionListPairPtr copy(PositionListPairPtr pair_tids,
                           const hype::ProcessingDeviceMemoryID& mem_id);
  LookupColumnPtr copy(LookupColumnPtr lookup_column,
                       const hype::ProcessingDeviceMemoryID& mem_id);
  JoinIndexPtr copy(JoinIndexPtr join_index,
                    const hype::ProcessingDeviceMemoryID& mem_id);
  BitmapPtr copy(BitmapPtr bitmap,
                 const hype::ProcessingDeviceMemoryID& mem_id);

  /*! create a copy only of data is not dorment in memory with ID mem_id,
      this copies data between processors on demand*/
  ColumnPtr copy_if_required(ColumnPtr col,
                             const hype::ProcessingDeviceMemoryID& mem_id);
  std::vector<ColumnPtr> copy_if_required(
      const std::vector<ColumnPtr>& col,
      const hype::ProcessingDeviceMemoryID& mem_id);
  PositionListPtr copy_if_required(
      PositionListPtr tids, const hype::ProcessingDeviceMemoryID& mem_id);
  PositionListPairPtr copy_if_required(
      PositionListPairPtr pair_tids,
      const hype::ProcessingDeviceMemoryID& mem_id);
  LookupColumnPtr copy_if_required(
      LookupColumnPtr lookup_column,
      const hype::ProcessingDeviceMemoryID& mem_id);
  JoinIndexPtr copy_if_required(JoinIndexPtr join_index,
                                const hype::ProcessingDeviceMemoryID& mem_id);
  BitmapPtr copy_if_required(BitmapPtr bitmap,
                             const hype::ProcessingDeviceMemoryID& mem_id);

  ColumnPtr copy_if_required(ColumnPtr col,
                             const ProcessorSpecification& proc_spec);
  std::vector<ColumnPtr> copy_if_required(
      const std::vector<ColumnPtr>& col,
      const ProcessorSpecification& proc_spec);
  PositionListPtr copy_if_required(PositionListPtr tids,
                                   const ProcessorSpecification& proc_spec);
  PositionListPairPtr copy_if_required(PositionListPairPtr pair_tids,
                                       const ProcessorSpecification& proc_spec);
  LookupColumnPtr copy_if_required(LookupColumnPtr lookup_column,
                                   const ProcessorSpecification& proc_spec);
  JoinIndexPtr copy_if_required(JoinIndexPtr join_index,
                                const ProcessorSpecification& proc_spec);
  BitmapPtr copy_if_required(BitmapPtr bitmap,
                             const ProcessorSpecification& proc_spec);

  /*! \brief decompresses input column if required and return pointer to
   * decompressed column
   * \detail if the input column is not compressed, it is returned as output
   * column
   *  and no work is done. */
  ColumnPtr decompress_if_required(ColumnPtr col);

  VectorPtr createVector(ColumnPtr col, BlockIteratorPtr it);

  /*! \brief computes the union of two position lists, e.g., to compute a
   * logical OR in a complex selection, the tid lists resulting from several
   * selections need to be united*/
  PositionListPtr computePositionListUnion(PositionListPtr tids1,
                                           PositionListPtr tids2);
  /*! \brief computes the union of two position lists, e.g., to compute a
   * logical AND in a complex selection, the tid lists resulting from several
   * selections need to be intersected*/
  PositionListPtr computePositionListIntersection(PositionListPtr tids1,
                                                  PositionListPtr tids2);

}  // end namespace CogaDB

// extend boost namespace to add serialization feature to my own types
namespace boost {
  namespace serialization {

    template <class Archive>
    void serialize(Archive& ar,
                   std::pair<CoGaDB::AttributeType, std::string>& pair,
                   const unsigned int)  // version)
    {
      ar& pair.first;
      ar& pair.second;
    }

  }  // namespace serialization
}  // namespace boost
