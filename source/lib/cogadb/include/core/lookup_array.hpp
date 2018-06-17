
#pragma once

#include <core/base_table.hpp>
#include <core/column.hpp>
#include <core/column_base_typed.hpp>
#include <lookup_table/lookup_column.hpp>

//#ifdef ENABLE_CDK_USAGE
//        #include <hardware_optimizations/primitives.hpp>
//#endif
//#include "runtime_configuration.hpp"

namespace CoGaDB {

  /*!
   *
   *
   *  \brief     A LookupArray is a LookupColumn which is applied on a
   *materialized column (of the table that is indexed by the Lookup column) and
   *hence has a Type.
   * 				This class represents a column with type T,
   *which
   *is
   *essentially a tid list describing which values of a typed materialized
   *column are included in the LookupArray.
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
  class LookupArray : public ColumnBaseTyped<T> {
   public:
    /***************** constructors and destructor *****************/
    LookupArray(const std::string& name, AttributeType db_type,
                ColumnPtr column, PositionListPtr tids);
    virtual ~LookupArray();

    virtual bool insert(const boost::any& new_Value);
    virtual bool insert(const T& new_Value);
    virtual bool update(TID tid, const boost::any& new_value);
    virtual bool update(PositionListPtr tid, const boost::any& new_value);

    virtual bool remove(TID tid);
    // assumes tid list is sorted ascending
    virtual bool remove(PositionListPtr tid);

    virtual bool append(boost::shared_ptr<ColumnBaseTyped<T> > typed_col);

    virtual bool clearContent();

#ifdef ENABLE_CDK_USAGE
    //        virtual const PositionListPtr selection(const boost::any&
    //        value_for_comparison, const ValueComparator comp);
    //        virtual const PositionListPtr selection(ColumnPtr
    //        comparison_column, const ValueComparator comp);
    virtual const PositionListPtr selection(const SelectionParam& param);
    virtual const BitmapPtr bitmap_selection(const SelectionParam& param);
    virtual const PositionListPairPtr hash_join(ColumnPtr join_column_);
    virtual const PositionListPairPtr radix_join(ColumnPtr join_column_);
#endif

    virtual const boost::any get(TID tid);
    // virtual const boost::any* const getRawData()=0;
    virtual void print() const throw();
    virtual size_t size() const throw();
    virtual size_t getSizeinBytes() const throw();
    virtual PositionListPtr getPositionList();
    virtual shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> >
    getIndexedColumn();

    virtual const ColumnPtr copy() const;
    virtual const ColumnPtr copy(
        const hype::ProcessingDeviceMemoryID& mem_id) const;
    virtual const typename ColumnBaseTyped<T>::DenseValueColumnPtr
    copyIntoDenseValueColumn(const ProcessorSpecification& proc_spec) const;
    virtual const ColumnPtr gather(PositionListPtr tid_list,
                                   const GatherParam&);

    virtual const ColumnGroupingKeysPtr createColumnGroupingKeys(
        const ProcessorSpecification& proc_spec) const;
    virtual size_t getNumberOfRequiredBits() const;

    //        virtual const AggregationResult
    //        aggregateByGroupingKeys(ColumnGroupingKeysPtr grouping_keys,
    //        AggregationMethod agg_meth, AggregationAlgorithm agg_alg);
    virtual const PositionListPairPtr join(ColumnPtr join_column,
                                           const JoinParam&);
    virtual const AggregationResult aggregate(const AggregationParam&);

    virtual const ColumnPtr column_algebra_operation(
        ColumnPtr source_column, const AlgebraOperationParam&);
    virtual const ColumnPtr column_algebra_operation(
        const boost::any& value, const AlgebraOperationParam&);

    virtual const PositionListPtr sort(const SortParam& param);

    virtual bool store(const std::string& path);
    virtual bool load(const std::string& path,
                      ColumnLoaderMode column_loader_mode);
    virtual bool isMaterialized() const throw();
    virtual bool isCompressed() const throw();
    virtual const ColumnPtr materialize() throw();
    virtual hype::ProcessingDeviceMemoryID getMemoryID() const;

    T* materializeToArray() throw();

    /*! \brief returns type information of internal values*/
    virtual T& operator[](const TID index);
    // inline T& operator[](const int index) __attribute__((always_inline));

   private:
    bool load_impl(const std::string& path,
                   boost::archive::binary_iarchive& ia);
    bool store_impl(const std::string& path,
                    boost::archive::binary_oarchive& oa);
    shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> > column_;
    PositionListPtr tids_;
  };

  ColumnPtr createLookupArrayForColumn(ColumnPtr col, PositionListPtr tids_);
  PositionListPtr getPositonListfromLookupArray(ColumnPtr col);

}  // end namespace CogaDB
