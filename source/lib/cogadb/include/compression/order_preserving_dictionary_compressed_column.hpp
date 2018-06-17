
#pragma once

#include <boost/serialization/map.hpp>
#include <core/compressed_column.hpp>
#include <util/dictionary_compressed_col.hpp>

#include <core/column.hpp>

namespace CoGaDB {

  /*class DictionaryCompressedCol {
  public:
      virtual const std::vector<uint32_t>& getCompressedValues() const throw ()
  = 0;
      virtual unsigned int getNumberofDistinctValues() const = 0;
      virtual uint32_t getLargestID() const = 0;
      virtual size_t getNumberOfRows() const = 0;
  };*/

  /*!
   *  \brief     This class represents a dictionary compressed column with type
   * T, is the base class for all compressed typed column classes.
   *  \author    Sebastian Breß
   *  \version   0.2
   *  \date      2013
   *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
   * http://www.gnu.org/licenses/lgpl-3.0.txt
   */

  template <class T>
  class OrderPreservingDictionaryCompressedColumn
      : public CompressedColumn<T>,
        public DictionaryCompressedCol {
   public:
    typedef std::map<T, uint32_t> Dictionary;
    typedef Column<uint32_t> IDColumn;
    typedef boost::shared_ptr<IDColumn> IDColumnPtr;
    typedef uint32_t CodeWordType;

    /***************** constructors and destructor *****************/
    OrderPreservingDictionaryCompressedColumn(
        const std::string& name, AttributeType db_type,
        const hype::ProcessingDeviceMemoryID& mem_id = hype::PD_Memory_0);
    // DictionaryCompressedColumn(DictionaryCompressedColumn&);

    OrderPreservingDictionaryCompressedColumn(
        const OrderPreservingDictionaryCompressedColumn&);

    OrderPreservingDictionaryCompressedColumn& operator=(
        const OrderPreservingDictionaryCompressedColumn&);

    virtual ~OrderPreservingDictionaryCompressedColumn();

    bool insert(const boost::any& new_Value);
    bool insert(const T& new_value);

    template <typename InputIterator>
    bool insert(InputIterator first, InputIterator last);

    virtual bool update(TID tid, const boost::any& new_value);
    virtual bool update(PositionListPtr tid, const boost::any& new_value);

    virtual bool remove(TID tid);
    // assumes tid list is sorted ascending
    virtual bool remove(PositionListPtr tid);
    bool clearContent();

    virtual const boost::any get(TID tid);
    // virtual const boost::any* const getRawData()=0;
    virtual void print() const throw();
    virtual size_t size() const throw();
    virtual size_t getSizeinBytes() const throw();

    virtual const ColumnPtr copy() const;
    virtual const ColumnPtr copy(
        const hype::ProcessingDeviceMemoryID& mem_id) const;
    virtual const typename ColumnBaseTyped<T>::DenseValueColumnPtr
    copyIntoDenseValueColumn(const ProcessorSpecification& proc_spec) const;
    const ColumnPtr gather(PositionListPtr tid_list, const GatherParam&);

    virtual const ColumnGroupingKeysPtr createColumnGroupingKeys(
        const ProcessorSpecification& proc_spec) const;
    virtual size_t getNumberOfRequiredBits() const;

    virtual bool isMaterialized() const throw();
    virtual const ColumnPtr materialize() throw();
    virtual hype::ProcessingDeviceMemoryID getMemoryID() const;

    virtual const PositionListPtr selection(const SelectionParam& param);
    virtual const BitmapPtr bitmap_selection(const SelectionParam& param);

    virtual bool isCompressed() const throw();

    const Dictionary& getDictionary() const throw();

    std::pair<bool, uint32_t> getDictionaryID(const T& value) const;
    std::pair<bool, uint32_t> getClosestDictionaryIDForPredicate(
        const T& value, const ValueComparator& comp,
        ValueComparator& rewritten_value_comparator) const;
    unsigned int getNumberofDistinctValues() const;
    uint32_t getLargestID() const;
    size_t getNumberOfRows() const;

    virtual uint32_t* getIdData();
    const T* const getReverseLookupVector() const;

    virtual T& operator[](const TID index);

    virtual T& reverseLookup(uint32_t id);

   private:
    void updateColumnPreserveringOrder(const u_int64_t index, const T& value);

    bool load_impl(const std::string& path,
                   boost::archive::binary_iarchive& ia);
    bool store_impl(const std::string& path,
                    boost::archive::binary_oarchive& oa);
    const ColumnPtr getDecompressedColumn_impl(
        const ProcessorSpecification& proc_spec);
    IDColumnPtr ids_;
    Dictionary dictionary_;
    std::vector<T> reverse_lookup_vector_;
    uint32_t maximal_id_;

    /** FOR UNIT TESTING: ALLOWS ACCESS TO PRIVATE MEMBERS **/
    /**
     * https://code.google.com/p/googletest/wiki/AdvancedGuide#Private_Class_Members
     * **/
    friend class OrderPreservingDictColTest;
  };

  template <typename T>
  template <typename InputIterator>
  bool OrderPreservingDictionaryCompressedColumn<T>::insert(
      InputIterator first, InputIterator last) {
    for (; first != last; ++first) {
      if (!this->insert(*first)) {
        return false;
      }
    }
    return true;
  }
}  // end namespace CogaDB
