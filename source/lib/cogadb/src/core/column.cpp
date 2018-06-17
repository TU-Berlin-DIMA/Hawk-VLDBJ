#include <core/column.hpp>
#include <core/lookup_array.hpp>
#include <hardware_optimizations/primitives.hpp>
#include <hardware_optimizations/simd_acceleration.hpp>

#include <hardware_optimizations/main_memory_joins/hash_joins.hpp>
#include <util/column_grouping_keys.hpp>
#include <util/column_grouping_keys.hpp>
#include <util/getname.hpp>
#include <util/types.hpp>
#include <util/utility_functions.hpp>

//#define ENABLE_SIMD_ACCELERATION
#include <backends/processor_backend.hpp>

#include <compression/rle_compressed_column.hpp>
#include <compression/void_compressed_column_int.hpp>

#include <boost/lexical_cast.hpp>
#include <core/copy_function_factory.hpp>
#include <util/functions.hpp>
#include <util/utility_functions.hpp>

#include <util/hardware_detector.hpp>
#include <util/utility_functions.hpp>

#include <query_compilation/ocl_data_cache.hpp>

#define COGADB_SELECTION_BODY_BRANCH(array, value, i, array_tids, pos, \
                                     COMPARATOR)                       \
  if (array[i] COMPARATOR value) array_tids[pos++] = i;
#define COGADB_SELECTION_BODY_NOBRANCH(array, value, i, array_tids, pos, \
                                       COMPARATOR)                       \
  array_tids[pos] = i;                                                   \
  pos += (array[i] COMPARATOR value);

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

namespace CoGaDB {

using namespace std;

/***************** Start of Implementation Section ******************/

template <class T>
Column<T>::Column(const std::string &name, AttributeType db_type,
                  hype::ProcessingDeviceMemoryID mem_id)
    : ColumnBaseTyped<T>(name, db_type, PLAIN_MATERIALIZED),
      type_tid_comparator(),
      values_(0),
      num_elements_(0),
      buffer_size_(0),
      mem_alloc(MemoryAllocator<T>::getMemoryAllocator(mem_id)) {
  this->reserve(DEFAULT_COLUMN_SIZE);
  assert(values_ != NULL);
}

// fill constructor
template <class T>
Column<T>::Column(const std::string &name, AttributeType db_type,
                  size_t number_of_elements, const T &val,
                  hype::ProcessingDeviceMemoryID mem_id)
    : ColumnBaseTyped<T>(name, db_type, PLAIN_MATERIALIZED),
      type_tid_comparator(),
      values_(0),
      num_elements_(0),
      buffer_size_(0),
      mem_alloc(MemoryAllocator<T>::getMemoryAllocator(mem_id)) {
  if (this->capacity() < number_of_elements) {
    this->reserve(number_of_elements);
  }
  assert(this->mem_alloc->getMemoryID() == hype::PD_Memory_0);
  std::fill_n(values_, number_of_elements, val);
  num_elements_ = number_of_elements;
}

template <class T>
Column<T>::Column(const Column<T> &x)
    : ColumnBaseTyped<T>(x.getName(), x.getType(), PLAIN_MATERIALIZED),
      type_tid_comparator(),
      values_(0),
      num_elements_(0),
      buffer_size_(0),
      mem_alloc(
          MemoryAllocator<T>::getMemoryAllocator(x.mem_alloc->getMemoryID())) {
  this->reserve(x.num_elements_);
  assert(values_ != NULL);

  if (typeid(T) != typeid(std::string)) {
    if (!x.empty()) {
      typedef typename CopyFunctionFactory<T>::CopyFunctionPtr CopyFunctionPtr;
      CopyFunctionPtr func = CopyFunctionFactory<T>::getCopyFunction(
          this->mem_alloc->getMemoryID(), x.mem_alloc->getMemoryID());
      assert(func != NULL);
      bool ret = (*func)(values_, x.values_, x.num_elements_ * sizeof(T));
      assert(ret == true);
    }
  } else {
    assert(this->mem_alloc->getMemoryID() == hype::PD_Memory_0);
    assert(x.mem_alloc->getMemoryID() == hype::PD_Memory_0);
    // make proper copies of objects
    std::copy(x.values_, x.values_ + x.num_elements_, values_);
  }
  num_elements_ = x.num_elements_;
}

template <class T>
Column<T> &Column<T>::operator=(const Column<T> &other) {
  if (this != &other)  // protect against invalid self-assignment
  {
    assert(this->mem_alloc->getMemoryID() == other.mem_alloc->getMemoryID());

    this->mem_alloc =
        MemoryAllocator<T>::getMemoryAllocator(other.mem_alloc->getMemoryID());

    // allocate new memory (using realloc, so we don't need to cleanup old
    // memory and malloc new memory!) and copy the elements
    this->reserve(other.num_elements_);  // calls realloc!
    assert(values_ != NULL);
    if (typeid(T) != typeid(std::string)) {
      if (!other.empty()) {
        typedef
            typename CopyFunctionFactory<T>::CopyFunctionPtr CopyFunctionPtr;
        CopyFunctionPtr func = CopyFunctionFactory<T>::getCopyFunction(
            this->mem_alloc->getMemoryID(), other.mem_alloc->getMemoryID());

        assert(func != NULL);
        bool ret =
            (*func)(values_, other.values_, other.num_elements_ * sizeof(T));
        assert(ret == true);
      }
    } else {
      assert(this->mem_alloc->getMemoryID() == hype::PD_Memory_0);
      assert(other.mem_alloc->getMemoryID() == hype::PD_Memory_0);
      // make proper copies of objects
      std::copy(other.values_, other.values_ + other.num_elements_, values_);
    }

    num_elements_ = other.num_elements_;
  }
  // by convention, always return *this
  return *this;
}

template <class T>
Column<T>::~Column() {
  OCL_DataCaches::instance().uncacheMemoryArea(values_,
                                               sizeof(T) * num_elements_);
  if (values_) this->mem_alloc->deallocate(values_);
}

template <class T>
bool Column<T>::insert(const boost::any &new_value) {
  T value;
  bool ret_success =
      getValueFromAny(this->name_, new_value, value, this->db_type_);
  if (!ret_success) return false;
  this->push_back(value);
  return true;
}

template <class T>
bool Column<T>::insert(const T &new_value) {
  this->push_back(new_value);
  return true;
}

template <class T>
bool Column<T>::update(TID tid, const boost::any &new_value) {
  if (new_value.empty()) return false;
  if (typeid(T) == new_value.type()) {
    T value = boost::any_cast<T>(new_value);
    values_[tid] = value;
    return true;
  } else {
    std::cout << "Fatal Error!!! Typemismatch for column " << this->name_
              << std::endl;
  }
  return false;
}

template <class T>
bool Column<T>::update(PositionListPtr tids, const boost::any &new_value) {
  if (!tids) return false;
  if (new_value.empty()) return false;
  if (typeid(T) == new_value.type()) {
    T value = boost::any_cast<T>(new_value);
    for (unsigned int i = 0; i < tids->size(); i++) {
      TID tid = (*tids)[i];
      values_[tid] = value;
    }
    return true;
  } else {
    std::cout << "Fatal Error!!! Typemismatch for column " << this->name_
              << std::endl;
  }
  return false;
}

template <class T>
bool Column<T>::remove(TID tid) {
  return false;
}

template <class T>
bool Column<T>::remove(PositionListPtr tids) {
  if (!tids) return false;
  // test whether tid list has at least one element, if not, return with error
  if (tids->empty()) return false;

  unsigned int loop_counter = tids->size();
  while (loop_counter > 0) {
    loop_counter--;
    this->remove((*tids)[loop_counter]);
  }

  return true;
}

template <class T>
bool Column<T>::append(boost::shared_ptr<ColumnBaseTyped<T> > typed_col) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr dense_column;
  if (typed_col->getColumnType() == PLAIN_MATERIALIZED) {
    dense_column = boost::dynamic_pointer_cast<DenseValueColumn>(typed_col);
  } else {
    boost::shared_ptr<ColumnBaseTyped<T> > typed_column =
        boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(typed_col);
    assert(typed_column != NULL);
    ProcessorSpecification proc_spec(
        hype::util::getProcessingDeviceID(this->getMemoryID()));
    dense_column = typed_column->copyIntoDenseValueColumn(proc_spec);
  }
  assert(dense_column != NULL);
  T *array = dense_column->data();
  size_t num_elements = dense_column->size();

  this->reserve(this->size() + num_elements);
  assert(values_ != NULL);

  if (typeid(T) != typeid(std::string)) {
    if (num_elements > 0) {
      typedef typename CopyFunctionFactory<T>::CopyFunctionPtr CopyFunctionPtr;
      CopyFunctionPtr func = CopyFunctionFactory<T>::getCopyFunction(
          this->mem_alloc->getMemoryID(),
          dense_column->mem_alloc->getMemoryID());
      assert(func != NULL);
      // append to end
      bool ret = (*func)(values_ + this->num_elements_, array,
                         num_elements * sizeof(T));
      assert(ret == true);
    }
  } else {
    assert(this->mem_alloc->getMemoryID() == hype::PD_Memory_0);
    assert(dense_column->mem_alloc->getMemoryID() == hype::PD_Memory_0);
    // make proper copies of objects, append at end
    std::copy(array, array + num_elements, values_ + this->num_elements_);
  }
  num_elements_ += num_elements;

  return true;
}

template <class T>
bool Column<T>::clearContent() {
  // values_.clear();
  // COGADB_FATAL_ERROR("Called unimplemented method!","");

  // FIXME: Maybe we should reduce memory footprint here?
  // recycling memory may be good for performance on CPU,
  // but reducing memory footprint is better for co-processors
  num_elements_ = 0;
  return true;
}

template <class T>
const boost::any Column<T>::get(TID tid) {
  if (tid < this->size())
    return boost::any(values_[tid]);
  else {
    std::cout << "fatal Error!!! Invalid TID!!! Attribute: " << this->name_
              << " TID: " << tid << std::endl;
  }
  return boost::any();
}

template <class T>
void Column<T>::print() const throw() {
  std::cout << "| " << this->name_ << " |" << std::endl;
  std::cout << "| Memory: " << (int)this->mem_alloc->getMemoryID() << " |"
            << std::endl;
  std::cout << "| Type: " << util::getName(this->getType())
            << " C++ type: " << typeid(T).name() << " |" << std::endl;
  std::cout << "| Compression: " << util::getName(this->getColumnType())
            << " C++ type: " << typeid(T).name() << " |" << std::endl;
  std::cout << "________________________" << std::endl;
  for (unsigned int i = 0; i < this->num_elements_; i++) {
    std::cout << "| " << values_[i] << " |" << std::endl;
  }
}

template <class T>
size_t Column<T>::size() const throw() {
  return num_elements_;
}

template <class T>
const ColumnPtr Column<T>::materialize() throw() {
  return copy();
}

template <class T>
hype::ProcessingDeviceMemoryID Column<T>::getMemoryID() const {
  return mem_alloc->getMemoryID();
}

template <class T>
const ColumnPtr Column<T>::copy() const {
  ColumnPtr col;
  try {
    col = ColumnPtr(new Column<T>(*this));
  } catch (std::bad_alloc &e) {
    COGADB_ERROR(
        "Out of memory on device memory "
            << (int)this->mem_alloc->getMemoryID() << " for column '"
            << this->getName() << "' requesting "
            << double(this->size() * sizeof(T)) / (1024 * 1024 * 2014)
            << " GB memory" << std::endl
            << "Total free memory: "
            << double(HardwareDetector::instance().getFreeMemorySizeInByte(
                   this->mem_alloc->getMemoryID())) /
                   (1024 * 1024 * 2014),
        "");
    return ColumnPtr();
  }
  return col;
}

template <class T>
const ColumnPtr Column<T>::copy(
    const hype::ProcessingDeviceMemoryID &mem_id) const {
  boost::shared_ptr<Column<T> > col;
  try {
    col = boost::shared_ptr<Column<T> >(
        new Column<T>(this->getName(), this->getType(), mem_id));
    col->resize(this->size());
  } catch (std::bad_alloc &e) {
    COGADB_ERROR(
        "Out of memory on device memory "
            << (int)mem_id << " for column '" << this->getName()
            << "' requesting "
            << double(this->size() * sizeof(T)) / (1024 * 1024 * 2014)
            << " GB memory" << std::endl
            << "Total free memory: "
            << double(HardwareDetector::instance().getFreeMemorySizeInByte(
                   mem_id)) /
                   (1024 * 1024 * 2014),
        "");
    return ColumnPtr();
  }

  if (col->size() > 0) {
    typedef typename CopyFunctionFactory<T>::CopyFunctionPtr CopyFunctionPtr;
    CopyFunctionPtr func = CopyFunctionFactory<T>::getCopyFunction(
        mem_id, this->mem_alloc->getMemoryID());
    assert(func != NULL);
    if (!(*func)(col->data(), values_, num_elements_ * sizeof(T)))
      return ColumnPtr();
  }
  return col;
}

template <class T>
const typename ColumnBaseTyped<T>::DenseValueColumnPtr
Column<T>::copyIntoDenseValueColumn(
    const ProcessorSpecification &proc_spec) const {
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  ColumnPtr copied_column =
      this->copy(hype::util::getMemoryID(proc_spec.proc_id));
  if (!copied_column) return DenseValueColumnPtr();

  DenseValueColumnPtr typed_copied_column =
      boost::dynamic_pointer_cast<DenseValueColumn>(copied_column);
  assert(typed_copied_column != NULL);
  return typed_copied_column;
}

template <class T>
const DoubleDenseValueColumnPtr Column<T>::convertToDenseValueDoubleColumn(
    const ProcessorSpecification &proc_spec) const {
  ProcessorBackend<T> *backend = ProcessorBackend<T>::get(proc_spec.proc_id);
  return backend->convertToDoubleDenseValueColumn(
      this->getName(), this->values_, this->size(), proc_spec);
}

template <typename T>
void toString(const T &val, std::string &result) {
  result = boost::lexical_cast<std::string>(val);
}

template <class T>
const StringDenseValueColumnPtr Column<T>::convertToDenseValueStringColumn()
    const {
  assert(mem_alloc->getMemoryID() == hype::PD_Memory_0);
  StringDenseValueColumnPtr string_column(
      new StringDenseValueColumn(this->getName(), VARCHAR, hype::PD_Memory_0));
  for (size_t i = 0; i < num_elements_; ++i) {
    std::string val;
    toString(this->values_[i], val);
    string_column->push_back(val);
  }
  return string_column;
}

template <>
const StringDenseValueColumnPtr
Column<std::string>::convertToDenseValueStringColumn() const {
  ColumnPtr col = this->copy();
  if (!col) return StringDenseValueColumnPtr();
  StringDenseValueColumnPtr result =
      boost::dynamic_pointer_cast<StringDenseValueColumn>(col);
  assert(result != NULL);
  return result;
}

template <class T>
const ColumnPtr Column<T>::gather(PositionListPtr tid_list,
                                  const GatherParam &param) {
  boost::shared_ptr<Column<T> > result(new Column<T>(
      this->name_, this->db_type_, this->mem_alloc->getMemoryID()));
  try {
    result->resize(tid_list->size());
  } catch (std::bad_alloc &e) {
    return ColumnPtr();
  }

  PositionListPtr copied_tids =
      copy_if_required(tid_list, this->mem_alloc->getMemoryID());
  if (!copied_tids) return ColumnPtr();
  tid_list = copied_tids;

  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);
  if (backend->gather(result->data(), this->data(), tid_list, param)) {
    return result;  // ColumnPtr(result);
  } else {
    return ColumnPtr();
  }
}

template <class T>
const ColumnGroupingKeysPtr Column<T>::createColumnGroupingKeys(
    const ProcessorSpecification &proc_spec) const {
  ProcessorBackend<T> *backend = ProcessorBackend<T>::get(proc_spec.proc_id);
  return backend->createColumnGroupingKeys(this->values_, this->size(),
                                           proc_spec);
}

template <>
const ColumnGroupingKeysPtr Column<std::string>::createColumnGroupingKeys(
    const ProcessorSpecification &proc_spec) const {
  return ColumnGroupingKeysPtr();
}

template <>
const ColumnGroupingKeysPtr Column<char *>::createColumnGroupingKeys(
    const ProcessorSpecification &proc_spec) const {
  return ColumnGroupingKeysPtr();
}

template <typename T>
size_t Column<T>::getNumberOfRequiredBits() const {
  auto &extended_statistics = this->getExtendedColumnStatistics();
  auto max = extended_statistics.max;

  if (!extended_statistics.statistics_up_to_date_) {
    T min = *std::min_element(values_, values_ + num_elements_);

    if (min < 0) {
      return sizeof(T) * 8;
    }

    max = *std::max_element(values_, values_ + num_elements_);
  }

  return getGreaterPowerOfTwo(max);
}

template <>
size_t Column<float>::getNumberOfRequiredBits() const {
  return sizeof(float) * 8;
}

template <>
size_t Column<double>::getNumberOfRequiredBits() const {
  return sizeof(float) * 8;
}

// we do not support groupby optimization for string columns
template <>
size_t Column<std::string>::getNumberOfRequiredBits() const {
  return 65;
}

template <>
size_t Column<char *>::getNumberOfRequiredBits() const {
  return 65;
}

template <class T>
const AggregationResult Column<T>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, const AggregationParam &param) {
  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);
  AggregationResult result;
  result = backend->aggregateByGroupingKeys(grouping_keys, this->data(),
                                            this->size(), param);
  if (result.second) result.second->setName(this->getName());

  return result;
}

template <class T>
const AggregationResult Column<T>::aggregate(const AggregationParam &param) {
  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);
  AggregationResult result =
      backend->aggregate(this->data(), this->size(), param);
  return result;
}

template <class T>
const PositionListPairPtr Column<T>::join(ColumnPtr join_column,
                                          const JoinParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;

  if (this->getType() != join_column->getType()) {
    COGADB_FATAL_ERROR("Cannot Join columns with different types", "");
    return PositionListPairPtr();
  }

  PositionListPairPtr result;
  if (this->getColumnType() == join_column->getColumnType()) {
    DenseValueColumnPtr dense_join_column =
        boost::dynamic_pointer_cast<DenseValueColumn>(join_column);
    assert(dense_join_column != NULL);
    result = this->join(*dense_join_column.get(), param);
  } else {
    boost::shared_ptr<ColumnBaseTyped<T> > typed_join_column =
        boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(join_column);
    assert(typed_join_column != NULL);
    DenseValueColumnPtr dense_join_column =
        typed_join_column->copyIntoDenseValueColumn(param.proc_spec);
    if (!dense_join_column) return PositionListPairPtr();
    result = dense_join_column->join(*this, param);
    if (result) std::swap(result->first, result->second);
    //}
  }
  return result;
}

template <class T>
const PositionListPairPtr Column<T>::join(Column<T> &join_column,
                                          const JoinParam &param) {
  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);
  return backend->join(this->data(), this->size(), join_column.data(),
                       join_column.size(), param);
}

template <class T>
const PositionListPtr Column<T>::tid_semi_join(ColumnPtr join_column,
                                               const JoinParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;

  if (this->getType() != join_column->getType()) {
    COGADB_FATAL_ERROR("Cannot Join columns with different types", "");
    return PositionListPtr();
  }

  PositionListPtr result;

  if (this->getColumnType() == join_column->getColumnType()) {
    DenseValueColumnPtr dense_join_column =
        boost::dynamic_pointer_cast<DenseValueColumn>(join_column);
    assert(dense_join_column != NULL);
    result = this->tid_semi_join(*dense_join_column.get(), param);
  } else {
    boost::shared_ptr<ColumnBaseTyped<T> > typed_join_column =
        boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(join_column);
    assert(typed_join_column != NULL);
    DenseValueColumnPtr dense_join_column =
        typed_join_column->copyIntoDenseValueColumn(param.proc_spec);
    if (!dense_join_column) return PositionListPtr();
    result = this->tid_semi_join(*dense_join_column.get(), param);
  }
  return result;
}

template <class T>
const PositionListPtr Column<T>::tid_semi_join(Column<T> &join_column,
                                               const JoinParam &param) {
  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);
  return backend->tid_semi_join(this->data(), this->size(), join_column.data(),
                                join_column.size(), param);
}

template <class T>
const BitmapPtr Column<T>::bitmap_semi_join(ColumnPtr join_column,
                                            const JoinParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;

  if (this->getType() != join_column->getType()) {
    COGADB_FATAL_ERROR("Cannot Join columns with different types", "");
    return BitmapPtr();
  }

  BitmapPtr result;

  if (this->getColumnType() == join_column->getColumnType()) {
    DenseValueColumnPtr dense_join_column =
        boost::dynamic_pointer_cast<DenseValueColumn>(join_column);
    assert(dense_join_column != NULL);
    result = this->bitmap_semi_join(*dense_join_column.get(), param);
  } else {
    boost::shared_ptr<ColumnBaseTyped<T> > typed_join_column =
        boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(join_column);
    assert(typed_join_column != NULL);
    DenseValueColumnPtr dense_join_column =
        typed_join_column->copyIntoDenseValueColumn(param.proc_spec);
    if (!dense_join_column) return BitmapPtr();
    result = this->bitmap_semi_join(*dense_join_column.get(), param);
  }
  return result;
}

template <class T>
const BitmapPtr Column<T>::bitmap_semi_join(Column<T> &join_column,
                                            const JoinParam &param) {
  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);
  return backend->bitmap_semi_join(this->data(), this->size(),
                                   join_column.data(), join_column.size(),
                                   param);
}

template <class T>
const ColumnPtr Column<T>::column_algebra_operation(
    ColumnPtr source_column, const AlgebraOperationParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;

  // we need to ensure that both columns have the same type, if not, then
  // convert both to double columns if required, and omit covnertions if
  // possible to save overhead
  if (this->getType() != source_column->getType()) {
    ColumnPtr converted_column = source_column;

    // check whether source_column already is a double column, or whether we
    // need to convert it into a double column
    if (source_column->getType() != DOUBLE) {
      //                //place column, so we can make the conversion on the
      //                specified processor
      //                ColumnPtr placed_col = copy_if_required(source_column,
      //                param.proc_spec);
      //                if(!placed_col) return ColumnPtr();
      //                if(this->getColumnType()==source_column->getColumnType()){
      //                    decompressed_source_column =
      //                    boost::dynamic_pointer_cast<DoubleDenseValueColumn>(placed_col);
      //                    assert(decompressed_source_column!=NULL);
      //                }else{
      //                    //decompress column or materialize intermediate
      //                    column
      //
      //                    DenseValueColumnPtr dense_value_col = placed_col->
      //                }
      //            }else{
      converted_column =
          source_column->convertToDenseValueDoubleColumn(param.proc_spec);
    }

    if (!converted_column) return ColumnPtr();

    // now check this column whether it is already a double or not,
    // and convert it to a double column if neccessary
    // call function again, this time the types match
    if (this->getType() == DOUBLE) {
      return this->column_algebra_operation(converted_column, param);
    } else {
      DoubleDenseValueColumnPtr double_column =
          this->convertToDenseValueDoubleColumn(param.proc_spec);
      if (!double_column) return ColumnPtr();
      return double_column->column_algebra_operation(converted_column, param);
    }
  }

  if (this->size() != source_column->size()) {
    COGADB_FATAL_ERROR(
        "Cannot perform algebra operation on columns with different number of "
        "elements!",
        "");
    return ColumnPtr();
  }
  // place column to correct processor memory
  ColumnPtr placed_col = copy_if_required(source_column, param.proc_spec);
  if (!placed_col) return ColumnPtr();

  // omit decompression or intermediate result construction if source column
  // is already a DenseValueColumn, materialize otherwise
  if (this->getColumnType() == source_column->getColumnType()) {
    DenseValueColumnPtr dense_source_column =
        boost::dynamic_pointer_cast<DenseValueColumn>(placed_col);
    assert(dense_source_column != NULL);
    return this->column_algebra_operation(*dense_source_column.get(), param);
  } else {
    boost::shared_ptr<ColumnBaseTyped<T> > typed_source_column =
        boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(placed_col);
    assert(typed_source_column != NULL);
    DenseValueColumnPtr dense_source_column =
        typed_source_column->copyIntoDenseValueColumn(param.proc_spec);
    if (!dense_source_column) return ColumnPtr();
    return this->column_algebra_operation(*dense_source_column.get(), param);
  }
}

template <class T>
const ColumnPtr Column<T>::column_algebra_operation(
    Column<T> &source_column, const AlgebraOperationParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);
  if (!dense_column) return ColumnPtr();
  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);
  if (backend->column_algebra_operation(dense_column->data(),
                                        source_column.data(),
                                        dense_column->size(), param)) {
    return dense_column;
  } else {
    return ColumnPtr();
  }
}

/* \todo add flag whether to perform this in place to save copy operation */
template <class T>
const ColumnPtr Column<T>::column_algebra_operation(
    const boost::any &value, const AlgebraOperationParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;

  T typed_value = T();

  if (value.type() == typeid(T)) {
    // same type, no problem
    typed_value = boost::any_cast<T>(value);
    DenseValueColumnPtr dense_column =
        this->copyIntoDenseValueColumn(param.proc_spec);
    if (!dense_column) return ColumnPtr();
    ProcessorBackend<T> *backend =
        ProcessorBackend<T>::get(param.proc_spec.proc_id);
    if (backend->column_algebra_operation(
            dense_column->data(), dense_column->size(), typed_value, param)) {
      return dense_column;
    } else {
      return ColumnPtr();
    }
  } else {
    // different type, we convert everything to double
    double double_value = 0;
    // convert constant to double
    bool ret_success =
        getValueFromAny(this->name_, value, double_value, this->db_type_);
    if (!ret_success) ColumnPtr();
    // convert this column in double column
    DoubleDenseValueColumnPtr double_column =
        this->convertToDenseValueDoubleColumn(param.proc_spec);
    if (!double_column) return ColumnPtr();
    return double_column->column_algebra_operation(boost::any(double_value),
                                                   param);
  }
}

template <class T>
const PositionListPtr Column<T>::sort(const SortParam &param) {
  return this->sort(param, false);  // false);
}

template <class T>
const PositionListPtr Column<T>::sort(const SortParam &param, bool no_copy) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;

  T *data = this->data();
  DenseValueColumnPtr copy_col;
  if (!no_copy) {
    copy_col = this->copyIntoDenseValueColumn(param.proc_spec);
    assert(copy_col != NULL);
    data = copy_col->data();
  }

  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);
  return backend->sort(data, this->size(), param);
}

template <>
const PositionListPtr Column<std::string>::sort(const SortParam &param) {
  PositionListPtr ids = createPositionList();
  ids->reserve(this->size());
  std::vector<std::pair<std::string, TID> > v;
  v.reserve(this->size());

  for (unsigned int i = 0; i < this->size(); i++) {
    v.push_back(std::pair<std::string, TID>((*this)[i], i));
  }

  if (param.order == ASCENDING) {
    std::stable_sort(v.begin(), v.end(),
                     std::less_equal<std::pair<std::string, TID> >());
  } else if (param.order == DESCENDING) {
    std::stable_sort(v.begin(), v.end(),
                     std::greater_equal<std::pair<std::string, TID> >());

  } else {
    std::cout << "FATAL ERROR: Column<T>::sort(): Unknown Sorting Order!"
              << std::endl;
  }

  for (unsigned int i = 0; i < v.size(); i++) {
    ids->push_back(v[i].second);
  }

  return ids;
}

template <>
const PositionListPtr Column<char *>::sort(const SortParam &param) {
  PositionListPtr ids = createPositionList();
  ids->reserve(this->size());
  std::vector<std::pair<char *, TID> > v;
  v.reserve(this->size());

  for (unsigned int i = 0; i < this->size(); i++) {
    v.push_back(std::pair<char *, TID>((*this)[i], i));
  }

  if (param.order == ASCENDING) {
    std::stable_sort(v.begin(), v.end(),
                     std::less_equal<std::pair<char *, TID> >());
  } else if (param.order == DESCENDING) {
    std::stable_sort(v.begin(), v.end(),
                     std::greater_equal<std::pair<char *, TID> >());

  } else {
    std::cout << "FATAL ERROR: Column<T>::sort(): Unknown Sorting Order!"
              << std::endl;
  }

  for (unsigned int i = 0; i < v.size(); i++) {
    ids->push_back(v[i].second);
  }

  return ids;
}

/***************** relational operations on Columns which return lookup tables
 * *****************/

template <class T>
bool Column<T>::store_impl(const std::string &path,
                           boost::archive::binary_oarchive &oa) {
  std::string path_to_column(path);
  path_to_column += "/";
  path_to_column += this->name_;
  // write additional information to output archive
  oa << this->num_elements_;
  // write actual data in seperate file
  FILE *file;
  std::string data_path = path_to_column + ".data";
  file = fopen(data_path.c_str(), "wb");
  if (!file) return false;
  size_t written_elements =
      fwrite(this->values_, sizeof(T), this->num_elements_, file);
  if (written_elements != this->num_elements_) {
    COGADB_FATAL_ERROR(
        "Did not write as many elements as required during store operation of "
        "column '"
            << this->getName() << "'!",
        "");
  }
  int err = fclose(file);
  assert(err == 0);
  return true;
}

template <>
bool Column<std::string>::store_impl(const std::string &path,
                                     boost::archive::binary_oarchive &oa) {
  // write additional information to output archive
  oa << this->num_elements_;
  for (size_t i = 0; i < this->num_elements_; ++i) {
    oa << values_[i];
  }
  // output is flushed by parent class
  return true;
}

template <class T>
bool Column<T>::load_impl(const std::string &path,
                          boost::archive::binary_iarchive &ia) {
  std::string path_to_column(path);
  path_to_column += "/";
  path_to_column += this->name_;

  ia >> this->num_elements_;

  FILE *file;
  std::string data_path = path_to_column + ".data";
  file = fopen(data_path.c_str(), "rb");
  if (!file) return false;
  this->reserve(this->num_elements_);
  size_t read_elements =
      fread(this->values_, sizeof(T), this->num_elements_, file);
  if (read_elements != this->num_elements_) {
    COGADB_FATAL_ERROR(
        "Did not read as many elements as required during load operation of "
        "column '"
            << this->getName() << "'!",
        "");
  }
  int err = fclose(file);
  assert(err == 0);

  return true;
}

template <>
bool Column<std::string>::load_impl(const std::string &path,
                                    boost::archive::binary_iarchive &ia) {
  ia >> this->num_elements_;

  this->reserve(this->num_elements_);

  for (size_t i = 0; i < this->num_elements_; ++i) {
    ia >> values_[i];
  }
  // input archive is closed by parent class
  return true;
}

template <class T>
bool Column<T>::isMaterialized() const throw() {
  return true;
}

template <class T>
bool Column<T>::isCompressed() const throw() {
  return false;
}

template <class T>
T *Column<T>::data() throw() {
  return this->values_;  // hype::util::begin_ptr(this->values_);
}

template <class T>
size_t Column<T>::getSizeinBytes() const throw() {
  return this->capacity() * sizeof(T)
         // this->has_primary_key_constraint_, this->has_foreign_key_constraint_
         + 2 * sizeof(bool)
         // this->fk_constr_
         + this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
         this->fk_constr_.getNameOfForeignKeyTable().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
}

// total template specialization

template <>
inline size_t Column<std::string>::getSizeinBytes() const throw() {
  unsigned long size_in_bytes = 0;
  for (unsigned int i = 0; i < this->size(); ++i) {
    size_in_bytes += values_[i].capacity();
  }

  return size_in_bytes + 2 * sizeof(bool) +
         this->fk_constr_.getNameOfForeignKeyColumn().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyColumn().capacity() +
         this->fk_constr_.getNameOfForeignKeyTable().capacity() +
         this->fk_constr_.getNameOfPrimaryKeyTable().capacity();
}

/***************** End of Implementation Section ******************/

#ifdef ENABLE_CDK_USAGE

template <class T>
const PositionListPairPtr Column<T>::hash_join(ColumnPtr join_column_) {
  assert(this->size() <= join_column_->size());
  if (join_column_->type() != typeid(T)) {
    std::cout << "Fatal Error!!! Typemismatch for columns " << this->name_
              << " and " << join_column_->getName() << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }

  T *input_column = NULL;
  if (join_column_->isMaterialized()) {
    shared_pointer_namespace::shared_ptr<Column<T> > join_column =
        shared_pointer_namespace::dynamic_pointer_cast<Column<T> >(
            join_column_);
    assert(join_column != NULL);
    input_column =
        join_column->data();  // hype::util::begin_ptr(join_column->values_);
    assert(input_column != NULL);

    T *this_array = this->data();  // hype::util::begin_ptr(this->values_);
    assert(this_array != NULL);
    return CDK::join::serial_hash_join(this_array, this->size(), input_column,
                                       join_column->size());

  } else if (!join_column_->isMaterialized() &&
             !join_column_->isCompressed()) {  // Lookup Array?
    shared_pointer_namespace::shared_ptr<LookupArray<T> > join_column =
        shared_pointer_namespace::dynamic_pointer_cast<LookupArray<T> >(
            join_column_);
    assert(join_column != NULL);
    T *input_column = join_column->materializeToArray();

    T *this_array = this->data();  // hype::util::begin_ptr(this->values_);
    assert(this_array != NULL);
    // PositionListPairPtr ret = CDK::join::serial_hash_join(this_array,
    // this->size(), input_column, join_column->size());
    PositionListPairPtr ret = CDK::join::serial_hash_join(
        this_array, this->size(), input_column, join_column->size());

    // ok, job done, cleanup temporary array
    free(input_column);
    return ret;
  } else if (join_column_->isCompressed()) {
    // if compressed, use old style hash join via ColumnBaseTyped class
    return CoGaDB::ColumnBaseTyped<T>::hash_join(join_column_);
  } else {
    // detected impossible case, abort
    COGADB_FATAL_ERROR(
        "Detected Invalid Column Class! Not MAterialized, No LookupARray and "
        "no CompressedColumn!",
        "");
    return PositionListPairPtr();
  }
}

template <class T>
const PositionListPairPtr Column<T>::radix_join(ColumnPtr join_column_) {
  COGADB_FATAL_ERROR(
      "called non INT instantiation of Column<T>::radix_join, which only "
      "supports integer columns!",
      "");
  return PositionListPairPtr();
}

template <>
const PositionListPairPtr Column<int>::radix_join(ColumnPtr join_column_) {
  assert(this->size() <= join_column_->size());

  if (join_column_->type() != typeid(int)) {
    std::cout << "Fatal Error!!! Typemismatch for columns " << this->name_
              << " and " << join_column_->getName() << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }

  int *input_column = NULL;
  if (join_column_->isMaterialized()) {
    shared_pointer_namespace::shared_ptr<Column<int> > join_column =
        shared_pointer_namespace::dynamic_pointer_cast<Column<int> >(
            join_column_);
    assert(join_column != NULL);
    input_column =
        join_column->data();  // hype::util::begin_ptr(join_column->values_);
    assert(input_column != NULL);

    int *this_array = this->data();  // hype::util::begin_ptr(this->values_);
    assert(this_array != NULL);
    return CDK::join::serial_hash_join(this_array, this->size(), input_column,
                                       join_column->size());

  } else if (!join_column_->isMaterialized() &&
             !join_column_->isCompressed()) {  // Lookup Array?
    shared_pointer_namespace::shared_ptr<LookupArray<int> > join_column =
        shared_pointer_namespace::dynamic_pointer_cast<LookupArray<int> >(
            join_column_);
    assert(join_column != NULL);

    int *input_column = join_column->materializeToArray();

    int *this_array = this->data();  // hype::util::begin_ptr(this->values_);
    assert(this_array != NULL);
    PositionListPairPtr ret = CDK::join::radix_join(
        this_array, this->size(), input_column,
        join_column->size());  // serial_hash_join(this_array, this->size(),
                               // input_column, join_column->size());
    // ok, job done, cleanup temporary array
    free(input_column);
    return ret;
  } else if (join_column_->isCompressed()) {
    // if compressed, use old style hash join via ColumnBaseTyped class
    return CoGaDB::ColumnBaseTyped<int>::hash_join(join_column_);
  } else {
    // detected impossible case, abort
    COGADB_FATAL_ERROR(
        "Detected Invalid Column Class! Not MAterialized, No LookupARray and "
        "no CompressedColumn!",
        "");
    return PositionListPairPtr();
  }
}

#endif

template <class T>
void Column<T>::reserve(size_t new_capacity) {
  if (new_capacity > buffer_size_) {
    // values_=(T*) realloc(values_,new_capacity*sizeof(T));
    T *tmp = this->mem_alloc->reallocate(values_, new_capacity);
    if (tmp == NULL) {
      throw std::bad_alloc();
    } else {
      /* did memory location move? => notify ocl data caches
       * that old memory range is outdated
       * as of now, we do not recache moved memory regions
       */
      if (values_ != tmp && values_ != NULL) {
        OCL_DataCaches::instance().uncacheMemoryArea(values_,
                                                     sizeof(T) * buffer_size_);
      }
      values_ = tmp;
    }
    buffer_size_ = new_capacity;
  } else if (new_capacity == 0) {
    return reserve(DEFAULT_COLUMN_SIZE);
  }
}

template <class T>
void Column<T>::resize(size_t new_size) {
  if (new_size > buffer_size_) {
    this->reserve(new_size);
    num_elements_ = new_size;
  } else {
    num_elements_ = new_size;
  }
}

template <class T>
T *Column<T>::begin() {
  if (this->empty()) {
    return NULL;
  } else {
    return values_;
  }
}

template <class T>
const T *Column<T>::begin() const {
  if (this->empty()) {
    return NULL;
  } else {
    return values_;
  }
}

template <class T>
T *Column<T>::end() {
  if (this->empty()) {
    return NULL;
  } else {
    return &values_[num_elements_];  // one element further than last element
  }
}

template <class T>
const T *Column<T>::end() const {
  if (this->empty()) {
    return NULL;
  } else {
    return &values_[num_elements_];  // one element further than last element
  }
}

template <class T>
bool Column<T>::empty() const {
  if (num_elements_ > 0)
    return false;
  else
    return true;
}

template <class T>
void Column<T>::clear() {
  num_elements_ = 0;
  // resize array to default size
  values_ = mem_alloc->reallocate(values_, DEFAULT_COLUMN_SIZE);
  this->buffer_size_ = DEFAULT_COLUMN_SIZE;
}

template <class T>
void selection_thread(unsigned int thread_id, unsigned int number_of_threads,
                      const T &value, const ValueComparator comp,
                      ColumnBaseTyped<T> *col, TID *result_tids,
                      size_t *result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  ColumnBaseTyped<T> &column_ref = dynamic_cast<ColumnBaseTyped<T> &>(*col);
  // number of elements per thread
  size_t chunk_size = column_ref.size() / number_of_threads;
  size_t begin_index = chunk_size * thread_id;
  size_t end_index;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    end_index = column_ref.size();
  } else {
    end_index = (chunk_size) * (thread_id + 1);
  }

  size_t pos = 0;

  if (comp == EQUAL) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (value == column_ref[i]) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else if (comp == LESSER) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (column_ref[i] < value) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else if (comp == LESSER_EQUAL) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (column_ref[i] <= value) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else if (comp == GREATER) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (column_ref[i] > value) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else if (comp == GREATER_EQUAL) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (column_ref[i] >= value) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else {
    COGADB_FATAL_ERROR("Unsupported Filter Predicate!", "");
  }
  //}

  // write result size to array
  *result_size = pos;
}

template <>
void selection_thread<int>(unsigned int thread_id,
                           unsigned int number_of_threads, const int &value,
                           const ValueComparator comp,
                           ColumnBaseTyped<int> *col, TID *result_tids,
                           size_t *result_size) {
  // std::cout << "Hi I'm thread" << thread_id << std::endl;
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  ColumnBaseTyped<int> &column_ref = dynamic_cast<ColumnBaseTyped<int> &>(*col);
  // number of elements per thread
  size_t chunk_size = column_ref.size() / number_of_threads;
  size_t begin_index = chunk_size * thread_id;
  size_t end_index;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    end_index = column_ref.size();
  } else {
    end_index = (chunk_size) * (thread_id + 1);
  }
  size_t pos = 0;

#ifdef ENABLE_SIMD_ACCELERATION
  // if column is materialized, we can use our SIMD SCAN; ensure we have enough
  // elements, otherwise SIMD SCAN might crash
  if (col->isMaterialized() && col->size() >= 300) {
    Column<int> *dense_value_column = dynamic_cast<Column<int> *>(col);
    assert(dense_value_column != NULL);
    vector<int> &v = dense_value_column->getContent();
    int *array = hype::util::begin_ptr(v);
    assert(array != NULL);
    unsigned int array_size = end_index - begin_index;
    simd_selection_int_thread(&array[begin_index], array_size, value, comp,
                              &result_tids[begin_index], &pos, begin_index);
    // write result size to array
    *result_size = pos;
    return;
  }
#endif

  if (comp == EQUAL) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (value == column_ref[i]) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else if (comp == LESSER) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (column_ref[i] < value) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else if (comp == LESSER_EQUAL) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (column_ref[i] <= value) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else if (comp == GREATER) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (column_ref[i] > value) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else if (comp == GREATER_EQUAL) {
    for (TID i = begin_index; i < end_index; ++i) {
      if (column_ref[i] >= value) {
        result_tids[pos + begin_index] = i;
        pos++;
      }
    }
  } else {
    COGADB_FATAL_ERROR("Unsupported Filter Predicate!", "");
  }

  // write result size to array
  *result_size = pos;
}

void write_selection_result_thread(
    unsigned int thread_id, unsigned int number_of_threads, size_t column_size,
    TID *result_tids_array, PositionListPtr result_tids, size_t result_size,
    size_t result_begin_index, size_t result_end_index) {
  assert(result_tids->size() == column_size);

  // number of elements per thread
  size_t chunk_size = column_size / number_of_threads;
  TID begin_index = chunk_size * thread_id;

  TID *position_list_output_array =
      result_tids->data();  // hype::util::begin_ptr(*result_tids);

  std::memcpy(&position_list_output_array[result_begin_index],
              &result_tids_array[begin_index], result_size * sizeof(TID));
}

void resize_PositionListPtr_thread(PositionListPtr tids, size_t new_size) {
  assert(tids != NULL);
  tids->resize(new_size);
}

template <class T>
const PositionListPtr ColumnBaseTyped<T>::parallel_selection(
    const boost::any &value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  PositionListPtr result_tids = createPositionList();
  // unsigned int number_of_threads=4;

  if (value_for_comparison.type() != typeid(T)) {
    std::cout << "Fatal Error!!! Typemismatch for column " << name_
              << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }

  T value = boost::any_cast<T>(value_for_comparison);

  TID *result_tids_array = (TID *)malloc(this->size() * sizeof(TID));
  if (result_tids_array == NULL) {
    COGADB_FATAL_ERROR("Malloc failed! While trying to allocate "
                           << this->size() * sizeof(TID) << " bytes",
                       "");
  }
  // resize PositionList, this will _NOT_ initialize the TIDs in the List!
  //               result_tids->resize(this->size());
  // std::vector<PositionListPtr> local_result_arrays;

  std::vector<size_t> result_sizes(number_of_threads);
  boost::thread_group threads;
  // create a PositionListPtr wof the maximal result size, so
  // that we can write the result tids in parallel to th vector
  // without the default latency (vector allocates + initializes each element)
  threads.add_thread(new boost::thread(boost::bind(
      &CoGaDB::resize_PositionListPtr_thread, result_tids, this->size())));
  for (unsigned int i = 0; i < number_of_threads; i++) {
    // create a selection thread
    threads.add_thread(new boost::thread(
        boost::bind(&CoGaDB::selection_thread<T>, i, number_of_threads, value,
                    comp, this, result_tids_array, &result_sizes[i])));
    // selection_thread(i, number_of_threads, value,  comp,
    // this,result_tids_array, &result_sizes[i]);
  }
  threads.join_all();

  std::vector<size_t> prefix_sum(number_of_threads + 1);
  prefix_sum[0] = 0;
  for (unsigned int i = 1; i < number_of_threads + 1; i++) {
    prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
  }

  // copy result chunks in vector
  // size_t chunk_size = this->size()/number_of_threads;
  for (unsigned int i = 0; i < number_of_threads; i++) {
    // cout << "thread " << i << " start index: " << begin_index << " end index:
    // " << begin_index+result_sizes[i] << endl;

    threads.add_thread(new boost::thread(
        boost::bind(&write_selection_result_thread, i, number_of_threads,
                    this->size(), result_tids_array, result_tids,
                    result_sizes[i], prefix_sum[i], prefix_sum[i + 1])));
    // write_selection_result_thread(i, number_of_threads, this->size(),
    // result_tids_array, result_tids, result_sizes[i], prefix_sum[i],
    // prefix_sum[i+1]);

    // result_tids->insert(result_tids->end(),
    // &result_tids_array[begin_index],&result_tids_array[begin_index+result_sizes[i]]);
  }
  threads.join_all();
  // fit positionlist to actual result length
  result_tids->resize(prefix_sum[number_of_threads]);

  free(result_tids_array);

  return result_tids;
}

template <class T>
const PositionListPtr Column<T>::selection(const SelectionParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);

  if (param.pred_type == ValueConstantPredicate) {
    T value;
    bool ret_success =
        getValueFromAny(this->name_, param.value, value, this->db_type_);
    if (!ret_success) PositionListPtr();
    // update parameter for correct type
    SelectionParam new_param(param);
    new_param.value = boost::any(value);
    return backend->tid_selection(this->data(), this->size(), new_param);
  } else if (param.pred_type == ValueValuePredicate) {
    assert(param.comparison_column != NULL);
    if (this->getType() != param.comparison_column->getType()) {
      if (this->getType() != VARCHAR &&
          param.comparison_column->getType() != VARCHAR) {
        SelectionParam new_param(param);
        // omit type cast in case comparison column has type double
        if (param.comparison_column->getType() != DOUBLE) {
          // comparison column can be an intermediate result
          // or compressed etc. and have an arbitrary type
          // besides VARCHAR
          //                              boost::shared_ptr<ColumnBaseTyped<T> >
          //                              typed_column =
          //                              boost::dynamic_pointer_cast<ColumnBaseTyped<T>
          //                              >(param.comparison_column);
          //                              assert(typed_column!=NULL);
          // copy into double column
          DoubleDenseValueColumnPtr dense_column =
              param.comparison_column->convertToDenseValueDoubleColumn(
                  param.proc_spec);
          if (!dense_column) return PositionListPtr();
          // adapt parameter and call the function again but
          // this time, the comparison column has type double
          new_param.comparison_column = dense_column;
        }
        // re-execute selection, but this time, the
        // comparison column has type double as well
        if (this->getType() == DOUBLE) {
          // save copy operation if the this column is
          // already a double
          return this->selection(new_param);
        } else {
          // convert this column to double type as well
          DoubleDenseValueColumnPtr this_dense_column =
              this->convertToDenseValueDoubleColumn(param.proc_spec);
          if (!this_dense_column) return PositionListPtr();
          return this_dense_column->selection(new_param);
        }

      } else {
        COGADB_FATAL_ERROR(
            "Cannot compare columns with different types, where one column is "
            "a VARCHAR Type and the other is not!",
            "");
        return PositionListPtr();
      }
    }

    PositionListPtr result;

    if (this->getColumnType() == param.comparison_column->getColumnType()) {
      DenseValueColumnPtr column =
          boost::dynamic_pointer_cast<DenseValueColumn>(
              param.comparison_column);
      assert(column != NULL);
      // place column
      ColumnPtr placed_column = copy_if_required(column, param.proc_spec);
      if (!placed_column) return PositionListPtr();
      // type cast column back and set it as new comparison column
      DenseValueColumnPtr typed_column =
          boost::dynamic_pointer_cast<DenseValueColumn>(placed_column);
      SelectionParam new_param(param);
      new_param.comparison_column = typed_column;

      assert(new_param.comparison_column->getMemoryID() ==
             hype::util::getMemoryID(new_param.proc_spec.proc_id));
      return backend->tid_selection(this->data(), this->size(), new_param);
    } else {
      boost::shared_ptr<ColumnBaseTyped<T> > typed_column =
          boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(
              param.comparison_column);
      assert(typed_column != NULL);
      DenseValueColumnPtr dense_column =
          typed_column->copyIntoDenseValueColumn(param.proc_spec);
      if (!dense_column) return PositionListPtr();

      SelectionParam new_param(param);
      new_param.comparison_column = dense_column;
      result = this->selection(new_param);
    }
    return result;
  } else {
    COGADB_FATAL_ERROR("Invalid Predicate Type!", "");
  }
  return PositionListPtr();
}

#ifdef ENABLE_SIMD_ACCELERATION
template <>
inline const PositionListPtr Column<int>::selection(
    const boost::any &value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  int value;

  if (value_for_comparison.type() != typeid(int)) {
    COGADB_FATAL_ERROR(std::string("Typemismatch for column") + this->name_ +
                           std::string(" Column Type: ") + typeid(int).name() +
                           std::string(" filter value type: ") +
                           value_for_comparison.type().name(),
                       "");
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<int>(value_for_comparison);
  }
  unsigned int array_size = this->size();
  int *array = hype::util::begin_ptr(values_);

  return CoGaDB::simd_selection_int(array, array_size, value, comp);
}
#endif

#ifdef ENABLE_SIMD_ACCELERATION
template <>
inline const PositionListPtr Column<float>::selection(
    const boost::any &value_for_comparison, const ValueComparator comp) {
  // std::cout << "SIMD SCAN" << std::endl;
  float value;

  if (value_for_comparison.type() != typeid(float)) {
    // allow comparison with itnegers as well
    if (value_for_comparison.type() == typeid(int)) {
      value = boost::any_cast<int>(value_for_comparison);
    } else {
      COGADB_FATAL_ERROR(std::string("Typemismatch for column") + this->name_ +
                             std::string(" Column Type: ") +
                             typeid(float).name() +
                             std::string(" filter value type: ") +
                             value_for_comparison.type().name(),
                         "");
    }
  } else {
    // everything fine, filter value matches type of column
    value = boost::any_cast<float>(value_for_comparison);
  }
  unsigned int array_size = this->size();
  float *array = hype::util::begin_ptr(values_);

  return CoGaDB::simd_selection_float(array, array_size, value, comp);

  //            PositionListPtr result_tids=createPositionList();
  //            //result_tids->reserve(array_size);
  ////            //ensure that at least one element is in output buffer, so we
  /// can get a pointer to it
  //            //result_tids->push_back(0);
  //            //get the pointer to the output buffer
  //            unsigned int* result_array = new unsigned int[array_size];
  //            //hype::util::begin_ptr(*result_tids); //new int[array_size];
  //            assert(result_array!=NULL);
  //
  //            unsigned int result_size=0;
  //            float comparison_value=value;
  //            if(comp==EQUAL){
  //                COGADB_SIMD_SCAN_FLOAT(array,array_size,result_array,comparison_value,_mm_cmpeq_ps,==,result_size);
  //            }else if(comp==LESSER){
  //                COGADB_SIMD_SCAN_FLOAT(array,array_size,result_array,comparison_value,_mm_cmplt_ps,<,result_size);
  //             }else if(comp==LESSER_EQUAL){
  //                //add one to comparison value to compare for less equal
  //                COGADB_SIMD_SCAN_FLOAT(array,array_size,result_array,comparison_value,_mm_cmple_ps,<=,result_size);
  //            }else if(comp==GREATER){
  //                COGADB_SIMD_SCAN_FLOAT(array,array_size,result_array,comparison_value,_mm_cmpgt_ps,>,result_size);
  //            }else if(comp==GREATER_EQUAL){
  //                //substract one of comparison value to compare for greater
  //                equal
  //                COGADB_SIMD_SCAN_FLOAT(array,array_size,result_array,comparison_value,_mm_cmpge_ps,>=,result_size);
  //            }else{
  //                COGADB_FATAL_ERROR("Invalid ValueComparator!","");
  //            }
  //
  //            /**********************************************************************************/
  //
  //
  //
  //            /**********************************************************************************/
  //
  //             result_tids->insert(result_tids->end(),result_array,result_array+result_size);
  //
  //             delete result_array;
  //
  //             //result_tids->resize(result_size);
  //             return result_tids;
}
#endif

template <typename T>
const BitmapPtr Column<T>::bitmap_selection(const SelectionParam &param) {
  assert(param.pred_type == ValueConstantPredicate);

  T value;
  bool ret_success =
      getValueFromAny(this->name_, param.value, value, this->db_type_);
  if (!ret_success) BitmapPtr();
  // update parameter for correct type
  SelectionParam new_param(param);
  new_param.value = boost::any(value);

  ProcessorBackend<T> *backend =
      ProcessorBackend<T>::get(param.proc_spec.proc_id);
  return backend->bitmap_selection(this->data(), this->size(), new_param);

  //            T* values=this->data();
  //            return CDK::selection::bitmap_selection(values, this->size(),
  //            value_for_comparison, comp);
}

template <typename T>
void ColumnBaseTyped<T>::hash_join_pruning_thread(
    ColumnBaseTyped<T> *join_column, HashTable *hashtable,
    TID *join_tids_table1, TID *join_tids_table2, size_t thread_id,
    size_t number_of_threads, size_t *result_size) {
  ColumnBaseTyped<T> &join_column_ref =
      dynamic_cast<ColumnBaseTyped<T> &>(*join_column);
  // number of elements per thread
  size_t chunk_size = join_column_ref.size() / number_of_threads;
  TID begin_index = chunk_size * thread_id;
  TID end_index;
  if (thread_id + 1 == number_of_threads) {
    // process until end of input array
    end_index = join_column_ref.size();
  } else {
    end_index = (chunk_size) * (thread_id + 1);
  }
  if (!quiet && verbose && debug)
    std::cout << "Parallel Hash Join for array of size: "
              << join_column_ref.size() << " thread id: " << thread_id
              << " chunksize: " << chunk_size
              << " elements  begin index: " << begin_index
              << " end index: " << end_index << std::endl;
  TID pos1 = begin_index;
  TID pos2 = begin_index;

  std::pair<typename HashTable::iterator, typename HashTable::iterator> range;
  typename HashTable::iterator it;
  for (size_t i = begin_index; i < end_index; i++) {
    range = hashtable->equal_range(join_column_ref[i]);
    for (it = range.first; it != range.second; ++it) {
      if (it->first == join_column_ref[i]) {  //(*join_column)[i]){
        join_tids_table1[pos1++] = it->second;
        join_tids_table2[pos2++] = i;
        if (!quiet && verbose && debug)
          std::cout << "match! " << it->second << ", " << i << "	"
                    << it->first << std::endl;
      }
    }
  }
  assert(pos1 == pos2);
  *result_size = pos1 - begin_index;
  if (!quiet && verbose && debug)
    cout << "Thread " << thread_id << " Result size: " << *result_size << endl;
}

template <typename T>
void ColumnBaseTyped<T>::join_write_result_chunk_thread(
    ColumnBaseTyped<T> *join_column, TID *join_tids_table1,
    TID *join_tids_table2, TID *join_tids_result_table1,
    TID *join_tids_result_table2, size_t thread_id, size_t number_of_threads,
    TID begin_index_result, TID end_index_result) {
  assert(join_column != NULL);
  // assert(end_index_result-begin_index_result==result_size);
  size_t result_size = end_index_result - begin_index_result;
  // ColumnBaseTyped<T>& join_column_ref = dynamic_cast< ColumnBaseTyped<T>&
  // >(*join_column);
  // number of elements per thread
  size_t chunk_size = join_column->size() / number_of_threads;
  TID begin_index = chunk_size * thread_id;
  TID end_index = (chunk_size * thread_id) + result_size;
  //            if(thread_id+1 == number_of_threads){
  //                //process until end of input array
  //                end_index=join_column->size();
  //            }else{
  //                end_index=(chunk_size*thread_id)+result_size;
  //            }

  assert(join_column != NULL);
  if (!quiet && verbose && debug)
    cout << "Maximal result tuples for parallel PK FK Join: "
         << join_column->size() << endl;
  // end_index=(chunk_size*thread_id)+result_size;
  if (!quiet && verbose && debug)
    std::cout << "Parallel Hash Join for array of size: " << join_column->size()
              << " thread id: " << thread_id << " chunksize: " << chunk_size
              << " elements  begin index: " << begin_index
              << " end index: " << end_index << std::endl;
  // size_t pos = begin_index;
  // size_t pos2 = begin_index;

  // copy memory chunk to result array
  std::memcpy(&join_tids_result_table1[begin_index_result],
              &join_tids_table1[begin_index], result_size * sizeof(TID));
  std::memcpy(&join_tids_result_table2[begin_index_result],
              &join_tids_table2[begin_index], result_size * sizeof(TID));
}

template <class T>
const PositionListPairPtr ColumnBaseTyped<T>::parallel_hash_join(
    ColumnPtr join_column_, unsigned int number_of_threads) {
  if (join_column_->type() != typeid(T)) {
    std::cout << "Fatal Error!!! Typemismatch for columns " << this->name_
              << " and " << join_column_->getName() << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }
  // this algorithms uses special optmizations for int and float values
  assert(join_column_->getType() != VARCHAR);

  shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> > join_column =
      shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<T> >(
          join_column_);  // static_cast<IntTypedColumnPtr>(column1);

  PositionListPairPtr join_tids(new PositionListPair());
  join_tids->first = createPositionList();
  join_tids->second = createPositionList();

  Timestamp build_hashtable_begin = getTimestamp();
  // create hash table
  HashTable hashtable;
  size_t hash_table_size = this->size();
  size_t join_column_size = join_column->size();

  assert(join_column_->size() >= this->size());
  //               unsigned int* join_tids_table1 =  new unsigned
  //               int[join_column_size];
  //               unsigned int* join_tids_table2 =  new unsigned
  //               int[join_column_size];

  TID *join_tids_table1 = (TID *)malloc(join_column_size * sizeof(TID));
  TID *join_tids_table2 = (TID *)malloc(join_column_size * sizeof(TID));

  //               ColumnBaseTyped<T>& join_column_ref = dynamic_cast<
  //               ColumnBaseTyped<T>& >(*join_column);

  for (unsigned int i = 0; i < hash_table_size; i++)
    hashtable.insert(std::pair<T, TID>((*this)[i], i));
  Timestamp build_hashtable_end = getTimestamp();

  // probe larger relation
  Timestamp prune_hashtable_begin = getTimestamp();
  std::vector<size_t> thread_result_sizes(number_of_threads);
  boost::thread_group threads;
  for (size_t i = 0; i < number_of_threads; i++) {
    // ColumnBaseTyped<T>::hash_join_pruning_thread(join_column.get(),
    // &hashtable, join_tids_table1, join_tids_table2, i, number_of_threads);
    // CoGaDB::hash_join_pruning_thread<T,HashTable>(join_column_ref,
    // &hashtable, join_tids_table1, join_tids_table2, i, number_of_threads);
    // threads.add_thread(new
    // boost::thread((&hash_join_pruning_thread<T,HashTable>),join_column_ref,
    // &hashtable, join_tids_table1, join_tids_table2, i, number_of_threads));
    // ColumnBaseTyped<T>& join_column_ref, HashTable* hashtable, unsigned int*
    // join_tids_table1, unsigned int* join_tids_table2, unsigned int thread_id,
    // unsigned int number_of_threads)
    threads.add_thread(new boost::thread(boost::bind(
        &ColumnBaseTyped<T>::hash_join_pruning_thread, join_column.get(),
        &hashtable, join_tids_table1, join_tids_table2, i,
        size_t(number_of_threads), &thread_result_sizes[i])));
  }
  threads.join_all();

  size_t total_result_size = 0;
  for (size_t i = 0; i < number_of_threads; i++) {
    total_result_size += thread_result_sizes[i];
  }
  // reserve memory
  join_tids->first->resize(total_result_size);
  join_tids->second->resize(total_result_size);

  TID *result_tids_table1 =
      join_tids->first->data();  // hype::util::begin_ptr(*join_tids->first);
  TID *result_tids_table2 =
      join_tids->second->data();  // hype::util::begin_ptr(*join_tids->second);
  assert(result_tids_table1 != NULL);
  assert(result_tids_table2 != NULL);
  assert(thread_result_sizes.size() > 0);
  // positions in result array the thread will write into
  TID begin_index = 0;
  TID end_index = thread_result_sizes[0];
  for (size_t i = 0; i < number_of_threads; i++) {
    if (!quiet && verbose && debug)
      cout << "Begin Index Result: " << begin_index
           << " End Index Result: " << end_index << endl;
    //(ColumnBaseTyped<T>* join_column, unsigned int* join_tids_table1, unsigned
    // int* join_tids_table2, unsigned int* join_tids_result_table1, unsigned
    // int* join_tids_result_table2, unsigned int thread_id, unsigned int
    // number_of_threads, unsigned int result_size, size_t begin_index_result,
    // size_t end_index_result)
    // threads.add_thread(new
    // boost::thread(boost::bind(&ColumnBaseTyped<T>::join_write_result_chunk_thread,
    // join_column.get(), join_tids_table1, join_tids_table2,
    // result_tids_table1, result_tids_table2, i, number_of_threads,
    // begin_index, end_index)));
    ColumnBaseTyped<T>::join_write_result_chunk_thread(
        join_column.get(), join_tids_table1, join_tids_table2,
        result_tids_table1, result_tids_table2, i, size_t(number_of_threads),
        begin_index, end_index);
    // if not last thread, then update indeces of result array
    if (i < (number_of_threads - 1)) {
      begin_index += thread_result_sizes[i];
      end_index += thread_result_sizes[i + 1];
      assert(end_index <= total_result_size);
      assert(begin_index <= total_result_size);
    }
    if (!quiet && verbose && debug)
      cout << "thread: " << i << " result size: " << thread_result_sizes[i]
           << " Total Result size: " << total_result_size << endl;
  }

  // put results in position list
  // join_tids->first->insert(join_tids->first->end(),join_tids_table1,join_tids_table1+join_column_size);
  // //assume PK FK Join!
  // join_tids->second->insert(join_tids->second->end(),join_tids_table2,join_tids_table2+join_column_size);
  // cleanup
  // delete join_tids_table1;
  // delete join_tids_table2;
  free(join_tids_table1);
  free(join_tids_table2);

  Timestamp prune_hashtable_end = getTimestamp();

  if (!quiet && verbose && debug)
    std::cout << "Hash Join: Build Phase: "
              << double(build_hashtable_end - build_hashtable_begin) /
                     (1000 * 1000)
              << "ms"
              << "Pruning Phase: "
              << double(prune_hashtable_end - prune_hashtable_begin) /
                     (1000 * 1000)
              << "ms" << std::endl;

  return join_tids;
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(Column)

}  // end namespace CoGaDB
