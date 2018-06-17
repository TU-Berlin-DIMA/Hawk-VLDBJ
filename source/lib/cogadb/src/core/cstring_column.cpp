
#include <backends/processor_backend.hpp>
#include <core/cstring_column.hpp>
#include <util/hardware_detector.hpp>
#include <util/types.hpp>

namespace CoGaDB {

StringHeap::StringHeap(const StringHeapType heap_type,
                       const size_t max_string_length,
                       const hype::ProcessingDeviceMemoryID mem_id)
    : num_elements_(0),
      max_string_length_(max_string_length),
      mem_id_(mem_id),
      heap_type_(heap_type) {}

bool StringHeap::load(const std::string& base_path,
                      const std::string& column_name,
                      boost::archive::binary_iarchive& ia) {
  ia >> num_elements_;
  ia >> heap_type_;
  ia >> max_string_length_;
  return load_impl(base_path, column_name, ia);
}

bool StringHeap::store(const std::string& base_path,
                       const std::string& column_name,
                       boost::archive::binary_oarchive& oa) {
  oa << num_elements_;
  oa << heap_type_;
  oa << max_string_length_;
  return store_impl(base_path, column_name, oa);
}

StringHeapType StringHeap::getStringHeapType() const { return heap_type_; }

size_t StringHeap::getMaxStringLength() const { return max_string_length_; }

hype::ProcessingDeviceMemoryID StringHeap::getMemoryID() const {
  return mem_id_;
}

size_t StringHeap::size() const { return num_elements_; }

FixedLengthStringHeap::FixedLengthStringHeap(
    const size_t max_string_length, const hype::ProcessingDeviceMemoryID mem_id)
    : StringHeap(FIXED_LENGTH_STRING_HEAP, max_string_length, mem_id),
      data_(new Column<char>("", CHAR)) {}

FixedLengthStringHeap::FixedLengthStringHeap(
    const size_t max_string_length, const size_t num_elements,
    const hype::ProcessingDeviceMemoryID mem_id)
    : StringHeap(FIXED_LENGTH_STRING_HEAP, max_string_length, mem_id),
      data_(new Column<char>("", CHAR)) {
  data_->reserve(num_elements * max_string_length);
}

char* FixedLengthStringHeap::getPointerArray() { return data_->data(); }

bool FixedLengthStringHeap::initStringArray(
    char** c_string_ptr_array, size_t num_elements,
    const hype::ProcessingDeviceMemoryID& mem_id) {
  if (mem_id != hype::PD_Memory_0) {
    COGADB_FATAL_ERROR("Did not implement special heap index array on "
                           << "others processors as CPU!",
                       "");
  }

  if (num_elements != this->size()) {
    COGADB_FATAL_ERROR("num_elements!=this->size()", "");
    return false;
  }
  char* data_array = data_->data();
  size_t max_string_length = this->getMaxStringLength();
  for (size_t i = 0; i < num_elements; ++i) {
    c_string_ptr_array[i] = &data_array[i * max_string_length];
  }
  return true;
}

char* FixedLengthStringHeap::push_back(const char* const val,
                                       bool& heap_memory_moved) {
  size_t required_bytes = getMaxStringLength();  // strlen(val)+1;
  if (required_bytes > getMaxStringLength()) {
    return NULL;
  }
  size_t current_size = data_->size();
  char* original_array = data_->data();
  data_->resize(data_->size() + getMaxStringLength());
  char* array = data_->data();
  /* did the realloc move the heap? */
  if (original_array == array) {
    heap_memory_moved = false;
  } else {
    heap_memory_moved = true;
  }
  char* new_cstring = &array[current_size];
  strncpy(new_cstring, val, required_bytes);
  ++num_elements_;
  return new_cstring;
}

void FixedLengthStringHeap::clear() { data_->clear(); }

const StringHeapPtr FixedLengthStringHeap::copy() const {
  return this->copy(this->getMemoryID());
}

const StringHeapPtr FixedLengthStringHeap::copy(
    const hype::ProcessingDeviceMemoryID& mem_id) const {
  boost::shared_ptr<FixedLengthStringHeap> copy(new FixedLengthStringHeap(
      this->getMaxStringLength(), this->size(), this->getMemoryID()));
  ColumnPtr heap_copy = data_->copy(mem_id);
  if (!heap_copy) {
    return StringHeapPtr();
  }
  copy->data_ = boost::dynamic_pointer_cast<Column<char> >(heap_copy);
  assert(copy->data_ != NULL);
  return copy;
}

void FixedLengthStringHeap::reserve(size_t new_capacity,
                                    bool& heap_memory_moved) {
  char* original_array = data_->data();
  data_->reserve(new_capacity * this->getMaxStringLength());
  char* new_array = data_->data();
  if (original_array != new_array) {
    heap_memory_moved = true;
  } else {
    heap_memory_moved = false;
  }
}

void FixedLengthStringHeap::resize(size_t new_size, bool& heap_memory_moved) {
  this->reserve(new_size, heap_memory_moved);
  data_->resize(new_size * this->getMaxStringLength());
  num_elements_ = new_size;
}

bool FixedLengthStringHeap::load_impl(const std::string& base_path,
                                      const std::string& column_name,
                                      boost::archive::binary_iarchive& ia) {
  std::stringstream name;
  name << column_name << ".fixed_length_string_heap";
  data_->setName(name.str());
  return data_->load(base_path, LOAD_ALL_DATA);
}

bool FixedLengthStringHeap::store_impl(const std::string& base_path,
                                       const std::string& column_name,
                                       boost::archive::binary_oarchive& oa) {
  std::stringstream name;
  name << column_name << ".fixed_length_string_heap";
  data_->setName(name.str());
  return data_->store(base_path);
}

size_t FixedLengthStringHeap::getSizeinBytes() const throw() {
  return data_->getSizeinBytes();
}

void FixedLengthStringHeap::print() const {
  char* array = data_->data();
  std::cout << " === HEAP START ========================= " << std::endl;
  for (size_t i = 0; i < data_->size(); ++i) {
    if (i % this->getMaxStringLength() == 0) {
      std::cout << std::endl;
    }
    std::cout << array[i];
  }
  std::cout << " === HEAP END   ========================= " << std::endl;
}

const StringHeapPtr createStringHeap(
    StringHeapType type, size_t max_string_length,
    const hype::ProcessingDeviceMemoryID mem_id) {
  assert(FIXED_LENGTH_STRING_HEAP == type);
  return StringHeapPtr(new FixedLengthStringHeap(max_string_length, mem_id));
}

/***************** constructors and destructor *****************/
CStringColumn::CStringColumn(const std::string& name,
                             size_t maximal_string_length,
                             hype::ProcessingDeviceMemoryID mem_id)
    : Column<char*>(name, CHAR, mem_id),
      heap_(createStringHeap(FIXED_LENGTH_STRING_HEAP, maximal_string_length,
                             mem_id)) {}

CStringColumn::CStringColumn(const CStringColumn& x)
    : Column<char*>(x), heap_() {
  heap_ = x.heap_->copy();
  if (!heap_) {
    throw std::bad_alloc();
  }
}

CStringColumn& CStringColumn::operator=(const CStringColumn& other) {
  if (this != &other)  // protect against invalid self-assignment
  {
    //                assert(this->getMemoryID()==other.getMemoryID());
    Column<char*>::operator=(other);
    heap_ = other.heap_->copy();
    if (!heap_) {
      throw std::bad_alloc();
    }
  }
  return *this;
}

CStringColumn::~CStringColumn() {}

bool CStringColumn::insert(const boost::any& new_value) {
  char* value;
  bool ret_success =
      getValueFromAny(this->name_, new_value, value, this->db_type_);
  if (!ret_success) return false;
  char* const_value = value;
  push_back(const_value);
  return true;
}

bool CStringColumn::insert(const value_type& new_value) {
  this->push_back(new_value);
  return true;
}

bool CStringColumn::update(TID tid, const boost::any& new_value) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return false;
}

bool CStringColumn::update(PositionListPtr tid, const boost::any& new_value) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return false;
}

bool CStringColumn::remove(TID tid) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return false;
}
// assumes tid list is sorted ascending
bool CStringColumn::remove(PositionListPtr tid) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return false;
}
bool CStringColumn::clearContent() {
  clear();
  return true;
}

const boost::any CStringColumn::get(TID tid) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return false;
}

size_t CStringColumn::size() const throw() { return Column<char*>::size(); }

size_t CStringColumn::getSizeinBytes() const throw() {
  return Column<char*>::getSizeinBytes() + heap_->getSizeinBytes();
}

size_t CStringColumn::getMaxStringLength() const {
  return heap_->getMaxStringLength();
}

const ColumnPtr CStringColumn::copy() const {
  ColumnPtr copy;
  try {
    copy = ColumnPtr(new CStringColumn(*this));
  } catch (std::bad_alloc& e) {
    return ColumnPtr();
  }
  return copy;
}
const ColumnPtr CStringColumn::copy(
    const hype::ProcessingDeviceMemoryID& mem_id) const {
  boost::shared_ptr<CStringColumn> col;
  try {
    col = boost::shared_ptr<CStringColumn>(new CStringColumn(
        this->getName(), this->heap_->getMaxStringLength(), mem_id));
    col->resize(this->size());
  } catch (std::bad_alloc& e) {
    COGADB_ERROR(
        "Out of memory on device memory "
            << (int)mem_id << " for column '" << this->getName()
            << "' requesting "
            << double(this->getSizeinBytes()) / (1024 * 1024 * 2014)
            << " GB memory" << std::endl
            << "Total free memory: "
            << double(HardwareDetector::instance().getFreeMemorySizeInByte(
                   mem_id)) /
                   (1024 * 1024 * 2014),
        "");
    return ColumnPtr();
  }
  if (col->size() > 0) {
    col->heap_ = this->heap_->copy(mem_id);
    if (!col->heap_) return ColumnPtr();
    col->heap_->initStringArray(col->data(), col->size(), mem_id);
    //                typedef typename
    //                CopyFunctionFactory<char*>::CopyFunctionPtr
    //                CopyFunctionPtr;
    //                CopyFunctionPtr func =
    //                CopyFunctionFactory<char*>::getCopyFunction(mem_id,
    //                this->getMemoryID());
    //                assert(func != NULL);
    //                char** original_array = this->data();
    //                if (!(*func)(col->data(), original_array, this->size() *
    //                sizeof (char*))) return ColumnPtr();
  }
  return col;
}
const Column<char*>::DenseValueColumnPtr
CStringColumn::copyIntoDenseValueColumn(
    const ProcessorSpecification& proc_spec) const {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return Column<char*>::DenseValueColumnPtr();
}
const DoubleDenseValueColumnPtr CStringColumn::convertToDenseValueDoubleColumn(
    const ProcessorSpecification& proc_spec) const {
  return DoubleDenseValueColumnPtr();
}
const ColumnPtr CStringColumn::materialize() throw() { return copy(); }
hype::ProcessingDeviceMemoryID CStringColumn::getMemoryID() const {
  assert(this->Column<char*>::getMemoryID() == heap_->getMemoryID());
  return this->Column<char*>::getMemoryID();
}
const ColumnPtr CStringColumn::gather(PositionListPtr tid_list,
                                      const GatherParam& param) {
  boost::shared_ptr<CStringColumn> result(
      new CStringColumn(this->name_, this->db_type_, this->getMemoryID()));
  try {
    // use resize of super class to not effect heap space
    result->Column<char*>::resize(tid_list->size());
  } catch (std::bad_alloc& e) {
    return ColumnPtr();
  }

  PositionListPtr copied_tids = copy_if_required(tid_list, this->getMemoryID());
  if (!copied_tids) return ColumnPtr();
  tid_list = copied_tids;

  ProcessorBackend<char*>* backend =
      ProcessorBackend<char*>::get(param.proc_spec.proc_id);
  if (backend->gather(result->data(), this->data(), tid_list, param)) {
    // share heap space between Columns
    result->heap_ = this->heap_;
    return result;
  } else {
    return ColumnPtr();
  }
}

const ColumnGroupingKeysPtr CStringColumn::createColumnGroupingKeys(
    const ProcessorSpecification& proc_spec) const {
  return ColumnGroupingKeysPtr();
}

size_t CStringColumn::getNumberOfRequiredBits() const {
  // cannot bitpack this column
  return 65;
}

const AggregationResult CStringColumn::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, const AggregationParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return AggregationResult();
}
const AggregationResult CStringColumn::aggregate(const AggregationParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return AggregationResult();
}

const PositionListPtr CStringColumn::selection(const SelectionParam& param) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return PositionListPtr();
}

const BitmapPtr CStringColumn::bitmap_selection(const SelectionParam& param) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return BitmapPtr();
}

#ifdef ENABLE_CDK_USAGE
const PositionListPairPtr CStringColumn::hash_join(ColumnPtr join_column) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return PositionListPairPtr();
}
const PositionListPairPtr CStringColumn::radix_join(ColumnPtr join_column) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return PositionListPairPtr();
}
#endif

const PositionListPairPtr CStringColumn::join(ColumnPtr join_column,
                                              const JoinParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return PositionListPairPtr();
}
const PositionListPairPtr CStringColumn::join(Column<char*>& join_column,
                                              const JoinParam& param) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return PositionListPairPtr();
}

const PositionListPtr CStringColumn::tid_semi_join(ColumnPtr join_column,
                                                   const JoinParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return PositionListPtr();
}
const PositionListPtr CStringColumn::tid_semi_join(
    Column<value_type>& join_column, const JoinParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return PositionListPtr();
}

const BitmapPtr CStringColumn::bitmap_semi_join(ColumnPtr join_column,
                                                const JoinParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return BitmapPtr();
}
const BitmapPtr CStringColumn::bitmap_semi_join(Column<value_type>& join_column,
                                                const JoinParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return BitmapPtr();
}

const ColumnPtr CStringColumn::column_algebra_operation(
    ColumnPtr source_column, const AlgebraOperationParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return ColumnPtr();
}
const ColumnPtr CStringColumn::column_algebra_operation(
    Column<value_type>& source_column, const AlgebraOperationParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return ColumnPtr();
}
const ColumnPtr CStringColumn::column_algebra_operation(
    const boost::any& value, const AlgebraOperationParam&) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return ColumnPtr();
}

const PositionListPtr CStringColumn::sort(const SortParam& param) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return PositionListPtr();
}
const PositionListPtr CStringColumn::sort(const SortParam& param,
                                          bool no_copy) {
  COGADB_FATAL_ERROR("Called Unimplemented Function!", "");
  return PositionListPtr();
}

/* BEGIN CONTAINER OPERATIONS*/

void CStringColumn::reserve(size_t new_capacity) {
  Column<char*>::reserve(new_capacity);
  bool heap_moved;
  heap_->reserve(new_capacity, heap_moved);
  /* if the realloc moved our memory, we are required to rebuild
   * our outdated pointer array to the strings in the heap
   */
  if (heap_moved) {
    if (!heap_->initStringArray(this->data(), heap_->size(),
                                this->getMemoryID())) {
      COGADB_FATAL_ERROR("Failed to initialize pointer array!", "");
    }
  }
}
void CStringColumn::resize(size_t new_size) {
  Column<char*>::resize(new_size);
  bool heap_moved;
  heap_->resize(new_size, heap_moved);
  /* if the realloc moved our memory, we are required to rebuild
   * our outdated pointer array to the strings in the heap
   */
  if (heap_moved) {
    if (!heap_->initStringArray(this->data(), heap_->size(),
                                this->getMemoryID())) {
      COGADB_FATAL_ERROR("Failed to initialize pointer array!", "");
    }
  }
}

void CStringColumn::push_back(const value_type& val) {
  push_back(static_cast<const char* const>(val));
}

void CStringColumn::push_back(const char* const val) {
  bool heap_moved;
  char* val_on_heap = heap_->push_back(val, heap_moved);
  if (!val_on_heap) {
    COGADB_FATAL_ERROR("Failed to extend heap!", "");
  } else {
    /* insert pointer to string on heap */
    Column<char*>::push_back(val_on_heap);
    /* if the realloc moved our memory, we are required to rebuild
     * our outdated pointer array to the strings in the heap
     */
    if (heap_moved) {
      //                    COGADB_WARNING("Heap expanded!","");
      assert(this->size() == heap_->size());
      if (!heap_->initStringArray(this->data(), heap_->size(),
                                  this->getMemoryID())) {
        COGADB_FATAL_ERROR("Failed to initialize pointer array!", "");
      }
      assert(strcmp(val_on_heap, val) == 0);
    }
  }
}

void CStringColumn::clear() {
  Column<char*>::clear();
  heap_ = StringHeapPtr(createStringHeap(FIXED_LENGTH_STRING_HEAP,
                                         heap_->getMaxStringLength(),
                                         this->getMemoryID()));
}

/* END CONTAINER OPERATIONS*/

void CStringColumn::printHeap() { heap_->print(); }

bool CStringColumn::load_impl(const std::string& path,
                              boost::archive::binary_iarchive& ia) {
  if (!Column<char*>::load_impl(path, ia)) {
    return false;
  }
  std::string path_to_column(path);
  path_to_column += "/";
  path_to_column += this->name_;
  int heap_type;
  size_t max_string_length;
  ia >> heap_type;
  ia >> max_string_length;
  StringHeapType str_heap_type = (StringHeapType)heap_type;

  heap_ = createStringHeap(str_heap_type, max_string_length,
                           hype::PD_Memory_0);  // load heap to CPU main memory
  bool ret = heap_->load(path, this->getName(), ia);
  if (!ret) return false;

  return heap_->initStringArray(this->data(), this->size(), hype::PD_Memory_0);
}

bool CStringColumn::store_impl(const std::string& path,
                               boost::archive::binary_oarchive& oa) {
  if (!Column<char*>::store_impl(path, oa)) {
    return false;
  }
  std::string path_to_column(path);
  path_to_column += "/";
  path_to_column += this->name_;
  int heap_type = (int)heap_->getStringHeapType();
  size_t max_string_length = heap_->getMaxStringLength();
  oa << heap_type;
  oa << max_string_length;
  return heap_->store(path, this->getName(), oa);
}

}  // end namespace CogaDB
