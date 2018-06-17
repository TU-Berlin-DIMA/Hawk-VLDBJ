#include <core/column.hpp>
#include <core/column_base_typed.hpp>
#include <core/vector_typed.hpp>
//#include <util/reduce_by_keys.hpp>
#include <util/is_equals.h>
#include <backends/processor_backend.hpp>
#include <core/runtime_configuration.hpp>
#include <util/column_grouping_keys.hpp>
#include <util/getname.hpp>
#include <util/types.hpp>
#include <util/utility_functions.hpp>

#include <core/block_iterator.hpp>

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

#include <boost/make_shared.hpp>

namespace CoGaDB {

using namespace std;

template <class T>
ColumnBaseTyped<T>::ColumnBaseTyped(const std::string &name,
                                    AttributeType db_type,
                                    ColumnType column_type)
    : ColumnBase(name, db_type, column_type),
      has_primary_key_constraint_(false),
      has_foreign_key_constraint_(false),
      fk_constr_(),
      statistics_() {}

template <class T>
ColumnBaseTyped<T>::~ColumnBaseTyped() {}

bool isInteger(float n) { return n - (float)(int)n == 0.0; }

bool isInteger(double n) { return n - (double)(int)n == 0.0; }

template <class T>
std::string ColumnBaseTyped<T>::getStringValue(TID tid) {
  std::stringstream ss;
  ss.setf(std::ios::fixed, std::ios::floatfield);
  ss.precision(3);
  ss << (*this)[tid];
  return ss.str();
}

template <>
std::string ColumnBaseTyped<uint32_t>::getStringValue(TID tid) {
  std::stringstream ss;
  //            ss.setf( std::ios::fixed, std::ios::floatfield );
  //            ss.precision(3);
  if (this->db_type_ == DATE) {
    std::string result;
    bool ret = convertInternalDateTypeToString((*this)[tid], result);
    assert(ret == true);
    return result;
  } else {
    ss << (*this)[tid];
  }
  return ss.str();
}

template <>
std::string ColumnBaseTyped<float>::getStringValue(TID tid) {
  std::stringstream ss;
  float val = (*this)[tid];
  if (!isInteger(val)) {
    ss.precision(3);
  } else {
    ss.precision(0);
  }
  ss.setf(std::ios::fixed, std::ios::floatfield);
  ss << val;
  return ss.str();
}

template <>
std::string ColumnBaseTyped<double>::getStringValue(TID tid) {
  std::stringstream ss;
  double val = (*this)[tid];
  if (!isInteger(val)) {
    ss.precision(3);
  } else {
    ss.precision(0);
  }
  ss.setf(std::ios::fixed, std::ios::floatfield);
  ss << val;
  return ss.str();
}

template <class T>
const std::type_info &ColumnBaseTyped<T>::type() const throw() {
  return typeid(T);
}

template <class T>
bool ColumnBaseTyped<T>::is_equal(ColumnPtr column) {
  if (column == nullptr) {
    return false;
  }

  if (column.get() == this) {
    return true;
  }

  if (column->type() != typeid(T)) {
    return false;
  }

  if (column->size() != this->size()) {
    if (!quiet && debug) {
      std::cout << "Columns have different size." << std::endl
                << "Candidate: " << column->size() << " This: " << this->size()
                << std::endl;
    }

    return false;
  }

  shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> > typed_column =
      shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<T> >(
          column);

  ProcessorSpecification proc_spec(hype::PD0);
  DenseValueColumnPtr this_col = this->copyIntoDenseValueColumn(proc_spec);
  DenseValueColumnPtr that_col =
      typed_column->copyIntoDenseValueColumn(proc_spec);

  T *this_col_data = this_col->data();
  T *that_col_data = that_col->data();

  for (unsigned int i = 0; i < this_col->size(); i++) {
    if (!CoGaDB::is_equal(this_col_data[i], that_col_data[i])) {
      if (!quiet && debug) {
        // added precision for a better readability
        std::cout.precision(15);
        std::cout << "Columnvalues in row " << i << " differ." << std::endl
                  << "Candidate: " << that_col_data[i]
                  << " This: " << this_col_data[i] << std::endl;
      }
      return false;
    }
  }

  return true;
}

template <class T>
int ColumnBaseTyped<T>::compareValuesAtIndexes(TID id1, TID id2) {
  if ((*this)[id1] == (*this)[id2])
    return 0;
  else if ((*this)[id1] < (*this)[id2])
    return 1;
  else
    return -1;
}

template <class T>
typename ColumnBaseTyped<T>::TypedVectorPtr
ColumnBaseTyped<T>::copyIntoPlainVector() {
  TypedVectorPtr v(new std::vector<T>());
  v->reserve(this->size());
  size_t vector_size = this->size();
  for (size_t i = 0; i < vector_size; ++i) {
    v->push_back((*this)[i]);
  }
  return v;
}

template <class T>
bool ColumnBaseTyped<T>::isApproximatelyEqual(ColumnPtr column) {
  std::cout << "Approximately Equal in Columns is getting called" << std::endl;

  if (column == nullptr) {
    return false;
  }

  if (column.get() == this) {
    return true;
  }

  if (column->type() != typeid(T)) {
    if (!quiet && debug) {
      std::cout << "Typemismatch for columns: " << this->name_ << " and "
                << column->getName() << std::endl;
    }

    return false;
  }

  if (column->size() != this->size()) {
    if (!quiet && debug) {
      std::cout << "Columns have different size." << std::endl
                << "Candidate: " << column->size() << " This: " << this->size()
                << std::endl;
    }

    return false;
  }

  shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> > typed_column =
      shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<T> >(
          column);

  ProcessorSpecification proc_spec(hype::PD0);
  DenseValueColumnPtr this_col = this->copyIntoDenseValueColumn(proc_spec);
  DenseValueColumnPtr that_col =
      typed_column->copyIntoDenseValueColumn(proc_spec);

  T *this_col_data = this_col->data();
  T *that_col_data = that_col->data();

  for (size_t i = 0; i < this_col->size(); i++) {
    if (!CoGaDB::approximatelyEqual(this_col_data[i], that_col_data[i])) {
      // if (!quiet && debug)
      {
        // added precision for a better readability
        std::cout.precision(15);
        std::cout << "Columnvalues in row " << i << " differ." << std::endl
                  << "Candidate: " << that_col_data[i]
                  << " This: " << this_col_data[i] << " for tuple_id " << i
                  << std::endl;
      }
      return false;
    }
  }

  return true;
}

template <class T>
bool ColumnBaseTyped<T>::setPrimaryKeyConstraint() {
  if (checkUniqueness()) {
    has_primary_key_constraint_ = true;
    return true;
  } else {
    has_primary_key_constraint_ = false;
    return false;
  }
}

template <class T>
bool ColumnBaseTyped<T>::hasPrimaryKeyConstraint() const throw() {
  return has_primary_key_constraint_;
}

template <class T>
bool ColumnBaseTyped<T>::hasForeignKeyConstraint() const throw() {
  return has_foreign_key_constraint_;
}

template <class T>
const ForeignKeyConstraint &ColumnBaseTyped<T>::getForeignKeyConstraint() {
  return this->fk_constr_;
}

template <class T>
const ColumnStatistics &ColumnBaseTyped<T>::getColumnStatistics() const {
  return statistics_;
}

template <class T>
const ExtendedColumnStatistics<T>
    &ColumnBaseTyped<T>::getExtendedColumnStatistics() const {
  return statistics_;
}

template <class T>
bool ColumnBaseTyped<T>::computeColumnStatistics() {
  bool ret = false;
  // only CPU
  ProcessorSpecification proc_spec(hype::PD0);
  DenseValueColumnPtr dense_value_column =
      this->copyIntoDenseValueColumn(proc_spec);

  if (dense_value_column) {
    auto array = dense_value_column->data();
    size_t array_size = dense_value_column->size();

    ret = statistics_.computeStatistics(array, array_size);
    if (ret) {
      if (!statistics_.store(this->path_to_column_)) {
        COGADB_ERROR("Could not store statistics of column '" << this->name_
                                                              << "'!",
                     "");
      }
    }
  }
  return ret;
}

template <class T>
bool ColumnBaseTyped<T>::setForeignKeyConstraint(
    const ForeignKeyConstraint &foreign_key_constraint) {
  // if we have already set a ForeignKeyConstraint, we cannot add a new one,
  // the old one needs to be deleted by the user!
  if (hasForeignKeyConstraint()) return false;
  ColumnPtr pk_column = foreign_key_constraint.getPrimaryKeyColumn();
  if (!pk_column) return false;
  // if(!pk_column->checkUniqueness()) return false;
  if (this->checkReferentialIntegrity(pk_column)) {
    this->fk_constr_ = foreign_key_constraint;
    this->has_foreign_key_constraint_ = true;
    return true;
  } else {
    return false;
  }
}

template <class T>
bool ColumnBaseTyped<T>::checkUniqueness() {
  std::set<T> value_set;
  for (TID i = 0; i < this->size(); ++i) {
    value_set.insert((*this)[i]);
  }
  if (value_set.size() == this->size()) {
    return true;
  } else {
    return false;
  }
}

template <class T>
bool ColumnBaseTyped<T>::checkReferentialIntegrity(ColumnPtr primary_key_col) {
  // if(primary_key_col->)
  if (primary_key_col->type() != typeid(T)) {
    COGADB_ERROR(
        std::string(
            "Check Referential Integrity Failed! Typemismatch for columns ") +
            this->name_ + std::string(" and ") + primary_key_col->getName(),
        "");
    return false;
  }

  shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> > pk_column =
      shared_pointer_namespace::dynamic_pointer_cast<ColumnBaseTyped<T> >(
          primary_key_col);  // static_cast<IntTypedColumnPtr>(column1);
  assert(pk_column != NULL);
  if (!pk_column->hasPrimaryKeyConstraint()) {
    COGADB_ERROR(
        std::string(
            "Cannot create ForeignKey constrained: referenced column '") +
            primary_key_col->getName() +
            std::string("' is not a primary key! "),
        "");
    return false;
  }
  if (!pk_column->checkUniqueness()) {
    COGADB_ERROR(
        std::string("Column marked as PrimaryKey contains duplicates!: Column "
                    "Name: '") +
            primary_key_col->getName(),
        "");
    return false;
  }

  // perfrom simple hash join to check referential integrity
  typedef boost::unordered_multimap<T, TID, boost::hash<T>, std::equal_to<T> >
      HashTable;

  // create hash table
  HashTable hashtable;
  unsigned int hash_table_size = pk_column->size();
  unsigned int join_column_size = this->size();

  for (unsigned int i = 0; i < hash_table_size; i++)
    hashtable.insert(std::pair<T, TID>((*pk_column)[i], i));

  std::pair<typename HashTable::iterator, typename HashTable::iterator> range;
  typename HashTable::iterator it;
  for (unsigned int i = 0; i < join_column_size; i++) {
    unsigned int number_of_matches = 0;
    range = hashtable.equal_range((*this)[i]);
    for (it = range.first; it != range.second; ++it) {
      if (it->first == (*this)[i]) {  //(*join_column)[i]){
        number_of_matches++;
      }
    }
    if (number_of_matches == 0) {
      std::stringstream ss;
      ss << "Cannot create ForeignKey constrained: broken reference: for value "
            "'"
         << (*this)[i] << "'! ";  // << endl;
      COGADB_ERROR(ss.str(), "");
      return false;
    }
    if (number_of_matches > 1) {
      std::stringstream ss;
      ss << "Cannot create ForeignKey constrained: referenced column '"
         << primary_key_col->getName() << "' has duplicate value: '"
         << (*this)[i] << "'!";
      COGADB_ERROR(ss.str(), "");
      return false;
    }
  }
  return true;
}

template <class T>
const VectorPtr ColumnBaseTyped<T>::getVector(ColumnPtr col,
                                              BlockIteratorPtr it) const {
  assert(this == col.get());
  boost::shared_ptr<ColumnBaseTyped<T> > typed_col;
  typed_col = boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(col);
  assert(typed_col != NULL);
  size_t block_size =
      std::min(it->getBlockSize(), col->size() - it->getOffset());
  VectorPtr vector(new VectorTyped<T>(it->getOffset(), block_size, typed_col));
  return vector;
}

template <class T>
const ColumnPtr ColumnBaseTyped<T>::changeCompression(
    const ColumnType &col_type) const {
  if (col_type == this->getColumnType()) {
    return this->copy();
  } else {
    typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
    typedef
        typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

    ProcessorSpecification proc_spec(hype::PD0);
    DenseValueColumnPtr decompressed_column =
        this->copyIntoDenseValueColumn(proc_spec);
    ColumnPtr new_col =
        createColumn(this->getType(), this->getName(), this->getColumnType());
    if (!new_col) return ColumnPtr();
    shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> > new_col_typed =
        shared_pointer_namespace::dynamic_pointer_cast<ColumnBaseTyped<T> >(
            new_col);
    T *array = decompressed_column->data();
    size_t num_elements = decompressed_column->size();
    for (size_t i = 0; i < num_elements; ++i) {
      new_col_typed->insert(array[i]);
    }

    return new_col_typed;
  }
}

template <class T>
const ColumnPtr ColumnBaseTyped<T>::copy(
    const hype::ProcessingDeviceMemoryID &mem_id) const {
  if (mem_id == hype::PD_Memory_0) {
    return this->copy();
  } else {
    COGADB_FATAL_ERROR("Cannot Handle Cross Processor Copy Operation!", "");
    return ColumnPtr();
  }
}

template <class T>
const DoubleDenseValueColumnPtr
ColumnBaseTyped<T>::convertToDenseValueDoubleColumn(
    const ProcessorSpecification &proc_spec) const {
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr decompressed = this->copyIntoDenseValueColumn(proc_spec);
  if (!decompressed) {
    return DoubleDenseValueColumnPtr();
  }

  ProcessorBackend<T> *backend = ProcessorBackend<T>::get(proc_spec.proc_id);
  return backend->convertToDoubleDenseValueColumn(
      this->getName(), decompressed->data(), decompressed->size(), proc_spec);
}

template <class T>
const StringDenseValueColumnPtr
ColumnBaseTyped<T>::convertToDenseValueStringColumn() const {
  ProcessorSpecification proc_spec(hype::PD0);
  hype::ProcessingDeviceMemoryID mem_id =
      hype::util::getMemoryID(proc_spec.proc_id);

  DenseValueColumnPtr decompressed;
  if (mem_id == hype::PD_Memory_0) {
    decompressed = this->copyIntoDenseValueColumn(proc_spec);
  } else {
    ColumnPtr placed_column = this->copy(mem_id);
    if (!placed_column) return StringDenseValueColumnPtr();
    boost::shared_ptr<ColumnBaseTyped<T> > typed_placed_column =
        boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(placed_column);
    decompressed = typed_placed_column->copyIntoDenseValueColumn(proc_spec);
  }

  if (!decompressed) {
    return StringDenseValueColumnPtr();
  }

  return decompressed->convertToDenseValueStringColumn();
}

template <class T>
bool ColumnBaseTyped<T>::append(ColumnPtr col) {
  if (!col) return false;
  if (this->type() != col->type()) {
    COGADB_FATAL_ERROR("Cannot append column with different data type!", "");
    return false;
  }
  boost::shared_ptr<ColumnBaseTyped<T> > col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(col);
  assert(col_typed != NULL);
  return this->append(col_typed);
}

template <class T>
bool ColumnBaseTyped<T>::append(
    boost::shared_ptr<ColumnBaseTyped<T> > typed_col) {
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
  T *array = dense_column->data();
  size_t num_elements = dense_column->size();
  bool ret = true;
  for (size_t i = 0; i < num_elements; ++i) {
    ret = ret && this->insert(array[i]);
  }
  return ret;
}

template <class T>
const ColumnPtr ColumnBaseTyped<T>::decompress(
    const ProcessorSpecification &proc_spec) const {
  return this->copyIntoDenseValueColumn(proc_spec);
}

template <class T>
hype::ProcessingDeviceMemoryID ColumnBaseTyped<T>::getMemoryID() const {
  return hype::PD_Memory_0;
}

template <class T>
const ColumnGroupingKeysPtr ColumnBaseTyped<T>::createColumnGroupingKeys(
    const ProcessorSpecification &proc_spec) const {
  COGADB_FATAL_ERROR(
      "Called createColumnGroupingKeys() in ColumnBaseTyped<T> for Type: "
          << typeid(T).name() << " !",
      "");
  return ColumnGroupingKeysPtr();
}

template <class T>
size_t ColumnBaseTyped<T>::getNumberOfRequiredBits() const {
  // if a column does not support the optimized groupby operator,
  // we just tell it that we need more bits than it can handle
  return 65;
}

template <class T>
const PositionListPtr ColumnBaseTyped<T>::sort(const SortParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);

  assert(dense_column != NULL);
  // sort using dense value sort, and do notcopy column but instead sort in
  // place
  return dense_column->sort(param);
}

typedef std::vector<int> Flags;
typedef boost::shared_ptr<Flags> FlagsPtr;

class TBB_Body_PrefixSum {
 public:
  TBB_Body_PrefixSum(std::vector<int> *y_, const std::vector<int> *x_)
      : sum(0), y(y_), x(x_) {}

  int get_sum() const { return sum; }

  template <typename Tag>
  void operator()(const tbb::blocked_range<int> &r, Tag) {
    int temp = sum;
    for (int i = r.begin(); i < r.end(); ++i) {
      temp = temp + (*x)[static_cast<uint32_t>(i)];
      if (Tag::is_final_scan()) {
        (*y)[i] = temp;
      }
    }
    sum = temp;
  }

  TBB_Body_PrefixSum(TBB_Body_PrefixSum &b, tbb::split)
      : sum(0), y(b.y), x(b.x) {}

  void reverse_join(TBB_Body_PrefixSum &a) { sum = a.sum + sum; }

  void assign(TBB_Body_PrefixSum &b) { sum = b.sum; }

 private:
  int sum;
  std::vector<int> *y;
  const std::vector<int> *x;
};

inline int TBB_Prefix_Sum(std::vector<int> &y, const std::vector<int> &x,
                          unsigned int number_of_threads) {
  assert(y.size() == x.size());
  auto chunk_size = x.size() / number_of_threads;
  TBB_Body_PrefixSum body(&y, &x);
  tbb::parallel_scan(
      tbb::blocked_range<int>(0, static_cast<int>(x.size()), chunk_size), body);
  return body.get_sum();
}

template <typename T>
void selection_thread_set_flag_array(ColumnBaseTyped<T> *col,
                                     unsigned int thread_id, FlagsPtr flags,
                                     unsigned int number_of_threads,
                                     const T &value_for_comparison,
                                     const ValueComparator comp) {
  assert(flags->size() == col->size());
  if (!quiet)
    std::cout << "Using CPU for Selection (parallel mode)..." << std::endl;
  auto array_size = col->size();
  auto chunk_size = col->size() / number_of_threads;
  unsigned int start_id = thread_id * chunk_size;
  unsigned int end_id = (thread_id * chunk_size) + chunk_size;
  // make sure that the last thread processes the rest of the array
  if (thread_id + 1 == number_of_threads) end_id = array_size;
  if (comp == EQUAL) {
    for (TID i = start_id; i < end_id; i++) {
      if (value_for_comparison == (*col)[i]) {
        (*flags)[i] = 1;
      }
    }
  } else if (comp == LESSER) {
    for (TID i = start_id; i < end_id; i++) {
      if ((*col)[i] < value_for_comparison) {
        (*flags)[i] = 1;
      }
    }
  } else if (comp == LESSER_EQUAL) {
    for (TID i = start_id; i < end_id; i++) {
      if ((*col)[i] <= value_for_comparison) {
        (*flags)[i] = 1;
      }
    }
  } else if (comp == GREATER) {
    for (TID i = start_id; i < end_id; i++) {
      if ((*col)[i] > value_for_comparison) {
        (*flags)[i] = 1;
      }
    }
  } else if (comp == GREATER_EQUAL) {
    for (TID i = start_id; i < end_id; i++) {
      if ((*col)[i] >= value_for_comparison) {
        (*flags)[i] = 1;
      }
    }
  } else {
    std::cerr << "FATAL Error! In CoGaDB::selection_thread_set_flag_array(): "
                 "Invalid Value Comparator! "
              << comp << std::endl;
  }
}

template <typename T>
void selection_thread_write_result_to_output_array(
    ColumnBaseTyped<T> *col, unsigned int thread_id, FlagsPtr flags,
    std::vector<int> *prefix_sum_array, PositionListPtr result_tids,
    unsigned int number_of_threads) {
  assert(flags->size() == col->size());
  assert(flags->size() == prefix_sum_array->size());
  auto array_size = col->size();
  auto chunk_size = col->size() / number_of_threads;
  auto start_id = thread_id * chunk_size;
  auto end_id = (thread_id * chunk_size) + chunk_size;
  // make sure that the last thread processes the rest of the array
  if (thread_id + 1 == number_of_threads) end_id = array_size;
  for (TID i = start_id; i < end_id; i++) {
    auto prefix_sum_val = (*prefix_sum_array)[i];
    if ((*flags)[i] == 1 && prefix_sum_val > 0) {
      auto write_id = static_cast<uint32_t>(prefix_sum_val - 1);
      (*result_tids)[write_id] = i;  // write matching TID to output buffer
    }
  }
}

template <class T>
const PositionListPtr ColumnBaseTyped<T>::lock_free_parallel_selection(
    const boost::any &value_for_comparison, const ValueComparator comp,
    unsigned int number_of_threads) {
  // unsigned int number_of_threads=4;

  if (value_for_comparison.type() != typeid(T)) {
    std::cout << "Fatal Error!!! Typemismatch for column " << name_
              << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }

  T value = boost::any_cast<T>(value_for_comparison);

  boost::thread_group threads;

  // create flag array of column size and init with zeros
  FlagsPtr flags(new Flags(this->size(), 0));
  // std::vector<PositionListPtr> local_result_arrays;
  for (unsigned int i = 0; i < number_of_threads; i++) {
    threads.add_thread(new boost::thread(
        boost::bind(&CoGaDB::selection_thread_set_flag_array<T>, this, i, flags,
                    number_of_threads, value, comp)));
  }
  threads.join_all();

  if (!quiet && verbose && debug) {
    std::cout << "FLAG Array:" << std::endl;
    for (unsigned int i = 0; i < flags->size(); ++i) {
      std::cout << (*flags)[i] << std::endl;
    }
  }

  std::vector<int> prefix_sum(this->size(), 0);

  // do prefix sum on threads
  TBB_Prefix_Sum(prefix_sum, *flags, number_of_threads);

  if (!quiet && verbose && debug) {
    std::cout << "Prefix Sum:" << std::endl;
    for (unsigned int i = 0; i < prefix_sum.size(); ++i) {
      std::cout << prefix_sum[i] << std::endl;
    }
  }
  auto resul_size = static_cast<uint32_t>(prefix_sum.back());
  PositionListPtr result_tids(createPositionList(resul_size));

  for (unsigned int i = 0; i < number_of_threads; i++) {
    threads.add_thread(new boost::thread(boost::bind(
        &CoGaDB::selection_thread_write_result_to_output_array<T>, this, i,
        flags, &prefix_sum, result_tids, number_of_threads)));
  }
  threads.join_all();

  if (!quiet && verbose && debug) {
    std::cout << "TIDS:" << std::endl;
    for (unsigned int i = 0; i < result_tids->size(); ++i) {
      std::cout << (*result_tids)[i] << std::endl;
    }
  }

  return result_tids;
}

template <class T>
const BitmapPtr ColumnBaseTyped<T>::bitmap_selection(
    const SelectionParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);
  assert(dense_column != NULL);
  return dense_column->bitmap_selection(param);
}

template <class T>
const PositionListPtr ColumnBaseTyped<T>::selection(
    const SelectionParam &param) {
  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);
  assert(dense_column != NULL);
  return dense_column->selection(param);
}

template <class T>
const PositionListPairPtr ColumnBaseTyped<T>::hash_join(
    ColumnPtr join_column_) {
  typedef boost::unordered_multimap<T, TID, boost::hash<T>, std::equal_to<T> >
      HashTable;

  if (join_column_->type() != typeid(T)) {
    std::cout << "Fatal Error!!! Typemismatch for columns " << this->name_
              << " and " << join_column_->getName() << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }

  shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> > join_column =
      shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<T> >(
          join_column_);  // static_cast<IntTypedColumnPtr>(column1);

  PositionListPairPtr join_tids(new PositionListPair());
  join_tids->first = createPositionList();
  join_tids->second = createPositionList();

  Timestamp build_hashtable_begin = getTimestamp();
  // create hash table
  HashTable hashtable;
  auto hash_table_size = this->size();
  auto join_column_size = join_column->size();

  assert(join_column_->size() >= this->size());

  T *join_tids_table1 = static_cast<T *>(malloc(join_column_size * sizeof(T)));
  T *join_tids_table2 = static_cast<T *>(malloc(join_column_size * sizeof(T)));

  unsigned int pos = 0;
  ColumnBaseTyped<T> &join_column_ref =
      dynamic_cast<ColumnBaseTyped<T> &>(*join_column);

  for (size_t i = 0; i < hash_table_size; i++)
    hashtable.insert(std::pair<T, TID>((*this)[i], i));
  Timestamp build_hashtable_end = getTimestamp();

  Timestamp prune_hashtable_begin = getTimestamp();

  std::pair<typename HashTable::iterator, typename HashTable::iterator> range;
  typename HashTable::iterator it;

  for (unsigned int i = 0; i < join_column_size; i++) {
    range = hashtable.equal_range(join_column_ref[i]);
    for (it = range.first; it != range.second; ++it) {
      if (it->first == join_column_ref[i]) {  //(*join_column)[i]){
        join_tids_table1[pos] = it->second;
        join_tids_table2[pos] = i;
        pos++;
      }
    }
  }

  // copy result in PositionList (vector)
  join_tids->first->insert(join_tids_table1, join_tids_table1 + pos);
  join_tids->second->insert(join_tids_table2, join_tids_table2 + pos);

  free(join_tids_table1);
  free(join_tids_table2);
  Timestamp prune_hashtable_end = getTimestamp();

  if (!quiet && verbose)
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

template <>
const PositionListPairPtr ColumnBaseTyped<char *>::hash_join(
    ColumnPtr join_column_) {
  COGADB_FATAL_ERROR("Called unimplemented method!", "");
}

template <class Type>
const PositionListPairPtr ColumnBaseTyped<Type>::sort_merge_join(
    ColumnPtr join_column_) {
  if (join_column_->type() != typeid(Type)) {
    std::cout << "Fatal Error!!! Typemismatch for columns " << this->name_
              << " and " << join_column_->getName() << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }

  shared_pointer_namespace::shared_ptr<ColumnBaseTyped<Type> > join_column =
      shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<Type> >(
          join_column_);  // static_cast<IntTypedColumnPtr>(column1);

  PositionListPairPtr join_tids(new PositionListPair());
  return join_tids;
}

template <class Type>
const PositionListPairPtr ColumnBaseTyped<Type>::nested_loop_join(
    ColumnPtr join_column_) {
  assert(join_column_ != NULL);
  if (join_column_->type() != typeid(Type)) {
    std::cout << "Fatal Error!!! Typemismatch for columns " << this->name_
              << " and " << join_column_->getName() << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }

  shared_pointer_namespace::shared_ptr<ColumnBaseTyped<Type> > join_column =
      shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<Type> >(
          join_column_);  // static_cast<IntTypedColumnPtr>(column1);

  PositionListPairPtr join_tids(new PositionListPair());
  join_tids->first = createPositionList();
  join_tids->second = createPositionList();

  auto join_column1_size = this->size();
  auto join_column2_size = join_column->size();

  for (unsigned int i = 0; i < join_column1_size; i++) {
    for (unsigned int j = 0; j < join_column2_size; j++) {
      if ((*this)[i] == (*join_column)[j]) {
        if (debug) std::cout << "MATCH: (" << i << "," << j << ")" << std::endl;
        join_tids->first->push_back(i);
        join_tids->second->push_back(j);
      }
    }
  }

  return join_tids;
}

template <class Type>
const PositionListPairPtr ColumnBaseTyped<Type>::radix_join(
    ColumnPtr join_column) {
  return this->hash_join(join_column);
}

template <class T>
bool ColumnBaseTyped<T>::operator==(ColumnBaseTyped<T> &column) {
  if (this->size() != column.size()) return false;
  for (unsigned int i = 0; i < this->size(); i++) {
    if ((*this)[i] != column[i]) {
      return false;
    }
  }
  return true;
}

template <class T>
bool ColumnBaseTyped<T>::operator==(ColumnBase &column) {
  ColumnBaseTyped<T> *typed_column =
      dynamic_cast<ColumnBaseTyped<T> *>(&column);
  if (!typed_column) return false;
  return ((*this) == *typed_column);
}

template <class T>
const AggregationResult ColumnBaseTyped<T>::aggregate(
    const AggregationParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);
  assert(dense_column != NULL);

  return dense_column->aggregate(param);
}

template <class T>
const PositionListPairPtr ColumnBaseTyped<T>::join(ColumnPtr join_column,
                                                   const JoinParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);
  assert(dense_column != NULL);
  return dense_column->join(join_column, param);
}

template <class T>
const PositionListPtr ColumnBaseTyped<T>::tid_semi_join(
    ColumnPtr join_column, const JoinParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);
  if (!dense_column) return PositionListPtr();
  return dense_column->tid_semi_join(join_column, param);
}

template <class T>
const BitmapPtr ColumnBaseTyped<T>::bitmap_semi_join(ColumnPtr join_column,
                                                     const JoinParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);
  if (!dense_column) return BitmapPtr();
  return dense_column->bitmap_semi_join(join_column, param);
}

template <class T>
const ColumnPtr ColumnBaseTyped<T>::column_algebra_operation(
    ColumnPtr source_column, const AlgebraOperationParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);
  if (!dense_column) return ColumnPtr();
  /* \todo: Here, we could save overhead by copy only once, because dense value
   columns also
   perform the computation on a copy of the "this" column */
  return dense_column->column_algebra_operation(source_column, param);
}

template <class T>
const ColumnPtr ColumnBaseTyped<T>::column_algebra_operation(
    const boost::any &value, const AlgebraOperationParam &param) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  DenseValueColumnPtr dense_column =
      this->copyIntoDenseValueColumn(param.proc_spec);
  if (!dense_column) return ColumnPtr();
  /* \todo: Here, we could save overhead by copy only once, because dense value
   columns also
   perform the computation on a copy of the "this" column */
  return dense_column->column_algebra_operation(value, param);
}

template <class T>
const AggregationResult ColumnBaseTyped<T>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, const AggregationParam &param) {
  DenseValueColumnPtr column = this->copyIntoDenseValueColumn(param.proc_spec);

  // typename ColumnBaseTyped<T>::TypedVectorPtr values =
  // this->copyIntoPlainVector();
  if (!column) return AggregationResult();
  return column->aggregateByGroupingKeys(grouping_keys, param);
}

template <class Type>
bool ColumnBaseTyped<Type>::store(const std::string &path_to_table_dir) {
  std::string path_to_column(path_to_table_dir);
  path_to_column += "/";
  path_to_column += this->name_;
  if (!quiet && verbose && debug)
    std::cout << "Writing Column " << this->getName() << " to File "
              << path_to_column << std::endl;
  std::ofstream outfile(path_to_column.c_str(),
                        std::ios_base::binary | std::ios_base::out);
  if (!outfile.good()) {
    COGADB_ERROR("Could not store column '"
                     << path_to_column << "'!" << std::endl
                     << "Check whether you have write access to the database "
                     << "directory: '"
                     << RuntimeConfiguration::instance().getPathToDatabase()
                     << "'",
                 "");
  }
  boost::archive::binary_oarchive oa(outfile);
  oa << this->has_primary_key_constraint_;
  oa << this->has_foreign_key_constraint_;
  oa << this->fk_constr_;

  if (!this->statistics_.store(path_to_column)) {
    COGADB_ERROR("Could not store statistics for column '"
                     << path_to_column << "'!" << std::endl
                     << "Check whether you have write access to the database "
                     << "directory: '"
                     << RuntimeConfiguration::instance().getPathToDatabase()
                     << "'",
                 "");
  }

  bool ret = this->store_impl(path_to_table_dir, oa);
  outfile.flush();
  outfile.close();
  return ret;
}

template <class Type>
bool ColumnBaseTyped<Type>::load(const std::string &path_to_table_dir,
                                 ColumnLoaderMode column_loader_mode) {
  std::string path_to_column(path_to_table_dir);
  if (!quiet && verbose && debug)
    std::cout << "Loading column '" << this->name_ << "' from path '"
              << path_to_column << "'..." << std::endl;
  path_to_column += "/";
  path_to_column += this->name_;

  std::ifstream infile(path_to_column.c_str(),
                       std::ios_base::binary | std::ios_base::in);
  boost::archive::binary_iarchive ia(infile);
  ia >> has_primary_key_constraint_;
  ia >> has_foreign_key_constraint_;
  ia >> fk_constr_;

  path_to_table_dir_ = path_to_table_dir;
  path_to_column_ = path_to_column;
  statistics_.load(path_to_column);

  if (column_loader_mode == LOAD_ALL_DATA) {
    return load_impl(path_to_table_dir, ia);
  } else {
    return true;
  }
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(ColumnBaseTyped)

}  // end namespace CoGaDB
