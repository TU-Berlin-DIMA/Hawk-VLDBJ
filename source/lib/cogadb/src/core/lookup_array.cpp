
#include <core/lookup_array.hpp>

#include <compression/dictionary_compressed_column.hpp>
#include <core/base_table.hpp>
#include <core/column.hpp>
#include <core/column_base_typed.hpp>
#include <lookup_table/lookup_column.hpp>
#include <persistence/storage_manager.hpp>

#ifdef ENABLE_CDK_USAGE
#include <hardware_optimizations/main_memory_joins/hash_joins.hpp>
#include <hardware_optimizations/primitives.hpp>
#endif
#include <core/runtime_configuration.hpp>

#include <core/processor_data_cache.hpp>
#include <util/utility_functions.hpp>
#include "core/copy_function_factory.hpp"

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

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
 *essentially a tid list describing which values of a typed materialized column
 *are included in the LookupArray.
 *  \details   This class is indentended to be a base class, so it has a virtual
 *destruktor and pure virtual methods, which need to be implemented in a derived
 *class.
 *  \author    Sebastian Breß
 *  \version   0.2
 *  \date      2013
 *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
 *http://www.gnu.org/licenses/lgpl-3.0.txt
 */

ColumnPtr createLookupArrayForColumn(ColumnPtr col, PositionListPtr tids_) {
  if (!col || !tids_) {
    return ColumnPtr();
  }

  AttributeType type = col->getType();
  std::string column_name = col->getName();

  switch (type) {
    case INT:
      return ColumnPtr(new LookupArray<int>(column_name, INT, col, tids_));
    case DATE:
    case UINT32:
      return ColumnPtr(
          new LookupArray<uint32_t>(column_name, type, col, tids_));
    case CHAR:
      return ColumnPtr(new LookupArray<char>(column_name, type, col, tids_));
    case FLOAT:
      return ColumnPtr(new LookupArray<float>(column_name, FLOAT, col, tids_));
    case DOUBLE:
      return ColumnPtr(
          new LookupArray<double>(column_name, DOUBLE, col, tids_));
    case OID:
      return ColumnPtr(new LookupArray<TID>(column_name, OID, col, tids_));
    case VARCHAR: {
      if (col->type() == typeid(std::string)) {
        return ColumnPtr(
            new LookupArray<std::string>(column_name, VARCHAR, col, tids_));
      } else {
        return ColumnPtr(
            new LookupArray<char*>(column_name, VARCHAR, col, tids_));
      }
    }
    case BOOLEAN: {
      std::cout << "Fatal Error! invalid AttributeType: " << type
                << " for Column: " << column_name
                << " Note: bool is currently not supported, will be added "
                   "again in the future!"
                << std::endl;
      return ColumnPtr();
    }
    default: {
      std::cout << "Fatal Error! invalid AttributeType: " << type
                << " for Column: " << column_name << std::endl;
      return ColumnPtr();
    }
  }
}

PositionListPtr getPositonListfromLookupArray(ColumnPtr col) {
  if (!col) return PositionListPtr();
  // is LookupArray?
  if (col->isMaterialized() || col->isCompressed()) {
    // input is not LookupArray, so no Positionlist can be retrieved
    return PositionListPtr();
  }

  AttributeType type = col->getType();

  if (type == INT) {
    shared_pointer_namespace::shared_ptr<LookupArray<int> > lookup_array =
        shared_pointer_namespace::dynamic_pointer_cast<LookupArray<int> >(col);
    return lookup_array->getPositionList();
  } else if (type == FLOAT) {
    shared_pointer_namespace::shared_ptr<LookupArray<float> > lookup_array =
        shared_pointer_namespace::dynamic_pointer_cast<LookupArray<float> >(
            col);
    return lookup_array->getPositionList();
  } else if (type == VARCHAR) {
    shared_pointer_namespace::shared_ptr<LookupArray<std::string> >
        lookup_array = shared_pointer_namespace::dynamic_pointer_cast<
            LookupArray<std::string> >(col);
    return lookup_array->getPositionList();
  } else if (type == BOOLEAN) {
    std::cout << "Fatal Error! invalid AttributeType: " << type
              << " for Column: " << col->getName()
              << " Note: bool is currently not supported, will be added again "
                 "in the future!"
              << std::endl;
    return PositionListPtr();
  } else {
    std::cout << "Fatal Error! invalid AttributeType: " << type
              << " for Column: " << col->getName() << std::endl;
    return PositionListPtr();
  }
}

// typedef shared_pointer_namespace::shared_ptr<LookupArray> LookupArrayPtr;

/***************** Start of Implementation Section ******************/

template <class T>
LookupArray<T>::LookupArray(const std::string& name, AttributeType db_type,
                            ColumnPtr column, PositionListPtr tids)
    : ColumnBaseTyped<T>(name, db_type, LOOKUP_ARRAY),
      column_(
          shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<T> >(
              column)),
      tids_(tids) {
  assert(column_ != NULL);
  assert(tids_ != NULL);
  assert(db_type == column->getType());
  this->is_loaded_ = column->isLoadedInMainMemory();
  this->fk_constr_ = column->getForeignKeyConstraint();
  this->has_foreign_key_constraint_ = column->hasForeignKeyConstraint();
  //    		if(!column_->isLoadedInMainMemory()){
  //			loadColumnFromDisk(column_);
  //		}
}

template <class T>
LookupArray<T>::~LookupArray() {}

template <class T>
bool LookupArray<T>::insert(const boost::any&) {
  return false;
}
template <class T>
bool LookupArray<T>::insert(const T& new_Value) {
  return false;
}
template <class T>
bool LookupArray<T>::update(TID, const boost::any&) {
  return false;
}

template <class T>
bool LookupArray<T>::update(PositionListPtr, const boost::any&) {
  return false;
}

template <class T>
bool LookupArray<T>::remove(TID) {
  return false;
}

// assumes tid list is sorted ascending
template <class T>
bool LookupArray<T>::remove(PositionListPtr) {
  return false;
}

template <class T>
bool LookupArray<T>::append(boost::shared_ptr<ColumnBaseTyped<T> > typed_col) {
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  typedef boost::shared_ptr<LookupArray> LookupArrayPtr;

  LookupArrayPtr intermediate_column;
  if (typed_col->getColumnType() == LOOKUP_ARRAY) {
    intermediate_column = boost::dynamic_pointer_cast<LookupArray>(typed_col);
  } else {
    COGADB_FATAL_ERROR("Cannot append a non lookup array to a lookup array!",
                       "");
    return false;
  }
  assert(intermediate_column != NULL);
  return this->tids_->append(intermediate_column->tids_);
}

template <class T>
bool LookupArray<T>::clearContent() {
  return false;
}

#ifdef ENABLE_CDK_USAGE
template <class T>
const PositionListPtr LookupArray<T>::selection(const SelectionParam& param) {
  GatherParam gather_param(ProcessorSpecification(param.proc_spec));
  if ((this->column_->size() == 0) || (this->tids_->size() == 0)) {
    return PositionListPtr();
  }

  ColumnPtr placed_column = copy_if_required(column_, param.proc_spec);
  if (!placed_column) return PositionListPtr();
  PositionListPtr placed_tids = copy_if_required(tids_, param.proc_spec);
  if (!placed_tids) return PositionListPtr();

  ColumnPtr materialized_column =
      placed_column->gather(placed_tids, gather_param);
  if (materialized_column) {
    // we assume that there is no Lookup_Array on a Lookup_Array,
    // so we will not get infinite recursion here
    return materialized_column->selection(param);
  }
  return PositionListPtr();

  //            if(this->column_->isMaterialized() &&
  //            !this->column_->isCompressed()){
  //                shared_pointer_namespace::shared_ptr<Column<T> > base_column
  //                = shared_pointer_namespace::dynamic_pointer_cast<Column<T>
  //                >(this->column_);
  //                assert(base_column!=NULL);
  //
  //                T* base_array = base_column->data();
  //                //hype::util::begin_ptr(base_column->getContent());
  //                assert(base_array!=NULL);
  //
  //                TID* tids = this->tids_->data();
  //                assert(tids!=NULL);
  //
  //                T* input_column = (T*)
  //                malloc(sizeof(T)*this->tids_->size());
  //                //CoGaDB::RuntimeConfiguration::instance().
  //                //fetch the relevant tids specified in TID list
  //                //for very small arrays, use serial gather to reduce
  //                threading overhead
  //                if(this->tids_->size()<300){
  //                        CDK::util::serial_gather(base_array, tids,
  //                        this->tids_->size(), input_column);
  //                }else{
  //                        CDK::util::parallel_gather(base_array, tids,
  //                        this->tids_->size(), input_column,
  //                        boost::thread::hardware_concurrency());
  //                }
  //
  //                //perform parallel selection on the generated input column
  //                PositionListPtr result =
  //                CDK::selection::parallel_selection(input_column,
  //                this->tids_->size(), value_for_comparison, comp,
  //                boost::thread::hardware_concurrency());
  //                free(input_column);
  //                return result;
  //            }else{
  //                //for strings we can use the default method
  //                return
  //                CoGaDB::ColumnBaseTyped<T>::selection(value_for_comparison,
  //                comp);
  //            }
}

//        template<class T>
//        const PositionListPtr LookupArray<T>::selection(ColumnPtr
//        comparison_column, const ValueComparator comp){
//            //same behaviour as in parent class
//            return CoGaDB::ColumnBaseTyped<T>::selection(comparison_column,
//            comp);
//        }

template <class T>
const BitmapPtr LookupArray<T>::bitmap_selection(const SelectionParam& param) {
  GatherParam gather_param(ProcessorSpecification(param.proc_spec));
  if ((this->column_->size() == 0) || (this->tids_->size() == 0)) {
    return BitmapPtr();
  }

  ColumnPtr placed_column = copy_if_required(column_, param.proc_spec);
  if (!placed_column) return BitmapPtr();
  PositionListPtr placed_tids = copy_if_required(tids_, param.proc_spec);
  if (!placed_tids) return BitmapPtr();

  ColumnPtr materialized_column =
      placed_column->gather(placed_tids, gather_param);
  if (materialized_column) {
    // we assume that there is no Lookup_Array on a Lookup_Array,
    // so we will not get infinite recursion here
    return materialized_column->bitmap_selection(param);
  }
  return BitmapPtr();
}

template <class T>
const PositionListPairPtr LookupArray<T>::hash_join(ColumnPtr join_column_) {
  assert(this->size() <= join_column_->size());
  if (join_column_->type() != typeid(T)) {
    std::cout << "Fatal Error!!! Typemismatch for columns " << this->name_
              << " and " << join_column_->getName() << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }
  T* this_array = this->materializeToArray();
  assert(this_array != NULL);
  T* input_column = NULL;
  PositionListPairPtr ret;
  if (join_column_->isMaterialized()) {
    shared_pointer_namespace::shared_ptr<Column<T> > join_column =
        shared_pointer_namespace::dynamic_pointer_cast<Column<T> >(
            join_column_);
    assert(join_column != NULL);
    input_column = join_column->data();
    assert(input_column != NULL);

    ret = CDK::join::serial_hash_join(this_array, this->size(), input_column,
                                      join_column->size());
    // ret = CDK::main_memory_joins::serial_hash_join(this_array, this->size(),
    // input_column, join_column->size());

  } else if (!join_column_->isMaterialized() &&
             !join_column_->isCompressed()) {  // Lookup Array?
    shared_pointer_namespace::shared_ptr<LookupArray<T> > join_column =
        shared_pointer_namespace::dynamic_pointer_cast<LookupArray<T> >(
            join_column_);
    assert(join_column != NULL);

    T* input_column = join_column->materializeToArray();

    assert(this_array != NULL);
    ret = CDK::join::serial_hash_join(this_array, this->size(), input_column,
                                      join_column->size());
    // ret = CDK::main_memory_joins::serial_hash_join(this_array, this->size(),
    // input_column, join_column->size());
    // ok, job done, cleanup temporary array
    free(input_column);
  } else if (join_column_->isCompressed()) {
    // if compressed, use old style hash join via ColumnBaseTyped class
    ret = CoGaDB::ColumnBaseTyped<T>::hash_join(join_column_);
  } else {
    // detected impossible case, abort
    COGADB_FATAL_ERROR(
        "Detected Invalid Column Class! Not MAterialized, No LookupARray and "
        "no CompressedColumn!",
        "");
    ret = PositionListPairPtr();
  }
  free(this_array);
  return ret;
}

//        template<>
//	const PositionListPairPtr LookupArray<int>::hash_join(ColumnPtr
// join_column_){
//            assert(this->size()<=join_column_->size());
//            if(join_column_->type()!=typeid(int)){
//                    std::cout << "Fatal Error!!! Typemismatch for columns " <<
//                    this->name_  << " and " << join_column_->getName() <<
//                    std::endl;
//                    std::cout << "File: " << __FILE__ << " Line: " << __LINE__
//                    << std::endl;
//                    exit(-1);
//            }
//            int* this_array=this->materializeToArray();
//            assert(this_array!=NULL);
//            int* input_column=NULL;
//            PositionListPairPtr ret;
//            if(join_column_->isMaterialized()){
//                shared_pointer_namespace::shared_ptr<Column<int> > join_column
//                = shared_pointer_namespace::dynamic_pointer_cast<Column<int>
//                >(join_column_);
//                assert(join_column!=NULL);
//                input_column=join_column->data();
//                assert(input_column!=NULL);
//
//                //ret = CDK::join::serial_hash_join(this_array, this->size(),
//                input_column, join_column->size());
//                ret = CDK::main_memory_joins::serial_hash_join(this_array,
//                this->size(), input_column, join_column->size());
//
//            }else if (!join_column_->isMaterialized() &&
//            !join_column_->isCompressed()){ //Lookup Array?
//                shared_pointer_namespace::shared_ptr<LookupArray<int> >
//                join_column =
//                shared_pointer_namespace::dynamic_pointer_cast<LookupArray<int>
//                >(join_column_);
//                assert(join_column!=NULL);
//
//                int* input_column = join_column->materializeToArray();
//
//                assert(this_array!=NULL);
////                ret = CDK::join::serial_hash_join(this_array, this->size(),
/// input_column, join_column->size());
//                ret = CDK::main_memory_joins::serial_hash_join(this_array,
//                this->size(), input_column, join_column->size());
//                //ok, job done, cleanup temporary array
//                free(input_column);
//            }else if(join_column_->isCompressed()){
//                //if compressed, use old style hash join via ColumnBaseTyped
//                class
//                ret = CoGaDB::ColumnBaseTyped<int>::hash_join(join_column_);
//            }else{
//                //detected impossible case, abort
//                COGADB_FATAL_ERROR("Detected Invalid Column Class! Not
//                MAterialized, No LookupARray and no CompressedColumn!","");
//                ret = PositionListPairPtr();
//            }
//            free(this_array);
//            return ret;
//        }

template <class T>
const PositionListPairPtr LookupArray<T>::radix_join(ColumnPtr join_column_) {
  COGADB_FATAL_ERROR(
      "called non INT instantiation of LookupArray<T>::radix_join, which only "
      "supports integer columns!",
      "");
  return PositionListPairPtr();
}

template <>
inline const PositionListPairPtr LookupArray<int>::radix_join(
    ColumnPtr join_column_) {
  assert(this->size() <= join_column_->size());
  if (join_column_->type() != typeid(int)) {
    std::cout << "Fatal Error!!! Typemismatch for columns " << this->name_
              << " and " << join_column_->getName() << std::endl;
    std::cout << "File: " << __FILE__ << " Line: " << __LINE__ << std::endl;
    exit(-1);
  }
  int* this_array = this->materializeToArray();
  assert(this_array != NULL);
  int* input_column = NULL;
  PositionListPairPtr ret;
  if (join_column_->isMaterialized()) {
    shared_pointer_namespace::shared_ptr<Column<int> > join_column =
        shared_pointer_namespace::dynamic_pointer_cast<Column<int> >(
            join_column_);
    assert(join_column != NULL);
    input_column = join_column->data();
    assert(input_column != NULL);

    ret = CDK::join::radix_join(this_array, this->size(), input_column,
                                join_column->size());

  } else if (!join_column_->isMaterialized() &&
             !join_column_->isCompressed()) {  // Lookup Array?
    shared_pointer_namespace::shared_ptr<LookupArray<int> > join_column =
        shared_pointer_namespace::dynamic_pointer_cast<LookupArray<int> >(
            join_column_);
    assert(join_column != NULL);

    int* input_column = join_column->materializeToArray();

    assert(this_array != NULL);
    ret = CDK::join::radix_join(this_array, this->size(), input_column,
                                join_column->size());
    // ok, job done, cleanup temporary array
    free(input_column);
  } else if (join_column_->isCompressed()) {
    // if compressed, use old style hash join via ColumnBaseTyped class
    ret = CoGaDB::ColumnBaseTyped<int>::hash_join(join_column_);
  } else {
    // detected impossible case, abort
    COGADB_FATAL_ERROR(
        "Detected Invalid Column Class! Not MAterialized, No LookupARray and "
        "no CompressedColumn!",
        "");
    ret = PositionListPairPtr();
  }
  free(this_array);
  return ret;
}

#endif

template <class T>
const boost::any LookupArray<T>::get(TID tid) {
  return boost::any((*this)[tid]);
}

template <class T>
void LookupArray<T>::print() const throw() {
  const shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> > column =
      column_;
  const PositionListPtr tids = tids_;

  std::cout << "Lookup Array for Column " << column_->getName() << " ";
  if (column_->isMaterialized()) {
    std::cout << "which is a materialized Column" << std::endl;
  } else {
    std::cout << "which is a LookupArray of a Lookup column" << std::endl;
  }
  std::cout << "| values | Translatetion TIDS | Index in Lookup Table |"
            << std::endl;
  for (unsigned int i = 0; i < tids->size(); i++) {
    std::cout << "| " << (*column_)[(*tids_)[i]] << " | " << (*tids_)[i]
              << " | " << i << " |" << std::endl;
  }
}
template <class T>
size_t LookupArray<T>::size() const throw() {
  return tids_->size();
}

template <class T>
const ColumnPtr LookupArray<T>::materialize() throw() {
  GatherParam gather_param{ProcessorSpecification{hype::PD0}};
  return column_->gather(tids_, gather_param);
}

template <class T>
hype::ProcessingDeviceMemoryID LookupArray<T>::getMemoryID() const {
  // in case memory locations differ, always return the memory ID of the non
  // CPU,
  // so the algorithms working with the column can detect whether
  // this column needs to be placed first
  if (column_->getMemoryID() == tids_->getMemoryID()) {
    return column_->getMemoryID();
  } else {
    if (column_->getMemoryID() == hype::PD_Memory_0) {
      return tids_->getMemoryID();
    } else if (tids_->getMemoryID() == hype::PD_Memory_0) {
      return column_->getMemoryID();
    } else {
      COGADB_FATAL_ERROR(
          "An Unhandled Error Condition Occurred, check the code for details!",
          "");
      return hype::PD_Memory_0;
    }
  }
}

/* \todo REMOVE this function, it is deprecated and should no longer be used!
 * Use copyIntoDenseValueColumn Instead!*/
template <class T>
T* LookupArray<T>::materializeToArray() throw() {
  if (this->size() == 0 || this->column_->size() == 0) return NULL;
  shared_pointer_namespace::shared_ptr<Column<T> > base_column =
      shared_pointer_namespace::dynamic_pointer_cast<Column<T> >(this->column_);
  assert(base_column != NULL);
  T* base_column_array =
      base_column->data();  // hype::util::begin_ptr(base_column->values_);
  assert(base_column_array != NULL);
  // allocate temporary array
  T* input_column = (T*)malloc(sizeof(T) * this->size());
  assert(input_column != NULL);
  // CoGaDB::RuntimeConfiguration::instance().
  // fetch the relevant tids specified in TID list
  // for very small arrays, use serial gather to reduce threading overhead
  //                CDK::util::serial_gather(base_column_array, tids->data(),
  //                tids->size(), input_column);
  if (this->size() < 300) {
    CDK::util::serial_gather(base_column_array, this->tids_->data(),
                             this->size(), input_column);
  } else {
    CDK::util::parallel_gather(base_column_array, this->tids_->data(),
                               this->size(), input_column,
                               boost::thread::hardware_concurrency());
  }
  return input_column;
}

template <class T>
const ColumnPtr LookupArray<T>::copy() const {
  PositionListPtr new_tids = boost::dynamic_pointer_cast<PositionList>(
      tids_->copy());  //(createPositionList(*tids_));
  return ColumnPtr(
      new LookupArray<T>(this->name_, this->db_type_, this->column_, new_tids));
}

template <class T>
const ColumnPtr LookupArray<T>::copy(
    const hype::ProcessingDeviceMemoryID& mem_id) const {
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;

  ColumnPtr placed_column;
  if (column_->getMemoryID() != mem_id && isCPUMemory(mem_id))
    placed_column = DataCacheManager::instance()
                        .getDataCache(column_->getMemoryID())
                        .getHostColumn(column_);
  if (!placed_column) placed_column = copy_if_required(column_, mem_id);
  if (!placed_column) return ColumnPtr();
  PositionListPtr placed_tids = copy_if_required(tids_, mem_id);
  if (!placed_tids) return ColumnPtr();
  // return ColumnPtr(new
  // LookupArray<T>(this->name_,this->db_type_,placed_column,placed_tids));
  //

  ProcessorSpecification proc_spec(hype::util::getProcessingDeviceID(mem_id));
  GatherParam gather_param(proc_spec);  // ProcessorSpecification(hype::PD0));
  // ColumnPtr col = this->column_->gather(tids_, gather_param);
  return placed_column->gather(placed_tids, gather_param);

  //            DenseValueColumnPtr dense_column =
  //            this->copyIntoDenseValueColumn(ProcessorSpecification(hype::PD0));
  //            assert(dense_column!=NULL);
  ////            return copy_if_required(dense_column, mem_id);
  //            return CoGaDB::copy(dense_column, mem_id);

  //            return this->copyIntoDenseValueColumn();
  //            return ColumnBaseTyped<T>::copy(mem_id);
}

template <class T>
const typename ColumnBaseTyped<T>::DenseValueColumnPtr
LookupArray<T>::copyIntoDenseValueColumn(
    const ProcessorSpecification& proc_spec) const {
  typedef typename ColumnBaseTyped<T>::DenseValueColumn DenseValueColumn;
  typedef typename ColumnBaseTyped<T>::DenseValueColumnPtr DenseValueColumnPtr;
  // assert(proc_spec.proc_id==hype::PD0);
  //
  ColumnPtr placed_column = copy_if_required(column_, proc_spec);
  if (!placed_column) return DenseValueColumnPtr();
  PositionListPtr placed_tids = copy_if_required(tids_, proc_spec);
  if (!placed_tids) return DenseValueColumnPtr();
  GatherParam gather_param(proc_spec);  // ProcessorSpecification(hype::PD0));
  //            ColumnPtr col = this->column_->gather(tids_, gather_param);
  ColumnPtr col = placed_column->gather(placed_tids, gather_param);
  if (!col) return DenseValueColumnPtr();
  if (col->getColumnType() != PLAIN_MATERIALIZED) {
    boost::shared_ptr<ColumnBaseTyped<T> > typed_col =
        boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(col);
    assert(typed_col != NULL);
    return typed_col->copyIntoDenseValueColumn(proc_spec);
  } else {
    return boost::dynamic_pointer_cast<DenseValueColumn>(col);
  }
}

template <class T>
const ColumnPtr LookupArray<T>::gather(PositionListPtr tid_list,
                                       const GatherParam& gather_param) {
  if (!tid_list) return ColumnPtr();
  // return column_->gather();
  hype::ProcessingDeviceMemoryID mem_id =
      hype::util::getMemoryID(gather_param.proc_spec.proc_id);
  //            Column<T>* result = new Column<T>(this->name_,this->db_type_,
  //            mem_id);
  ColumnPtr copied_column = copy_if_required(column_, mem_id);
  PositionListPtr copied_tids = copy_if_required(tids_, mem_id);
  if (!copied_column || !copied_tids) return ColumnPtr();

  column_ = boost::dynamic_pointer_cast<ColumnBaseTyped<T> >(copied_column);
  tids_ = copied_tids;

  PositionListPtr placed_tid_list = copy_if_required(tid_list, mem_id);
  if (!placed_tid_list) return ColumnPtr();

  ColumnPtr new_tid_list = this->tids_->gather(placed_tid_list, gather_param);

  if (!new_tid_list) return ColumnPtr();

  PositionListPtr new_tids =
      boost::dynamic_pointer_cast<PositionList>(new_tid_list);
  assert(new_tids != NULL);
  return column_->gather(new_tids, gather_param);

  //            result->resize(tid_list->size());
  //            TID* tids = tid_list->data();
  //            T* data = result->data();
  //            for(unsigned int i=0;i<tid_list->size();i++){
  //		data[i]=(*this)[(*tid_list)[i]];
  //            }
  //            return ColumnPtr(result);
}

template <class T>
const ColumnGroupingKeysPtr LookupArray<T>::createColumnGroupingKeys(
    const ProcessorSpecification& proc_spec) const {
  ColumnPtr placed_column = copy_if_required(column_, proc_spec);
  if (!placed_column) return ColumnGroupingKeysPtr();
  //            PositionListPtr placed_tids = copy_if_required(tids_,
  //            proc_spec);
  //            if(!placed_tids) return ColumnGroupingKeysPtr();
  GatherParam gather_param(proc_spec);
  ColumnPtr materialized_column = placed_column->gather(tids_, gather_param);
  if (materialized_column) {
    // we assume that there is no Lookup_Array on a Lookup_Array,
    // so we will not get infinite recursion here
    return materialized_column->createColumnGroupingKeys(proc_spec);
  }
  return ColumnGroupingKeysPtr();
}

template <class T>
size_t LookupArray<T>::getNumberOfRequiredBits() const {
  return this->column_->getNumberOfRequiredBits();
}

//        template<class T>
//        const AggregationResult
//        LookupArray<T>::aggregateByGroupingKeys(ColumnGroupingKeysPtr
//        grouping_keys, AggregationMethod agg_meth, AggregationAlgorithm
//        agg_alg){ //const AggregationParam& param){
//            AggregationParam param(ProcessorSpecification(hype::PD0),
//            agg_meth, agg_alg);
//
//            ColumnPtr copied_column = copy_if_required(column_,
//            param.proc_spec);
//            PositionListPtr copied_tids = copy_if_required(tids_,
//            param.proc_spec);
//            if(!copied_column || !copied_tids) return AggregationResult();
//            GatherParam gather_param(param.proc_spec);
//            ColumnPtr materialized_column = copied_column->gather(copied_tids,
//            gather_param);
////            ColumnPtr materialized_column = column_->gather(tids_,
/// GatherParam(ProcessorSpecification(param.proc_spec)));
//            if(materialized_column){
//                //we assume that there is no Lookup_Array on a Lookup_Array,
//                //so we will not get infinite recursion here
//                return
//                materialized_column->aggregateByGroupingKeys(grouping_keys,
//                param);
//            }
//            assert(materialized_column!=NULL);
//            return AggregationResult();
//        }

template <class T>
const AggregationResult LookupArray<T>::aggregate(
    const AggregationParam& param) {
  if ((this->column_->size() == 0) || (this->tids_->size() == 0)) {
    return AggregationResult();
  }
  ColumnPtr copied_column = copy_if_required(column_, param.proc_spec);
  PositionListPtr copied_tids = copy_if_required(tids_, param.proc_spec);
  if (!copied_column || !copied_tids) return AggregationResult();
  GatherParam gather_param(param.proc_spec);
  ColumnPtr materialized_column =
      copied_column->gather(copied_tids, gather_param);
  if (materialized_column) {
    // we assume that there is no Lookup_Array on a Lookup_Array,
    // so we will not get infinite recursion here
    return materialized_column->aggregate(param);
  }
  return AggregationResult();
}

template <class T>
const PositionListPairPtr LookupArray<T>::join(ColumnPtr join_column,
                                               const JoinParam& param) {
  if (!join_column) return PositionListPairPtr();
  ColumnPtr copied_column = copy_if_required(column_, param.proc_spec);
  PositionListPtr copied_tids = copy_if_required(tids_, param.proc_spec);
  if (!copied_column || !copied_tids) return PositionListPairPtr();
  // Empty result
  if ((this->column_->size() == 0) || (this->tids_->size() == 0) ||
      join_column->size() == 0) {
    PositionListPairPtr r(new PositionListPair());
    r->first =
        createPositionList(0, hype::util::getMemoryID(param.proc_spec.proc_id));
    r->second =
        createPositionList(0, hype::util::getMemoryID(param.proc_spec.proc_id));
    return r;
  }
  GatherParam gather_param(param.proc_spec);
  ColumnPtr materialized_column =
      copied_column->gather(copied_tids, gather_param);
  if (materialized_column) {
    // we assume that there is no Lookup_Array on a Lookup_Array,
    // so we will not get infinite recursion here
    return materialized_column->join(join_column, param);
  }
  return PositionListPairPtr();
}

template <class T>
const ColumnPtr LookupArray<T>::column_algebra_operation(
    ColumnPtr source_column, const AlgebraOperationParam& param) {
  //            COGADB_FATAL_ERROR("Called column_algebra_operation on Lookup
  //            array, which is not allowed!","");
  ColumnPtr copied_column = copy_if_required(column_, param.proc_spec);
  PositionListPtr copied_tids = copy_if_required(tids_, param.proc_spec);
  if (!copied_column || !copied_tids) return ColumnPtr();
  GatherParam gather_param(param.proc_spec);
  ColumnPtr materialized_column =
      copied_column->gather(copied_tids, gather_param);
  if (!materialized_column) return ColumnPtr();
  return materialized_column->column_algebra_operation(source_column, param);
}

template <class T>
const ColumnPtr LookupArray<T>::column_algebra_operation(
    const boost::any& value, const AlgebraOperationParam& param) {
  //            COGADB_FATAL_ERROR("Called column_algebra_operation on Lookup
  //            array, which is not allowed!","");
  ColumnPtr copied_column = copy_if_required(column_, param.proc_spec);
  PositionListPtr copied_tids = copy_if_required(tids_, param.proc_spec);
  if (!copied_column || !copied_tids) return ColumnPtr();
  GatherParam gather_param(param.proc_spec);
  ColumnPtr materialized_column =
      copied_column->gather(copied_tids, gather_param);
  if (!materialized_column) return ColumnPtr();
  return materialized_column->column_algebra_operation(value, param);
}

template <class T>
const PositionListPtr LookupArray<T>::sort(const SortParam& param) {
  ColumnPtr copied_column = copy_if_required(column_, param.proc_spec);
  PositionListPtr copied_tids = copy_if_required(tids_, param.proc_spec);
  if (!copied_column || !copied_tids) return PositionListPtr();
  GatherParam gather_param(param.proc_spec);
  ColumnPtr materialized_column =
      copied_column->gather(copied_tids, gather_param);
  if (materialized_column) {
    return materialized_column->sort(param);
  }
  assert(materialized_column != NULL);
  return PositionListPtr();
}

/***************** relational operations on LookupArrays which return lookup
 * tables *****************/

template <class T>
bool LookupArray<T>::load_impl(const std::string& path,
                               boost::archive::binary_iarchive& ia) {
  return false;
}
template <class T>
bool LookupArray<T>::store_impl(const std::string& path,
                                boost::archive::binary_oarchive& oa) {
  return false;
}
template <class T>
bool LookupArray<T>::store(const std::string&) {
  return false;
}
template <class T>
bool LookupArray<T>::load(const std::string& path,
                          ColumnLoaderMode column_loader_mode) {
  if (column_->isLoadedInMainMemory()) {
    return false;
  } else {
    //                std::cout << "Loading Column " << column_->getName() << "
    //                (" << column_->type().name() << ")" << std::endl;
    column_->load(path, LOAD_ALL_DATA);
    column_->setStatusLoadedInMainMemory(true);
    this->is_loaded_ = true;
    return true;
  }
}
template <class T>
bool LookupArray<T>::isMaterialized() const throw() {
  return false;
}
template <class T>
bool LookupArray<T>::isCompressed() const throw() {
  return false;
}
template <class T>
T& LookupArray<T>::operator[](const TID index) {
  return (*column_)[(*tids_)[index]];
}

template <class T>
size_t LookupArray<T>::getSizeinBytes() const throw() {
  return tids_->capacity() * sizeof(typename PositionList::value_type);
}

template <class T>
PositionListPtr LookupArray<T>::getPositionList() {
  return this->tids_;
}

template <class T>
shared_pointer_namespace::shared_ptr<ColumnBaseTyped<T> >
LookupArray<T>::getIndexedColumn() {
  return this->column_;
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(LookupArray)

/***************** End of Implementation Section ******************/

}  // end namespace CogaDB
