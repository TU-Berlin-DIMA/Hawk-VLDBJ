
#include <boost/make_shared.hpp>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/partition.h>
#include <thrust/sort.h>

#include <backends/gpu/aggregation.hpp>
#include <backends/gpu/gpu_backend.hpp>
#include <backends/gpu/stream_manager.hpp>
#include <backends/processor_backend.hpp>
#include <util/column_grouping_keys.hpp>
#include <util/functions.hpp>
#include <util/getname.hpp>
#include <util/types.hpp>

namespace CoGaDB {

template <typename Type>
const AggregationResult gpu_aggregation(ColumnGroupingKeysPtr grouping_keys,
                                        Type* values, size_t num_elements,
                                        const AggregationParam& param) {
  // expects data sorted after group_column (keys)

  // here starts aggregation
  typedef GroupingKeys::value_type GroupingKeysType;
  typedef boost::shared_ptr<Column<GroupingKeysType> > KeyColumnPtr;
  typedef boost::shared_ptr<Column<Type> > ValueColumnPtr;

  GroupingKeysType* keys = grouping_keys->keys->data();

  AggregationMethod agg_meth = param.agg_func;

  KeyColumnPtr result_keys(new Column<GroupingKeysType>(
      grouping_keys->keys->getName(), grouping_keys->keys->getType(),
      getMemoryID(param.proc_spec)));

  ValueColumnPtr result_values(new Column<Type>(
      "", getAttributeType(typeid(Type)), getMemoryID(param.proc_spec)));
  if (grouping_keys->keys->size() == 0) {
    return std::pair<ColumnPtr, ColumnPtr>(std::make_pair(
        result_keys, result_values));  // ColumnPtr(aggregated_values));
  }

  try {
    result_keys->resize(num_elements);
    result_values->resize(num_elements);
  } catch (std::bad_alloc& e) {
    size_t free;
    size_t total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    assert(err == cudaSuccess);
    COGADB_ERROR("Ran out of memory during Aggregation!"
                     << "Requested "
                     << double(num_elements *
                               (sizeof(GroupingKeysType) + sizeof(Type))) /
                            (1024 * 1024)
                     << " MB GPU memory (#Elements: " << num_elements << ")"
                     << "Currently free GPU memory: "
                     << double(free) / (1024 * 1024) << " MB memory",
                 "");
    return AggregationResult();
  }

  //  thrust::pair<thrust::device_vector<int>::iterator,typename
  //  thrust::device_vector<Type>::iterator> new_end;

  thrust::pair<thrust::device_ptr<GroupingKeysType>, thrust::device_ptr<Type> >
      new_end;
  // we need to convince thrust to call the GPU reduce_by_keys, so we convert
  // our plain
  // pointers to thrust::device_ptr objects
  thrust::device_ptr<GroupingKeysType> keys_begin_ptr =
      thrust::device_pointer_cast(keys);
  thrust::device_ptr<GroupingKeysType> keys_end_ptr =
      thrust::device_pointer_cast(keys + num_elements);
  thrust::device_ptr<Type> values_ptr = thrust::device_pointer_cast(values);
  thrust::device_ptr<GroupingKeysType> result_keys_ptr =
      thrust::device_pointer_cast(result_keys->data());
  thrust::device_ptr<Type> result_values_ptr =
      thrust::device_pointer_cast(result_values->data());

  try {
    if (agg_meth == SUM) {
      thrust::equal_to<GroupingKeysType> binary_pred;
      thrust::plus<Type> binary_op;
      new_end = thrust::reduce_by_key(keys_begin_ptr, keys_end_ptr, values_ptr,
                                      result_keys_ptr, result_values_ptr,
                                      binary_pred, binary_op);
    } else if (agg_meth == MIN) {
      thrust::equal_to<GroupingKeysType> binary_pred;
      thrust::minimum<Type> binary_op;
      new_end = thrust::reduce_by_key(keys_begin_ptr, keys_end_ptr, values_ptr,
                                      result_keys_ptr, result_values_ptr,
                                      binary_pred, binary_op);
    } else if (agg_meth == MAX) {
      thrust::equal_to<GroupingKeysType> binary_pred;
      thrust::maximum<Type> binary_op;
      new_end = thrust::reduce_by_key(keys_begin_ptr, keys_end_ptr, values_ptr,
                                      result_keys_ptr, result_values_ptr,
                                      binary_pred, binary_op);
    } else if (agg_meth == COUNT) {
      thrust::equal_to<GroupingKeysType> binary_pred;
      thrust::plus<TID> binary_op;
      thrust::device_vector<TID> dummy_values(num_elements, 1);
      TID* dummy_values_ptr = thrust::raw_pointer_cast(&dummy_values[0]);
      new_end = thrust::reduce_by_key(
          keys_begin_ptr, keys_end_ptr,
          thrust::device_pointer_cast(dummy_values_ptr), result_keys_ptr,
          result_values_ptr, binary_pred, binary_op);
    } else if (agg_meth == AVERAGE) {
      //                COGADB_FATAL_ERROR("AVERAGE ON GPU not implemented!",
      //                "");
      AggregationParam count_param(param);
      AggregationParam sum_param(param);
      count_param.agg_func = COUNT;
      sum_param.agg_func = SUM;
      AggregationResult count_result;
      AggregationResult sum_result;

      try {
        count_result =
            gpu_aggregation(grouping_keys, values, num_elements, count_param);
      } catch (std::bad_alloc& e) {
        COGADB_ERROR("Ran out of memory during Aggregation!"
                         << std::endl
                         << "GPU Average computation failed!" << std::endl
                         << e.what(),
                     "");
        return AggregationResult();
      }
      try {
        sum_result =
            gpu_aggregation(grouping_keys, values, num_elements, sum_param);
      } catch (std::bad_alloc& e) {
        COGADB_ERROR("Ran out of memory during Aggregation!"
                         << std::endl
                         << "GPU Average computation failed!" << std::endl
                         << e.what(),
                     "");
        return AggregationResult();
      }
      if (!count_result.first || !sum_result.first) return AggregationResult();

      AlgebraOperationParam algebra_param(param.proc_spec, DIV);
      ColumnPtr average = sum_result.second->column_algebra_operation(
          count_result.second, algebra_param);
      if (!average) {
        COGADB_ERROR("GPU Average computation failed!", "");
        return AggregationResult();
      }

      AggregationResult result(sum_result.first, average);
      return result;
    } else {
      std::cerr << "FATAL Error! Unknown Aggregation Method!" << agg_meth
                << std::endl;
    }
  } catch (std::bad_alloc& e) {
    COGADB_ERROR("Ran out of memory during Aggregation!"
                     << std::endl
                     << "For aggregation function " << util::getName(agg_meth)
                     << std::endl
                     << e.what(),
                 "");
    return AggregationResult();
  }
  size_t num_result_elements =
      new_end.first - thrust::device_pointer_cast(result_keys->begin());
  result_keys->resize(num_result_elements);
  result_values->resize(num_result_elements);

  // std::pair<GPU_Base_ColumnPtr,GPU_Base_ColumnPtr>
  // result(GPU_Base_ColumnPtr(result_key_column),GPU_Base_ColumnPtr(result_value_column));
  return AggregationResult(std::make_pair(result_keys, result_values));
}

template <typename T>
const AggregationResult GPU_Aggregation<T>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, T* aggregation_column,
    size_t num_elements, const AggregationParam& param) {
  AggregationResult value_computation_result =
      gpu_aggregation(grouping_keys, aggregation_column, num_elements, param);

  if (!value_computation_result.second) return AggregationResult();

  // compute TIDs for fetching values from grouping columns
  PositionListPtr tids =
      createPositionList(num_elements, getMemoryID(param.proc_spec));
  if (!tids) return AggregationResult();

  thrust::sequence(
      thrust::device_pointer_cast(getPointer(*tids)),
      thrust::device_pointer_cast(getPointer(*tids) + num_elements), 0);

  AggregationParam new_param(param);
  new_param.agg_func = MIN;
  AggregationResult tid_computation_result = gpu_aggregation(
      grouping_keys, getPointer(*tids), num_elements, new_param);
  if (!tid_computation_result.second) return AggregationResult();

  AggregationResult result;
  result.first = tid_computation_result.second;
  result.second = value_computation_result.second;

  return result;
}

template <typename T>
const AggregationResult GPU_Aggregation<T>::aggregate(
    T* aggregation_column, size_t num_elements, const AggregationParam& param) {
  thrust::device_ptr<T> values_ptr_begin =
      thrust::device_pointer_cast(aggregation_column);
  thrust::device_ptr<T> values_ptr_end =
      thrust::device_pointer_cast(aggregation_column + num_elements);

  T aggregate = T();

  AggregationResult result;
  try {
    if (param.agg_func == COUNT) {
      // create column with a single element, which is the number of elements
      result.second = boost::make_shared<Column<T> >(
          std::string(""), INT, 1, num_elements, hype::PD_Memory_0);
      return result;
    } else if (param.agg_func == MIN) {
      aggregate =
          thrust::reduce(values_ptr_begin, values_ptr_end,
                         std::numeric_limits<T>::max(), thrust::minimum<T>());
      result.second = boost::make_shared<Column<T> >(
          std::string(""), getAttributeType(typeid(T)), 1, aggregate,
          hype::PD_Memory_0);
      return result;
    } else if (param.agg_func == MAX) {
      aggregate =
          thrust::reduce(values_ptr_begin, values_ptr_end,
                         std::numeric_limits<T>::min(), thrust::maximum<T>());
      result.second = boost::make_shared<Column<T> >(
          std::string(""), getAttributeType(typeid(T)), 1, aggregate,
          hype::PD_Memory_0);
      return result;
    } else if (param.agg_func == SUM || param.agg_func == AVERAGE) {
      aggregate = thrust::reduce(values_ptr_begin, values_ptr_end, T(0),
                                 thrust::plus<T>());

      if (param.agg_func == AVERAGE) aggregate = aggregate / num_elements;
      // create column with a single element, which is the number of elements
      result.second = boost::make_shared<Column<T> >(
          std::string(""), getAttributeType(typeid(T)), 1, aggregate,
          hype::PD_Memory_0);
      return result;

    } else {
      COGADB_FATAL_ERROR(
          "Unknown or unsupported Algebra Operation: " << param.agg_func, "");
    }
  } catch (std::bad_alloc& e) {
    COGADB_ERROR("Run out of memory in aggregation!", "");
    return AggregationResult();
  }
  return AggregationResult();
}

template <typename T>
const ColumnGroupingKeysPtr GPU_Aggregation<T>::createColumnGroupingKeys(
    T* column, size_t num_elements, const ProcessorSpecification& proc_spec) {
  ColumnGroupingKeysPtr result(new ColumnGroupingKeys(getMemoryID(proc_spec)));
  result->keys->resize(num_elements);

  ProcessorBackend<T>* backend = ProcessorBackend<T>::get(proc_spec.proc_id);
  result->required_number_of_bits =
      backend->getNumberOfRequiredBits(column, num_elements, proc_spec);

  // GroupingKeyType needs to be at least as large as T
  assert(sizeof(T) <= sizeof(GroupingKeys::value_type));

  GroupingKeys::value_type* key_array = result->keys->data();

  thrust::copy(thrust::device_pointer_cast(column),
               thrust::device_pointer_cast(column + num_elements),
               thrust::device_pointer_cast(key_array));

  return result;
}

template <typename T>
size_t GPU_Aggregation<T>::getNumberOfRequiredBits(
    T* column, size_t num_elements, const ProcessorSpecification& proc_spec) {
  thrust::device_ptr<T> begin = thrust::device_pointer_cast(column);
  thrust::device_ptr<T> end =
      thrust::device_pointer_cast(column + num_elements);
  T max = *thrust::max_element(begin, end);
  T min = *thrust::min_element(begin, end);
  //        T max = *thrust::max_element(column, column + num_elements);
  //        T min = *thrust::min_element(column, column + num_elements);
  // check if we need all bits (in case negativ bit is set))
  if (min < 0) return sizeof(T) * 8;
  return getGreaterPowerOfTwo(max);
}

template <>
size_t GPU_Aggregation<uint32_t>::getNumberOfRequiredBits(
    uint32_t* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) {
  thrust::device_ptr<uint32_t> begin = thrust::device_pointer_cast(column);
  thrust::device_ptr<uint32_t> end =
      thrust::device_pointer_cast(column + num_elements);
  uint32_t max = *thrust::max_element(begin, end);
  return getGreaterPowerOfTwo(max);
}

template <>
size_t GPU_Aggregation<uint64_t>::getNumberOfRequiredBits(
    uint64_t* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) {
  thrust::device_ptr<uint64_t> begin = thrust::device_pointer_cast(column);
  thrust::device_ptr<uint64_t> end =
      thrust::device_pointer_cast(column + num_elements);
  uint64_t max = *thrust::max_element(begin, end);
  return getGreaterPowerOfTwo(max);
}

template <>
const AggregationResult GPU_Aggregation<std::string>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, std::string* aggregation_column,
    size_t num_elements, const AggregationParam& param) {
  return AggregationResult();
}

template <>
const AggregationResult GPU_Aggregation<std::string>::aggregate(
    std::string* aggregation_column, size_t num_elements,
    const AggregationParam&) {
  return AggregationResult();
}

template <>
const ColumnGroupingKeysPtr
GPU_Aggregation<std::string>::createColumnGroupingKeys(
    std::string* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) {
  return ColumnGroupingKeysPtr();
}

template <>
size_t GPU_Aggregation<std::string>::getNumberOfRequiredBits(
    std::string* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) {
  return 65;
}

template <>
const AggregationResult GPU_Aggregation<C_String>::aggregateByGroupingKeys(
    ColumnGroupingKeysPtr grouping_keys, C_String* aggregation_column,
    size_t num_elements, const AggregationParam& param) {
  return AggregationResult();
}

template <>
const AggregationResult GPU_Aggregation<C_String>::aggregate(
    C_String* aggregation_column, size_t num_elements,
    const AggregationParam&) {
  return AggregationResult();
}

template <>
const ColumnGroupingKeysPtr GPU_Aggregation<C_String>::createColumnGroupingKeys(
    C_String* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) {
  return ColumnGroupingKeysPtr();
}

template <>
size_t GPU_Aggregation<C_String>::getNumberOfRequiredBits(
    C_String* column, size_t num_elements,
    const ProcessorSpecification& proc_spec) {
  return 65;
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(GPU_Aggregation);

};  // end namespace CoGaDB