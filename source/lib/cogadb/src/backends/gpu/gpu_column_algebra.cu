

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <backends/gpu/column_algebra.hpp>
#include <backends/gpu/stream_manager.hpp>

namespace CoGaDB {

template <typename T>
bool GPU_ColumnAlgebra<T>::column_algebra_operation(
    T* target_column, T* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  thrust::device_ptr<T> target_column_begin =
      thrust::device_pointer_cast(target_column);
  thrust::device_ptr<T> target_column_end =
      thrust::device_pointer_cast(target_column + num_elements);
  thrust::device_ptr<T> source_column_begin =
      thrust::device_pointer_cast(source_column);

  // using thrust::placeholders;
  // the third agument is an unnamed functor!, see
  // http://stackoverflow.com/questions/9671104/how-to-decrement-each-element-of-a-device-vector-by-a-constant
  // for details
  if (param.alg_op == ADD) {
    thrust::transform(target_column_begin, target_column_end,
                      source_column_begin, target_column_begin,
                      thrust::plus<T>());
  } else if (param.alg_op == SUB) {
    thrust::transform(target_column_begin, target_column_end,
                      source_column_begin, target_column_begin,
                      thrust::minus<T>());
  } else if (param.alg_op == MUL) {
    thrust::transform(target_column_begin, target_column_end,
                      source_column_begin, target_column_begin,
                      thrust::multiplies<T>());
  } else if (param.alg_op == DIV) {
    thrust::transform(target_column_begin, target_column_end,
                      source_column_begin, target_column_begin,
                      thrust::divides<T>());
  } else {
    COGADB_FATAL_ERROR(
        "Unknown Algebra Operation! Value: " << (int)param.alg_op, "");
  }

  return true;
}

template <typename T>
bool GPU_ColumnAlgebra<T>::double_precision_column_algebra_operation(
    double* target_column, T* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  if (num_elements == 0) return true;

  thrust::device_ptr<T> source_column_begin =
      thrust::device_pointer_cast(source_column);
  thrust::device_ptr<T> source_column_end =
      thrust::device_pointer_cast(source_column + num_elements);

  thrust::device_vector<double> v;

  try {
    v.insert(v.begin(), source_column_begin, source_column_end);
  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during "
                 "GPU_ColumnAlgebra<T>::double_precision_column_algebra_"
                 "operation!"
              << std::endl;
    return false;
  }

  double* double_source_column = thrust::raw_pointer_cast(&v.front());
  assert(double_source_column != NULL);

  return GPU_ColumnAlgebra<double>::column_algebra_operation(
      target_column, double_source_column, num_elements, param);
}

template <typename T>
bool GPU_ColumnAlgebra<T>::column_algebra_operation(
    T* column, size_t num_elements, T value,
    const AlgebraOperationParam& param) {
  thrust::device_ptr<T> data_begin = thrust::device_pointer_cast(column);
  thrust::device_ptr<T> data_end =
      thrust::device_pointer_cast(column + num_elements);
  // using thrust::placeholders;
  // the third agument is an unnamed functor!, see
  // http://stackoverflow.com/questions/9671104/how-to-decrement-each-element-of-a-device-vector-by-a-constant
  // for details
  if (param.alg_op == ADD) {
    thrust::for_each(data_begin, data_end, thrust::placeholders::_1 += value);
  } else if (param.alg_op == SUB) {
    thrust::for_each(data_begin, data_end, thrust::placeholders::_1 -= value);
  } else if (param.alg_op == MUL) {
    thrust::for_each(data_begin, data_end, thrust::placeholders::_1 *= value);
  } else if (param.alg_op == DIV) {
    thrust::for_each(data_begin, data_end, thrust::placeholders::_1 /= value);
  } else {
    COGADB_FATAL_ERROR(
        "Unknown Algebra Operation! Value: " << (int)param.alg_op, "");
  }

  return true;
}

template <>
bool GPU_ColumnAlgebra<std::string>::column_algebra_operation(
    std::string* target_column, std::string* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  return false;
}

template <>
bool GPU_ColumnAlgebra<std::string>::double_precision_column_algebra_operation(
    double* target_column, std::string* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  return false;
}

template <>
bool GPU_ColumnAlgebra<std::string>::column_algebra_operation(
    std::string* column, size_t num_elements, std::string value,
    const AlgebraOperationParam& param) {
  return false;
}

template <>
bool GPU_ColumnAlgebra<C_String>::column_algebra_operation(
    C_String* target_column, C_String* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  return false;
}

template <>
bool GPU_ColumnAlgebra<C_String>::double_precision_column_algebra_operation(
    double* target_column, C_String* source_column, size_t num_elements,
    const AlgebraOperationParam& param) {
  return false;
}

template <>
bool GPU_ColumnAlgebra<C_String>::column_algebra_operation(
    C_String* column, size_t num_elements, C_String value,
    const AlgebraOperationParam& param) {
  return false;
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(GPU_ColumnAlgebra);

};  // end namespace CoGaDB