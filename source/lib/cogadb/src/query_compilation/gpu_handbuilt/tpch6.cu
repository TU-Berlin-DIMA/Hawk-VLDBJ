#include <query_compilation/gpu_handbuilt/queries.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <boost/make_shared.hpp>
#include <core/column.hpp>
#include <core/global_definitions.hpp>
#include <core/memory_allocator.hpp>
#include <iostream>
#include <limits>
#include <persistence/storage_manager.hpp>
#include "core/variable_manager.hpp"

using namespace std;

namespace CoGaDB {

const TablePtr tpch_q6(bool holistic, bool gpu);

const TablePtr inner_tpch_q6(const size_t& num_elements,
                             float* l_extended_price, int* l_quantity,
                             float* l_discount, uint32_t* l_shipdate);

const TablePtr inner_tpch_q6_holistic(const size_t& num_elements,
                                      float* l_extended_price, int* l_quantity,
                                      float* l_discount, uint32_t* l_shipdate);

const TablePtr inner_tpch_q6_cpu(size_t num_elements, float* l_extended_price,
                                 int* l_quantity, float* l_discount,
                                 uint32_t* l_shipdate);

bool tpch6_hand_compiled(ClientPtr client) {
  tpch_q6(false, false);
  return true;
}

bool tpch6_hand_compiled_kernel(ClientPtr client) {
  tpch_q6(false, true);
  return true;
}

bool tpch6_hand_compiled_holistic_kernel(ClientPtr client) {
  tpch_q6(true, true);
  return true;
}

const TablePtr tpch_q6(bool holistic, bool gpu) {
  // cudaSetDevice(0);
  // cudaDeviceSynchronize();

  CUDA_CHECK_ERROR_RETURN("Initial failure");

  TablePtr lineitem = getTablebyName("LINEITEM");
  assert(lineitem != NULL);
  assert(lineitem->isMaterialized());

  ColumnPtr col_lineitem_shipdate = lineitem->getColumnbyName("L_SHIPDATE");
  ColumnPtr col_lineitem_discount = lineitem->getColumnbyName("L_DISCOUNT");
  ColumnPtr col_lineitem_quantity = lineitem->getColumnbyName("L_QUANTITY");
  ColumnPtr col_lineitem_extended_price =
      lineitem->getColumnbyName("L_EXTENDEDPRICE");
  size_t num_elements = lineitem->getNumberofRows();

  //        if(gpu) {
  //        col_lineitem_shipdate = copy_if_required(col_lineitem_shipdate,
  //        hype::PD_Memory_1);
  //        col_lineitem_discount = copy_if_required(col_lineitem_discount,
  //        hype::PD_Memory_1);
  //        col_lineitem_quantity = copy_if_required(col_lineitem_quantity,
  //        hype::PD_Memory_1);
  //        col_lineitem_extended_price =
  //        copy_if_required(col_lineitem_extended_price, hype::PD_Memory_1);
  //        if(!col_lineitem_shipdate | !col_lineitem_discount |
  //        !col_lineitem_quantity | !col_lineitem_extended_price){
  //            cout << "Error while transfering columns to GPU memory" << endl;
  //            return TablePtr();
  //        }
  //        }

  boost::shared_ptr<Column<float>> typed_col_lineitem_extended_price =
      boost::dynamic_pointer_cast<Column<float>>(col_lineitem_extended_price);
  boost::shared_ptr<Column<int>> typed_col_lineitem_quantity =
      boost::dynamic_pointer_cast<Column<int>>(col_lineitem_quantity);
  boost::shared_ptr<Column<float>> typed_col_lineitem_discount =
      boost::dynamic_pointer_cast<Column<float>>(col_lineitem_discount);
  boost::shared_ptr<Column<uint32_t>> typed_col_lineitem_shipdate =
      boost::dynamic_pointer_cast<Column<uint32_t>>(col_lineitem_shipdate);

  float* d_extended_price = NULL;
  int* d_quantity = NULL;
  float* d_discount = NULL;
  uint32_t* d_shipdate = NULL;

  if (gpu) {
    d_extended_price = customMalloc<float>(DEVICE_MEMORY, num_elements);
    d_quantity = customMalloc<int>(DEVICE_MEMORY, num_elements);
    d_discount = customMalloc<float>(DEVICE_MEMORY, num_elements);
    d_shipdate = customMalloc<uint32_t>(DEVICE_MEMORY, num_elements);
    CUDA_CHECK_ERROR_RETURN("Failure while allocating gpu columns");
    cudaMemcpy(d_extended_price, typed_col_lineitem_extended_price->data(),
               num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_quantity, typed_col_lineitem_quantity->data(),
               num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_discount, typed_col_lineitem_discount->data(),
               num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shipdate, typed_col_lineitem_shipdate->data(),
               num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR_RETURN("Failure while transfering column data to gpu");
  } else {
    d_extended_price = typed_col_lineitem_extended_price->data();
    d_quantity = typed_col_lineitem_quantity->data();
    d_discount = typed_col_lineitem_discount->data();
    d_shipdate = typed_col_lineitem_shipdate->data();
  }

  // Final checkup before computation
  cout << "pointer to l_extended_price " << d_extended_price << endl;
  cout << "pointer to l_quantity " << d_quantity << endl;
  cout << "pointer to l_discount " << d_discount << endl;
  cout << "pointer to l_shipdate " << d_shipdate << endl;

  if (gpu) {
    //        cudaMemset((void *)d_extended_price, 0, sizeof(float) *
    //        num_elements);
    //        cudaDeviceSynchronize();
    //        CUDA_CHECK_ERROR_RETURN("memset 1a failed");
    //        cudaMemset((void *)d_quantity, 0, sizeof(int) * num_elements);
    //        cudaDeviceSynchronize();
    //        CUDA_CHECK_ERROR_RETURN("memset 1b failed");
    //        cudaMemset((void *)d_discount, 0, sizeof(float) * num_elements);
    //        cudaDeviceSynchronize();
    //        CUDA_CHECK_ERROR_RETURN("memset 1c failed");
    //        cudaMemset((void *)d_shipdate, 0, sizeof(uint32_t) *
    //        num_elements);
    //        cudaDeviceSynchronize();
    //        CUDA_CHECK_ERROR_RETURN("memset 1d failed");
    if (holistic) {
      inner_tpch_q6_holistic(num_elements, d_extended_price, d_quantity,
                             d_discount, d_shipdate);
    } else {
      inner_tpch_q6(num_elements, d_extended_price, d_quantity, d_discount,
                    d_shipdate);
    }
  } else {
    inner_tpch_q6_cpu(num_elements, d_extended_price, d_quantity, d_discount,
                      d_shipdate);
  }

  if (gpu) {
    customFree<float>(DEVICE_MEMORY, d_extended_price);
    customFree<int>(DEVICE_MEMORY, d_quantity);
    customFree<float>(DEVICE_MEMORY, d_discount);
    customFree<uint32_t>(DEVICE_MEMORY, d_shipdate);
    CUDA_CHECK_ERROR_RETURN("Failure while freeing gpu columns");
  }

  return TablePtr();
}

#include "tpch6.cuh"

const TablePtr inner_tpch_q6(const size_t& num_elements,
                             float* l_extended_price, int* l_quantity,
                             float* l_discount, uint32_t* l_shipdate) {
  CUDA_CHECK_ERROR_RETURN("Initial failure");

  cout << "TPCH6 - input size: " << num_elements << " rows" << endl;

  char* d_flags = customMalloc<char>(DEVICE_MEMORY, num_elements);
  char* h_flags = customMalloc<char>(HOST_MEMORY, num_elements);
  float* d_products = customMalloc<float>(DEVICE_MEMORY, num_elements);
  float* h_products = customMalloc<float>(HOST_MEMORY, num_elements);
  cudaMemset((void*)d_flags, 0, sizeof(char) * num_elements);
  cudaMemset((void*)d_products, 0, sizeof(float) * num_elements);
  CUDA_CHECK_ERROR_RETURN(
      "Error in device memory allocation for intermediate computation data");

  int gridSize = 128;

  Timestamp begin_query = getTimestamp();

  kernel_select_tpch_q6<<<gridSize, blockSize>>>(num_elements, l_extended_price,
                                                 l_quantity, l_discount,
                                                 l_shipdate, d_flags);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR_RETURN("kernel_select_tpch_q6");

  kernel_arithmetic_tpch_q6<<<gridSize, blockSize>>>(
      num_elements, l_extended_price, l_discount, d_flags, d_products);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR_RETURN("kernel_arithmetic_tpch_q6");

  float sum =
      thrust::reduce(thrust::device_pointer_cast(d_products),
                     thrust::device_pointer_cast(d_products + num_elements));

  Timestamp end_query = getTimestamp();
  double query_time = double(end_query - begin_query) / (1000 * 1000);

  cout << "TPCH6 execution time " << query_time << " ms" << endl;
  cout << "TPCH6 - result: " << sum << endl;

  cudaMemcpy(h_flags, d_flags, num_elements * sizeof(char),
             cudaMemcpyDeviceToHost);
  size_t resultSize = 0;
  for (int i = 0; i < num_elements; i++) {
    if (h_flags[i] > 0) resultSize++;
  }
  cout << "TPCH6 - selection result size: " << resultSize << endl;

  customFree<char>(DEVICE_MEMORY, d_flags);
  customFree<char>(HOST_MEMORY, h_flags);
  customFree<float>(DEVICE_MEMORY, d_products);
  customFree<float>(HOST_MEMORY, h_products);

  return TablePtr();
}

const TablePtr inner_tpch_q6_holistic(const size_t& num_elements,
                                      float* l_extended_price, int* l_quantity,
                                      float* l_discount, uint32_t* l_shipdate) {
  CUDA_CHECK_ERROR_RETURN("Initial failure");

  cout << "TPCH6 - input size: " << num_elements << " rows" << endl;

  dim3 gridDim = computeGridDim(num_elements);
  size_t gridSize = gridDim.x * gridDim.y * gridDim.z;

  float* d_blockSums = customMalloc<float>(DEVICE_MEMORY, gridSize);
  float* h_blockSums = customMalloc<float>(HOST_MEMORY, gridSize);
  cudaMemset((void*)d_blockSums, 0, sizeof(float) * gridSize);
  CUDA_CHECK_ERROR_RETURN(
      "Error in device memory allocation for intermediate computation data");

  Timestamp begin_query = getTimestamp();

  kernel_tpch_q6_holistic<<<computeGridDim(num_elements), blockSize>>>(
      num_elements, l_extended_price, l_quantity, l_discount, l_shipdate,
      d_blockSums);
  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR_RETURN("kernel_tpch_q6_holistic failed");

  float result =
      thrust::reduce(thrust::device_pointer_cast(d_blockSums),
                     thrust::device_pointer_cast(d_blockSums + gridSize));
  CUDA_CHECK_ERROR_RETURN("thrust::reduce failed");

  Timestamp end_query = getTimestamp();
  double query_time = double(end_query - begin_query) / (1000 * 1000);

  cout << "TPCH6 execution time " << query_time << " ms" << endl;
  cout << "TPCH6 - result: " << result << endl;

  customFree<float>(DEVICE_MEMORY, d_blockSums);
  customFree<float>(HOST_MEMORY, h_blockSums);
  CUDA_CHECK_ERROR_RETURN("kernel_tpch_q6_holistic failed");

  return TablePtr();
}

const TablePtr inner_tpch_q6_cpu(size_t num_elements, float* l_extended_price,
                                 int* l_quantity, float* l_discount,
                                 uint32_t* l_shipdate) {
  Timestamp begin_query = getTimestamp();

  double result = 0.0;
  bool selected;
  //        size_t counter = 0;

  for (size_t tid = 0; tid < num_elements; tid++) {
    //            if( (l_shipdate[tid]  >=  19940101)
    //            &&  (l_shipdate[tid]  <   19950101)
    //            &&  (l_discount[tid]  >=      0.05)
    //            &&  (l_discount[tid]  <=      0.07)
    //            &&  (l_quantity[tid]  <         24)){
    //                result += l_extended_price[tid] * l_discount[tid];
    //                counter++;
    //            }

    selected = true;
    selected = selected && (l_shipdate[tid] >= 19940101);
    selected = selected && (l_shipdate[tid] < 19950101);
    selected = selected && (l_discount[tid] >= 0.05f);
    selected = selected && (l_discount[tid] <= 0.07f);
    selected = selected && (l_quantity[tid] < 24);

    if (selected) {
      result += l_extended_price[tid] * l_discount[tid];
    }
  }

  Timestamp end_query = getTimestamp();
  double query_time = double(end_query - begin_query) / (1000 * 1000);

  cout << "TPCH6 CPU execution time " << query_time << " ms" << endl;
  cout << "TPCH6 - result: " << result << endl;
  //        cout << "Matching Tuples: " << counter << endl;
  cout << "Table Size: " << num_elements << "rows" << std::endl;

  return TablePtr();
}

}  // end namespace CoGaDB
