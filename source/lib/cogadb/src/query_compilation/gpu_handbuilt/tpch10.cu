

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
#include <vector>
#include "core/variable_manager.hpp"
// for dictionary compressed column
#include <compression/dictionary_compressed_column.hpp>
#include <util/getname.hpp>
// hashtable
#include <query_compilation/gpu_utilities/util/cuda_griddim.h>
#include <query_compilation/gpu_utilities/util/divup.h>
#include <query_compilation/gpu_utilities/compilation_hashtables/GPU_CHT32_2.cuh>
// reduce (sort, limit)
#include <algorithm>
#include <query_compilation/gpu_utilities/util/sequence.cuh>

#include "tpch10.cuh"

// -- TPC-H Query 10
// select
//         c_custkey,
//         c_name,
//         sum(l_extendedprice * (1 - l_discount)) as revenue,
//         c_acctbal,
//         n_name,
//         c_address,
//         c_phone,
//         c_comment
// from
//         customer,
//         orders,
//         lineitem,
//         nation
// where
//         c_custkey = o_custkey
//         and l_orderkey = o_orderkey
//         and o_orderdate >= date '1993-10-01'
//         and o_orderdate < date '1994-01-01'
//         and l_returnflag = 'R'
//         and c_nationkey = n_nationkey
// group by
//         c_custkey,
//         c_name,
//         c_acctbal,
//         c_phone,
//         n_name,
//         c_address,
//         c_comment
// order by
//         revenue desc
// limit 20

using namespace std;

namespace CoGaDB {

TablePtr tpch10();

bool tpch10_holistic_kernel(ClientPtr client) {
  tpch10();
  return true;
}

TablePtr tpch10() {
  CUDA_CHECK_ERROR_RETURN("Initial failure");

  // build customer ht

  // build orders ht (filter)

  // build nation ht

  // probe (filter) lineitem -> orders -> customer -> nation
  // aggregate revenue group by customer

  int numGroups = 25;
  int numProbeBlocks = divUp(lineitem_num_elements, (size_t)kProbeBlockSize);
  float *d_groupAggregatePerBlock =
      customMalloc<float>(DEVICE_MEMORY, numProbeBlocks * numGroups);
  int *d_groupCountPerBlock =
      customMalloc<int>(DEVICE_MEMORY, numProbeBlocks * numGroups);
  vector<float> groupAggregatePerBlock(numProbeBlocks * numGroups);
  vector<int> groupCountPerBlock(numProbeBlocks * numGroups);
  Timestamp begin_query = getTimestamp();

  double region_selectivity = 1;
  GPU_CHT32_2::HashTable regionHashTable = GPU_CHT32_2::createHashTable(
      region_num_elements * region_selectivity, 1.25);
  dim3 gridDim =
      computeGridDim(region_num_elements, kGridSize, kBuildBlockSize);

  regionBuildKernel<<<gridDim, kBuildBlockSize>>>(
      regionHashTable, d_region_regionkey, d_region_name, region_num_elements);

  double nation_selectivity = 1;
  GPU_CHT32_2::HashTable nationHashTable = GPU_CHT32_2::createHashTable(
      nation_num_elements * nation_selectivity, 1.25);
  gridDim = computeGridDim(nation_num_elements, kGridSize, kBuildBlockSize);

  nationBuildKernel<<<gridDim, kBuildBlockSize>>>(
      nationHashTable, d_nation_nationkey, nation_num_elements);

  double supplier_selectivity = 1;
  GPU_CHT32_2::HashTable supplierHashTable = GPU_CHT32_2::createHashTable(
      supplier_num_elements * supplier_selectivity, 1.25);
  gridDim = computeGridDim(supplier_num_elements, kGridSize, kBuildBlockSize);

  supplierBuildKernel<<<gridDim, kBuildBlockSize>>>(
      supplierHashTable, d_supplier_suppkey, supplier_num_elements);

  double customer_selectivity = 1;
  GPU_CHT32_2::HashTable customerHashTable = GPU_CHT32_2::createHashTable(
      customer_num_elements * customer_selectivity, 1.25);
  gridDim = computeGridDim(customer_num_elements, kGridSize, kBuildBlockSize);

  customerBuildKernel<<<gridDim, kBuildBlockSize>>>(
      customerHashTable, d_customer_custkey, customer_num_elements);

  double orders_selectivity = 1;
  GPU_CHT32_2::HashTable ordersHashTable = GPU_CHT32_2::createHashTable(
      orders_num_elements * orders_selectivity, 1.25);
  gridDim = computeGridDim(orders_num_elements, kGridSize, kBuildBlockSize);

  ordersBuildKernel<<<gridDim, kBuildBlockSize>>>(
      ordersHashTable, d_orders_orderkey, d_orders_orderdate,
      orders_num_elements);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR_RETURN("build failed");

  Timestamp end_build_start_probe = getTimestamp();

  gridDim = computeGridDim(lineitem_num_elements, kGridSize, kProbeBlockSize);
  lineitemProbeKernel<<<gridDim, kProbeBlockSize>>>(
      regionHashTable, nationHashTable, supplierHashTable, customerHashTable,
      ordersHashTable, d_lineitem_suppkey, d_lineitem_orderkey,
      d_supplier_nationkey, d_orders_custkey, d_nation_regionkey,
      d_customer_nationkey, lineitem_num_elements, d_lineitem_extended_price,
      d_lineitem_discount, d_nation_name, numGroups, d_groupAggregatePerBlock,
      d_groupCountPerBlock);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR_RETURN("probe failed");

  Timestamp end_probe_start_reduce = getTimestamp();

  cudaMemcpy(&groupAggregatePerBlock[0], d_groupAggregatePerBlock,
             sizeof(float) * numProbeBlocks * numGroups,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(&groupCountPerBlock[0], d_groupCountPerBlock,
             sizeof(int) * numProbeBlocks * numGroups, cudaMemcpyDeviceToHost);
  CUDA_CHECK_ERROR_RETURN("probe failed");
  vector<float> aggregateResult(numGroups);
  vector<int> aggregateCount(numGroups);
  for (int i = 0; i < numProbeBlocks; i++) {
    for (int j = 0; j < numGroups; j++) {
      aggregateResult[j] += groupAggregatePerBlock[i * numGroups + j];
      aggregateCount[j] += groupCountPerBlock[i * numGroups + j];
    }
  }

  Timestamp end_query = getTimestamp();

  GPU_CHT32_2::printHashTable(regionHashTable);
  GPU_CHT32_2::printHashTable(nationHashTable);
  GPU_CHT32_2::printHashTable(supplierHashTable);
  GPU_CHT32_2::printHashTable(customerHashTable);
  GPU_CHT32_2::printHashTable(ordersHashTable);

  for (int j = 0; j < numGroups; j++) {
    if (aggregateResult[j] > 0.0)
      cout << nation_name->reverseLookup(j) << ", " << aggregateResult[j]
           << ", " << aggregateCount[j] << endl;
  }

  double build_time =
      double(end_build_start_probe - begin_query) / (1000 * 1000);
  double probe_time =
      double(end_probe_start_reduce - end_build_start_probe) / (1000 * 1000);
  double reduce_time =
      double(end_query - end_probe_start_reduce) / (1000 * 1000);
  double query_time = double(end_query - begin_query) / (1000 * 1000);

  cout << "TPCH5 execution time " << query_time << " ms" << endl;
  cout << "build: " << build_time << " ms, "
       << "probe: " << probe_time << " ms, "
       << "reduce: " << reduce_time << " ms, " << endl;

  GPU_CHT32_2::freeHashTable(regionHashTable);
  GPU_CHT32_2::freeHashTable(nationHashTable);
  GPU_CHT32_2::freeHashTable(supplierHashTable);
  GPU_CHT32_2::freeHashTable(customerHashTable);
  GPU_CHT32_2::freeHashTable(ordersHashTable);

  customFree<float>(DEVICE_MEMORY, d_groupAggregatePerBlock);
  customFree<int>(DEVICE_MEMORY, d_groupCountPerBlock);

  customFree<int>(DEVICE_MEMORY, d_lineitem_orderkey);
  customFree<int>(DEVICE_MEMORY, d_lineitem_suppkey);
  customFree<float>(DEVICE_MEMORY, d_lineitem_discount);
  customFree<float>(DEVICE_MEMORY, d_lineitem_extended_price);
  customFree<int>(DEVICE_MEMORY, d_customer_custkey);
  customFree<int>(DEVICE_MEMORY, d_customer_nationkey);
  customFree<uint32_t>(DEVICE_MEMORY, d_orders_orderdate);
  customFree<int>(DEVICE_MEMORY, d_orders_custkey);
  customFree<int>(DEVICE_MEMORY, d_orders_orderkey);
  customFree<int>(DEVICE_MEMORY, d_supplier_suppkey);
  customFree<int>(DEVICE_MEMORY, d_supplier_nationkey);
  customFree<int>(DEVICE_MEMORY, d_nation_nationkey);
  customFree<uint32_t>(DEVICE_MEMORY, d_nation_name);
  customFree<int>(DEVICE_MEMORY, d_nation_regionkey);
  customFree<int>(DEVICE_MEMORY, d_region_regionkey);
  customFree<uint32_t>(DEVICE_MEMORY, d_region_name);

  return TablePtr();
}
}
