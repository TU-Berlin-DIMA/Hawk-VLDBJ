#ifdef TPCH5QQQ

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

#include "tpch5.cuh"

// -- TPC-H Query 5
// select
//         n_name,
//         sum(l_extendedprice * (1 - l_discount)) as revenue
// from
//         customer,
//         orders,
//         lineitem,
//         supplier,
//         nation,
//         region
// where
//         c_custkey = o_custkey
//         and l_orderkey = o_orderkey
//         and l_suppkey = s_suppkey
//         and c_nationkey = s_nationkey
//         and s_nationkey = n_nationkey
//         and n_regionkey = r_regionkey
//         and r_name = 'ASIA'
//         and o_orderdate >= date '1994-01-01'
//         and o_orderdate < date '1995-01-01'
// group by
//         n_name
// order by
//         revenue desc

using namespace std;

namespace CoGaDB {

TablePtr tpch5();

bool tpch5_holistic_kernel(ClientPtr client) {
  tpch5();
  return true;
}

// typedef struct {
//     int orderkey;
//     float revenue;
//     uint32_t orderdate;
//     int shippriority;
// } ResultTuple;

TablePtr tpch5() {
  CUDA_CHECK_ERROR_RETURN("Initial failure");

  TablePtr lineitem = getTablebyName("LINEITEM");
  assert(lineitem != NULL);
  assert(lineitem->isMaterialized());
  ColumnPtr col_lineitem_orderkey = lineitem->getColumnbyName("L_ORDERKEY");
  ColumnPtr col_lineitem_suppkey = lineitem->getColumnbyName("L_SUPPKEY");
  ColumnPtr col_lineitem_discount = lineitem->getColumnbyName("L_DISCOUNT");
  ColumnPtr col_lineitem_extended_price =
      lineitem->getColumnbyName("L_EXTENDEDPRICE");
  size_t lineitem_num_elements = lineitem->getNumberofRows();

  TablePtr customer = getTablebyName("CUSTOMER");
  assert(customer != NULL);
  assert(customer->isMaterialized());
  ColumnPtr col_customer_custkey = customer->getColumnbyName("C_CUSTKEY");
  ColumnPtr col_customer_nationkey = customer->getColumnbyName("C_NATIONKEY");
  size_t customer_num_elements = customer->getNumberofRows();

  TablePtr orders = getTablebyName("ORDERS");
  assert(orders != NULL);
  assert(orders->isMaterialized());
  ColumnPtr col_orders_orderdate = orders->getColumnbyName("O_ORDERDATE");
  ColumnPtr col_orders_custkey = orders->getColumnbyName("O_CUSTKEY");
  ColumnPtr col_orders_orderkey = orders->getColumnbyName("O_ORDERKEY");
  size_t orders_num_elements = orders->getNumberofRows();

  TablePtr supplier = getTablebyName("SUPPLIER");
  assert(supplier != NULL);
  assert(supplier->isMaterialized());
  ColumnPtr col_supplier_suppkey = supplier->getColumnbyName("S_SUPPKEY");
  ColumnPtr col_supplier_nationkey = supplier->getColumnbyName("S_NATIONKEY");
  size_t supplier_num_elements = supplier->getNumberofRows();

  TablePtr nation = getTablebyName("NATION");
  assert(nation != NULL);
  assert(nation->isMaterialized());
  ColumnPtr col_nation_nationkey = nation->getColumnbyName("N_NATIONKEY");
  ColumnPtr col_nation_name = nation->getColumnbyName("N_NAME");
  ColumnPtr col_nation_regionkey = nation->getColumnbyName("N_REGIONKEY");
  size_t nation_num_elements = nation->getNumberofRows();

  TablePtr region = getTablebyName("REGION");
  assert(region != NULL);
  assert(region->isMaterialized());
  ColumnPtr col_region_regionkey = region->getColumnbyName("R_REGIONKEY");
  ColumnPtr col_region_name = region->getColumnbyName("R_NAME");
  size_t region_num_elements = region->getNumberofRows();

  boost::shared_ptr<Column<int>> lineitem_orderkey =
      boost::dynamic_pointer_cast<Column<int>>(col_lineitem_orderkey);
  boost::shared_ptr<Column<int>> lineitem_suppkey =
      boost::dynamic_pointer_cast<Column<int>>(col_lineitem_suppkey);
  boost::shared_ptr<Column<float>> lineitem_discount =
      boost::dynamic_pointer_cast<Column<float>>(col_lineitem_discount);
  boost::shared_ptr<Column<float>> lineitem_extended_price =
      boost::dynamic_pointer_cast<Column<float>>(col_lineitem_extended_price);

  boost::shared_ptr<Column<int>> customer_custkey =
      boost::dynamic_pointer_cast<Column<int>>(col_customer_custkey);
  boost::shared_ptr<Column<int>> customer_nationkey =
      boost::dynamic_pointer_cast<Column<int>>(col_customer_nationkey);

  boost::shared_ptr<Column<uint32_t>> orders_orderdate =
      boost::dynamic_pointer_cast<Column<uint32_t>>(col_orders_orderdate);
  boost::shared_ptr<Column<int>> orders_custkey =
      boost::dynamic_pointer_cast<Column<int>>(col_orders_custkey);
  boost::shared_ptr<Column<int>> orders_orderkey =
      boost::dynamic_pointer_cast<Column<int>>(col_orders_orderkey);

  boost::shared_ptr<Column<int>> supplier_suppkey =
      boost::dynamic_pointer_cast<Column<int>>(col_supplier_suppkey);
  boost::shared_ptr<Column<int>> supplier_nationkey =
      boost::dynamic_pointer_cast<Column<int>>(col_supplier_nationkey);

  boost::shared_ptr<Column<int>> nation_nationkey =
      boost::dynamic_pointer_cast<Column<int>>(col_nation_nationkey);
  boost::shared_ptr<DictionaryCompressedColumn<std::string>> nation_name =
      boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>>(
          col_nation_name);
  boost::shared_ptr<Column<int>> nation_regionkey =
      boost::dynamic_pointer_cast<Column<int>>(col_nation_regionkey);

  boost::shared_ptr<Column<int>> region_regionkey =
      boost::dynamic_pointer_cast<Column<int>>(col_region_regionkey);
  boost::shared_ptr<DictionaryCompressedColumn<std::string>> region_name =
      boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>>(
          col_region_name);

  DictionaryCompressedColumn<std::string>& refdict = *nation_name;
  AttributeType attType = refdict.getType();
  ColumnType colType = refdict.getColumnType();
  cout << "AtrributeType: " << util::getName(attType)
       << ", ColumnType: " << util::getName(colType) << endl;
  uint32_t* idData = refdict.getIdData();
  for (int i = 0; i < refdict.getNumberOfRows(); i++) {
    uint32_t compressed = idData[i];
    cout << compressed << ", " << refdict.reverseLookup(compressed) << "."
         << endl;
  }

  int* d_lineitem_orderkey =
      customMalloc<int>(DEVICE_MEMORY, lineitem_num_elements);
  int* d_lineitem_suppkey =
      customMalloc<int>(DEVICE_MEMORY, lineitem_num_elements);
  float* d_lineitem_discount =
      customMalloc<float>(DEVICE_MEMORY, lineitem_num_elements);
  float* d_lineitem_extended_price =
      customMalloc<float>(DEVICE_MEMORY, lineitem_num_elements);

  int* d_customer_custkey =
      customMalloc<int>(DEVICE_MEMORY, customer_num_elements);
  int* d_customer_nationkey =
      customMalloc<int>(DEVICE_MEMORY, customer_num_elements);

  uint32_t* d_orders_orderdate =
      customMalloc<uint32_t>(DEVICE_MEMORY, orders_num_elements);
  int* d_orders_custkey = customMalloc<int>(DEVICE_MEMORY, orders_num_elements);
  int* d_orders_orderkey =
      customMalloc<int>(DEVICE_MEMORY, orders_num_elements);

  int* d_supplier_suppkey =
      customMalloc<int>(DEVICE_MEMORY, supplier_num_elements);
  int* d_supplier_nationkey =
      customMalloc<int>(DEVICE_MEMORY, supplier_num_elements);

  int* d_nation_nationkey =
      customMalloc<int>(DEVICE_MEMORY, nation_num_elements);
  uint32_t* d_nation_name =
      customMalloc<uint32_t>(DEVICE_MEMORY, nation_num_elements);
  int* d_nation_regionkey =
      customMalloc<int>(DEVICE_MEMORY, nation_num_elements);

  int* d_region_regionkey =
      customMalloc<int>(DEVICE_MEMORY, region_num_elements);
  uint32_t* d_region_name =
      customMalloc<uint32_t>(DEVICE_MEMORY, region_num_elements);
  CUDA_CHECK_ERROR_RETURN("Failure while allocating gpu columns");

  cudaMemcpy(d_lineitem_orderkey, lineitem_orderkey->data(),
             lineitem_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lineitem_suppkey, lineitem_suppkey->data(),
             lineitem_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lineitem_discount, lineitem_discount->data(),
             lineitem_num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lineitem_extended_price, lineitem_extended_price->data(),
             lineitem_num_elements * sizeof(float), cudaMemcpyHostToDevice);

  cudaMemcpy(d_customer_custkey, customer_custkey->data(),
             customer_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_customer_nationkey, customer_nationkey->data(),
             customer_num_elements * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_orders_orderdate, orders_orderdate->data(),
             orders_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders_custkey, orders_orderkey->data(),
             orders_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders_orderkey, orders_orderkey->data(),
             orders_num_elements * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_supplier_suppkey, supplier_suppkey->data(),
             supplier_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_supplier_nationkey, supplier_nationkey->data(),
             supplier_num_elements * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_nation_nationkey, nation_nationkey->data(),
             nation_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nation_name, nation_name->getIdData(),
             nation_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_nation_regionkey, nation_regionkey->data(),
             nation_num_elements * sizeof(int), cudaMemcpyHostToDevice);

  cudaMemcpy(d_region_regionkey, region_regionkey->data(),
             region_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_region_name, region_name->getIdData(),
             region_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR_RETURN("Failure while allocating gpu columns");

  // -- TPC-H Query 5
  // select  n_name,
  //         sum(l_extendedprice * (1 - l_discount)) as revenue
  // from
  //         customer,
  //         orders,
  //         lineitem,
  //         supplier,
  //         nation,
  //         region
  // where
  //         c_custkey = o_custkey
  //         and l_orderkey = o_orderkey
  //         and l_suppkey = s_suppkey
  //         and c_nationkey = s_nationkey
  //         and s_nationkey = n_nationkey
  //         and n_regionkey = r_regionkey
  //         and r_name = 'ASIA'
  //         and o_orderdate >= date '1994-01-01'
  //         and o_orderdate < date '1995-01-01'
  // group by
  //         n_name
  // order by
  //         revenue desc
  //

  int numGroups = 25;
  int numProbeBlocks = divUp(lineitem_num_elements, (size_t)kProbeBlockSize);
  float* d_groupAggregatePerBlock =
      customMalloc<float>(DEVICE_MEMORY, numProbeBlocks * numGroups);
  int* d_groupCountPerBlock =
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

#endif
