//#ifdef TPCH3QQQ

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

#include <query_compilation/gpu_utilities/compilation_hashtables/GPU_CHT32_2.cuh>

#include <query_compilation/gpu_utilities/util/cuda_griddim.h>
#include <query_compilation/gpu_utilities/util/divup.h>
// reduce (sort, limit)
#include <algorithm>
#include <query_compilation/gpu_utilities/util/sequence.cuh>
#include "tpch3.cuh"

// -- TPC-H Query 3
// select  l_orderkey,
//         sum(l_extendedprice * (1 - l_discount)) as revenue,
//         o_orderdate,
//         o_shippriority
// from    customer, orders, lineitem
// where   c_mktsegment = 'BUILDING'
//         and c_custkey = o_custkey
//         and l_orderkey = o_orderkey
//         and o_orderdate < date '1995-03-15'
//         and l_shipdate > date '1995-03-15'
// group by
//         l_orderkey,
//         o_orderdate,
//         o_shippriority
// order by
//         revenue desc,
//         o_orderdate
// limit 10

using namespace std;

namespace CoGaDB {

TablePtr tpch3();

bool tpch3_holistic_kernel(ClientPtr client) {
  tpch3();
  return true;
}

typedef struct {
  int orderkey;
  float revenue;
  uint32_t orderdate;
  int shippriority;
} ResultTuple;

TablePtr tpch3() {
  CUDA_CHECK_ERROR_RETURN("Initial failure");

  TablePtr lineitem = getTablebyName("LINEITEM");
  assert(lineitem != NULL);
  assert(lineitem->isMaterialized());
  ColumnPtr col_lineitem_orderkey = lineitem->getColumnbyName("L_ORDERKEY");
  ColumnPtr col_lineitem_discount = lineitem->getColumnbyName("L_DISCOUNT");
  ColumnPtr col_lineitem_extended_price =
      lineitem->getColumnbyName("L_EXTENDEDPRICE");
  ColumnPtr col_lineitem_shipdate = lineitem->getColumnbyName("L_SHIPDATE");
  size_t lineitem_num_elements = lineitem->getNumberofRows();

  TablePtr customer = getTablebyName("CUSTOMER");
  assert(customer != NULL);
  assert(customer->isMaterialized());
  ColumnPtr col_customer_mktsegment = customer->getColumnbyName("C_MKTSEGMENT");
  ColumnPtr col_customer_custkey = customer->getColumnbyName("C_CUSTKEY");
  size_t customer_num_elements = customer->getNumberofRows();

  TablePtr orders = getTablebyName("ORDERS");
  assert(orders != NULL);
  assert(orders->isMaterialized());
  ColumnPtr col_orders_shippriority = orders->getColumnbyName("O_SHIPPRIORITY");
  ColumnPtr col_orders_orderdate = orders->getColumnbyName("O_ORDERDATE");
  ColumnPtr col_orders_orderkey = orders->getColumnbyName("O_ORDERKEY");
  ColumnPtr col_orders_custkey = orders->getColumnbyName("O_CUSTKEY");
  size_t orders_num_elements = orders->getNumberofRows();

  boost::shared_ptr<Column<int>> tcol_lineitem_orderkey =
      boost::dynamic_pointer_cast<Column<int>>(col_lineitem_orderkey);
  boost::shared_ptr<Column<float>> tcol_lineitem_discount =
      boost::dynamic_pointer_cast<Column<float>>(col_lineitem_discount);
  boost::shared_ptr<Column<float>> tcol_lineitem_extended_price =
      boost::dynamic_pointer_cast<Column<float>>(col_lineitem_extended_price);
  boost::shared_ptr<Column<uint32_t>> tcol_lineitem_shipdate =
      boost::dynamic_pointer_cast<Column<uint32_t>>(col_lineitem_shipdate);
  boost::shared_ptr<DictionaryCompressedColumn<std::string>>
      dict_customer_mktsegment =
          boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>>(
              col_customer_mktsegment);
  boost::shared_ptr<Column<int>> tcol_customer_custkey =
      boost::dynamic_pointer_cast<Column<int>>(col_customer_custkey);
  boost::shared_ptr<Column<int>> tcol_orders_shippriority =
      boost::dynamic_pointer_cast<Column<int>>(col_orders_shippriority);
  boost::shared_ptr<Column<uint32_t>> tcol_orders_orderdate =
      boost::dynamic_pointer_cast<Column<uint32_t>>(col_orders_orderdate);
  boost::shared_ptr<Column<int>> tcol_orders_orderkey =
      boost::dynamic_pointer_cast<Column<int>>(col_orders_orderkey);
  boost::shared_ptr<Column<int>> tcol_orders_custkey =
      boost::dynamic_pointer_cast<Column<int>>(col_orders_custkey);

  DictionaryCompressedColumn<std::string>& refdict = *dict_customer_mktsegment;
  AttributeType attType = refdict.getType();
  ColumnType colType = refdict.getColumnType();
  cout << "AtrributeType: " << util::getName(attType)
       << ", ColumnType: " << util::getName(colType) << endl;
  uint32_t* idData = refdict.getIdData();
  for (int i = 0; i < 20; i++) {
    uint32_t compressed = idData[i];
    cout << compressed << ", " << refdict.reverseLookup(compressed) << "."
         << endl;
  }

  int* d_lineitem_orderkey =
      customMalloc<int>(DEVICE_MEMORY, lineitem_num_elements);
  float* d_lineitem_discount =
      customMalloc<float>(DEVICE_MEMORY, lineitem_num_elements);
  float* d_lineitem_extended_price =
      customMalloc<float>(DEVICE_MEMORY, lineitem_num_elements);
  uint32_t* d_lineitem_shipdate =
      customMalloc<uint32_t>(DEVICE_MEMORY, lineitem_num_elements);
  uint32_t* d_customer_mktsegment =
      customMalloc<uint32_t>(DEVICE_MEMORY, customer_num_elements);
  int* d_customer_custkey =
      customMalloc<int>(DEVICE_MEMORY, customer_num_elements);
  int* d_orders_shippriority =
      customMalloc<int>(DEVICE_MEMORY, orders_num_elements);
  uint32_t* d_orders_orderdate =
      customMalloc<uint32_t>(DEVICE_MEMORY, orders_num_elements);
  int* d_orders_orderkey =
      customMalloc<int>(DEVICE_MEMORY, orders_num_elements);
  int* d_orders_custkey = customMalloc<int>(DEVICE_MEMORY, orders_num_elements);
  CUDA_CHECK_ERROR_RETURN("Failure while allocating gpu columns");

  cudaMemcpy(d_lineitem_orderkey, tcol_lineitem_orderkey->data(),
             lineitem_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lineitem_discount, tcol_lineitem_discount->data(),
             lineitem_num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lineitem_extended_price, tcol_lineitem_extended_price->data(),
             lineitem_num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lineitem_shipdate, tcol_lineitem_shipdate->data(),
             lineitem_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_customer_mktsegment, dict_customer_mktsegment->getIdData(),
             customer_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_customer_custkey, tcol_customer_custkey->data(),
             customer_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders_shippriority, tcol_orders_shippriority->data(),
             orders_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders_orderdate, tcol_orders_orderdate->data(),
             orders_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders_orderkey, tcol_orders_orderkey->data(),
             orders_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders_custkey, tcol_orders_custkey->data(),
             orders_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR_RETURN("Failure while allocating gpu columns");

  // aggregate over primary key attribute
  size_t cardinality = orders_num_elements;
  float* d_aggregateRevenue = customMalloc<float>(DEVICE_MEMORY, cardinality);
  cudaMemset(d_aggregateRevenue, 0x00, sizeof(float) * cardinality);
  vector<float> h_aggregateRevenue(cardinality);

  // size_t numResultBlocks = divUp(maxOrderKey, (size_t)kReduceBlockSize);
  int* d_ResultOrderKey = customMalloc<int>(DEVICE_MEMORY, cardinality);
  float* d_ResultRevenue = customMalloc<float>(DEVICE_MEMORY, cardinality);
  uint32_t* d_ResultOrderDate =
      customMalloc<uint32_t>(DEVICE_MEMORY, cardinality);
  int* d_ResultShipPriority = customMalloc<int>(DEVICE_MEMORY, cardinality);

  // -- TPC-H Query 3
  // select  l_orderkey,
  //         sum(l_extendedprice * (1 - l_discount)) as revenue,
  //         o_orderdate,
  //         o_shippriority
  // from    customer, orders, lineitem
  // where   c_mktsegment = 'BUILDING'
  //         and c_custkey = o_custkey
  //         and l_orderkey = o_orderkey
  //         and o_orderdate < date '1995-03-15'
  //         and l_shipdate > date '1995-03-15'
  // group by
  //         l_orderkey,
  //         o_orderdate,
  //         o_shippriority
  // order by
  //         revenue desc,
  //         o_orderdate
  // limit 10

  // build primary key tables
  Timestamp begin_query = getTimestamp();

  double orders_selectivity = 0.7;
  GPU_CHT32_2::HashTable* ordersHashTable = GPU_CHT32_2::createHashTable(
      orders_num_elements * orders_selectivity, 1.25);
  dim3 gridDim =
      computeGridDim(orders_num_elements, kGridSize, kBuildBlockSize);

  ordersBuildKernel<<<gridDim, kBuildBlockSize>>>(
      *ordersHashTable, d_orders_orderkey, d_orders_orderdate,
      orders_num_elements);

  double customer_selectivity = 0.3;
  GPU_CHT32_2::HashTable* customerHashTable = GPU_CHT32_2::createHashTable(
      customer_num_elements * customer_selectivity, 1.25);
  gridDim = computeGridDim(customer_num_elements, kGridSize, kBuildBlockSize);

  customerBuildKernel<<<gridDim, kBuildBlockSize>>>(
      *customerHashTable, d_customer_custkey, d_customer_mktsegment,
      customer_num_elements);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR_RETURN("build failed");
  Timestamp end_build_start_probe = getTimestamp();

  gridDim = computeGridDim(lineitem_num_elements, kGridSize, kProbeBlockSize);
  lineorderProbeKernel<<<gridDim, kProbeBlockSize>>>(
      *customerHashTable, *ordersHashTable, d_lineitem_orderkey,
      d_orders_custkey, d_lineitem_extended_price, d_lineitem_discount,
      d_lineitem_shipdate, lineitem_num_elements, d_aggregateRevenue);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR_RETURN("probe failed");
  Timestamp end_probe_start_reduce = getTimestamp();

  char* d_flags;
  cudaMalloc((void**)&d_flags, sizeof(char) * cardinality);
  cudaMemset(d_flags, 0x00, cardinality);
  uint32_t *d_indices, *d_selectedIndices;
  int* d_numSelectedIndices;
  int numSelectedIndices;
  cudaMalloc((void**)&d_numSelectedIndices, sizeof(int));
  cudaMalloc((void**)&d_indices, sizeof(uint32_t) * cardinality);
  cudaMalloc((void**)&d_selectedIndices, sizeof(uint32_t) * cardinality);
  sequence<<<256, 256>>>(d_indices, cardinality);
  flagAggregate<<<256, 256>>>(d_aggregateRevenue, cardinality, d_flags);
  cudaDeviceSynchronize();

  CUDA_CHECK_ERROR_RETURN("reduce failed");

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_indices,
                             d_flags, d_selectedIndices, d_numSelectedIndices,
                             cardinality);
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  // Run selection
  cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_indices,
                             d_flags, d_selectedIndices, d_numSelectedIndices,
                             cardinality);
  CUDA_CHECK_ERROR_RETURN("reduce failed");

  // order by with limit for aggregation on primary key column
  cudaMemcpy(&h_aggregateRevenue[0], d_aggregateRevenue,
             cardinality * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&numSelectedIndices, d_numSelectedIndices, sizeof(int),
             cudaMemcpyDeviceToHost);
  vector<uint32_t> h_selectedIndices(numSelectedIndices);
  cudaMemcpy(&h_selectedIndices[0], d_selectedIndices,
             numSelectedIndices * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  CUDA_CHECK_ERROR_RETURN("reduce failed");

  Timestamp reduce_gpu_end = getTimestamp();

  vector<ResultTuple> topTuples;
  float minInsertRevenue = 0.0;
  int limit = 10;

  for (size_t i = 0; i < numSelectedIndices; i++) {
    size_t orderIndex = h_selectedIndices[i];

    if (h_aggregateRevenue[orderIndex] > minInsertRevenue) {
      ResultTuple tuple;
      tuple.orderkey = (tcol_orders_orderkey->data())[orderIndex];
      tuple.revenue = h_aggregateRevenue[orderIndex];
      tuple.orderdate = (tcol_orders_orderdate->data())[orderIndex];
      tuple.shippriority = (tcol_orders_shippriority->data())[orderIndex];

      int insert = 0;
      while (insert < topTuples.size() &&
             topTuples[insert].revenue > tuple.revenue)
        insert++;

      topTuples.insert(topTuples.begin() + insert, tuple);
      int minRevenueIndex = min((int)topTuples.size(), limit) - 1;
      minInsertRevenue = topTuples[minRevenueIndex].revenue;
    }
  }

  Timestamp end_query = getTimestamp();

  GPU_CHT32_2::printHashTable(*ordersHashTable, 20);
  GPU_CHT32_2::printHashTable(*customerHashTable, 20);

  double build_time =
      double(end_build_start_probe - begin_query) / (1000 * 1000);
  double probe_time =
      double(end_probe_start_reduce - end_build_start_probe) / (1000 * 1000);
  double reduce_time =
      double(end_query - end_probe_start_reduce) / (1000 * 1000);
  double query_time = double(end_query - begin_query) / (1000 * 1000);
  double reduce_time_gpu =
      double(reduce_gpu_end - end_probe_start_reduce) / (1000 * 1000);

  cout << "TPCH3 execution time " << query_time << " ms" << endl;
  cout << "build: " << build_time << " ms, "
       << "probe: " << probe_time << " ms, "
       << "reduce: " << reduce_time << " ms, "
       << "reduce (only gpu): " << reduce_time_gpu << " ms." << endl;

  cout << "topTuples.size(): " << topTuples.size() << endl;
  cout << "numSelectedIndices: " << numSelectedIndices << endl;

  cout << "orderkey, revenue, orderdate, shippriority" << endl;
  for (int i = 0; i < 10; i++) {
    cout << topTuples[i].orderkey << ", " << topTuples[i].revenue << ", "
         << topTuples[i].orderdate << ", " << topTuples[i].shippriority << endl;
  }

  GPU_CHT32_2::freeHashTable(ordersHashTable);
  GPU_CHT32_2::freeHashTable(customerHashTable);

  customFree<int>(DEVICE_MEMORY, d_ResultOrderKey);
  customFree<float>(DEVICE_MEMORY, d_ResultRevenue);
  customFree<uint32_t>(DEVICE_MEMORY, d_ResultOrderDate);
  customFree<int>(DEVICE_MEMORY, d_ResultShipPriority);

  cudaFree(d_flags);
  cudaFree(d_numSelectedIndices);
  cudaFree(d_indices);
  cudaFree(d_selectedIndices);
  cudaFree(d_temp_storage);

  customFree<float>(DEVICE_MEMORY, d_aggregateRevenue);
  customFree<int>(DEVICE_MEMORY, d_lineitem_orderkey);
  customFree<float>(DEVICE_MEMORY, d_lineitem_discount);
  customFree<float>(DEVICE_MEMORY, d_lineitem_extended_price);
  customFree<uint32_t>(DEVICE_MEMORY, d_lineitem_shipdate);
  customFree<uint32_t>(DEVICE_MEMORY, d_customer_mktsegment);
  customFree<int>(DEVICE_MEMORY, d_customer_custkey);
  customFree<int>(DEVICE_MEMORY, d_orders_shippriority);
  customFree<uint32_t>(DEVICE_MEMORY, d_orders_orderdate);
  customFree<int>(DEVICE_MEMORY, d_orders_orderkey);
  customFree<int>(DEVICE_MEMORY, d_orders_custkey);
  CUDA_CHECK_ERROR_RETURN("Failure while deallocating gpu columns");

  return TablePtr();
}
}

//#endif
