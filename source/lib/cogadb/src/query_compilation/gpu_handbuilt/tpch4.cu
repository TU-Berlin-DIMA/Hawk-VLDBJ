#ifdef TPCH4QQQ

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
#include <query_compilation/gpu_utilities/compilation_hashtables/GPU_CHF32_4.cuh>

#include "tpch4.cuh"

// -- TPC-H Query 4
// select o_orderpriority, count(*) as order_count
// from orders
// where   o_orderdate >= date '1993-07-01'
//         and o_orderdate < date '1993-10-01'
//         and exists (
//                 select * from lineitem
//                 where l_orderkey = o_orderkey
//                 and l_commitdate < l_receiptdate
//         )
// group by o_orderpriority
// order by o_orderpriority

using namespace std;

namespace CoGaDB {

TablePtr tpch4();

bool tpch4_holistic_kernel(ClientPtr client) {
  tpch4();
  return true;
}

TablePtr tpch4() {
  CUDA_CHECK_ERROR_RETURN("Initial failure");

  TablePtr lineitem = getTablebyName("LINEITEM");
  assert(lineitem != NULL);
  assert(lineitem->isMaterialized());
  ColumnPtr col_lineitem_orderkey = lineitem->getColumnbyName("L_ORDERKEY");
  ColumnPtr col_lineitem_commitdate = lineitem->getColumnbyName("L_COMMITDATE");
  ColumnPtr col_lineitem_receiptdate =
      lineitem->getColumnbyName("L_RECEIPTDATE");
  size_t lineitem_num_elements = lineitem->getNumberofRows();

  TablePtr orders = getTablebyName("ORDERS");
  assert(orders != NULL);
  assert(orders->isMaterialized());
  ColumnPtr col_orders_orderpriority =
      orders->getColumnbyName("O_ORDERPRIORITY");
  ColumnPtr col_orders_orderdate = orders->getColumnbyName("O_ORDERDATE");
  ColumnPtr col_orders_orderkey = orders->getColumnbyName("O_ORDERKEY");
  size_t orders_num_elements = orders->getNumberofRows();

  boost::shared_ptr<Column<int>> typed_col_lineitem_orderkey =
      boost::dynamic_pointer_cast<Column<int>>(col_lineitem_orderkey);
  boost::shared_ptr<Column<uint32_t>> typed_col_lineitem_commitdate =
      boost::dynamic_pointer_cast<Column<uint32_t>>(col_lineitem_commitdate);
  boost::shared_ptr<Column<uint32_t>> typed_col_lineitem_receiptdate =
      boost::dynamic_pointer_cast<Column<uint32_t>>(col_lineitem_receiptdate);
  boost::shared_ptr<DictionaryCompressedColumn<std::string>>
      dict_orders_orderpriority =
          boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>>(
              col_orders_orderpriority);
  boost::shared_ptr<Column<uint32_t>> typed_col_orders_orderdate =
      boost::dynamic_pointer_cast<Column<uint32_t>>(col_orders_orderdate);
  boost::shared_ptr<Column<int>> typed_col_orders_orderkey =
      boost::dynamic_pointer_cast<Column<int>>(col_orders_orderkey);

  DictionaryCompressedColumn<std::string> &refdict = *dict_orders_orderpriority;
  AttributeType attType = refdict.getType();
  ColumnType colType = refdict.getColumnType();
  cout << "AtrributeType: " << util::getName(attType)
       << ", ColumnType: " << util::getName(colType) << endl;
  uint32_t *idData = refdict.getIdData();
  for (int i = 0; i < 20; i++) {
    uint32_t compressed = idData[i];
    cout << compressed << ", " << refdict.reverseLookup(compressed) << "."
         << endl;
  }

  int *d_lineitem_orderkey =
      customMalloc<int>(DEVICE_MEMORY, lineitem_num_elements);
  uint32_t *d_lineitem_commitdate =
      customMalloc<uint32_t>(DEVICE_MEMORY, lineitem_num_elements);
  uint32_t *d_lineitem_receiptdate =
      customMalloc<uint32_t>(DEVICE_MEMORY, lineitem_num_elements);
  uint32_t *d_orders_orderpriority =
      customMalloc<uint32_t>(DEVICE_MEMORY, orders_num_elements);
  uint32_t *d_orders_orderdate =
      customMalloc<uint32_t>(DEVICE_MEMORY, orders_num_elements);
  int *d_orders_orderkey =
      customMalloc<int>(DEVICE_MEMORY, orders_num_elements);
  CUDA_CHECK_ERROR_RETURN("Failure while allocating gpu columns");

  cudaMemcpy(d_lineitem_orderkey, typed_col_lineitem_orderkey->data(),
             lineitem_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lineitem_commitdate, typed_col_lineitem_commitdate->data(),
             lineitem_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_lineitem_receiptdate, typed_col_lineitem_receiptdate->data(),
             lineitem_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders_orderpriority, dict_orders_orderpriority->getIdData(),
             orders_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders_orderdate, typed_col_orders_orderdate->data(),
             orders_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_orders_orderkey, typed_col_orders_orderkey->data(),
             orders_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR_RETURN("Failure while transfering column data to gpu");

  int numGroups = 5;
  int numProbeBlocks = divUp(orders_num_elements, (size_t)kProbeBlockSize);
  int *d_groupCountPerBlock =
      customMalloc<int>(DEVICE_MEMORY, numProbeBlocks * numGroups);

  Timestamp begin_query = getTimestamp();

  double lineitem_selectivity = 0.5;
  GPU_CHF32_4::HashTable lineitemHashTable = GPU_CHF32_4::createHashTable(
      lineitem_num_elements * lineitem_selectivity, 1.25);
  dim3 gridDim =
      computeGridDim(lineitem_num_elements, kGridSize, kBuildBlockSize);

  lineitemBuildKernel<<<gridDim, kBuildBlockSize>>>(
      lineitemHashTable, d_lineitem_orderkey, d_lineitem_commitdate,
      d_lineitem_receiptdate, lineitem_num_elements);

  cudaDeviceSynchronize();
  Timestamp end_build_start_probe = getTimestamp();

  gridDim = computeGridDim(orders_num_elements, kGridSize, kProbeBlockSize);
  ordersProbeKernel<<<gridDim, kProbeBlockSize>>>(
      lineitemHashTable, d_orders_orderkey, d_orders_orderdate,
      d_orders_orderpriority, orders_num_elements, d_groupCountPerBlock);

  cudaDeviceSynchronize();

  Timestamp end_probe_start_reduce = getTimestamp();

  vector<int> h_groupCountPerBlock(numProbeBlocks * numGroups);
  cudaMemcpy(&h_groupCountPerBlock[0], d_groupCountPerBlock,
             sizeof(int) * numProbeBlocks * numGroups, cudaMemcpyDeviceToHost);

  int result[5] = {0, 0, 0, 0, 0};
  for (int i = 0; i < numProbeBlocks; i++) {
    result[0] += h_groupCountPerBlock[i * 5];
    result[1] += h_groupCountPerBlock[i * 5 + 1];
    result[2] += h_groupCountPerBlock[i * 5 + 2];
    result[3] += h_groupCountPerBlock[i * 5 + 3];
    result[4] += h_groupCountPerBlock[i * 5 + 4];
  }

  Timestamp end_query = getTimestamp();

  GPU_CHF32_4::printHashTable(lineitemHashTable, 50);

  double build_time =
      double(end_build_start_probe - begin_query) / (1000 * 1000);
  double probe_time =
      double(end_probe_start_reduce - end_build_start_probe) / (1000 * 1000);
  double reduce_time =
      double(end_query - end_probe_start_reduce) / (1000 * 1000);
  double query_time = double(end_query - begin_query) / (1000 * 1000);

  cout << "TPCH4 execution time " << query_time << " ms" << endl;
  cout << "build: " << build_time << " ms, "
       << "probe: " << probe_time << " ms, "
       << "reduce: " << reduce_time << " ms." << endl;

  for (int i = 0; i < 5; i++) cout << i << ", " << result[i] << endl;

  GPU_CHF32_4::freeHashTable(lineitemHashTable);

  customFree<int>(DEVICE_MEMORY, d_lineitem_orderkey);
  customFree<uint32_t>(DEVICE_MEMORY, d_lineitem_commitdate);
  customFree<uint32_t>(DEVICE_MEMORY, d_lineitem_receiptdate);
  customFree<uint32_t>(DEVICE_MEMORY, d_orders_orderpriority);
  customFree<uint32_t>(DEVICE_MEMORY, d_orders_orderdate);
  customFree<int>(DEVICE_MEMORY, d_orders_orderkey);

  cout << "tpch4: i was called" << endl;

  return TablePtr();
}
}

#endif
