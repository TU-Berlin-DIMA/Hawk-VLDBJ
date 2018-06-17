#ifdef SSB4HC

#include <query_compilation/gpu_handbuilt/queries.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <boost/make_shared.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <core/column.hpp>
#include <core/global_definitions.hpp>
#include <core/memory_allocator.hpp>
#include <iostream>
#include <limits>
#include <persistence/storage_manager.hpp>
#include <util/getname.hpp>
#include "core/variable_manager.hpp"
// compiled query specific"
//#include
//<query_compilation/gpu_utilities/compilation-hashtables/compilation_hashtables.cuh>
#include <query_compilation/gpu_utilities/util/cuda_griddim.h>
#include <query_compilation/gpu_utilities/util/divup.h>

#include <query_compilation/gpu_utilities/compilation-hashtables/GPU_CHT32_2.h>

// select d_year, c_nation, lo_revenue - lo_supplycost as profit from lineorder
// JOIN supplier ON (lo_suppkey = s_suppkey) JOIN customer ON (lo_custkey =
// c_custkey)
// JOIN part ON (lo_partkey = p_partkey) JOIN dates ON (lo_orderdate =
// d_datekey)
// where d_year between 1991 and 1995 and (lo_revenue - lo_supplycost) < 40000
// and c_region = 'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or
// p_mfgr = 'MFGR#2');

#define CUDA_CHECK_ERROR_RETURN(errorMessage)                             \
  {                                                                       \
    cudaError_t err = cudaGetLastError();                                 \
    if (cudaSuccess != err) {                                             \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",   \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err)); \
      CoGaDB::exit(EXIT_FAILURE);                                         \
    }                                                                     \
  }

using namespace std;

namespace CoGaDB {

TablePtr ssb4_q3();

bool ssb4_hand_compiled_holistic_kernel(ClientPtr client) {
  ssb4_q3();
  return true;
}

#include "ssb4.cuh"

TablePtr ssb4_q3() {
  cout << "SSB4 - variant - holistic kernel" << endl;

  CUDA_CHECK_ERROR_RETURN("Initial failure");

  TablePtr table_lineorder = getTablebyName("LINEORDER");
  assert(table_lineorder != NULL);
  assert(table_lineorder->isMaterialized());
  size_t lineorder_num_elements = table_lineorder->getNumberofRows();

  TablePtr table_dates = getTablebyName("DATES");
  assert(table_dates != NULL);
  assert(table_dates->isMaterialized());
  size_t dates_num_elements = table_dates->getNumberofRows();

  TablePtr table_supplier = getTablebyName("SUPPLIER");
  assert(table_supplier != NULL);
  assert(table_supplier->isMaterialized());
  size_t supplier_num_elements = table_supplier->getNumberofRows();

  TablePtr table_part = getTablebyName("PART");
  assert(table_part != NULL);
  assert(table_part->isMaterialized());
  size_t part_num_elements = table_part->getNumberofRows();

  TablePtr table_customer = getTablebyName("CUSTOMER");
  assert(table_customer != NULL);
  assert(table_customer->isMaterialized());
  size_t customer_num_elements = table_customer->getNumberofRows();

  ColumnPtr col_lo_custkey = table_lineorder->getColumnbyName("LO_CUSTKEY");
  ColumnPtr col_lo_suppkey = table_lineorder->getColumnbyName("LO_SUPPKEY");
  ColumnPtr col_lo_partkey = table_lineorder->getColumnbyName("LO_PARTKEY");
  ColumnPtr col_lo_orderdate = table_lineorder->getColumnbyName("LO_ORDERDATE");
  ColumnPtr col_lo_revenue = table_lineorder->getColumnbyName("LO_REVENUE");
  ColumnPtr col_lo_supplycost =
      table_lineorder->getColumnbyName("LO_SUPPLYCOST");
  ColumnPtr col_c_custkey = table_customer->getColumnbyName("C_CUSTKEY");
  ColumnPtr col_c_region = table_customer->getColumnbyName("C_REGION");
  ColumnPtr col_c_nation = table_customer->getColumnbyName("C_NATION");
  ColumnPtr col_s_region = table_supplier->getColumnbyName("S_REGION");
  ColumnPtr col_s_suppkey = table_supplier->getColumnbyName("S_SUPPKEY");
  ColumnPtr col_p_mfgr = table_part->getColumnbyName("P_MFGR");
  ColumnPtr col_p_partkey = table_part->getColumnbyName("P_PARTKEY");
  ColumnPtr col_d_datekey = table_dates->getColumnbyName("D_DATEKEY");
  ColumnPtr col_d_year = table_dates->getColumnbyName("D_YEAR");

  cout << (col_lo_custkey->type()).name() << endl;
  cout << (col_lo_suppkey->type()).name() << endl;
  cout << (col_lo_partkey->type()).name() << endl;
  cout << (col_lo_orderdate->type()).name() << endl;
  cout << (col_lo_revenue->type()).name() << endl;
  cout << (col_lo_supplycost->type()).name() << endl;
  cout << (col_c_custkey->type()).name() << endl;
  cout << (col_c_region->type()).name() << endl;
  cout << (col_c_nation->type()).name() << endl;
  cout << (col_s_region->type()).name() << endl;
  cout << (col_s_suppkey->type()).name() << endl;
  cout << (col_p_mfgr->type()).name() << endl;
  cout << (col_p_partkey->type()).name() << endl;
  cout << (col_d_datekey->type()).name() << endl;
  cout << (col_d_year->type()).name() << endl;

  boost::shared_ptr<Column<int>> typed_col_lo_custkey =
      boost::dynamic_pointer_cast<Column<int>>(col_lo_custkey);
  typed_col_lo_custkey->data();

  boost::shared_ptr<Column<int>> typed_col_lo_suppkey =
      boost::dynamic_pointer_cast<Column<int>>(col_lo_suppkey);
  typed_col_lo_suppkey->data();

  boost::shared_ptr<Column<int>> typed_col_lo_partkey =
      boost::dynamic_pointer_cast<Column<int>>(col_lo_partkey);
  typed_col_lo_partkey->data();

  boost::shared_ptr<Column<int>> typed_col_lo_orderdate =
      boost::dynamic_pointer_cast<Column<int>>(col_lo_orderdate);
  typed_col_lo_orderdate->data();

  boost::shared_ptr<Column<float>> typed_col_lo_revenue =
      boost::dynamic_pointer_cast<Column<float>>(col_lo_revenue);
  typed_col_lo_revenue->data();

  boost::shared_ptr<Column<float>> typed_col_lo_supplycost =
      boost::dynamic_pointer_cast<Column<float>>(col_lo_supplycost);
  typed_col_lo_supplycost->data();

  boost::shared_ptr<Column<int>> typed_col_c_custkey =
      boost::dynamic_pointer_cast<Column<int>>(col_c_custkey);
  typed_col_c_custkey->data();

  boost::shared_ptr<DictionaryCompressedColumn<std::string>>
      dict_customer_region =
          boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>>(
              col_c_region);

  boost::shared_ptr<DictionaryCompressedColumn<std::string>>
      dict_customer_nation =
          boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>>(
              col_c_nation);

  boost::shared_ptr<DictionaryCompressedColumn<std::string>>
      dict_supplier_region =
          boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>>(
              col_s_region);

  boost::shared_ptr<Column<int>> typed_col_s_suppkey =
      boost::dynamic_pointer_cast<Column<int>>(col_s_suppkey);
  typed_col_s_suppkey->data();

  boost::shared_ptr<DictionaryCompressedColumn<std::string>> dict_part_mfgr =
      boost::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>>(
          col_p_mfgr);

  boost::shared_ptr<Column<int>> typed_col_p_partkey =
      boost::dynamic_pointer_cast<Column<int>>(col_p_partkey);
  typed_col_p_partkey->data();

  boost::shared_ptr<Column<int>> typed_col_d_datekey =
      boost::dynamic_pointer_cast<Column<int>>(col_d_datekey);
  typed_col_d_datekey->data();

  boost::shared_ptr<Column<int>> typed_col_d_year =
      boost::dynamic_pointer_cast<Column<int>>(col_d_year);
  typed_col_d_year->data();

  // DictionaryCompressedColumn<std::string>& refdict = *dict_customer_region;
  // AttributeType attType = refdict.getType();
  // ColumnType colType  = refdict.getColumnType();
  // cout << "AtrributeType: " << util::getName(attType) << ", ColumnType: " <<
  // util::getName(colType) << endl;
  // uint32_t* h_supplier_region = refdict.getIdData();
  // for(int i=0; i < 100; i++) {
  //     uint32_t compressed = h_supplier_region[i];
  //     cout << compressed << ", " << refdict.reverseLookup(compressed) << "."
  //     << endl;
  // }

  // DictionaryCompressedColumn<std::string>& refdict = *dict_part_mfgr;
  // AttributeType attType = refdict.getType();
  // ColumnType colType  = refdict.getColumnType();
  // cout << "AtrributeType: " << util::getName(attType) << ", ColumnType: " <<
  // util::getName(colType) << endl;
  // uint32_t* h_supplier_region = refdict.getIdData();
  // for(int i=0; i < 20; i++) {
  //     uint32_t compressed = h_supplier_region[i];
  //     cout << compressed << ", " << refdict.reverseLookup(compressed) << "."
  //     << endl;
  // }

  DictionaryCompressedColumn<std::string>& refdict = *dict_customer_region;
  AttributeType attType = refdict.getType();
  ColumnType colType = refdict.getColumnType();
  cout << "AtrributeType: " << util::getName(attType)
       << ", ColumnType: " << util::getName(colType) << endl;
  uint32_t* h_supplier_region = refdict.getIdData();
  for (int i = 0; i < 20; i++) {
    uint32_t compressed = h_supplier_region[i];
    cout << compressed << ", " << refdict.reverseLookup(compressed) << "."
         << endl;
  }

  int* p_col_lo_custkey;
  int* p_col_lo_suppkey;
  int* p_col_lo_partkey;
  int* p_col_lo_orderdate;
  float* p_col_lo_revenue;
  float* p_col_lo_supplycost;
  int* p_col_c_custkey;
  uint32_t* p_col_c_region;
  uint32_t* p_col_c_nation;
  uint32_t* p_col_s_region;
  int* p_col_s_suppkey;
  uint32_t* p_col_p_mfgr;
  int* p_col_p_partkey;
  int* p_col_d_datekey;
  int* p_col_d_year;

  p_col_lo_custkey = customMalloc<int>(DEVICE_MEMORY, lineorder_num_elements);
  p_col_lo_suppkey = customMalloc<int>(DEVICE_MEMORY, lineorder_num_elements);
  p_col_lo_partkey = customMalloc<int>(DEVICE_MEMORY, lineorder_num_elements);
  p_col_lo_orderdate = customMalloc<int>(DEVICE_MEMORY, lineorder_num_elements);
  p_col_lo_revenue = customMalloc<float>(DEVICE_MEMORY, lineorder_num_elements);
  p_col_lo_supplycost =
      customMalloc<float>(DEVICE_MEMORY, lineorder_num_elements);
  p_col_c_custkey = customMalloc<int>(DEVICE_MEMORY, customer_num_elements);
  p_col_c_region = customMalloc<uint32_t>(DEVICE_MEMORY, customer_num_elements);
  p_col_c_nation = customMalloc<uint32_t>(DEVICE_MEMORY, customer_num_elements);
  p_col_s_region = customMalloc<uint32_t>(DEVICE_MEMORY, supplier_num_elements);
  p_col_s_suppkey = customMalloc<int>(DEVICE_MEMORY, supplier_num_elements);
  p_col_p_mfgr = customMalloc<uint32_t>(DEVICE_MEMORY, part_num_elements);
  p_col_p_partkey = customMalloc<int>(DEVICE_MEMORY, part_num_elements);
  p_col_d_datekey = customMalloc<int>(DEVICE_MEMORY, dates_num_elements);
  p_col_d_year = customMalloc<int>(DEVICE_MEMORY, dates_num_elements);

  cudaMemcpy(p_col_lo_custkey, typed_col_lo_custkey->data(),
             lineorder_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_lo_suppkey, typed_col_lo_suppkey->data(),
             lineorder_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_lo_partkey, typed_col_lo_partkey->data(),
             lineorder_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_lo_orderdate, typed_col_lo_orderdate->data(),
             lineorder_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_lo_revenue, typed_col_lo_revenue->data(),
             lineorder_num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_lo_supplycost, typed_col_lo_supplycost->data(),
             lineorder_num_elements * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_c_custkey, typed_col_c_custkey->data(),
             customer_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_c_region, dict_customer_region->getIdData(),
             customer_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_c_nation, dict_customer_nation->getIdData(),
             customer_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_s_region, dict_supplier_region->getIdData(),
             supplier_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_s_suppkey, typed_col_s_suppkey->data(),
             supplier_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_p_mfgr, dict_part_mfgr->getIdData(),
             part_num_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_p_partkey, typed_col_p_partkey->data(),
             part_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_d_datekey, typed_col_d_datekey->data(),
             dates_num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_col_d_year, typed_col_d_year->data(),
             dates_num_elements * sizeof(int), cudaMemcpyHostToDevice);

  int maxResultsPerBlock = kProbeBlockSize / 16;
  int* result_d_year;
  uint32_t* result_c_nation;
  float* result_profit;

  int numBlocks = divUp(lineorder_num_elements, (size_t)kProbeBlockSize);
  int* numResultsEachBlock;

  cudaMallocManaged((void**)&result_d_year,
                    sizeof(int) * numBlocks * maxResultsPerBlock);
  cudaMallocManaged((void**)&result_c_nation,
                    sizeof(uint32_t) * numBlocks * maxResultsPerBlock);
  cudaMallocManaged((void**)&result_profit,
                    sizeof(float) * numBlocks * maxResultsPerBlock);
  cudaMallocManaged((void**)&numResultsEachBlock, sizeof(int) * numBlocks);

  for (int i = 0; i < numBlocks; i++) {
    numResultsEachBlock[i] = 0;
  }

  cout << "Holding memory:" << endl;
  ;
  size_t avail;
  size_t total;
  cudaMemGetInfo(&avail, &total);
  size_t used = (total - avail) / (1000 * 1000);
  cout << "Device memory used: " << used << "MB" << endl;

  // --------------- Query processing -------------------
  // select d_year, c_nation, lo_revenue - lo_supplycost as profit from
  // lineorder
  // JOIN supplier ON (lo_suppkey = s_suppkey) JOIN customer ON (lo_custkey =
  // c_custkey)
  // JOIN part ON (lo_partkey = p_partkey) JOIN dates ON (lo_orderdate =
  // d_datekey)
  // where d_year between 1991 and 1995 and (lo_revenue - lo_supplycost) < 40000
  // and c_region = 'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or
  // p_mfgr = 'MFGR#2');

  Timestamp begin_query = getTimestamp();

  // ----------------- invocation: build kernel1  ------------------------
  // build hashtable for supplier
  // JOIN supplier ON (lo_suppkey = s_suppkey)
  // s_region = 'AMERICA'
  size_t supplier_selected_num_elements = supplier_num_elements * 0.25 * 2;
  GPU_CHT32_2::HashTable supplierHashTable =
      GPU_CHT32_2::createHashTable(supplier_selected_num_elements, 1.25);
  dim3 gridDim =
      computeGridDim(supplier_num_elements, supplierHashTable.gridSize,
                     supplierHashTable.blockSize);
  supplierBuildKernel<<<gridDim, supplierHashTable.blockSize>>>(
      supplierHashTable, p_col_s_region, p_col_s_suppkey,
      supplier_num_elements);

  // ----------------- invocation: build kernel2  ------------------------
  // build hash table for dates
  // JOIN dates ON (lo_orderdate = d_datekey)
  // where d_year between 1991 and 1995
  size_t dates_selected_num_elements = dates_num_elements * 0.6 * 2;
  GPU_CHT32_2::HashTable datesHashTable =
      GPU_CHT32_2::createHashTable(dates_selected_num_elements, 1.25);
  gridDim = computeGridDim(dates_num_elements, datesHashTable.gridSize,
                           datesHashTable.blockSize);
  datesBuildKernel<<<gridDim, datesHashTable.blockSize>>>(
      datesHashTable, p_col_d_year, p_col_d_datekey, dates_num_elements);

  // ----------------- invocation: build kernel3  ------------------------
  // build hash table for customer
  // JOIN customer ON (lo_custkey = c_custkey)
  // c_region = 'AMERICA'
  size_t customer_selected_num_elements = customer_num_elements * 0.2 * 2;
  GPU_CHT32_2::HashTable customerHashTable =
      GPU_CHT32_2::createHashTable(customer_selected_num_elements, 1.25);
  gridDim = computeGridDim(customer_num_elements, customerHashTable.gridSize,
                           customerHashTable.blockSize);
  customerBuildKernel<<<gridDim, customerHashTable.blockSize>>>(
      customerHashTable, p_col_c_region, p_col_c_custkey,
      customer_num_elements);

  // ----------------- invocation: build kernel4  ------------------------
  // build hash table for customer
  // JOIN part ON (lo_partkey = p_partkey)
  // (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2');
  size_t part_selected_num_elements = part_num_elements * 0.5 * 2;
  GPU_CHT32_2::HashTable partHashTable =
      GPU_CHT32_2::createHashTable(part_selected_num_elements, 1.25);
  gridDim = computeGridDim(part_num_elements, partHashTable.gridSize,
                           partHashTable.blockSize);
  partBuildKernel<<<gridDim, partHashTable.blockSize>>>(
      partHashTable, p_col_p_mfgr, p_col_p_partkey, part_num_elements);

  cudaDeviceSynchronize();

  Timestamp end_build_start_probe = getTimestamp();

  // select d_year, c_nation, lo_revenue - lo_supplycost as profit from
  // lineorder
  // JOIN supplier ON (lo_suppkey = s_suppkey) JOIN customer ON (lo_custkey =
  // c_custkey)
  // JOIN part ON (lo_partkey = p_partkey) JOIN dates ON (lo_orderdate =
  // d_datekey)
  // where d_year between 1991 and 1995 and (lo_revenue - lo_supplycost) < 40000
  // and c_region = 'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or
  // p_mfgr = 'MFGR#2');

  gridDim = computeGridDim(lineorder_num_elements, 16384, kProbeBlockSize);

  probeKernel<<<gridDim, kProbeBlockSize>>>(
      supplierHashTable, datesHashTable, customerHashTable, partHashTable,
      p_col_lo_suppkey, p_col_lo_orderdate, p_col_lo_custkey, p_col_lo_partkey,
      p_col_lo_revenue, p_col_lo_supplycost, lineorder_num_elements,
      p_col_d_year, p_col_c_nation, result_d_year, result_c_nation,
      result_profit, maxResultsPerBlock, numResultsEachBlock);

  cudaDeviceSynchronize();

  Timestamp end_query = getTimestamp();

  double build_time =
      double(end_build_start_probe - begin_query) / (1000 * 1000);
  double probe_time = double(end_query - end_build_start_probe) / (1000 * 1000);
  double query_time = build_time + probe_time;

  int max = 0;
  int min = kProbeBlockSize;
  int result = 0;
  for (int i = 0; i < numBlocks; i++) {
    result += numResultsEachBlock[i];
    if (numResultsEachBlock[i] > max) max = numResultsEachBlock[i];
    if (numResultsEachBlock[i] < min) min = numResultsEachBlock[i];

    for (int j = 0; j < numResultsEachBlock[i]; j++) {
      size_t index = maxResultsPerBlock * i + j;
      cout << "d_year: " << result_d_year[index]
           << ", c_nation: " << result_c_nation[index]
           << ", profit: " << result_profit[index] << endl;
    }
  }

  cout << "SSB4 (modified) execution time " << query_time << " ms" << endl;
  cout << "build: " << build_time << " ms, probe: " << probe_time << " ms"
       << endl;
  cout << "Query result has " << result << " elements" << endl;
  cout << "numBlocks " << numBlocks << endl;
  cout << "min results per block: " << min << endl;
  cout << "max results per block: " << max << endl;

  GPU_CHT32_2::printHashTable(customerHashTable);
  GPU_CHT32_2::printHashTable(supplierHashTable);
  GPU_CHT32_2::printHashTable(datesHashTable);
  GPU_CHT32_2::printHashTable(partHashTable);

  GPU_CHT32_2::freeHashTable(customerHashTable);
  GPU_CHT32_2::freeHashTable(supplierHashTable);
  GPU_CHT32_2::freeHashTable(datesHashTable);
  GPU_CHT32_2::freeHashTable(partHashTable);

  cudaFree(result_d_year);
  cudaFree(result_c_nation);
  cudaFree(result_profit);

  customFree<int>(DEVICE_MEMORY, p_col_lo_custkey);
  customFree<int>(DEVICE_MEMORY, p_col_lo_suppkey);
  customFree<int>(DEVICE_MEMORY, p_col_lo_partkey);
  customFree<int>(DEVICE_MEMORY, p_col_lo_orderdate);
  customFree<float>(DEVICE_MEMORY, p_col_lo_revenue);
  customFree<float>(DEVICE_MEMORY, p_col_lo_supplycost);
  customFree<int>(DEVICE_MEMORY, p_col_c_custkey);
  customFree<uint32_t>(DEVICE_MEMORY, p_col_c_region);
  customFree<uint32_t>(DEVICE_MEMORY, p_col_c_nation);
  customFree<uint32_t>(DEVICE_MEMORY, p_col_s_region);
  customFree<int>(DEVICE_MEMORY, p_col_s_suppkey);
  customFree<uint32_t>(DEVICE_MEMORY, p_col_p_mfgr);
  customFree<int>(DEVICE_MEMORY, p_col_p_partkey);
  customFree<int>(DEVICE_MEMORY, p_col_d_datekey);
  customFree<int>(DEVICE_MEMORY, p_col_d_year);

  cout << "After freeing memory:" << endl;
  ;
  cudaMemGetInfo(&avail, &total);
  used = (total - avail) / (1000 * 1000);
  cout << "Device memory used: " << used << "MB" << endl;

  CUDA_CHECK_ERROR_RETURN("Initial failure");

  return TablePtr();
}
}

#endif
