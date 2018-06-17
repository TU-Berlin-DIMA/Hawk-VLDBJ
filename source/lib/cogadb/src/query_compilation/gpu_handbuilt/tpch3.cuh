#include <cub/cub.cuh>


const int kGridSize = 16384;
const int kBuildBlockSize = 64;
const int kProbeBlockSize = 128;
const int kReduceBlockSize = 256;


__global__ void customerBuildKernel(GPU_CHT32_2::HashTable ht,
                                   int* custkey,
                                   uint32_t* mktsegment,
                                   size_t num_elements) {

    uint32_t thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    bool selected = true;
    selected &= ( mktsegment[thread_index] == 0 );

    if(selected)
        GPU_CHT32_2::insertEntry(ht, (uint32_t)custkey[thread_index], thread_index);
}




__global__ void ordersBuildKernel(GPU_CHT32_2::HashTable ht,
                               int* orderkey,
                               uint32_t* orderdate,
                               size_t num_elements) {

     uint32_t thread_index = threadIdx.x +
                               blockIdx.x * blockDim.x +
                               blockIdx.y * blockDim.x * gridDim.x;

     if(thread_index >= num_elements) return;

     bool selected = true;
     selected &= ( orderdate[thread_index] < 19950315 );

     if(selected)
         GPU_CHT32_2::insertEntry(ht, (uint32_t)orderkey[thread_index], thread_index);
}


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


__global__ void lineorderProbeKernel
                             (GPU_CHT32_2::HashTable customerHashTable,
                              GPU_CHT32_2::HashTable ordersHashTable,
                              int*                   lineitem_orderkey,
                              int*                   orders_custkey,
                              float*                 lineitem_extended_price,
                              float*                 lineitem_discount,
                              uint32_t*              lineitem_shipdate,
                              size_t                 lineitem_num_elements,
                              float*                 aggregateRevenue) {

    uint32_t thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= lineitem_num_elements) return;

    uint32_t probe1, probe2;
    bool selected = true;

    selected &= (lineitem_shipdate[thread_index] > 19950315);

    if(selected) {
       probe1 = GPU_CHT32_2::probeKey(ordersHashTable, lineitem_orderkey[thread_index]);
       selected &= (probe1 != GPU_CHT32_2::kEmpty);
    }

    if(selected) {
       probe2 = GPU_CHT32_2::probeKey(customerHashTable, orders_custkey[probe1]);
       selected &= (probe2 != GPU_CHT32_2::kEmpty);
    }

    if(selected) {
      float revenue = lineitem_extended_price[thread_index] * (1.0 - lineitem_discount[thread_index]);
      atomicAdd(&aggregateRevenue[probe1], revenue);
    }
}




__global__ void flagAggregate(float *aggregateRevenue, size_t cardinality, char* d_flags) {

  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  while(tid < cardinality) {

      if(aggregateRevenue[tid] > 0.0) d_flags[tid] = 1;
      else d_flags[tid] = 0;
      tid += blockDim.x * gridDim.x;
  }

}
