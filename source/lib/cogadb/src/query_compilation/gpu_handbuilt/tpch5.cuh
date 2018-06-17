#include <cub/cub.cuh>


const int kGridSize = 16384;
const int kBuildBlockSize = 64;
const int kProbeBlockSize = 128;
//const int kReduceBlockSize = 256;




__global__ void regionBuildKernel
                      ( GPU_CHT32_2::HashTable  regionHashTable,
                        int*                    region_regionkey,
                        uint32_t*               region_name,
                        size_t                  num_elements ) {

    uint32_t thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    bool selected = true;
    selected &= ( region_name[thread_index] == 2 );

    if(selected)
        GPU_CHT32_2::insertEntry(regionHashTable, (uint32_t)region_regionkey[thread_index], thread_index);
}

__global__ void nationBuildKernel
                      ( GPU_CHT32_2::HashTable  nationHashTable,
                        int*                    nation_nationkey,
                        size_t                  num_elements) {

    uint32_t thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    GPU_CHT32_2::insertEntry(nationHashTable, (uint32_t)nation_nationkey[thread_index], thread_index);

}

__global__ void supplierBuildKernel
                        ( GPU_CHT32_2::HashTable  supplierHashTable,
                          int*                    supplier_suppkey,
                          size_t                  num_elements) {

    uint32_t thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    GPU_CHT32_2::insertEntry(supplierHashTable, (uint32_t)supplier_suppkey[thread_index], thread_index);


}

__global__ void customerBuildKernel
                        ( GPU_CHT32_2::HashTable  customerHashTable,
                          int*                    customer_custkey,
                          size_t                  num_elements) {

    uint32_t thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    GPU_CHT32_2::insertEntry(customerHashTable, (uint32_t)customer_custkey[thread_index], thread_index);

}

__global__ void ordersBuildKernel
                      ( GPU_CHT32_2::HashTable  ordersHashTable,
                        int*                    orders_orderkey,
                        uint32_t*               orders_orderdate,
                        size_t                  num_elements) {

    uint32_t thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    bool selected = true;
    selected &= ( orders_orderdate[thread_index] >= 19940101 );
    selected &= ( orders_orderdate[thread_index] <  19950101 );

    if(selected)
        GPU_CHT32_2::insertEntry(ordersHashTable, (uint32_t)orders_orderkey[thread_index], thread_index);
}


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

__global__ void lineitemProbeKernel(GPU_CHT32_2::HashTable  regionHashTable,
                                    GPU_CHT32_2::HashTable  nationHashTable,
                                    GPU_CHT32_2::HashTable  supplierHashTable,
                                    GPU_CHT32_2::HashTable  customerHashTable,
                                    GPU_CHT32_2::HashTable  ordersHashTable,
                                    int*                    lineitem_suppkey,
                                    int*                    lineitem_orderkey,
                                    int*                    supplier_nationkey,
                                    int*                    orders_custkey,
                                    int*                    nation_regionkey,
                                    int*                    customer_nationkey,
                                    size_t                  lineitem_num_elements,
                                    float*                  lineitem_extended_price,
                                    float*                  lineitem_discount,
                                    uint32_t*               nation_name,
                                    int                     numGroups,
                                    float*                  groupAggregatesPerBlock,
                                    int*                    groupCountsPerBlock ) {

    uint32_t block_index = blockIdx.y * gridDim.x + blockIdx.x;

    uint32_t thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= lineitem_num_elements) return;

    __shared__ float aggregates[25];
    __shared__ int counts[25];
    if(threadIdx.x < 25) {
        aggregates[threadIdx.x] = 0.0;
        counts[threadIdx.x] = 0;
    }
    __syncthreads();

    uint32_t probe1, probe2, probe3, probe4, probe5, probe6, probe7;

    bool selected = true;

    //Probe lineitem -> supplier -> nation -> region
    if(selected) {
        probe1 = GPU_CHT32_2::probeKey(supplierHashTable, lineitem_suppkey[thread_index]);
        selected &= (probe1 != GPU_CHT32_2::kEmpty);
    }

    if(selected) {
        probe2 = GPU_CHT32_2::probeKey(nationHashTable, supplier_nationkey[probe1]);
        selected &= (probe2 != GPU_CHT32_2::kEmpty);
    }

    if(selected) {
        probe3 = GPU_CHT32_2::probeKey(regionHashTable, nation_regionkey[probe2]);
        selected &= (probe3 != GPU_CHT32_2::kEmpty);
    }

    //Probe lineitem -> order -> customer -> nation -> region
    if(selected) {
        probe4 = GPU_CHT32_2::probeKey(ordersHashTable, lineitem_orderkey[thread_index]);
        selected &= (probe4 != GPU_CHT32_2::kEmpty);
    }

    if(selected) {
        probe5 = GPU_CHT32_2::probeKey(customerHashTable, orders_custkey[probe4]);
        selected &= (probe5 != GPU_CHT32_2::kEmpty);
    }

    if(selected) {
        probe6 = GPU_CHT32_2::probeKey(nationHashTable, customer_nationkey[probe5]);
        selected &= (probe6 != GPU_CHT32_2::kEmpty);
    }

    if(selected) {
        probe7 = GPU_CHT32_2::probeKey(regionHashTable, nation_regionkey[probe6]);
        selected &= (probe7 != GPU_CHT32_2::kEmpty);
    }

    if(selected) {
        selected &= (probe6 != probe2);
    }

    if(selected) {
        float revenue = lineitem_extended_price[thread_index] * (1.0 - lineitem_discount[thread_index]);
        atomicAdd(&aggregates[nation_name[probe2]], revenue);
        atomicAdd(&counts[nation_name[probe2]], 1);
    }
    __syncthreads();

    if(threadIdx.x < 25) {
        groupAggregatesPerBlock[block_index * 25 + threadIdx.x] = aggregates[threadIdx.x];
        groupCountsPerBlock[block_index * 25 + threadIdx.x] = counts[threadIdx.x];
    }
}
