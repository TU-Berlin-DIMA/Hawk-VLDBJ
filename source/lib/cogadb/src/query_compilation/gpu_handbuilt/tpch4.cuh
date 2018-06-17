#include <cub/cub.cuh>


const int kGridSize = 16384;

const int kBuildBlockSize = 64;

__global__ void  lineitemBuildKernel(GPU_CHF32_4::HashTable ht,
                                     int* l_orderkey,
                                     uint32_t* l_commitdate,
                                     uint32_t* l_receiptdate,
                                     size_t lineitem_num_elements) {

    HtUnsigned thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= lineitem_num_elements) return;

    bool selected = (l_commitdate < l_receiptdate);

    if(selected)
        GPU_CHF32_4::insertEntry(ht, (HtUnsigned)l_orderkey[thread_index]);
}

const int kProbeBlockSize = 128;


__global__ void ordersProbeKernel(GPU_CHF32_4::HashTable ht,
                                  int*                   o_orderkey,
                                  uint32_t*              o_orderdate,
                                  uint32_t*              o_orderpriority,
                                  size_t                 orders_num_elements,
                                  int*                   groupCountPerBlock) {

    HtUnsigned block_index = blockIdx.y * gridDim.x + blockIdx.x;

    HtUnsigned thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    __shared__ int groupCounts[5];
    if(threadIdx.x < 5) {
        groupCounts[threadIdx.x] = 0;
    }
    __syncthreads();

    if(thread_index >= orders_num_elements)
        return;

    bool selected = true;
    selected &= (o_orderdate[thread_index]  >=  19930701);
    selected &= (o_orderdate[thread_index]  <   19931001);

    HtUnsigned probe1 = GPU_CHF32_4::probeKey(ht, o_orderkey[thread_index]);
    selected &= (probe1 != GPU_CHF32_4::kEmpty);

    if(selected) {
        atomicAdd(&groupCounts[o_orderpriority[thread_index]], 1);
    }
    __syncthreads();

    if(threadIdx.x < 5) {
        groupCountPerBlock[block_index * 5 + threadIdx.x] = groupCounts[threadIdx.x];
    }

}
