#include <cub/cub.cuh>


__global__ void supplierBuildKernel(GPU_CHT32_2::HashTable ht,
                                   uint32_t* region,
                                   int* suppkey,
                                   size_t num_elements) {

    HtUnsigned thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    bool selected = false;

    if(region[thread_index] == 2)
        selected = true;

    if(selected)
        GPU_CHT32_2::insertEntry(ht, (HtUnsigned)suppkey[thread_index], thread_index);
}


__global__ void datesBuildKernel(GPU_CHT32_2::HashTable ht,
                               int* year,
                               int* datekey,
                               size_t num_elements) {

    HtUnsigned thread_index = threadIdx.x +
                              blockIdx.x * blockDim.x +
                              blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    bool selected = false;

    if(year[thread_index] >= 1991 && year[thread_index] <= 1995)
        selected = true;

    if(selected)
        GPU_CHT32_2::insertEntry(ht, datekey[thread_index], thread_index);
}


__global__ void customerBuildKernel(GPU_CHT32_2::HashTable ht,
                                   uint32_t* region,
                                   int* customerkey,
                                   size_t num_elements) {

    HtUnsigned thread_index = threadIdx.x +
                             blockIdx.x * blockDim.x +
                             blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    bool selected = false;

    if(region[thread_index] == 2)
       selected = true;

    if(selected)
       GPU_CHT32_2::insertEntry(ht, (HtUnsigned)customerkey[thread_index], thread_index);
}


__global__ void partBuildKernel(GPU_CHT32_2::HashTable ht,
                                   uint32_t* mfgr,
                                   int* partkey,
                                   size_t num_elements) {

    HtUnsigned thread_index = threadIdx.x +
                             blockIdx.x * blockDim.x +
                             blockIdx.y * blockDim.x * gridDim.x;

    if(thread_index >= num_elements) return;

    bool selected = false;

    if(mfgr[thread_index] == 0)
       selected = true;
    if(mfgr[thread_index] == 3)
       selected = true;

    if(selected)
       GPU_CHT32_2::insertEntry(ht, (HtUnsigned)partkey[thread_index], thread_index);
}


//select d_year, c_nation, lo_revenue - lo_supplycost as profit from lineorder
//JOIN supplier ON (lo_suppkey = s_suppkey) JOIN customer ON (lo_custkey = c_custkey)
//JOIN part ON (lo_partkey = p_partkey) JOIN dates ON (lo_orderdate = d_datekey)
//where d_year between 1991 and 1995 and (lo_revenue - lo_supplycost) < 40000
//and c_region = 'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2');

const int kProbeBlockSize = 64;

__global__ void probeKernel(GPU_CHT32_2::HashTable supplierHashTable,
                            GPU_CHT32_2::HashTable datesHashTable,
                            GPU_CHT32_2::HashTable customerHashTable,
                            GPU_CHT32_2::HashTable partHashTable,
                            int*         lo_suppkey,
                            int*         lo_orderdate,
                            int*         lo_custkey,
                            int*         lo_partkey,
                            float*       lo_revenue,
                            float*       lo_supplycost,
                            size_t       lineorder_num_elements,
                            int*         d_year,
                            uint32_t*    c_nation,
                            int*         result_d_year,
                            uint32_t*    result_c_nation,
                            float*       result_profit,
                            int          maxResultsPerBlock,
                            int*         numResultsEachBlock) {


    typedef cub::BlockScan<int, kProbeBlockSize> BlockScan;

    __shared__ typename BlockScan::TempStorage temp_storage;

    HtUnsigned block_index = blockIdx.y * gridDim.x + blockIdx.x;

    HtUnsigned thread_index = threadIdx.x +
                             blockIdx.x * blockDim.x +
                             blockIdx.y * blockDim.x * gridDim.x;

    bool selected = false;
    float profit = 0.0;
    HtUnsigned probe1, probe2, probe3, probe4;

    if(thread_index < lineorder_num_elements) {

        selected = true;
        if(selected) {
            probe3 = GPU_CHT32_2::probeKey(customerHashTable, lo_custkey[thread_index]);
            if(probe3 == GPU_CHT32_2::kEmpty) selected = false;
        }
        if(selected) {
            probe1 = GPU_CHT32_2::probeKey(supplierHashTable, lo_suppkey[thread_index]);
            if(probe1 == GPU_CHT32_2::kEmpty) selected = false;
        }
        if(selected) {
            probe2 = GPU_CHT32_2::probeKey(datesHashTable, lo_orderdate[thread_index]);
            if(probe2 == GPU_CHT32_2::kEmpty) selected = false;
        }
        if(selected) {
            probe4 = GPU_CHT32_2::probeKey(partHashTable, lo_partkey[thread_index]);
            if(probe4 == GPU_CHT32_2::kEmpty) selected = false;
        }

        if(selected) {
            profit = lo_revenue[thread_index] - lo_supplycost[thread_index];
            if(profit > 40000.0) {
              selected = false;
            }
        }
    }

    int flag = 0;
    int block_scan = 0;
    if(selected) flag = 1;

    BlockScan(temp_storage).ExclusiveSum(flag, block_scan);

    if(selected) {
        size_t outIndex = block_index * maxResultsPerBlock + block_scan;
        result_d_year[outIndex] = d_year[probe2];
        result_c_nation[outIndex] = c_nation[probe3];
        result_profit[outIndex] = profit;

    }


    if(threadIdx.x == blockDim.x - 1) {
        numResultsEachBlock[block_index] = flag + block_scan;
    }



}
