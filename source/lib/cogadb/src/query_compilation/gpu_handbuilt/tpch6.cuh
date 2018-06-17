

#include <cub/cub.cuh>


const unsigned int blockSize = 256;
const unsigned int kGridSize  = 16384;

    dim3 computeGridDim(size_t n) {
        // Round up in order to make sure all items are hashed in.
        dim3 grid( (n + blockSize-1) / blockSize );
        if (grid.x > kGridSize) {
            grid.y = (grid.x + kGridSize - 1) / kGridSize;
            grid.x = kGridSize;
        }
        if(grid.x == 0) grid.x = 1;
        return grid;
    }

    __global__ void kernel_select_tpch_q6(
                    const size_t num_elements,
                    const float    * l_extended_price,
                    const int      * l_quantity,
                    const float    * l_discount,
                    const uint32_t * l_shipdate,
                    char           * d_flags) {

        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        bool selected;

        while(tid < num_elements) {

//            if(tid % 10000000 == 0)
//                printf("shipdate %u\n", l_shipdate[tid]);
//
            if(tid == num_elements-1)
                printf("num_elements %u\n", num_elements);

            selected = true;
            selected    &=     (l_shipdate[tid]  >=  19940101);
            selected    &=     (l_shipdate[tid]  <   19950101);
            selected    &=     (l_discount[tid]  >=      0.05f);
            selected    &=     (l_discount[tid]  <=      0.07f);
            selected    &=     (l_quantity[tid]  <         24);

            if(selected) d_flags[tid] = 1;

            tid += gridDim.x * blockDim.x;
        }
    }

    __global__ void kernel_arithmetic_tpch_q6(const size_t num_elements,
                    const float    * l_extended_price,
                    const float    * l_discount,
                    char           * d_flags,
                    float          * d_products) {

        size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

        while(tid < num_elements) {

            if(d_flags[tid] > 0)
                d_products[tid] = l_extended_price[tid] * l_discount[tid];
            else
                d_products[tid] = 0.0;

            tid += gridDim.x * blockDim.x;
        }
    }


    __global__ void kernel_tpch_q6_holistic(const size_t num_elements,
                    const float    * l_extended_price,
                    const int      * l_quantity,
                    const float    * l_discount,
                    const uint32_t * l_shipdate,
                    float          * d_blockResults) {

        typedef cub::BlockScan<float, blockSize> BlockScan;

        __shared__ typename BlockScan::TempStorage temp_storage;

        TID tid = threadIdx.x +
                          blockIdx.x * blockDim.x +
                          blockIdx.y * blockDim.x * gridDim.x;

        TID block_index = blockIdx.y * gridDim.x + blockIdx.x;

        bool selected;
        float scanSum;
        float arithmetic;

        scanSum = 0.0;
        arithmetic = 0.0;
        selected = true;

        if(tid < num_elements) {
            selected    &=     (l_shipdate[tid]  >=  19940101);
            selected    &=     (l_shipdate[tid]  <   19950101);
            selected    &=     (l_discount[tid]  >=      0.05f);
            selected    &=     (l_discount[tid]  <=      0.07f);
            selected    &=     (l_quantity[tid]  <         24);

            if(selected) {
                arithmetic = l_extended_price[tid] * l_discount[tid];
            }
        }

        BlockScan(temp_storage).ExclusiveSum(arithmetic, scanSum);

        if(threadIdx.x == blockDim.x - 1) {
            d_blockResults[block_index] = scanSum + arithmetic;
        }
    }
