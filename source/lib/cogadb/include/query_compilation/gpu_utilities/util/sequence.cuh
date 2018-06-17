




template<typename T>
__global__ void sequence(T* data, size_t size) {

    T tid = blockDim.x * blockIdx.x + threadIdx.x;

    while(tid < size) {
    
        data[tid] = tid;
        tid += blockDim.x * gridDim.x;
    }

}
