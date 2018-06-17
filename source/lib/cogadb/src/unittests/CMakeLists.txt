add_source_cogadb(cdk_primitives_unittests.cpp
                  histogram_unittests.cpp
                  join_unittests.cpp
                  positionlist_unittests.cpp
                  unittests.cpp)               
                  
if(ENABLE_GPU_ACCELERATION)
    add_source_cuda(gpu_work_unittests.cu)
endif(ENABLE_GPU_ACCELERATION)
