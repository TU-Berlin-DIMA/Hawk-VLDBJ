add_subdirectory(hashtable)

add_source_cuda(gpu_aggregation.cu
                gpu_backend.cpp
                gpu_bit_operations.cu
                gpu_bitmap_set_operation.cu
                gpu_column_algebra.cu
                gpu_conversion_operations.cu
                gpu_fetch_join.cu
                gpu_join.cu
                gpu_selection.cu
                gpu_util.cu
                memory_cost_models.cpp
                positionlist_set_operation.cu
                stream_manager.cpp)