add_subdirectory(code_generators)
add_subdirectory(primitives)
add_subdirectory(execution_strategy)
if(ENABLE_GPU_ACCELERATION)
    add_subdirectory(gpu_handbuilt)
endif(ENABLE_GPU_ACCELERATION)
if(PERSEUS_FOUND)
    add_subdirectory(variant_tuning)
endif()

add_source_cogadb(aggregate_specification.cpp
                  algebraic_aggregate_specification.cpp
                  code_generator.cpp
                  kernel_execution.cpp
                  minimal_api.cpp
                  minimal_api_c.cpp
                  hash_table_generator.cpp
                  ocl_api.cpp
                  ocl_data_cache.cpp
                  pipeline.cpp
                  pipeline_info.cpp
                  pipeline_job.cpp
                  user_defined_code.cpp
                  predicate_expression.cpp
                  predicate_specification.cpp
                  query_context.cpp
                  ocl_cuckoo2hashes.cpp
                  ocl_lin_genhash.cpp
                  pipeline_selectivity_estimates.cpp
                  )


if(PERSEUS_FOUND)
	add_source_cogadb(variant_generator_wrapper.cpp)
endif()

