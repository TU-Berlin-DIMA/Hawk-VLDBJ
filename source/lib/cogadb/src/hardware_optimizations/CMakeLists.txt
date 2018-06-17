add_subdirectory(main_memory_joins)

add_source_cogadb(#loop_unrolling.cpp
                  primitives.cpp
                  #radix_cluster.cpp
                  #radix.cpp
                  simd_acceleration.cpp)
                  