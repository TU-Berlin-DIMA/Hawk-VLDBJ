add_subdirectory(serial_hash_join)
add_source_cogadb(hash_joins.cpp)

if(NOT APPLE)
#add_subdirectory(parallel_radix)
#add_source_cogadb(radix_hash_joins.cpp)
endif()
