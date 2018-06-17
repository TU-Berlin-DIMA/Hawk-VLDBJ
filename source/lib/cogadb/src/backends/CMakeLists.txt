add_subdirectory(cpu)

if(ENABLE_GPU_ACCELERATION)
    add_subdirectory(gpu)
endif(ENABLE_GPU_ACCELERATION)

#add_subdirectory(xeonphi)

add_source_cogadb(processor_backend.cpp)
