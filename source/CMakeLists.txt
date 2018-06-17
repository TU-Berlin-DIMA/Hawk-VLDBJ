macro (add_source PROP_NAME SOURCE_FILES)
    set(SOURCE_FILES_ABSOLUTE )
    foreach(it ${SOURCE_FILES})
        get_filename_component(ABSOLUTE_PATH ${it} ABSOLUTE)
        set(SOURCE_FILES_ABSOLUTE ${SOURCE_FILES_ABSOLUTE} ${ABSOLUTE_PATH})
    endforeach()

    get_property(OLD_PROP_VAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set_property(GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP" ${SOURCE_FILES_ABSOLUTE} ${OLD_PROP_VAL})
endmacro()

macro (get_source PROP_NAME SOURCE_FILES)
    get_property(SOURCE_FILES_LOCAL GLOBAL PROPERTY "${PROP_NAME}_SOURCE_PROP")
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
endmacro()

macro (add_source_cogadb)
    add_source(COGADB "${ARGN}")
endmacro()

macro (get_source_cogadb SOURCE_FILES)
    get_source(COGADB SOURCE_FILES_LOCAL)
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
endmacro()

macro (add_source_cuda)
    add_source(CUDACOGABD "${ARGN}")
endmacro()

macro (get_source_cuda SOURCE_FILES)
    get_source(CUDACOGABD SOURCE_FILES_LOCAL)
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
endmacro()

project(CoGaDB)
set(COGADB_TARGET "cogadb")
cmake_minimum_required(VERSION 2.6)
# do not treat constants as variable names
cmake_policy(SET CMP0012 NEW)

set(COGADB_VERSION "\"0.4.2\"")
set(PATH_TO_COGADB_EXECUTABLE "\"${PROJECT_BINARY_DIR}\"")
set(PATH_TO_COGADB_SOURCE_CODE "\"${PROJECT_SOURCE_DIR}\"")
#set(PATH_TO_DATA_OF_TESTS "\"${PROJECT_BINARY_DIR}/test/testdata\"")
set(PATH_TO_DATA_OF_TESTS "\"${PROJECT_BINARY_DIR}/test/testdata\"")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMake"
    "${PROJECT_SOURCE_DIR}/CMake/FindTBB")
include(CheckSymbolExists)
include(CheckIncludeFileCXX)

# Configure CCache if available
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "CCACHE found")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
else(CCACHE_FOUND)
    message(STATUS "CCACHE not found")
endif(CCACHE_FOUND)

set(COGADB_INCLUDE_DIRS
    "${PROJECT_BINARY_DIR}"
    "${PROJECT_SOURCE_DIR}/lib/cogadb/include"
    "${PROJECT_SOURCE_DIR}/lib/cogadb/include/hardware_optimizations/main_memory_joins/serial_hash_join"
    "${PROJECT_SOURCE_DIR}/lib/hype/include")

# Subsequent FIND_PACKAGE(HyPE) commands will find
# the internal HyPEConfig.cmake file instead of
# any installed one.
set(HyPE_DIR "${PROJECT_SOURCE_DIR}/CMake"
    CACHE PATH "The directory containing a CMake configuration file for HyPE.")

find_package(Thrust REQUIRED)
set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} ${THRUST_INCLUDE_DIR})

add_subdirectory(external/IntelPerformanceCounterMonitorV2.7)
set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} ${INTEL_PCM_INCLUDE_DIRS})
set(EXTRA_SOURCES ${EXTRA_SOURCES} ${INTEL_PCM_SOURCES})

# General options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(USE_GOOGLE_PERF_TOOLS "Use Google performance tools if available" OFF)

# Configure required Boost libraries
set(BOOST_ROOT "" CACHE PATH "Boost build root (useful on Windows)")
option(Boost_USE_STATIC_LIBS
       "Search for static boost libs" OFF)
option(Boost_USE_MULTITHREADED
       "Search for multithreaded boost libs" ON)
option(Boost_USE_STATIC_RUNTIME
       "Search for boost libs linked against static C++ runtime" OFF)
find_package(Boost 1.47.0 REQUIRED
             filesystem system thread program_options serialization
             chrono random date_time iostreams regex log)
# ensure that dependant libraries not explicitly specified here
# are found by the linker:
link_directories(${Boost_LIBRARY_DIRS})
set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# Configure FindTBB
find_package(TBB REQUIRED)
set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS})
set(LIBS ${LIBS} ${TBB_LIBRARIES})

# CPU ISA extensions
find_package(CPUISAXs REQUIRED)

# BoostCompute
find_package(BoostCompute REQUIRED)
set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} ${BoostCompute_INCLUDE_DIRS})
add_definitions(-DBOOST_COMPUTE_THREAD_SAFE -DBOOST_COMPUTE_HAVE_THREAD_LOCAL)

# OpenCL
find_package(OpenCL REQUIRED)
set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OPENCL_LIBRARIES})
add_definitions(-DCL_USE_DEPRECATED_OPENCL_1_2_APIS -DCL_USE_DEPRECATED_OPENCL_1_1_APIS)

# Configure CUDA
option(ENABLE_GPU_ACCELERATION "GPU acceleration enabled" OFF)

if(ENABLE_GPU_ACCELERATION)
    set(CUDA_GENERATED_OUTPUT_DIR "${PROJECT_BINARY_DIR}/CMakeFiles")
    set(CUDA_NVCC_FLAGS "-arch=sm_30"
        CACHE STRING "CUDA nvcc flags (semicolon separated)")
    find_package(CUDA 6.5 REQUIRED)
endif ()

# Check for cudaMemGetInfo() availability
set(CMAKE_REQUIRED_INCLUDES ${CUDA_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${CUDA_LIBRARIES})
# avoid caching
unset(HAVE_CUDAMEMGETINFO CACHE)
check_symbol_exists(cudaMemGetInfo "cuda.h;cuda_runtime_api.h"
		    HAVE_CUDAMEMGETINFO)

# Configure Flex and Bison
find_package(FLEX REQUIRED)
find_package(BISON 2.5 REQUIRED)
execute_process(COMMAND ${BISON_EXECUTABLE} --print-datadir
		OUTPUT_VARIABLE BISON_DATA_DIR
		OUTPUT_STRIP_TRAILING_WHITESPACE)

#message(INFO "Bison Version: ${BISON_VERSION}")
#option(BISON_VERSION_3_OR_HIGHER "Bison Version three or higher found" ON)
if(BISON_VERSION VERSION_EQUAL 3.0.2 OR BISON_VERSION  VERSION_GREATER 3.0.2)
    set(BISON_VERSION_3_OR_HIGHER true)
else(BISON_VERSION VERSION_LESS 3.0.2)
    set(BISON_VERSION_3_OR_HIGHER false)
    #option(BISON_VERSION_3_OR_HIGHER "Bison Version three or higher found" OFF)
endif()

# Configure BAM - for genomics support
find_package(BAM)
if (BAM_FOUND)
  set(LIBS ${LIBS} ${BAM_LIBRARIES})
  message(INFO " Enabling genomics importer for FASTA, SAM and BAM files.")
else (!BAM_FOUND)
  message(WARNING " Disabling genomics importer for FASTA, SAM and BAM files. Reason: " ${BAM_ERROR_REASON})
endif()


# Configure Perseus - for Perseus integration
option(ENABLE_PERSEUS "Enable perseus" OFF)
if (ENABLE_PERSEUS)
  if (PERSEUS_FOUND)
    set(LIBS ${LIBS} "perseus")
    message(INFO " Enabling Perseus as Variant Tuning Framework for CoGaDB.")
  else (!PERSEUS_FOUND)
    message(WARNING " Disabling Perseus as Variant Tuning Framework for CoGaDB. Reason: " ${PERSEUS_ERROR_REASON})
  endif()
endif(ENABLE_PERSEUS)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(USE_GOOGLE_PERF_TOOLS)
  find_package(GooglePerfTools)
  if (GOOGLE_PERFTOOLS_FOUND)
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${GOOGLE_PERFTOOLS_INCLUDE_DIR}")
      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${GOOGLE_PERFTOOLS_INCLUDE_DIR}")
      set(LIBS ${LIBS} ${TCMALLOC_LIBRARIES})
      set(LIBS ${LIBS} ${STACKTRACE_LIBRARIES})
      set(LIBS ${LIBS} ${PROFILER_LIBRARIES})
  endif()
else()
  message(INFO " Google performance tools will not be used.")
endif()

# Configure GNU Readline (optional)
find_package(LibReadline)
if (LIBREADLINE_FOUND)
  set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} ${LIBREADLINE_INCLUDE_DIRS})
  set(LIBS ${LIBS} ${LIBREADLINE_LIBRARIES})
else ()
  message(WARNING "GNU Readline not found! Falling back to simple command line reader.")
endif ()

# Configure LLVM and clang
find_package(LLVM 3.6 REQUIRED)
find_package(clang REQUIRED)
set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
set(LIBS ${LIBS} ${CLANG_LIBS} ${LLVM_MODULE_LIBS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CFLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LLVM_LDFLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${LLVM_LDFLAGS}")
add_definitions(-DLLVM_VERSION=${LLVM_VERSION_SIMPLIFIED})

# Configure sparsehash
find_package(sparsehash REQUIRED)

# Configure RapidJSON
find_package(RapidJSON REQUIRED)

# Enable JSON schema validation support
CHECK_INCLUDE_FILE_CXX(rapidjson/schema.h JSON_VALIDATION_ENABLED)
if(JSON_VALIDATION_ENABLED)
  add_definitions( -DJSON_VALIDATION_ENABLED )
  message(STATUS "Enabled JSON schema validation")
else()
  message(STATUS "Disabled JSON schema validation")
endif(JSON_VALIDATION_ENABLED)

# We need thread support
find_package(Threads REQUIRED)

# Enable ExternalProject CMake module
include(ExternalProject)

# Download and install GoogleTest
ExternalProject_Add(
  gtest
  URL https://github.com/google/googletest/archive/master.zip
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
  # Disable install step
  INSTALL_COMMAND ""
)

# Get GTest source and binary directories from CMake project
ExternalProject_Get_Property(gtest source_dir binary_dir)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest gtest_main)

# Set libgtest properties
set_target_properties(libgtest PROPERTIES
  "IMPORTED_LOCATION" "${binary_dir}/googlemock/gtest/libgtest.a"
  "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

add_library(libgtest_main IMPORTED STATIC GLOBAL)
add_dependencies(libgtest_main gtest_main)

set_target_properties(libgtest_main PROPERTIES
  "IMPORTED_LOCATION" "${binary_dir}/googlemock/gtest/libgtest_main.a"
  "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

# Create a libgmock target to be used as a dependency by test programs
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gtest)

# Set libgmock properties
set_target_properties(libgmock PROPERTIES
  "IMPORTED_LOCATION" "${binary_dir}/googlemock/libgmock.a"
  "IMPORTED_LINK_INTERFACE_LIBRARIES" "${CMAKE_THREAD_LIBS_INIT}"
)

set(GMOCK_INCLUDE_DIRS "${source_dir}/googlemock/include")

include_directories("${source_dir}/googletest/include"
                    "${source_dir}/googlemock/include")

# CoGaDB-specific options
option(ENABLE_TWO_PHASE_PHYSICAL_OPTIMIZATION "Two phase physical optimization enabled" ON)
option(ENABLE_SIMD_ACCELERATION "SIMD acceleration enabled" OFF)
option(ENABLE_BRANCHING_SCAN "enables branching inside scan operator variants" ON)
option(ENABLE_CPU_NESTED_LOOP_JOIN "CPU nested loop join enabled" OFF)

set(CMAKE_CXX_FLAGS "-msse4.1" CACHE STRING "flags used for all build types")
set(CMAKE_CXX_FLAGS_RELEASE "-O6 -fno-omit-frame-pointer -falign-functions=4 -falign-loops=4 -falign-jumps=4 -ftree-vectorize -fexpensive-optimizations -frerun-cse-after-loop  -funroll-loops -frerun-loop-opt -finline-functions" CACHE STRING "C++ Optimization flags used when build in release mode")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS "-std=gnu99")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-omit-frame-pointer -falign-functions=4 -falign-loops=4 -falign-jumps=4 -ftree-vectorize -fexpensive-optimizations -frerun-cse-after-loop  -funroll-loops -frerun-loop-opt -finline-functions")

set(CUDA_NVCC_FLAGS "-rdc=true -gencode arch=compute_35,code=sm_35 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_20,code=sm_20" CACHE STRING "Configures for which GPU Compute Capabilities CoGaDB is compiled.")

# set clang binary for query compilation
execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --prefix OUTPUT_VARIABLE LLVM_INSTALL_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
set(QUERY_COMPILATION_CC "\"${LLVM_INSTALL_PREFIX}/bin/clang\"")
set(QUERY_COMPILATION_CXX "\"${LLVM_INSTALL_PREFIX}/bin/clang++\"")

execute_process(COMMAND ${LLVM_CONFIG_EXECUTABLE} --version OUTPUT_VARIABLE CLANG_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
set(QUERY_COMPILATION_CLANG_VERSION "\"${CLANG_VERSION}\"")

# Generate Autoconf-like config.h
configure_file("${PROJECT_SOURCE_DIR}/config.h.in" "${PROJECT_BINARY_DIR}/config.h")
add_definitions(-DHAVE_CONFIG_H)

if(NOT APPLE)
add_source_cogadb(${PROJECT_SOURCE_DIR}/external/IntelPerformanceCounterMonitorV2.7/client_bw.cpp
                  ${PROJECT_SOURCE_DIR}/external/IntelPerformanceCounterMonitorV2.7/cpucounters.cpp
                  ${PROJECT_SOURCE_DIR}/external/IntelPerformanceCounterMonitorV2.7/msr.cpp
                  ${PROJECT_SOURCE_DIR}/external/IntelPerformanceCounterMonitorV2.7/pci.cpp
                  ${PROJECT_SOURCE_DIR}/external/IntelPerformanceCounterMonitorV2.7/utils.cpp)
endif()

add_subdirectory(lib)
add_definitions(${HyPE_DEFINITIONS})
set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} ${HyPE_INCLUDE_DIRS})
set(LIBS ${LIBS} ${HyPE_LIBRARIES})

# SQL Parser generation
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/src/sql/server")
set(COGADB_INCLUDE_DIRS ${COGADB_INCLUDE_DIRS} "${PROJECT_BINARY_DIR}/src/sql/server")

set(BISON_SQLParser_FLAGS "--defines=${PROJECT_BINARY_DIR}/src/sql/server/sql_parser.hpp --xml=${PROJECT_BINARY_DIR}/doc/sql_grammar.xml")
bison_target(SQLParser lib/cogadb/src/sql/server/sql_parser.ypp
             "${PROJECT_BINARY_DIR}/src/sql/server/sql_parser.cpp"
             COMPILE_FLAGS "${BISON_SQLParser_FLAGS}")
add_custom_target(SQLParser DEPENDS ${BISON_SQLParser_OUTPUTS})

flex_target(SQLScanner lib/cogadb/src/sql/server/sql_scanner.lpp
            "${PROJECT_BINARY_DIR}/src/sql/server/sql_scanner.cpp")


add_flex_bison_dependency(SQLScanner SQLParser)

add_source_cogadb(${BISON_SQLParser_OUTPUTS}
                  ${FLEX_SQLScanner_OUTPUTS})

# CoGaDB includes
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION include/cogadb
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE)

if(ENABLE_GPU_ACCELERATION)
    add_source_cuda(${PROJECT_SOURCE_DIR}/external/moderngpu/src/mgpucontext.cu
                    ${PROJECT_SOURCE_DIR}/external/moderngpu/src/mgpuutil.cpp)

    get_source_cuda(CUDA_SOURCE)

    cuda_add_library(cuda_gpu_backend
                     ${CUDA_SOURCE})

    set(LIBS ${LIBS} cuda_gpu_backend)
    # static, shared and import libraries are put in lib/,
    # while DLLs are put in bin/
    set_target_properties(cuda_gpu_backend PROPERTIES
                          ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
                          LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
                          RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
else()
    message(WARNING "GPU acceleration is disabled, I will not build the CUDA backend!")
endif()

get_source_cogadb(COGADB_SOURCE_FILES)

file(GLOB_RECURSE COGADB_HEADER_FILES "lib/cogadb/include/*.h"
    "lib/cogadb/include/*.hpp")

# enable c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(ENABLE_GPU_ACCELERATION)
    cuda_add_library(cogadb ${COGADB_SOURCE_FILES} ${COGADB_HEADER_FILES})
else()
    add_library(cogadb ${COGADB_SOURCE_FILES} ${COGADB_HEADER_FILES})
endif()

set(CMAKE_C_FLAGS "-std=gnu99")

target_link_libraries(cogadb ${LIBS} hype)
target_include_directories(cogadb PUBLIC ${COGADB_INCLUDE_DIRS})

#http://stackoverflow.com/questions/2368811/how-to-set-warning-level-in-cmake
if(MSVC)
    target_compile_options(cogadb PRIVATE /W4)
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(cogadb PRIVATE -Wall -Wno-unused-variable
        -Wno-unused-value
        -Wno-deprecated-declarations -pedantic -Wno-ignored-attributes)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(cogadb PRIVATE -Wall -Wno-unused-variable
        -Wno-unused-value -Wno-long-long -Wno-deprecated-register
        -Wno-unused-private-field -Wno-mismatched-tags
        -Wno-sometimes-uninitialized -Wno-overloaded-virtual
        -Wno-nested-anon-types -Wno-deprecated-declarations
        -Wno-format -Wno-self-assign-field -Wno-pessimizing-move
        -Wno-vla-extension -Wno-infinite-recursion
        -Wno-tautological-undefined-compare -pedantic)
endif()

# static, shared and import libraries are put in lib/,
# while DLLs are put in bin/
set_target_properties(cogadb PROPERTIES
		      ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
		      LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
		      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

add_subdirectory(tools)

# CoGaDB installation
install(TARGETS cogadb
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "CoGaDB Development Team <cogadb@googlegroups.com>")
SET(CPACK_PACKAGE_VERSION 0.4.2)
SET(CPACK_PACKAGE_NAME "cogadb")
SET(CPACK_PACKAGE_VERSION_MAJOR 0)
SET(CPACK_PACKAGE_VERSION_MINOR 4)
SET(CPACK_PACKAGE_VERSION_PATCH 2)
SET(CPACK_SYSTEM_NAME ubuntu-14.04-lts-amd64)

#first comes the short description, then a "\n<white space>", then the long description
SET(CPACK_DEBIAN_PACKAGE_DESCRIPTION "A Column-oriented GPU-accelerated DBMS\n CoGaDB is a column-oriented GPU-accelerated database management system developed at TU Dortmund University and University of Magdeburg.
    Its purpose is to investigate advanced coprocessing techniques for effective GPU utilization during database query processing.
    It utilizes our hybrid query processing engine (HyPE) for the physical optimization process.\n Website: http://cogadb.cs.tu-dortmund.de/wordpress/")

SET(CPACK_DEBIAN_PACKAGE_VERSION ${CPACK_PACKAGE_VERSION})
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libgcc1, libboost-chrono1.54-dev, libboost-chrono1.54.0,
    libboost-date-time1.54-dev,libboost-date-time1.54.0, libboost-filesystem1.54-dev, libboost-filesystem1.54.0,
    libboost-iostreams1.54-dev, libboost-iostreams1.54.0, libboost-program-options1.54-dev, libboost-program-options1.54.0,
    libboost-random1.54-dev, libboost-random1.54.0, libboost-regex1.54-dev, libboost-regex1.54.0, libboost-serialization1.54-dev,
    libboost-serialization1.54.0, libboost-system1.54-dev, libboost-system1.54.0, libboost-thread1.54-dev, libboost-thread1.54.0, libtbb2, zlib1g, libreadline6, libgomp1")

INCLUDE(CPack)

###
### Test programs and benchmarks
### FIXME: should probably be tests defined in
### benchmarks/CMakeLists.txt and run by CTest
###

add_executable(sql_test_ssb_diagrams lib/cogadb/src/sql/server/test_ssb_diagrams.cpp)
target_link_libraries(sql_test_ssb_diagrams cogadb)
# put executable in bin/
set_target_properties(sql_test_ssb_diagrams PROPERTIES
		      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

add_executable(sql_benchmark_parser lib/cogadb/src/sql/server/benchmark_parser.cpp)
target_link_libraries(sql_benchmark_parser cogadb)
# put executable in bin/
set_target_properties(sql_benchmark_parser PROPERTIES
		      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

if(ENABLE_GPU_ACCELERATION)
    add_executable(aggregation_benchmark benchmarks/aggregation_benchmark.cpp)
    target_link_libraries(aggregation_benchmark cogadb)
    # put executable in bin/
    set_target_properties(aggregation_benchmark PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

    add_executable(selection_benchmark benchmarks/selection_benchmark.cpp)
    target_link_libraries(selection_benchmark cogadb)
    # put executable in bin/
    set_target_properties(selection_benchmark PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

    add_executable(sort_benchmark benchmarks/sort_benchmark.cpp)
    target_link_libraries(sort_benchmark cogadb)
    # put executable in bin/
    set_target_properties(sort_benchmark PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

    add_executable(join_benchmark benchmarks/join_benchmark.cpp)
    target_link_libraries(join_benchmark cogadb)
    # put executable in bin/
    set_target_properties(join_benchmark PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

    add_executable(generic_selection_benchmark benchmarks/generic_selection_benchmark.cpp)
    target_link_libraries(generic_selection_benchmark cogadb)
    # put executable in bin/
    set_target_properties(generic_selection_benchmark PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

    cuda_add_executable(gpu_memory_allocator benchmarks/gpu_memory_allocator.cu)
    #set_target_properties(gpu_memory_allocator PROPERTIES LINKER_LANGUAGE CXX)
    # put executable in bin/
    set_target_properties(gpu_memory_allocator PROPERTIES
                          RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
else ()
        message(WARNING "GPU acceleration is disabled, I will not build the hybrid CPU/GPU operator benchmarks!")
endif()

# Output directory required by libhype
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/output")

# Documentation
add_subdirectory(doc)

#unittests
add_subdirectory(test)
add_subdirectory(lib/cogadb/test)

enable_testing()
add_custom_target(cogadb_tests COMMAND ${CMAKE_CTEST_COMMAND}
                  DEPENDS ${UNITTESTS})

if(APPLE)
  target_link_libraries(cogadb z readline ncurses omp)
endif(APPLE)

file(COPY res/parser DESTINATION share/cogadb)

# Unified doc target
add_custom_target(doc DEPENDS hype-doc cogadb-doc)

# Testing: create top-level test configuration
enable_testing()
