project(HyPE)
cmake_minimum_required(VERSION 2.6)
# do not treat constants as variable names
cmake_policy(SET CMP0012 NEW)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/CMake"
    "${PROJECT_SOURCE_DIR}/CMake/MonetDBOcelotConfig")
include(CheckSymbolExists)

find_package(PkgConfig)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMake")

# General options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(HYPE_ENABLE_MONETDB_OCELOT_DRIVER "Builds Driver Component for MonetDB/Ocelot" OFF)
option(MONETDB_OCELOT_ROOT_PATH "Path to MonetDB/Ocelot" /home/${USER}/monetdb-opencl)
#option(HYPE_ENABLE_INTERACTION_WITH_COGADB "Allows HyPE to call functions of CoGaDB" ON)

# Configure required Boost libraries
set(BOOST_ROOT "" CACHE PATH "Boost build root (useful on Windows)")
option(Boost_USE_STATIC_LIBS
       "Search for static boost libs" OFF)
option(Boost_USE_MULTITHREADED
       "Search for multithreaded boost libs" ON)
option(Boost_USE_STATIC_RUNTIME
       "Search for boost libs linked against static C++ runtime" OFF)
find_package(Boost 1.47.0 REQUIRED
             filesystem system thread program_options chrono serialization)
# ensure that dependant libraries not explicitly specified here
# are found by the linker:
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# Ensure that we link against Boost::Thread static library
# on MinGW if so requested
if (WIN32 AND __COMPILER_GNU AND
    Boost_USE_STATIC_LIBS)
        add_definitions(-DBOOST_THREAD_USE_LIB)
endif ()
if (APPLE)
  set(CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS} -undefined dynamic_lookup")
endif (APPLE)

# Configure TR1 headers: prefer Boost TR1 library
#find_path(TR1_INCLUDE_DIR memory
#	  HINTS "${Boost_INCLUDE_DIR}/boost/tr1/tr1"
#	  DOC "TR1 header location")
#if (NOT TR1_INCLUDE_DIR)
#	message(FATAL_ERROR "TR1 headers not found!")
#endif ()
#include_directories(${TR1_INCLUDE_DIR})

# Configure ALGLIB
option(INTERNAL_ALGLIB "Enforce using internal ALGLIB" OFF)
if (NOT ${INTERNAL_ALGLIB})
        # try to use pkg-config to configure ALGLIB
        pkg_check_modules(ALGLIB alglib>=3.6.0)
        if (${ALGLIB_FOUND})
                include_directories(${ALGLIB_INCLUDE_DIRS})
                set(LIBS ${LIBS} ${ALGLIB_LIBRARIES})
        endif ()
endif ()
if (NOT "${ALGLIB_FOUND}")
        message(STATUS "Using included ALGLIB")
        add_subdirectory(alglib)
        include_directories(${ALGLIB_INCLUDE_DIRS})
        set(EXTRA_SOURCES ${EXTRA_SOURCES} ${ALGLIB_SOURCES})
endif ()

if(HYPE_ENABLE_MONETDB_OCELOT_DRIVER)
# Configure MonetDBOcelot
find_package(MonetDBOcelot REQUIRED)
add_definitions(${MonetDBOcelot_DEFINITIONS})
include_directories(${MonetDBOcelot_INCLUDE_DIRS})
set(LIBS ${LIBS} ${MonetDBOcelot_LIBRARIES})
endif(HYPE_ENABLE_MONETDB_OCELOT_DRIVER)

# enable c++11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# HyPE includes
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/../cogadb/include")
#include_directories("${PROJECT_SOURCE_DIR}/lib/cogadb/include")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
        DESTINATION include/hype
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
        PATTERN ".svn" EXCLUDE)

file(GLOB_RECURSE HYPE_HEADER_FILES "include/*.h" "include/*.hpp")

add_library(hype
            src/algorithm.cpp
            src/algorithm_measurement.cpp
            src/algorithm_name_conversion.cpp
            src/algorithm_statistics.cpp
            src/architecture.cpp
            src/c_interface.cpp
            src/device_memory.cpp
            src/device_operator_queue.cpp
            src/drivers/monetdb_ocelot/monetdb_ocelot.cpp
            src/event_based_recomputation.cpp
            src/get_name.cpp
            src/hype.cpp
            src/knn_regression.cpp
            src/least_squares.cpp
            src/loadchange_estimator.cpp
            src/measurementpair.cpp
            src/measurementpair_ringbuffer.cpp
            src/multi_dim_least_squares.cpp
            src/node.cpp
            src/offline_algorithm.cpp
            src/oneshot_computation.cpp
            src/operation.cpp
            src/operator.cpp
            src/optimization_criterion.cpp
            src/periodic_recomputation.cpp
            src/plotscriptgenerator.cpp
            src/pluginloader.cpp
            src/print.cpp
            src/probability_based_outsourcing.cpp
            src/processing_device.cpp
            src/query_optimization/c_qep.cpp
            src/query_optimization/qep.cpp
            src/recomputation_heuristic.cpp
            src/report.cpp
            src/response_time_advanced.cpp
            src/response_time.cpp
            src/runtime_configuration.cpp
            src/scheduler.cpp
            src/scheduling_decision.cpp
            src/simple_round_robin_throughput.cpp
            src/specification.cpp
            src/statistical_method.cpp
            src/statistics_gatherer.cpp
            src/throughput2.cpp
            src/throughput.cpp
            src/time_measurement.cpp
            src/utility_functions.cpp
            src/virtual_processing_device.cpp
            ${EXTRA_SOURCES}
            ${HYPE_HEADER_FILES})
target_link_libraries(hype ${LIBS})

# Define HYPE_MAKE_SHARED macro when building a shared library;
# static, shared and import libraries are put in lib/,
# while DLLs are put in bin/
set_target_properties(hype PROPERTIES
                      DEFINE_SYMBOL HYPE_MAKE_SHARED
          ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
                      LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
                      RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")

# libhype installation
install(TARGETS hype EXPORT HyPETargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# CMake project config files
#install(EXPORT HyPETargets DESTINATION share/hype/CMake)
#install(FILES CMake/HyPEConfig.cmake DESTINATION share/hype/CMake)

# Testing
#enable_testing()
#add_subdirectory(examples)

# Documentation
add_subdirectory(doc)

# Generate Autoconf-like config.h
configure_file("${PROJECT_SOURCE_DIR}/config.h.in"
               "${PROJECT_BINARY_DIR}/config.h")
#configure_file("${PROJECT_SOURCE_DIR}/config.h.in"
#	       "${PROJECT_SOURCE_DIR}/config.h")
add_definitions(-DHAVE_CONFIG_H)

# HyPE includes
include_directories("${PROJECT_BINARY_DIR}")
#include_directories("${PROJECT_BINARY_DIR}"
#		    "${PROJECT_SOURCE_DIR}/include"
#		    "${PROJECT_SOURCE_DIR}/..")

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
MESSAGE(${dirs})
