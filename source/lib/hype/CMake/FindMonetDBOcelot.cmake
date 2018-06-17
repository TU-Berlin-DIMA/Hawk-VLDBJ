#include(${CMAKE_CURRENT_LIST_DIR}/HyPETargets.cmake)

#get_filename_component(PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(MONETDB_OCELOT_INCLUDE_DIRS "${MONETDB_OCELOT_ROOT_PATH}"
    CACHE PATH "MonetDB/Ocelot include directories")
# hype exists as an imported target now
set(MONETDB_OCELOT_LIBRARIES hype
    CACHE STRING "Ocelot library or target")


find_path(MONETDB_OCELOT_INCLUDE_DIRS monetdb5/optimizer/opt_qep.h
          HINTS ${MONETDB_OCELOT_ROOT_PATH}
          PATH_SUFFIXES monetdb5/optimizer )

find_library(MONETDB_OCELOT_LIBRARIES NAMES liboptimizer
             HINTS ${MONETDB_OCELOT_ROOT_PATH}/monetdb5/optimizer )

MESSAGE("PATH to Ocelot: '${MONETDB_OCELOT_ROOT_PATH}'")

#get_target_property(HyPE_TYPE hype TYPE)
#if (${HyPE_TYPE} STREQUAL SHARED_LIBRARY)
#	set(HyPE_DEFINITIONS "-DHYPE_USE_SHARED")
#endif ()
    
mark_as_advanced(MONETDB_OCELOT_INCLUDE_DIRS MONETDB_OCELOT_LIBRARIES)
