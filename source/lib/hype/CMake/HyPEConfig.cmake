###
### HyPE CMake configuration file:
### Assumes that it will be installed into ${PREFIX}/share/hype/CMake
### to calculate the PREFIX from its script location
###

include(${CMAKE_CURRENT_LIST_DIR}/HyPETargets.cmake)

get_filename_component(PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(HyPE_INCLUDE_DIRS "${PREFIX}/include/hype"
    CACHE PATH "HyPE include directories")
# hype exists as an imported target now
set(HyPE_LIBRARIES hype
    CACHE STRING "HyPE library or target")

get_target_property(HyPE_TYPE hype TYPE)
if (${HyPE_TYPE} STREQUAL SHARED_LIBRARY)
	set(HyPE_DEFINITIONS "-DHYPE_USE_SHARED")
endif ()
    
mark_as_advanced(HyPE_INCLUDE_DIRS HyPE_LIBRARIES)
