# Configure HyPE package for CoGaDB as part of the same
# master project.
# The usual way of FIND_PACKAGE finding the ordinary config file
# will not work unless HyPE is installed before CoGaDB is
# configured
# See also ../hype-library/CMake/HyPEConfig.cmake

get_filename_component(_GPUDBMS "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

#set(HyPE_INCLUDE_DIRS "${_GPUDBMS}/lib/hype-library/include"
#    CACHE PATH "HyPE include directories")
set(HyPE_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/lib/hype-library/include" "${PROJECT_SOURCE_DIR}/include" "/home/sebastian/Software/gpudbms/cogadb/include"
    CACHE PATH "HyPE include directories")
# "hype" target exists since HyPE was added to the master project
set(HyPE_LIBRARIES hype
    CACHE STRING "HyPE library or target")

get_target_property(HyPE_TYPE hype TYPE)
if (${HyPE_TYPE} STREQUAL SHARED_LIBRARY)
	set(HyPE_DEFINITIONS "-DHYPE_USE_SHARED")
endif ()

mark_as_advanced(HyPE_INCLUDE_DIRS HyPE_LIBRARIES)
