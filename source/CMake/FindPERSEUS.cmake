# - Try to find perseus
# Once done this will define
#  PERSEUS_FOUND - System has libbam
#  PERSEUS_INCLUDE_DIRS - The libbam include directories
#  PERSEUS_LIBRARIES - The libraries needed to use libbam
#
# adapted from FindBAM.cmake

find_library(PERSEUS_LIBRARIES NAMES libperseus.so libperseus.dylib)
find_path(PERSEUS_INCLUDE_DIRS NAMES perseus/VariantPool.hpp)

if (PERSEUS_LIBRARIES)
	if(PERSEUS_INCLUDE_DIRS)
		set(PERSEUS_LIBRARIES ${PERSEUS_LIBRARIES})
		set(PERSEUS_INCLUDE_DIRS ${PERSEUS_INCLUDE_DIRS})
	else()
    		set(PERSEUS_ERROR_REASON "Perseus header files not found.")
	endif()

else()
	if(PERSEUS_INCLUDE_DIRS)
    		set(PERSEUS_ERROR_REASON "Library 'libperseus' not found.")
	else()
    		set(PERSEUS_ERROR_REASON "Perseus not found. If you want to use Perseus, clone it from: https://gitlab.tubit.tu-berlin.de/viktor-rosenfeld/perseus.git.")
	endif()

endif()

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set BAM_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(PERSEUS DEFAULT_MSG PERSEUS_LIBRARIES PERSEUS_INCLUDE_DIRS)
