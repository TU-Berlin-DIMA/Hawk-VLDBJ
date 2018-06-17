# - Try to find libbam
# Once done this will define
#  BAM_FOUND - System has libbam
#  BAM_INCLUDE_DIRS - The libbam include directories
#  BAM_LIBRARIES - The libraries needed to use libbam
#
# adapted from http://www.cmake.org/Wiki/CMake:How_To_Find_Libraries#Writing_find_modules

find_library(BAM_LIBRARIES NAMES bam libbam)

if (BAM_LIBRARIES)
    find_package(ZLIB)
    if(ZLIB_INCLUDE_DIRS)
        set(BAM_LIBRARIES ${BAM_LIBRARIES} ${ZLIB_LIBRARIES})
    else()
        set(BAM_ERROR_REASON "zlib.h not found. You should install following package on Ubuntu to fix this: zlib1g-dev.")
        return()
    endif()

    find_package(Threads)
    set(BAM_LIBRARIES ${BAM_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

else()
    set(BAM_ERROR_REASON "libbam not found. You should install following package on Ubuntu to fix this: libbam-dev.")
endif()

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set BAM_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(BAM DEFAULT_MSG BAM_LIBRARIES)