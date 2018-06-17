###
### ALGLIB
### -- will be linked into HyPE, and we don't want to install it.
### Building a static library is not possible since the object files built
### must be built with relocatable symbols. Libtool supports convenience
### libraries for that matter.
### For CMake we must add ALGLIB to HyPE's sources.
###
### Defines: ALGLIB_SOURCES, ALGLIB_INCLUDE_DIRS
###

foreach (source IN ITEMS
	 alglibinternal.cpp
	 ap.cpp
	 diffequations.cpp
	 integration.cpp
	 linalg.cpp
	 solvers.cpp
	 statistics.cpp
	 alglibmisc.cpp
	 dataanalysis.cpp
	 fasttransforms.cpp
	 interpolation.cpp
	 optimization.cpp
	 specialfunctions.cpp)
	set(ALGLIB_SOURCES ${ALGLIB_SOURCES}
	    "${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/${source}")
endforeach (source)

set(ALGLIB_SOURCES ${ALGLIB_SOURCES} PARENT_SCOPE)
set(ALGLIB_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/src"
    PARENT_SCOPE)
