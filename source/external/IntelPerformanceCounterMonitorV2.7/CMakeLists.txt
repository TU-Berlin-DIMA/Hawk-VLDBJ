
foreach (source IN ITEMS
client_bw.cpp
cpucounters.cpp
msr.cpp
pci.cpp)
	set(INTEL_PCM_SOURCES ${INTEL_PCM_SOURCES}
	    "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
endforeach (source)

set(INTEL_PCM_SOURCES ${INTEL_PCM_SOURCES} PARENT_SCOPE)
set(INTEL_PCM_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/MacMSRDriver"
    PARENT_SCOPE)
