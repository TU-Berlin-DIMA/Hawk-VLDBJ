cmake_minimum_required(VERSION 2.8)

macro (add_source_perseus)
    add_source(PERSEUS "${ARGN}")
endmacro()

macro (get_source_perseus SOURCE_FILES)
    get_source(PERSEUS SOURCE_FILES_LOCAL)
    set(${SOURCE_FILES} ${SOURCE_FILES_LOCAL})
endmacro()

project("perseus")

if(POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Werror")

set(PERSEUS_TARGET "perseus")

include_directories("${PROJECT_SOURCE_DIR}/include")

enable_testing()

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

get_source_perseus(PERSEUS_SOURCE_FILES)
file(GLOB_RECURSE PERSEUS_HEADER_FILES "include/*.h" "include/*.hpp")

add_library(${PERSEUS_TARGET} SHARED ${PERSEUS_SOURCE_FILES}
    ${PERSEUS_HEADER_FILES})

target_link_libraries(${PERSEUS_TARGET} ${Boost_LIBRARIES})
install(TARGETS ${PERSEUS_TARGET} LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
