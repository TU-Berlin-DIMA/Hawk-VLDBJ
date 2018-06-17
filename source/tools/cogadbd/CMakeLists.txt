project(CoGaDBD)

add_executable(cogadbd main.cpp)
target_link_libraries(cogadbd cogadb)

# put executable in bin/
set_target_properties(cogadbd PROPERTIES
		      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
		      
install(TARGETS cogadbd
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	RUNTIME DESTINATION bin)
