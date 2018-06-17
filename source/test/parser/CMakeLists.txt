#
# JSON parser tests
#
add_executable(json_parser-tests
               json_parser.cpp)

target_link_libraries(json_parser-tests
                      libgtest libgtest_main
                      cogadb)

add_test(json_parser-tests json_parser-tests)

if(JSON_VALIDATION_ENABLED)
  #
  # Json validation tests
  #
  add_executable(json_validator-tests
                 json_validator.cpp)

  target_link_libraries(json_validator-tests
                        libgtest libgtest_main
                        cogadb)

add_test(json_validator-tests json_validator-tests)

endif(JSON_VALIDATION_ENABLED)
