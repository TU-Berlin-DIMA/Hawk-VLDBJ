#google testing framework
enable_testing(true)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(code-generation-tests "feature_tests/cpp_code_generation.cpp")
target_link_libraries(code-generation-tests libgtest libgtest_main cogadb ${CMAKE_DL_LIBS})
add_test(code-generation-tests code-generation-tests)

add_executable(wah_bitmap_column-tests "unittests/compression_tests/wah_bitmap_column_tests.cpp")
target_link_libraries(wah_bitmap_column-tests libgtest libgtest_main cogadb)
#add_test(wah_bitmap_column-tests wah_bitmap_column-tests)

add_executable(dictionary_compressed_column-tests "unittests/compression_tests/dictionary_compressed_column_tests.cpp")
target_link_libraries(dictionary_compressed_column-tests libgtest libgtest_main cogadb)
add_test(dictionary_compressed_column-tests dictionary_compressed_column-tests)

add_executable(string_column_tests "unittests/storage/string_column_tests.cpp")
target_link_libraries(string_column_tests libgtest libgtest_main cogadb ${CMAKE_DL_LIBS})
add_test(string_column_tests string_column_tests)

add_executable(order_preserving_dictionary_compressed_column-tests "unittests/compression_tests/order_preserving_dictionary_compressed_column_tests.cpp")
target_link_libraries(order_preserving_dictionary_compressed_column-tests libgtest libgtest_main cogadb)
add_test(order_preserving_dictionary_compressed_column-tests order_preserving_dictionary_compressed_column-tests)

add_subdirectory(util)
