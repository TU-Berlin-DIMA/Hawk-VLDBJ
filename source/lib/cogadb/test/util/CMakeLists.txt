
add_executable(variant-configurator-tests "variant_configurator.cpp")
target_link_libraries(variant-configurator-tests libgtest libgtest_main cogadb ${CMAKE_DL_LIBS})
add_test(variant-configurator-tests variant-configurator-tests)

add_executable(variant-measurement-tests "variant_measurement.cpp")
target_link_libraries(variant-measurement-tests libgtest libgtest_main cogadb ${CMAKE_DL_LIBS})
add_test(variant-measurement-tests variant-measurement-tests)
