enable_testing(true)
set(TEST_TARGET "perseus_unit_tests")

set(SOURCES
        ../utils/test_utils.cpp
        ConfigurationTest.cpp
        DefaultVariantPoolTest.cpp
        DiscreteFeatureTest.cpp
        IterativeUpdateStrategyTest.cpp
        MarkovUpdateStrategyTest.cpp
        RandomizedInitializationStrategyTest.cpp
        RandomizedUpdateStrategyTest.cpp
        RandomizedSelectionStrategyTest.cpp
        VWGreedySelectionStrategyTest.cpp)

add_executable(${TEST_TARGET} ${SOURCES})

target_link_libraries(${TEST_TARGET}
        ${PERSEUS_TARGET}
        libgmock libgtest_main)

if(NOT APPLE)
   target_link_libraries(${TEST_TARGET}
           pthread)
endif()

add_test(${TEST_TARGET} ${TEST_TARGET})
