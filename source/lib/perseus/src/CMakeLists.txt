find_package(Boost COMPONENTS log)

add_source_perseus(
        CurrentRuntimeVariantScorer.cpp
        ElitismUpdateStrategy.cpp
        Configuration.cpp
        DefaultVariantPool.cpp
        ErrorFunctionVariantScorer.cpp
        FullPoolBaselineStrategy.cpp
        GeneticUpdateStrategy.cpp
        IterativeUpdateStrategy.cpp
        MarkovUpdateStrategy.cpp
        RandomizedInitializationStrategy.cpp
        RandomizedUpdateStrategy.cpp
        RandomizedSelectionStrategy.cpp
        RandomizedVariantFactoryImpl.cpp
        TotalCallsVariantScorer.cpp
        TotalRuntimeVariantScorer.cpp
        utils.cpp
        Variant.cpp
        VWGreedyBaselineStrategy.cpp
        VWGreedySelectionStrategy.cpp)



