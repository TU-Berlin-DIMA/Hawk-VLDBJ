
#pragma once
//#ifndef HAVE_CONFIG_H
//#define HAVE_CONFIG_H
//#include "config.h"
//#endif

#define HYPE_ENABLE_PARALLEL_QUERY_PLAN_EVALUATION
//#define HYPE_USE_MEMORY_COST_MODELS
#define HYPE_INCLUDE_INPUT_DATA_LOCALITY_IN_FEATURE_VECTOR
#define HYPE_ENABLE_INTERACTION_WITH_COGADB

//#include <limits>

#ifdef __cplusplus
#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
namespace CoGaDB {
  void exit(int status) __attribute__((noreturn));
}
#endif

namespace hype {

#define HYPE_WARNING(X, OUTSTREAM)                                  \
  {                                                                 \
    std::cout << "WARNING: " << X << ": In " << __PRETTY_FUNCTION__ \
              << std::endl;                                         \
    OUTSTREAM << "In File: " << __FILE__ << " Line: " << __LINE__   \
              << std::endl;                                         \
  }

#define HYPE_ERROR(X, OUTSTREAM)                                  \
  {                                                               \
    std::cout << "ERROR: " << X << ": In " << __PRETTY_FUNCTION__ \
              << std::endl;                                       \
    OUTSTREAM << "In File: " << __FILE__ << " Line: " << __LINE__ \
              << std::endl;                                       \
  }

#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
#define HYPE_FATAL_ERROR(X, OUTSTREAM)                                  \
  {                                                                     \
    std::cout << "FATAL ERROR: " << X << ": In " << __PRETTY_FUNCTION__ \
              << std::endl;                                             \
    OUTSTREAM << "In File: " << __FILE__ << " Line: " << __LINE__       \
              << std::endl;                                             \
    OUTSTREAM << "Aborting Execution..." << std::endl;                  \
    CoGaDB::exit(-1);                                                   \
  }
#else
#define HYPE_FATAL_ERROR(X, OUTSTREAM)                                  \
  {                                                                     \
    std::cout << "FATAL ERROR: " << X << ": In " << __PRETTY_FUNCTION__ \
              << std::endl;                                             \
    OUTSTREAM << "In File: " << __FILE__ << " Line: " << __LINE__       \
              << std::endl;                                             \
    OUTSTREAM << "Aborting Execution..." << std::endl;                  \
    std::exit(-1);                                                      \
  }
#endif

#endif

  // enum
  // ProcessingDevice{PD0,PD1,PD2,PD3,PD4,PD5,PD6,PD7,PD8,PD9,PD10,PD11,PD12,PD13,PD14,PD15,PD16,PD17,PD18,PD19,PD20,PD21,PD22,PD23,PD24,PD25,PD26,PD27,PD28,PD29,PD30,PD31,PD32,PD33,PD34,PD35,PD36,PD37,PD38,PD39,PD40,PD41,PD42,PD43,PD44,PD45,PD46,PD47,PD48,PD49,PD50,PD51,PD52,PD53,PD54,PD55,PD56,PD57,PD58,PD59,PD60,PD61,PD62,PD63,PD64,PD65,PD66,PD67,PD68,PD69,PD70,PD71,PD72,PD73,PD74,PD75,PD76,PD77,PD78,PD79,PD80,PD81,PD82,PD83,PD84,PD85,PD86,PD87,PD88,PD89,PD90,PD91,PD92,PD93,PD94,PD95,PD96,PD97,PD98,PD99,PD100};

  // enum
  // ProcessingDeviceMemory{PD_Memory_0,PD_Memory_1,PD_Memory_2,PD_Memory_3,PD_Memory_4,PD_Memory_5,PD_Memory_6,PD_Memory_7,PD_Memory_8,PD_Memory_9,PD_Memory_10,PD_Memory_11,PD_Memory_12,PD_Memory_13,PD_Memory_14,PD_Memory_15,PD_Memory_16,PD_Memory_17,PD_Memory_18,PD_Memory_19,PD_Memory_20,PD_Memory_21,PD_Memory_22,PD_Memory_23,PD_Memory_24,PD_Memory_25,PD_Memory_26,PD_Memory_27,PD_Memory_28,PD_Memory_29,PD_Memory_30,PD_Memory_31,PD_Memory_32,PD_Memory_33,PD_Memory_34,PD_Memory_35,PD_Memory_36,PD_Memory_37,PD_Memory_38,PD_Memory_39,PD_Memory_40,PD_Memory_41,PD_Memory_42,PD_Memory_43,PD_Memory_44,PD_Memory_45,PD_Memory_46,PD_Memory_47,PD_Memory_48,PD_Memory_49,PD_Memory_50,PD_Memory_51,PD_Memory_52,PD_Memory_53,PD_Memory_54,PD_Memory_55,PD_Memory_56,PD_Memory_57,PD_Memory_58,PD_Memory_59,PD_Memory_60,PD_Memory_61,PD_Memory_62,PD_Memory_63,PD_Memory_64,PD_Memory_65,PD_Memory_66,PD_Memory_67,PD_Memory_68,PD_Memory_69,PD_Memory_70,PD_Memory_71,PD_Memory_72,PD_Memory_73,PD_Memory_74,PD_Memory_75,PD_Memory_76,PD_Memory_77,PD_Memory_78,PD_Memory_79,PD_Memory_80,PD_Memory_81,PD_Memory_82,PD_Memory_83,PD_Memory_84,PD_Memory_85,PD_Memory_86,PD_Memory_87,PD_Memory_88,PD_Memory_89,PD_Memory_90,PD_Memory_91,PD_Memory_92,PD_Memory_93,PD_Memory_94,PD_Memory_95,PD_Memory_96,PD_Memory_97,PD_Memory_98,PD_Memory_99,PD_Memory_100};

  //	typedef enum {PD0,PD1,PD2,PD3,PD4,PD5,PD6,PD7,PD8,PD9,PD10}
  // ProcessingDeviceID;

  // PD_DMA_0 copies CPU -> CP, whereas PD_DMA_1 copies CP -> CPU
  // typedef enum
  // {PD0,PD1,PD2,PD3,PD4,PD5,PD6,PD7,PD8,PD9,PD10,PD11,PD12,PD13,PD14,PD15,PD16,PD17,PD18,PD19,PD20,PD_DMA0,PD_DMA1}
  // ProcessingDeviceID;
  typedef enum {
    PD0,
    PD1,
    PD2,
    PD3,
    PD4,
    PD5,
    PD6,
    PD7,
    PD8,
    PD9,
    PD10,
    PD11,
    PD12,
    PD13,
    PD14,
    PD15,
    PD16,
    PD17,
    PD18,
    PD19,
    PD20,
    PD21,
    PD22,
    PD23,
    PD24,
    PD25,
    PD26,
    PD27,
    PD28,
    PD29,
    PD30,
    PD31,
    PD32,
    PD33,
    PD34,
    PD35,
    PD36,
    PD37,
    PD38,
    PD39,
    PD40,
    PD41,
    PD42,
    PD43,
    PD44,
    PD45,
    PD46,
    PD47,
    PD48,
    PD49,
    PD50,
    PD51,
    PD52,
    PD53,
    PD54,
    PD55,
    PD56,
    PD57,
    PD58,
    PD59,
    PD60,
    PD61,
    PD62,
    PD63,
    PD64,
    PD65,
    PD66,
    PD67,
    PD68,
    PD69,
    PD70,
    PD71,
    PD72,
    PD73,
    PD74,
    PD75,
    PD76,
    PD77,
    PD78,
    PD79,
    PD80,
    PD81,
    PD82,
    PD83,
    PD84,
    PD85,
    PD86,
    PD87,
    PD88,
    PD89,
    PD90,
    PD91,
    PD92,
    PD93,
    PD94,
    PD95,
    PD96,
    PD97,
    PD98,
    PD99,
    PD100,
    PD_DMA0,
    PD_DMA1
  } ProcessingDeviceID;

  typedef enum {
    PD_Memory_0,
    PD_Memory_1,
    PD_Memory_2,
    PD_Memory_3,
    PD_Memory_4,
    PD_Memory_5,
    PD_Memory_6,
    PD_Memory_7,
    PD_Memory_8,
    PD_Memory_9,
    PD_Memory_10,
    PD_Memory_11,
    PD_Memory_12,
    PD_Memory_13,
    PD_Memory_14,
    PD_Memory_15,
    PD_Memory_16,
    PD_Memory_17,
    PD_Memory_18,
    PD_Memory_19,
    PD_Memory_20,
    PD_NO_Memory
  } ProcessingDeviceMemoryID;

  typedef enum {
    CPU,
    GPU,
    FPGA,
    NP,
    XEON_PHI,
    DMA
  } ProcessingDeviceType;  // CPU,GPU,FPGA, Network Processor

  // enum DeviceTypeConstraint{ANY_DEVICE,CPU_ONLY,GPU_ONLY,FPGA_ONLY,NP_ONLY};
  // //ALL,CPU,GPU,FPGA, Network Processor

  typedef enum {
    ANY_DEVICE,
    CPU_ONLY,
    GPU_ONLY,
    FPGA_ONLY,
    NP_ONLY,
    XEON_PHI_ONLY
  } DeviceTypeConstraint;  // ALL,CPU,GPU,FPGA, Network Processor

  typedef enum {
    NO_COPY,
    COPY_CPU_TO_GPU,
    COPY_GPU_TO_CPU,
    COPY_CPU_TO_GPU_TO_CPU
  } CopyDirection;

  typedef enum { Architecture_32Bit, Architecture_64Bit } Architecture;

  // typedef enum {CPU_ONLY,GPU_ONLY,HYBRID} SchedulingConfiguration;

  // struct StatisticalMethods{
  typedef enum {
    Least_Squares_1D,
    Multilinear_Fitting_2D,
    KNN_Regression
  } StatisticalMethod;
  //};

  typedef enum {
    No_Recomputation,
    Periodic  //,
    // RelativeErrorBased
  } RecomputationHeuristic;

  typedef enum {
    ResponseTime,
    WaitingTimeAwareResponseTime,
    Throughput,
    Simple_Round_Robin,
    ProbabilityBasedOutsourcing,
    Throughput2
  } OptimizationCriterion;

  typedef enum {
    GREEDY_HEURISTIC,    // Use Greedy Heuristic from Bress et al. 2012 (ADBIS
                         // 2012)
    BACKTRACKING,        // consider all possible plans
    TWO_COPY_HEURISTIC,  // explores a limited optimization space according to
                         // Bress et al. 2012 (Control and Cybernetics Journal)
    CPU_CP_SEQUENCE_ALLOCATION,     // creates a plan for each (co-)processor
    INTERACTIVE_USER_OPTIMIZATION,  // allows user to interactively optimize
                                    // query plans
    GREEDY_CHAINER_HEURISTIC,  // works as greedy heuristic, but tries to avoid
                               // copy operations by chaining operators on the
                               // same device type
    CRITICAL_PATH_HEURISTIC,  // identifies the critical path and accelerates it
                              // on co-processor
    BEST_EFFORT_GPU_HEURISTIC  // performs all operators that have a GPU
                               // operator on GPU, if an operator aborts, fall
                               // back to CPU
  } QueryOptimizationHeuristic;

//#ifdef __cplusplus
// struct StatisticalMethods{typedef hype::StatisticalMethod
// StatisticalMethod;};
// struct RecomputationHeuristics{typedef hype::RecomputationHeuristic
// RecomputationHeuristic;};
// struct OptimizationCriterions{typedef hype::OptimizationCriterion
// OptimizationCriterion;};

//#endif

#ifdef __cplusplus
  namespace core {
#endif

    typedef enum {
      quiet = 1,
      verbose = 0,
      debug = 0,
      print_time_measurement = 0
    } DebugMode;  //*/

#ifdef __cplusplus
  }  // end namespace core

  namespace queryprocessing {
    enum SchedulingConfiguration { CPU_ONLY, GPU_ONLY, HYBRID };
  }
}  // end namespace hype
#endif
