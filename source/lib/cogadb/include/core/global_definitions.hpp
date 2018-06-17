
#pragma once

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <inttypes.h>
#include <boost/any.hpp>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <vector>

#include <boost/shared_ptr.hpp>

// Compile Time configuration
//#define ENABLE_PARALLEL_SELECTION
//#define ENABLE_GPU_ACCELERATION
#define ENABLE_TWO_PHASE_PHYSICAL_OPTIMIZATION

#define COGADB_USE_KNN_REGRESSION_LEARNER
//#define COGADB_USE_INVISIBLE_JON_PLANS_ONLY

//#define VALIDATE_GPU_MEMORY_COST_MODELS

#ifndef __APPLE__
#define COGADB_USE_INTEL_PERFORMANCE_COUNTER
#endif
//#define COGADB_VALIDATE_GPU_PREFIX_SUM

//#define PRINT_QUERY_STARTINGTIME
//#define FORCE_USE_UNSTABLE_SORT_FOR_BENCHMARK

// enables use of CDK primitives
#define ENABLE_CDK_USAGE

#define ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//#define ENABLE_GPU_JOIN
#define ENABLE_GPU_PK_FK_JOIN
#define ENABLE_GPU_FETCH_JOIN
#ifndef HYPE_ENABLE_PARALLEL_QUERY_PLAN_EVALUATION
#define HYPE_ENABLE_PARALLEL_QUERY_PLAN_EVALUATION
#endif

namespace CoGaDB {
  void printStackTrace(std::ostream& out);
  void exit(int status) __attribute__((noreturn));
}

#define COGADB_WARNING(X, Y)                                        \
  {                                                                 \
    std::cout << "WARNING: " << X << ": In " << __PRETTY_FUNCTION__ \
              << std::endl;                                         \
    std::cout << "In File: " << __FILE__ << " Line: " << __LINE__   \
              << std::endl;                                         \
  }

#define COGADB_ERROR(X, Y)                                        \
  {                                                               \
    std::cout << "ERROR: " << X << ": In " << __PRETTY_FUNCTION__ \
              << std::endl;                                       \
    std::cout << "In File: " << __FILE__ << " Line: " << __LINE__ \
              << std::endl;                                       \
    CoGaDB::printStackTrace(std::cout);                           \
  }

#define COGADB_FATAL_ERROR(X, Y)                                        \
  {                                                                     \
    std::cout << "FATAL ERROR: " << X << ": In " << __PRETTY_FUNCTION__ \
              << std::endl;                                             \
    std::cout << "In File: " << __FILE__ << " Line: " << __LINE__       \
              << std::endl;                                             \
    std::cout << "Aborting Execution..." << std::endl;                  \
    CoGaDB::printStackTrace(std::cout);                                 \
    CoGaDB::exit(-1);                                                   \
  }

#define COGADB_NOT_IMPLEMENTED COGADB_FATAL_ERROR("Not Implemented!", "")

//#define TID_NUMBER_OF_BITS 32
#define TID_NUMBER_OF_BITS 64

#if TID_NUMBER_OF_BITS == 32
typedef uint32_t TID;
#define COGADB_USE_32_BIT_TIDS
#else
typedef uint64_t TID;
#endif

typedef char* C_String;

#define COGADB_INSTANTIATE_TEMPLATE_FOR_SUPPORTED_TYPES(TEMPLATE, TYPE) \
  template TYPE TEMPLATE<int>;                                          \
  template TYPE TEMPLATE<float>;                                        \
  template TYPE TEMPLATE<double>;                                       \
  template TYPE TEMPLATE<uint64_t>;                                     \
  template TYPE TEMPLATE<uint32_t>;                                     \
  template TYPE TEMPLATE<char>;                                         \
  template TYPE TEMPLATE<std::string>;                                  \
  template TYPE TEMPLATE<char*>;

#define COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(TEMPLATE) \
  COGADB_INSTANTIATE_TEMPLATE_FOR_SUPPORTED_TYPES(TEMPLATE, class)

#define COGADB_INSTANTIATE_STRUCT_TEMPLATE_FOR_SUPPORTED_TYPES(TEMPLATE) \
  COGADB_INSTANTIATE_TEMPLATE_FOR_SUPPORTED_TYPES(TEMPLATE, struct)

namespace shared_pointer_namespace = boost;  // std::tr1

namespace CoGaDB {

  enum AttributeType {
    INT,
    FLOAT,
    VARCHAR,
    BOOLEAN,
    UINT32,
    OID,
    DOUBLE,
    CHAR,
    DATE
  };

  enum ColumnType {
    PLAIN_MATERIALIZED,
    LOOKUP_ARRAY,
    DICTIONARY_COMPRESSED,
    RUN_LENGTH_COMPRESSED,
    DELTA_COMPRESSED,
    BIT_VECTOR_COMPRESSED,
    BITPACKED_DICTIONARY_COMPRESSED,
    RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER,
    VOID_COMPRESSED_NUMBER,
    REFERENCE_BASED_COMPRESSED,
    RUN_LENGTH_COMPRESSED_PREFIX,
    RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER_PREFIX,
    DICTIONARY_COMPRESSED_ORDER_PRESERVING
  };

  enum TableLoaderMode {
    LOAD_ALL_COLUMNS,
    LOAD_NO_COLUMNS,
    LOAD_ELEMENTARY_COLUMNS
  };

  enum ColumnLoaderMode { LOAD_META_DATA_ONLY, LOAD_ALL_DATA };

  enum ComputeDevice { CPU, GPU };

  enum AggregationMethod {
    COUNT,
    SUM,
    MIN,
    MAX,
    AVERAGE,
    AGG_GENOTYPE,
    AGG_CONCAT_BASES,
    AGG_IS_HOMOPOLYMER,
    AGG_GENOTYPE_STATISTICS,
    UDF_AGGREGATION
  };

  enum AggregationAlgorithm { SORT_BASED_AGGREGATION, HASH_BASED_AGGREGATION };

  enum PredicationMode { BRANCHED_EXECUTION, PREDICATED_EXECUTION };

  enum ValueComparator {
    LESSER,
    GREATER,
    EQUAL,
    LESSER_EQUAL,
    GREATER_EQUAL,
    UNEQUAL
  };

  enum PredicateType {
    ValueValuePredicate,
    ValueConstantPredicate,
    ValueRegularExpressionPredicate
  };

  enum SortOrder { ASCENDING, DESCENDING };

  enum Operation {
    SELECTION,
    PROJECTION,
    JOIN,
    GROUPBY,
    SORT,
    COPY,
    AGGREGATION,
    FULL_SCAN,
    INDEX_SCAN
  };

  //    enum JoinAlgorithm {
  //        SORT_MERGE_JOIN, SORT_MERGE_JOIN_2, NESTED_LOOP_JOIN, HASH_JOIN,
  //        PARALLEL_HASH_JOIN, RADIX_JOIN
  //    };
  enum JoinAlgorithm {
    SORT_MERGE_JOIN,
    SORT_MERGE_JOIN_2,
    NESTED_LOOP_JOIN,
    HASH_JOIN,
    PARALLEL_HASH_JOIN,
    RADIX_JOIN,
    INDEX_NESTED_LOOP_JOIN
  };

  enum JoinType {
    INNER_JOIN,
    LEFT_SEMI_JOIN,
    RIGHT_SEMI_JOIN,
    LEFT_ANTI_SEMI_JOIN,
    RIGHT_ANTI_SEMI_JOIN,
    LEFT_OUTER_JOIN,
    RIGHT_OUTER_JOIN,
    FULL_OUTER_JOIN,
    GATHER_JOIN
  };

  enum MaterializationStatus { MATERIALIZE, LOOKUP };

  enum ParallelizationMode { SERIAL, PARALLEL };

  enum ColumnAlgebraOperation { ADD, SUB, MUL, DIV };

  enum SetOperation {
    UNION,
    UNION_ALL,
    INTERSECT,
    INTERSECT_ALL,
    EXCEPT,
    EXCEPT_ALL
  };

  enum PositionListOperation { POSITIONLIST_INTERSECTION, POSITIONLIST_UNION };

  enum BitmapOperation { BITMAP_AND, BITMAP_OR };

  enum BitShiftOperation { SHIFT_BITS_LEFT, SHIFT_BITS_RIGHT };

  enum BitwiseCombinationOperation { BITWISE_AND, BITWISE_OR };

  enum GPUBufferManagementStrategy {
    LEAST_RECENTLY_USED,
    LEAST_FREQUENTLY_USED,
    DISABLED_GPU_BUFFER
  };

  typedef GPUBufferManagementStrategy BufferManagementStrategy;

  enum SelectivityEstimationStrategy {
    NO_SELECTIVITY_ESTIMATION,
    EQUI_HEIGHT_HISTOGRAM
  };

  /* Enum that steers how the result is written in query compiler */
  enum PipelineEndType {
    MATERIALIZE_FROM_ARRAY_TO_ARRAY,
    MATERIALIZE_FROM_ARRAY_TO_JOIN_HASH_TABLE_AND_ARRAY,
    MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY
  };

  enum LogicalOperation { LOGICAL_AND, LOGICAL_OR };

  enum DebugMode {
    quiet = 1,
    verbose = 0,
    debug = 0,
    print_time_measurement = 0
  };

  //    enum DebugMode {
  //        quiet = 0,
  //        verbose = 1,
  //        debug = 1,
  //        print_time_measurement = 0
  //    };

  /*
  enum DebugMode{quiet=0,
                                          verbose=1,
                                          debug=0,
                                          print_time_measurement=0};
   */

  typedef std::pair<TID, TID> TID_Pair;

  typedef std::pair<AttributeType, std::string> Attribut;

  typedef std::list<Attribut> TableSchema;

  typedef std::pair<std::string, ColumnType> CompressionSpecification;
  typedef std::map<std::string, ColumnType> CompressionSpecifications;

  typedef std::vector<boost::any> Tuple;

  // method plus resulting aggregation column name
  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;

  // for Rename Operator
  typedef std::pair<std::string, std::string> RenameEntry;
  typedef std::list<RenameEntry> RenameList;

  // a SortAttribute determines a specific SortOrder for a specific attribute
  // name
  typedef std::pair<std::string, SortOrder> SortAttribute;
  typedef std::list<SortAttribute> SortAttributeList;

  // struct Attribut {

  //	AttributeType type_;
  //	std::string name_;
  //	ColumnPtr column_;

  //	AttributeType& first;
  //	std::string& second;

  //}
}  // end namespace CogaDB
