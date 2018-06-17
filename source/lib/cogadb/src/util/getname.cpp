

#include <util/getname.hpp>

namespace CoGaDB {

namespace util {

const std::string getName(AttributeType x) {
  const char* const names[] = {"INT", "FLOAT",  "VARCHAR", "BOOLEAN", "UINT32",
                               "OID", "DOUBLE", "CHAR",    "DATE"};

  return std::string(names[x]);
}

const std::string getName(ColumnType x) {
  const char* const names[] = {"PLAIN_MATERIALIZED",
                               "LOOKUP_ARRAY",
                               "DICTIONARY_COMPRESSED",
                               "RUN_LENGTH_COMPRESSED",
                               "DELTA_COMPRESSED",
                               "BIT_VECTOR_COMPRESSED",
                               "BITPACKED_DICTIONARY_COMPRESSED",
                               "RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER",
                               "VOID_COMPRESSED_NUMBER",
                               "REFERENCE_BASED_COMPRESSED",
                               "RUN_LENGTH_COMPRESSED_PREFIX",
                               "RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER_PREFIX",
                               "DICTIONARY_COMPRESSED_ORDER_PRESERVING"};

  return std::string(names[x]);
}

const std::string getName(ComputeDevice x) {
  const char* const names[] = {"CPU", "GPU"};

  return std::string(names[x]);
}

const std::string getName(AggregationMethod x) {
  const char* const names[] = {"COUNT",
                               "SUM",
                               "MIN",
                               "MAX",
                               "AVERAGE",
                               "AGG_GENOTYPE",
                               "AGG_CONCAT_BASES",
                               "AGG_IS_HOMOPOLYMER",
                               "AGG_GENOTYPE_STATISTICS",
                               "UDF_AGGREGATION"};

  return std::string(names[x]);
}

const std::string getName(AggregationAlgorithm x) {
  const char* const names[] = {"SORT_BASED_AGGREGATION",
                               "HASH_BASED_AGGREGATION"};
  return std::string(names[x]);
}

const std::string getName(ValueComparator x) {
  const char* const names[] = {"<", ">", "=", "<=", ">=", "<>"};

  return std::string(names[x]);
}

const std::string getName(SortOrder x) {
  const char* const names[] = {"ASCENDING", "DESCENDING"};

  return std::string(names[x]);
}

const std::string getName(Operation x) {
  const char* const names[] = {"SELECTION",   "PROJECTION", "JOIN",
                               "GROUPBY",     "SORT",       "COPY",
                               "AGGREGATION", "FULL_SCAN",  "INDEX_SCAN"};

  return std::string(names[x]);
}

const std::string getName(JoinAlgorithm x) {
  const char* const names[] = {"SORT_MERGE_JOIN", "NESTED_LOOP_JOIN",
                               "HASH_JOIN"};

  return std::string(names[x]);
}

const std::string getName(JoinType x) {
  const char* const names[] = {"JOIN",
                               "LEFT_SEMI_JOIN",
                               "RIGHT_SEMI_JOIN",
                               "LEFT_ANTI_SEMI_JOIN",
                               "RIGHT_ANTI_SEMI_JOIN",
                               "LEFT_OUTER_JOIN",
                               "RIGHT_OUTER_JOIN",
                               "FULL_OUTER_JOIN",
                               "GATHER_JOIN"};

  return std::string(names[x]);
}

const std::string getName(MaterializationStatus x) {
  const char* const names[] = {"MATERIALIZE", "LOOKUP"};

  return std::string(names[x]);
}

const std::string getName(ParallelizationMode x) {
  const char* const names[] = {"SERIAL", "PARALLEL"};

  return std::string(names[x]);
}

const std::string getName(ColumnAlgebraOperation x) {
  const char* const names[] = {"ADD", "SUB", "MUL", "DIV"};
  return std::string(names[x]);
}

const std::string getName(PositionListOperation x) {
  const char* const names[] = {"POSITIONLIST_INTERSECTION",
                               "POSITIONLIST_UNION"};
  return std::string(names[x]);
}

const std::string getName(BitmapOperation x) {
  const char* const names[] = {"BITMAP_AND", "BITMAP_OR"};
  return std::string(names[x]);
}

const std::string getName(GPUBufferManagementStrategy x) {
  const char* const names[] = {"LEAST_RECENTLY_USED", "LEAST_FREQUENTLY_USED",
                               "DISABLED_GPU_BUFFER"};
  return std::string(names[x]);
}

const std::string getName(TableLoaderMode x) {
  const char* const names[] = {"LOAD_ALL_COLUMNS", "LOAD_NO_COLUMNS",
                               "LOAD_ELEMENTARY_COLUMNS"};
  return std::string(names[x]);
}

const std::string getName(LogicalOperation x) {
  const char* const names[] = {"AND", "OR"};
  return std::string(names[x]);
}

const std::string getName(SetOperation x) {
  const char* const names[] = {"UNION",         "UNION_ALL", "INTERSECT",
                               "INTERSECT_ALL", "EXCEPT",    "EXCEPT_ALL"};
  return std::string(names[x]);
}
}
}  // end namespace CogaDB
