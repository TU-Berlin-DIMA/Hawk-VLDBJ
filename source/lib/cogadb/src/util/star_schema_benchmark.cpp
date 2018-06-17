
#include <boost/tokenizer.hpp>
#include <core/table.hpp>
#include <parser/commandline_interpreter.hpp>
#include <persistence/storage_manager.hpp>
#include <util/star_schema_benchmark.hpp>

#include <core/runtime_configuration.hpp>
#include <unittests/unittests.hpp>
#include <util/star_schema_benchmark.hpp>
#include <util/tpch_benchmark.hpp>

#include <query_processing/query_processor.hpp>

#include <boost/lexical_cast.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/make_shared.hpp>
#include <boost/random.hpp>

#include <algorithm>
#include <string>
#include <utility>

#include <parser/generated/Parser.h>
#include <query_processing/query_processor.hpp>
#include <sql/server/sql_driver.hpp>
#include <statistics/statistics_manager.hpp>
#include <util/statistics.hpp>

//#define COGADB_MEASURE_ENERGY_CONSUMPTION
#ifdef COGADB_MEASURE_ENERGY_CONSUMPTION
#include <IntelPerformanceCounterMonitorV2.7/cpucounters.h>
#endif

using namespace std;

#define USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
#define USE_FETCH_JOIN
//#define USE_ORDER_PRESERVING_DICTIONARY_COMPRESSION

namespace CoGaDB {
using namespace query_processing;

bool import_table_dates(const std::string& path_to_files, ClientPtr client) {
  std::cout << "Create Table 'DATES' ..." << std::endl;

  TableSchema schema;
  schema.push_back(Attribut(OID, "D_DATEKEY"));  // primary key
  schema.push_back(Attribut(VARCHAR, "D_DATE"));
  schema.push_back(Attribut(
      VARCHAR,
      "D_DAYOFWEEK"));  // fixed text, size 8, Sunday, Monday, ..., Saturday)
  schema.push_back(Attribut(
      VARCHAR, "D_MONTH"));  //  fixed text, size 9: January, ..., December
  schema.push_back(Attribut(INT, "D_YEAR"));  //  unique value 1992-1998
  schema.push_back(
      Attribut(INT, "D_YEARMONTHNUM"));  //  numeric (YYYYMM) -- e.g. 199803
  schema.push_back(Attribut(
      VARCHAR, "D_YEARMONTH"));  //  fixed text, size 7: Mar1998 for example
  schema.push_back(Attribut(INT, "D_DAYNUMINWEEK"));    //  numeric 1-7
  schema.push_back(Attribut(INT, "D_DAYNUMINMONTH"));   //  numeric 1-31
  schema.push_back(Attribut(INT, "D_DAYNUMINYEAR"));    //  numeric 1-366
  schema.push_back(Attribut(INT, "D_MONTHNUMINYEAR"));  //  numeric 1-12
  schema.push_back(Attribut(INT, "D_WEEKNUMINYEAR"));   //  numeric 1-53
  schema.push_back(Attribut(
      VARCHAR, "D_SELLINGSEASON"));  //  text, size 12 (Christmas, Summer,...)
  schema.push_back(Attribut(INT, "D_LASTDAYINWEEKFL"));   //  1 bit
  schema.push_back(Attribut(INT, "D_LASTDAYINMONTHFL"));  //  1 bit
  schema.push_back(Attribut(INT, "D_HOLIDAYFL"));         //  1 bit
  schema.push_back(Attribut(INT, "D_WEEKDAYFL"));         //  1 bit

  TablePtr tab(new Table("DATES", schema));

  if (tab->loadDatafromFile(path_to_files + "date.tbl")) {
    std::cout << "Store Table 'DATES' ..." << std::endl;
    if (!tab->setPrimaryKeyConstraint("DATES.D_DATEKEY")) {
      COGADB_ERROR("Failed to set Primary Key Constraint!", "");
      return false;
    }
    storeTable(tab);
    addToGlobalTableList(tab);
    return true;
  } else {
    return false;
  }
}

bool import_lineorder(const std::string& path_to_files, ClientPtr client) {
  std::cout << "Create Table 'LINEORDER' ..." << std::endl;
  Timestamp begin = getTimestamp();

  TableSchema schema;
  schema.push_back(Attribut(OID, "LO_ORDERKEY"));    //"L_ORDERKEY");
  schema.push_back(Attribut(INT, "LO_LINENUMBER"));  // l_linenumber);
  schema.push_back(Attribut(INT, "LO_CUSTKEY"));
  schema.push_back(Attribut(INT, "LO_PARTKEY"));
  schema.push_back(Attribut(INT, "LO_SUPPKEY"));
  schema.push_back(Attribut(INT, "LO_ORDERDATE"));
  schema.push_back(Attribut(VARCHAR, "LO_ORDERPRIORITY"));
  schema.push_back(Attribut(INT, "LO_SHIPPRIORITY"));
  schema.push_back(Attribut(INT, "LO_QUANTITY"));
  schema.push_back(Attribut(FLOAT, "LO_EXTENDEDPRICE"));
  schema.push_back(Attribut(FLOAT, "LO_ORDTOTALPRICE"));
  schema.push_back(Attribut(FLOAT, "LO_DISCOUNT"));      // l_quantity );
  schema.push_back(Attribut(FLOAT, "LO_REVENUE"));       // l_extendedprice);
  schema.push_back(Attribut(FLOAT, "LO_SUPPLYCOST"));    // l_discount);
  schema.push_back(Attribut(FLOAT, "LO_TAX"));           // l_tax);
  schema.push_back(Attribut(VARCHAR, "LO_COMMITDATE"));  // l_commitdate);
  schema.push_back(Attribut(VARCHAR, "LO_SHIPMODE"));    // l_shipmode);

  CompressionSpecifications compr_specs;
  //        PLAIN_MATERIALIZED, LOOKUP_ARRAY, DICTIONARY_COMPRESSED,
  //        RUN_LENGTH_COMPRESSED, DELTA_COMPRESSED, BIT_VECTOR_COMPRESSED,
  //        BITPACKED_DICTIONARY_COMPRESSED,
  //        RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER,
  //        VOID_COMPRESSED_NUMBER, REFERENCE_BASED_COMPRESSED
  compr_specs["LO_ORDERPRIORITY"] = BITPACKED_DICTIONARY_COMPRESSED;
  compr_specs["LO_TAX"] = BITPACKED_DICTIONARY_COMPRESSED;
  compr_specs["LO_LINENUMBER"] = RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER;
  compr_specs["LO_QUANTITY"] = BITPACKED_DICTIONARY_COMPRESSED;

  TablePtr tab(new Table("LINEORDER", schema, compr_specs));

  if (tab->loadDatafromFile(path_to_files + "lineorder.tbl")) {
    Timestamp end = getTimestamp();
    assert(end >= begin);
    cout << "Needed " << double(end - begin) / (1000 * 1000 * 1000)
         << "s for import..." << endl;
    // tab->print();
    cout << "Store Table 'LINEORDER' ..." << endl;
    if (!tab->setForeignKeyConstraint("LINEORDER.LO_CUSTKEY",
                                      "CUSTOMER.C_CUSTKEY", "CUSTOMER")) {
      COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
      return false;
    }
    if (!tab->setForeignKeyConstraint("LINEORDER.LO_PARTKEY", "PART.P_PARTKEY",
                                      "PART")) {
      COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
      return false;
    }
    if (!tab->setForeignKeyConstraint("LINEORDER.LO_SUPPKEY",
                                      "SUPPLIER.S_SUPPKEY", "SUPPLIER")) {
      COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
      return false;
    }
    if (!tab->setForeignKeyConstraint("LINEORDER.LO_ORDERDATE",
                                      "DATES.D_DATEKEY", "DATES")) {
      COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
      return false;
    }
    storeTable(tab);
    addToGlobalTableList(tab);
    //    LO_CustKey bigint not null,
    //    LO_PartKey bigint not null,
    //    LO_SuppKey int not null,
    //    LO_OrderDateKey bigint not null,
    return true;
  } else {
    return false;
  }
  return true;
}

bool Unittest_Create_Star_Schema_Benchmark_Database(
    const std::string& path_to_files, ClientPtr client) {
  // DATE Table Layout (7 years of days: 7366 days)
  // DATEKEY identifier, unique id -- e.g. 19980327 (what we use)
  // DATE fixed text, size 18, longest: December 22, 1998
  // DAYOFWEEK fixed text, size 8, Sunday, Monday, ..., Saturday)
  // MONTH fixed text, size 9: January, ..., December
  // YEAR unique value 1992-1998
  // YEARMONTHNUM numeric (YYYYMM) -- e.g. 199803
  // YEARMONTH fixed text, size 7: Mar1998 for example
  // DAYNUMINWEEK numeric 1-7
  // DAYNUMINMONTH numeric 1-31
  // DAYNUMINYEAR numeric 1-366
  // MONTHNUMINYEAR numeric 1-12
  // WEEKNUMINYEAR numeric 1-53
  // SELLINGSEASON text, size 12 (Christmas, Summer,...)
  // LASTDAYINWEEKFL 1 bit
  // LASTDAYINMONTHFL 1 bit
  // HOLIDAYFL 1 bit
  // WEEKDAYFL 1 bit
  // Primary Key: DATEKEY

  {
    cout << "Create Table 'DATES' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(OID, "D_DATEKEY"));  // primary key
    schema.push_back(Attribut(VARCHAR, "D_DATE"));
    schema.push_back(Attribut(
        VARCHAR,
        "D_DAYOFWEEK"));  // fixed text, size 8, Sunday, Monday, ..., Saturday)
    schema.push_back(Attribut(
        VARCHAR, "D_MONTH"));  //  fixed text, size 9: January, ..., December
    schema.push_back(Attribut(INT, "D_YEAR"));  //  unique value 1992-1998
    schema.push_back(
        Attribut(INT, "D_YEARMONTHNUM"));  //  numeric (YYYYMM) -- e.g. 199803
    schema.push_back(Attribut(
        VARCHAR, "D_YEARMONTH"));  //  fixed text, size 7: Mar1998 for example
    schema.push_back(Attribut(INT, "D_DAYNUMINWEEK"));    //  numeric 1-7
    schema.push_back(Attribut(INT, "D_DAYNUMINMONTH"));   //  numeric 1-31
    schema.push_back(Attribut(INT, "D_DAYNUMINYEAR"));    //  numeric 1-366
    schema.push_back(Attribut(INT, "D_MONTHNUMINYEAR"));  //  numeric 1-12
    schema.push_back(Attribut(INT, "D_WEEKNUMINYEAR"));   //  numeric 1-53
    schema.push_back(Attribut(
        VARCHAR, "D_SELLINGSEASON"));  //  text, size 12 (Christmas, Summer,...)
    schema.push_back(Attribut(INT, "D_LASTDAYINWEEKFL"));   //  1 bit
    schema.push_back(Attribut(INT, "D_LASTDAYINMONTHFL"));  //  1 bit
    schema.push_back(Attribut(INT, "D_HOLIDAYFL"));         //  1 bit
    schema.push_back(Attribut(INT, "D_WEEKDAYFL"));         //  1 bit

    CompressionSpecifications compression_specs;
#ifdef USE_ORDER_PRESERVING_DICTIONARY_COMPRESSION
    compression_specs.insert(CompressionSpecification(
        "DATES.D_DATE", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "DATES.D_DAYOFWEEK", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "DATES.D_MONTH", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "DATES.D_YEARMONTH", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "DATES.D_SELLINGSEASON", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
#endif

    TablePtr tab(new Table("DATES", schema, compression_specs));

    if (tab->loadDatafromFile(path_to_files + "date.tbl")) {
      cout << "Store Table 'DATES' ..." << endl;
      if (!tab->setPrimaryKeyConstraint("DATES.D_DATEKEY")) {
        COGADB_ERROR("Failed to set Primary Key Constraint!", "");
        return false;
      }
      storeTable(tab);
      addToGlobalTableList(tab);
    }
  }

  // PART Table Layout (200,000*1+log2SF populated)
  // PARTKEY identifier
  // NAME variable text, size 22 (Not unique per PART but never was)
  // MFGR fixed text, size 6 (MFGR#1-5, CARD = 5)
  // CATEGORY fixed text, size 7 ('MFGR#'||1-5||1-5: CARD = 25)
  // BRAND1 fixed text, size 9 (CATEGORY||1-40: CARD = 1000)
  // COLOR variable text, size 11 (CARD = 94)
  // TYPE variable text, size 25 (CARD = 150)
  // SIZE numeric 1-50 (CARD = 50)
  // CONTAINER fixed text(10) (CARD = 40)
  // Primary Key: PARTKEY

  {
    cout << "Create Table 'PART' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(OID, "P_PARTKEY"));  // primary key
    schema.push_back(Attribut(VARCHAR, "P_NAME"));
    schema.push_back(Attribut(VARCHAR, "P_MFGR"));
    schema.push_back(Attribut(VARCHAR, "P_CATEGORY"));
    schema.push_back(Attribut(VARCHAR, "P_BRAND"));
    schema.push_back(Attribut(VARCHAR, "P_COLOR"));
    schema.push_back(Attribut(VARCHAR, "P_TYPE"));
    schema.push_back(Attribut(INT, "P_SIZE"));
    schema.push_back(Attribut(VARCHAR, "P_CONTAINER"));

    CompressionSpecifications compression_specs;
#ifdef USE_ORDER_PRESERVING_DICTIONARY_COMPRESSION
    compression_specs.insert(CompressionSpecification(
        "PART.P_NAME", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "PART.P_MFGR", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "PART.P_CATEGORY", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "PART.P_BRAND", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "PART.P_COLOR", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "PART.P_TYPE", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "PART.P_CONTAINER", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
#endif

    TablePtr tab(new Table("PART", schema, compression_specs));

    if (tab->loadDatafromFile(path_to_files + "part.tbl")) {
      cout << "Store Table 'PART' ..." << endl;
      if (!tab->setPrimaryKeyConstraint("PART.P_PARTKEY")) {
        COGADB_ERROR("Failed to set Primary Key Constraint!", "");
        return false;
      }
      storeTable(tab);
      addToGlobalTableList(tab);
    }
  }

  // SUPPLIER Table Layout (SF*10,000 are populated)
  // SUPPKEY identifier
  // NAME fixed text, size 25: 'Supplier'||SUPPKEY
  // ADDRESS variable text, size 25 (city below)
  // CITY fixed text, size 10 (10/nation: nation_prefix||(0-9))
  // NATION fixed text(15) (25 values, longest UNITED KINGDOM)
  // REGION fixed text, size 12 (5 values: longest MIDDLE EAST)
  // PHONE fixed text, size 15 (many values, format: 43-617-354-1222)
  // Primary Key: SUPPKEY

  {
    cout << "Create Table 'SUPPLIER' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(OID, "S_SUPPKEY"));  // primary key
    schema.push_back(Attribut(VARCHAR, "S_NAME"));
    schema.push_back(Attribut(VARCHAR, "S_ADDRESS"));
    schema.push_back(Attribut(VARCHAR, "S_CITY"));
    schema.push_back(Attribut(VARCHAR, "S_NATION"));
    schema.push_back(Attribut(VARCHAR, "S_REGION"));
    schema.push_back(Attribut(VARCHAR, "S_PHONE"));

    CompressionSpecifications compression_specs;
#ifdef USE_ORDER_PRESERVING_DICTIONARY_COMPRESSION
    compression_specs.insert(CompressionSpecification(
        "SUPPLIER.S_NAME", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "SUPPLIER.S_ADDRESS", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "SUPPLIER.S_CITY", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "SUPPLIER.S_NATION", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "SUPPLIER.S_REGION", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "SUPPLIER.S_PHONE", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
#endif

    TablePtr tab(new Table("SUPPLIER", schema, compression_specs));

    if (tab->loadDatafromFile(path_to_files + "supplier.tbl")) {
      cout << "Store Table 'SUPPLIER' ..." << endl;
      if (!tab->setPrimaryKeyConstraint("SUPPLIER.S_SUPPKEY")) {
        COGADB_ERROR("Failed to set Primary Key Constraint!", "");
        return false;
      }
      storeTable(tab);
      addToGlobalTableList(tab);
    }
  }

  // CUSTOMER Table Layout (SF*30,000 are populated)
  // CUSTKEY numeric identifier
  // NAME variable text, size 25 'Customer'||CUSTKEY
  // ADDRESS variable text, size 25 (city below)
  // CITY fixed text, size 10 (10/nation: NATION_PREFIX||(0-9)
  // NATION fixed text(15) (25 values, longest UNITED KINGDOM)
  // REGION fixed text, size 12 (5 values: longest MIDDLE EAST)
  // PHONE fixed text, size 15 (many values, format: 43-617-354-1222)
  // MKTSEGMENT fixed text, size 10 (longest is AUTOMOBILE)
  // Primary Key: CUSTKEY

  {
    cout << "Create Table 'CUSTOMER' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(OID, "C_CUSTKEY"));
    schema.push_back(Attribut(VARCHAR, "C_NAME"));
    schema.push_back(Attribut(VARCHAR, "C_ADDRESS"));
    schema.push_back(Attribut(VARCHAR, "C_CITY"));
    schema.push_back(Attribut(VARCHAR, "C_NATION"));
    schema.push_back(Attribut(VARCHAR, "C_REGION"));
    schema.push_back(Attribut(VARCHAR, "C_PHONE"));
    schema.push_back(Attribut(VARCHAR, "C_MKTSEGMENT"));

    CompressionSpecifications compression_specs;
#ifdef USE_ORDER_PRESERVING_DICTIONARY_COMPRESSION
    compression_specs.insert(CompressionSpecification(
        "CUSTOMER.C_NAME", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "CUSTOMER.C_ADDRESS", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "CUSTOMER.C_CITY", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "CUSTOMER.C_NATION", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "CUSTOMER.C_REGION", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "CUSTOMER.C_PHONE", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "CUSTOMER.C_MKTSEGMENT", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
#endif

    TablePtr tab(new Table("CUSTOMER", schema, compression_specs));

    if (tab->loadDatafromFile(path_to_files + "customer.tbl")) {
      cout << "Store Table 'CUSTOMER' ..." << endl;
      if (!tab->setPrimaryKeyConstraint("CUSTOMER.C_CUSTKEY")) {
        COGADB_ERROR("Failed to set Primary Key Constraint!", "");
        return false;
      }
      storeTable(tab);
      addToGlobalTableList(tab);
    }
  }

  // LINEORDER Table Layout (SF*6,000,000 are populated)
  // ORDERKEY numeric (int up to SF 300) first 8 of each 32 keys used
  // LINENUMBER numeric 1-7
  // CUSTKEY numeric identifier foreign key reference to C_CUSTKEY
  // PARTKEY identifier foreign key reference to P_PARTKEY
  // SUPPKEY numeric identifier foreign key reference to S_SUPPKEY
  // ORDERDATE identifier foreign key reference to D_DATEKEY
  // ORDERPRIORITY fixed text, size 15 (5 Priorities: 1-URGENT, etc.)
  // SHIPPRIORITY fixed text, size 1
  // QUANTITY numeric 1-50 (for PART)
  // EXTENDEDPRICE numeric, MAX about 55,450 (for PART)
  // ORDTOTALPRICE numeric, MAX about 388,000 (for ORDER)
  // DISCOUNT numeric 0-10 (for PART) -- (Represents PERCENT)
  // REVENUE numeric (for PART: (extendedprice*(100-discount))/100)
  // SUPPLYCOST numeric (for PART, cost from supplier, max = ?)
  // TAX numeric 0-8 (for PART)
  // COMMITDATE Foreign Key reference to D_DATEKEY
  // SHIPMODE fixed text, size 10 (Modes: REG AIR, AIR, etc.)
  // Compound Primary Key: ORDERKEY, LINENUMBER

  {
    cout << "Create Table 'LINEORDER' ..." << endl;
    Timestamp begin = getTimestamp();
    TableSchema schema;
    schema.push_back(Attribut(OID, "LO_ORDERKEY"));    //"L_ORDERKEY");
    schema.push_back(Attribut(OID, "LO_LINENUMBER"));  // l_linenumber);
    schema.push_back(Attribut(OID, "LO_CUSTKEY"));
    schema.push_back(Attribut(OID, "LO_PARTKEY"));
    schema.push_back(Attribut(OID, "LO_SUPPKEY"));
    schema.push_back(Attribut(OID, "LO_ORDERDATE"));
    schema.push_back(Attribut(VARCHAR, "LO_ORDERPRIORITY"));
    schema.push_back(Attribut(INT, "LO_SHIPPRIORITY"));
    schema.push_back(Attribut(INT, "LO_QUANTITY"));
    schema.push_back(Attribut(FLOAT, "LO_EXTENDEDPRICE"));
    schema.push_back(Attribut(FLOAT, "LO_ORDTOTALPRICE"));
    schema.push_back(Attribut(FLOAT, "LO_DISCOUNT"));      // l_quantity );
    schema.push_back(Attribut(FLOAT, "LO_REVENUE"));       // l_extendedprice);
    schema.push_back(Attribut(FLOAT, "LO_SUPPLYCOST"));    // l_discount);
    schema.push_back(Attribut(FLOAT, "LO_TAX"));           // l_tax);
    schema.push_back(Attribut(VARCHAR, "LO_COMMITDATE"));  // l_commitdate);
    schema.push_back(Attribut(VARCHAR, "LO_SHIPMODE"));    // l_shipmode);

    CompressionSpecifications compression_specs;
#ifdef USE_ORDER_PRESERVING_DICTIONARY_COMPRESSION
    compression_specs.insert(CompressionSpecification(
        "LINEORDER.LO_ORDERPRIORITY", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "LINEORDER.LO_COMMITDATE", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
    compression_specs.insert(CompressionSpecification(
        "LINEORDER.LO_SHIPMODE", DICTIONARY_COMPRESSED_ORDER_PRESERVING));
#endif

    TablePtr tab(new Table("LINEORDER", schema, compression_specs));

    if (tab->loadDatafromFile(path_to_files + "lineorder.tbl")) {
      Timestamp end = getTimestamp();
      assert(end >= begin);
      cout << "Needed " << double(end - begin) / (1000 * 1000 * 1000)
           << "s for import..." << endl;
      // tab->print();
      cout << "Store Table 'LINEORDER' ..." << endl;
      if (!tab->setForeignKeyConstraint("LINEORDER.LO_CUSTKEY",
                                        "CUSTOMER.C_CUSTKEY", "CUSTOMER")) {
        COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
        return false;
      }
      if (!tab->setForeignKeyConstraint("LINEORDER.LO_PARTKEY",
                                        "PART.P_PARTKEY", "PART")) {
        COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
        return false;
      }
      if (!tab->setForeignKeyConstraint("LINEORDER.LO_SUPPKEY",
                                        "SUPPLIER.S_SUPPKEY", "SUPPLIER")) {
        COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
        return false;
      }
      if (!tab->setForeignKeyConstraint("LINEORDER.LO_ORDERDATE",
                                        "DATES.D_DATEKEY", "DATES")) {
        COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
        return false;
      }
      storeTable(tab);
      addToGlobalTableList(tab);
      //    LO_CustKey bigint not null,
      //    LO_PartKey bigint not null,
      //    LO_SuppKey int not null,
      //    LO_OrderDateKey bigint not null,
    }
  }

  cout << "Building Statistics..." << std::endl;
  computeStatisticsOnTable("DATES", client);
  computeStatisticsOnTable("SUPPLIER", client);
  computeStatisticsOnTable("PART", client);
  computeStatisticsOnTable("CUSTOMER", client);
  computeStatisticsOnTable("LINEORDER", client);

  cout << "Successfully created Star Schema Benchmark Database!" << endl;
  return true;
}

bool optimize_execute_print(const std::string& query_name,
                            LogicalQueryPlan& log_plan, ClientPtr client) {
  uint64_t begin = getTimestamp();

  ostream& out = client->getOutputStream();
#ifdef COGADB_MEASURE_ENERGY_CONSUMPTION
  PCM* pcm = PCM::getInstance();
  SystemCounterState systemState_before;
  SystemCounterState systemState_after;
  systemState_before = pcm->getSystemCounterState();
#endif
  CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
      CoGaDB::query_processing::optimize_and_execute(query_name, log_plan,
                                                     client);
  assert(plan != NULL);
  TablePtr result = plan->getResult();
  if (!result) return false;
  uint64_t end = getTimestamp();
#ifdef COGADB_MEASURE_ENERGY_CONSUMPTION
  systemState_after = pcm->getSystemCounterState();
#endif
  // is Result Table:
  result->setName("");
//                 result->print();
#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
  out << result->toString() << std::endl;
#endif
  out << "Execution Time: " << double(end - begin) / (1000 * 1000) << " ms"
      << std::endl;
  out << "\\e[31m"
      << "Expected Execution Time: "
      << plan->getExpectedExecutionTime() / (1000 * 1000) << "ms"
      << "\\e[39m" << std::endl;
  double total_delay_time = plan->getTotalSchedulingDelay();
  out << "Total Scheduling Delay Time: " << total_delay_time / (1000 * 1000)
      << "ms" << std::endl;
  StatisticsManager::instance().addToValue(
      "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_SECONDS",
      total_delay_time / (1000 * 1000 * 1000));
  StatisticsManager::instance().addToValue(
      "TOTAL_LOST_TIME_DUE_TO_DELAYED_SCHEDULING_IN_NS", total_delay_time);

#ifdef COGADB_MEASURE_ENERGY_CONSUMPTION
  double consumed_joules_processor =
      getConsumedJoules(systemState_before, systemState_after);
  double consumed_joules_dram =
      getDRAMConsumedJoules(systemState_before, systemState_after);
  double total_joules_for_query =
      consumed_joules_processor + consumed_joules_dram;

  out << "Consumed Joules (CPU): " << consumed_joules_processor << std::endl;
  out << "Consumed Joules (DRAM): " << consumed_joules_dram << std::endl;
  out << "Consumed Joules (Total): " << total_joules_for_query << std::endl;
  out << "Energy Product (Joule*seconds): "
      << total_joules_for_query * double(end - begin) / (1000 * 1000 * 1000)
      << std::endl;
#endif
  return true;
}

bool optimize_execute_print(const std::string& query_name,
                            query_processing::LogicalQueryPlan& log_plan) {
  ClientPtr client(new LocalClient());
  return optimize_execute_print(query_name, log_plan, client);
}

/*QUERIES*/

//#ifdef ENABLE_TWO_PHASE_PHYSICAL_OPTIMIZATION
//
//
//
//#ifndef USE_FETCH_JOIN
//    LogicalQueryPlanPtr SSB_Q11_plan(){
//        //*****************************************************************
//        //        --Q1.1
//        //        select sum(lo_extendedprice*lo_discount) as revenue
//        //        from lineorder, dates
//        //        where lo_orderdate =  d_datekey
//        //        and d_year = 1993
//        //        and lo_discount between 1 and 3
//        //        and lo_quantity < 25;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//
//        //Attribut(INT,"YEARMONTHNUM")); //  numeric (YYYYMM) -- e.g. 199803
//
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_year(new logical_operator::Logical_Selection("D_YEAR",
//        boost::any(1993), EQUAL, LOOKUP, default_device_constraint));
//        //        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_quantity(new
//        logical_operator::Logical_Selection("LO_QUANTITY", boost::any(25),
//        LESSER, LOOKUP, default_device_constraint));
//        //        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_discount_lower(new
//        logical_operator::Logical_Selection("LO_DISCOUNT",
//        boost::any(float(0.99)), GREATER, LOOKUP, default_device_constraint));
//        //        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_discount_upper(new
//        logical_operator::Logical_Selection("LO_DISCOUNT",
//        boost::any(float(3.01)), LESSER, LOOKUP, default_device_constraint));
//
//        KNF_Selection_Expression knf_expr; //LO_QUANTITY<25 AND
//        LO_DISCOUNT>0.99 AND LO_DISCOUNT<3.01
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(25),
//            ValueConstantPredicate, LESSER)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(1)),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(3)),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT<3.01
//            knf_expr.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_lineorder(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_Join> join(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_EXTENDEDPRICE",
//        "LO_DISCOUNT", "REVENUE", MUL, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("REVENUE", SUM));
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
////        selection_year->setLeft(scan_date);
////        complex_selection_on_lineorder->setLeft(scan_lineorder);
////        join->setLeft(selection_year);
////        join->setRight(complex_selection_on_lineorder);
////        column_algebra_operation_lineorder->setLeft(join);
////        order->setLeft(column_algebra_operation_lineorder);
////        group->setLeft(order);
//
//
//        selection_year->setLeft(scan_date);
//        complex_selection_on_lineorder->setLeft(scan_lineorder);
//        join->setLeft(selection_year);
//        join->setRight(complex_selection_on_lineorder);
//        column_algebra_operation_lineorder->setLeft(join);
//
////        selection_year->setLeft(scan_date);
////        join->setLeft(selection_year);
////        join->setRight(scan_lineorder);
////        complex_selection_on_lineorder->setLeft(join);
////
/// column_algebra_operation_lineorder->setLeft(complex_selection_on_lineorder);
//
//        //column_algebra_operation_lineorder->setLeft(complex_selection_on_lineorder);
//        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(order);
//
//        std::list<std::string> column_list;
//        column_list.push_back("REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Projection> projection(new
//        logical_operator::Logical_Projection(column_list)); //GPU Projection
//        not supported
//
//        projection->setLeft(group);
//        return boost::make_shared<LogicalQueryPlan>(projection);
//    }
//#else
//
//    LogicalQueryPlanPtr SSB_Q11_plan(){
//        //*****************************************************************
//        //        --Q1.1
//        //        select sum(lo_extendedprice*lo_discount) as revenue
//        //        from lineorder, dates
//        //        where lo_orderdate =  d_datekey
//        //        and d_year = 1993
//        //        and lo_discount between 1 and 3
//        //        and lo_quantity < 25;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//
//        //Attribut(INT,"YEARMONTHNUM")); //  numeric (YYYYMM) -- e.g. 199803
//
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_year(new logical_operator::Logical_Selection("D_YEAR",
//        boost::any(1993), EQUAL, LOOKUP, default_device_constraint));
//        //        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_quantity(new
//        logical_operator::Logical_Selection("LO_QUANTITY", boost::any(25),
//        LESSER, LOOKUP, default_device_constraint));
//        //        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_discount_lower(new
//        logical_operator::Logical_Selection("LO_DISCOUNT",
//        boost::any(float(0.99)), GREATER, LOOKUP, default_device_constraint));
//        //        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_discount_upper(new
//        logical_operator::Logical_Selection("LO_DISCOUNT",
//        boost::any(float(3.01)), LESSER, LOOKUP, default_device_constraint));
//
//        KNF_Selection_Expression knf_expr; //LO_QUANTITY<25 AND
//        LO_DISCOUNT>0.99 AND LO_DISCOUNT<3.01
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(25),
//            ValueConstantPredicate, LESSER)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(1)),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(3)),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT<3.01
//            knf_expr.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_lineorder(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        //boost::shared_ptr<logical_operator::Logical_Join> join(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Fetch_Join> join(new
//        logical_operator::Logical_Fetch_Join("D_DATEKEY", "LO_ORDERDATE",
//        LOOKUP, default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_EXTENDEDPRICE",
//        "LO_DISCOUNT", "REVENUE", MUL, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("REVENUE", SUM));
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
////        selection_year->setLeft(scan_date);
////        complex_selection_on_lineorder->setLeft(scan_lineorder);
////        join->setLeft(selection_year);
////        join->setRight(complex_selection_on_lineorder);
////        column_algebra_operation_lineorder->setLeft(join);
////        order->setLeft(column_algebra_operation_lineorder);
////        group->setLeft(order);
//
//
//        selection_year->setLeft(scan_date);
//        join->setLeft(selection_year);
//        join->setRight(scan_lineorder);
//        complex_selection_on_lineorder->setLeft(join);
//        column_algebra_operation_lineorder->setLeft(complex_selection_on_lineorder);
//
////        selection_year->setLeft(scan_date);
////        join->setLeft(selection_year);
////        join->setRight(scan_lineorder);
////        complex_selection_on_lineorder->setLeft(join);
////
/// column_algebra_operation_lineorder->setLeft(complex_selection_on_lineorder);
//
//        //column_algebra_operation_lineorder->setLeft(complex_selection_on_lineorder);
////        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(column_algebra_operation_lineorder);
//
//        std::list<std::string> column_list;
//        column_list.push_back("REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Projection> projection(new
//        logical_operator::Logical_Projection(column_list)); //GPU Projection
//        not supported
//
//        projection->setLeft(group);
//        return boost::make_shared<LogicalQueryPlan>(projection);
//    }
//
//#endif
//
//#else
//    LogicalQueryPlanPtr SSB_Q11_plan(){
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//
//        //Attribut(INT,"YEARMONTHNUM")); //  numeric (YYYYMM) -- e.g. 199803
//
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_year(new logical_operator::Logical_Selection("D_YEAR",
//        boost::any(1993), EQUAL, LOOKUP, default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_quantity(new
//        logical_operator::Logical_Selection("LO_QUANTITY", boost::any(25),
//        LESSER, LOOKUP, default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_discount_lower(new
//        logical_operator::Logical_Selection("LO_DISCOUNT",
//        boost::any(float(1)), GREATER_EQUAL, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_discount_upper(new
//        logical_operator::Logical_Selection("LO_DISCOUNT",
//        boost::any(float(3)), LESSER_EQUAL, LOOKUP,
//        default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_Join> join(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_EXTENDEDPRICE",
//        "LO_DISCOUNT", "REVENUE", MUL, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("REVENUE", SUM));
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        selection_year->setLeft(scan_date);
//        selection_quantity->setLeft(scan_lineorder);
//
//        selection_discount_lower->setLeft(selection_quantity);
//        selection_discount_upper->setLeft(selection_discount_lower);
//
//        join->setLeft(selection_year);
//        join->setRight(selection_discount_upper);
//        column_algebra_operation_lineorder->setLeft(join);
//
//
//
//        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(order);
//
//        std::list<std::string> column_list;
//        column_list.push_back("REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Projection> projection(new
//        logical_operator::Logical_Projection(column_list)); //GPU Projection
//        not supported
//
//        projection->setLeft(group);
//        return boost::make_shared<LogicalQueryPlan>(projection);
//    }
//#endif
//
//    bool SSB_Q11(ClientPtr client) {
//	return optimize_execute_print("SSB Query 1.1", *SSB_Q11_plan(),client);
//    }
//#ifndef USE_FETCH_JOIN
//    LogicalQueryPlanPtr SSB_Q12_plan() {
//        //*****************************************************************
//        //--Q1.2
//        //select sum(lo_extendedprice*lo_discount) as
//        //revenue
//        //from lineorder, dates
//        //where lo_orderdate =  d_datekey
//        //and d_yearmonthnum = 199401
//        //and lo_discount between 4 and 6
//        //and lo_quantity between 26 and 35;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//
//        //Attribut(INT,"YEARMONTHNUM")); //  numeric (YYYYMM) -- e.g. 199803
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(26),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(35),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(4)),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(6)),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT<3.01
//            knf_expr.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_lineorder(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_year(new
//        logical_operator::Logical_Selection("D_YEARMONTHNUM",
//        boost::any(199401), EQUAL, LOOKUP, default_device_constraint));
//
//
//        boost::shared_ptr<logical_operator::Logical_Join> join(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_EXTENDEDPRICE",
//        "LO_DISCOUNT", "REVENUE", MUL, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        selection_year->setLeft(scan_date);
//        complex_selection_on_lineorder->setLeft(scan_lineorder);
//
//
//        join->setLeft(selection_year);
//        join->setRight(complex_selection_on_lineorder);
//        column_algebra_operation_lineorder->setLeft(join);
//        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(order);
//
//        std::list<std::string> column_list;
//        column_list.push_back("REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Projection> projection(new
//        logical_operator::Logical_Projection(column_list)); //GPU Projection
//        not supported
//
//        projection->setLeft(group);
//        return boost::make_shared<LogicalQueryPlan>(projection);
//    }
//#else
//    LogicalQueryPlanPtr SSB_Q12_plan() {
//        //*****************************************************************
//        //--Q1.2
//        //select sum(lo_extendedprice*lo_discount) as
//        //revenue
//        //from lineorder, dates
//        //where lo_orderdate =  d_datekey
//        //and d_yearmonthnum = 199401
//        //and lo_discount between 4 and 6
//        //and lo_quantity between 26 and 35;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//
//        //Attribut(INT,"YEARMONTHNUM")); //  numeric (YYYYMM) -- e.g. 199803
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(26),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(35),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(4)),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(6)),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT<3.01
//            knf_expr.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_lineorder(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_year(new
//        logical_operator::Logical_Selection("D_YEARMONTHNUM",
//        boost::any(199401), EQUAL, LOOKUP, default_device_constraint));
//
//
//        //boost::shared_ptr<logical_operator::Logical_Join> join(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Fetch_Join> join(new
//        logical_operator::Logical_Fetch_Join("D_DATEKEY", "LO_ORDERDATE",
//        LOOKUP, default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_EXTENDEDPRICE",
//        "LO_DISCOUNT", "REVENUE", MUL, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        selection_year->setLeft(scan_date);
//        join->setLeft(selection_year);
//        join->setRight(scan_lineorder);
//        complex_selection_on_lineorder->setLeft(join);
//        column_algebra_operation_lineorder->setLeft(complex_selection_on_lineorder);
////        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(column_algebra_operation_lineorder);
//
//        std::list<std::string> column_list;
//        column_list.push_back("REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Projection> projection(new
//        logical_operator::Logical_Projection(column_list)); //GPU Projection
//        not supported
//
//        projection->setLeft(group);
//        return boost::make_shared<LogicalQueryPlan>(projection);
//    }
//#endif
//
//    bool SSB_Q12(ClientPtr client) {
//	return optimize_execute_print("SSB Query 1.2", *SSB_Q12_plan(),client);
//    }
//
//#ifndef USE_FETCH_JOIN
//    LogicalQueryPlanPtr SSB_Q13_plan() {
//        //*****************************************************************
//        //--Q1.3
//        //select sum(lo_extendedprice*lo_discount) as
//        //revenue
//        //from lineorder, dates
//        //where lo_orderdate =  d_datekey
//        //and d_weeknuminyear = 6
//        //and d_year = 1994
//        //and lo_discount between 5 and 7
//        //and lo_quantity between 26 and 35;
//        //*****************************************************************
//
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(26),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(35),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(5)),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(7)),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT<3.01
//            knf_expr.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_lineorder(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        KNF_Selection_Expression knf_expr2;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1994),
//            ValueConstantPredicate, EQUAL));
//            knf_expr2.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_WEEKNUMINYEAR", boost::any(6),
//            ValueConstantPredicate, EQUAL));
//            knf_expr2.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_date(new
//        logical_operator::Logical_ComplexSelection(knf_expr2, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_Join> join(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_EXTENDEDPRICE",
//        "LO_DISCOUNT", "REVENUE", MUL, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        complex_selection_on_date->setLeft(scan_date);
//        complex_selection_on_lineorder->setLeft(scan_lineorder);
//        join->setLeft(complex_selection_on_date);
//        join->setRight(complex_selection_on_lineorder);
//        column_algebra_operation_lineorder->setLeft(join);
//        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(order);
//
//        std::list<std::string> column_list;
//        column_list.push_back("REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Projection> projection(new
//        logical_operator::Logical_Projection(column_list)); //GPU Projection
//        not supported
//
//        projection->setLeft(group);
//        return boost::make_shared<LogicalQueryPlan>(projection);
//    }
//
//#else
//    LogicalQueryPlanPtr SSB_Q13_plan() {
//        //*****************************************************************
//        //--Q1.3
//        //select sum(lo_extendedprice*lo_discount) as
//        //revenue
//        //from lineorder, dates
//        //where lo_orderdate =  d_datekey
//        //and d_weeknuminyear = 6
//        //and d_year = 1994
//        //and lo_discount between 5 and 7
//        //and lo_quantity between 26 and 35;
//        //*****************************************************************
//
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(26),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_QUANTITY", boost::any(35),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(5)),
//            ValueConstantPredicate, GREATER_EQUAL)); //LO_DISCOUNT>0.99
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("LO_DISCOUNT", boost::any(float(7)),
//            ValueConstantPredicate, LESSER_EQUAL)); //LO_DISCOUNT<3.01
//            knf_expr.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_lineorder(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        KNF_Selection_Expression knf_expr2;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1994),
//            ValueConstantPredicate, EQUAL));
//            knf_expr2.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_WEEKNUMINYEAR", boost::any(6),
//            ValueConstantPredicate, EQUAL));
//            knf_expr2.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_date(new
//        logical_operator::Logical_ComplexSelection(knf_expr2, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        //boost::shared_ptr<logical_operator::Logical_Join> join(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Fetch_Join> join(new
//        logical_operator::Logical_Fetch_Join("D_DATEKEY", "LO_ORDERDATE",
//        LOOKUP, default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_EXTENDEDPRICE",
//        "LO_DISCOUNT", "REVENUE", MUL, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        complex_selection_on_date->setLeft(scan_date);
//        join->setLeft(complex_selection_on_date);
//        //it is crucial that we do NOT filter the foreign key table before
//        using the join index!
//        join->setRight(scan_lineorder);
//        //filter after join, saved costs for join do more than compensate for
//        a
//        //little extra overhead for fitler operations
//        complex_selection_on_lineorder->setLeft(join);
//        column_algebra_operation_lineorder->setLeft(complex_selection_on_lineorder);
////        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(column_algebra_operation_lineorder);
//
//
//        std::list<std::string> column_list;
//        column_list.push_back("REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Projection> projection(new
//        logical_operator::Logical_Projection(column_list)); //GPU Projection
//        not supported
//
//        projection->setLeft(group);
//        return boost::make_shared<LogicalQueryPlan>(projection);
//    }
//#endif
//    bool SSB_Q13(ClientPtr client) {
//	return optimize_execute_print("SSB Query 1.3", *SSB_Q13_plan(), client);
//    }
//
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q21_plan() {
//        //*****************************************************************
//        //--Q2.1
//        //select sum(lo_revenue), d_year, p_brand
//        //from lineorder, dates, part, supplier
//        //where lo_orderdate =  d_datekey
//        //and lo_partkey = p_partkey
//        //and lo_suppkey = s_suppkey
//        //and p_category = 'MFGR#12'
//        //and s_region = 'AMERICA'
//        //group by d_year, p_brand
//        //order by d_year, p_brand;
//        //*****************************************************************
//
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_part(new
//        logical_operator::Logical_Scan("PART"));
//
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_category(new
//        logical_operator::Logical_Selection("P_CATEGORY",
//        boost::any(std::string("MFGR#12")), EQUAL, LOOKUP,
//        default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_category(new
//        logical_operator::Logical_Selection("P_CATEGORY",
//        boost::any(std::string("MFGR#12")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//        boost::shared_ptr<logical_operator::Logical_Join> join_part(new
//        logical_operator::Logical_Join("LO_PARTKEY", "P_PARTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("P_BRAND");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        selection_region->setLeft(scan_supplier);
//        selection_category->setLeft(scan_part);
//
//        join_part->setLeft(scan_lineorder);
//        join_part->setRight(selection_category);
//
//        join_supplier->setLeft(selection_region);
//        join_supplier->setRight(join_part);
//
//        join_date->setLeft(scan_date);
//        join_date->setRight(join_supplier);
//
//
//        order->setLeft(join_date);
//        group->setLeft(order);
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(group);
//#else
//	return boost::make_shared<LogicalQueryPlan>(group);
//#endif
//    }
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q21_plan() {
//        //*****************************************************************
//        //--Q2.1
//        //select sum(lo_revenue), d_year, p_brand
//        //from lineorder, dates, part, supplier
//        //where lo_orderdate =  d_datekey
//        //and lo_partkey = p_partkey
//        //and lo_suppkey = s_suppkey
//        //and p_category = 'MFGR#12'
//        //and s_region = 'AMERICA'
//        //group by d_year, p_brand
//        //order by d_year, p_brand;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
////        boost::shared_ptr<logical_operator::Logical_Selection>
/// selection_region(new logical_operator::Logical_Selection("S_REGION",
/// boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
/// default_device_constraint));
////        boost::shared_ptr<logical_operator::Logical_Selection>
/// selection_category(new logical_operator::Logical_Selection("P_CATEGORY",
/// boost::any(std::string("MFGR#12")), EQUAL, LOOKUP,
/// default_device_constraint));
////
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
////        {
////            Disjunction d;
////            d.push_back(Predicate("D_YEAR", boost::any(1997),
/// ValueConstantPredicate, EQUAL));
////            d.push_back(Predicate("D_YEAR", boost::any(1998),
/// ValueConstantPredicate, EQUAL));
////            knf_expr_date.disjunctions.push_back(d);
////        }
////        KNF_Selection_Expression knf_expr_customer;
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_REGION",
//            boost::any(std::string("AMERICA")), ValueConstantPredicate,
//            EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_part;
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_CATEGORY",
//            boost::any(std::string("MFGR#12")), ValueConstantPredicate,
//            EQUAL)); //YEAR<2013
//            knf_expr_part.disjunctions.push_back(d);
//        }
//        InvisibleJoinSelectionList dimensions;
////        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
/// Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
/// EQUAL),
/// knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//        dimensions.push_back(InvisibleJoinSelection("PART",
//        Predicate("P_PARTKEY", std::string("LO_PARTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_part));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, hype::CPU_ONLY));
//
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("P_BRAND");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        inv_join->setLeft(scan_lineorder);
//        group->setLeft(inv_join);
//        order->setLeft(group);
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//
//    }
//#endif
//    bool SSB_Q21(ClientPtr client) {
//	return optimize_execute_print("SSB Query 2.1", *SSB_Q21_plan(),client);
//    }
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q22_plan() {
//        //*****************************************************************
//        //--Q2.2
//        //select sum(lo_revenue), d_year, p_brand
//        //from lineorder, dates, part, supplier
//        //where lo_orderdate =  d_datekey
//        //and lo_partkey = p_partkey
//        //and lo_suppkey = s_suppkey
//        //and p_brand between 'MFGR#2221'
//        //and 'MFGR#2228'
//        //and s_region = 'ASIA'
//        //group by d_year, p_brand
//        //order by d_year, p_brand;
//        //*****************************************************************
//
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_part(new
//        logical_operator::Logical_Scan("PART"));
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("ASIA")), EQUAL, LOOKUP,
//        default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("ASIA")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_BRAND",
//            boost::any(std::string("MFGR#2221")), ValueConstantPredicate,
//            GREATER_EQUAL));
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_BRAND",
//            boost::any(std::string("MFGR#2228")), ValueConstantPredicate,
//            LESSER_EQUAL));
//            knf_expr.disjunctions.push_back(d);
//        }
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_parts(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#else
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_parts(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//        boost::shared_ptr<logical_operator::Logical_Join> join_part(new
//        logical_operator::Logical_Join("LO_PARTKEY", "P_PARTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("P_BRAND");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        selection_region->setLeft(scan_supplier);
//        complex_selection_on_parts->setLeft(scan_part);
//
////        join_part->setLeft(scan_lineorder);
////        join_part->setRight(complex_selection_on_parts);
////
////        join_supplier->setLeft(selection_region);
////        join_supplier->setRight(join_part);
////
////        join_date->setLeft(scan_date);
////        join_date->setRight(join_supplier);
////
////        order->setLeft(join_date);
//
//
//        join_supplier->setLeft(selection_region);
//        join_supplier->setRight(scan_lineorder);
//
//        join_part->setLeft(join_supplier);
//        join_part->setRight(complex_selection_on_parts);
//
//
//
//        join_date->setLeft(scan_date);
//        join_date->setRight(join_part);
//
//        order->setLeft(join_date);
//
//        group->setLeft(order);
//
//        return boost::make_shared<LogicalQueryPlan>(group);
//    }
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q22_plan() {
//        //*****************************************************************
//        //--Q2.2
//        //select sum(lo_revenue), d_year, p_brand
//        //from lineorder, dates, part, supplier
//        //where lo_orderdate =  d_datekey
//        //and lo_partkey = p_partkey
//        //and lo_suppkey = s_suppkey
//        //and p_brand between 'MFGR#2221'
//        //and 'MFGR#2228'
//        //and s_region = 'ASIA'
//        //group by d_year, p_brand
//        //order by d_year, p_brand;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
////        {
////            Disjunction d;
////            d.push_back(Predicate("D_YEAR", boost::any(1997),
/// ValueConstantPredicate, EQUAL));
////            d.push_back(Predicate("D_YEAR", boost::any(1998),
/// ValueConstantPredicate, EQUAL));
////            knf_expr_date.disjunctions.push_back(d);
////        }
////        KNF_Selection_Expression knf_expr_customer;
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_REGION", boost::any(std::string("ASIA")),
//            ValueConstantPredicate, EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_part;
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_BRAND",
//            boost::any(std::string("MFGR#2221")), ValueConstantPredicate,
//            GREATER_EQUAL));
//            knf_expr_part.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_BRAND",
//            boost::any(std::string("MFGR#2228")), ValueConstantPredicate,
//            LESSER_EQUAL));
//            knf_expr_part.disjunctions.push_back(d);
//        }
////        {
////            Disjunction d;
////            d.push_back(Predicate("P_CATEGORY",
/// boost::any(std::string("MFGR#12")), ValueConstantPredicate, EQUAL));
/////YEAR<2013
////            knf_expr_part.disjunctions.push_back(d);
////        }
//        InvisibleJoinSelectionList dimensions;
////        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
/// Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
/// EQUAL),
/// knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//        dimensions.push_back(InvisibleJoinSelection("PART",
//        Predicate("P_PARTKEY", std::string("LO_PARTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_part));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, hype::CPU_ONLY));
//
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("P_BRAND");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        inv_join->setLeft(scan_lineorder);
//        group->setLeft(inv_join);
//        order->setLeft(group);
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//
//    }
//#endif
//    bool SSB_Q22(ClientPtr client) {
//	return optimize_execute_print("SSB Query 2.2", *SSB_Q22_plan(),client);
//    }
//
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q23_plan() {
//        //*****************************************************************
//        //--Q2.3
//        //select sum(lo_revenue), d_year, p_brand
//        //from lineorder, dates, part, supplier
//        //where lo_orderdate =  d_datekey
//        //and lo_partkey = p_partkey
//        //and lo_suppkey = s_suppkey
//        //and p_brand= 'MFGR#2239'
//        //and s_region = 'EUROPE'
//        //group by d_year, p_brand
//        //order by d_year, p_brand;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_part(new
//        logical_operator::Logical_Scan("PART"));
//
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("EUROPE")), EQUAL,
//        LOOKUP,default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_category(new logical_operator::Logical_Selection("P_BRAND",
//        boost::any(std::string("MFGR#2239")), EQUAL,
//        LOOKUP,default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("EUROPE")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_category(new logical_operator::Logical_Selection("P_BRAND",
//        boost::any(std::string("MFGR#2239")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//
//        boost::shared_ptr<logical_operator::Logical_Join> join_part(new
//        logical_operator::Logical_Join("LO_PARTKEY", "P_PARTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("P_BRAND");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        selection_region->setLeft(scan_supplier);
//        selection_category->setLeft(scan_part);
//
//        join_part->setLeft(scan_lineorder);
//        join_part->setRight(selection_category);
//
//        join_supplier->setLeft(selection_region);
//        join_supplier->setRight(join_part);
//
//        join_date->setLeft(scan_date);
//        join_date->setRight(join_supplier);
//
//        order->setLeft(join_date);
//        group->setLeft(order);
//
//        return boost::make_shared<LogicalQueryPlan>(group);
//    }
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q23_plan() {
//        //*****************************************************************
//        //--Q2.3
//        //select sum(lo_revenue), d_year, p_brand
//        //from lineorder, dates, part, supplier
//        //where lo_orderdate =  d_datekey
//        //and lo_partkey = p_partkey
//        //and lo_suppkey = s_suppkey
//        //and p_brand= 'MFGR#2239'
//        //and s_region = 'EUROPE'
//        //group by d_year, p_brand
//        //order by d_year, p_brand;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
////        boost::shared_ptr<logical_operator::Logical_Selection>
/// selection_region(new logical_operator::Logical_Selection("S_REGION",
/// boost::any(std::string("EUROPE")), EQUAL,
/// LOOKUP,default_device_constraint));
////        boost::shared_ptr<logical_operator::Logical_Selection>
/// selection_category(new logical_operator::Logical_Selection("P_BRAND",
/// boost::any(std::string("MFGR#2239")), EQUAL,
/// LOOKUP,default_device_constraint));
////
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
////        {
////            Disjunction d;
////            d.push_back(Predicate("D_YEAR", boost::any(1997),
/// ValueConstantPredicate, EQUAL));
////            d.push_back(Predicate("D_YEAR", boost::any(1998),
/// ValueConstantPredicate, EQUAL));
////            knf_expr_date.disjunctions.push_back(d);
////        }
////        KNF_Selection_Expression knf_expr_customer;
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_REGION",
//            boost::any(std::string("EUROPE")), ValueConstantPredicate,
//            EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_part;
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_BRAND",
//            boost::any(std::string("MFGR#2239")), ValueConstantPredicate,
//            EQUAL));
//            knf_expr_part.disjunctions.push_back(d);
//        }
////        {
////            Disjunction d;
////            d.push_back(Predicate("P_CATEGORY",
/// boost::any(std::string("MFGR#12")), ValueConstantPredicate, EQUAL));
/////YEAR<2013
////            knf_expr_part.disjunctions.push_back(d);
////        }
//        InvisibleJoinSelectionList dimensions;
////        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
/// Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
/// EQUAL),
/// knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//        dimensions.push_back(InvisibleJoinSelection("PART",
//        Predicate("P_PARTKEY", std::string("LO_PARTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_part));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, hype::CPU_ONLY));
//
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("P_BRAND");
//
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(sorting_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        inv_join->setLeft(scan_lineorder);
//        group->setLeft(inv_join);
//        order->setLeft(group);
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//
//    }
//#endif
//    bool SSB_Q23(ClientPtr client) {
//	return optimize_execute_print("SSB Query 2.3", *SSB_Q23_plan(), client);
//    }
//
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q31_plan() {
//        //*****************************************************************
//        //--Q3.1
//        //select c_nation, s_nation, d_year,
//        //sum(lo_revenue)  as  revenue
//        //from customer, lineorder, supplier, dates
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_orderdate = d_datekey
//        //and c_region = 'ASIA'
//        //and s_region = 'ASIA'
//        //and d_year >= 1992 and d_year <= 1997
//        //group by c_nation, s_nation, d_year
//        //order by d_year asc,  revenue desc;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
//        logical_operator::Logical_Scan("CUSTOMER"));
//
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("ASIA")), EQUAL,
//        LOOKUP,default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_c_region(new logical_operator::Logical_Selection("C_REGION",
//        boost::any(std::string("ASIA")), EQUAL,
//        LOOKUP,default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("ASIA")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_c_region(new logical_operator::Logical_Selection("C_REGION",
//        boost::any(std::string("ASIA")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1992),
//            ValueConstantPredicate, GREATER_EQUAL));
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, LESSER_EQUAL));
//            knf_expr.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_date(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_Join> join_customer(new
//        logical_operator::Logical_Join("C_CUSTKEY", "LO_CUSTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//        std::list<std::string> groupsorting_column_names;
//        groupsorting_column_names.push_back("C_NATION");
//        groupsorting_column_names.push_back("S_NATION");
//        groupsorting_column_names.push_back("D_YEAR");
//        boost::shared_ptr<logical_operator::Logical_Sort> grouporder(new
//        logical_operator::Logical_Sort(groupsorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("C_NATION");
//        grouping_column_names.push_back("S_NATION");
//        grouping_column_names.push_back("D_YEAR");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("LO_REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//
//        selection_s_region->setLeft(scan_supplier);
//        selection_c_region->setLeft(scan_customer);
//
//        complex_selection_on_date->setLeft(scan_date);
//
//        join_customer->setLeft(selection_c_region);
//        join_customer->setRight(scan_lineorder);
//
//        join_supplier->setLeft(selection_s_region);
//        join_supplier->setRight(join_customer);
//
//        join_date->setLeft(complex_selection_on_date);
//        join_date->setRight(join_supplier);
//
//
//        grouporder->setLeft(join_date);
//        group->setLeft(grouporder);
//        order->setLeft(group);
//
//        return boost::make_shared<LogicalQueryPlan>(order);
//    }
//
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q31_plan() {
//        //*****************************************************************
//        //--Q3.1
//        //select c_nation, s_nation, d_year,
//        //sum(lo_revenue)  as  revenue
//        //from customer, lineorder, supplier, dates
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_orderdate = d_datekey
//        //and c_region = 'ASIA'
//        //and s_region = 'ASIA'
//        //and d_year >= 1992 and d_year <= 1997
//        //group by c_nation, s_nation, d_year
//        //order by d_year asc,  revenue desc;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1992),
//            ValueConstantPredicate, GREATER_EQUAL));
//            knf_expr_date.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, LESSER_EQUAL));
//            knf_expr_date.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_customer;
//        {
//            Disjunction d;
//            d.push_back(Predicate("C_REGION", boost::any(std::string("ASIA")),
//            ValueConstantPredicate, EQUAL));
//            knf_expr_customer.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_REGION", boost::any(std::string("ASIA")),
//            ValueConstantPredicate, EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//
//        InvisibleJoinSelectionList dimensions;
//        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
//        Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, default_device_constraint));
//
//        std::list<std::string> groupsorting_column_names;
//        groupsorting_column_names.push_back("C_NATION");
//        groupsorting_column_names.push_back("S_NATION");
//        groupsorting_column_names.push_back("D_YEAR");
//        boost::shared_ptr<logical_operator::Logical_Sort> grouporder(new
//        logical_operator::Logical_Sort(groupsorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("C_NATION");
//        grouping_column_names.push_back("S_NATION");
//        grouping_column_names.push_back("D_YEAR");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("LO_REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//
//        inv_join->setLeft(scan_lineorder);
//
////        grouporder->setLeft(inv_join);
//        group->setLeft(inv_join);
//        order->setLeft(group);
//
//
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//    }
//#endif
//
//    bool SSB_Q31(ClientPtr client) {
//	return optimize_execute_print("SSB Query 3.1", *SSB_Q31_plan(), client);
//    }
//
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q32_plan() {
//        //*****************************************************************
//        //--Q3.2
//        //select c_city, s_city, d_year, sum(lo_revenue)
//        //as  revenue
//        //from customer, lineorder, supplier, dates
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_orderdate = d_datekey
//        //and c_nation = 'UNITED STATES'
//        //and s_nation = 'UNITED STATES'
//        //and d_year >= 1992 and d_year <= 1997
//        //group by c_city, s_city, d_year
//        //order by d_year asc,  revenue desc;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
//        logical_operator::Logical_Scan("CUSTOMER"));
//
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_nation(new logical_operator::Logical_Selection("S_NATION",
//        boost::any(std::string("UNITED STATES")), EQUAL, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_c_nation(new logical_operator::Logical_Selection("C_NATION",
//        boost::any(std::string("UNITED STATES")), EQUAL, LOOKUP,
//        default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_nation(new logical_operator::Logical_Selection("S_NATION",
//        boost::any(std::string("UNITED STATES")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_c_nation(new logical_operator::Logical_Selection("C_NATION",
//        boost::any(std::string("UNITED STATES")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1992),
//            ValueConstantPredicate, GREATER_EQUAL));
//            knf_expr.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, LESSER_EQUAL));
//            knf_expr.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_date(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_Join> join_customer(new
//        logical_operator::Logical_Join("C_CUSTKEY", "LO_CUSTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//        std::list<std::string> ordersorting_column_names;
//        ordersorting_column_names.push_back("C_CITY");
//        ordersorting_column_names.push_back("S_CITY");
//        ordersorting_column_names.push_back("D_YEAR");
//        boost::shared_ptr<logical_operator::Logical_Sort> grouporder(new
//        logical_operator::Logical_Sort(ordersorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("C_CITY");
//        grouping_column_names.push_back("S_CITY");
//        grouping_column_names.push_back("D_YEAR");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("LO_REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//
//        selection_s_nation->setLeft(scan_supplier);
//        selection_c_nation->setLeft(scan_customer);
//
//        complex_selection_on_date->setLeft(scan_date);
//
//        join_customer->setLeft(selection_c_nation);
//        join_customer->setRight(scan_lineorder);
//
//        join_supplier->setLeft(selection_s_nation);
//        join_supplier->setRight(join_customer);
//
//        join_date->setLeft(complex_selection_on_date);
//        join_date->setRight(join_supplier);
//
//
//        grouporder->setLeft(join_date);
//        group->setLeft(grouporder);
//        order->setLeft(group);
//
//        return boost::make_shared<LogicalQueryPlan>(order);
//    }
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q32_plan() {
//        //*****************************************************************
//        //--Q3.2
//        //select c_city, s_city, d_year, sum(lo_revenue)
//        //as  revenue
//        //from customer, lineorder, supplier, dates
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_orderdate = d_datekey
//        //and c_nation = 'UNITED STATES'
//        //and s_nation = 'UNITED STATES'
//        //and d_year >= 1992 and d_year <= 1997
//        //group by c_city, s_city, d_year
//        //order by d_year asc,  revenue desc;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1992),
//            ValueConstantPredicate, GREATER_EQUAL));
//            knf_expr_date.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, LESSER_EQUAL));
//            knf_expr_date.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_customer;
//        {
//            Disjunction d;
//            d.push_back(Predicate("C_NATION", boost::any(std::string("UNITED
//            STATES")), ValueConstantPredicate, EQUAL));
//            knf_expr_customer.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_NATION", boost::any(std::string("UNITED
//            STATES")), ValueConstantPredicate, EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//
//        InvisibleJoinSelectionList dimensions;
//        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
//        Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, default_device_constraint));
//
//        std::list<std::string> ordersorting_column_names;
//        ordersorting_column_names.push_back("C_CITY");
//        ordersorting_column_names.push_back("S_CITY");
//        ordersorting_column_names.push_back("D_YEAR");
//        boost::shared_ptr<logical_operator::Logical_Sort> grouporder(new
//        logical_operator::Logical_Sort(ordersorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("C_CITY");
//        grouping_column_names.push_back("S_CITY");
//        grouping_column_names.push_back("D_YEAR");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("LO_REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//
//        inv_join->setLeft(scan_lineorder);
//
////        grouporder->setLeft(inv_join);
//        group->setLeft(inv_join);
//        order->setLeft(group);
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//    }
//#endif
//
//    bool SSB_Q32(ClientPtr client) {
//	return optimize_execute_print("SSB Query 3.2", *SSB_Q32_plan(), client);
//    }
//
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q33_plan() {
//
//        //*****************************************************************
//        //--Q3.3
//        //select c_city, s_city, d_year, sum(lo_revenue)
//        //as  revenue
//        //from customer, lineorder, supplier, dates
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_orderdate = d_datekey
//        //and  (c_city='UNITED KI1'
//        //or c_city='UNITED KI5')
//        //and (s_city='UNITED KI1'
//        //or s_city='UNITED KI5')
//        //and d_year >= 1992 and d_year <= 1997
//        //group by c_city, s_city, d_year
//        //order by d_year asc,  revenue desc;
//        //*****************************************************************
//
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
//        logical_operator::Logical_Scan("CUSTOMER"));
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("C_CITY", boost::any(std::string("UNITED
//            KI1")), ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("C_CITY", boost::any(std::string("UNITED
//            KI5")), ValueConstantPredicate, EQUAL)); //Vergleich wert in
//            Spalte mit Konstante
//            knf_expr.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr2;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_CITY", boost::any(std::string("UNITED
//            KI1")), ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("S_CITY", boost::any(std::string("UNITED
//            KI5")), ValueConstantPredicate, EQUAL)); //Vergleich wert in
//            Spalte mit Konstante
//            knf_expr2.disjunctions.push_back(d);
//        }
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_c_city(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_s_city(new
//        logical_operator::Logical_ComplexSelection(knf_expr2, LOOKUP,
//        default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_c_city(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_s_city(new
//        logical_operator::Logical_ComplexSelection(knf_expr2, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//        KNF_Selection_Expression knf_expr3;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1992),
//            ValueConstantPredicate, GREATER_EQUAL));
//            knf_expr3.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, LESSER_EQUAL));
//            knf_expr3.disjunctions.push_back(d);
//        }
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        complex_selection_on_date(new
//        logical_operator::Logical_ComplexSelection(knf_expr3, LOOKUP,
//        default_device_constraint));
//        //hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        boost::shared_ptr<logical_operator::Logical_Join> join_customer(new
//        logical_operator::Logical_Join("C_CUSTKEY", "LO_CUSTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//        std::list<std::string> groupsorting_column_names;
//        groupsorting_column_names.push_back("C_CITY");
//        groupsorting_column_names.push_back("S_CITY");
//        groupsorting_column_names.push_back("D_YEAR");
//        boost::shared_ptr<logical_operator::Logical_Sort> grouporder(new
//        logical_operator::Logical_Sort(groupsorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("C_CITY");
//        grouping_column_names.push_back("S_CITY");
//        grouping_column_names.push_back("D_YEAR");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("LO_REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//
//        selection_s_city->setLeft(scan_supplier);
//        selection_c_city->setLeft(scan_customer);
//
//        complex_selection_on_date->setLeft(scan_date);
//
//        join_customer->setLeft(selection_c_city);
//        join_customer->setRight(scan_lineorder);
//
//        join_supplier->setLeft(selection_s_city);
//        join_supplier->setRight(join_customer);
//
//        join_date->setLeft(complex_selection_on_date);
//        join_date->setRight(join_supplier);
//
//
//        grouporder->setLeft(join_date);
//        group->setLeft(grouporder);
//        order->setLeft(group);
//
//        return boost::make_shared<LogicalQueryPlan>(order);
//    }
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q33_plan() {
//        //*****************************************************************
//        //--Q3.3
//        //select c_city, s_city, d_year, sum(lo_revenue)
//        //as  revenue
//        //from customer, lineorder, supplier, dates
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_orderdate = d_datekey
//        //and  (c_city='UNITED KI1'
//        //or c_city='UNITED KI5')
//        //and (s_city='UNITED KI1'
//        //or s_city='UNITED KI5')
//        //and d_year >= 1992 and d_year <= 1997
//        //group by c_city, s_city, d_year
//        //order by d_year asc,  revenue desc;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1992),
//            ValueConstantPredicate, GREATER_EQUAL));
//            knf_expr_date.disjunctions.push_back(d);
//        }
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, LESSER_EQUAL));
//            knf_expr_date.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_customer;
//        {
//            Disjunction d;
//            d.push_back(Predicate("C_CITY", boost::any(std::string("UNITED
//            KI1")), ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("C_CITY", boost::any(std::string("UNITED
//            KI5")), ValueConstantPredicate, EQUAL)); //Vergleich wert in
//            Spalte mit Konstante
//            knf_expr_customer.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_CITY", boost::any(std::string("UNITED
//            KI1")), ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("S_CITY", boost::any(std::string("UNITED
//            KI5")), ValueConstantPredicate, EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//
//        InvisibleJoinSelectionList dimensions;
//        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
//        Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, default_device_constraint));
//
//        std::list<std::string> ordersorting_column_names;
//        ordersorting_column_names.push_back("C_CITY");
//        ordersorting_column_names.push_back("S_CITY");
//        ordersorting_column_names.push_back("D_YEAR");
//        boost::shared_ptr<logical_operator::Logical_Sort> grouporder(new
//        logical_operator::Logical_Sort(ordersorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("C_CITY");
//        grouping_column_names.push_back("S_CITY");
//        grouping_column_names.push_back("D_YEAR");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("LO_REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//
//        inv_join->setLeft(scan_lineorder);
//
////        grouporder->setLeft(inv_join);
//        group->setLeft(inv_join);
//        order->setLeft(group);
//
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//    }
//#endif
//
//    bool SSB_Q33(ClientPtr client) {
//	return optimize_execute_print("SSB Query 3.3", *SSB_Q33_plan(), client);
//    }
//
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q34_plan() {
//       //*****************************************************************
//        //--Q3.4
//        //select c_city, s_city, d_year, sum(lo_revenue)
//        //as  revenue
//        //from customer, lineorder, supplier, dates
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_orderdate = d_datekey
//        //and  (c_city='UNITED KI1'
//        //or c_city='UNITED KI5')
//        //and (s_city='UNITED KI1'
//        //or s_city='UNITED KI5')
//        //and d_yearmonth = 'Dec1997'
//        //group by c_city, s_city, d_year
//        //order by d_year asc,  revenue desc;
//       //*****************************************************************
//
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
//        logical_operator::Logical_Scan("CUSTOMER"));
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("C_CITY", boost::any(std::string("UNITED
//            KI1")), ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("C_CITY", boost::any(std::string("UNITED
//            KI5")), ValueConstantPredicate, EQUAL)); //Vergleich wert in
//            Spalte mit Konstante
//            knf_expr.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr2;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_CITY", boost::any(std::string("UNITED
//            KI1")), ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("S_CITY", boost::any(std::string("UNITED
//            KI5")), ValueConstantPredicate, EQUAL)); //Vergleich wert in
//            Spalte mit Konstante
//            knf_expr2.disjunctions.push_back(d);
//        }
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_c_city(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_s_city(new
//        logical_operator::Logical_ComplexSelection(knf_expr2, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_date(new logical_operator::Logical_Selection("D_YEARMONTH",
//        boost::any(std::string("Dec1997")), EQUAL, LOOKUP,
//        default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_c_city(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_s_city(new
//        logical_operator::Logical_ComplexSelection(knf_expr2, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_date(new logical_operator::Logical_Selection("D_YEARMONTH",
//        boost::any(std::string("Dec1997")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//        boost::shared_ptr<logical_operator::Logical_Join> join_customer(new
//        logical_operator::Logical_Join("C_CUSTKEY", "LO_CUSTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//        std::list<std::string> groupsorting_column_names;
//        groupsorting_column_names.push_back("C_CITY");
//        groupsorting_column_names.push_back("S_CITY");
//        groupsorting_column_names.push_back("D_YEAR");
//        boost::shared_ptr<logical_operator::Logical_Sort> grouporder(new
//        logical_operator::Logical_Sort(groupsorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("C_CITY");
//        grouping_column_names.push_back("S_CITY");
//        grouping_column_names.push_back("D_YEAR");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("LO_REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//
//        selection_s_city->setLeft(scan_supplier);
//        selection_c_city->setLeft(scan_customer);
//
//        selection_date->setLeft(scan_date);
//
//        join_customer->setLeft(selection_c_city);
//        join_customer->setRight(scan_lineorder);
//
//        join_supplier->setLeft(selection_s_city);
//        join_supplier->setRight(join_customer);
//
//        join_date->setLeft(selection_date);
//        join_date->setRight(join_supplier);
//
//
//        grouporder->setLeft(join_date);
//        group->setLeft(grouporder);
//        order->setLeft(group);
//
//        return boost::make_shared<LogicalQueryPlan>(order);
//    }
//
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q34_plan() {
//       //*****************************************************************
//        //--Q3.4
//        //select c_city, s_city, d_year, sum(lo_revenue)
//        //as  revenue
//        //from customer, lineorder, supplier, dates
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_orderdate = d_datekey
//        //and  (c_city='UNITED KI1'
//        //or c_city='UNITED KI5')
//        //and (s_city='UNITED KI1'
//        //or s_city='UNITED KI5')
//        //and d_yearmonth = 'Dec1997'
//        //group by c_city, s_city, d_year
//        //order by d_year asc,  revenue desc;
//       //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEARMONTH",
//            boost::any(std::string("Dec1997")), ValueConstantPredicate,
//            EQUAL));
//            knf_expr_date.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_customer;
//        {
//            Disjunction d;
//            d.push_back(Predicate("C_CITY", boost::any(std::string("UNITED
//            KI1")), ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("C_CITY", boost::any(std::string("UNITED
//            KI5")), ValueConstantPredicate, EQUAL)); //Vergleich wert in
//            Spalte mit Konstante
//            knf_expr_customer.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_CITY", boost::any(std::string("UNITED
//            KI1")), ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("S_CITY", boost::any(std::string("UNITED
//            KI5")), ValueConstantPredicate, EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//
//        InvisibleJoinSelectionList dimensions;
//        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
//        Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, default_device_constraint));
//
//        std::list<std::string> ordersorting_column_names;
//        ordersorting_column_names.push_back("C_CITY");
//        ordersorting_column_names.push_back("S_CITY");
//        ordersorting_column_names.push_back("D_YEAR");
//        boost::shared_ptr<logical_operator::Logical_Sort> grouporder(new
//        logical_operator::Logical_Sort(ordersorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("C_CITY");
//        grouping_column_names.push_back("S_CITY");
//        grouping_column_names.push_back("D_YEAR");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("LO_REVENUE", SUM));
//
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("LO_REVENUE");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, default_device_constraint));
//
//        inv_join->setLeft(scan_lineorder);
//
////        grouporder->setLeft(inv_join);
////        group->setLeft(grouporder);
//        group->setLeft(inv_join);
//        order->setLeft(group);
//
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//    }
//#endif
//
//
//    bool SSB_Q34(ClientPtr client) {
//	return optimize_execute_print("SSB Query 3.4", *SSB_Q34_plan(), client);
//    }
//
//
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q41_plan() {
//        //*****************************************************************
//        //--Q4.1
//        //select d_year, c_nation,
//        //sum(lo_revenue - lo_supplycost) as profit
//        //from dates, customer, supplier, part, lineorder
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_partkey = p_partkey
//        //and lo_orderdate = d_datekey
//        //and c_region = 'AMERICA'
//        //and s_region = 'AMERICA'
//        //and (p_mfgr = 'MFGR#1'
//        //or p_mfgr = 'MFGR#2')
//        //group by d_year, c_nation
//        //order by d_year, c_nation;
//        //*****************************************************************
//
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
//        logical_operator::Logical_Scan("CUSTOMER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_part(new
//        logical_operator::Logical_Scan("PART"));
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_MFGR", boost::any(std::string("MFGR#1")),
//            ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("P_MFGR", boost::any(std::string("MFGR#2")),
//            ValueConstantPredicate, EQUAL)); //Vergleich wert in Spalte mit
//            Konstante
//            knf_expr.disjunctions.push_back(d);
//        }
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_part(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_c_region(new logical_operator::Logical_Selection("C_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_part(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_c_region(new logical_operator::Logical_Selection("C_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_REVENUE",
//        "LO_SUPPLYCOST", "PROFIT", SUB, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_Join> join_customer(new
//        logical_operator::Logical_Join("C_CUSTKEY", "LO_CUSTKEY", LOOKUP,
//        default_device_constraint)); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        default_device_constraint)); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        default_device_constraint)); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_part(new
//        logical_operator::Logical_Join("P_PARTKEY", "LO_PARTKEY", LOOKUP,
//        default_device_constraint)); //GPU Join not supported
//
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("C_NATION");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("D_YEAR");
//        grouping_column_names.push_back("C_NATION");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("PROFIT", SUM));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//
//        selection_s_region->setLeft(scan_supplier);
//        selection_c_region->setLeft(scan_customer);
//        selection_part->setLeft(scan_part);
//
//
//        join_customer->setLeft(selection_c_region);
//        join_customer->setRight(scan_lineorder);
//        join_supplier->setLeft(selection_s_region);
//        join_supplier->setRight(join_customer);
//
//        join_date->setLeft(scan_date);
//        join_date->setRight(join_supplier);
//
//        join_part->setLeft(selection_part);
//        join_part->setRight(join_date);
//
//        column_algebra_operation_lineorder->setLeft(join_part);
//        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(order);
//
//
//        return boost::make_shared<LogicalQueryPlan>(group);
//    }
//
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q41_plan() {
//        //*****************************************************************
//        //--Q4.1
//        //select d_year, c_nation,
//        //sum(lo_revenue - lo_supplycost) as profit
//        //from dates, customer, supplier, part, lineorder
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_partkey = p_partkey
//        //and lo_orderdate = d_datekey
//        //and c_region = 'AMERICA'
//        //and s_region = 'AMERICA'
//        //and (p_mfgr = 'MFGR#1'
//        //or p_mfgr = 'MFGR#2')
//        //group by d_year, c_nation
//        //order by d_year, c_nation;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
////        KNF_Selection_Expression knf_expr;
////        {
////            Disjunction d;
////            d.push_back(Predicate("P_MFGR",
/// boost::any(std::string("MFGR#1")), ValueConstantPredicate, EQUAL));
/////YEAR<2013
////            d.push_back(Predicate("P_MFGR",
/// boost::any(std::string("MFGR#2")), ValueConstantPredicate, EQUAL));
/////Vergleich wert in Spalte mit Konstante
////            knf_expr.disjunctions.push_back(d);
////        }
////#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
////        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
/// selection_part(new logical_operator::Logical_ComplexSelection(knf_expr,
/// LOOKUP, default_device_constraint));
////        boost::shared_ptr<logical_operator::Logical_Selection>
/// selection_s_region(new logical_operator::Logical_Selection("S_REGION",
/// boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
/// default_device_constraint));
////        boost::shared_ptr<logical_operator::Logical_Selection>
/// selection_c_region(new logical_operator::Logical_Selection("C_REGION",
/// boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
/// default_device_constraint));
////#else
//
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
////        {
////            Disjunction d;
////            d.push_back(Predicate("D_YEARMONTH",
/// boost::any(std::string("Dec1997")), ValueConstantPredicate, EQUAL));
////            knf_expr_date.disjunctions.push_back(d);
////        }
//        KNF_Selection_Expression knf_expr_customer;
//        {
//            Disjunction d;
//            d.push_back(Predicate("C_REGION",
//            boost::any(std::string("AMERICA")), ValueConstantPredicate,
//            EQUAL));
//            knf_expr_customer.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_REGION",
//            boost::any(std::string("AMERICA")), ValueConstantPredicate,
//            EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_part;
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_MFGR", boost::any(std::string("MFGR#1")),
//            ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("P_MFGR", boost::any(std::string("MFGR#2")),
//            ValueConstantPredicate, EQUAL)); //Vergleich wert in Spalte mit
//            Konstante
//            knf_expr_part.disjunctions.push_back(d);
//        }
//        InvisibleJoinSelectionList dimensions;
//        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
//        Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//        dimensions.push_back(InvisibleJoinSelection("PART",
//        Predicate("P_PARTKEY", std::string("LO_PARTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_part));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_REVENUE",
//        "LO_SUPPLYCOST", "PROFIT", SUB, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("C_NATION");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("D_YEAR");
//        grouping_column_names.push_back("C_NATION");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("PROFIT", SUM));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        inv_join->setLeft(scan_lineorder);
//
//        column_algebra_operation_lineorder->setLeft(inv_join);
//
//        group->setLeft(column_algebra_operation_lineorder);
//        order->setLeft(group);
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//
//    }
//#endif
//
//    bool SSB_Q41(ClientPtr client) {
//	return optimize_execute_print("SSB Query 4.1", *SSB_Q41_plan(), client);
//    }
//
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q42_plan() {
//        //*****************************************************************
//        //--Q4.2
//        //select d_year, s_nation, p_category,
//        //sum(lo_revenue - lo_supplycost) as profit
//        //from dates, customer, supplier, part, lineorder
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_partkey = p_partkey
//        //and lo_orderdate = d_datekey
//        //and c_region = 'AMERICA'
//        //and s_region = 'AMERICA'
//        //and (d_year = 1997 or d_year = 1998)
//        //and (p_mfgr = 'MFGR#1'
//        //or p_mfgr = 'MFGR#2')
//        //group by d_year, s_nation, p_category
//        //order by d_year, s_nation, p_category;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
//        logical_operator::Logical_Scan("CUSTOMER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_part(new
//        logical_operator::Logical_Scan("PART"));
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_MFGR", boost::any(std::string("MFGR#1")),
//            ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("P_MFGR", boost::any(std::string("MFGR#2")),
//            ValueConstantPredicate, EQUAL)); //Vergleich wert in Spalte mit
//            Konstante
//            knf_expr.disjunctions.push_back(d);
//        }
//
//
//
//        KNF_Selection_Expression knf_expr2;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("D_YEAR", boost::any(1998),
//            ValueConstantPredicate, EQUAL)); //Vergleich wert in Spalte mit
//            Konstante
//            knf_expr2.disjunctions.push_back(d);
//        }
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_part(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_date(new
//        logical_operator::Logical_ComplexSelection(knf_expr2, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_c_region(new logical_operator::Logical_Selection("C_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_part(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_date(new
//        logical_operator::Logical_ComplexSelection(knf_expr2, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_region(new logical_operator::Logical_Selection("S_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_c_region(new logical_operator::Logical_Selection("C_REGION",
//        boost::any(std::string("AMERICA")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_REVENUE",
//        "LO_SUPPLYCOST", "PROFIT", SUB, default_device_constraint));
//
//
//        boost::shared_ptr<logical_operator::Logical_Join> join_customer(new
//        logical_operator::Logical_Join("C_CUSTKEY", "LO_CUSTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_part(new
//        logical_operator::Logical_Join("P_PARTKEY", "LO_PARTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("S_NATION");
//        sorting_column_names.push_back("P_CATEGORY");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//        //default_device_constraint));
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("D_YEAR");
//        grouping_column_names.push_back("S_NATION");
//        grouping_column_names.push_back("P_CATEGORY");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("PROFIT", SUM));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        selection_s_region->setLeft(scan_supplier);
//        selection_c_region->setLeft(scan_customer);
//        selection_part->setLeft(scan_part);
//        selection_date->setLeft(scan_date);
//        column_algebra_operation_lineorder->setLeft(scan_lineorder);
//
//        join_customer->setLeft(selection_c_region);
//        join_customer->setRight(scan_lineorder);
//
//        join_supplier->setLeft(selection_s_region);
//        join_supplier->setRight(join_customer);
//
//        join_date->setLeft(selection_date);
//        join_date->setRight(join_supplier);
//
//        join_part->setLeft(selection_part);
//        join_part->setRight(join_date);
//        column_algebra_operation_lineorder->setLeft(join_part);
//
//        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(order);
//
//        return boost::make_shared<LogicalQueryPlan>(group);
//    }
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q42_plan() {
//        //*****************************************************************
//        //--Q4.2
//        //select d_year, s_nation, p_category,
//        //sum(lo_revenue - lo_supplycost) as profit
//        //from dates, customer, supplier, part, lineorder
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_partkey = p_partkey
//        //and lo_orderdate = d_datekey
//        //and c_region = 'AMERICA'
//        //and s_region = 'AMERICA'
//        //and (d_year = 1997 or d_year = 1998)
//        //and (p_mfgr = 'MFGR#1'
//        //or p_mfgr = 'MFGR#2')
//        //group by d_year, s_nation, p_category
//        //order by d_year, s_nation, p_category;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, EQUAL));
//            d.push_back(Predicate("D_YEAR", boost::any(1998),
//            ValueConstantPredicate, EQUAL));
//            knf_expr_date.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_customer;
//        {
//            Disjunction d;
//            d.push_back(Predicate("C_REGION",
//            boost::any(std::string("AMERICA")), ValueConstantPredicate,
//            EQUAL));
//            knf_expr_customer.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_REGION",
//            boost::any(std::string("AMERICA")), ValueConstantPredicate,
//            EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_part;
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_MFGR", boost::any(std::string("MFGR#1")),
//            ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("P_MFGR", boost::any(std::string("MFGR#2")),
//            ValueConstantPredicate, EQUAL)); //Vergleich wert in Spalte mit
//            Konstante
//            knf_expr_part.disjunctions.push_back(d);
//        }
//        InvisibleJoinSelectionList dimensions;
//        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
//        Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//        dimensions.push_back(InvisibleJoinSelection("PART",
//        Predicate("P_PARTKEY", std::string("LO_PARTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_part));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_REVENUE",
//        "LO_SUPPLYCOST", "PROFIT", SUB, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("S_NATION");
//        sorting_column_names.push_back("P_CATEGORY");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//        //default_device_constraint));
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("D_YEAR");
//        grouping_column_names.push_back("S_NATION");
//        grouping_column_names.push_back("P_CATEGORY");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("PROFIT", SUM));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        inv_join->setLeft(scan_lineorder);
//
//        column_algebra_operation_lineorder->setLeft(inv_join);
//
//        group->setLeft(column_algebra_operation_lineorder);
//        order->setLeft(group);
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//    }
//#endif
//
//    bool SSB_Q42(ClientPtr client) {
//	return optimize_execute_print("SSB Query 4.2", *SSB_Q42_plan(), client);
//    }
//
//#ifndef USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES
//    LogicalQueryPlanPtr SSB_Q43_plan() {
//        //*****************************************************************
//        //--Q4.3
//        //select d_year, s_city, p_brand,
//        //sum(lo_revenue - lo_supplycost) as profit
//        //from dates, customer, supplier, part, lineorder
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_partkey = p_partkey
//        //and lo_orderdate = d_datekey
//        //and s_nation = 'UNITED STATES'
//        //and (d_year = 1997 or d_year = 1998)
//        //and p_category = 'MFGR#14'
//        //group by d_year, s_city, p_brand
//        //order by d_year, s_city, p_brand;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_date(new
//        logical_operator::Logical_Scan("DATES"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
//        logical_operator::Logical_Scan("SUPPLIER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
//        logical_operator::Logical_Scan("CUSTOMER"));
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_part(new
//        logical_operator::Logical_Scan("PART"));
//
//
//        KNF_Selection_Expression knf_expr;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, EQUAL)); //YEAR<2013
//            d.push_back(Predicate("D_YEAR", boost::any(1998),
//            ValueConstantPredicate, EQUAL)); //Vergleich wert in Spalte mit
//            Konstante
//            knf_expr.disjunctions.push_back(d);
//        }
//#ifdef ENABLE_GPU_ACCELERATED_VARCHAR_SCAN
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_date(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_nation(new logical_operator::Logical_Selection("S_NATION",
//        boost::any(std::string("UNITED STATES")), EQUAL, LOOKUP,
//        default_device_constraint));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_part(new logical_operator::Logical_Selection("P_CATEGORY",
//        boost::any(std::string("MFGR#14")), EQUAL, LOOKUP,
//        default_device_constraint));
//#else
//        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
//        selection_date(new
//        logical_operator::Logical_ComplexSelection(knf_expr, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_s_nation(new logical_operator::Logical_Selection("S_NATION",
//        boost::any(std::string("UNITED STATES")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//        boost::shared_ptr<logical_operator::Logical_Selection>
//        selection_part(new logical_operator::Logical_Selection("P_CATEGORY",
//        boost::any(std::string("MFGR#14")), EQUAL, LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY)));
//#endif
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_REVENUE",
//        "LO_SUPPLYCOST", "PROFIT", SUB, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_Join> join_customer(new
//        logical_operator::Logical_Join("C_CUSTKEY", "LO_CUSTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_supplier(new
//        logical_operator::Logical_Join("S_SUPPKEY", "LO_SUPPKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_date(new
//        logical_operator::Logical_Join("D_DATEKEY", "LO_ORDERDATE", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//        boost::shared_ptr<logical_operator::Logical_Join> join_part(new
//        logical_operator::Logical_Join("P_PARTKEY", "LO_PARTKEY", LOOKUP,
//        hype::DeviceConstraint(hype::CPU_ONLY))); //GPU Join not supported
//
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("S_CITY");
//        sorting_column_names.push_back("P_BRAND");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//        //default_device_constraint));
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("D_YEAR");
//        grouping_column_names.push_back("S_CITY");
//        grouping_column_names.push_back("P_BRAND");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("PROFIT", SUM));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        selection_s_nation->setLeft(scan_supplier);
//        selection_part->setLeft(scan_part);
//        selection_date->setLeft(scan_date);
//
//
//        join_customer->setLeft(scan_customer);
//        join_customer->setRight(scan_lineorder);
//
//        join_supplier->setLeft(selection_s_nation);
//        join_supplier->setRight(join_customer);
//
//        join_date->setLeft(selection_date);
//        join_date->setRight(join_supplier);
//
//        join_part->setLeft(selection_part);
//        join_part->setRight(join_date);
//
//        column_algebra_operation_lineorder->setLeft(join_part);
//        order->setLeft(column_algebra_operation_lineorder);
//        group->setLeft(order);
//
//        return boost::make_shared<LogicalQueryPlan>(group);
//    }
//#else
//    //USE_INVISIBLE_JOIN_FOR_STORED_PROCEDURES defined
//    LogicalQueryPlanPtr SSB_Q43_plan() {
//        //*****************************************************************
//        //--Q4.3
//        //select d_year, s_city, p_brand,
//        //sum(lo_revenue - lo_supplycost) as profit
//        //from dates, customer, supplier, part, lineorder
//        //where lo_custkey = c_custkey
//        //and lo_suppkey = s_suppkey
//        //and lo_partkey = p_partkey
//        //and lo_orderdate = d_datekey
//        //and s_nation = 'UNITED STATES'
//        //and (d_year = 1997 or d_year = 1998)
//        //and p_category = 'MFGR#14'
//        //group by d_year, s_city, p_brand
//        //order by d_year, s_city, p_brand;
//        //*****************************************************************
//        hype::DeviceConstraint default_device_constraint =
//        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
//
////
////        boost::shared_ptr<logical_operator::Logical_Selection>
/// selection_s_nation(new logical_operator::Logical_Selection("S_NATION",
/// boost::any(std::string("UNITED STATES")), EQUAL, LOOKUP,
/// default_device_constraint));
////        boost::shared_ptr<logical_operator::Logical_Selection>
/// selection_part(new logical_operator::Logical_Selection("P_CATEGORY",
/// boost::any(std::string("MFGR#14")), EQUAL, LOOKUP,
/// default_device_constraint));
////
//
//
//        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(new
//        logical_operator::Logical_Scan("LINEORDER"));
//        KNF_Selection_Expression knf_expr_date;
//        {
//            Disjunction d;
//            d.push_back(Predicate("D_YEAR", boost::any(1997),
//            ValueConstantPredicate, EQUAL));
//            d.push_back(Predicate("D_YEAR", boost::any(1998),
//            ValueConstantPredicate, EQUAL));
//            knf_expr_date.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_customer;
////        {
////            Disjunction d;
////            d.push_back(Predicate("C_REGION",
/// boost::any(std::string("AMERICA")), ValueConstantPredicate, EQUAL));
////            knf_expr_customer.disjunctions.push_back(d);
////        }
//        KNF_Selection_Expression knf_expr_supplier;
//        {
//            Disjunction d;
//            d.push_back(Predicate("S_NATION", boost::any(std::string("UNITED
//            STATES")), ValueConstantPredicate, EQUAL));
//            knf_expr_supplier.disjunctions.push_back(d);
//        }
//        KNF_Selection_Expression knf_expr_part;
//        {
//            Disjunction d;
//            d.push_back(Predicate("P_CATEGORY",
//            boost::any(std::string("MFGR#14")), ValueConstantPredicate,
//            EQUAL)); //YEAR<2013
//            knf_expr_part.disjunctions.push_back(d);
//        }
//        InvisibleJoinSelectionList dimensions;
//        dimensions.push_back(InvisibleJoinSelection("CUSTOMER",
//        Predicate("C_CUSTKEY", std::string("LO_CUSTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_customer));
//        dimensions.push_back(InvisibleJoinSelection("SUPPLIER",
//        Predicate("S_SUPPKEY", std::string("LO_SUPPKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_supplier));
//        dimensions.push_back(InvisibleJoinSelection("DATES",
//        Predicate("D_DATEKEY",
//        std::string("LO_ORDERDATE"),ValueValuePredicate, EQUAL),
//        knf_expr_date));
//        dimensions.push_back(InvisibleJoinSelection("PART",
//        Predicate("P_PARTKEY", std::string("LO_PARTKEY"),ValueValuePredicate,
//        EQUAL), knf_expr_part));
//
//        boost::shared_ptr<query_processing::logical_operator::Logical_InvisibleJoin>
//        inv_join (new
//        query_processing::logical_operator::Logical_InvisibleJoin(
//                                                                                               dimensions, LOOKUP, default_device_constraint));
//
//        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
//        column_algebra_operation_lineorder(new
//        logical_operator::Logical_ColumnAlgebraOperator("LO_REVENUE",
//        "LO_SUPPLYCOST", "PROFIT", SUB, default_device_constraint));
//
//        std::list<std::string> sorting_column_names;
//        sorting_column_names.push_back("D_YEAR");
//        sorting_column_names.push_back("S_CITY");
//        sorting_column_names.push_back("P_BRAND");
//        boost::shared_ptr<logical_operator::Logical_Sort> order(new
//        logical_operator::Logical_Sort(sorting_column_names, ASCENDING,
//        LOOKUP, hype::DeviceConstraint(hype::CPU_ONLY)));
//        //default_device_constraint));
//
//        std::list<std::string> grouping_column_names;
//        grouping_column_names.push_back("D_YEAR");
//        grouping_column_names.push_back("S_CITY");
//        grouping_column_names.push_back("P_BRAND");
//        std::list<std::pair<string, AggregationMethod> >
//        aggregation_functions;
//        aggregation_functions.push_back(make_pair("PROFIT", SUM));
//        boost::shared_ptr<logical_operator::Logical_Groupby> group(new
//        logical_operator::Logical_Groupby(grouping_column_names,
//        aggregation_functions, LOOKUP, default_device_constraint));
//
//        inv_join->setLeft(scan_lineorder);
//
//        column_algebra_operation_lineorder->setLeft(inv_join);
////        order->setLeft(column_algebra_operation_lineorder);
////        group->setLeft(order);
//
//        group->setLeft(column_algebra_operation_lineorder);
//        order->setLeft(group);
//
//#ifndef COGADB_USE_INVISIBLE_JON_PLANS_ONLY
//        return boost::make_shared<LogicalQueryPlan>(order);
//#else
//	return boost::make_shared<LogicalQueryPlan>(inv_join);
//#endif
//    }
//#endif
//
//    bool SSB_Q43(ClientPtr client) {
//	return optimize_execute_print("SSB Query 4.3", *SSB_Q43_plan(), client);
//    }

//    bool SSB_Q11(ClientPtr client) {
//        return SQL::commandlineExec("select sum(lo_extendedprice*lo_discount)
//        as revenue from lineorder, dates where lo_orderdate = d_datekey and
//        d_year = 1993 and lo_discount between 1 and 3 and lo_quantity < 25;",
//                client);
//    }
//
//    bool SSB_Q12(ClientPtr client) {
//        return SQL::commandlineExec("select sum(lo_extendedprice*lo_discount)
//        as revenue from lineorder, dates where lo_orderdate = d_datekey and
//        d_yearmonthnum = 199401 and lo_discount between 4 and 6 and
//        lo_quantity between 26 and 35;",
//                client);
//    }
//
//    bool SSB_Q13(ClientPtr client) {
//        return SQL::commandlineExec("select sum(lo_extendedprice*lo_discount)
//        as revenue from lineorder, dates where lo_orderdate = d_datekey and
//        d_weeknuminyear = 6 and d_year = 1994 and lo_discount between 5 and 7
//        and lo_quantity between 26 and 35;",
//                client);
//    }
//
//    bool SSB_Q21(ClientPtr client) {
//        return SQL::commandlineExec("select sum(lo_revenue), d_year, p_brand
//        from lineorder, dates, part, supplier where lo_orderdate = d_datekey
//        and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_category =
//        'MFGR#12' and s_region = 'AMERICA' group by d_year, p_brand order by
//        d_year, p_brand;",
//                client);
//    }
//
//    bool SSB_Q22(ClientPtr client) {
//        return SQL::commandlineExec("select sum(lo_revenue), d_year, p_brand
//        from lineorder, dates, part, supplier where lo_orderdate = d_datekey
//        and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_brand
//        between 'MFGR#2221' and 'MFGR#2228' and s_region = 'ASIA' group by
//        d_year, p_brand order by d_year, p_brand;",
//                client);
//    }
//
//    bool SSB_Q23(ClientPtr client) {
//        return SQL::commandlineExec("select sum(lo_revenue), d_year, p_brand
//        from lineorder, dates, part, supplier where lo_orderdate = d_datekey
//        and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_brand=
//        'MFGR#2239' and s_region = 'EUROPE' group by d_year, p_brand order by
//        d_year, p_brand;",
//                client);
//    }
//
//    bool SSB_Q31(ClientPtr client) {
//        return SQL::commandlineExec("select c_nation, s_nation, d_year,
//        sum(lo_revenue) from customer, lineorder, supplier, dates where
//        lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate =
//        d_datekey and c_region = 'ASIA' and s_region = 'ASIA' and d_year >=
//        1992 and d_year <= 1997 group by c_nation, s_nation, d_year order by
//        d_year asc, lo_revenue desc;",
//                client);
//    }
//
//    bool SSB_Q32(ClientPtr client) {
//        return SQL::commandlineExec("select c_city, s_city, d_year,
//        sum(lo_revenue) from customer, lineorder, supplier, dates where
//        lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate =
//        d_datekey and c_nation = 'UNITED STATES' and s_nation = 'UNITED
//        STATES' and d_year >= 1992 and d_year <= 1997 group by c_city, s_city,
//        d_year order by d_year asc, lo_revenue desc;",
//                client);
//    }
//
//    bool SSB_Q33(ClientPtr client) {
//        return SQL::commandlineExec("select c_city, s_city, d_year,
//        sum(lo_revenue) from customer, lineorder, supplier, dates where
//        lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate =
//        d_datekey and (c_city='UNITED KI1' or c_city='UNITED KI5') and
//        (s_city='UNITED KI1' or s_city='UNITED KI5') and d_year >= 1992 and
//        d_year <= 1997 group by c_city, s_city, d_year order by d_year asc,
//        lo_revenue desc;",
//                client);
//    }
//
//    bool SSB_Q34(ClientPtr client) {
//        return SQL::commandlineExec("select c_city, s_city, d_year,
//        sum(lo_revenue) from customer, lineorder, supplier, dates where
//        lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_orderdate =
//        d_datekey and (c_city='UNITED KI1' or c_city='UNITED KI5') and
//        (s_city='UNITED KI1' or s_city='UNITED KI5') and d_yearmonth =
//        'Dec1997' group by c_city, s_city, d_year order by d_year asc,
//        lo_revenue desc;",
//                client);
//    }
//
//    bool SSB_Q41(ClientPtr client) {
//        return SQL::commandlineExec("select d_year, c_nation, sum(lo_revenue -
//        lo_supplycost) as profit from dates, customer, supplier, part,
//        lineorder where lo_custkey = c_custkey and lo_suppkey = s_suppkey and
//        lo_partkey = p_partkey and lo_orderdate = d_datekey and c_region =
//        'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or p_mfgr =
//        'MFGR#2') group by d_year, c_nation order by d_year, c_nation;",
//                client);
//    }
//
//    bool SSB_Q42(ClientPtr client) {
//        return SQL::commandlineExec("select d_year, s_nation, p_category,
//        sum(lo_revenue - lo_supplycost) as profit from dates, customer,
//        supplier, part, lineorder where lo_custkey = c_custkey and lo_suppkey
//        = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey
//        and c_region = 'AMERICA' and s_region = 'AMERICA' and (d_year = 1997
//        or d_year = 1998) and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') group
//        by d_year, s_nation, p_category order by d_year, s_nation,
//        p_category;",
//                client);
//    }
//
//
//    bool SSB_Q43(ClientPtr client) {
//        return SQL::commandlineExec("select d_year, s_city, p_brand,
//        sum(lo_revenue - lo_supplycost) as profit from dates, customer,
//        supplier, part, lineorder where lo_custkey = c_custkey and lo_suppkey
//        = s_suppkey and lo_partkey = p_partkey and lo_orderdate = d_datekey
//        and s_nation = 'UNITED STATES' and (d_year = 1997 or d_year = 1998)
//        and p_category = 'MFGR#14' group by d_year, s_city, p_brand order by
//        d_year, s_city, p_brand;",
//                client);
//    }

bool SSB_Q11(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(lo_extendedprice*lo_discount) as revenue from lineorder JOIN "
      "dates ON (lo_orderdate = d_datekey) where d_year = 1993 and lo_discount "
      "between 1 and 3 and lo_quantity < 25;",
      client);
}

bool SSB_Q12(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(lo_extendedprice*lo_discount) as revenue from lineorder JOIN "
      "dates ON (lo_orderdate = d_datekey) where d_yearmonthnum = 199401 and "
      "lo_discount between 4 and 6 and lo_quantity between 26 and 35;",
      client);
}

bool SSB_Q13(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(lo_extendedprice*lo_discount) as revenue from lineorder JOIN "
      "dates ON (lo_orderdate = d_datekey) where d_weeknuminyear = 6 and "
      "d_year = 1994 and lo_discount between 5 and 7 and lo_quantity between "
      "26 and 35;",
      client);
}

bool SSB_Q21(ClientPtr client) {
  //        return SQL::commandlineExec("select sum(lo_revenue), d_year, p_brand
  //        from lineorder JOIN supplier ON (lo_suppkey = s_suppkey) JOIN part
  //        ON (lo_partkey = p_partkey) JOIN dates ON (lo_orderdate = d_datekey)
  //        where p_category = 'MFGR#12' and s_region = 'AMERICA' group by
  //        d_year, p_brand order by d_year, p_brand;",
  //                client);
  return SQL::commandlineExec(
      "select sum(lo_revenue), d_year, p_brand from dates JOIN (part JOIN "
      "(supplier JOIN lineorder ON (s_suppkey = lo_suppkey)) ON "
      "(p_partkey=lo_partkey)) ON (d_datekey=lo_orderdate) where p_category = "
      "'MFGR#12' and s_region = 'AMERICA' group by d_year, p_brand order by "
      "d_year, p_brand;",
      client);
}

bool SSB_Q22(ClientPtr client) {
  //        return SQL::commandlineExec("select sum(lo_revenue), d_year, p_brand
  //        from lineorder JOIN supplier ON (lo_suppkey = s_suppkey) JOIN part
  //        ON (lo_partkey = p_partkey) JOIN dates ON (lo_orderdate = d_datekey)
  //        where p_brand between 'MFGR#2221' and 'MFGR#2228' and s_region =
  //        'ASIA' group by d_year, p_brand order by d_year, p_brand;",
  //                client);
  return SQL::commandlineExec(
      "select sum(lo_revenue), d_year, p_brand from dates JOIN (part JOIN "
      "(supplier JOIN lineorder ON (s_suppkey = lo_suppkey)) ON "
      "(p_partkey=lo_partkey)) ON (d_datekey=lo_orderdate) where p_brand "
      "between 'MFGR#2221' and 'MFGR#2228' and s_region = 'ASIA' group by "
      "d_year, p_brand order by d_year, p_brand;",
      client);
}

bool SSB_Q23(ClientPtr client) {
  //        return SQL::commandlineExec("select sum(lo_revenue), d_year, p_brand
  //        from lineorder JOIN supplier ON (lo_suppkey = s_suppkey) JOIN part
  //        ON (lo_partkey = p_partkey) JOIN dates ON (lo_orderdate = d_datekey)
  //        where p_brand= 'MFGR#2239' and s_region = 'EUROPE' group by d_year,
  //        p_brand order by d_year, p_brand;",
  //                client);
  return SQL::commandlineExec(
      "select sum(lo_revenue), d_year, p_brand from dates JOIN (part JOIN "
      "(supplier JOIN lineorder ON (s_suppkey = lo_suppkey)) ON "
      "(p_partkey=lo_partkey)) ON (d_datekey=lo_orderdate) where p_brand= "
      "'MFGR#2239' and s_region = 'EUROPE' group by d_year, p_brand order by "
      "d_year, p_brand;",
      client);
}

bool SSB_Q31(ClientPtr client) {
  //        return SQL::commandlineExec("select c_nation, s_nation, d_year,
  //        sum(lo_revenue) from lineorder JOIN supplier ON (lo_suppkey =
  //        s_suppkey) JOIN customer ON (lo_custkey = c_custkey) JOIN dates ON
  //        (lo_orderdate = d_datekey) where c_region = 'ASIA' and s_region =
  //        'ASIA' and d_year >= 1992 and d_year <= 1997 group by c_nation,
  //        s_nation, d_year order by d_year asc, lo_revenue desc;",
  //                client);
  return SQL::commandlineExec(
      "select c_nation, s_nation, d_year, sum(lo_revenue) from dates JOIN "
      "(customer JOIN (supplier JOIN lineorder ON (s_suppkey = lo_suppkey)) ON "
      "(c_custkey=lo_custkey)) ON (d_datekey=lo_orderdate) where c_region = "
      "'ASIA' and s_region = 'ASIA' and d_year >= 1992 and d_year <= 1997 "
      "group by c_nation, s_nation, d_year order by d_year asc, lo_revenue "
      "desc;",
      client);
}

bool SSB_Q32(ClientPtr client) {
  //        return SQL::commandlineExec("select c_city, s_city, d_year,
  //        sum(lo_revenue) from lineorder JOIN supplier ON (lo_suppkey =
  //        s_suppkey) JOIN customer ON (lo_custkey = c_custkey) JOIN dates ON
  //        (lo_orderdate = d_datekey) where c_nation = 'UNITED STATES' and
  //        s_nation = 'UNITED STATES' and d_year >= 1992 and d_year <= 1997
  //        group by c_city, s_city, d_year order by d_year asc, lo_revenue
  //        desc;",
  //                client);
  return SQL::commandlineExec(
      "select c_city, s_city, d_year, sum(lo_revenue) from dates JOIN "
      "(customer JOIN (supplier JOIN lineorder ON (s_suppkey = lo_suppkey)) ON "
      "(c_custkey=lo_custkey)) ON (d_datekey=lo_orderdate) where c_nation = "
      "'UNITED STATES' and s_nation = 'UNITED STATES' and d_year >= 1992 and "
      "d_year <= 1997 group by c_city, s_city, d_year order by d_year asc, "
      "lo_revenue desc;",
      client);
}

bool SSB_Q33(ClientPtr client) {
  //        return SQL::commandlineExec("select c_city, s_city, d_year,
  //        sum(lo_revenue) from lineorder JOIN supplier ON (lo_suppkey =
  //        s_suppkey) JOIN customer ON (lo_custkey = c_custkey) JOIN dates ON
  //        (lo_orderdate = d_datekey) where (c_city='UNITED KI1' or
  //        c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED
  //        KI5') and d_year >= 1992 and d_year <= 1997 group by c_city, s_city,
  //        d_year order by d_year asc, lo_revenue desc;",
  //                client);
  return SQL::commandlineExec(
      "select c_city, s_city, d_year, sum(lo_revenue) from dates JOIN "
      "(customer JOIN (supplier JOIN lineorder ON (s_suppkey = lo_suppkey)) ON "
      "(c_custkey=lo_custkey)) ON (d_datekey=lo_orderdate) where "
      "(c_city='UNITED KI1' or c_city='UNITED KI5') and (s_city='UNITED KI1' "
      "or s_city='UNITED KI5') and d_year >= 1992 and d_year <= 1997 group by "
      "c_city, s_city, d_year order by d_year asc, lo_revenue desc;",
      client);
}

bool SSB_Q34(ClientPtr client) {
  //        return SQL::commandlineExec("select c_city, s_city, d_year,
  //        sum(lo_revenue) from lineorder JOIN supplier ON (lo_suppkey =
  //        s_suppkey) JOIN customer ON (lo_custkey = c_custkey) JOIN dates ON
  //        (lo_orderdate = d_datekey) where (c_city='UNITED KI1' or
  //        c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED
  //        KI5') and d_yearmonth = 'Dec1997' group by c_city, s_city, d_year
  //        order by d_year asc, lo_revenue desc;",
  //                client);
  return SQL::commandlineExec(
      "select c_city, s_city, d_year, sum(lo_revenue) from dates JOIN "
      "(customer JOIN (supplier JOIN lineorder ON (s_suppkey = lo_suppkey)) ON "
      "(c_custkey=lo_custkey)) ON (d_datekey=lo_orderdate) where "
      "(c_city='UNITED KI1' or c_city='UNITED KI5') and (s_city='UNITED KI1' "
      "or s_city='UNITED KI5') and d_yearmonth = 'Dec1997' group by c_city, "
      "s_city, d_year order by d_year asc, lo_revenue desc;",
      client);
}

bool SSB_Q41(ClientPtr client) {
  //        return SQL::commandlineExec("select d_year, c_nation, sum(lo_revenue
  //        - lo_supplycost) as profit from lineorder JOIN supplier ON
  //        (lo_suppkey = s_suppkey) JOIN customer ON (lo_custkey = c_custkey)
  //        JOIN part ON (lo_partkey = p_partkey) JOIN dates ON (lo_orderdate =
  //        d_datekey) where c_region = 'AMERICA' and s_region = 'AMERICA' and
  //        (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year, c_nation
  //        order by d_year, c_nation;",
  //                client);
  return SQL::commandlineExec(
      "select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit from "
      "dates JOIN (part JOIN (customer JOIN (supplier JOIN lineorder ON "
      "(s_suppkey = lo_suppkey)) ON (c_custkey=lo_custkey)) ON "
      "(p_partkey=lo_partkey)) ON (d_datekey=lo_orderdate) where c_region = "
      "'AMERICA' and s_region = 'AMERICA' and (p_mfgr = 'MFGR#1' or p_mfgr = "
      "'MFGR#2') group by d_year, c_nation order by d_year, c_nation;",
      client);
}

bool SSB_Q42(ClientPtr client) {
  //        return SQL::commandlineExec("select d_year, s_nation, p_category,
  //        sum(lo_revenue - lo_supplycost) as profit from lineorder JOIN
  //        supplier ON (lo_suppkey = s_suppkey) JOIN dates ON (lo_orderdate =
  //        d_datekey) JOIN part ON (lo_partkey = p_partkey) JOIN customer ON
  //        (lo_custkey = c_custkey) where c_region = 'AMERICA' and s_region =
  //        'AMERICA' and (d_year = 1997 or d_year = 1998) and (p_mfgr =
  //        'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year, s_nation, p_category
  //        order by d_year, s_nation, p_category;",
  //                client);
  return SQL::commandlineExec(
      "select d_year, s_nation, p_category, sum(lo_revenue - lo_supplycost) as "
      "profit from customer JOIN ( part JOIN (dates JOIN (supplier JOIN "
      "lineorder ON (s_suppkey = lo_suppkey)) ON (d_datekey=lo_orderdate)) ON "
      "(p_partkey=lo_partkey)) ON (c_custkey=lo_custkey) where c_region = "
      "'AMERICA' and s_region = 'AMERICA' and (d_year = 1997 or d_year = 1998) "
      "and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year, s_nation, "
      "p_category order by d_year, s_nation, p_category;",
      client);
}

bool SSB_Q43(ClientPtr client) {
  //        return SQL::commandlineExec("select d_year, s_city, p_brand,
  //        sum(lo_revenue - lo_supplycost) as profit from lineorder JOIN
  //        supplier ON (lo_suppkey = s_suppkey) JOIN dates ON (lo_orderdate =
  //        d_datekey) JOIN part ON (lo_partkey = p_partkey) JOIN customer ON
  //        (lo_custkey = c_custkey) where s_nation = 'UNITED STATES' and
  //        (d_year = 1997 or d_year = 1998) and p_category = 'MFGR#14' group by
  //        d_year, s_city, p_brand order by d_year, s_city, p_brand;",
  //                client);
  return SQL::commandlineExec(
      "select d_year, s_city, p_brand, sum(lo_revenue - lo_supplycost) as "
      "profit from customer JOIN ( part JOIN (dates JOIN (supplier JOIN "
      "lineorder ON (s_suppkey = lo_suppkey)) ON (d_datekey=lo_orderdate)) ON "
      "(p_partkey=lo_partkey)) ON (c_custkey=lo_custkey) where s_nation = "
      "'UNITED STATES' and (d_year = 1997 or d_year = 1998) and p_category = "
      "'MFGR#14' group by d_year, s_city, p_brand order by d_year, s_city, "
      "p_brand;",
      client);
}

//    lineorder -> supplier -> dates -> part -> customer
//
// customer JOIN ( part JOIN (dates JOIN (supplier JOIN lineorder ON (s_suppkey
// = lo_suppkey)) ON (d_datekey=lo_orderdate)) ON (p_partkey=lo_partkey))  ON
// (c_custkey=lo_custkey)
//
//    lineorder -> supplier -> customer -> part -> dates
//            dates JOIN (part JOIN (customer JOIN (supplier JOIN lineorder ON
//            (s_suppkey = lo_suppkey)) ON (c_custkey=lo_custkey)) ON
//            (p_partkey=lo_partkey)) ON (d_datekey=lo_orderdate)

bool SSB_Selection_Query(ClientPtr client) {
  return SQL::commandlineExec(
      "select * from lineorder where lo_discount between 4 and 6 and "
      "lo_quantity between 26 and 35;",
      client);
}

bool SSB_SemiJoin_Query(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(
      new logical_operator::Logical_Scan("LINEORDER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_date(
      new logical_operator::Logical_Scan("DATES"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
  //        logical_operator::Logical_Scan("SUPPLIER"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
  //        logical_operator::Logical_Scan("CUSTOMER"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_part(new
  //        logical_operator::Logical_Scan("PART"));

  KNF_Selection_Expression knf_expr;
  {
    Disjunction d;
    d.push_back(Predicate("D_YEAR", boost::any(1995), ValueConstantPredicate,
                          GREATER_EQUAL));
    knf_expr.disjunctions.push_back(d);
  }
  {
    Disjunction d;
    d.push_back(Predicate("D_YEAR", boost::any(1996), ValueConstantPredicate,
                          LESSER_EQUAL));
    knf_expr.disjunctions.push_back(d);
  }
  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_date(new logical_operator::Logical_ComplexSelection(
          knf_expr, LOOKUP,
          default_device_constraint));  // hype::DeviceConstraint(hype::CPU_ONLY)));

  boost::shared_ptr<logical_operator::Logical_Join> join(
      new logical_operator::Logical_Join(
          "D_DATEKEY", "LO_ORDERDATE", LEFT_SEMI_JOIN,
          hype::DeviceConstraint(hype::CPU_ONLY)));  // GPU Join not supported

  complex_selection_on_date->setLeft(scan_date);
  join->setLeft(complex_selection_on_date);
  join->setRight(scan_lineorder);

  LogicalQueryPlanPtr log_plan = boost::make_shared<LogicalQueryPlan>(join);

  return optimize_execute_print("ssb_semi_join", *log_plan, client);
}

}  // end namespace CogaDB
