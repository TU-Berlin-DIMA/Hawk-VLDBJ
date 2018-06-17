
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

using namespace std;

namespace CoGaDB {
using namespace query_processing;

bool Unittest_Create_Denormalized_Star_Schema_Benchmark_Database(
    ClientPtr client) {
  std::ostream& out = client->getOutputStream();

  out << "Denormalizing existing Star Schema Database: '"
      << RuntimeConfiguration::instance().getPathToDatabase() << "'"
      << std::endl;

  std::string query =
      "select D_DATE, D_DAYOFWEEK, D_MONTH, D_YEAR, D_YEARMONTHNUM, "
      "D_YEARMONTH, D_DAYNUMINWEEK, D_DAYNUMINMONTH, D_DAYNUMINYEAR, "
      "D_MONTHNUMINYEAR, D_WEEKNUMINYEAR, D_SELLINGSEASON, D_LASTDAYINWEEKFL, "
      "D_LASTDAYINMONTHFL, D_HOLIDAYFL, D_WEEKDAYFL, P_NAME, P_MFGR, "
      "P_CATEGORY, P_BRAND, P_COLOR, P_TYPE, P_SIZE, P_CONTAINER, S_NAME, "
      "S_ADDRESS, S_CITY, S_NATION, S_REGION, S_PHONE, C_NAME, C_ADDRESS, "
      "C_CITY, C_NATION, C_REGION, C_PHONE, C_MKTSEGMENT, LO_ORDERPRIORITY, "
      "LO_SHIPPRIORITY, LO_QUANTITY, LO_EXTENDEDPRICE, LO_ORDTOTALPRICE, "
      "LO_DISCOUNT, LO_REVENUE, LO_SUPPLYCOST, LO_TAX, LO_COMMITDATE, "
      "LO_SHIPMODE from PART JOIN (dates JOIN (customer JOIN (supplier JOIN "
      "lineorder ON (s_suppkey = lo_suppkey)) ON (c_custkey=lo_custkey)) ON "
      "(d_datekey=lo_orderdate)) ON (p_partkey=lo_partkey);";
  TablePtr denormalized_star_schema_database = SQL::executeSQL(query, client);

  denormalized_star_schema_database->setName("DENORMALIZED_SSBM_TABLE");

  RuntimeConfiguration::instance().setPathToDatabase(
      RuntimeConfiguration::instance().getPathToDatabase() + "_denormalized");
  out << "Store Denormalized Star Schema Database: '"
      << RuntimeConfiguration::instance().getPathToDatabase() << "'"
      << std::endl;

  //  storeTableAsSelfContainedCSV(denormalized_star_schema_database,"./","tmp.csv");
  //  denormalized_star_schema_database=loadTableFromSelfContainedCSV("tmp.csv");
  storeTable(denormalized_star_schema_database);
  return true;
}

bool Denormalized_SSB_Q11(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(lo_extendedprice*lo_discount) as revenue from "
      "denormalized_ssbm_table where d_year = 1993 and lo_discount between 1 "
      "and 3 and lo_quantity < 25;",
      client);
}

bool Denormalized_SSB_Q12(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(lo_extendedprice*lo_discount) as revenue from "
      "denormalized_ssbm_table where d_yearmonthnum = 199401 and lo_discount "
      "between 4 and 6 and lo_quantity between 26 and 35;",
      client);
}

bool Denormalized_SSB_Q13(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(lo_extendedprice*lo_discount) as revenue from "
      "denormalized_ssbm_table where d_weeknuminyear = 6 and d_year = 1994 and "
      "lo_discount between 5 and 7 and lo_quantity between 26 and 35;",
      client);
}

bool Denormalized_SSB_Q21(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(lo_revenue), d_year, p_brand from denormalized_ssbm_table "
      "where p_category = 'MFGR#12' and s_region = 'AMERICA' group by d_year, "
      "p_brand order by d_year, p_brand;",
      client);
}

bool Denormalized_SSB_Q22(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(lo_revenue), d_year, p_brand from denormalized_ssbm_table "
      "where p_brand between 'MFGR#2221' and 'MFGR#2228' and s_region = 'ASIA' "
      "group by d_year, p_brand order by d_year, p_brand;",
      client);
}

bool Denormalized_SSB_Q23(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(lo_revenue), d_year, p_brand from denormalized_ssbm_table "
      "where p_brand= 'MFGR#2239' and s_region = 'EUROPE' group by d_year, "
      "p_brand order by d_year, p_brand;",
      client);
}

bool Denormalized_SSB_Q31(ClientPtr client) {
  return SQL::commandlineExec(
      "select c_nation, s_nation, d_year, sum(lo_revenue) from "
      "denormalized_ssbm_table where c_region = 'ASIA' and s_region = 'ASIA' "
      "and d_year >= 1992 and d_year <= 1997 group by c_nation, s_nation, "
      "d_year order by d_year asc, lo_revenue desc;",
      client);
}

bool Denormalized_SSB_Q32(ClientPtr client) {
  return SQL::commandlineExec(
      "select c_city, s_city, d_year, sum(lo_revenue) from "
      "denormalized_ssbm_table where c_nation = 'UNITED STATES' and s_nation = "
      "'UNITED STATES' and d_year >= 1992 and d_year <= 1997 group by c_city, "
      "s_city, d_year order by d_year asc, lo_revenue desc;",
      client);
}

bool Denormalized_SSB_Q33(ClientPtr client) {
  return SQL::commandlineExec(
      "select c_city, s_city, d_year, sum(lo_revenue) from "
      "denormalized_ssbm_table where (c_city='UNITED KI1' or c_city='UNITED "
      "KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5') and d_year >= "
      "1992 and d_year <= 1997 group by c_city, s_city, d_year order by d_year "
      "asc, lo_revenue desc;",
      client);
}

bool Denormalized_SSB_Q34(ClientPtr client) {
  return SQL::commandlineExec(
      "select c_city, s_city, d_year, sum(lo_revenue) from "
      "denormalized_ssbm_table where (c_city='UNITED KI1' or c_city='UNITED "
      "KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5') and d_yearmonth "
      "= 'Dec1997' group by c_city, s_city, d_year order by d_year asc, "
      "lo_revenue desc;",
      client);
}

bool Denormalized_SSB_Q41(ClientPtr client) {
  return SQL::commandlineExec(
      "select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit from "
      "denormalized_ssbm_table where c_region = 'AMERICA' and s_region = "
      "'AMERICA' and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year, "
      "c_nation order by d_year, c_nation;",
      client);
}

bool Denormalized_SSB_Q42(ClientPtr client) {
  return SQL::commandlineExec(
      "select d_year, s_nation, p_category, sum(lo_revenue - lo_supplycost) as "
      "profit from denormalized_ssbm_table where c_region = 'AMERICA' and "
      "s_region = 'AMERICA' and (d_year = 1997 or d_year = 1998) and (p_mfgr = "
      "'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year, s_nation, p_category "
      "order by d_year, s_nation, p_category;",
      client);
}

bool Denormalized_SSB_Q43(ClientPtr client) {
  return SQL::commandlineExec(
      "select d_year, s_city, p_brand, sum(lo_revenue - lo_supplycost) as "
      "profit from denormalized_ssbm_table where s_nation = 'UNITED STATES' "
      "and (d_year = 1997 or d_year = 1998) and p_category = 'MFGR#14' group "
      "by d_year, s_city, p_brand order by d_year, s_city, p_brand;",
      client);
}

}  // end namespace CogaDB
