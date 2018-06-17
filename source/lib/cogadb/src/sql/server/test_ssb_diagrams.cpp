
#include <optimizer/optimizer.hpp>
#include <query_processing/query_processor.hpp>
#include <sql/server/sql_driver.hpp>
#include <util/star_schema_benchmark.hpp>

#include <exception>
#include <iostream>
#include <string>

#include <assert.h>

using namespace std;
using namespace CoGaDB;
using namespace query_processing;

SQL::Driver driver;

// static void
// sql_generate_graph(const string &exp, const string &filename)
//{
//	ofstream os(filename.c_str());
//	std::cout << "Generating " << filename << "..." << std::endl;

//	SQL::ParseTree::SequencePtr seq(driver.parse(exp));

//	seq->explain(os);
//}

// static void
// manual_generate_graph(LogicalQueryPlanPtr plan, const string &filename)
//{
//	ofstream os(filename.c_str());
//	std::cout << "Generating " << filename << "..." << std::endl;

//        //optimizer::Logical_Optimizer::instance().optimize(plan);

//	plan->print_graph(os);
//}

int main(int argc, char **argv) {
  RuntimeConfiguration::instance().setPathToDatabase("./ssb_sf1");
  ClientPtr client = ClientPtr(new LocalClient());
  assert(loadTables(client));

  //	manual_generate_graph(SSB_Q11_plan(), "manual_q1_1.gv");
  //	sql_generate_graph(
  //		"select sum(lo_extendedprice*lo_discount) as revenue from
  // lineorder,
  // dates where lo_orderdate = d_datekey and d_year = 1993 and lo_discount
  // between 1 and 3 and lo_quantity < 25;",
  //		"sql_q1_1.gv"
  //	);
  //	manual_generate_graph(SSB_Q12_plan(), "manual_q1_2.gv");
  //	sql_generate_graph(
  //		"select sum(lo_extendedprice*lo_discount) as revenue from
  // lineorder,
  // dates where lo_orderdate = d_datekey and d_yearmonthnum = 199401 and
  // lo_discount between 4 and 6 and lo_quantity between 26 and 35;",
  //		"sql_q1_2.gv"
  //	);
  //	manual_generate_graph(SSB_Q13_plan(), "manual_q1_3.gv");
  //	sql_generate_graph(
  //		"select sum(lo_extendedprice*lo_discount) as revenue from
  // lineorder,
  // dates where lo_orderdate = d_datekey and d_weeknuminyear = 6 and d_year =
  // 1994 and lo_discount between 5 and 7 and lo_quantity between 26 and 35;",
  //		"sql_q1_3.gv"
  //	);
  //
  //	manual_generate_graph(SSB_Q21_plan(), "manual_q2_1.gv");
  //	sql_generate_graph(
  //		"select sum(lo_revenue), d_year, p_brand from lineorder, dates,
  // part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey
  // and
  // lo_suppkey = s_suppkey and p_category = 'MFGR#12' and s_region = 'AMERICA'
  // group by d_year, p_brand order by d_year, p_brand;",
  //		"sql_q2_1.gv"
  //	);
  //	manual_generate_graph(SSB_Q22_plan(), "manual_q2_2.gv");
  //	sql_generate_graph(
  //		"select sum(lo_revenue), d_year, p_brand from lineorder, dates,
  // part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey
  // and
  // lo_suppkey = s_suppkey and p_brand between 'MFGR#2221' and 'MFGR#2228' and
  // s_region = 'ASIA' group by d_year, p_brand order by d_year, p_brand;",
  //		"sql_q2_2.gv"
  //	);
  //	manual_generate_graph(SSB_Q23_plan(), "manual_q2_3.gv");
  //	sql_generate_graph(
  //		"select sum(lo_revenue), d_year, p_brand from lineorder, dates,
  // part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey
  // and
  // lo_suppkey = s_suppkey and p_brand= 'MFGR#2239' and s_region = 'EUROPE'
  // group by d_year, p_brand order by d_year, p_brand;",
  //		"sql_q2_3.gv"
  //	);
  //
  //	manual_generate_graph(SSB_Q31_plan(), "manual_q3_1.gv");
  //	sql_generate_graph(
  //		"select c_nation, s_nation, d_year, sum(lo_revenue) as revenue
  // from
  // customer, lineorder, supplier, dates where lo_custkey = c_custkey and
  // lo_suppkey = s_suppkey and lo_orderdate = d_datekey and c_region = 'ASIA'
  // and s_region = 'ASIA' and d_year >= 1992 and d_year <= 1997 group by
  // c_nation, s_nation, d_year order by d_year asc, revenue desc;",
  //		"sql_q3_1.gv"
  //	);
  //	manual_generate_graph(SSB_Q32_plan(), "manual_q3_2.gv");
  //	sql_generate_graph(
  //		"select c_city, s_city, d_year, sum(lo_revenue) as revenue from
  // customer, lineorder, supplier, dates where lo_custkey = c_custkey and
  // lo_suppkey = s_suppkey and lo_orderdate = d_datekey and c_nation = 'UNITED
  // STATES' and s_nation = 'UNITED STATES' and d_year >= 1992 and d_year <=
  // 1997
  // group by c_city, s_city, d_year order by d_year asc, revenue desc;",
  //		"sql_q3_2.gv"
  //	);
  //	manual_generate_graph(SSB_Q33_plan(), "manual_q3_3.gv");
  //	sql_generate_graph(
  //		"select c_city, s_city, d_year, sum(lo_revenue) as revenue from
  // customer, lineorder, supplier, dates where lo_custkey = c_custkey and
  // lo_suppkey = s_suppkey and lo_orderdate = d_datekey and (c_city='UNITED
  // KI1'
  // or c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5')
  // and
  // d_year >= 1992 and d_year <= 1997 group by c_city, s_city, d_year order by
  // d_year asc, revenue desc;",
  //		"sql_q3_3.gv"
  //	);
  //	manual_generate_graph(SSB_Q34_plan(), "manual_q3_4.gv");
  //	sql_generate_graph(
  //		"select c_city, s_city, d_year, sum(lo_revenue) as revenue from
  // customer, lineorder, supplier, dates where lo_custkey = c_custkey and
  // lo_suppkey = s_suppkey and lo_orderdate = d_datekey and (c_city='UNITED
  // KI1'
  // or c_city='UNITED KI5') and (s_city='UNITED KI1' or s_city='UNITED KI5')
  // and
  // d_yearmonth = 'Dec1997' group by c_city, s_city, d_year order by d_year
  // asc,
  // revenue desc;",
  //		"sql_q3_4.gv"
  //	);
  //
  //	manual_generate_graph(SSB_Q41_plan(), "manual_q4_1.gv");
  //	sql_generate_graph(
  //		"select d_year, c_nation, sum(lo_revenue - lo_supplycost) as
  // profit
  // from dates, customer, supplier, part, lineorder where lo_custkey =
  // c_custkey
  // and lo_suppkey = s_suppkey and lo_partkey = p_partkey and lo_orderdate =
  // d_datekey and c_region = 'AMERICA' and s_region = 'AMERICA' and (p_mfgr =
  //'MFGR#1' or p_mfgr = 'MFGR#2') group by d_year, c_nation order by d_year,
  // c_nation;",
  //		"sql_q4_1.gv"
  //	);
  //	manual_generate_graph(SSB_Q42_plan(), "manual_q4_2.gv");
  //	sql_generate_graph(
  //		"select d_year, s_nation, p_category, sum(lo_revenue -
  // lo_supplycost) as profit from dates, customer, supplier, part, lineorder
  // where lo_custkey = c_custkey and lo_suppkey = s_suppkey and lo_partkey =
  // p_partkey and lo_orderdate = d_datekey and c_region = 'AMERICA' and
  // s_region
  //= 'AMERICA' and (d_year = 1997 or d_year = 1998) and (p_mfgr = 'MFGR#1' or
  // p_mfgr = 'MFGR#2') group by d_year, s_nation, p_category order by d_year,
  // s_nation, p_category;",
  //		"sql_q4_2.gv"
  //	);
  //	manual_generate_graph(SSB_Q43_plan(), "manual_q4_3.gv");
  //	sql_generate_graph(
  //		"select d_year, s_city, p_brand, sum(lo_revenue - lo_supplycost)
  // as
  // profit from dates, customer, supplier, part, lineorder where lo_custkey =
  // c_custkey and lo_suppkey = s_suppkey and lo_partkey = p_partkey and
  // lo_orderdate = d_datekey and s_nation = 'UNITED STATES' and (d_year = 1997
  // or d_year = 1998) and p_category = 'MFGR#14' group by d_year, s_city,
  // p_brand order by d_year, s_city, p_brand;",
  //		"sql_q4_3.gv"
  //	);

  return 0;
}
