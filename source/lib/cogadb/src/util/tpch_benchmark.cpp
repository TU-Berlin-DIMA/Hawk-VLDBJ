#include <parser/generated/Parser.h>
#include <parser/client.hpp>
#include <persistence/storage_manager.hpp>
#include <query_processing/query_processor.hpp>
#include <sql/server/sql_driver.hpp>
#include <statistics/statistics_manager.hpp>
#include <util/star_schema_benchmark.hpp>
#include <util/statistics.hpp>
#include <util/tpch_benchmark.hpp>

#include <query_processing/query_processor.hpp>

using namespace std;

namespace CoGaDB {

typedef boost::any UDF_Parameter;
typedef std::vector<UDF_Parameter> UDF_Parameters;

using namespace query_processing;

bool Unittest_Create_TPCH_Database(const std::string& path_to_files,
                                   ClientPtr client) {
  // TPCH Table
  // CREATE TABLE NATION  ( N_NATIONKEY  INTEGER NOT NULL,
  //                            N_NAME       CHAR(25) NOT NULL,
  //                            N_REGIONKEY  INTEGER NOT NULL,
  //                            N_COMMENT    VARCHAR(152));

  {
    cout << "Create Table 'NATION' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(INT, "NATION.N_NATIONKEY"));
    schema.push_back(Attribut(VARCHAR, "NATION.N_NAME"));
    schema.push_back(Attribut(INT, "NATION.N_REGIONKEY"));
    schema.push_back(Attribut(VARCHAR, "NATION.N_COMMENT"));

    TablePtr tab1(new Table("NATION", schema));
    if (!tab1->setPrimaryKeyConstraint("NATION.N_NATIONKEY")) {
      COGADB_ERROR("Failed to set Primary Key Constraint!", "");
      return false;
    }
    if (tab1->loadDatafromFile(path_to_files + "nation.tbl")) {
      // tab1->print();
      cout << "Store Table 'NATION' ..." << endl;
      if (storeTable(tab1)) {
        std::cout << "SUCCESS" << std::endl;
        getGlobalTableList().push_back(tab1);
      } else {
        std::cout << "FAILED" << std::endl;
        return false;
      }
    }
  }

  // CREATE TABLE REGION  ( R_REGIONKEY  INTEGER NOT NULL,
  //                            R_NAME       CHAR(25) NOT NULL,
  //                            R_COMMENT    VARCHAR(152));

  {
    cout << "Create Table 'REGION' ..." << endl;
    TableSchema schema2;
    schema2.push_back(Attribut(INT, "REGION.R_REGIONKEY"));
    schema2.push_back(Attribut(VARCHAR, "REGION.R_NAME"));
    schema2.push_back(Attribut(VARCHAR, "REGION.R_COMMENT"));
    TablePtr tab2(new Table("REGION", schema2));
    if (!tab2->setPrimaryKeyConstraint("REGION.R_REGIONKEY")) {
      COGADB_ERROR("Failed to set Primary Key Constraint!", "");
      return false;
    }
    if (tab2->loadDatafromFile(path_to_files + "region.tbl")) {
      // tab2->print();
      cout << "Store Table 'REGION' ..." << endl;
      if (storeTable(tab2)) {
        std::cout << "SUCCESS" << std::endl;
        getGlobalTableList().push_back(tab2);
      } else {
        std::cout << "FAILED" << std::endl;
        return false;
      }
    }
  }

  /*
  CREATE TABLE part  ( P_PARTKEY     INTEGER NOT NULL,
                            P_NAME        VARCHAR(55) NOT NULL,
                            P_MFGR        CHAR(25) NOT NULL,
                            P_BRAND       CHAR(10) NOT NULL,
                            P_TYPE        VARCHAR(25) NOT NULL,
                            P_SIZE        INTEGER NOT NULL,
                            P_CONTAINER   CHAR(10) NOT NULL,
                            P_RETAILPRICE DECIMAL(15,2) NOT NULL,
                            P_COMMENT     VARCHAR(23) NOT NULL )
  engine=brighthouse;
  //*/

  {
    cout << "Create Table 'PART' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(INT, "PART.P_PARTKEY"));
    schema.push_back(Attribut(VARCHAR, "PART.P_NAME"));
    schema.push_back(Attribut(VARCHAR, "PART.P_MFGR"));
    schema.push_back(Attribut(VARCHAR, "PART.P_BRAND"));
    schema.push_back(Attribut(VARCHAR, "PART.P_TYPE"));
    schema.push_back(Attribut(INT, "PART.P_SIZE"));
    schema.push_back(Attribut(VARCHAR, "PART.P_CONTAINER"));
    schema.push_back(Attribut(FLOAT, "PART.P_RETAILPRICE"));
    schema.push_back(Attribut(VARCHAR, "PART.P_COMMENT"));

    TablePtr tab(new Table("PART", schema));
    if (!tab->setPrimaryKeyConstraint("PART.P_PARTKEY")) {
      COGADB_ERROR("Failed to set Primary Key Constraint!", "");
      return false;
    }
    if (tab->loadDatafromFile(path_to_files + "part.tbl")) {
      cout << "Store Table 'PART' ..." << endl;
      if (storeTable(tab)) {
        std::cout << "SUCCESS" << std::endl;
        getGlobalTableList().push_back(tab);
      } else {
        std::cout << "FAILED" << std::endl;
        return false;
      }
    }
  }

  /*
  CREATE TABLE supplier ( S_SUPPKEY     INTEGER NOT NULL,
                               S_NAME        CHAR(25) NOT NULL,
                               S_ADDRESS     VARCHAR(40) NOT NULL,
                               S_NATIONKEY   INTEGER NOT NULL,
                               S_PHONE       CHAR(15) NOT NULL,
                               S_ACCTBAL     DECIMAL(15,2) NOT NULL,
                               S_COMMENT     VARCHAR(101) NOT NULL)
  engine=brighthouse;
  //*/
  {
    cout << "Create Table 'SUPPLIER' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(INT, "SUPPLIER.S_SUPPKEY"));
    schema.push_back(Attribut(VARCHAR, "SUPPLIER.S_NAME"));
    schema.push_back(Attribut(VARCHAR, "SUPPLIER.S_ADDRESS"));
    schema.push_back(Attribut(INT, "SUPPLIER.S_NATIONKEY"));
    schema.push_back(Attribut(VARCHAR, "SUPPLIER.S_PHONE"));
    schema.push_back(Attribut(FLOAT, "SUPPLIER.S_ACCTBAL"));
    schema.push_back(Attribut(VARCHAR, "SUPPLIER.S_COMMENT"));
    TablePtr tab(new Table("SUPPLIER", schema));
    if (!tab->setPrimaryKeyConstraint("SUPPLIER.S_SUPPKEY")) {
      COGADB_ERROR("Failed to set Primary Key Constraint!", "");
      return false;
    }
    if (tab->loadDatafromFile(path_to_files + "supplier.tbl")) {
      cout << "Store Table 'SUPPLIER' ..." << endl;
      if (!tab->setForeignKeyConstraint("SUPPLIER.S_NATIONKEY",
                                        "NATION.N_NATIONKEY", "NATION")) {
        COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
        return false;
      }
      if (storeTable(tab)) {
        std::cout << "SUCCESS" << std::endl;
        getGlobalTableList().push_back(tab);
      } else {
        std::cout << "FAILED" << std::endl;
        return false;
      }
    }
  }

  /*
  CREATE TABLE partsupp ( PS_PARTKEY     INTEGER NOT NULL,
                               PS_SUPPKEY     INTEGER NOT NULL,
                               PS_AVAILQTY    INTEGER NOT NULL,
                               PS_SUPPLYCOST  DECIMAL(15,2)  NOT NULL,
                               PS_COMMENT     VARCHAR(199) NOT NULL )
  engine=brighthouse;
  //*/

  {
    cout << "Create Table 'PARTSUPP' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(INT, "PARTSUPP.PS_PARTKEY"));
    schema.push_back(Attribut(INT, "PARTSUPP.PS_SUPPKEY"));
    schema.push_back(Attribut(INT, "PARTSUPP.PS_AVAILQTY"));
    schema.push_back(Attribut(FLOAT, "PARTSUPP.PS_SUPPLYCOST"));
    schema.push_back(Attribut(VARCHAR, "PARTSUPP.PS_COMMENT"));
    TablePtr tab(new Table("PARTSUPP", schema));

    if (tab->loadDatafromFile(path_to_files + "partsupp.tbl")) {
      cout << "Store Table 'PARTSUPP' ..." << endl;

      if (storeTable(tab)) {
        if (!tab->setForeignKeyConstraint("PARTSUPP.PS_PARTKEY",
                                          "PART.P_PARTKEY", "PART")) {
          COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
          return false;
        }
        if (!tab->setForeignKeyConstraint("PARTSUPP.PS_SUPPKEY",
                                          "SUPPLIER.S_SUPPKEY", "SUPPLIER")) {
          COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
          return false;
        }
        std::cout << "SUCCESS" << std::endl;
        getGlobalTableList().push_back(tab);
      } else {
        std::cout << "FAILED" << std::endl;
        return false;
      }
    }
  }
  /*
  CREATE TABLE customer ( C_CUSTKEY     INTEGER NOT NULL,
                               C_NAME        VARCHAR(25) NOT NULL,
                               C_ADDRESS     VARCHAR(40) NOT NULL,
                               C_NATIONKEY   INTEGER NOT NULL,
                               C_PHONE       CHAR(15) NOT NULL,
                               C_ACCTBAL     DECIMAL(15,2)   NOT NULL,
                               C_MKTSEGMENT  CHAR(10) NOT NULL,
                               C_COMMENT     VARCHAR(117) NOT NULL)
  engine=brighthouse;
  //*/

  {
    cout << "Create Table 'CUSTOMER' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(INT, "CUSTOMER.C_CUSTKEY"));
    schema.push_back(Attribut(VARCHAR, "CUSTOMER.C_NAME"));
    schema.push_back(Attribut(VARCHAR, "CUSTOMER.C_ADDRESS"));
    schema.push_back(Attribut(INT, "CUSTOMER.C_NATIONKEY"));
    schema.push_back(Attribut(VARCHAR, "CUSTOMER.C_PHONE"));
    schema.push_back(Attribut(FLOAT, "CUSTOMER.C_ACCTBAL"));
    schema.push_back(Attribut(VARCHAR, "CUSTOMER.C_MKTSEGMENT"));
    schema.push_back(Attribut(VARCHAR, "CUSTOMER.C_COMMENT"));
    TablePtr tab(new Table("CUSTOMER", schema));
    if (!tab->setPrimaryKeyConstraint("CUSTOMER.C_CUSTKEY")) {
      COGADB_ERROR("Failed to set Primary Key Constraint!", "");
      return false;
    }
    if (tab->loadDatafromFile(path_to_files + "customer.tbl")) {
      cout << "Store Table 'CUSTOMER' ..." << endl;
      if (!tab->setForeignKeyConstraint("CUSTOMER.C_NATIONKEY",
                                        "NATION.N_NATIONKEY", "NATION")) {
        COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
        return false;
      }
      if (storeTable(tab)) {
        std::cout << "SUCCESS" << std::endl;
        getGlobalTableList().push_back(tab);
      } else {
        std::cout << "FAILED" << std::endl;
        return false;
      }
    }
  }

  /*
  CREATE TABLE orders  ( O_ORDERKEY       INTEGER NOT NULL,
                             O_CUSTKEY        INTEGER NOT NULL,
                             O_ORDERSTATUS    CHAR(1) NOT NULL,
                             O_TOTALPRICE     DECIMAL(15,2) NOT NULL,
                             O_ORDERDATE      DATE NOT NULL,
                             O_ORDERPRIORITY  CHAR(15) NOT NULL,
                             O_CLERK          CHAR(15) NOT NULL,
                             O_SHIPPRIORITY   INTEGER NOT NULL,
                             O_COMMENT        VARCHAR(79) NOT NULL)
  engine=brighthouse;
  //*/

  {
    cout << "Create Table 'ORDERS' ..." << endl;
    TableSchema schema;
    schema.push_back(Attribut(INT, "ORDERS.O_ORDERKEY"));
    schema.push_back(Attribut(INT, "ORDERS.O_CUSTKEY"));
    schema.push_back(Attribut(CHAR, "ORDERS.O_ORDERSTATUS"));
    schema.push_back(Attribut(FLOAT, "ORDERS.O_TOTALPRICE"));
    schema.push_back(Attribut(DATE, "ORDERS.O_ORDERDATE"));
    schema.push_back(Attribut(VARCHAR, "ORDERS.O_ORDERPRIORITY"));
    schema.push_back(Attribut(VARCHAR, "ORDERS.O_CLERK"));
    schema.push_back(Attribut(INT, "ORDERS.O_SHIPPRIORITY"));
    schema.push_back(Attribut(VARCHAR, "ORDERS.O_COMMENT"));
    TablePtr tab(new Table("ORDERS", schema));
    if (!tab->setPrimaryKeyConstraint("ORDERS.O_ORDERKEY")) {
      COGADB_ERROR("Failed to set Primary Key Constraint!", "");
      return false;
    }
    if (tab->loadDatafromFile(path_to_files + "orders.tbl")) {
      cout << "Store Table 'ORDERS' ..." << endl;
      if (!tab->setForeignKeyConstraint("ORDERS.O_CUSTKEY",
                                        "CUSTOMER.C_CUSTKEY", "CUSTOMER")) {
        COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
        return false;
      }
      if (storeTable(tab)) {
        std::cout << "SUCCESS" << std::endl;
        getGlobalTableList().push_back(tab);
      } else {
        std::cout << "FAILED" << std::endl;
        return false;
      }
    }
  }

  // create table lineitem (
  // l_orderkey    decimal(12,0) not null,
  // l_partkey     decimal(10,0) not null,
  // l_suppkey     decimal(8,0) not null,
  // l_LINEITEM  decimal(1,0) not null,
  // l_quantity    decimal(2,0) not null,
  // l_extendedprice  decimal(8,2) not null,
  // l_discount    decimal(3,2) not null,
  // l_tax         decimal(3,2) not null,
  // l_returnflag  char(1) not null,
  // l_linestatus  char(1) not null,
  // l_shipdate    date not null,
  // l_commitdate  date not null,
  // l_receiptdate date not null,
  // l_shipinstruct char(25) not null,
  // l_shipmode     char(10) not null,
  // l_comment      varchar(44) not null
  //) TYPE=MyISAM

  {
    cout << "Create Table 'LINEITEM' ..." << endl;
    Timestamp begin = getTimestamp();
    TableSchema schema;
    schema.push_back(Attribut(INT, "LINEITEM.L_ORDERKEY"));  //"L_ORDERKEY");
    schema.push_back(Attribut(INT, "LINEITEM.L_PARTKEY"));   // l_partkey);
    schema.push_back(Attribut(INT, "LINEITEM.L_SUPPKEY"));   // l_suppkey);
    schema.push_back(Attribut(INT, "LINEITEM.L_LINEITEM"));  // l_LINEITEM);
    schema.push_back(Attribut(INT, "LINEITEM.L_QUANTITY"));  // l_quantity );
    schema.push_back(
        Attribut(FLOAT, "LINEITEM.L_EXTENDEDPRICE"));  // l_extendedprice);
    schema.push_back(Attribut(FLOAT, "LINEITEM.L_DISCOUNT"));  // l_discount);
    schema.push_back(Attribut(FLOAT, "LINEITEM.L_TAX"));       // l_tax);
    schema.push_back(
        Attribut(CHAR, "LINEITEM.L_RETURNFLAG"));  // l_returnflag);
    schema.push_back(
        Attribut(CHAR, "LINEITEM.L_LINESTATUS"));             // l_linestatus );
    schema.push_back(Attribut(DATE, "LINEITEM.L_SHIPDATE"));  // l_shipdate);
    schema.push_back(
        Attribut(DATE, "LINEITEM.L_COMMITDATE"));  // l_commitdate);
    schema.push_back(
        Attribut(DATE, "LINEITEM.L_RECEIPTDATE"));  // l_receiptdate);
    schema.push_back(
        Attribut(VARCHAR, "LINEITEM.L_SHIPINSTRUCT"));  // l_shipinstruct);
    schema.push_back(Attribut(VARCHAR, "LINEITEM.L_SHIPMODE"));  // l_shipmode);
    schema.push_back(Attribut(VARCHAR, "LINEITEM.L_COMMENT"));   // l_comment);

    //            schema.push_back(Attribut(INT, "L_ORDERKEY"));
    //            //"L_ORDERKEY");
    //            schema.push_back(Attribut(INT, "L_PARTKEY")); //l_partkey);
    //            schema.push_back(Attribut(INT, "L_SUPPKEY")); //l_suppkey);
    //            schema.push_back(Attribut(INT, "L_LINEITEM")); //l_LINEITEM);
    //            schema.push_back(Attribut(DOUBLE, "L_QUANTITY")); //l_quantity
    //            );
    //            schema.push_back(Attribut(DOUBLE, "L_EXTENDEDPRICE"));
    //            //l_extendedprice);
    //            schema.push_back(Attribut(DOUBLE, "L_DISCOUNT"));
    //            //l_discount);
    //            schema.push_back(Attribut(DOUBLE, "L_TAX")); //l_tax);
    //            schema.push_back(Attribut(CHAR, "L_RETURNFLAG"));
    //            //l_returnflag);
    //            schema.push_back(Attribut(CHAR, "L_LINESTATUS"));
    //            //l_linestatus );
    //            schema.push_back(Attribut(DATE, "L_SHIPDATE")); //l_shipdate);
    //            schema.push_back(Attribut(DATE, "L_COMMITDATE"));
    //            //l_commitdate);
    //            schema.push_back(Attribut(DATE, "L_RECEIPTDATE"));
    //            //l_receiptdate);
    //            schema.push_back(Attribut(VARCHAR, "L_SHIPINSTRUCT"));
    //            //l_shipinstruct);
    //            schema.push_back(Attribut(VARCHAR, "L_SHIPMODE"));
    //            //l_shipmode);
    //            schema.push_back(Attribut(VARCHAR, "L_COMMENT"));
    //            //l_comment);

    TablePtr tab(new Table("LINEITEM", schema));

    if (tab->loadDatafromFile(path_to_files + "lineitem.tbl")) {
      Timestamp end = getTimestamp();
      assert(end >= begin);
      cout << "Needed " << end - begin << "ns for import..." << endl;
      // tab->print();
      cout << "Store Table 'LINEITEM' ..." << endl;
      if (storeTable(tab)) {
        // if(!tab->setForeignKeyConstraint("L_ORDERKEY","O_ORDERKEY",
        // "ORDERS")){ COGADB_ERROR("Failed to set Foreign Key Constraint!","");
        // return false;}
        // if(!tab->setForeignKeyConstraint("L_PARTKEY","PS_PARTKEY",
        // "PARTSUPP")){ COGADB_ERROR("Failed to set Foreign Key
        // Constraint!",""); return false;}
        // if(!tab->setForeignKeyConstraint("L_SUPPKEY","PS_SUPPKEY",
        // "PARTSUPP")){ COGADB_ERROR("Failed to set Foreign Key
        // Constraint!",""); return false;}
        std::cout << "SUCCESS" << std::endl;
        getGlobalTableList().push_back(tab);
      } else {
        std::cout << "FAILED" << std::endl;
        return false;
      }
    }
  }

  cout << "Building Statistics..." << std::endl;
  computeStatisticsOnTable("NATION", client);
  computeStatisticsOnTable("REGION", client);
  computeStatisticsOnTable("SUPPLIER", client);
  computeStatisticsOnTable("PART", client);
  computeStatisticsOnTable("CUSTOMER", client);
  computeStatisticsOnTable("PARTSUPP", client);
  computeStatisticsOnTable("ORDERS", client);
  computeStatisticsOnTable("LINEITEM", client);

  cout << "Successfully created TPCH Database!" << endl;
  return true;
}

hype::DeviceConstraint convertToSemiJoinConstraint(
    const hype::DeviceConstraint& dev_constr) {
  //        return hype::CPU_ONLY;
  return dev_constr;
}

hype::DeviceConstraint convertToRegularExpressionSelectionConstraint(
    const hype::DeviceConstraint& dev_constr) {
  return hype::CPU_ONLY;
  //        return dev_constr;
}

bool TPCH_Q1(ClientPtr client) {
  return SQL::commandlineExec(
      "select l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, "
      "sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - "
      "l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) "
      "* (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, "
      "avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, "
      "count(l_discount) as count_order from lineitem where l_shipdate <= "
      "'1998-9-02' group by l_returnflag, l_linestatus order by l_returnflag, "
      "l_linestatus;",
      client);
}

TypedNodePtr getTPCHQ2Subplan() {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  /* Table SCANs */
  boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(
      new logical_operator::Logical_Scan("SUPPLIER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_nation(
      new logical_operator::Logical_Scan("NATION"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_region(
      new logical_operator::Logical_Scan("REGION"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_partsupp(
      new logical_operator::Logical_Scan("PARTSUPP"));

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_region;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("REGION.R_NAME", boost::any(std::string("EUROPE")),
                            ValueConstantPredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_region =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  complex_selection_on_region->setLeft(scan_region);

  /* Now we compute the joins. Note the additional selection that applys
   * the additional join condition. */
  boost::shared_ptr<logical_operator::Logical_Join> join_nation_region(
      new logical_operator::Logical_Join("NATION.N_REGIONKEY",
                                         "REGION.R_REGIONKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_supplier(
      new logical_operator::Logical_Join("NATION.N_NATIONKEY",
                                         "SUPPLIER.S_NATIONKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_partsupp(
      new logical_operator::Logical_Join("SUPPLIER.S_SUPPKEY",
                                         "PARTSUPP.PS_SUPPKEY", INNER_JOIN,
                                         default_device_constraint));

  join_nation_region->setLeft(scan_nation);
  join_nation_region->setRight(complex_selection_on_region);

  join_supplier->setLeft(join_nation_region);
  join_supplier->setRight(scan_supplier);

  join_partsupp->setLeft(join_supplier);
  join_partsupp->setRight(scan_partsupp);

  /* Group By operation. We perform renaming later! */

  /* This subquery is tricky: At first sight, the query looks independent from
   * the
   * outer query, and computing a single minimum looks right. However, the
   * condition
   * p_partkey = ps_partkey in the sub query references the part table, which
   * is not referencing the PART table. Therefore, the query is correlated.
   * That means, we have to compute the minimum supplier for each part
   * (partkey).
   * We can optimize this by grouping by the PARTKEY contained in PARTSUPP
   * (PS_PARTKEY),
   * which is included in the sub query. Thus, we can first compute the minimum
   * supplier cost
   * for each partkey, and then peform an equality inner join with the part
   * table (outer query).
   */
  std::list<std::string> sorting_column_names;
  //        sorting_column_names.push_back("_GROUPING_COLUMN");
  sorting_column_names.push_back("PARTSUPP.PS_PARTKEY");

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      make_pair("PARTSUPP.PS_SUPPLYCOST", Aggregate(MIN, "MIN_SUPPLYCOST")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(sorting_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));

  group->setLeft(join_partsupp);

  /* Rename PS_PARTKEY, because the outer query also joins with PARTSUPP, and we
   * have to be able to reference this column in a later join.
   */
  boost::shared_ptr<logical_operator::Logical_Rename> rename;
  {
    RenameList rename_list;
    rename_list.push_back(RenameEntry("PARTSUPP.PS_PARTKEY", "PS2.PS_PARTKEY"));
    rename = boost::make_shared<logical_operator::Logical_Rename>(rename_list);
  }
  rename->setLeft(group);

  return rename;
}

bool TPCH_Q2(ClientPtr client) {
  LogicalQueryPlanPtr log_plan = getPlan_TPCH_Q2(client);
  return optimize_execute_print("tpch2", *log_plan, client);
}

LogicalQueryPlanPtr getPlan_TPCH_Q2(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  /* Table SCANs */
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(new
  //        logical_operator::Logical_Scan("LINEITEM"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_orders(new
  //        logical_operator::Logical_Scan("ORDERS"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(
      new logical_operator::Logical_Scan("SUPPLIER"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
  //        logical_operator::Logical_Scan("CUSTOMER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_nation(
      new logical_operator::Logical_Scan("NATION"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_region(
      new logical_operator::Logical_Scan("REGION"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_part(
      new logical_operator::Logical_Scan("PART"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_partsupp(
      new logical_operator::Logical_Scan("PARTSUPP"));

  //        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
  //        complex_selection_on_part;
  //        {
  //            KNF_Selection_Expression knf_expr;
  //            {
  //                Disjunction d;
  //                d.push_back(Predicate("P_TYPE",
  //                boost::any(std::string(".*BRASS")),
  //                ValueRegularExpressionPredicate, EQUAL));
  //                knf_expr.disjunctions.push_back(d);
  //            }
  //            {
  //                Disjunction d;
  //                d.push_back(Predicate("P_SIZE", boost::any(int32_t(15)),
  //                ValueConstantPredicate, EQUAL));
  //                knf_expr.disjunctions.push_back(d);
  //            }
  //            complex_selection_on_part =
  //            boost::make_shared<logical_operator::Logical_ComplexSelection>(knf_expr,
  //            LOOKUP, hype::CPU_ONLY); //default_device_constraint);
  //        }

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_part;
  {
    {
      KNF_Selection_Expression knf_expr;
      {
        Disjunction d;
        d.push_back(Predicate("PART.P_SIZE", boost::any(int32_t(15)),
                              ValueConstantPredicate, EQUAL));
        knf_expr.disjunctions.push_back(d);
      }
      complex_selection_on_part =
          boost::make_shared<logical_operator::Logical_ComplexSelection>(
              knf_expr, LOOKUP, default_device_constraint);
    }
    boost::shared_ptr<logical_operator::Logical_ComplexSelection>
        complex_selection_on_part2;
    {
      KNF_Selection_Expression knf_expr;
      {
        Disjunction d;
        d.push_back(Predicate("PART.P_TYPE", boost::any(std::string(".*BRASS")),
                              ValueRegularExpressionPredicate, EQUAL));
        knf_expr.disjunctions.push_back(d);
      }
      complex_selection_on_part2 =
          boost::make_shared<logical_operator::Logical_ComplexSelection>(
              knf_expr, LOOKUP,
              convertToRegularExpressionSelectionConstraint(
                  default_device_constraint));  // hype::CPU_ONLY);
    }
    complex_selection_on_part2->setLeft(scan_part);
    complex_selection_on_part->setLeft(complex_selection_on_part2);
  }

  //        complex_selection_on_part->setLeft(scan_part);

  /* Now we compute the joins. Note the additional selection that applys
   * the additional join condition (PS_SUPPLYCOST=MIN_SUPPLYCOST) */
  boost::shared_ptr<logical_operator::Logical_Join> join_nation_region(
      new logical_operator::Logical_Join("NATION.N_REGIONKEY",
                                         "REGION.R_REGIONKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_supplier(
      new logical_operator::Logical_Join("NATION.N_NATIONKEY",
                                         "SUPPLIER.S_NATIONKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_part_subquery(
      new logical_operator::Logical_Join("PART.P_PARTKEY", "PS2.PS_PARTKEY",
                                         INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_part_subquery_partsupp(
      new logical_operator::Logical_Join("PART.P_PARTKEY",
                                         "PARTSUPP.PS_PARTKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_all(
      new logical_operator::Logical_Join("SUPPLIER.S_SUPPKEY",
                                         "PARTSUPP.PS_SUPPKEY", INNER_JOIN,
                                         default_device_constraint));

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_part_subplan_partsupp;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      // d.push_back(Predicate("N2.N_NAME", boost::any(std::string("FRANCE")),
      // ValueConstantPredicate, EQUAL));
      d.push_back(Predicate("PARTSUPP.PS_SUPPLYCOST",
                            std::string("MIN_SUPPLYCOST"), ValueValuePredicate,
                            EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_part_subplan_partsupp = boost::make_shared<
        logical_operator::Logical_ComplexSelection>(
        knf_expr, LOOKUP,
        default_device_constraint);  // hype::CPU_ONLY);//default_device_constraint);
  }

  join_nation_region->setLeft(scan_nation);
  join_nation_region->setRight(scan_region);

  join_supplier->setLeft(join_nation_region);
  join_supplier->setRight(scan_supplier);

  TypedNodePtr subplan = getTPCHQ2Subplan();

  join_part_subquery->setLeft(complex_selection_on_part);
  join_part_subquery->setRight(subplan);

  join_part_subquery_partsupp->setLeft(join_part_subquery);
  join_part_subquery_partsupp->setRight(scan_partsupp);

  complex_selection_on_part_subplan_partsupp->setLeft(
      join_part_subquery_partsupp);

  join_all->setLeft(join_supplier);
  join_all->setRight(complex_selection_on_part_subplan_partsupp);

  SortAttributeList sorting_attributes;
  sorting_attributes.push_back(SortAttribute("SUPPLIER.S_ACCTBAL", DESCENDING));
  sorting_attributes.push_back(SortAttribute("NATION.N_NAME", ASCENDING));
  sorting_attributes.push_back(SortAttribute("SUPPLIER.S_NAME", ASCENDING));
  sorting_attributes.push_back(SortAttribute("PART.P_PARTKEY", ASCENDING));

  boost::shared_ptr<logical_operator::Logical_Sort> order(
      new logical_operator::Logical_Sort(sorting_attributes, LOOKUP,
                                         default_device_constraint));
  order->setLeft(join_all);

  UDF_Parameters limit_param;
  limit_param.push_back(UDF_Parameter(size_t(100)));

  boost::shared_ptr<logical_operator::Logical_UDF> limit(
      new logical_operator::Logical_UDF("LIMIT", limit_param,
                                        default_device_constraint));
  limit->setLeft(order);

  boost::shared_ptr<logical_operator::Logical_Projection> project;
  {
    std::list<std::string> projected_columns;
    projected_columns.push_back("SUPPLIER.S_ACCTBAL");
    projected_columns.push_back("SUPPLIER.S_NAME");
    projected_columns.push_back("NATION.N_NAME");
    projected_columns.push_back("PART.P_PARTKEY");
    projected_columns.push_back("PART.P_MFGR");
    projected_columns.push_back("SUPPLIER.S_ADDRESS");
    projected_columns.push_back("SUPPLIER.S_PHONE");
    projected_columns.push_back("SUPPLIER.S_COMMENT");
    projected_columns.push_back("PARTSUPP.PS_SUPPLYCOST");
    project = boost::make_shared<logical_operator::Logical_Projection>(
        projected_columns);
  }

  project->setLeft(limit);  // order);

  LogicalQueryPlanPtr log_plan =
      boost::make_shared<LogicalQueryPlan>(project);  // order);

  return log_plan;
}

bool TPCH_Q3(ClientPtr client) {
  //        return SQL::commandlineExec("select l_orderkey,
  //        sum(l_extendedprice*(1-l_discount)) as revenue, o_orderdate,
  //        o_shippriority from customer, orders, lineitem where
  //        c_mktsegment='BUILDING' and c_custkey=o_custkey and
  //        l_orderkey=o_orderkey and o_orderdate<'1995-03-15' and l_shipdate >
  //        '1995-03-15' group by l_orderkey, o_orderdate, o_shippriority order
  //        by revenue desc, o_orderdate limit 10;",
  return SQL::commandlineExec(
      "select l_orderkey, sum(l_extendedprice*(1-l_discount)) as revenue, "
      "o_orderdate, o_shippriority from customer JOIN orders ON "
      "(c_custkey=o_custkey) JOIN lineitem ON (o_orderkey=l_orderkey) where "
      "c_mktsegment='BUILDING' and o_orderdate<'1995-03-15' and l_shipdate > "
      "'1995-03-15' group by l_orderkey, o_orderdate, o_shippriority order by "
      "revenue desc, o_orderdate limit 10;",
      client);
}

bool TPCH_Q4(ClientPtr client) {
  LogicalQueryPlanPtr log_plan = getPlan_TPCH_Q4(client);
  return optimize_execute_print("tpch4", *log_plan, client);
}

LogicalQueryPlanPtr getPlan_TPCH_Q4(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_orders(
      new logical_operator::Logical_Scan("ORDERS"));

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_orders;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("ORDERS.O_ORDERDATE",
                            boost::any(std::string("1993-07-01")),
                            ValueConstantPredicate, GREATER_EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(Predicate("ORDERS.O_ORDERDATE",
                            boost::any(std::string("1993-10-01")),
                            ValueConstantPredicate, LESSER));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_orders =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_lineitem;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("LINEITEM.L_COMMITDATE",
                            std::string("LINEITEM.L_RECEIPTDATE"),
                            ValueValuePredicate, LESSER));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_lineitem =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  boost::shared_ptr<logical_operator::Logical_Join> join(
      new logical_operator::Logical_Join(
          "ORDERS.O_ORDERKEY", "LINEITEM.L_ORDERKEY", LEFT_SEMI_JOIN,
          convertToSemiJoinConstraint(
              default_device_constraint)));  // hype::DeviceConstraint(hype::CPU_ONLY)));
                                             // //GPU Join not supported

  complex_selection_on_lineitem->setLeft(scan_lineitem);
  complex_selection_on_orders->setLeft(scan_orders);
  join->setLeft(complex_selection_on_orders);
  join->setRight(complex_selection_on_lineitem);

  std::list<std::string> grouping_column_names;
  grouping_column_names.push_back("ORDERS.O_ORDERPRIORITY");

  SortAttributeList sorting_attributes;
  sorting_attributes.push_back(
      SortAttribute("ORDERS.O_ORDERPRIORITY", ASCENDING));

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      make_pair("ORDERS.O_ORDERKEY", Aggregate(COUNT, "ORDER_COUNT")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(grouping_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Sort> order(
      new logical_operator::Logical_Sort(sorting_attributes, LOOKUP,
                                         default_device_constraint));

  group->setLeft(join);
  order->setLeft(group);

  LogicalQueryPlanPtr log_plan = boost::make_shared<LogicalQueryPlan>(order);

  return log_plan;
}

bool TPCH_Q5(ClientPtr client) {
  return SQL::commandlineExec(
      "select n_name, sum(l_extendedprice * (1 - l_discount)) as revenue from "
      "customer JOIN orders ON (c_custkey = o_custkey) JOIN lineitem ON "
      "(o_orderkey = l_orderkey) JOIN supplier ON (l_suppkey = s_suppkey) JOIN "
      "(select * from nation JOIN region ON (n_regionkey = r_regionkey)) ON "
      "(s_nationkey = n_nationkey) where c_nationkey = s_nationkey and r_name "
      "= 'ASIA' and o_orderdate >= '1994-01-01' and o_orderdate < '1995-01-01' "
      "group by n_name order by revenue desc;",
      client);
}

bool TPCH_Q6(ClientPtr client) {
  return SQL::commandlineExec(
      "select sum(l_extendedprice * l_discount) as revenue from lineitem where "
      "l_shipdate >= '1994-01-01' and l_shipdate < '1995-01-01' and l_discount "
      "between 0.05 and 0.07 and l_quantity < 24;",
      client);
}

bool TPCH_Q7(ClientPtr client) {
  LogicalQueryPlanPtr log_plan = getPlan_TPCH_Q7(client);
  return optimize_execute_print("tpch7", *log_plan, client);
}

LogicalQueryPlanPtr getPlan_TPCH_Q7(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  /* Table SCANs */
  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_orders(
      new logical_operator::Logical_Scan("ORDERS"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(
      new logical_operator::Logical_Scan("SUPPLIER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(
      new logical_operator::Logical_Scan("CUSTOMER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_nation1(
      new logical_operator::Logical_Scan("NATION"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_nation2(
      new logical_operator::Logical_Scan("NATION"));

  /* We have not yet a nice mechanism to deal with self joins, so we need
   * to workaround this. First we project columns (from materialized tables),
   * so renaming works properly. The nwe rename columns, perform a CROSS_JOIN.
   * Then, we apply a complex selection to the computed table. */
  boost::shared_ptr<logical_operator::Logical_Projection> project_nation1;
  boost::shared_ptr<logical_operator::Logical_Projection> project_nation2;
  {
    std::list<std::string> projected_columns;
    projected_columns.push_back("NATION.N_NATIONKEY");
    projected_columns.push_back("NATION.N_NAME");
    projected_columns.push_back("NATION.N_REGIONKEY");
    project_nation1 = boost::make_shared<logical_operator::Logical_Projection>(
        projected_columns);
    project_nation2 = boost::make_shared<logical_operator::Logical_Projection>(
        projected_columns);
  }

  boost::shared_ptr<logical_operator::Logical_Rename> rename_nation1;
  {
    RenameList rename_list;
    rename_list.push_back(RenameEntry("NATION.N_NATIONKEY", "N1.N_NATIONKEY"));
    rename_list.push_back(RenameEntry("NATION.N_NAME", "N1.N_NAME"));
    rename_list.push_back(RenameEntry("NATION.N_REGIONKEY", "N1.N_REGIONKEY"));
    //            rename_list.push_back(RenameEntry("N_COMMENT","N1.N_COMMENT"));
    rename_nation1 =
        boost::make_shared<logical_operator::Logical_Rename>(rename_list);
  }

  boost::shared_ptr<logical_operator::Logical_Rename> rename_nation2;
  {
    RenameList rename_list;
    rename_list.push_back(RenameEntry("NATION.N_NATIONKEY", "N2.N_NATIONKEY"));
    rename_list.push_back(RenameEntry("NATION.N_NAME", "N2.N_NAME"));
    rename_list.push_back(RenameEntry("NATION.N_REGIONKEY", "N2.N_REGIONKEY"));
    //            rename_list.push_back(RenameEntry("N_COMMENT","N2.N_COMMENT"));
    rename_nation2 =
        boost::make_shared<logical_operator::Logical_Rename>(rename_list);
  }

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_lineitem;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("LINEITEM.L_SHIPDATE",
                            boost::any(std::string("1995-01-01")),
                            ValueConstantPredicate, GREATER_EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(Predicate("LINEITEM.L_SHIPDATE",
                            boost::any(std::string("1996-12-31")),
                            ValueConstantPredicate, LESSER_EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_lineitem =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_joined_nations;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("N1.N_NAME", boost::any(std::string("GERMANY")),
                            ValueConstantPredicate, EQUAL));
      d.push_back(Predicate("N1.N_NAME", boost::any(std::string("FRANCE")),
                            ValueConstantPredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(Predicate("N2.N_NAME", boost::any(std::string("FRANCE")),
                            ValueConstantPredicate, EQUAL));
      d.push_back(Predicate("N2.N_NAME", boost::any(std::string("GERMANY")),
                            ValueConstantPredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      // d.push_back(Predicate("N2.N_NAME", boost::any(std::string("FRANCE")),
      // ValueConstantPredicate, EQUAL));
      d.push_back(Predicate("N1.N_NAME", std::string("N2.N_NAME"),
                            ValueValuePredicate, UNEQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_joined_nations =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, hype::CPU_ONLY);
  }

  boost::shared_ptr<logical_operator::Logical_CrossJoin> join1(
      new logical_operator::Logical_CrossJoin());  // GPU Join not supported

  complex_selection_on_lineitem->setLeft(scan_lineitem);

  project_nation1->setLeft(scan_nation1);
  project_nation2->setLeft(scan_nation2);

  rename_nation1->setLeft(project_nation1);
  rename_nation2->setLeft(project_nation2);

  join1->setLeft(rename_nation1);
  join1->setRight(rename_nation2);

  complex_selection_on_joined_nations->setLeft(join1);
  complex_selection_on_lineitem->setLeft(scan_lineitem);

  /* Now we compute the joins. Note the additional selection that applys
   * the additional join condition. */
  boost::shared_ptr<logical_operator::Logical_Join> join_customer(
      new logical_operator::Logical_Join("N1.N_NATIONKEY",
                                         "CUSTOMER.C_NATIONKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_orders(
      new logical_operator::Logical_Join("CUSTOMER.C_CUSTKEY",
                                         "ORDERS.O_CUSTKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_lineitem(
      new logical_operator::Logical_Join("ORDERS.O_ORDERKEY",
                                         "LINEITEM.L_ORDERKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_supplier(
      new logical_operator::Logical_Join("LINEITEM.L_SUPPKEY",
                                         "SUPPLIER.S_SUPPKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_s_nation_and_n_nation;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      // d.push_back(Predicate("N2.N_NAME", boost::any(std::string("FRANCE")),
      // ValueConstantPredicate, EQUAL));
      d.push_back(Predicate("SUPPLIER.S_NATIONKEY",
                            std::string("N2.N_NATIONKEY"), ValueValuePredicate,
                            EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_s_nation_and_n_nation =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  join_customer->setLeft(complex_selection_on_joined_nations);
  join_customer->setRight(scan_customer);

  join_orders->setLeft(join_customer);
  join_orders->setRight(scan_orders);

  join_lineitem->setLeft(join_orders);
  join_lineitem->setRight(complex_selection_on_lineitem);

  join_supplier->setLeft(join_lineitem);
  join_supplier->setRight(scan_supplier);

  complex_selection_on_s_nation_and_n_nation->setLeft(join_supplier);

  /* Now we compute the aggregation expression (1-L_DISCOUNT)*L_EXTENDEDPRICE.
   * First we create a column representeing the 1. Then we perform the
   * substract and multiply operations. */
  boost::shared_ptr<logical_operator::Logical_AddConstantValueColumn>
      one_column =
          boost::make_shared<logical_operator::Logical_AddConstantValueColumn>(
              "1", DOUBLE, boost::any(double(1)), default_device_constraint);
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_one_minus_discount(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "1", "L_DISCOUNT", "(1-L_DISCOUNT)", SUB,
              default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_mul_extendedprice(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "(1-L_DISCOUNT)", "L_EXTENDEDPRICE", "VOLUME", MUL,
              default_device_constraint));

  one_column->setLeft(complex_selection_on_s_nation_and_n_nation);
  column_algebra_operation_one_minus_discount->setLeft(one_column);
  column_algebra_operation_mul_extendedprice->setLeft(
      column_algebra_operation_one_minus_discount);

  UDF_Parameters extract_year_param;
  extract_year_param.push_back(
      UDF_Parameter(std::string("LINEITEM.L_SHIPDATE")));
  extract_year_param.push_back(UDF_Parameter(std::string("LINEITEM.L_YEAR")));

  boost::shared_ptr<logical_operator::Logical_UDF> extract_year(
      new logical_operator::Logical_UDF("EXTRACT_YEAR", extract_year_param,
                                        default_device_constraint));
  extract_year->setLeft(column_algebra_operation_mul_extendedprice);

  /* Group By and Order By operations. */
  std::list<std::string> grouping_column_names;
  grouping_column_names.push_back("N2.N_NAME");
  grouping_column_names.push_back("N1.N_NAME");
  //        grouping_column_names.push_back("L_SHIPDATE");
  grouping_column_names.push_back("LINEITEM.L_YEAR");

  SortAttributeList sorting_attributes;
  sorting_attributes.push_back(SortAttribute("N2.N_NAME", ASCENDING));
  sorting_attributes.push_back(SortAttribute("N1.N_NAME", ASCENDING));
  //        sorting_attributes.push_back(SortAttribute("L_SHIPDATE",ASCENDING));
  sorting_attributes.push_back(SortAttribute("LINEITEM.L_YEAR", ASCENDING));

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      make_pair("VOLUME", Aggregate(SUM, "REVENUE")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(grouping_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Sort> order(
      new logical_operator::Logical_Sort(sorting_attributes, LOOKUP,
                                         default_device_constraint));

  /* Rename the columns, create and execute plan. */
  boost::shared_ptr<logical_operator::Logical_Rename> rename_final;
  {
    RenameList rename_list;
    rename_list.push_back(RenameEntry("N2.N_NAME", "SUPP_NATION"));
    rename_list.push_back(RenameEntry("N1.N_NAME", "CUST_NATION"));
    rename_final =
        boost::make_shared<logical_operator::Logical_Rename>(rename_list);
  }

  group->setLeft(extract_year);  // column_algebra_operation_mul_extendedprice);
  order->setLeft(group);
  rename_final->setLeft(order);

  LogicalQueryPlanPtr log_plan =
      boost::make_shared<LogicalQueryPlan>(rename_final);  // order);

  return log_plan;
}

bool TPCH_Q9(ClientPtr client) {
  LogicalQueryPlanPtr log_plan = getPlan_TPCH_Q9(client);
  return optimize_execute_print("tpch9", *log_plan, client);
}

LogicalQueryPlanPtr getPlan_TPCH_Q9(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  /* Table SCANs */
  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_orders(
      new logical_operator::Logical_Scan("ORDERS"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(
      new logical_operator::Logical_Scan("SUPPLIER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(
      new logical_operator::Logical_Scan("CUSTOMER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_nation(
      new logical_operator::Logical_Scan("NATION"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_part(
      new logical_operator::Logical_Scan("PART"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_partsupp(
      new logical_operator::Logical_Scan("PARTSUPP"));

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_part;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("PART.P_NAME", boost::any(std::string(".*green.*")),
                            ValueRegularExpressionPredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_part =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP,
            convertToRegularExpressionSelectionConstraint(
                default_device_constraint));  // default_device_constraint);
  }

  complex_selection_on_part->setLeft(scan_part);

  /* Now we compute the joins. Note the additional selection that applys
   * the additional join condition. */
  boost::shared_ptr<logical_operator::Logical_Join> join_nation_supplier(
      new logical_operator::Logical_Join("NATION.N_NATIONKEY",
                                         "SUPPLIER.S_NATIONKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_part_partsupp(
      new logical_operator::Logical_Join("PART.P_PARTKEY",
                                         "PARTSUPP.PS_PARTKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join>
      join_nation_supplier_part_partsupp(new logical_operator::Logical_Join(
          "SUPPLIER.S_SUPPKEY", "PARTSUPP.PS_SUPPKEY", INNER_JOIN,
          default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_lineitem(
      new logical_operator::Logical_Join("PART.P_PARTKEY", "LINEITEM.L_PARTKEY",
                                         INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_orders(
      new logical_operator::Logical_Join("LINEITEM.L_ORDERKEY",
                                         "ORDERS.O_ORDERKEY", INNER_JOIN,
                                         default_device_constraint));

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      join_selection_lineitem;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("PARTSUPP.PS_PARTKEY",
                            std::string("LINEITEM.L_PARTKEY"),
                            ValueValuePredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(Predicate("PARTSUPP.PS_SUPPKEY",
                            std::string("LINEITEM.L_SUPPKEY"),
                            ValueValuePredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(Predicate("SUPPLIER.S_SUPPKEY",
                            std::string("LINEITEM.L_SUPPKEY"),
                            ValueValuePredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    join_selection_lineitem =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  join_nation_supplier->setLeft(scan_nation);
  join_nation_supplier->setRight(scan_supplier);

  join_part_partsupp->setLeft(complex_selection_on_part);
  join_part_partsupp->setRight(scan_partsupp);

  join_nation_supplier_part_partsupp->setLeft(join_nation_supplier);
  join_nation_supplier_part_partsupp->setRight(join_part_partsupp);

  /* Perform join with four predicates. */
  join_lineitem->setLeft(join_nation_supplier_part_partsupp);
  join_lineitem->setRight(scan_lineitem);
  join_selection_lineitem->setLeft(join_lineitem);

  join_orders->setLeft(join_selection_lineitem);
  join_orders->setRight(scan_orders);

  /* Now we compute the aggregation expression (l_extendedprice * (1 -
   * l_discount) - ps_supplycost * l_quantity) as amount.
   * First we create a column representing the 1. Then we perform the
   * substract and multiply operations. */
  boost::shared_ptr<logical_operator::Logical_AddConstantValueColumn>
      one_column =
          boost::make_shared<logical_operator::Logical_AddConstantValueColumn>(
              "1", DOUBLE, boost::any(double(1)), default_device_constraint);
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_one_minus_discount(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "1", "LINEITEM.L_DISCOUNT", "(1-LINEITEM.L_DISCOUNT)", SUB,
              default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_mul_extendedprice(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "(1-LINEITEM.L_DISCOUNT)", "LINEITEM.L_EXTENDEDPRICE",
              "((1-LINEITEM.L_DISCOUNT)*LINEITEM.L_EXTENDEDPRICE)", MUL,
              default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_mul_supplycost_quantity(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "PARTSUPP.PS_SUPPLYCOST", "LINEITEM.L_QUANTITY",
              "(PARTSUPP.PS_SUPPLYCOST*LINEITEM.L_QUANTITY)", MUL,
              default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_final_sub(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "((1-LINEITEM.L_DISCOUNT)*LINEITEM.L_EXTENDEDPRICE)",
              "(PARTSUPP.PS_SUPPLYCOST*LINEITEM.L_QUANTITY)", "AMOUNT", SUB,
              default_device_constraint));

  one_column->setLeft(join_orders);
  column_algebra_operation_one_minus_discount->setLeft(one_column);
  column_algebra_operation_mul_extendedprice->setLeft(
      column_algebra_operation_one_minus_discount);
  column_algebra_operation_mul_supplycost_quantity->setLeft(
      column_algebra_operation_mul_extendedprice);
  column_algebra_operation_final_sub->setLeft(
      column_algebra_operation_mul_supplycost_quantity);

  UDF_Parameters extract_year_param;
  extract_year_param.push_back(
      UDF_Parameter(std::string("ORDERS.O_ORDERDATE")));
  extract_year_param.push_back(UDF_Parameter(std::string("O_YEAR")));

  boost::shared_ptr<logical_operator::Logical_UDF> extract_year(
      new logical_operator::Logical_UDF("EXTRACT_YEAR", extract_year_param,
                                        default_device_constraint));
  extract_year->setLeft(column_algebra_operation_final_sub);

  /* Group By and Order By operations. We perform renaming later! */
  std::list<std::string> grouping_column_names;
  grouping_column_names.push_back("NATION.N_NAME");
  //        grouping_column_names.push_back("O_ORDERDATE");
  grouping_column_names.push_back("O_YEAR");

  SortAttributeList sorting_attributes;
  sorting_attributes.push_back(SortAttribute("NATION.N_NAME", ASCENDING));
  //        sorting_attributes.push_back(SortAttribute("O_ORDERDATE",DESCENDING));
  sorting_attributes.push_back(SortAttribute("O_YEAR", DESCENDING));

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      make_pair("AMOUNT", Aggregate(SUM, "SUM_PROFIT")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(grouping_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Sort> order(
      new logical_operator::Logical_Sort(sorting_attributes, LOOKUP,
                                         default_device_constraint));

  /* Rename the columns, create and execute plan. */
  boost::shared_ptr<logical_operator::Logical_Rename> rename_final;
  {
    RenameList rename_list;
    rename_list.push_back(RenameEntry("NATION.N_NAME", "NATION"));
    rename_final =
        boost::make_shared<logical_operator::Logical_Rename>(rename_list);
  }

  group->setLeft(extract_year);  // column_algebra_operation_final_sub);
  order->setLeft(group);
  rename_final->setLeft(order);

  LogicalQueryPlanPtr log_plan =
      boost::make_shared<LogicalQueryPlan>(rename_final);  // order);
  return log_plan;
}

// gets error during execution in group by operator

bool TPCH_Q10(ClientPtr client) {
  //        return SQL::commandlineExec("select c_custkey, c_name,
  //        sum(l_extendedprice * (1 - l_discount)) as revenue from customer,
  //        orders, lineitem, nation where c_custkey = o_custkey and l_orderkey
  //        = o_orderkey and o_orderdate >= '1993-10-01' and o_orderdate <
  //        '1994-01-01' and l_returnflag = 'R' and c_nationkey = n_nationkey
  //        group by c_custkey, c_name order by revenue desc;",
  //                client);
  //        return SQL::commandlineExec("select c_custkey, c_name, c_acctbal,
  //        c_phone, n_name, c_address, c_comment, sum(l_extendedprice * (1 -
  //        l_discount)) as revenue from customer, orders, lineitem, nation
  //        where c_custkey = o_custkey and l_orderkey = o_orderkey and
  //        o_orderdate >= '1993-10-01' and o_orderdate < '1994-01-01' and
  //        l_returnflag = 'R' and c_nationkey = n_nationkey group by c_custkey,
  //        c_name, c_acctbal, c_phone, n_name, c_address, c_comment order by
  //        revenue desc limit 20;",
  return SQL::commandlineExec(
      "select c_custkey, c_name, c_acctbal, c_phone, n_name, c_address, "
      "c_comment, sum(l_extendedprice * (1 - l_discount)) as revenue from "
      "customer JOIN orders ON (c_custkey = o_custkey) JOIN nation ON "
      "(c_nationkey = n_nationkey) JOIN lineitem ON (o_orderkey = l_orderkey) "
      "where o_orderdate >= '1993-10-01' and o_orderdate < '1994-01-01' and "
      "l_returnflag = 'R' group by c_custkey, c_name, c_acctbal, c_phone, "
      "n_name, c_address, c_comment order by revenue desc limit 20;",
      client);
}

TypedNodePtr getTPCHQ15RevenueSubplan() {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  /* Table SCANs */
  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_lineitem;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("LINEITEM.L_SHIPDATE",
                            boost::any(std::string("1996-04-01")),
                            ValueConstantPredicate, LESSER));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(Predicate("LINEITEM.L_SHIPDATE",
                            boost::any(std::string("1996-01-01")),
                            ValueConstantPredicate, GREATER_EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_lineitem =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  complex_selection_on_lineitem->setLeft(scan_lineitem);

  /* Now we compute the aggregation expression (1-L_DISCOUNT)*L_EXTENDEDPRICE.
   * First we create a column representeing the 1. Then we perform the
   * substract and multiply operations. */
  boost::shared_ptr<logical_operator::Logical_AddConstantValueColumn>
      one_column =
          boost::make_shared<logical_operator::Logical_AddConstantValueColumn>(
              "1", DOUBLE, boost::any(double(1)), default_device_constraint);
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_one_minus_discount(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "1", "LINEITEM.L_DISCOUNT", "(1-LINEITEM.L_DISCOUNT)", SUB,
              default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_mul_extendedprice(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "(1-LINEITEM.L_DISCOUNT)", "LINEITEM.L_EXTENDEDPRICE", "VOLUME",
              MUL, default_device_constraint));

  one_column->setLeft(complex_selection_on_lineitem);
  column_algebra_operation_one_minus_discount->setLeft(one_column);
  column_algebra_operation_mul_extendedprice->setLeft(
      column_algebra_operation_one_minus_discount);

  /* Group By and Order By operations. */
  std::list<std::string> sorting_column_names;
  sorting_column_names.push_back("LINEITEM.L_SUPPKEY");

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      make_pair("VOLUME", Aggregate(SUM, "TOTAL_REVENUE")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(sorting_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));

  group->setLeft(column_algebra_operation_mul_extendedprice);

  boost::shared_ptr<logical_operator::Logical_Rename> rename;
  {
    RenameList rename_list;
    rename_list.push_back(RenameEntry("LINEITEM.L_SUPPKEY", "SUPPLIER_NO"));
    rename = boost::make_shared<logical_operator::Logical_Rename>(rename_list);
  }
  rename->setLeft(group);

  return rename;
}

TypedNodePtr getTPCHQ15MaximumRevenueSubplan() {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
  TypedNodePtr revenue_sub_plan = getTPCHQ15RevenueSubplan();

  boost::shared_ptr<logical_operator::Logical_Groupby> groupby;
  {
    std::list<std::string> grouping_column_names;
    //        grouping_column_names.push_back("_GROUPING_COLUMN");

    typedef std::pair<AggregationMethod, std::string> Aggregate;
    // column_name_to_aggregate, aggregate
    typedef std::pair<std::string, Aggregate> ColumnAggregation;
    std::list<ColumnAggregation> aggregation_functions;
    aggregation_functions.push_back(
        make_pair("TOTAL_REVENUE", Aggregate(MAX, "MAX_TOTAL_REVENUE")));

    groupby = boost::make_shared<logical_operator::Logical_Groupby>(
        grouping_column_names, aggregation_functions, LOOKUP,
        default_device_constraint);
  }
  groupby->setLeft(revenue_sub_plan);
  return groupby;
}

bool TPCH_Q15(ClientPtr client) {
  LogicalQueryPlanPtr log_plan = getPlan_TPCH_Q15(client);
  return optimize_execute_print("tpch15", *log_plan, client);
}

LogicalQueryPlanPtr getPlan_TPCH_Q15(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  TypedNodePtr revenue_sub_plan = getTPCHQ15RevenueSubplan();
  LogicalQueryPlanPtr log_plan_subquery =
      boost::make_shared<LogicalQueryPlan>(revenue_sub_plan);

  CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
      CoGaDB::query_processing::optimize_and_execute(
          "revenue_view", *log_plan_subquery, client);
  assert(plan != NULL);
  TablePtr result_revenue_subquery = plan->getResult();
  //        {
  //            std::list<SortAttribute> sort_attributes;
  //            sort_attributes.push_back(SortAttribute("TOTAL_REVENUE",DESCENDING));
  //// sort_attributes.push_back(SortAttribute("SUPPLIER_NO",ASCENDING));
  //
  //            result_revenue_subquery=BaseTable::sort(result_revenue_subquery,
  //            sort_attributes, LOOKUP, CPU);
  //            result_revenue_subquery->print();
  //        }

  result_revenue_subquery->setName("REVENUE_VIEW");
  /* Table SCANs */
  boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(
      new logical_operator::Logical_Scan("SUPPLIER"));
  boost::shared_ptr<logical_operator::Logical_Scan>
      scan_revenue_sub_query_result(
          new logical_operator::Logical_Scan(result_revenue_subquery));
  boost::shared_ptr<logical_operator::Logical_Scan>
      scan_revenue_sub_query_result2(
          new logical_operator::Logical_Scan(result_revenue_subquery));

  //        TypedNodePtr maximal_revenue_sub_plan =
  //        getTPCHQ15MaximumRevenueSubplan();

  boost::shared_ptr<logical_operator::Logical_Groupby>
      groupby_on_revenue_subquery;
  {
    std::list<std::string> grouping_column_names;

    typedef std::pair<AggregationMethod, std::string> Aggregate;
    typedef std::pair<std::string, Aggregate> ColumnAggregation;
    std::list<ColumnAggregation> aggregation_functions;
    aggregation_functions.push_back(
        make_pair("TOTAL_REVENUE", Aggregate(MAX, "MAX_TOTAL_REVENUE")));

    groupby_on_revenue_subquery =
        boost::make_shared<logical_operator::Logical_Groupby>(
            grouping_column_names, aggregation_functions, LOOKUP,
            default_device_constraint);
  }
  groupby_on_revenue_subquery->setLeft(
      scan_revenue_sub_query_result);  // revenue_sub_plan);

  /* Now we compute the joins. */
  boost::shared_ptr<logical_operator::Logical_Join> join_revenue_max_revenue(
      new logical_operator::Logical_Join("TOTAL_REVENUE", "MAX_TOTAL_REVENUE",
                                         INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join>
      join_revenue_max_revenue_supplier(new logical_operator::Logical_Join(
          "SUPPLIER_NO", "SUPPLIER.S_SUPPKEY", INNER_JOIN,
          default_device_constraint));

  join_revenue_max_revenue->setLeft(
      scan_revenue_sub_query_result2);  // revenue_sub_plan);
  join_revenue_max_revenue->setRight(
      groupby_on_revenue_subquery);  // maximal_revenue_sub_plan);

  join_revenue_max_revenue_supplier->setLeft(join_revenue_max_revenue);
  join_revenue_max_revenue_supplier->setRight(scan_supplier);

  boost::shared_ptr<logical_operator::Logical_Projection> project;
  {
    std::list<std::string> projected_columns;
    projected_columns.push_back("SUPPLIER.S_SUPPKEY");
    projected_columns.push_back("SUPPLIER.S_NAME");
    projected_columns.push_back("SUPPLIER.S_ADDRESS");
    projected_columns.push_back("SUPPLIER.S_PHONE");
    projected_columns.push_back("TOTAL_REVENUE");
    project = boost::make_shared<logical_operator::Logical_Projection>(
        projected_columns);
  }

  project->setLeft(join_revenue_max_revenue_supplier);

  LogicalQueryPlanPtr log_plan =
      boost::make_shared<LogicalQueryPlan>(project);  // order);

  return log_plan;
}

/* select
    sum(l_extendedprice) / 7.0 as avg_yearly
from
    lineitem,
    part
where
    p_partkey = l_partkey
    and p_brand = 'Brand#23'
    and p_container = 'MED BOX'
    and l_quantity < (
            select
                    0.2 * avg(l_quantity)
            from
                    lineitem
            where
                    l_partkey = p_partkey
    )*/

TypedNodePtr getTPCHQ17Subplan() {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));

  std::list<std::string> grouping_column_names;
  grouping_column_names.push_back("LINEITEM.L_PARTKEY");

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      make_pair("LINEITEM.L_QUANTITY", Aggregate(AVERAGE, "AVG_QUANTITY")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(grouping_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));
  group->setLeft(scan_lineitem);

  /* Now we compute the aggregation expression (0.2*AVG_QUANTITY).
   * First we create a column representeing the 0.2. Then we perform the
   * multiplication operation. */
  boost::shared_ptr<logical_operator::Logical_AddConstantValueColumn>
      constant_column =
          boost::make_shared<logical_operator::Logical_AddConstantValueColumn>(
              "0.2", DOUBLE, boost::any(double(0.2)),
              default_device_constraint);
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_mul(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "0.2", "AVG_QUANTITY", "SUBQUERY_AGGREGATION_RESULT", MUL,
              default_device_constraint));

  constant_column->setLeft(group);
  column_algebra_operation_mul->setLeft(constant_column);

  return column_algebra_operation_mul;

  //        boost::shared_ptr<logical_operator::Logical_Rename> rename;
  //        {
  //            RenameList rename_list;
  //            rename_list.push_back(RenameEntry("L_PARTKEY", "L2.L_PARTKEY"));
  //            rename =
  //            boost::make_shared<logical_operator::Logical_Rename>(rename_list);
  // //        rename->setLeft(column_algebra_operation_mul);
  ////        return rename;
  //
  //        boost::shared_ptr<logical_operator::Logical_Projection> project;
  //        {
  //            std::list<std::string> projected_columns;
  //            projected_columns.push_back("L_PARTKEY");
  //            projected_columns.push_back("SUBQUERY_AGGREGATION_RESULT");
  //            project =
  //            boost::make_shared<logical_operator::Logical_Projection>(projected_columns);
  //        }
  //
  //
  //
  //        project->setLeft(column_algebra_operation_mul);
  //        return project;
  //
  //        rename->setLeft(project);
  //
  //        return rename;         }
  ////        rename->setLeft(column_algebra_operation_mul);
  ////        return rename;
  //
  //        boost::shared_ptr<logical_operator::Logical_Projection> project;
  //        {
  //            std::list<std::string> projected_columns;
  //            projected_columns.push_back("L_PARTKEY");
  //            projected_columns.push_back("SUBQUERY_AGGREGATION_RESULT");
  //            project =
  //            boost::make_shared<logical_operator::Logical_Projection>(projected_columns);
  //        }
  //
  //
  //
  //        project->setLeft(column_algebra_operation_mul);
  //        return project;
  //
  //        rename->setLeft(project);
  //
  //        return rename;

  //        return column_algebra_operation_mul;
}

bool TPCH_Q17(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  /* Table SCANs */
  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_part(
      new logical_operator::Logical_Scan("PART"));

  TypedNodePtr sub_query = getTPCHQ17Subplan();

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_part;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("PART.P_BRAND", boost::any(std::string("Brand#23")),
                            ValueConstantPredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(Predicate("PART.P_CONTAINER",
                            boost::any(std::string("MED BOX")),
                            ValueConstantPredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_part =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  complex_selection_on_part->setLeft(scan_part);

  boost::shared_ptr<logical_operator::Logical_Join> join_part_lineitem(
      new logical_operator::Logical_Join("PART.P_PARTKEY", "LINEITEM.L_PARTKEY",
                                         INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_part_lineitem_subquery(
      new logical_operator::Logical_Join("PART.P_PARTKEY", "LINEITEM.L_PARTKEY",
                                         INNER_JOIN,
                                         default_device_constraint));

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_part_lineitem_subquery;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      // d.push_back(Predicate("N2.N_NAME", boost::any(std::string("FRANCE")),
      // ValueConstantPredicate, EQUAL));
      d.push_back(Predicate("LINEITEM.L_QUANTITY",
                            std::string("SUBQUERY_AGGREGATION_RESULT"),
                            ValueValuePredicate, LESSER));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_part_lineitem_subquery =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  join_part_lineitem->setLeft(complex_selection_on_part);
  join_part_lineitem->setRight(scan_lineitem);

  join_part_lineitem_subquery->setLeft(join_part_lineitem);
  join_part_lineitem_subquery->setRight(sub_query);

  complex_selection_on_part_lineitem_subquery->setLeft(
      join_part_lineitem_subquery);

  /* Aggregation Expression of outer query. First comes the SUM aggregation. */
  //        boost::shared_ptr<logical_operator::Logical_AddConstantValueColumn>
  //        create_grouping_column =
  //        boost::make_shared<logical_operator::Logical_AddConstantValueColumn>
  //                ("_GROUPING_COLUMN", INT, boost::any(int32_t(1),
  //                default_device_constraint));
  //
  //        create_grouping_column->setLeft(complex_selection_on_part_lineitem_subquery);
  //
  std::list<std::string> group_column_names;
  //        group_column_names.push_back("_GROUPING_COLUMN");

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(make_pair(
      "LINEITEM.L_EXTENDEDPRICE", Aggregate(SUM, "SUM_EXTENDEDPRICE")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(group_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));

  group->setLeft(
      complex_selection_on_part_lineitem_subquery);  // create_grouping_column);

  /* Now we compute the aggregation expression (sum(l_extendedprice) / 7.0).
   * First we create a column representeing the 7.0. Then we perform the
   * division operation. */
  boost::shared_ptr<logical_operator::Logical_AddConstantValueColumn>
      constant_column =
          boost::make_shared<logical_operator::Logical_AddConstantValueColumn>(
              "7.0", DOUBLE, boost::any(double(7.0)),
              default_device_constraint);
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_div(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "SUM_EXTENDEDPRICE", "7.0", "AVG_YEARLY", DIV,
              default_device_constraint));

  constant_column->setLeft(group);
  column_algebra_operation_div->setLeft(constant_column);

  {
    SortAttributeList sorting_attributes;
    sorting_attributes.push_back(
        SortAttribute("LINEITEM.L_PARTKEY", DESCENDING));

    boost::shared_ptr<logical_operator::Logical_Sort> order(
        new logical_operator::Logical_Sort(sorting_attributes, LOOKUP,
                                           default_device_constraint));

    order->setLeft(complex_selection_on_part_lineitem_subquery);

    std::list<std::string> group_column_names;
    group_column_names.push_back("LINEITEM.L_PARTKEY");

    typedef std::pair<AggregationMethod, std::string> Aggregate;
    // column_name_to_aggregate, aggregate
    typedef std::pair<std::string, Aggregate> ColumnAggregation;
    std::list<ColumnAggregation> aggregation_functions;
    aggregation_functions.push_back(make_pair(
        "LINEITEM.L_ORDERKEY", Aggregate(COUNT, "LINEITEM.L_ORDERKEY")));

    boost::shared_ptr<logical_operator::Logical_Groupby> group(
        new logical_operator::Logical_Groupby(group_column_names,
                                              aggregation_functions, LOOKUP,
                                              default_device_constraint));

    //        group->setLeft(join_part_lineitem_subquery);
    //        order->setLeft(group);
    LogicalQueryPlanPtr log_plan = boost::make_shared<LogicalQueryPlan>(
        order);  // join_part_lineitem_subquery); //create_grouping_column);
                 // //column_algebra_operation_div);
    return optimize_execute_print("tpch17", *log_plan, client);
  }

  LogicalQueryPlanPtr log_plan = boost::make_shared<LogicalQueryPlan>(
      complex_selection_on_part_lineitem_subquery);  // column_algebra_operation_div);
  return optimize_execute_print("tpch17", *log_plan, client);
}

TypedNodePtr getTPCHQ18Subplan() {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));

  std::list<std::string> sorting_column_names;
  sorting_column_names.push_back("LINEITEM.L_ORDERKEY");

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      make_pair("LINEITEM.L_QUANTITY", Aggregate(SUM, "SUM_QUANTITY")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(sorting_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));
  group->setLeft(scan_lineitem);

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_sum_quantity;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("SUM_QUANTITY", boost::any(double(300)),
                            ValueConstantPredicate, GREATER));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_sum_quantity =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }
  complex_selection_on_sum_quantity->setLeft(group);

  boost::shared_ptr<logical_operator::Logical_Projection> project;
  {
    std::list<std::string> projected_columns;
    projected_columns.push_back("LINEITEM.L_ORDERKEY");
    project = boost::make_shared<logical_operator::Logical_Projection>(
        projected_columns);
  }

  project->setLeft(complex_selection_on_sum_quantity);

  boost::shared_ptr<logical_operator::Logical_Rename> rename;
  {
    RenameList rename_list;
    rename_list.push_back(RenameEntry("LINEITEM.L_ORDERKEY", "L2.L_ORDERKEY"));
    rename = boost::make_shared<logical_operator::Logical_Rename>(rename_list);
  }
  rename->setLeft(project);

  return rename;
}

bool TPCH_Q18(ClientPtr client) {
  LogicalQueryPlanPtr log_plan = getPlan_TPCH_Q18(client);
  return optimize_execute_print("tpch18", *log_plan, client);
}

LogicalQueryPlanPtr getPlan_TPCH_Q18(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  /* Table SCANs */
  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_orders(
      new logical_operator::Logical_Scan("ORDERS"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(
      new logical_operator::Logical_Scan("CUSTOMER"));

  TypedNodePtr sub_query = getTPCHQ18Subplan();

  /* Now we compute the joins. */
  boost::shared_ptr<logical_operator::Logical_Join> join_subquery_orders(
      new logical_operator::Logical_Join(
          "LINEITEM.L_ORDERKEY", "ORDERS.O_ORDERKEY", RIGHT_SEMI_JOIN,
          convertToSemiJoinConstraint(default_device_constraint)));
  boost::shared_ptr<logical_operator::Logical_Join> join_customer(
      new logical_operator::Logical_Join("ORDERS.O_CUSTKEY",
                                         "CUSTOMER.C_CUSTKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join_lineitem(
      new logical_operator::Logical_Join("ORDERS.O_ORDERKEY",
                                         "LINEITEM.L_ORDERKEY", INNER_JOIN,
                                         default_device_constraint));

  join_subquery_orders->setLeft(sub_query);
  join_subquery_orders->setRight(scan_orders);

  join_customer->setLeft(join_subquery_orders);
  join_customer->setRight(scan_customer);

  join_lineitem->setLeft(join_customer);
  join_lineitem->setRight(scan_lineitem);

  std::list<std::string> group_column_names;
  group_column_names.push_back("CUSTOMER.C_NAME");
  group_column_names.push_back("CUSTOMER.C_CUSTKEY");
  group_column_names.push_back("ORDERS.O_ORDERKEY");
  group_column_names.push_back("ORDERS.O_ORDERDATE");
  group_column_names.push_back("ORDERS.O_TOTALPRICE");

  SortAttributeList sorting_attributes;
  sorting_attributes.push_back(
      SortAttribute("ORDERS.O_TOTALPRICE", DESCENDING));
  sorting_attributes.push_back(SortAttribute("ORDERS.O_ORDERDATE", ASCENDING));

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      make_pair("LINEITEM.L_QUANTITY", Aggregate(SUM, "SUM_QUANTITY")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(group_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Sort> order(
      new logical_operator::Logical_Sort(sorting_attributes, LOOKUP,
                                         default_device_constraint));

  group->setLeft(join_lineitem);
  order->setLeft(group);

  UDF_Parameters limit_param;
  limit_param.push_back(UDF_Parameter(size_t(100)));

  boost::shared_ptr<logical_operator::Logical_UDF> limit(
      new logical_operator::Logical_UDF("LIMIT", limit_param,
                                        default_device_constraint));
  limit->setLeft(order);

  LogicalQueryPlanPtr log_plan =
      boost::make_shared<LogicalQueryPlan>(limit);  // order);
  return log_plan;
}

TypedNodePtr getTPCHQ20Subplan() {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_lineitem;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("LINEITEM.L_SHIPDATE",
                            boost::any(std::string("1994-01-01")),
                            ValueConstantPredicate, GREATER_EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(Predicate("LINEITEM.L_SHIPDATE",
                            boost::any(std::string("1995-01-01")),
                            ValueConstantPredicate, LESSER));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_lineitem =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  complex_selection_on_lineitem->setLeft(scan_lineitem);

  std::list<std::string> grouping_column_names;
  grouping_column_names.push_back("LINEITEM.L_PARTKEY");
  grouping_column_names.push_back("LINEITEM.L_SUPPKEY");

  typedef std::pair<AggregationMethod, std::string> Aggregate;
  // column_name_to_aggregate, aggregate
  typedef std::pair<std::string, Aggregate> ColumnAggregation;
  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      make_pair("LINEITEM.L_QUANTITY", Aggregate(SUM, "SUM_QUANTITY")));

  boost::shared_ptr<logical_operator::Logical_Groupby> group(
      new logical_operator::Logical_Groupby(grouping_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));
  group->setLeft(complex_selection_on_lineitem);

  /* Now we compute the aggregation expression (0.2*AVG_QUANTITY).
   * First we create a column representeing the 0.2. Then we perform the
   * multiplication operation. */
  boost::shared_ptr<logical_operator::Logical_AddConstantValueColumn>
      constant_column =
          boost::make_shared<logical_operator::Logical_AddConstantValueColumn>(
              "0.5", DOUBLE, boost::any(double(0.5)),
              default_device_constraint);
  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_mul(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "0.5", "SUM_QUANTITY", "SUBQUERY_AGGREGATION_RESULT", MUL,
              default_device_constraint));

  constant_column->setLeft(group);
  column_algebra_operation_mul->setLeft(constant_column);

  return column_algebra_operation_mul;
}

bool TPCH_Q20(ClientPtr client) {
  LogicalQueryPlanPtr log_plan = getPlan_TPCH_Q20(client);
  return optimize_execute_print("tpch20", *log_plan, client);
}

LogicalQueryPlanPtr getPlan_TPCH_Q20(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(
      new logical_operator::Logical_Scan("SUPPLIER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_nation(
      new logical_operator::Logical_Scan("NATION"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_part(
      new logical_operator::Logical_Scan("PART"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_partsupp(
      new logical_operator::Logical_Scan("PARTSUPP"));

  TypedNodePtr subplan = getTPCHQ20Subplan();

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_part;
  {
    {
      KNF_Selection_Expression knf_expr;
      {
        Disjunction d;
        d.push_back(Predicate("PART.P_NAME",
                              boost::any(std::string("forest.*")),
                              ValueRegularExpressionPredicate, EQUAL));
        knf_expr.disjunctions.push_back(d);
      }
      complex_selection_on_part =
          boost::make_shared<logical_operator::Logical_ComplexSelection>(
              knf_expr, LOOKUP, convertToRegularExpressionSelectionConstraint(
                                    default_device_constraint));
    }
  }
  complex_selection_on_part->setLeft(scan_part);

  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_nation;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("NATION.N_NAME", boost::any(std::string("CANADA")),
                            ValueConstantPredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_nation =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }
  complex_selection_on_nation->setLeft(scan_nation);

  boost::shared_ptr<logical_operator::Logical_Join> join_nation_supplier(
      new logical_operator::Logical_Join("NATION.N_NATIONKEY",
                                         "SUPPLIER.S_NATIONKEY", INNER_JOIN,
                                         default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join>
      right_semi_join_part_partsupp(new logical_operator::Logical_Join(
          "PART.P_PARTKEY", "PARTSUPP.PS_PARTKEY", RIGHT_SEMI_JOIN,
          convertToSemiJoinConstraint(
              default_device_constraint)));  // hype::CPU_ONLY));
  boost::shared_ptr<logical_operator::Logical_Join> join_part_partsupp_subquery(
      new logical_operator::Logical_Join("PARTSUPP.PS_PARTKEY",
                                         "LINEITEM.L_PARTKEY", INNER_JOIN,
                                         default_device_constraint));

  // additional join conditions
  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection_on_join_part_partsupp_subquery;
  {
    KNF_Selection_Expression knf_expr;
    {
      Disjunction d;
      d.push_back(Predicate("PARTSUPP.PS_SUPPKEY",
                            std::string("LINEITEM.L_SUPPKEY"),
                            ValueValuePredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(Predicate("PARTSUPP.PS_AVAILQTY",
                            std::string("SUBQUERY_AGGREGATION_RESULT"),
                            ValueValuePredicate, GREATER));
      knf_expr.disjunctions.push_back(d);
    }
    complex_selection_on_join_part_partsupp_subquery =
        boost::make_shared<logical_operator::Logical_ComplexSelection>(
            knf_expr, LOOKUP, default_device_constraint);
  }

  boost::shared_ptr<logical_operator::Logical_Join>
      left_semi_join_nation_supplier_part_partsupp_subplan(
          new logical_operator::Logical_Join(
              "SUPPLIER.S_SUPPKEY", "PARTSUPP.PS_SUPPKEY", LEFT_SEMI_JOIN,
              convertToSemiJoinConstraint(
                  default_device_constraint)));  // hype::CPU_ONLY));

  join_nation_supplier->setLeft(complex_selection_on_nation);
  join_nation_supplier->setRight(scan_supplier);

  right_semi_join_part_partsupp->setLeft(complex_selection_on_part);
  right_semi_join_part_partsupp->setRight(scan_partsupp);

  join_part_partsupp_subquery->setLeft(right_semi_join_part_partsupp);
  join_part_partsupp_subquery->setRight(subplan);

  complex_selection_on_join_part_partsupp_subquery->setLeft(
      join_part_partsupp_subquery);

  left_semi_join_nation_supplier_part_partsupp_subplan->setLeft(
      join_nation_supplier);
  left_semi_join_nation_supplier_part_partsupp_subplan->setRight(
      complex_selection_on_join_part_partsupp_subquery);

  SortAttributeList sorting_attributes;
  sorting_attributes.push_back(SortAttribute("SUPPLIER.S_NAME", ASCENDING));

  boost::shared_ptr<logical_operator::Logical_Sort> order(
      new logical_operator::Logical_Sort(sorting_attributes, LOOKUP,
                                         default_device_constraint));
  order->setLeft(left_semi_join_nation_supplier_part_partsupp_subplan);

  boost::shared_ptr<logical_operator::Logical_Projection> project;
  {
    std::list<std::string> projected_columns;
    projected_columns.push_back("SUPPLIER.S_NAME");
    projected_columns.push_back("SUPPLIER.S_ADDRESS");
    project = boost::make_shared<logical_operator::Logical_Projection>(
        projected_columns);
  }

  project->setLeft(order);

  LogicalQueryPlanPtr log_plan = boost::make_shared<LogicalQueryPlan>(
      project);  // complex_selection_on_join_part_partsupp_subquery);
                 // //project);
  return log_plan;
}

bool TPCH_Q21(ClientPtr client) {
  //        hype::DeviceConstraint default_device_constraint =
  //        CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
  //
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
  //        logical_operator::Logical_Scan("SUPPLIER"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_nation(new
  //        logical_operator::Logical_Scan("NATION"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_nation(new
  //        logical_operator::Logical_Scan("ORDERS"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem1(new
  //        logical_operator::Logical_Scan("LINEITEM"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem2(new
  //        logical_operator::Logical_Scan("LINEITEM"));
  //        boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem3(new
  //        logical_operator::Logical_Scan("LINEITEM"));
  //
  //        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
  //        complex_selection_on_nation;
  //        {
  //            KNF_Selection_Expression knf_expr;
  //            {
  //                Disjunction d;
  //                d.push_back(Predicate("N_NAME",
  //                boost::any(std::string("CANADA")), ValueConstantPredicate,
  //                EQUAL));
  //                knf_expr.disjunctions.push_back(d);
  //            }
  //            complex_selection_on_nation =
  //            boost::make_shared<logical_operator::Logical_ComplexSelection>(knf_expr,
  //            LOOKUP, default_device_constraint);
  //        }
  //        complex_selection_on_nation->setLeft(scan_nation);
  //
  //        boost::shared_ptr<logical_operator::Logical_Join>
  //        join_nation_supplier(new
  //        logical_operator::Logical_Join("N_NATIONKEY", "S_NATIONKEY",
  //        INNER_JOIN, default_device_constraint));
  //        boost::shared_ptr<logical_operator::Logical_Join> join_lineitem(new
  //        logical_operator::Logical_Join("S_SUPPKEY", "L_SUPPKEY", INNER_JOIN,
  //        default_device_constraint));
  //        boost::shared_ptr<logical_operator::Logical_Join> join_orders(new
  //        logical_operator::Logical_Join("O_ORDERKEY", "L_ORDERKEY",
  //        INNER_JOIN, default_device_constraint));
  //
  //        boost::shared_ptr<logical_operator::Logical_Join>
  //        left_anti_semi_join_lineitem2(new
  //        logical_operator::Logical_Join("P_PARTKEY", "PS_PARTKEY",
  //        LEFT_ANTI_SEMI_JOIN, hype::CPU_ONLY));
  //
  //        //additional join conditions
  //        boost::shared_ptr<logical_operator::Logical_ComplexSelection>
  //        complex_selection_on_left_anti_semi_join;
  //        {
  //            KNF_Selection_Expression knf_expr;
  //            {
  //                Disjunction d;
  //                d.push_back(Predicate("PS_SUPPKEY",
  //                std::string("L_SUPPKEY"), ValueValuePredicate, EQUAL));
  //                knf_expr.disjunctions.push_back(d);
  //            }
  //            {
  //                Disjunction d;
  //                d.push_back(Predicate("PS_AVAILQTY",
  //                std::string("SUBQUERY_AGGREGATION_RESULT"),
  //                ValueValuePredicate, GREATER));
  //                knf_expr.disjunctions.push_back(d);
  //            }
  //            complex_selection_on_left_anti_semi_join =
  //            boost::make_shared<logical_operator::Logical_ComplexSelection>(knf_expr,
  //            LOOKUP, default_device_constraint);
  //        }
  //
  //        boost::shared_ptr<logical_operator::Logical_Join>
  //        left_semi_join_lineitem3(new
  //        logical_operator::Logical_Join("P_PARTKEY", "PS_PARTKEY",
  //        LEFT_SEMI_JOIN, hype::CPU_ONLY));
  //
  //
  //
  //
  //        boost::shared_ptr<logical_operator::Logical_Join>
  //        right_semi_join_part_partsupp(new
  //        logical_operator::Logical_Join("P_PARTKEY", "PS_PARTKEY",
  //        RIGHT_SEMI_JOIN, hype::CPU_ONLY));
  //

  return SQL::commandlineExec("", client);
}

bool TPCH_Q1_hand_compiled_cpu_single_threaded(ClientPtr client) {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  /* Table SCANs */
  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(
      new logical_operator::Logical_Scan("LINEITEM"));

  UDF_Parameters param;

  boost::shared_ptr<logical_operator::Logical_UDF> tpch1_compiled(
      new logical_operator::Logical_UDF(
          "UDF.EXPERIMENTAL.HAND_COMPILED_TPCH_Q1", param,
          default_device_constraint));
  tpch1_compiled->setLeft(scan_lineitem);

  LogicalQueryPlanPtr log_plan =
      boost::make_shared<LogicalQueryPlan>(tpch1_compiled);  // order);
  return optimize_execute_print("tpch1_hand_compiled", *log_plan, client);
}

bool TPCH_QX(ClientPtr client) { return SQL::commandlineExec("", client); }

}  // end namespace CogaDB
