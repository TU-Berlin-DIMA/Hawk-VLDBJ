#include <fstream>
#include <iomanip>
#include <iostream>

#include <dlfcn.h>
#include <stdlib.h>

#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>

#include <core/global_definitions.hpp>
#include <core/selection_expression.hpp>
#include <core/user_defined_function.hpp>
#include <core/variable_manager.hpp>
#include <parser/commandline_interpreter.hpp>
#include <parser/json_parser.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/cpp_code_generator.hpp>
#include <query_compilation/pipeline_job.hpp>
#include <query_compilation/query_context.hpp>
#include <query_compilation/user_defined_code.hpp>
#include <util/getname.hpp>
#include <util/tests.hpp>
#include <util/time_measurement.hpp>
#include <util/variant_measurement.hpp>

using namespace CoGaDB;

const TablePtr execute(CodeGeneratorPtr code_gen, QueryContextPtr context,
                       bool print = false) {
  PipelinePtr pipeline = code_gen->compile();
  if (!pipeline) return TablePtr();
  if (!execute(pipeline, context)) return TablePtr();
  TablePtr result = pipeline->getResult();
  if (print) {
    if (!result) {
      std::cerr << "Could access pipeline result successfully!" << std::endl;
      return TablePtr();
    } else {
      result->print();
    }
  }
  return result;
}

typedef std::pair<bool, int> HashAggregationOptimizationState;

const HashAggregationOptimizationState setSizeOfHashAggregationTable(
    int num_groups) {
  /* save parameters */
  bool enabled_manual_ht_size =
      VariableManager::instance().getVariableValueBoolean(
          "code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_size");
  int manual_ht_size = VariableManager::instance().getVariableValueInteger(
      "code_gen.opt.ocl_grouped_aggregation.hack.ht_size");

  /* set new parameter */
  VariableManager::instance().setVariableValue(
      "code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_size",
      "true");
  VariableManager::instance().setVariableValue(
      "code_gen.opt.ocl_grouped_aggregation.hack.ht_size",
      boost::lexical_cast<std::string>(num_groups));
  return HashAggregationOptimizationState(enabled_manual_ht_size,
                                          manual_ht_size);
}

void resetSizeOfHashAggregationTable(
    const HashAggregationOptimizationState state) {
  bool enabled_manual_ht_size = state.first;
  int manual_ht_size = state.second;
  /* reset previously overridden parameters */
  if (enabled_manual_ht_size) {
    VariableManager::instance().setVariableValue(
        "code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_size",
        "true");
  } else {
    VariableManager::instance().setVariableValue(
        "code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_size",
        "false");
  }
  VariableManager::instance().setVariableValue(
      "code_gen.opt.ocl_grouped_aggregation.hack.ht_size",
      boost::lexical_cast<std::string>(manual_ht_size));
}

const TablePtr qcrev_tpch1(CodeGeneratorType code_generator) {
  // json available
  ProjectionParam param;
  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));
  QueryContextPtr context = createQueryContext();
  context->markAsOriginalContext();

  AttributeReferencePtr l_shipdate = boost::make_shared<AttributeReference>(
      getTablebyName("LINEITEM"), "LINEITEM.L_SHIPDATE", "SHIPDATE");

  std::vector<PredicateExpressionPtr> conjunctions;
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      l_shipdate, boost::any(std::string("1998-09-01")), LESSER_EQUAL));

  PredicateExpressionPtr selection_expr =
      createPredicateExpression(conjunctions, LOGICAL_AND);

  if (!code_gen->consumeSelection(selection_expr)) COGADB_FATAL_ERROR("", "");

  std::pair<bool, AttributeReference> ret = code_gen->consumeAlgebraComputation(
      boost::any(double(1)),
      AttributeReference(getTablebyName("LINEITEM"), "LINEITEM.L_DISCOUNT"),
      SUB);

  if (!ret.first) {
    COGADB_FATAL_ERROR("", "");
  }

  AttributeReference one_minus_discount = ret.second;

  ret = code_gen->consumeAlgebraComputation(
      AttributeReference(getTablebyName("LINEITEM"),
                         "LINEITEM.L_EXTENDEDPRICE"),
      one_minus_discount, MUL);

  if (!ret.first) {
    COGADB_FATAL_ERROR("", "");
  }

  AttributeReference extended_price_multiply_one_minus_discount = ret.second;

  ProcessorSpecification proc_spec(hype::PD0);
  GroupingAttributes grouping_columns;
  grouping_columns.push_back(AttributeReference(
      getTablebyName("LINEITEM"), "LINEITEM.L_RETURNFLAG", "RETURNFLAG"));
  AggregateSpecifications agg_specs;

  AttributeReference l_extendedprice(
      getTablebyName("LINEITEM"), "LINEITEM.L_EXTENDEDPRICE", "NUMBER_OF_ROWS");

  agg_specs.push_back(
      createAggregateSpecification(l_extendedprice, COUNT, "COUNT_ORDER"));
  agg_specs.push_back(createAggregateSpecification(
      AttributeReference(getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY",
                         "QUANTITY"),
      SUM, "SUM"));
  agg_specs.push_back(createAggregateSpecification(
      extended_price_multiply_one_minus_discount, SUM, "SUM_DISCOUNT_PRICE"));

  GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);
  if (!code_gen->consumeHashGroupAggregate(groupby_param))
    COGADB_FATAL_ERROR("", "");

  return execute(code_gen, context, true);
  //  PipelinePtr pipeline = code_gen->compile();
  //  if (!pipeline) return TablePtr();

  //  return execute(pipeline, true);
}

const TablePtr qcrev_tpch5_join(CodeGeneratorType code_generator) {
  return TablePtr();
}

// bool qcrev_tpch5_join(CodeGeneratorType code_generator) {
//  /* currently not working: FATAL ERROR: Column R_REGIONKEY.1 not found  in
//   * table REGION!: */
//  // json available

//  //-- TPC-H Query 5
//  //
//  // select
//  //        sum(l_extendedprice * (1 - l_discount)) as revenue
//  // from
//  //        customer,
//  //        orders,
//  //        lineitem,
//  //        supplier,
//  //        nation,
//  //        region
//  // where
//  //        c_custkey = o_custkey
//  //        and l_orderkey = o_orderkey
//  //        and l_suppkey = s_suppkey
//  //        and c_nationkey = s_nationkey
//  //        and s_nationkey = n_nationkey
//  //        and n_regionkey = r_regionkey
//  //        and r_name = 'ASIA'
//  //        and o_orderdate >= date '1994-01-01'
//  //        and o_orderdate < date '1995-01-01'

//  TablePtr region_result;
//  QueryContextPtr region_context;
//  {
//    PipelinePtr pipeline;
//    ProjectionParam param;
//    param.push_back(AttributeReference(getTablebyName("REGION"),
//    "R_REGIONKEY",
//                                       "R_REGIONKEY"));

//    CodeGeneratorPtr code_gen =
//        createCodeGenerator(code_generator, param, getTablebyName("REGION"));
//    region_context = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

//    AttributeReferencePtr r_regionkey =
//    boost::make_shared<AttributeReference>(
//        getTablebyName("REGION"), "R_REGIONKEY", "R_REGIONKEY");
//    AttributeReferencePtr r_name = boost::make_shared<AttributeReference>(
//        getTablebyName("REGION"), "R_NAME", "R_NAME");

//    std::vector<PredicateExpressionPtr> conjunctions;
//    conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
//        r_name, boost::any(uint32_t(2)), EQUAL));
//    PredicateExpressionPtr selection_expr =
//        createPredicateExpression(conjunctions, LOGICAL_AND);
//    if (!code_gen->consumeSelection(selection_expr)) COGADB_FATAL_ERROR("",
//    "");

//    if (!code_gen->consumeBuildHashTable(*r_regionkey))
//      COGADB_FATAL_ERROR("", "");

////    pipeline = code_gen->compile();
////    if (!execute(pipeline)) {
////      std::cerr << "Could not execute query successfully!" << std::endl;
////      return TablePtr();
////    }
//    region_result = execute(code_gen, region_context, true);
//  }

//  TablePtr nation_result;
//  QueryContextPtr nation_context;
//  {
//    PipelinePtr pipeline;
//    ProjectionParam param;
//    param.push_back(AttributeReference(getTablebyName("NATION"),
//    "N_REGIONKEY",
//                                       "N_REGIONKEY"));
//    param.push_back(AttributeReference(getTablebyName("NATION"),
//    "N_NATIONKEY",
//                                       "N_NATIONKEY"));

//    CodeGeneratorPtr code_gen =
//        createCodeGenerator(code_generator, param, getTablebyName("NATION"));
//    nation_context = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

//    AttributeReferencePtr n_nationkey =
//    boost::make_shared<AttributeReference>(
//        getTablebyName("NATION"), "N_NATIONKEY", "N_NATIONKEY");

//    if (!code_gen->consumeBuildHashTable(*n_nationkey))
//      COGADB_FATAL_ERROR("", "");

////    pipeline = code_gen->compile();
////    if (!execute(pipeline)) {
////      std::cerr << "Could not execute query successfully!" << std::endl;
////      return TablePtr();
////    }
////    nation_result = pipeline->getResult();
//    nation_result = execute(code_gen, nation_context, true);
//  }

//  TablePtr supplier_result;
//  QueryContextPtr supplier_context;
//  {
//    PipelinePtr pipeline;
//    ProjectionParam param;
//    param.push_back(AttributeReference(getTablebyName("SUPPLIER"),
//    "S_SUPPKEY",
//                                       "S_SUPPKEY"));
//    param.push_back(AttributeReference(getTablebyName("SUPPLIER"),
//                                       "S_NATIONKEY", "S_NATIONKEY"));

//    CodeGeneratorPtr code_gen =
//        createCodeGenerator(code_generator, param,
//        getTablebyName("SUPPLIER"));
//    supplier_context = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

//    AttributeReferencePtr s_suppkey = boost::make_shared<AttributeReference>(
//        getTablebyName("SUPPLIER"), "S_SUPPKEY", "S_SUPPKEY");

//    if (!code_gen->consumeBuildHashTable(*s_suppkey))
//      COGADB_FATAL_ERROR("", "");

//    pipeline = code_gen->compile();
//    if (!execute(pipeline)) {
//      std::cerr << "Could not execute query successfully!" << std::endl;
//      return TablePtr();
//    }
//    supplier_result = execute(code_gen, supplier_context, true);
//  }

//  TablePtr customer_result;
//  {
//    PipelinePtr pipeline;
//    ProjectionParam param;
//    param.push_back(AttributeReference(getTablebyName("CUSTOMER"),
//                                       "C_NATIONKEY", "C_NATIONKEY"));
//    param.push_back(AttributeReference(getTablebyName("CUSTOMER"),
//    "C_CUSTKEY",
//                                       "C_CUSTKEY"));

//    CodeGeneratorPtr code_gen =
//        createCodeGenerator(code_generator, param,
//        getTablebyName("CUSTOMER"));

//    AttributeReferencePtr c_custkey = boost::make_shared<AttributeReference>(
//        getTablebyName("CUSTOMER"), "C_CUSTKEY", "C_CUSTKEY");
//    AttributeReferencePtr c_nationkey =
//    boost::make_shared<AttributeReference>(
//        getTablebyName("CUSTOMER"), "C_NATIONKEY", "C_NATIONKEY");

//    if (!code_gen->consumeBuildHashTable(*c_custkey))
//      COGADB_FATAL_ERROR("", "");

//    pipeline = code_gen->compile();
//    if (!execute(pipeline)) {
//      std::cerr << "Could not execute query successfully!" << std::endl;
//      return TablePtr();
//    }
//    customer_result = pipeline->getResult();
//  }

//  TablePtr orders_result;
//  {
//    PipelinePtr pipeline;
//    ProjectionParam param;
//    param.push_back(AttributeReference(getTablebyName("ORDERS"), "O_ORDERKEY",
//                                       "O_ORDERKEY"));
//    param.push_back(
//        AttributeReference(getTablebyName("ORDERS"), "O_CUSTKEY",
//        "O_CUSTKEY"));
//    param.push_back(AttributeReference(getTablebyName("ORDERS"),
//    "O_ORDERDATE",
//                                       "O_ORDERDATE"));

//    CodeGeneratorPtr code_gen =
//        createCodeGenerator(code_generator, param, getTablebyName("ORDERS"));

//    AttributeReferencePtr o_custkey = boost::make_shared<AttributeReference>(
//        getTablebyName("ORDERS"), "O_CUSTKEY", "O_CUSTKEY");
//    AttributeReferencePtr o_orderkey = boost::make_shared<AttributeReference>(
//        getTablebyName("ORDERS"), "O_ORDERKEY", "O_ORDERKEY");
//    AttributeReferencePtr o_orderdate =
//    boost::make_shared<AttributeReference>(
//        getTablebyName("ORDERS"), "O_ORDERDATE", "O_ORDERDATE");

//    std::vector<PredicateExpressionPtr> conjunctions;
//    conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
//        o_orderdate, boost::any(int(19940101)), GREATER_EQUAL));
//    conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
//        o_orderdate, boost::any(int(19950101)), LESSER));
//    PredicateExpressionPtr selection_expr =
//        createPredicateExpression(conjunctions, LOGICAL_AND);
//    if (!code_gen->consumeSelection(selection_expr)) COGADB_FATAL_ERROR("",
//    "");

//    if (!code_gen->consumeBuildHashTable(*o_orderkey))
//      COGADB_FATAL_ERROR("", "");

//    pipeline = code_gen->compile();
//    if (!execute(pipeline)) {
//      std::cerr << "Could not execute query successfully!" << std::endl;
//      return TablePtr();
//    }
//    orders_result = pipeline->getResult();
//  }

//  // probe pipeline
//  TablePtr lineitem_result;
//  {
//    PipelinePtr pipeline;
//    ProjectionParam param;

//    CodeGeneratorPtr code_gen =
//        createCodeGenerator(code_generator, param,
//        getTablebyName("LINEITEM"));

//    AttributeReferencePtr l_extendedprice =
//        boost::make_shared<AttributeReference>(
//            getTablebyName("LINEITEM"), "L_EXTENDEDPRICE", "L_EXTENDEDPRICE");
//    AttributeReferencePtr l_discount = boost::make_shared<AttributeReference>(
//        getTablebyName("LINEITEM"), "L_DISCOUNT", "L_DISCOUNT");
//    AttributeReferencePtr l_orderkey = boost::make_shared<AttributeReference>(
//        getTablebyName("LINEITEM"), "L_ORDERKEY", "L_ORDERKEY");
//    AttributeReferencePtr l_suppkey = boost::make_shared<AttributeReference>(
//        getTablebyName("LINEITEM"), "L_SUPPKEY", "L_SUPPKEY");

//    // line 1
//    if (!code_gen->consumeProbeHashTable(
//            AttributeReference(orders_result, "O_ORDERKEY"), *l_orderkey))
//      COGADB_FATAL_ERROR("", "");

//    if (!code_gen->consumeProbeHashTable(
//            AttributeReference(customer_result, "C_CUSTKEY"),
//            AttributeReference(orders_result, "O_CUSTKEY")))
//      COGADB_FATAL_ERROR("", "");

//    // line 2
//    if (!code_gen->consumeProbeHashTable(
//            AttributeReference(supplier_result, "S_SUPPKEY"), *l_suppkey))
//      COGADB_FATAL_ERROR("", "");

//    // not supported anymore
//    //// --- FK/FK equality condition ---
//    // if(!code_gen->consumeJoinEqualityCondition(
//    //        AttributeReference(customer_result, "C_NATIONKEY"),
//    //        AttributeReference(supplier_result, "S_NATIONKEY")))
//    //    COGADB_FATAL_ERROR("", "");

//    if (!code_gen->consumeProbeHashTable(
//            AttributeReference(nation_result, "N_NATIONKEY"),
//            AttributeReference(supplier_result, "S_NATIONKEY")))
//      COGADB_FATAL_ERROR("", "");

//    if (!code_gen->consumeProbeHashTable(
//            AttributeReference(region_result, "R_REGIONKEY"),
//            AttributeReference(nation_result, "N_REGIONKEY")))
//      COGADB_FATAL_ERROR("", "");

//    std::pair<bool, AttributeReference> algebra_ret =
//        code_gen->consumeAlgebraComputation(*l_extendedprice, *l_discount,
//        MUL);
//    if (!algebra_ret.first) {
//      COGADB_FATAL_ERROR("", "");
//    }
//    AttributeReference extended_price_mul_discount = algebra_ret.second;
//    code_gen->addAttributeProjection(extended_price_mul_discount);
//    code_gen->addAttributeProjection(
//        AttributeReference(orders_result, "O_ORDERDATE"));

//    pipeline = code_gen->compile();
//    if (!execute(pipeline, true)) {
//      std::cerr << "Could not execute query successfully!" << std::endl;
//      return TablePtr();
//    }
//    lineitem_result = pipeline->getResult();
//  }
//  return true;
//}

const TablePtr qcrev_tpch9(CodeGeneratorType code_generator) {
  /*
-- TPC-H Query 9
select
   nation,
   o_year,
   sum(amount) as sum_profit
from
   (
      select
      n_name as nation,
      extract(year from o_orderdate) as o_year,
      l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
   from
      part,
      supplier,
      lineitem,
      partsupp,
      orders,
      nation
   where
      s_suppkey = l_suppkey
      and ps_suppkey = l_suppkey
      and ps_partkey = l_partkey
      and p_partkey = l_partkey
      and o_orderkey = l_orderkey
      and s_nationkey = n_nationkey
      and p_name like '%green%'
   ) as profit
group by
   nation,
   o_year
order by
   nation,
   o_year desc
  */
  Timestamp begin_total_time = getTimestamp();

  TablePtr result_nation;
  QueryContextPtr context_nation;
  {
    PipelinePtr pipeline;
    ProjectionParam param;

    AttributeReferencePtr n_nationkey = boost::make_shared<AttributeReference>(
        getTablebyName("NATION"), "NATION.N_NATIONKEY");
    AttributeReferencePtr n_nationname = boost::make_shared<AttributeReference>(
        getTablebyName("NATION"), "NATION.N_NAME");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("NATION"));
    context_nation = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

    code_gen->addAttributeProjection(*n_nationkey);
    code_gen->addAttributeProjection(*n_nationname);

    if (!code_gen->consumeBuildHashTable(*n_nationkey)) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }

    result_nation = execute(code_gen, context_nation);

    //    pipeline = code_gen->compile();
    //    if (!execute(pipeline, context_nation)) {
    //      std::cerr << "error: could not execute query" << std::endl;
    //      return TablePtr();
    //    }
    //    result_nation = pipeline->getResult();
    if (!result_nation->getHashTablebyName("NATION.N_NATIONKEY.1")) {
      std::cerr << "error: could not get hash table" << std::endl;
      return TablePtr();
    }

    //    std::cout << "1. Schema: ";
    //    result_nation->printSchema();
    //    std::cout << "1. Number of rows: " << result_nation->getNumberofRows()
    //              << std::endl;
    // std::cout << "1. Result: " << std::endl; result_nation->print();
  }

  TablePtr result_supplier;
  QueryContextPtr context_supplier;
  {
    PipelinePtr pipeline;
    ProjectionParam param;

    AttributeReferencePtr s_nationkey = boost::make_shared<AttributeReference>(
        getTablebyName("SUPPLIER"), "SUPPLIER.S_NATIONKEY");
    AttributeReferencePtr s_suppkey = boost::make_shared<AttributeReference>(
        getTablebyName("SUPPLIER"), "SUPPLIER.S_SUPPKEY");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("SUPPLIER"));
    context_supplier = createQueryContext(NORMAL_PIPELINE);
    // context_supplier->fetchInformationFromParentContext(context_nation);

    // code_gen->addAttributeProjection(*s_nationkey);
    code_gen->addAttributeProjection(*s_suppkey);
    AttributeReference n_name(result_nation, "NATION.N_NAME");
    code_gen->addAttributeProjection(n_name);

    if (!code_gen->consumeProbeHashTable(
            AttributeReference(result_nation, "NATION.N_NATIONKEY"),
            *s_nationkey)) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }

    if (!code_gen->consumeBuildHashTable(*s_suppkey)) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }

    result_supplier = execute(code_gen, context_supplier);
    //    pipeline = code_gen->compile();
    //    if (!execute(pipeline, context_supplier)) {
    //      std::cerr << "error: could not execute query" << std::endl;
    //      return TablePtr();
    //    }
    //    result_supplier = pipeline->getResult();
    //    context_supplier->updateStatistics(context_nation);
    storeResultTableAttributes(code_gen, context_supplier, result_supplier);

    //    std::cout << "2. Schema: ";
    //    result_supplier->printSchema();
    //    std::cout << "2. Number of rows: " <<
    //    result_supplier->getNumberofRows()
    //              << std::endl;
    // std::cout << "2. Result: " << std::endl; result_supplier->print();
  }

  TablePtr result_part;
  QueryContextPtr context_part;
  {
    PipelinePtr pipeline;
    ProjectionParam param;

    AttributeReferencePtr p_partkey = boost::make_shared<AttributeReference>(
        getTablebyName("PART"), "PART.P_PARTKEY");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("PART"));
    context_part = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

    code_gen->addAttributeProjection(*p_partkey);

    std::vector<PredicateExpressionPtr> conjunctions;
    conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
        p_partkey, boost::any(int(108783)), LESSER));
    PredicateExpressionPtr selection_expr =
        createPredicateExpression(conjunctions, LOGICAL_AND);

    if (!code_gen->consumeSelection(selection_expr)) {
      std::cerr << "error: could not consume selection" << std::endl;
      return TablePtr();
    }

    if (!code_gen->consumeBuildHashTable(*p_partkey)) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }

    result_part = execute(code_gen, context_part);

    //    pipeline = code_gen->compile();
    //    if (!execute(pipeline, context_part)) {
    //      std::cerr << "error: could not execute query" << std::endl;
    //      return TablePtr();
    //    }
    //    result_part = pipeline->getResult();
    if (!result_part->getHashTablebyName("PART.P_PARTKEY.1")) {
      std::cerr << "error: could not get hash table" << std::endl;
      return TablePtr();
    }

    //    std::cout << "3. Schema: ";
    //    result_part->printSchema();
    //    std::cout << "3. Number of rows: " << result_part->getNumberofRows()
    //              << std::endl;
    // std::cout << "3. Result: " << std::endl; result_part->print();
  }

  TablePtr result_partsupp;
  QueryContextPtr context_partsupp;
  {
    PipelinePtr pipeline;
    ProjectionParam param;

    AttributeReferencePtr ps_partkey = boost::make_shared<AttributeReference>(
        getTablebyName("PARTSUPP"), "PARTSUPP.PS_PARTKEY");
    AttributeReferencePtr ps_suppkey = boost::make_shared<AttributeReference>(
        getTablebyName("PARTSUPP"), "PARTSUPP.PS_SUPPKEY");
    AttributeReferencePtr ps_supplycost =
        boost::make_shared<AttributeReference>(getTablebyName("PARTSUPP"),
                                               "PARTSUPP.PS_SUPPLYCOST");
    AttributeReferencePtr s_suppkey = boost::make_shared<AttributeReference>(
        result_supplier, "SUPPLIER.S_SUPPKEY");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("PARTSUPP"));
    context_partsupp = createQueryContext(NORMAL_PIPELINE);
    // context_partsupp->fetchInformationFromParentContext(context_part);
    // context_partsupp->fetchInformationFromParentContext(context_supplier);

    // code_gen->addAttributeProjection(*ps_partkey);
    // code_gen->addAttributeProjection(*ps_suppkey);
    code_gen->addAttributeProjection(*ps_supplycost);
    code_gen->addAttributeProjection(*ps_partkey);
    code_gen->addAttributeProjection(*ps_suppkey);
    code_gen->addAttributeProjection(
        AttributeReference(result_supplier, "NATION.N_NAME"));
    code_gen->addAttributeProjection(*s_suppkey);

    if (!code_gen->consumeProbeHashTable(
            AttributeReference(result_part, "PART.P_PARTKEY"), *ps_partkey)) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }

    if (!code_gen->consumeProbeHashTable(*s_suppkey, *ps_suppkey)) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }

    if (!code_gen->consumeBuildHashTable(*ps_partkey)) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }

    /*if (!code_gen->consumeBuildHashTable(*ps_suppkey)) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }

    if (!code_gen->consumeBuildHashTable(*s_suppkey)) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }*/

    result_partsupp = execute(code_gen, context_partsupp);
    //    pipeline = code_gen->compile();
    //    if (!execute(pipeline, context_partsupp)) {
    //      std::cerr << "error: could not execute query" << std::endl;
    //      return TablePtr();
    //    }
    //    result_partsupp = pipeline->getResult();
    //    context_partsupp->updateStatistics(context_part);
    //    context_partsupp->updateStatistics(context_supplier);
    storeResultTableAttributes(code_gen, context_partsupp, result_partsupp);

    //    std::cout << "4. Schema: ";
    //    result_partsupp->printSchema();
    //    std::cout << "4. Number of rows: " <<
    //    result_partsupp->getNumberofRows()
    //              << std::endl;
    // std::cout << "4. Result: " << std::endl; result_partsupp->print();
  }

  TablePtr result_orders;
  QueryContextPtr context_orders;
  {
    PipelinePtr pipeline;
    ProjectionParam param;

    AttributeReferencePtr o_orderkey = boost::make_shared<AttributeReference>(
        getTablebyName("ORDERS"), "ORDERS.O_ORDERKEY");
    AttributeReferencePtr o_orderdate = boost::make_shared<AttributeReference>(
        getTablebyName("ORDERS"), "ORDERS.O_ORDERDATE");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("ORDERS"));
    context_orders = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

    code_gen->addAttributeProjection(*o_orderkey);
    code_gen->addAttributeProjection(*o_orderdate);

    if (!code_gen->consumeBuildHashTable(*o_orderkey)) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }

    result_orders = execute(code_gen, context_orders);
    //    pipeline = code_gen->compile();
    //    if (!execute(pipeline, context_orders)) {
    //      std::cerr << "error: could not execute query" << std::endl;
    //      return TablePtr();
    //    }
    //    result_orders = pipeline->getResult();
    if (!result_orders->getHashTablebyName("ORDERS.O_ORDERKEY.1")) {
      std::cerr << "error: could not get hash table" << std::endl;
      return TablePtr();
    }

    storeResultTableAttributes(code_gen, context_orders, result_orders);

    //    std::cout << "5. Schema: ";
    //    result_orders->printSchema();
    //    std::cout << "5. Number of rows: " << result_orders->getNumberofRows()
    //              << std::endl;
    // std::cout << "5. Result: " << std::endl; result_orders->print();
  }

  TablePtr result_lineitem;
  QueryContextPtr context_lineitem;
  {
    PipelinePtr pipeline;
    ProjectionParam param;

    AttributeReferencePtr l_partkey = boost::make_shared<AttributeReference>(
        getTablebyName("LINEITEM"), "LINEITEM.L_PARTKEY");
    AttributeReferencePtr l_suppkey = boost::make_shared<AttributeReference>(
        getTablebyName("LINEITEM"), "LINEITEM.L_SUPPKEY");
    AttributeReferencePtr l_orderkey = boost::make_shared<AttributeReference>(
        getTablebyName("LINEITEM"), "LINEITEM.L_ORDERKEY");

    AttributeReferencePtr l_discount = boost::make_shared<AttributeReference>(
        getTablebyName("LINEITEM"), "LINEITEM.L_DISCOUNT");
    AttributeReferencePtr l_quantity = boost::make_shared<AttributeReference>(
        getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY");
    AttributeReferencePtr l_extendedprice =
        boost::make_shared<AttributeReference>(getTablebyName("LINEITEM"),
                                               "LINEITEM.L_EXTENDEDPRICE");

    AttributeReferencePtr o_orderdate = boost::make_shared<AttributeReference>(
        result_orders, "ORDERS.O_ORDERDATE");
    AttributeReferencePtr n_name = boost::make_shared<AttributeReference>(
        result_partsupp, "NATION.N_NAME");
    AttributeReferencePtr ps_supplycost =
        boost::make_shared<AttributeReference>(result_partsupp,
                                               "PARTSUPP.PS_SUPPLYCOST");
    AttributeReferencePtr ps_partkey = boost::make_shared<AttributeReference>(
        result_partsupp, "PARTSUPP.PS_PARTKEY");
    AttributeReferencePtr ps_suppkey = boost::make_shared<AttributeReference>(
        result_partsupp, "PARTSUPP.PS_SUPPKEY");
    AttributeReferencePtr s_suppkey = boost::make_shared<AttributeReference>(
        result_partsupp, "SUPPLIER.S_SUPPKEY");

    /* dirty workaround: we know that the number of bits for the grouping key is
     * sufficient! */
    VariableManager::instance().setVariableValue(
        "code_gen.opt.hack.ignore_bitpacking_max_bits", "true");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));
    context_lineitem = createQueryContext(NORMAL_PIPELINE);
    //    context_lineitem->fetchInformationFromParentContext(context_orders);
    //    context_lineitem->fetchInformationFromParentContext(context_partsupp);

    code_gen->addToScannedAttributes(*o_orderdate);

    // code_gen->addAttributeProjection(*ps_partkey);
    // code_gen->addAttributeProjection(*ps_suppkey);
    // code_gen->addAttributeProjection(*o_orderdate);
    // code_gen->addAttributeProjection(*n_name);

    if (!code_gen->consumeProbeHashTable(*ps_partkey, *l_partkey)) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }

    /*if (!code_gen->consumeProbeHashTable(
            *ps_suppkey, *l_suppkey)) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }

    if (!code_gen->consumeProbeHashTable(
            *s_suppkey, *l_suppkey)) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }*/

    if (!code_gen->consumeProbeHashTable(
            AttributeReference(result_orders, "ORDERS.O_ORDERKEY"),
            *l_orderkey)) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }

    std::vector<PredicateExpressionPtr> conjunctions;
    conjunctions.push_back(createColumnColumnComparisonPredicateExpression(
        ps_suppkey, l_suppkey, EQUAL));
    conjunctions.push_back(createColumnColumnComparisonPredicateExpression(
        s_suppkey, l_suppkey, EQUAL));
    PredicateExpressionPtr selection_expr =
        createPredicateExpression(conjunctions, LOGICAL_AND);

    if (!code_gen->consumeSelection(selection_expr)) {
      std::cerr << "error: could not consume selection" << std::endl;
      return TablePtr();
    }

    std::pair<bool, AttributeReference> algebra_ret =
        code_gen->consumeAlgebraComputation(*ps_supplycost, *l_quantity, MUL);
    if (!algebra_ret.first) {
      std::cerr << "error: could not consume algebra computation" << std::endl;
      return TablePtr();
    }
    AttributeReference supplycost_quantity_mul = algebra_ret.second;

    algebra_ret = code_gen->consumeAlgebraComputation(boost::any(double(1.0)),
                                                      *l_discount, SUB);
    if (!algebra_ret.first) {
      std::cerr << "error: could not consume algebra computation" << std::endl;
      return TablePtr();
    }
    AttributeReference discount_sub = algebra_ret.second;

    algebra_ret = code_gen->consumeAlgebraComputation(*l_extendedprice,
                                                      discount_sub, MUL);
    if (!algebra_ret.first) {
      std::cerr << "error: could not consume algebra computation" << std::endl;
      return TablePtr();
    }
    AttributeReference extendedprice_discount_sub_mul = algebra_ret.second;

    algebra_ret = code_gen->consumeAlgebraComputation(
        extendedprice_discount_sub_mul, supplycost_quantity_mul, SUB);
    if (!algebra_ret.first) {
      std::cerr << "error: could not consume algebra computation" << std::endl;
      return TablePtr();
    }
    AttributeReference amount = algebra_ret.second;
    // name =
    // "LINEITEM.L_EXTENDEDPRICE1_MUL_1SUBLINEITEM.L_DISCOUNT1_SUB_PARTSUPP.PS_SUPPLYCOST1_MUL_LINEITEM.L_QUANTITY1"
    // code_gen->addAttributeProjection(amount);

    /*algebra_ret =
        code_gen->consumeAlgebraComputation(
            *o_orderdate, boost::any(double(10000.0)), DIV);
    if (!algebra_ret.first) {
      std::cerr << "error: could not consume algebra computation" << std::endl;
      return TablePtr();
    }
    AttributeReference o_year = algebra_ret.second;
    // name = "ORDERS.O_ORDERDATE1DIV10000"
    //code_gen->addAttributeProjection(o_year);*/

    std::vector<OutputAttributePtr> output_attrs;
    /* hier gibt man den Namen und TYPEN des berechneten Attributs an */
    OutputAttributePtr field =
        boost::make_shared<OutputAttribute>(UINT32, "o_year", "O_YEAR");
    // OutputAttributePtr field = boost::make_shared<OutputAttribute>(CHAR,
    // "o_year","O_YEAR");
    output_attrs.push_back(field);
    std::vector<std::string> code_lines;
    code_lines.push_back("#<OUT>.O_YEAR#=#ORDERS.O_ORDERDATE#/10000;");
    // code_lines.push_back("#<OUT>.O_YEAR#=#ORDERS.O_ORDERDATE#/10000-1900;");
    // // CHAR code
    UDF_CodePtr udf_code = parseUDFCode(code_lines, output_attrs, MAP_UDF);
    std::vector<StructFieldPtr> declared_variables;
    for (size_t i = 0; i < output_attrs.size(); ++i) {
      StructFieldPtr declared_var(new StructField(
          output_attrs.at(i)->field_type, output_attrs.at(i)->field_name, "0"));
      declared_variables.push_back(declared_var);
    }
    Map_UDFPtr map_udf(new Generic_Map_UDF(udf_code, declared_variables));
    Map_UDF_ParamPtr map_param(new Map_UDF_Param(map_udf));
    std::pair<bool, std::vector<AttributeReferencePtr> > computed_attrs;
    computed_attrs = code_gen->consumeMapUDF(map_param);
    AttributeReferencePtr o_year = computed_attrs.second.at(0);
    code_gen->addAttributeProjection(*o_year);
    // std::cout << "computed_attrs.second.size() = " <<
    // computed_attrs.second.size() << std::endl;

    ProcessorSpecification proc_spec(hype::PD0);
    GroupingAttributes grouping_columns;
    grouping_columns.push_back(*n_name);
    grouping_columns.push_back(*o_year);
    AggregateSpecifications agg_specs;
    agg_specs.push_back(
        createAggregateSpecification(amount, SUM, "SUM_AMOUNT.1"));
    GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);
    // std::cout << "groupby_param.aggregation_specs.size() = " <<
    // groupby_param.aggregation_specs.size() << std::endl;
    // std::cout << "groupby_param.aggregation_specs.max_size() = " <<
    // groupby_param.aggregation_specs.max_size() << std::endl;
    // std::cout << "groupby_param.aggregation_specs.capacity() = " <<
    // groupby_param.aggregation_specs.capacity() << std::endl;

    if (!code_gen->consumeHashGroupAggregate(groupby_param)) {
      std::cerr << "error: could not consume hash group aggregate" << std::endl;
      return TablePtr();
    }

    //    std::cout << "compile ..." << std::endl;
    //    pipeline = code_gen->compile();
    //    std::cout << "execute ..." << std::endl;
    //    if (!execute(pipeline, context_lineitem)) {
    //      std::cerr << "error: could not execute query" << std::endl;
    //      return TablePtr();
    //    }
    // result_lineitem = pipeline->getResult();
    result_lineitem = execute(code_gen, context_lineitem);

    //    {
    //    auto elapsed =
    //          (double)(getTimestamp() - begin_total_time) / (1000 * 1000 *
    //          1000);
    //    std::cout << "TOTAL TIME UNTIL LINEITEM: " << elapsed << "s" <<
    //    std::endl;
    //    VariantMeasurement vm = createVariantMeasurement(elapsed,
    //    context_lineitem);
    //    print(ClientPtr(new LocalClient()), vm);
    //    }

    //    context_lineitem->updateStatistics(context_orders);
    //    context_lineitem->updateStatistics(context_partsupp);
    storeResultTableAttributes(code_gen, context_lineitem, result_lineitem);

    // ORDER BY on CPU
    SortAttributeList sort_list;
    sort_list.push_back(SortAttribute("NATION.N_NAME.1", ASCENDING));
    sort_list.push_back(SortAttribute("O_YEAR", DESCENDING));
    Timestamp begin = getTimestamp();
    result_lineitem = BaseTable::sort(result_lineitem, sort_list);
    Timestamp end = getTimestamp();
    double sort_execution_time = double(end - begin) / (1000 * 1000 * 1000);
    context_lineitem->addExecutionTime(sort_execution_time);

    //    std::cout << "6. Schema: ";
    //    result_lineitem->printSchema();
    //    std::cout << "6. Number of rows: " <<
    //    result_lineitem->getNumberofRows()
    //              << std::endl;
    //    std::cout << "6. Result: " << std::endl; result_lineitem->print();

    VariableManager::instance().setVariableValue(
        "code_gen.opt.hack.ignore_bitpacking_max_bits", "false");
  }
  Timestamp end_total_time = getTimestamp();
  auto elapsed =
      (double)(end_total_time - begin_total_time) / (1000 * 1000 * 1000);

  std::cout << "LINEITEM: " << std::endl;
  print(ClientPtr(new LocalClient()),
        createVariantMeasurement(elapsed, context_lineitem));
  std::cout << "ORDERS: " << std::endl;
  print(ClientPtr(new LocalClient()),
        createVariantMeasurement(elapsed, context_orders));

  std::cout << "PARTSUPP: " << std::endl;
  print(ClientPtr(new LocalClient()),
        createVariantMeasurement(elapsed, context_partsupp));
  std::cout << "PART: " << std::endl;
  print(ClientPtr(new LocalClient()),
        createVariantMeasurement(elapsed, context_part));
  std::cout << "SUPPLIER: " << std::endl;
  print(ClientPtr(new LocalClient()),
        createVariantMeasurement(elapsed, context_supplier));
  std::cout << "NATION: " << std::endl;
  print(ClientPtr(new LocalClient()),
        createVariantMeasurement(elapsed, context_nation));

  context_lineitem->updateStatistics(context_orders);
  context_lineitem->updateStatistics(context_partsupp);
  context_lineitem->updateStatistics(context_part);
  context_lineitem->updateStatistics(context_supplier);
  context_lineitem->updateStatistics(context_nation);

  result_lineitem->print();
  std::cout << "TOTAL TIME: " << elapsed << "s" << std::endl;
  VariantMeasurement vm = createVariantMeasurement(elapsed, context_lineitem);
  print(ClientPtr(new LocalClient()), vm);

  return result_lineitem;
}

const TablePtr qcrev_tpch13(CodeGeneratorType code_generator) {
  /*-- TPC-H Query 13

  select
          c_count,
          count(*) as custdist
  from
          (
                  select
                          c_custkey,
                          count(o_orderkey) c_count
                  from
                          customer left outer join orders on
                                  c_custkey = o_custkey
                                  and o_comment not like '%special%requests%'
                  group by
                          c_custkey
          ) as c_orders
  group by
          c_count
  order by
          custdist desc,
          c_count desc
   */

  /*
  -- Our version of TPC-H Query 13

  select
          c_count,
          count(*) as custdist
  from
          (
                  select
                          c_custkey,
                          count(o_orderkey) c_count
                  from
                          customer join orders on
                                  c_custkey = o_custkey
                  group by
                          c_custkey
          ) as c_orders
  group by
          c_count
  order by
          custdist desc,
          c_count desc
  */

  Timestamp begin_total_time = getTimestamp();
  TablePtr result_customer;
  QueryContextPtr context_customer;
  {
    PipelinePtr pipeline;
    ProjectionParam param;

    AttributeReferencePtr c_custkey = boost::make_shared<AttributeReference>(
        getTablebyName("CUSTOMER"), "CUSTOMER.C_CUSTKEY");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("CUSTOMER"));
    context_customer = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

    code_gen->addAttributeProjection(*c_custkey);
    // code_gen->addToScannedAttributes(*c_custkey);

    if (!code_gen->consumeBuildHashTable(*c_custkey)) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }
    result_customer = execute(code_gen, context_customer);
    //    pipeline = code_gen->compile();
    //    if (!execute(pipeline, context_customer)) {
    //      std::cerr << "error: could not execute query" << std::endl;
    //      return TablePtr();
    //    }
    //    result_customer = pipeline->getResult();
    if (!result_customer->getHashTablebyName("CUSTOMER.C_CUSTKEY.1")) {
      std::cerr << "error: could not get hash table" << std::endl;
      return TablePtr();
    }

    //    std::cout << "1. Schema: ";
    //    result_customer->printSchema();
    //    std::cout << "1. Number of rows: " <<
    //    result_customer->getNumberofRows()
    //              << std::endl;
    // std::cout << "1. Result: " << std::endl; result_customer->print();
  }

  TablePtr result_orders;
  QueryContextPtr context_orders;
  {
    PipelinePtr pipeline;
    ProjectionParam param;

    AttributeReferencePtr o_custkey = boost::make_shared<AttributeReference>(
        getTablebyName("ORDERS"), "ORDERS.O_CUSTKEY");
    AttributeReferencePtr o_orderkey = boost::make_shared<AttributeReference>(
        getTablebyName("ORDERS"), "ORDERS.O_ORDERKEY");
    AttributeReferencePtr c_custkey = boost::make_shared<AttributeReference>(
        result_customer, "CUSTOMER.C_CUSTKEY");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("ORDERS"));
    context_orders = createQueryContext(NORMAL_PIPELINE);
    context_orders->fetchInformationFromParentContext(context_customer);

    // code_gen->addToScannedAttributes(*c_custkey);

    if (!code_gen->consumeProbeHashTable(*c_custkey, *o_custkey)) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }

    ProcessorSpecification proc_spec(hype::PD0);
    GroupingAttributes grouping_columns;
    grouping_columns.push_back(
        AttributeReference(result_customer, "CUSTOMER.C_CUSTKEY"));
    AggregateSpecifications agg_specs;
    agg_specs.push_back(
        createAggregateSpecification(*o_orderkey, COUNT, "C_COUNT.1"));
    GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);

    if (!code_gen->consumeHashGroupAggregate(groupby_param)) {
      std::cerr << "error: could not consume hash group aggregate" << std::endl;
      return TablePtr();
    }

    result_orders = execute(code_gen, context_orders);
    //    pipeline = code_gen->compile();
    //    if (!execute(pipeline, context_orders)) {
    //      std::cerr << "error: could not execute query" << std::endl;
    //      return TablePtr();
    //    }
    //    result_orders = pipeline->getResult();
    context_orders->updateStatistics(context_customer);
    storeResultTableAttributes(code_gen, context_orders, result_orders);

    std::cout << "2. Schema: ";
    result_orders->printSchema();
    // std::cout <<
    // CoGaDB::util::getName(result_orders->getColumnbyId(1)->getType()) <<
    // std::endl;
    std::cout << "2. Number of rows: " << result_orders->getNumberofRows()
              << std::endl;
    // std::cout << "2. Result: " << std::endl; result_orders->print();
  }

  TablePtr result;
  QueryContextPtr context;
  {
    PipelinePtr pipeline;
    ProjectionParam param;

    AttributeReferencePtr c_custkey = boost::make_shared<AttributeReference>(
        result_orders, "CUSTOMER.C_CUSTKEY");
    AttributeReferencePtr c_count =
        boost::make_shared<AttributeReference>(result_orders, "C_COUNT");
    // std::cout << CoGaDB::toString(*c_count) << "->" <<
    // c_count->getResultAttributeName() << std::endl;

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, result_orders);
    context = createQueryContext(NORMAL_PIPELINE);
    context->fetchInformationFromParentContext(context_orders);

    // code_gen->addToScannedAttributes(*c_custkey);
    // code_gen->addToScannedAttributes(*c_count);

    ProcessorSpecification proc_spec(hype::PD0);
    GroupingAttributes grouping_columns;
    grouping_columns.push_back(*c_count);
    AggregateSpecifications agg_specs;
    agg_specs.push_back(
        createAggregateSpecification(*c_custkey, COUNT, "CUSTDIST"));
    GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);

    if (!code_gen->consumeHashGroupAggregate(groupby_param)) {
      std::cerr << "error: could not consume hash group aggregate" << std::endl;
      return TablePtr();
    }

    result = execute(code_gen, context);
    //    pipeline = code_gen->compile();
    //    if (!execute(pipeline, context)) {
    //      std::cerr << "error: could not execute query" << std::endl;
    //      return TablePtr();
    //    }
    //    result = pipeline->getResult();
    context->updateStatistics(context_orders);
    storeResultTableAttributes(code_gen, context, result);

    // result->print();

    // ORDER BY on CPU
    SortAttributeList sort_list;
    sort_list.push_back(SortAttribute("CUSTDIST.1", DESCENDING));
    // sort_list.push_back(SortAttribute("1", DESCENDING));
    Timestamp begin = getTimestamp();
    result = BaseTable::sort(result, sort_list);
    Timestamp end = getTimestamp();
    double sort_execution_time = double(end - begin) / (1000 * 1000 * 1000);
    context->addExecutionTime(sort_execution_time);

    //    std::cout << "3. Schema: ";
    //    result->printSchema();
    //    std::cout << "3. Number of rows: " << result->getNumberofRows()
    //              << std::endl;
    //    std::cout << "3. Result: " << std::endl; result->print();
  }

  Timestamp end_total_time = getTimestamp();
  auto elapsed =
      (double)(end_total_time - begin_total_time) / (1000 * 1000 * 1000);

  //  std::cout << "LINEITEM: " << std::endl;
  //  print(ClientPtr(new LocalClient()), createVariantMeasurement(elapsed,
  //  context_lineitem));
  //  std::cout << "ORDERS: " << std::endl;
  //  print(ClientPtr(new LocalClient()), createVariantMeasurement(elapsed,
  //  context_orders));

  //  std::cout << "PARTSUPP: " << std::endl;
  //  print(ClientPtr(new LocalClient()), createVariantMeasurement(elapsed,
  //  context_partsupp));
  //  std::cout << "PART: " << std::endl;
  //  print(ClientPtr(new LocalClient()), createVariantMeasurement(elapsed,
  //  context_part));
  //  std::cout << "SUPPLIER: " << std::endl;
  //  print(ClientPtr(new LocalClient()), createVariantMeasurement(elapsed,
  //  context_supplier));
  //  std::cout << "NATION: " << std::endl;
  //  print(ClientPtr(new LocalClient()), createVariantMeasurement(elapsed,
  //  context_nation));

  //  context_lineitem->updateStatistics(context_orders);
  //  context_lineitem->updateStatistics(context_partsupp);
  //  context_lineitem->updateStatistics(context_part);
  //  context_lineitem->updateStatistics(context_supplier);
  //  context_lineitem->updateStatistics(context_nation);

  result->print();
  std::cout << "TOTAL TIME: " << elapsed << "s" << std::endl;
  VariantMeasurement vm = createVariantMeasurement(elapsed, context);
  print(ClientPtr(new LocalClient()), vm);

  return result;
}

const TablePtr qcrev_tpch17(CodeGeneratorType code_generator) {
  /*-- TPC-H Query 17

  select
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
        )
  */

  /* set sufficient size for aggregation hash table */
  uint64_t num_groups =
      uint64_t(getTablebyName("PART")->getNumberofRows() * 0.002);
  std::cout << "Num Groups: " << num_groups << std::endl;

  /* save parameters and set new number of groups */
  HashAggregationOptimizationState hash_aggr_state =
      setSizeOfHashAggregationTable(num_groups);

  Timestamp begin_total_time = getTimestamp();
  TablePtr result1;
  QueryContextPtr context1;
  {
    ProjectionParam param;

    AttributeReferencePtr p_partkey = boost::make_shared<AttributeReference>(
        getTablebyName("PART"), "PART.P_PARTKEY");
    AttributeReferencePtr p_brand = boost::make_shared<AttributeReference>(
        getTablebyName("PART"), "PART.P_BRAND");
    AttributeReferencePtr p_container = boost::make_shared<AttributeReference>(
        getTablebyName("PART"), "PART.P_CONTAINER");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("PART"));
    context1 = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

    code_gen->addAttributeProjection(*p_partkey);

    std::vector<PredicateExpressionPtr> conjunctions;
    conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
        p_brand, boost::any(std::string("Brand#23")), EQUAL));
    conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
        p_container, boost::any(std::string("MED BOX")), EQUAL));
    PredicateExpressionPtr selection_expr =
        createPredicateExpression(conjunctions, LOGICAL_AND);

    if (!code_gen->consumeSelection(selection_expr)) {
      std::cerr << "error: could not consume selection" << std::endl;
      return TablePtr();
    }

    if (!code_gen->consumeBuildHashTable(*p_partkey)) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }

    result1 = execute(code_gen, context1);

    std::cout << "Pipeline Execution Time: " << context1->getExecutionTimeSec()
              << "s" << std::endl;

    if (!result1->getHashTablebyName("PART.P_PARTKEY.1")) {
      std::cerr << "error: could not get hash table" << std::endl;
      return TablePtr();
    }
  }

  TablePtr result2;
  QueryContextPtr context2;
  {
    ProjectionParam param;

    AttributeReferencePtr l_partkey = boost::make_shared<AttributeReference>(
        getTablebyName("LINEITEM"), "LINEITEM.L_PARTKEY");
    AttributeReferencePtr l_quantity = boost::make_shared<AttributeReference>(
        getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY");

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));
    context2 = createQueryContext(NORMAL_PIPELINE);
    context2->fetchInformationFromParentContext(context1);

    code_gen->addAttributeProjection(
        AttributeReference(result1, "PART.P_PARTKEY"));

    if (!code_gen->consumeProbeHashTable(
            AttributeReference(result1, "PART.P_PARTKEY"), *l_partkey)) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }

    ProcessorSpecification proc_spec(hype::PD0);
    GroupingAttributes grouping_columns;
    grouping_columns.push_back(AttributeReference(result1, "PART.P_PARTKEY"));
    AggregateSpecifications agg_specs;
    agg_specs.push_back(
        createAggregateSpecification(*l_quantity, SUM, "SUM_QUANTITY.1"));
    GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);

    if (!code_gen->consumeHashGroupAggregate(groupby_param)) {
      std::cerr << "error: could not consume hash group aggregate" << std::endl;
      return TablePtr();
    }

    result2 = execute(code_gen, context2);

    std::cout << "Pipeline Execution Time: " << context2->getExecutionTimeSec()
              << "s" << std::endl;

    context2->updateStatistics(context1);
    storeResultTableAttributes(code_gen, context2, result2);
  }

  TablePtr result3;
  QueryContextPtr context3;
  {
    ProjectionParam param;

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, result2);
    context3 = createQueryContext(BUILD_HASH_TABLE_PIPELINE);
    context3->fetchInformationFromParentContext(context2);

    std::pair<bool, AttributeReference> algebra_ret =
        code_gen->consumeAlgebraComputation(
            AttributeReference(result2, "SUM_QUANTITY"),
            boost::any(double(0.2)), MUL);
    if (!algebra_ret.first) {
      std::cerr << "error: could not consume algebra computation" << std::endl;
      return TablePtr();
    }

    AttributeReference quantity_mul = algebra_ret.second;
    code_gen->addAttributeProjection(quantity_mul);
    code_gen->addAttributeProjection(
        AttributeReference(result2, "PART.P_PARTKEY"));

    if (!code_gen->consumeBuildHashTable(
            AttributeReference(result2, "PART.P_PARTKEY"))) {
      std::cerr << "error: could not consume build hash table" << std::endl;
      return TablePtr();
    }

    result3 = execute(code_gen, context3);

    if (!result3->getHashTablebyName("PART.P_PARTKEY.1")) {
      std::cerr << "error: could not get hash table" << std::endl;
      return TablePtr();
    }
    std::cout << "Pipeline Execution Time: " << context3->getExecutionTimeSec()
              << "s" << std::endl;
    context3->updateStatistics(context2);
    // storeResultTableAttributes(code_gen, context3, result3);
  }

  TablePtr result4;
  QueryContextPtr context4;
  {
    ProjectionParam param;

    uint32_t version = 1;
    CodeGeneratorPtr code_gen = createCodeGenerator(
        code_generator, param, getTablebyName("LINEITEM"), boost::any(), 2);
    context4 = createQueryContext(NORMAL_PIPELINE);

    /* workaround, so the function storeResultTableAttributes does
     * not add the computed attribute to the code generator*/
    context4->addColumnToProjectionList("PART.P_PARTKEY.1");
    storeResultTableAttributes(code_gen, context4, result3);

    context4->fetchInformationFromParentContext(context3);

    code_gen->addAttributeProjection(AttributeReference(
        getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY", "", 2));
    code_gen->addAttributeProjection(AttributeReference(
        getTablebyName("LINEITEM"), "LINEITEM.L_EXTENDEDPRICE", "", 2));
    context4->addColumnToProjectionList("LINEITEM.L_EXTENDEDPRICE.2");
    context4->addColumnToProjectionList("PART.P_PARTKEY.1");

    AttributeReferencePtr computed_attr(
        new AttributeReference(result3,
                               "SUM_QUANTITY1MUL0.2"));  // =
    // context4->getAttributeFromOtherPipelineByName("SUM_QUANTITY.1");
    AttributeReferencePtr build_attr =
        context4->getAttributeFromOtherPipelineByName("PART.P_PARTKEY.1");
    code_gen->addAttributeProjection(*computed_attr);

    /* do not add AttributeReferences after this call! */
    retrieveScannedAndProjectedAttributesFromScannedTable(code_gen, context4,
                                                          result3, version);
    if (!code_gen->consumeProbeHashTable(
            *build_attr,  // AttributeReference(result3, "PART.P_PARTKEY"),
            AttributeReference(getTablebyName("LINEITEM"), "LINEITEM.L_PARTKEY",
                               "LINEITEM.L_PARTKEY", 2))) {
      std::cerr << "error: could not consume probe hash table" << std::endl;
      return TablePtr();
    }

    AttributeReferencePtr l_quantity = boost::make_shared<AttributeReference>(
        getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY", "", 2);
    AttributeReferencePtr l_quantity_mul =
        boost::make_shared<AttributeReference>(result3, "SUM_QUANTITY1MUL0.2");
    std::vector<PredicateExpressionPtr> conjunctions;
    conjunctions.push_back(createColumnColumnComparisonPredicateExpression(
        l_quantity, l_quantity_mul, LESSER));
    PredicateExpressionPtr selection_expr =
        createPredicateExpression(conjunctions, LOGICAL_AND);

    if (!code_gen->consumeSelection(selection_expr)) {
      std::cerr << "error: could not consume selection" << std::endl;
      return TablePtr();
    }

    AggregateSpecifications agg_specs;
    agg_specs.push_back(createAggregateSpecification(
        AttributeReference(getTablebyName("LINEITEM"),
                           "LINEITEM.L_EXTENDEDPRICE", "", 2),
        SUM, "SUM_EXTENDEDPRICE"));

    if (!code_gen->consumeAggregate(agg_specs)) {
      std::cerr << "error: could not consume hash group aggregate" << std::endl;
      return TablePtr();
    }
    result4 = execute(code_gen, context4);

    std::cout << "Pipeline Execution Time: " << context4->getExecutionTimeSec()
              << "s" << std::endl;
    context4->updateStatistics(context3);
  }

  Timestamp end_total_time = getTimestamp();
  auto elapsed =
      (double)(end_total_time - begin_total_time) / (1000 * 1000 * 1000);
  result4->print();
  std::cout << "TOTAL TIME: " << elapsed << "s" << std::endl;
  VariantMeasurement vm = createVariantMeasurement(elapsed, context4);
  print(ClientPtr(new LocalClient()), vm);

  // not working: "FATAL ERROR: Column SUM_EXTENDEDPRICE.1 not found  in table
  // LINEITEM!"
  /*TablePtr result5;
  QueryContextPtr context5;
  {
      PipelinePtr pipeline;
      ProjectionParam param;

      CodeGeneratorPtr code_gen = createCodeGenerator(code_generator,
              param, result4);
      context5 = createQueryContext(NORMAL_PIPELINE);
      context5->fetchInformationFromParentContext(context4);

      std::pair<bool, AttributeReference> algebra_ret =
          code_gen->consumeAlgebraComputation(
              AttributeReference(result4, "SUM_EXTENDEDPRICE"),
              boost::any(double(7.0)), DIV);
      if (!algebra_ret.first) {
          std::cerr << "error: could not consume algebra computation" <<
  std::endl;
          return TablePtr();
      }

      AttributeReference avg_yearly = algebra_ret.second;
      code_gen->addAttributeProjection(avg_yearly);

      pipeline = code_gen->compile();
      if (!execute(pipeline, context5)) {
          std::cerr << "error: could not execute query" << std::endl;
          return TablePtr();
      }
      result5 = pipeline->getResult();
      context5->updateStatistics(context4);

      std::cout << "5. Schema: "; result5->printSchema();
      std::cout << "5. Number of rows: " << result5->getNumberofRows() <<
  std::endl;
      std::cout << "5. Result: " << std::endl; result5->print();
  }*/

  /* reset previously overridden parameters */
  resetSizeOfHashAggregationTable(hash_aggr_state);

  return result4;
}

const TablePtr qcrev_tpch18(CodeGeneratorType code_generator) {
  return nullptr;
  //  ProcessorSpecification proc_spec(hype::PD0);

  //  /* set sufficient size for aggregation hash table */
  //  uint64_t num_groups =
  //  uint64_t(getTablebyName("ORDERS")->getNumberofRows()*1.2);
  //  std::cout << "Num Groups: " << num_groups << std::endl;

  //  /* save parameters and set new number of groups */
  //  HashAggregationOptimizationState hash_aggr_state =
  //  setSizeOfHashAggregationTable(num_groups);

  //  Timestamp begin_total_time = getTimestamp();
  //  TablePtr result1;
  //  QueryContextPtr context1;
  //  {
  //    ProjectionParam param;
  //    PipelinePtr pipeline;

  //    context1 = createQueryContext(NORMAL_PIPELINE);
  //    CodeGeneratorPtr code_gen =
  //        createCodeGenerator(code_generator, param,
  //        getTablebyName("LINEITEM"));

  //    AttributeReferencePtr l_orderkey =
  //    boost::make_shared<AttributeReference>(
  //        getTablebyName("LINEITEM"), "LINEITEM.L_ORDERKEY",
  //        "LINEITEM.L_ORDERKEY.1");
  //    code_gen->addAttributeProjection(*l_orderkey);

  //    AttributeReferencePtr l_quantity =
  //    boost::make_shared<AttributeReference>(
  //        getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY",
  //        "LINEITEM.L_QUANTITY.1");

  //    GroupingAttributes grouping_columns;
  //    grouping_columns.push_back(*l_orderkey);

  //    AggregateSpecifications agg_specs;
  //    agg_specs.push_back(
  //        createAggregateSpecification(*l_quantity, SUM, "SUM.1"));

  //    GroupByAggregateParam groupby_param(proc_spec, grouping_columns,
  //    agg_specs);
  //    if (!code_gen->consumeHashGroupAggregate(groupby_param)) {
  //      std::cerr << "error: could not consume hash group aggregate" <<
  //      std::endl;
  //      return TablePtr();
  //    }

  ////
  /// VariableManager::instance().setVariableValue("code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_size","true");
  ////
  /// VariableManager::instance().setVariableValue("code_gen.opt.ocl_grouped_aggregation.hack.ht_size",boost::lexical_cast<std::string>(num_groups));

  //    result1 = execute(code_gen, context1);

  //    storeResultTableAttributes(code_gen, context1, result1);

  //    std::cout << "Pipeline Execution Time: " <<
  //    context1->getExecutionTimeSec() << "s" << std::endl;

  //  }

  //  TablePtr result2;
  //  QueryContextPtr context2;
  //  {
  //    ProjectionParam param;
  //    PipelinePtr pipeline;

  //    CodeGeneratorPtr code_gen =
  //        createCodeGenerator(code_generator, param, result1);
  //    context2 = createQueryContext(BUILD_HASH_TABLE_PIPELINE);
  //    context2->fetchInformationFromParentContext(context1);

  //    AttributeReferencePtr sum_quantity(new AttributeReference(result1,
  //    "SUM"));
  //    code_gen->addAttributeProjection(*sum_quantity);
  //    code_gen->addAttributeProjection(
  //        AttributeReference(result1, "LINEITEM.L_ORDERKEY"));

  //    std::vector<PredicateExpressionPtr> conjunctions;
  //    conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
  //        sum_quantity, boost::any(uint32_t(300)), GREATER));
  //    PredicateExpressionPtr selection_expr =
  //        createPredicateExpression(conjunctions, LOGICAL_AND);

  //    if (!code_gen->consumeSelection(selection_expr)) {
  //      std::cerr << "error: could not consume selection" << std::endl;
  //      return TablePtr();
  //    }

  //    if (!code_gen->consumeBuildHashTable(
  //            AttributeReference(result1, "LINEITEM.L_ORDERKEY"),
  //            LEFT_SEMI_JOIN)) {
  //      std::cerr << "error: could not consume build hash table" << std::endl;
  //      return TablePtr();
  //    }

  //    result2 = execute(code_gen, context2);

  //    std::cout << "Pipeline Execution Time: " <<
  //    context2->getExecutionTimeSec() << "s" << std::endl;

  //    if (!result2->getHashTablebyName("LINEITEM.L_ORDERKEY.1")) {
  //      std::cerr << "error: could not get hash table" << std::endl;
  //      return TablePtr();
  //    }

  //    context2->updateStatistics(context1);

  //  }

  //  TablePtr result3;
  //  QueryContextPtr context3;
  //  {
  //    ProjectionParam param;
  //    PipelinePtr pipeline;

  //    CodeGeneratorPtr code_gen =
  //        createCodeGenerator(code_generator, param,
  //        getTablebyName("CUSTOMER"));
  //    context3 = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

  //    AttributeReferencePtr c_custkey =
  //    boost::make_shared<AttributeReference>(
  //        getTablebyName("CUSTOMER"), "CUSTOMER.C_CUSTKEY",
  //        "CUSTOMER.C_CUSTKEY.1");
  //    code_gen->addAttributeProjection(*c_custkey);
  //    AttributeReferencePtr c_name = boost::make_shared<AttributeReference>(
  //        getTablebyName("CUSTOMER"), "CUSTOMER.C_NAME", "CUSTOMER.C_NAME.1");
  //    code_gen->addAttributeProjection(*c_name);

  //    if (!code_gen->consumeBuildHashTable(*c_custkey)) {
  //      std::cerr << "error: could not consume build hash table" << std::endl;
  //      return TablePtr();
  //    }

  //    result3 = execute(code_gen, context3);

  //    std::cout << "Pipeline Execution Time: " <<
  //    context3->getExecutionTimeSec() << "s" << std::endl;

  //    if (!result3->getHashTablebyName("CUSTOMER.C_CUSTKEY.1")) {
  //      std::cerr << "error: could not get hash table" << std::endl;
  //      return TablePtr();
  //    }

  //    context3->updateStatistics(context2);

  //  }

  //  TablePtr result4;
  //  QueryContextPtr context4;
  //  {
  //    ProjectionParam param;
  //    PipelinePtr pipeline;

  //    CodeGeneratorPtr code_gen =
  //        createCodeGenerator(code_generator, param,
  //        getTablebyName("ORDERS"));
  //    context4 = createQueryContext(NORMAL_PIPELINE);

  //    code_gen->addAttributeProjection(
  //        AttributeReference(result3, "CUSTOMER.C_NAME"));
  //    code_gen->addAttributeProjection(
  //        AttributeReference(result3, "CUSTOMER.C_CUSTKEY"));

  //    AttributeReferencePtr o_orderkey =
  //    boost::make_shared<AttributeReference>(
  //        getTablebyName("ORDERS"), "ORDERS.O_ORDERKEY",
  //        "ORDERS.O_ORDERKEY.1");
  //    code_gen->addAttributeProjection(*o_orderkey);

  //    AttributeReferencePtr o_orderdate =
  //    boost::make_shared<AttributeReference>(
  //        getTablebyName("ORDERS"), "ORDERS.O_ORDERDATE",
  //        "ORDERS.O_ORDERDATE.1");
  //    code_gen->addAttributeProjection(*o_orderdate);

  //    AttributeReferencePtr o_totalprice =
  //    boost::make_shared<AttributeReference>(
  //        getTablebyName("ORDERS"), "ORDERS.O_TOTALPRICE",
  //        "ORDERS.O_TOTALPRICE.1");
  //    code_gen->addAttributeProjection(*o_totalprice);

  //    code_gen->addAttributeProjection(AttributeReference(result2, "SUM"));

  //    AttributeReferencePtr o_custkey =
  //    boost::make_shared<AttributeReference>(
  //        getTablebyName("ORDERS"), "ORDERS.O_CUSTKEY", "ORDERS.O_CUSTKEY.1");

  //    if (!code_gen->consumeProbeHashTable(
  //            AttributeReference(result2, "LINEITEM.L_ORDERKEY"), *o_orderkey,
  //            LEFT_SEMI_JOIN)) {
  //      std::cerr << "error: could not consume probe hash table" << std::endl;
  //      return TablePtr();
  //    }

  //    if (!code_gen->consumeProbeHashTable(
  //            AttributeReference(result3, "CUSTOMER.C_CUSTKEY"), *o_custkey))
  //            {
  //      std::cerr << "error: could not consume probe hash table" << std::endl;
  //      return TablePtr();
  //    }
  //    /* set sufficient size for aggregation hash table */
  //    VariableManager::instance().setVariableValue("code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_size","true");
  //    VariableManager::instance().setVariableValue("code_gen.opt.ocl_grouped_aggregation.hack.ht_size","300");

  //    result4 = execute(code_gen, context4);
  //    std::cout << "Pipeline Execution Time: " <<
  //    context4->getExecutionTimeSec() << "s" << std::endl;

  //    context4->updateStatistics(context3);

  //    SortAttributeList sort_list;
  //    sort_list.push_back(SortAttribute("ORDERS.O_TOTALPRICE.1", DESCENDING));
  //    sort_list.push_back(SortAttribute("ORDERS.O_ORDERDATE.1", ASCENDING));

  //    std::vector<boost::any> limit_param;
  //    limit_param.push_back(boost::any(size_t(100)));

  //    Timestamp begin = getTimestamp();
  //    result4 = BaseTable::sort(result4, sort_list);
  //    result4 = limit(result4, "LIMIT", limit_param, proc_spec);
  //    Timestamp end = getTimestamp();
  //    double sort_execution_time = double(end - begin) / (1000 * 1000 * 1000);
  //    context4->addExecutionTime(sort_execution_time);
  //  }

  //  Timestamp end_total_time = getTimestamp();
  //  auto elapsed =
  //      (double)(end_total_time - begin_total_time) / (1000 * 1000 * 1000);
  //  result4->print();
  //  std::cout << "TOTAL TIME: " << elapsed << "s" << std::endl;
  //  VariantMeasurement vm = createVariantMeasurement(elapsed, context4);
  //  print(ClientPtr(new LocalClient()), vm);

  //  /* reset previously overridden parameters */
  //  resetSizeOfHashAggregationTable(hash_aggr_state);

  //  return result4;
}

const TablePtr qcrev_tpch19(CodeGeneratorType code_generator) {
  // json available
  return TablePtr();
}

const TablePtr qcrev_tpch21(CodeGeneratorType code_generator) {
  // json available
  return TablePtr();
}
