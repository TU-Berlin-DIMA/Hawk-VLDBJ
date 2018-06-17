
#include <fstream>
#include <iostream>

#include <dlfcn.h>
#include <stdlib.h>

#include <core/global_definitions.hpp>

#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generators/cpp_code_generator.hpp>

#include <core/selection_expression.hpp>
#include <parser/commandline_interpreter.hpp>
#include <util/time_measurement.hpp>

#include <boost/make_shared.hpp>
#include <iomanip>
#include <util/getname.hpp>

#include <boost/program_options.hpp>

#include <util/tests.hpp>

#include "core/variable_manager.hpp"
#include "query_compilation/code_generators/code_generator_utils.hpp"
#include "query_compilation/query_context.hpp"

using namespace CoGaDB;

std::string directory_containing_reference_results =
    PATH_TO_DATA_OF_TESTS "/unittests/code_generators";

typedef bool (*TestFunctionPtr)(CodeGeneratorType code_generator,
                                const std::string& file_name_reference_result,
                                bool write_new_reference_result);
typedef std::map<std::string, TestFunctionPtr> Tests;

bool repeatedPipelineExecution(PipelinePtr pipeline,
                               const std::string& file_name_reference_result,
                               bool write_new_reference_result,
                               size_t num_of_repetitions = 1) {
  assert(pipeline != NULL);

  TablePtr result;
  for (size_t i = 0; i < num_of_repetitions; ++i) {
    Timestamp begin_execute = getTimestamp();
    bool ret = pipeline->execute();
    if (!ret) {
      std::cout << "Execution of Pipeline failed!" << std::endl;
      return false;
    }
    //        assert(ret == true);
    result = pipeline->getResult();
    Timestamp end_execute = getTimestamp();

    if (!result) {
      std::cerr << "Could not execute query successfully!" << std::endl;
      return false;
    } else {
      if (i == 0) {
        result->print();
        /* \todo read previously stored table and compare with current result
         * (write_new_reference_result==false),
         * or store this result as new reference result
         * (write_new_reference_result==true)
         */
        if (write_new_reference_result) {
          std::cout << "Write result file " << file_name_reference_result
                    << ".csv in directory '"
                    << directory_containing_reference_results << "'"
                    << std::endl;
          if (!storeTableAsSelfContainedCSV(
                  result, directory_containing_reference_results,
                  file_name_reference_result + ".csv")) {
            COGADB_FATAL_ERROR("Could not store reference result!", "");
          }
        } else {
          TablePtr reference_table = loadTableFromSelfContainedCSV(
              directory_containing_reference_results + std::string("/") +
              file_name_reference_result + ".csv");
          if (!reference_table) {
            COGADB_FATAL_ERROR(
                "Could not load reference result for "
                "test '"
                    << file_name_reference_result << "'! "
                    << "Check whether a file "
                    << file_name_reference_result + ".csv' exists in "
                    << "directory '" << directory_containing_reference_results
                    << "' and that this file is a valid CoGaDB self-"
                    << "contained CSV file!",
                "");
          }
          /* store result in self contained CSV format and then reload it.
           * This mechanims avoids comparison problems with floating point
           * columns. */
          if (!storeTableAsSelfContainedCSV(result, ".", "tmp_unittest.csv")) {
            COGADB_FATAL_ERROR("Could not store reference result!", "");
          }
          TablePtr reloaded_result =
              loadTableFromSelfContainedCSV("tmp_unittest.csv");
          assert(reloaded_result != NULL);
          boost::filesystem::remove("tmp_unittest.csv");

          if (!BaseTable::equals(reloaded_result, reference_table)) {
            std::cout << "Reference Result: " << std::endl;
            reference_table->print();
            std::cout << "Computed Result: " << std::endl;
            reloaded_result->print();
            COGADB_FATAL_ERROR(
                "Result not equal to reference result for "
                "test '"
                    << file_name_reference_result << "'!",
                "");
          } else {
            std::cout << "Result table and reference table are equal!"
                      << std::endl;
          }
        }
      }
      std::cout << "Execute query successfully!" << std::endl;
      std::cout << "Compile Time: " << pipeline->getCompileTimeSec() << "s"
                << std::endl;
      std::cout << "Execution Time: "
                << double(end_execute - begin_execute) / (1000 * 1000 * 1000)
                << "s" << std::endl;
    }
  }
  return true;
}

bool Selection_Test(CodeGeneratorType code_generator,
                    const std::string& file_name_reference_result,
                    bool write_new_reference_result) {
  AttributeReferencePtr l_quantity = boost::make_shared<AttributeReference>(
      getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY", "QUANTITY");
  AttributeReferencePtr l_shipdate = boost::make_shared<AttributeReference>(
      getTablebyName("LINEITEM"), "LINEITEM.L_SHIPDATE", "SHIPDATE");

  std::vector<PredicateExpressionPtr> conjunctions;
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      l_quantity, boost::any(double(49)), GREATER));
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      l_shipdate, boost::any(std::string("1998-11-02")), GREATER));

  PredicateExpressionPtr selection_expr =
      createPredicateExpression(conjunctions, LOGICAL_AND);

  ProjectionParam param;
  param.push_back(AttributeReference(getTablebyName("LINEITEM"),
                                     "LINEITEM.L_QUANTITY", "QUANTITY"));
  param.push_back(AttributeReference(getTablebyName("LINEITEM"),
                                     "LINEITEM.L_SHIPDATE", "SHIPDATE"));
  param.push_back(AttributeReference(getTablebyName("LINEITEM"),
                                     "LINEITEM.L_ORDERKEY", "ORDERKEY"));

  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));
  assert(code_gen != NULL);

  if (!code_gen->consumeSelection(selection_expr)) COGADB_FATAL_ERROR("", "");

  //    code_gen->printCode(std::cout);

  PipelinePtr pipeline = code_gen->compile();

  if (!pipeline) return false;

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool Selection_On_String_Test(CodeGeneratorType code_generator,
                              const std::string& file_name_reference_result,
                              bool write_new_reference_result) {
  ProjectionParam param;
  param.push_back(
      AttributeReference(getTablebyName("PART"), "PART.P_BRAND", "BRAND"));
  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("PART"));

  AttributeReferencePtr p_brand = boost::make_shared<AttributeReference>(
      getTablebyName("PART"), "PART.P_BRAND", "BRAND");

  std::vector<PredicateExpressionPtr> conjunctions;
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      p_brand, boost::any(std::string("Brand#45")), EQUAL));

  PredicateExpressionPtr selection_expr =
      createPredicateExpression(conjunctions, LOGICAL_AND);

  if (!code_gen->consumeSelection(selection_expr)) COGADB_FATAL_ERROR("", "");

  PipelinePtr pipeline = code_gen->compile();

  if (!pipeline) return false;

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool Selection_Projection_Test(CodeGeneratorType code_generator,
                               const std::string& file_name_reference_result,
                               bool write_new_reference_result) {
  ProjectionParam param;
  param.push_back(
      AttributeReference(getTablebyName("LINEITEM"), "L_QUANTITY", "QUANTITY"));
  param.push_back(
      AttributeReference(getTablebyName("LINEITEM"), "L_SHIPDATE", "SHIPDATE"));
  param.push_back(
      AttributeReference(getTablebyName("LINEITEM"), "L_ORDERKEY", "ORDERKEY"));

  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));

  AttributeReferencePtr l_quantity = boost::make_shared<AttributeReference>(
      getTablebyName("LINEITEM"), "L_QUANTITY", "QUANTITY");
  AttributeReferencePtr l_shipdate = boost::make_shared<AttributeReference>(
      getTablebyName("LINEITEM"), "L_SHIPDATE", "SHIPDATE");

  std::vector<PredicateExpressionPtr> conjunctions;
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      l_quantity, boost::any(double(49)), GREATER));
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      l_shipdate, boost::any(std::string("1998-11-02")), GREATER));

  PredicateExpressionPtr selection_expr =
      createPredicateExpression(conjunctions, LOGICAL_AND);

  if (!code_gen->consumeSelection(selection_expr)) COGADB_FATAL_ERROR("", "");

  //    code_gen.printCode(std::cout);

  PipelinePtr pipeline = code_gen->compile();

  TablePtr result;
  for (size_t i = 0; i < 5; ++i) {
    Timestamp begin_execute = getTimestamp();
    bool ret = pipeline->execute();
    assert(ret == true);
    result = pipeline->getResult();
    Timestamp end_execute = getTimestamp();

    if (!result) {
      std::cerr << "Could not execute query successfully!" << std::endl;
    } else {
      if (i == 0) result->print();
      std::cout << "Execute query successfully!" << std::endl;
      std::cout << "Compile Time: " << pipeline->getCompileTimeSec() << "s"
                << std::endl;
      std::cout << "Execution Time: "
                << double(end_execute - begin_execute) / (1000 * 1000 * 1000)
                << "s" << std::endl;
    }
  }

  {
    ProjectionParam param;
    param.push_back(AttributeReference(result, "QUANTITY"));
    param.push_back(AttributeReference(result, "SHIPDATE"));

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, result);

    pipeline = code_gen->compile();
  }

  if (!pipeline) return false;

  assert(result->getColumnbyName("QUANTITY"));

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool Algebra_Projection_Test(CodeGeneratorType code_generator,
                             const std::string& file_name_reference_result,
                             bool write_new_reference_result) {
  ProjectionParam param;
  param.push_back(AttributeReference(getTablebyName("LINEITEM"),
                                     "LINEITEM.L_QUANTITY", "QUANTITY"));
  param.push_back(AttributeReference(
      getTablebyName("LINEITEM"), "LINEITEM.L_EXTENDEDPRICE", "EXTENDEDPRICE"));
  param.push_back(AttributeReference(getTablebyName("LINEITEM"),
                                     "LINEITEM.L_SHIPDATE", "SHIPDATE"));
  param.push_back(AttributeReference(getTablebyName("LINEITEM"),
                                     "LINEITEM.L_DISCOUNT", "DISCOUNT"));
  param.push_back(AttributeReference(getTablebyName("LINEITEM"),
                                     "LINEITEM.L_ORDERKEY", "ORDERKEY"));

  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));

  AttributeReferencePtr l_quantity = boost::make_shared<AttributeReference>(
      getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY", "QUANTITY");
  AttributeReferencePtr l_shipdate = boost::make_shared<AttributeReference>(
      getTablebyName("LINEITEM"), "LINEITEM.L_SHIPDATE", "SHIPDATE");

  std::vector<PredicateExpressionPtr> conjunctions;
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      l_quantity, boost::any(double(49)), GREATER));
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      l_shipdate, boost::any(std::string("1998-11-02")), GREATER));

  PredicateExpressionPtr selection_expr =
      createPredicateExpression(conjunctions, LOGICAL_AND);
  if (!code_gen->consumeSelection(selection_expr)) COGADB_FATAL_ERROR("", "");

  std::pair<bool, AttributeReference> algebra_ret =
      code_gen->consumeAlgebraComputation(
          AttributeReference(getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY"),
          AttributeReference(getTablebyName("LINEITEM"),
                             "LINEITEM.L_EXTENDEDPRICE"),
          MUL);
  if (!algebra_ret.first) {
    COGADB_FATAL_ERROR("", "");
  }

  AttributeReference quantity_mul_extended_price = algebra_ret.second;

  algebra_ret = code_gen->consumeAlgebraComputation(
      quantity_mul_extended_price,
      AttributeReference(getTablebyName("LINEITEM"), "LINEITEM.L_DISCOUNT",
                         "L_DISCOUNT"),
      MUL);
  if (!algebra_ret.first) {
    COGADB_FATAL_ERROR("", "");
  }
  /* add computed attribute to projection list */
  code_gen->addAttributeProjection(algebra_ret.second);

  PipelinePtr pipeline = code_gen->compile();

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool Selection_HashBuild_HashProbe_Test(
    CodeGeneratorType code_generator,
    const std::string& file_name_reference_result,
    bool write_new_reference_result) {
  AttributeReferencePtr l_quantity = boost::make_shared<AttributeReference>(
      getTablebyName("LINEITEM"), "LINEITEM.L_QUANTITY");
  AttributeReferencePtr l_shipdate = boost::make_shared<AttributeReference>(
      getTablebyName("LINEITEM"), "LINEITEM.L_SHIPDATE");

  ProjectionParam param2;

  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param2, getTablebyName("LINEITEM"));

  QueryContextPtr build_context = createQueryContext(BUILD_HASH_TABLE_PIPELINE);

  code_gen->addAttributeProjection(*l_quantity);
  code_gen->addAttributeProjection(*l_shipdate);
  code_gen->addAttributeProjection(
      AttributeReference(getTablebyName("LINEITEM"), "LINEITEM.L_ORDERKEY"));

  std::vector<PredicateExpressionPtr> conjunctions;
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      l_quantity, boost::any(double(49)), GREATER));
  conjunctions.push_back(createColumnConstantComparisonPredicateExpression(
      l_shipdate, boost::any(std::string("1998-11-02")), GREATER));

  PredicateExpressionPtr selection_expr =
      createPredicateExpression(conjunctions, LOGICAL_AND);

  if (!code_gen->consumeSelection(selection_expr)) COGADB_FATAL_ERROR("", "");
  if (!code_gen->consumeBuildHashTable(AttributeReference(
          getTablebyName("LINEITEM"), "LINEITEM.L_ORDERKEY")))
    COGADB_FATAL_ERROR("", "");
  //    code_gen->print();
  PipelinePtr pipeline = code_gen->compile();

  TablePtr result;
  for (size_t i = 0; i < 5; ++i) {
    Timestamp begin_execute = getTimestamp();
    bool ret = pipeline->execute();
    assert(ret == true);
    result = pipeline->getResult();
    Timestamp end_execute = getTimestamp();

    if (!result) {
      std::cerr << "Could not execute query successfully!" << std::endl;
    } else {
      if (i == 0) result->print();
      std::cout << "Execute query successfully!" << std::endl;
      std::cout << "Compile Time: " << pipeline->getCompileTimeSec() << "s"
                << std::endl;
      std::cout << "Execution Time: "
                << double(end_execute - begin_execute) / (1000 * 1000 * 1000)
                << "s" << std::endl;
    }
  }
  result->printSchema();
  //    std::string tmp;
  //    std::cin >> tmp;
  assert(result->getHashTablebyName("LINEITEM.L_ORDERKEY.1") != NULL);

  result->print();
  {
    AttributeReference o_orderkey(getTablebyName("ORDERS"), "ORDERS.O_ORDERKEY",
                                  "O_ORDERKEY");
    AttributeReference o_orderdate(getTablebyName("ORDERS"),
                                   "ORDERS.O_ORDERDATE", "O_ORDERDATE");

    ProjectionParam param;

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("ORDERS"));
    QueryContextPtr context = createQueryContext(NORMAL_PIPELINE);

    code_gen->addAttributeProjection(
        AttributeReference(result, "LINEITEM.L_QUANTITY", "QUANTITY"));
    code_gen->addAttributeProjection(
        AttributeReference(result, "LINEITEM.L_SHIPDATE", "SHIPDATE"));
    code_gen->addAttributeProjection(o_orderkey);
    code_gen->addAttributeProjection(o_orderdate);
    code_gen->addAttributeProjection(
        AttributeReference(result, "LINEITEM.L_ORDERKEY", "ORDERKEY"));

    storeResultTableAttributes(code_gen, context, result);

    code_gen->consumeProbeHashTable(
        AttributeReference(result, "LINEITEM.L_ORDERKEY"), o_orderkey);
    //                AttributeReference(getTablebyName("ORDERS"),
    //                "O_ORDERKEY"));
    pipeline = code_gen->compile();
  }

  //    assert(result->getColumnbyName("QUANTITY"));

  //    result->print();

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool Groupby_One_Attribute_Test(CodeGeneratorType code_generator,
                                const std::string& file_name_reference_result,
                                bool write_new_reference_result) {
  ProjectionParam param;
  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));

  ProcessorSpecification proc_spec(hype::PD0);
  GroupingAttributes grouping_columns;
  grouping_columns.push_back(AttributeReference(
      getTablebyName("LINEITEM"), "LINEITEM.L_RETURNFLAG", "RETURNFLAG"));

  AttributeReference l_extendedprice(
      getTablebyName("LINEITEM"), "LINEITEM.L_EXTENDEDPRICE", "NUMBER_OF_ROWS");
  AggregateSpecifications agg_specs;
  agg_specs.push_back(
      createAggregateSpecification(l_extendedprice, COUNT, "COUNT_ORDER"));
  agg_specs.push_back(
      createAggregateSpecification(l_extendedprice, SUM, "SUM_EXTENDEDPRICE"));

  GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);

  if (!code_gen->consumeHashGroupAggregate(groupby_param))
    COGADB_FATAL_ERROR("", "");

  PipelinePtr pipeline = code_gen->compile();

  if (!pipeline) return false;

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool Groupby_Two_Attributes_Test(CodeGeneratorType code_generator,
                                 const std::string& file_name_reference_result,
                                 bool write_new_reference_result) {
  ProjectionParam param;
  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));

  ProcessorSpecification proc_spec(hype::PD0);
  GroupingAttributes grouping_columns;
  grouping_columns.push_back(AttributeReference(
      getTablebyName("LINEITEM"), "LINEITEM.L_RETURNFLAG", "RETURNFLAG"));
  grouping_columns.push_back(AttributeReference(
      getTablebyName("LINEITEM"), "LINEITEM.L_LINESTATUS", "LINESTATUS"));

  AttributeReference l_extendedprice(
      getTablebyName("LINEITEM"), "LINEITEM.L_EXTENDEDPRICE", "NUMBER_OF_ROWS");
  AggregateSpecifications agg_specs;
  agg_specs.push_back(
      createAggregateSpecification(l_extendedprice, COUNT, "COUNT_ORDER"));
  agg_specs.push_back(
      createAggregateSpecification(l_extendedprice, SUM, "SUM_EXTENDEDPRICE"));

  GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);

  if (!code_gen->consumeHashGroupAggregate(groupby_param))
    COGADB_FATAL_ERROR("", "");

  PipelinePtr pipeline = code_gen->compile();

  if (!pipeline) return false;

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool Groupby_String_Attribute_Test(
    CodeGeneratorType code_generator,
    const std::string& file_name_reference_result,
    bool write_new_reference_result) {
  /* select c_name, sum(C_ACCTBAL) from customer where c_acctbal > 9999 group by
   * c_name; */

  ProjectionParam param;
  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("CUSTOMER"));

  AttributeReferencePtr c_acctbal = createInputAttribute(
      getTablebyName("CUSTOMER"), "CUSTOMER.C_ACCTBAL", "C_ACCTBAL");
  PredicateExpressionPtr selection_expr =
      createColumnConstantComparisonPredicateExpression(
          c_acctbal, boost::any(9999), GREATER);

  if (!code_gen->consumeSelection(selection_expr)) COGADB_FATAL_ERROR("", "");

  ProcessorSpecification proc_spec(hype::PD0);
  GroupingAttributes grouping_columns;
  grouping_columns.push_back(AttributeReference(getTablebyName("CUSTOMER"),
                                                "CUSTOMER.C_NAME", "C_NAME"));

  //    AttributeReference l_extendedprice(getTablebyName("CUSTOMER"),
  //    "C_ACCTBAL", "C_ACCTBAL");
  AggregateSpecifications agg_specs;
  agg_specs.push_back(createAggregateSpecification(*c_acctbal, SUM, "ACCTBAL"));

  GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);
  if (!code_gen->consumeHashGroupAggregate(groupby_param))
    COGADB_FATAL_ERROR("", "");

  PipelinePtr pipeline = code_gen->compile();

  if (!pipeline) return false;

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool Aggregation_Test(CodeGeneratorType code_generator,
                      const std::string& file_name_reference_result,
                      bool write_new_reference_result) {
  ProjectionParam param;
  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));

  ProcessorSpecification proc_spec(hype::PD0);

  AggregateSpecifications agg_specs;

  AttributeReference l_extendedprice(getTablebyName("LINEITEM"),
                                     "LINEITEM.L_EXTENDEDPRICE",
                                     "NUMBER_OF_ROWS", 1);
  agg_specs.push_back(
      createAggregateSpecification(l_extendedprice, COUNT, "COUNT_ORDER"));
  agg_specs.push_back(
      createAggregateSpecification(l_extendedprice, SUM, "SUM_EXTENDEDPRICE"));

  if (!code_gen->consumeAggregate(agg_specs)) COGADB_FATAL_ERROR("", "");

  PipelinePtr pipeline = code_gen->compile();
  if (!pipeline) return false;

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool TPCH1_Test(CodeGeneratorType code_generator,
                const std::string& file_name_reference_result,
                bool write_new_reference_result) {
  ProjectionParam param;
  CodeGeneratorPtr code_gen =
      createCodeGenerator(code_generator, param, getTablebyName("LINEITEM"));

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
      SUM, "SUM_QTY"));
  agg_specs.push_back(createAggregateSpecification(
      extended_price_multiply_one_minus_discount, SUM, "SUM_DISCOUNT_PRICE"));

  GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);
  if (!code_gen->consumeHashGroupAggregate(groupby_param))
    COGADB_FATAL_ERROR("", "");

  PipelinePtr pipeline = code_gen->compile();
  if (!pipeline) return false;

  return repeatedPipelineExecution(pipeline, file_name_reference_result,
                                   write_new_reference_result);
}

bool Genome_Query_Test(CodeGeneratorType code_generator,
                       const std::string& file_name_reference_result,
                       bool write_new_reference_result) {
  /* select sb_read_id, r_qname, r_flag, c_name, rb_position,
   *        r_mapq, sb_base_value, sb_insert_offset, sb_base_call_quality
   *  from sample_base join reference_base on sb_rb_id = rb_id
   *      join contig on rb_c_id = c_id join read on sb_read_id = r_id;
   */

  AttributeReferencePtr sb_read_id = boost::make_shared<AttributeReference>(
      getTablebyName("SAMPLE_BASE"), "SB_READ_ID");
  AttributeReferencePtr sb_insert_offset =
      boost::make_shared<AttributeReference>(getTablebyName("SAMPLE_BASE"),
                                             "SB_INSERT_OFFSET");
  AttributeReferencePtr sb_base_call_quality =
      boost::make_shared<AttributeReference>(getTablebyName("SAMPLE_BASE"),
                                             "SB_BASE_CALL_QUALITY");
  AttributeReferencePtr sb_base_value = boost::make_shared<AttributeReference>(
      getTablebyName("SAMPLE_BASE"), "SB_BASE_VALUE");
  AttributeReferencePtr sb_rb_id = boost::make_shared<AttributeReference>(
      getTablebyName("SAMPLE_BASE"), "SB_RB_ID");
  AttributeReferencePtr rb_id = boost::make_shared<AttributeReference>(
      getTablebyName("REFERENCE_BASE"), "RB_ID");
  AttributeReferencePtr rb_c_id = boost::make_shared<AttributeReference>(
      getTablebyName("REFERENCE_BASE"), "RB_C_ID");
  AttributeReferencePtr rb_position = boost::make_shared<AttributeReference>(
      getTablebyName("REFERENCE_BASE"), "RB_POSITION");

  //    AttributeReferencePtr rg_name = boost::make_shared<AttributeReference>(
  //            getTablebyName("REFERENCE_GENOME"), "RG_NAME");

  AttributeReferencePtr r_id =
      boost::make_shared<AttributeReference>(getTablebyName("READ"), "R_ID");
  AttributeReferencePtr r_qname =
      boost::make_shared<AttributeReference>(getTablebyName("READ"), "R_QNAME");
  AttributeReferencePtr r_flag =
      boost::make_shared<AttributeReference>(getTablebyName("READ"), "R_FLAG");
  AttributeReferencePtr r_mapq =
      boost::make_shared<AttributeReference>(getTablebyName("READ"), "R_MAPQ");

  AttributeReferencePtr c_id =
      boost::make_shared<AttributeReference>(getTablebyName("CONTIG"), "C_ID");
  AttributeReferencePtr c_name = boost::make_shared<AttributeReference>(
      getTablebyName("CONTIG"), "C_NAME");

  Timestamp begin_query = getTimestamp();

  PipelinePtr hb_read_r_id_pipeline;
  PipelinePtr hb_refbase_rb_id_pipeline;
  PipelinePtr hb_contig_c_id_pipeline;
  PipelinePtr hp_samplebase;

  /* hash build read.r_id */
  {
    ProjectionParam param;
    param.push_back(*r_id);
    param.push_back(*r_flag);
    param.push_back(*r_mapq);
    param.push_back(*r_qname);

    CPPCodeGenerator code_gen(param, getTablebyName("READ"));
    if (!code_gen.consumeBuildHashTable(*r_id)) COGADB_FATAL_ERROR("", "");
    code_gen.printCode(std::cout);
    hb_read_r_id_pipeline = code_gen.compile();
  }

  /* hash build reference_base.rb_id */
  {
    ProjectionParam param;
    param.push_back(*rb_id);
    param.push_back(*rb_c_id);
    param.push_back(*rb_position);

    CodeGeneratorPtr code_gen = createCodeGenerator(
        code_generator, param, getTablebyName("REFERENCE_BASE"));

    if (!code_gen->consumeBuildHashTable(*rb_id)) COGADB_FATAL_ERROR("", "");

    hb_refbase_rb_id_pipeline = code_gen->compile();
    assert(hb_refbase_rb_id_pipeline != NULL);
  }

  /* hash build contig.c_id */
  {
    ProjectionParam param;
    param.push_back(*c_id);
    param.push_back(*c_name);

    CodeGeneratorPtr code_gen =
        createCodeGenerator(code_generator, param, getTablebyName("CONTIG"));

    if (!code_gen->consumeBuildHashTable(*c_id)) COGADB_FATAL_ERROR("", "");

    //        code_gen.printCode(std::cout);
    hb_contig_c_id_pipeline = code_gen->compile();
  }

  assert(hb_read_r_id_pipeline != NULL);
  assert(hb_refbase_rb_id_pipeline != NULL);
  assert(hb_contig_c_id_pipeline != NULL);

  Timestamp begin_hashbuild = getTimestamp();
  hb_read_r_id_pipeline->execute();
  hb_refbase_rb_id_pipeline->execute();
  hb_contig_c_id_pipeline->execute();
  Timestamp end_hashbuild = getTimestamp();

  /* probe phase */
  {
    assert(hb_read_r_id_pipeline->getResult()->getHashTablebyName("R_ID") !=
           NULL);
    /* create attribute references for attributes for input tables */
    AttributeReference r_id_tmp(hb_read_r_id_pipeline->getResult(), "R_ID");
    AttributeReference rb_id_tmp(hb_refbase_rb_id_pipeline->getResult(),
                                 "RB_ID");
    AttributeReference c_id_tmp(hb_contig_c_id_pipeline->getResult(), "C_ID");

    AttributeReferencePtr r_qname_tmp = createInputAttributeForNewTable(
        *r_qname, hb_read_r_id_pipeline->getResult());
    AttributeReferencePtr r_flag_tmp = createInputAttributeForNewTable(
        *r_flag, hb_read_r_id_pipeline->getResult());
    AttributeReferencePtr r_mapq_tmp = createInputAttributeForNewTable(
        *r_mapq, hb_read_r_id_pipeline->getResult());
    AttributeReferencePtr c_name_tmp = createInputAttributeForNewTable(
        *c_name, hb_contig_c_id_pipeline->getResult());
    AttributeReferencePtr rb_position_tmp = createInputAttributeForNewTable(
        *rb_position, hb_refbase_rb_id_pipeline->getResult());

    ProjectionParam param;
    param.push_back(*sb_read_id);
    param.push_back(*r_qname_tmp);
    param.push_back(*r_flag_tmp);
    param.push_back(*c_name_tmp);
    param.push_back(*rb_position_tmp);
    param.push_back(*r_mapq_tmp);
    param.push_back(*sb_base_value);
    param.push_back(*sb_insert_offset);
    param.push_back(*sb_base_call_quality);

    CodeGeneratorPtr code_gen = createCodeGenerator(
        code_generator, param, getTablebyName("SAMPLE_BASE"));

    /* hash probes */
    assert(r_id_tmp.getHashTable() != NULL);
    assert(hb_read_r_id_pipeline->getResult()->getHashTablebyName(
               r_id_tmp.getUnversionedAttributeName()) != NULL);
    if (!code_gen->consumeProbeHashTable(r_id_tmp, *sb_read_id))
      COGADB_FATAL_ERROR("", "");
    if (!code_gen->consumeProbeHashTable(rb_id_tmp, *sb_rb_id))
      COGADB_FATAL_ERROR("", "");
    if (!code_gen->consumeProbeHashTable(c_id_tmp, *rb_c_id))
      COGADB_FATAL_ERROR("", "");

    hp_samplebase = code_gen->compile();
  }

  assert(hp_samplebase != NULL);
  Timestamp begin_hashprobe = getTimestamp();
  hp_samplebase->execute();
  Timestamp end_hashprobe = getTimestamp();
  Timestamp end_query = getTimestamp();

  if (hp_samplebase && hp_samplebase->getResult()->getNumberofRows() < 100000) {
    hp_samplebase->getResult()->print();
  }

  double hash_build_in_sec =
      double(end_hashbuild - begin_hashbuild) / (1000 * 1000 * 1000);
  double hash_probe_in_sec =
      double(end_hashprobe - begin_hashprobe) / (1000 * 1000 * 1000);
  double total_query_time_in_sec =
      double(end_query - begin_query) / (1000 * 1000 * 1000);

  std::cout << std::setprecision(6);
  std::cout << "Hash Build (Total): " << hash_build_in_sec << "s" << std::endl;
  std::cout << "Hash Probe (Total): " << hash_probe_in_sec << "s" << std::endl;
  std::cout << "Execution Time: " << hash_build_in_sec + hash_probe_in_sec
            << "s" << std::endl;
  std::cout << "Execution Time (Including compilation times): "
            << total_query_time_in_sec << "s" << std::endl;

  return true;
}

int performTests(const Tests& tests, const CodeGeneratorType& code_generator,
                 bool write_new_reference_results) {
  Tests::const_iterator cit;
  for (cit = tests.begin(); cit != tests.end(); ++cit) {
    bool ret =
        (cit->second(code_generator, cit->first, write_new_reference_results));
    if (ret) {
      std::cout << "Test '" << cit->first << "' was successful!" << std::endl;
    } else {
      std::cout << "Test '" << cit->first << "' failed!" << std::endl;
      return -1;
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
  /* defines which code generator should be tested */
  CodeGeneratorType code_generator = C_CODE_GENERATOR;
  /* defines whether the tests should check for correctness by comparing with
   *  old saved csv files (false) or generate new csv result files for result
   * comparison with later test results. */
  bool write_new_reference_results = false;

  std::cout << "Unittests for Code Generators" << std::endl;
  boost::program_options::options_description desc("Options");
  std::string name_of_code_generator = "all";
  desc.add_options()("help,h", "Print help messages")(
      "write_new_reference_results",
      "For all registered tests, the result is written as self-contained csv "
      "file. "
      "NOTE: This will by default update the files in the current build "
      "directory only! To update "
      "the persistent reference results, you have to specify the directory in "
      "the source tree: "
      "'gpudbms/cogadb/test/testdata/unittests/code_generators/'")(
      "path_to_reference_results", boost::program_options::value<std::string>(
                                       &directory_containing_reference_results),
      "Specify path to directory where reference results are read from or "
      "written to")(
      "code_generator",
      boost::program_options::value<std::string>(&name_of_code_generator),
      "Specify code generator that should be tested. Valid Values are: all, c, "
      "multi_staged");

  boost::program_options::variables_map vm;
  try {
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, desc), vm);
  } catch (boost::program_options::error& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }

  if (vm.count("help")) {
    desc.print(std::cout);
    return 0;
  }
  // throws on error, so we call this after we checked for the help option
  // note that this command sets the correct values to all variables were we
  // passed a pointer to the options_description object earlier
  boost::program_options::notify(vm);

  if (vm.count("write_new_reference_results")) {
    write_new_reference_results = true;
  }

  /* read instructions to load database from config file */
  ClientPtr client(new LocalClient());
  loadReferenceDatabaseTPCHScaleFactor1(client);

  Tests tests;
  /* register tests */
  if (getTablebyName("LINEITEM") != NULL) {
    if (getTablebyName("LINEITEM")->getNumberofRows() != 6001215) {
      COGADB_FATAL_ERROR(
          "TPC-H test queries require a scale factor 1 database!", "");
    }
    /* Tests on TPC-H database with scale factor 1 */
    std::cout << "Detected TPC-H database, executing test queries for TPC-H "
                 "databases..."
              << std::endl;
    tests.insert(std::make_pair("Selection_Test", &Selection_Test));
    //        tests.insert(std::make_pair("Selection_Projection_Test",
    //        &Selection_Projection_Test));
    tests.insert(
        std::make_pair("Selection_On_String_Test", &Selection_On_String_Test));
    tests.insert(std::make_pair("Selection_HashBuild_HashProbe_Test",
                                &Selection_HashBuild_HashProbe_Test));
    tests.insert(
        std::make_pair("Algebra_Projection_Test", &Algebra_Projection_Test));
    tests.insert(std::make_pair("Groupby_Test", &Groupby_One_Attribute_Test));
    tests.insert(std::make_pair("Groupby_String_Attribute_Test",
                                &Groupby_String_Attribute_Test));
    tests.insert(std::make_pair("Groupby_Two_Attributes_Test",
                                &Groupby_Two_Attributes_Test));
    tests.insert(std::make_pair("TPCH1_Test", &TPCH1_Test));
    tests.insert(std::make_pair("Aggregation_Test", &Aggregation_Test));
  } else if (getTablebyName("SAMPLE_BASE") != NULL) {
    /* Tests on genome database with simple key schema and default .bam test
     * file */
    std::cout << "Detected genome database (simple key), executing test "
                 "queries for genome databases..."
              << std::endl;
    tests.insert(std::make_pair("Genome_Query_Test", &Genome_Query_Test));
  } else {
    COGADB_FATAL_ERROR(
        "Unknown database type (TPC-H, Genome), or tables were not loaded!"
            << "Check the startup.coga whether a correct database is loaded!",
        "");
  }

  if (name_of_code_generator == "all") {
    std::cout << "Testing C_CodeGenerator..." << std::endl;
    if (performTests(tests, C_CODE_GENERATOR, write_new_reference_results)) {
#ifndef __APPLE__
      quick_exit(-1);
#endif
      return -1;
    }
    std::cout << "Testing Multi_Staged CodeGenerator..." << std::endl;
    if (performTests(tests, MULTI_STAGE_CODE_GENERATOR,
                     write_new_reference_results)) {
#ifndef __APPLE__
      quick_exit(-1);
#endif
      return -1;
    }
  } else {
    convertToCodeGenerator(name_of_code_generator, code_generator);
    if (performTests(tests, code_generator, write_new_reference_results)) {
#ifndef __APPLE__
      quick_exit(-1);
#endif
      return -1;
    }
  }

  if (!write_new_reference_results) {
    std::cout << "All tests successfully completed!" << std::endl;
  } else {
    std::cout << "Successfully wrote new reference results to directory: '"
              << directory_containing_reference_results << "'" << std::endl;
  }
#ifndef __APPLE__
  quick_exit(0);
#endif
  return 0;
}
