
#include <iostream>
#include <persistence/storage_manager.hpp>
#include <util/getname.hpp>
#include <util/tests.hpp>

#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/thread/lock_guard.hpp>
#include <core/block_iterator.hpp>
#include <parser/commandline_interpreter.hpp>
#include <query_compilation/algebraic_aggregate_specification.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/pipeline_info.hpp>
#include <query_compilation/predicate_expression.hpp>
#include <query_compilation/query_context.hpp>

using namespace CoGaDB;

// class PipelineJob;
// typedef boost::shared_ptr<PipelineJob> PipelineJobPtr;
//
// class PipelineJob{
// public:
//    PipelineJob(TablePtr source_table,
//        size_t block_size,
//        PipelinePtr _pipeline);
//    const TablePtr getNextBlock(const ProcessorSpecification& proc_spec);
// private:
//    TablePtr source_table;
//    BlockIteratorPtr it;
//    PipelinePtr pipeline;
//    boost::mutex mutex;
//};
//
//    PipelineJob::PipelineJob(TablePtr _source_table,
//        size_t block_size,
//        PipelinePtr _pipeline)
//    : source_table(_source_table), it(new BlockIterator(block_size)),
//    pipeline(_pipeline), mutex()
//    {
//
//    }
//
// const TablePtr PipelineJob::getNextBlock(const ProcessorSpecification&
// proc_spec){
//    boost::lock_guard<boost::mutex> lock(mutex);
//    TablePtr block;
//    if(it->getOffset()<source_table->getNumberofRows()){
//        block = BaseTable::getNext(source_table, it, proc_spec);
//    }
//    return block;
//}
//
// void worker_thread(PipelineJobPtr job,
//        PipelinePtr pipeline,
//        TablePtr source_table,
//        const ProcessorSpecification& proc_spec,
//        TablePtr* result_table
//        ){
//
//
//    TablePtr previous_block;
//    TablePtr block;
//    TablePtr result;
//
//    block = job->getNextBlock(proc_spec);
//
//    while(block){
//
//        std::map<TablePtr,TablePtr> table_replacement;
//        if(previous_block){
//            table_replacement.insert(std::make_pair(previous_block, block));
//        }else{
//            table_replacement.insert(std::make_pair(source_table, block));
//        }
//        pipeline->replaceInputTables(table_replacement);
//
//        assert(block!=NULL);
////        std::cout << "=== Input Block ===" << std::endl;
////        block->print();
//
//        if(!pipeline->execute()){
//            COGADB_FATAL_ERROR("","");
//        }
////        std::cout << "=== Result ===" << std::endl;
////        pipeline->getResult()->print();
//
//        if(!result){
//            result = pipeline->getResult();
//        }else{
//            result->append(pipeline->getResult());
//        }
//
//        previous_block=block;
//        block = job->getNextBlock(proc_spec);
//    }
//
//    *result_table=result;
//}
//
// const TablePtr postAggregation(PipelinePtr pipeline, TablePtr
// unmerged_result){
//
//}
//
// typedef boost::shared_ptr<AggregationParam> AggregationParamPtr;
//
// bool isBuildHashTablePipeline(PipelinePtr pipeline);
//
// bool isAggregationPipeline(PipelinePtr pipeline);
//
////bool containsOnlyAlgebraicAggregation(PipelinePtr pipeline);
////
////bool containsUDFAggregation(PipelinePtr pipeline);
////
////bool containsAverageAggregation(PipelinePtr pipeline);
//
// bool getCombineAggregationFunction(const AggregationFunction& agg_func,
// AggregationFunction& combiner_agg_func);
//
// AggregationParamPtr toAggregationParam(AggregateSpecificationPtr agg_spec,
//        const ProcessorSpecification& proc_spec);
//
//
// bool isBuildHashTablePipeline(PipelinePtr pipeline){
//    return
//    (pipeline->getPipelineInfo()->getPipelineType()==MATERIALIZE_FROM_ARRAY_TO_JOIN_HASH_TABLE_AND_ARRAY);
//}
//
// bool isAggregationPipeline(PipelinePtr pipeline){
//    return
//    (pipeline->getPipelineInfo()->getPipelineType()==MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY);
//}
//
////bool containsOnlyAlgebraicAggregation(PipelinePtr pipeline);
////
////bool containsUDFAggregation(PipelinePtr pipeline);
////
////bool containsAverageAggregation(PipelinePtr pipeline){
////
////}
//
// bool isMergeAggregationPossible(PipelinePtr pipeline){
//    assert(pipeline!=NULL);
//    PipelineInfoPtr pipe_info = pipeline->getPipelineInfo();
//    assert(pipe_info!=NULL);
//    GroupByAggregateParamPtr groupby = pipe_info->getGroupByAggregateParam();
//    assert(groupby!=NULL);
//
//    ProcessorSpecification proc_spec(hype::PD0);
//    for(size_t i=0;i<groupby->aggregation_specs.size();++i){
//
//        AggregationParamPtr agg_param
//                = toAggregationParam(groupby->aggregation_specs[i],
//                proc_spec);
//        if(!agg_param){
//            return false;
//        }
//    }
//    return true;
//}
//
// bool getCombineAggregationFunction(const AggregationFunction& agg_func,
//        AggregationFunction& combiner_agg_func){
//
//    if(agg_func==COUNT){
//        combiner_agg_func=SUM;
//    }else if(agg_func==SUM
//            || agg_func==MIN
//            || agg_func==MAX){
//        combiner_agg_func=agg_func;
//    }else{
//        return false;
//    }
//
//    return true;
//}
//
// AggregationParamPtr toAggregationParam(AggregateSpecificationPtr agg_spec,
//        const ProcessorSpecification& proc_spec){
//        boost::shared_ptr<AlgebraicAggregateSpecification> alg_agg_spec;
//        alg_agg_spec=boost::dynamic_pointer_cast<AlgebraicAggregateSpecification>(agg_spec);
//        if(!alg_agg_spec)
//            return AggregationParamPtr();
//
//        AggregationFunction combiner_agg_func=SUM;
//        if(!getCombineAggregationFunction(alg_agg_spec->agg_func,
//        combiner_agg_func)){
//            return AggregationParamPtr();
//        }
//
//        AggregationParamPtr result(new AggregationParam(proc_spec,
//        combiner_agg_func, HASH_BASED_AGGREGATION,
//                alg_agg_spec->result_attr.getResultAttributeName()));
//
//        return result;
//}
//
// const TablePtr parallel_execute(PipelinePtr pipeline){
//
//    size_t num_threads = boost::thread::hardware_concurrency();
////    size_t num_threads = 1;
//    ProcessorSpecification proc_spec(hype::PD0);
//
//    PipelineInfoPtr pipe_info = pipeline->getPipelineInfo();
//    assert(pipe_info!=NULL);
//    TablePtr source_table = pipe_info->getSourceTable();
////    pipeline->
//
//    PipelineJobPtr job(new PipelineJob(source_table, 100000, pipeline));
//
//    std::vector<PipelinePtr> pipelines(num_threads);
//    std::vector<TablePtr> result_tables(num_threads);
//
//    boost::thread_group threads;
//    for(size_t i=0;i<num_threads;++i){
//        pipelines[i]=pipeline->copy();
//        boost::thread* t = new boost::thread(&worker_thread, job,
//                pipelines[i],
//                source_table,
//                proc_spec,
//                &result_tables[i]);
//        threads.add_thread(t);
////        worker_thread(job,
////                pipelines[i],
////                source_table,
////                proc_spec,
////                &result_tables[i]);
//
////    boost::thread_group threads;
////    boost::thread* th = new boost::thread(&increment_count);
////    threads.add_thread(th);
////    BOOST_TEST(! threads.is_this_thread_in());
////    threads.join_all();
//    }
//    threads.join_all();
//
//    TablePtr result;
//    /* it may happen that a worker thread did not get a chunk to process,
//    because
//     the table is too small. So we need to find a valid result table first. */
//    for(size_t i=0;i<num_threads;++i){
//        if(result_tables[i]){
//            result=result_tables[i];
//            break;
//        }
//    }
//    assert(result!=NULL);
//    for(size_t i=1;i<num_threads;++i){
//        if(result_tables[i]!=NULL && result_tables[i]!=result){
//            result->append(result_tables[i]);
//        }
//    }
//
//    GroupByAggregateParamPtr groupby = pipe_info->getGroupByAggregateParam();
//
//
//
//    std::cout << "Intermediate Result: " << std::endl;
//    result->print();
////    result->setName("CHEESE");
//
//    GroupingColumns grouping;
//    for(size_t i=0;i<groupby->grouping_attrs.size();++i){
//        grouping.push_back(groupby->grouping_attrs[i].getResultAttributeName());
//    }
//    AggregationFunctions aggs;
//    for(size_t i=0;i<groupby->aggregation_specs.size();++i){
//
//        AggregationParamPtr agg_param
//                = toAggregationParam(groupby->aggregation_specs[i],
//                proc_spec);
//        assert(agg_param);
//
//        aggs.push_back(std::make_pair(agg_param->new_column_name,
//                *agg_param));
//    }
////    aggs.push_back(std::make_pair(std::string("COUNT_ORDER"),
/// AggregationParam(proc_spec, SUM, HASH_BASED_AGGREGATION, "COUNT_ORDER")));
////    aggs.push_back(std::make_pair(std::string("SUM_EXTENDEDPRICE"),
/// AggregationParam(proc_spec, SUM, HASH_BASED_AGGREGATION,
///"SUM_EXTENDEDPRICE")));
//
//    GroupbyParam groupby_param(
//        proc_spec,
//        grouping,
//        aggs);
//
////    std::string groupby_string="GROUPBY";
////    groupby_string+=" (";
////    std::list<std::string>::const_iterator cit;
////
/// for(cit=groupby_param.grouping_columns.begin();cit!=groupby_param.grouping_columns.end();++cit){
////       groupby_string+=*cit;
////       if(cit!=--groupby_param.grouping_columns.end())
////           groupby_string+=",";
////    }
////    groupby_string+=")";
////    groupby_string+=" USING (";
////    AggregationFunctions::const_iterator agg_func_cit;
////    for(agg_func_cit=groupby_param.aggregation_functions.begin();
////           agg_func_cit!=groupby_param.aggregation_functions.end();
////           ++agg_func_cit){
////       groupby_string+=agg_func_cit->first;
////       groupby_string+="(";
////       groupby_string+=agg_func_cit->first;
////       groupby_string+=")";
////    }
////    groupby_string+=")";
////    std::cout << groupby_string << std::endl;
//
//    result=BaseTable::groupby(result, groupby_param);
//    assert(result!=NULL);
//    result->setName("");
////
////    ProjectionParam param;
////    CodeGeneratorPtr code_gen = createCodeGenerator(CPP_CODE_GENERATOR,
////            param, result);
////
////    GroupingAttributes grouping_columns;
////    grouping_columns.push_back(AttributeReference(result, "SHIPMODE",
///"SHIPMODE"));
////
//////    AttributeReference l_extendedprice(result, "NUMBER_OF_ROWS",
///"NUMBER_OF_ROWS");
////    AttributeReference count(result, "COUNT_ORDER", "COUNT_ORDER");
////    AttributeReference sum(result, "SUM_EXTENDEDPRICE",
///"SUM_EXTENDEDPRICE");
////
////    AggregateSpecifications agg_specs;
////    agg_specs.push_back(createAggregateSpecification(count, SUM,
///"COUNT_ORDER"));
////    agg_specs.push_back(createAggregateSpecification(sum, SUM,
///"SUM_EXTENDEDPRICE"));
////
////        GroupByAggregateParam groupby_param2(proc_spec, grouping_columns,
/// agg_specs);
////
//////    ProcessorSpecification proc_spec(hype::PD0);
////    if (!code_gen->consumeHashGroupAggregate(groupby_param2))
////        COGADB_FATAL_ERROR("", "");
////
////    PipelinePtr merge_pipeline = code_gen->compile();
////    assert(merge_pipeline!=NULL);
//////    std::map<TablePtr,TablePtr> table_replacement;
//////    table_replacement.insert(std::make_pair(source_table, result));
//////    merge_pipeline->replaceInputTables(table_replacement);
////
////
////    if(!merge_pipeline->execute()){
////        COGADB_FATAL_ERROR("","");
////    }
////
////    assert(merge_pipeline->getResult()!=NULL);
////    result = merge_pipeline->getResult();
//
//    return result;
//}
//
// const TablePtr execute(PipelinePtr pipeline){
//    if(isAggregationPipeline(pipeline)){
////        if(containsOnlyAlgebraicAggregation(pipeline)
////                && !containsUDFAggregation(pipeline)
////                && !containsAverageAggregation(pipeline)){
//        if(isMergeAggregationPossible(pipeline)){
//            return parallel_execute(pipeline);
//        }else{
//            /* Either we cannot build a merge pipeline or we cannot
//             split the aggregation because at least one is holistic.
//             Thus we execute the pipeline serially. */
//            pipeline->execute();
//            return pipeline->getResult();
//        }
//    }else if(isBuildHashTablePipeline(pipeline)){
//        /* we do not yet support parallel build of shared hash table! */
//        pipeline->execute();
//        return pipeline->getResult();
//    }else{
//        return parallel_execute(pipeline);
//    }
//}

int main(int argc, char** argv) {
  //
  //    if(argc<2){
  //        std::cerr << "Missing Parameter: Expect path to JSON file!" <<
  //        std::endl;
  //        return -1;
  //    }else if(argc>2){
  //        std::cerr << "Too many Parameters: Expect path to JSON file!" <<
  //        std::endl;
  //        return -1;
  //    }
  //
  //    std::string json_file=argv[1];

  CoGaDB::ClientPtr client(new CoGaDB::LocalClient());
  if (!CoGaDB::loadReferenceDatabaseStarSchemaScaleFactor1(client)) {
    COGADB_FATAL_ERROR("Failed to load database!", "");
  }

  PipelinePtr pipeline;
  TablePtr table;
  QueryContextPtr context = createQueryContext();
  {
      //    table = getTablebyName("LINEORDER");
      //
      //    std::list<std::string> cols;
      //    cols.push_back("LO_QUANTITY");
      //    cols.push_back("LO_DISCOUNT");
      //    cols.push_back("LO_REVENUE");
      //    table=table->projection(table, cols);
      //
      //    table->setName("LINEORDER");
      //
      //    AttributeReferencePtr lo_quantity =
      //    boost::make_shared<AttributeReference>(
      //            table, "LO_QUANTITY", "QUANTITY");
      //    AttributeReferencePtr lo_discount =
      //    boost::make_shared<AttributeReference>(
      //            table, "LO_DISCOUNT", "DISCOUNT");
      //    AttributeReferencePtr lo_revenue =
      //    boost::make_shared<AttributeReference>(
      //            table, "LO_REVENUE", "REVENUE");
      //
      //    /* select LO_QUANTITY, LO_DISCOUNT, LO_REVENUE
      //     * from lineorder
      //     * where LO_QUANTITY<25 AND lo_discount<=3 AND lo_discount>=1 AND
      //     lo_revenue>4800000; */
      //
      //    std::vector<PredicateExpressionPtr> conjunctions;
      //    conjunctions.push_back(
      //            createColumnConstantComparisonPredicateExpression(
      //            lo_quantity, boost::any(int(25)), LESSER));
      //    conjunctions.push_back(
      //            createColumnConstantComparisonPredicateExpression(
      //            lo_discount, boost::any(int(3)), LESSER_EQUAL));
      //    conjunctions.push_back(
      //            createColumnConstantComparisonPredicateExpression(
      //            lo_discount, boost::any(int(1)), GREATER_EQUAL));
      //    conjunctions.push_back(
      //            createColumnConstantComparisonPredicateExpression(
      //            lo_revenue, boost::any(int(4800000)), GREATER));
      //
      //    PredicateExpressionPtr selection_expr =
      //    createPredicateExpression(conjunctions, LOGICAL_AND);
      //
      //    ProjectionParam param;
      //    param.push_back(*lo_quantity);
      //    param.push_back(*lo_discount);
      //    param.push_back(*lo_revenue);
      //
      //    CodeGeneratorPtr code_gen = createCodeGenerator(CPP_CODE_GENERATOR,
      //            param, table);
      //    assert(code_gen != NULL);
      //
      //    if (!code_gen->consumeSelection(selection_expr))
      //        COGADB_FATAL_ERROR("", "");
      //
      //    pipeline = code_gen->compile();
      //    assert(pipeline!=NULL);
  } {
    //    table = getTablebyName("PART");
    //    ProjectionParam param;
    //    param.push_back(AttributeReference(getTablebyName("PART"), "P_BRAND",
    //    "BRAND"));
    //    CodeGeneratorPtr code_gen = createCodeGenerator(CPP_CODE_GENERATOR,
    //            param, table);
    //
    //    AttributeReferencePtr p_brand =
    //    boost::make_shared<AttributeReference>(
    //            getTablebyName("PART"), "P_BRAND", "BRAND");
    //
    //    std::vector<PredicateExpressionPtr> conjunctions;
    //    conjunctions.push_back(createColumnConstantComparisonPredicateExpression(p_brand,
    //    boost::any(std::string("Brand#45")), UNEQUAL));
    //
    //    PredicateExpressionPtr selection_expr =
    //    createPredicateExpression(conjunctions, LOGICAL_AND);
    //
    //    if (!code_gen->consumeSelection(selection_expr))
    //        COGADB_FATAL_ERROR("", "");
    //    pipeline = code_gen->compile();
    //    assert(pipeline!=NULL);
  }

  table = getTablebyName("LINEORDER");
  ProjectionParam param;
  CodeGeneratorPtr code_gen =
      createCodeGenerator(CPP_CODE_GENERATOR, param, table);

  ProcessorSpecification proc_spec(hype::PD0);
  GroupingAttributes grouping_columns;
  grouping_columns.push_back(
      AttributeReference(table, "LO_SHIPMODE", "SHIPMODE"));

  AttributeReference l_extendedprice(table, "LO_EXTENDEDPRICE",
                                     "NUMBER_OF_ROWS");
  AggregateSpecifications agg_specs;
  agg_specs.push_back(
      createAggregateSpecification(l_extendedprice, COUNT, "COUNT_ORDER"));
  agg_specs.push_back(
      createAggregateSpecification(l_extendedprice, SUM, "SUM_EXTENDEDPRICE"));

  GroupByAggregateParam groupby_param(proc_spec, grouping_columns, agg_specs);

  if (!code_gen->consumeHashGroupAggregate(groupby_param))
    COGADB_FATAL_ERROR("", "");

  pipeline = code_gen->compile();
  assert(pipeline != NULL);

  for (size_t i = 0; i < 5; ++i) {
    if (!pipeline->execute()) {
      COGADB_FATAL_ERROR("", "");
    }
  }

  pipeline->getResult()->print();

  std::cout << "Compile time: " << pipeline->getCompileTimeSec() << std::endl;
  std::cout << "Exec time: " << pipeline->getExecutionTimeSec() << std::endl;

  TablePtr result;
  Timestamp begin, end;
  for (size_t i = 0; i < 5; ++i) {
    begin = getTimestamp();
    result = execute(pipeline, context);
    end = getTimestamp();
  }
  std::cout << "=== Result ===" << std::endl;
  result->print();
  std::cout << "Execution Time: " << double(end - begin) / (1000 * 1000 * 1000)
            << "s" << std::endl;

  return 0;
}
