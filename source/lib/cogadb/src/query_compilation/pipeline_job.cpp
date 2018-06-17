#include <query_compilation/pipeline_job.hpp>

#include <core/base_table.hpp>
#include <core/block_iterator.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/algebraic_aggregate_specification.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/ocl_api.hpp>
#include <query_compilation/pipeline_info.hpp>
#include <query_compilation/predicate_expression.hpp>
#include <query_compilation/query_context.hpp>
#include <util/getname.hpp>

#include <boost/make_shared.hpp>
#include <boost/thread.hpp>
#include <boost/thread/lock_guard.hpp>

#include <iostream>

#include <omp.h>

namespace CoGaDB {

PipelineJob::PipelineJob(TablePtr _source_table, size_t block_size,
                         PipelinePtr _pipeline)
    : source_table(_source_table),
      it(new BlockIterator(block_size)),
      pipeline(_pipeline),
      mutex() {}

const TablePtr PipelineJob::getNextBlock(
    const ProcessorSpecification& proc_spec) {
  boost::lock_guard<boost::mutex> lock(mutex);
  TablePtr block;
  if (it->getOffset() < source_table->getNumberofRows()) {
    block = BaseTable::getNext(source_table, it, proc_spec);
    //    std::cout << "Block: " << block->getNumberofRows() << " rows" <<
    //    std::endl;
  }
  return block;
}

size_t PipelineJob::getBlockSize() const { return this->it->getBlockSize(); }

const std::string PipelineJob::toString() const {
  std::stringstream out;
  for (size_t i = 0; i < produced_blocks.size(); ++i) {
    out << "Block " << i << ": " << (void*)produced_blocks[i].get()
        << std::endl;
  }
  return out.str();
}

typedef boost::shared_ptr<GroupbyParam> GroupbyParamPtr;
const GroupbyParamPtr convertGroupbyParamToLegacyGroupByParam(
    GroupByAggregateParamPtr groupby);

void worker_thread(PipelineJobPtr job, PipelinePtr pipeline,
                   TablePtr source_table,
                   const ProcessorSpecification& proc_spec,
                   TablePtr* result_table) {
  TablePtr previous_block;
  TablePtr block;
  TablePtr result;

  block = job->getNextBlock(proc_spec);
  //        GroupByAggregateParamPtr groupby =
  //        pipeline->getPipelineInfo()->getGroupByAggregateParam();
  //        GroupbyParamPtr groupby_param =
  //        convertGroupbyParamToLegacyGroupByParam(groupby);
  //        assert(groupby_param!=NULL);

  while (block) {
    std::map<TablePtr, TablePtr> table_replacement;
    if (previous_block) {
      table_replacement.insert(std::make_pair(previous_block, block));
    } else {
      table_replacement.insert(std::make_pair(source_table, block));
    }
    pipeline->replaceInputTables(table_replacement);

    assert(block != NULL);

    if (!pipeline->execute()) {
      COGADB_FATAL_ERROR("", "");
    }

    if (!result) {
      result = pipeline->getResult();
    } else {
      //                size_t current_size = result->getNumberofRows();
      //                size_t last_intermediate_result_size =
      //                pipeline->getResult()->getNumberofRows();
      //                std::cout << "temporary result: " << std::endl;
      //                pipeline->getResult()->print();
      result->append(pipeline->getResult());
      //                /* if we perform an aggregation in parallel,
      //                 aggregate intermediate result again if it gets too
      //                 large */
      //                if(groupby_param!=NULL
      //                && result->getNumberofRows()>4*job->getBlockSize()){
      //                    result = BaseTable::groupby(result, *groupby_param);
      //                }
    }

    previous_block = block;
    block = job->getNextBlock(proc_spec);
  }

  *result_table = result;
  ocl_api_reset_thread_local_variables();
}

bool isBuildHashTablePipeline(PipelinePtr pipeline) {
  return (pipeline->getPipelineInfo()->getPipelineType() ==
          MATERIALIZE_FROM_ARRAY_TO_JOIN_HASH_TABLE_AND_ARRAY);
}

bool isAggregationPipeline(PipelinePtr pipeline) {
  return (pipeline->getPipelineInfo()->getPipelineType() ==
          MATERIALIZE_FROM_AGGREGATION_HASH_TABLE_TO_ARRAY);
}

bool isMergeAggregationPossible(PipelinePtr pipeline) {
  assert(pipeline != NULL);
  PipelineInfoPtr pipe_info = pipeline->getPipelineInfo();
  assert(pipe_info != NULL);
  GroupByAggregateParamPtr groupby = pipe_info->getGroupByAggregateParam();
  assert(groupby != NULL);

  ProcessorSpecification proc_spec(hype::PD0);
  for (size_t i = 0; i < groupby->aggregation_specs.size(); ++i) {
    AggregationParamPtr agg_param =
        toAggregationParam(groupby->aggregation_specs[i], proc_spec);
    if (!agg_param) {
      return false;
    }
  }
  return true;
}

bool getCombineAggregationFunction(const AggregationFunction& agg_func,
                                   AggregationFunction& combiner_agg_func) {
  if (agg_func == COUNT) {
    combiner_agg_func = SUM;
  } else if (agg_func == SUM || agg_func == MIN || agg_func == MAX) {
    combiner_agg_func = agg_func;
  } else {
    return false;
  }

  return true;
}

AggregationParamPtr toAggregationParam(
    AggregateSpecificationPtr agg_spec,
    const ProcessorSpecification& proc_spec) {
  boost::shared_ptr<AlgebraicAggregateSpecification> alg_agg_spec;
  alg_agg_spec =
      boost::dynamic_pointer_cast<AlgebraicAggregateSpecification>(agg_spec);
  if (!alg_agg_spec) return AggregationParamPtr();

  AggregationFunction combiner_agg_func = SUM;
  if (!getCombineAggregationFunction(alg_agg_spec->agg_func_,
                                     combiner_agg_func)) {
    return AggregationParamPtr();
  }

  AggregationParamPtr result(new AggregationParam(
      proc_spec, combiner_agg_func, HASH_BASED_AGGREGATION,
      alg_agg_spec->result_attr_.getResultAttributeName()));

  return result;
}

const GroupbyParamPtr convertGroupbyParamToLegacyGroupByParam(
    GroupByAggregateParamPtr groupby) {
  if (!groupby) return GroupbyParamPtr();
  ProcessorSpecification proc_spec(hype::PD0);
  GroupingColumns grouping;
  for (size_t i = 0; i < groupby->grouping_attrs.size(); ++i) {
    grouping.push_back(groupby->grouping_attrs[i].getResultAttributeName());
  }
  AggregationFunctions aggs;
  for (size_t i = 0; i < groupby->aggregation_specs.size(); ++i) {
    AggregationParamPtr agg_param =
        toAggregationParam(groupby->aggregation_specs[i], proc_spec);
    assert(agg_param);

    aggs.push_back(std::make_pair(agg_param->new_column_name, *agg_param));
  }

  GroupbyParamPtr groupby_param =
      boost::make_shared<GroupbyParam>(proc_spec, grouping, aggs);
  return groupby_param;
}

const TablePtr parallel_execute(PipelinePtr pipeline, QueryContextPtr context) {
  Timestamp begin = getTimestamp();
  /* add compile time, execution time is still zero and will be added at the end
   * of this function */
  context->updateStatistics(pipeline);

  size_t num_threads = VariableManager::instance().getVariableValueInteger(
      "code_gen.num_threads");
  std::cout << "Num Threads: " << num_threads << std::endl;
  // boost::thread::hardware_concurrency();
  //  size_t num_threads = 2;
  size_t block_size = VariableManager::instance().getVariableValueInteger(
      "code_gen.block_size");

  ProcessorSpecification proc_spec(hype::PD0);

  PipelineInfoPtr pipe_info = pipeline->getPipelineInfo();
  assert(pipe_info != NULL);
  TablePtr source_table = pipe_info->getSourceTable();
  //    pipeline->

  PipelineJobPtr job(new PipelineJob(source_table, block_size, pipeline));

  std::vector<PipelinePtr> pipelines(num_threads);
  std::vector<TablePtr> result_tables(num_threads);

  omp_set_dynamic(0);                // Explicitly disable dynamic teams
  omp_set_num_threads(num_threads);  // Use specified number of threads

  boost::thread_group threads;
#pragma omp parallel for
  for (size_t i = 0; i < num_threads; ++i) {
    pipelines[i] = pipeline->copy();
    TablePtr* tab_ptr = &result_tables[i];
    worker_thread(job, pipelines[i], source_table, proc_spec, tab_ptr);
  }
  threads.join_all();

  TablePtr result;
  /* it may happen that a worker thread did not get a chunk to process, because
   the table is too small. So we need to find a valid result table first. */
  for (size_t i = 0; i < num_threads; ++i) {
    if (result_tables[i]) {
      result = result_tables[i];
      break;
    }
  }
  assert(result != NULL);
  for (size_t i = 0; i < num_threads; ++i) {
    if (result_tables[i] != NULL && result_tables[i] != result) {
      result->append(result_tables[i]);
    }
  }
  GroupByAggregateParamPtr groupby = pipe_info->getGroupByAggregateParam();
  if (groupby) {
    GroupbyParamPtr groupby_param =
        convertGroupbyParamToLegacyGroupByParam(groupby);
    assert(groupby_param != NULL);
    result = BaseTable::groupby(result, *groupby_param);
    assert(result != NULL);
    result->setName("");
  }
  Timestamp end = getTimestamp();

  double total_execution_time = double(end - begin) / (1000 * 1000 * 1000);
  context->addExecutionTime(total_execution_time);

  return result;
}

const TablePtr serial_execute(PipelinePtr pipeline, QueryContextPtr context) {
  if (!pipeline->execute()) {
    COGADB_FATAL_ERROR("Pipeline Execution Failed!", "");
  }
  context->updateStatistics(pipeline);
  ocl_api_reset_thread_local_variables();
  return pipeline->getResult();
}

const TablePtr execute_intern(PipelinePtr pipeline, QueryContextPtr context) {
  if (!VariableManager::instance().getVariableValueBoolean(
          "enable_parallel_pipelines") ||
      isDummyPipeline(pipeline)) {
    return serial_execute(pipeline, context);
  }
  if (isAggregationPipeline(pipeline)) {
    //        if(containsOnlyAlgebraicAggregation(pipeline)
    //                && !containsUDFAggregation(pipeline)
    //                && !containsAverageAggregation(pipeline)){
    if (isMergeAggregationPossible(pipeline)) {
      return parallel_execute(pipeline, context);
    } else {
      /* Either we cannot build a merge pipeline or we cannot
       split the aggregation because at least one is holistic.
       Thus we execute the pipeline serially. */
      return serial_execute(pipeline, context);
    }
  } else if (isBuildHashTablePipeline(pipeline)) {
    /* we do not yet support parallel build of shared hash table! */
    return serial_execute(pipeline, context);
  } else {
    return parallel_execute(pipeline, context);
  }
}

const TablePtr execute(PipelinePtr pipeline, QueryContextPtr context) {
  TablePtr result = execute_intern(pipeline, context);
  return result;
}

}  // end namespace CoGaDB
