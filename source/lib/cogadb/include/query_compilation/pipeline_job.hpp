/*
 * File:   pipeline_job.hpp
 * Author: sebastian
 *
 * Created on 30. Dezember 2015, 14:55
 */

#ifndef PIPELINE_JOB_HPP
#define PIPELINE_JOB_HPP

#include <boost/shared_ptr.hpp>
#include <boost/thread/mutex.hpp>
#include <core/global_definitions.hpp>

namespace CoGaDB {

  typedef AggregationMethod AggregationFunction;

  class PipelineJob;
  typedef boost::shared_ptr<PipelineJob> PipelineJobPtr;

  class BlockIterator;
  typedef boost::shared_ptr<BlockIterator> BlockIteratorPtr;

  class Pipeline;
  typedef boost::shared_ptr<Pipeline> PipelinePtr;

  class BaseTable;
  typedef boost::shared_ptr<BaseTable> TablePtr;

  class AggregationParam;
  typedef boost::shared_ptr<AggregationParam> AggregationParamPtr;

  struct AggregateSpecification;
  typedef boost::shared_ptr<AggregateSpecification> AggregateSpecificationPtr;

  class ProcessorSpecification;

  class PipelineJob {
   public:
    PipelineJob(TablePtr source_table, size_t block_size,
                PipelinePtr _pipeline);
    const TablePtr getNextBlock(const ProcessorSpecification& proc_spec);
    size_t getBlockSize() const;
    const std::string toString() const;

   private:
    TablePtr source_table;
    BlockIteratorPtr it;
    PipelinePtr pipeline;
    boost::mutex mutex;
    std::vector<TablePtr> produced_blocks;
  };

  bool isBuildHashTablePipeline(PipelinePtr pipeline);

  bool isAggregationPipeline(PipelinePtr pipeline);

  bool getCombineAggregationFunction(const AggregationFunction& agg_func,
                                     AggregationFunction& combiner_agg_func);

  AggregationParamPtr toAggregationParam(
      AggregateSpecificationPtr agg_spec,
      const ProcessorSpecification& proc_spec);

}  // end namespace CoGaDB

#endif /* PIPELINE_JOB_HPP */
