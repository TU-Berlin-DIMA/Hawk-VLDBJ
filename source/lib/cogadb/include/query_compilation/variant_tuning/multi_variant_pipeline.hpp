#ifndef COGADB_MULTI_VARIANT_PIPELINE_HPP
#define COGADB_MULTI_VARIANT_PIPELINE_HPP

#include <perseus/DefaultVariantPool.hpp>
#include <perseus/RandomizedInitializationStrategy.hpp>
#include <perseus/VWGreedySelectionStrategy.hpp>
#include <query_compilation/pipeline.hpp>
#include <query_processing/definitions.hpp>
#include <util/variant_configurator.hpp>

namespace CoGaDB {
  class PipelineExecutionContext;

  class CodeGenerator;

  typedef boost::shared_ptr<CodeGenerator> CodeGeneratorPtr;

  class MultiVariantPipeline : public Pipeline {
   private:
    query_processing::LogicalQueryPlan& logical_query_plan_;
    VariantIterator variant_iterator_;
    std::unique_ptr<perseus::VariantPool> variant_pool_;
    std::unique_ptr<PipelineExecutionContext> executionContext_;

   public:
    MultiVariantPipeline(query_processing::LogicalQueryPlan& logical_query_plan,
                         VariantIterator variant_iterator);

    ~MultiVariantPipeline();

    virtual double getCompileTimeSec() const { return 0; }

    virtual const PipelinePtr copy() const;

    virtual bool replaceInputTables(
        const std::map<TablePtr, TablePtr>& table_replacement);

    // Each copy has an independent variant pool. Not sure, if
    // this is the right semantics.  Otherwise, we should use a
    // shared_ptr. Also, each copy is initialized with an empty
    // execution context. This should be the right semantics
    // because the execution context is created by
    // replaceInputTables which should be local for every
    // pipeline.
    MultiVariantPipeline(const MultiVariantPipeline& other);

   protected:
    virtual TablePtr execute_impl();

   private:
    const TablePtr executeSelectedVariant(const ScanParam& scan_param,
                                          StatePtr state);

    std::unique_ptr<perseus::VariantPool> createVariantPool(
        VariantIterator variant_iterator);

    void initializeMembersFromFirstVariant();

    const Pipeline& getFirstPipeline() const;
  };
  typedef boost::shared_ptr<MultiVariantPipeline> MultiVariantPipelinePtr;
}

#endif  // COGADB_MULTI_VARIANT_PIPELINE_HPP
