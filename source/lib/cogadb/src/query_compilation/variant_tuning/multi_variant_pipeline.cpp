#include <perseus/FullPoolBaselineStrategy.hpp>
#include <perseus/TotalRuntimeVariantScorer.hpp>
#include <query_compilation/pipeline_info.hpp>
#include <query_compilation/variant_tuning/multi_variant_pipeline.hpp>
#include <query_compilation/variant_tuning/variant_generator_wrapper.hpp>

using perseus::VariantGenerator;
using perseus::InitializationStrategy;
using perseus::DefaultVariantPool;
using perseus::SelectionStrategy;
using perseus::RandomizedInitializationStrategy;
using perseus::VWGreedySelectionStrategy;
using perseus::FullPoolBaselineStrategy;
using perseus::VariantScorer;
using perseus::TotalRuntimeVariantScorer;

namespace CoGaDB {

MultiVariantPipeline::MultiVariantPipeline(
    query_processing::LogicalQueryPlan& logical_query_plan,
    VariantIterator variant_iterator)
    : Pipeline(0, ScanParam(), PipelineInfoPtr()),
      logical_query_plan_(logical_query_plan),
      variant_iterator_(variant_iterator),
      variant_pool_(createVariantPool(variant_iterator)) {
  initializeMembersFromFirstVariant();
}

MultiVariantPipeline::~MultiVariantPipeline() = default;

MultiVariantPipeline::MultiVariantPipeline(const MultiVariantPipeline& other)
    : MultiVariantPipeline(other.logical_query_plan_, other.variant_iterator_) {
}

void MultiVariantPipeline::initializeMembersFromFirstVariant() {
  auto& pipeline = getFirstPipeline();
  mPipeInfo = pipeline.getPipelineInfo();
  mScanParam = pipeline.getScannedAttributes();
}

const Pipeline& MultiVariantPipeline::getFirstPipeline() const {
  auto variants = variant_pool_->variants();
  assert(variants.size() > 0);
  auto variant = variants[0];
  auto pipelineVariant = dynamic_cast<PipelineVariant*>(variant);
  auto pipeline = pipelineVariant->getPipeline();
  return *pipeline;
}

TablePtr MultiVariantPipeline::execute_impl() {
  return executeSelectedVariant(mScanParam, mPipeInfo->getGlobalState());
}

const PipelinePtr MultiVariantPipeline::copy() const {
  MultiVariantPipeline* copy = new MultiVariantPipeline(*this);
  return PipelinePtr(copy);
}

bool MultiVariantPipeline::replaceInputTables(
    const std::map<TablePtr, TablePtr>& table_replacement) {
  assert(table_replacement.size() == 1);
  for (perseus::Variant* variant : variant_pool_->variants()) {
    PipelineVariant* pipeline_variant = static_cast<PipelineVariant*>(variant);
    if (!pipeline_variant->replaceInputTables(table_replacement)) {
      return false;
    }
  }
  auto& variant = variant_pool_->getVariant();
  auto& pipeline_variant = static_cast<PipelineVariant&>(variant);
  executionContext_.reset(new PipelineExecutionContext(
      table_replacement.begin()->second->getNumberofRows(), *variant_pool_,
      pipeline_variant));
  return true;
}

std::unique_ptr<VariantPool> MultiVariantPipeline::createVariantPool(
    VariantIterator variant_iterator) {
  auto poolSize = VariableManager::instance().getVariableValueInteger(
      "perseus.minimal_pool_size");
  auto elitism =
      VariableManager::instance().getVariableValueInteger("perseus.elitism");
  auto explore_period = VariableManager::instance().getVariableValueInteger(
      "perseus.explore_period");
  auto explore_length = VariableManager::instance().getVariableValueInteger(
      "perseus.explore_length");
  auto exploit_period = VariableManager::instance().getVariableValueInteger(
      "perseus.exploit_period");
  auto skip_length = VariableManager::instance().getVariableValueInteger(
      "perseus.skip_length");
  auto generator_wrapper =
      std::unique_ptr<VariantGenerator>(new MultiStageCodeGeneratorWrapper(
          logical_query_plan_, variant_iterator));
  auto initializationStrategy = std::unique_ptr<InitializationStrategy>(
      new RandomizedInitializationStrategy);
  std::shared_ptr<VariantScorer> variantScorer =
      std::make_shared<TotalRuntimeVariantScorer>();
  auto strategy =
      std::make_shared<FullPoolBaselineStrategy>(variantScorer, elitism);
  auto selectionStrategy =
      std::unique_ptr<SelectionStrategy>(new VWGreedySelectionStrategy(
          explore_period, explore_length, exploit_period, skip_length));
  auto chunkSize = 0u;
  auto pool = std::unique_ptr<VariantPool>(new DefaultVariantPool(
      poolSize, 1.0, poolSize, chunkSize, std::move(generator_wrapper),
      strategy, std::move(selectionStrategy), strategy));
  pool->initialize();
  return pool;
}

const TablePtr MultiVariantPipeline::executeSelectedVariant(
    const ScanParam& scan_param, StatePtr state) {
  executionContext_->getPipelineVariant().invoke(executionContext_.get());
  return executionContext_->getResult();
}
}