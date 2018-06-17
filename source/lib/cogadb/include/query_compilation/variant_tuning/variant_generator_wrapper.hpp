/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   variant_generator_wrapper.hpp
 * Author: dbronesk
 *
 * Created on March 8, 2016, 3:23 PM
 */
#pragma once

#include <atomic>
#include <perseus/Configuration.hpp>
#include <perseus/DiscreteFeature.hpp>
#include <perseus/ExecutionContext.hpp>
#include <perseus/Variant.hpp>
#include <perseus/VariantGenerator.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/code_generators/multi_stage_code_generator.hpp>
#include <query_compilation/pipeline.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/definitions.hpp>
#include <util/variant_configurator.hpp>

#include <core/variable_manager.hpp>

using perseus::ExecutionContext;
using perseus::VariantPool;
using perseus::Configuration;
using perseus::VariantGenerator;
using perseus::Feature;
using StringFeature = perseus::DiscreteFeature<std::string>;

namespace CoGaDB {

  class PipelineVariant;

  class PipelineExecutionContext : public ExecutionContext {
   public:
    PipelineExecutionContext(const unsigned long long numConsumedTuples,
                             perseus::VariantPool& pool,
                             PipelineVariant& pipeline_variant)
        : numConsumedTuples_(numConsumedTuples),
          pool_(pool),
          pipeline_variant_(pipeline_variant) {}

    virtual VariantPool& variantPool() const { return pool_; }

    const unsigned long long getConsumedTuples() const {
      return numConsumedTuples_;
    }

    PipelineVariant& getPipelineVariant() const { return pipeline_variant_; }

    void setResult(TablePtr result) { result_ = result; }

    TablePtr getResult() { return result_; }

   private:
    const unsigned long long numConsumedTuples_;
    VariantPool& pool_;
    PipelineVariant& pipeline_variant_;
    TablePtr result_;
  };

  class PipelineVariant : public perseus::Variant {
   private:
    const std::string name_;
    std::unique_ptr<Configuration> configuration_;
    PipelinePtr pipeline_;
    std::atomic<unsigned long long> totalRuntime_{0};
    std::atomic<unsigned long long> totalTuples_{0};
    std::atomic<unsigned long long> totalCalls_{0};
    std::atomic<double> currentRuntimePerTuple_{0};

   public:
    PipelineVariant(const std::string& name,
                    std::unique_ptr<Configuration> configuration,
                    PipelinePtr pipeline)
        : name_(name),
          configuration_(std::move(configuration)),
          pipeline_(pipeline) {}

    virtual void invoke(ExecutionContext* context) override {
      PipelineExecutionContext* pipeline_execution_context =
          static_cast<PipelineExecutionContext*>(context);
      pipeline_->execute();
      pipeline_execution_context->setResult(pipeline_->getResult());

      // CoGaDB returns a double (for seconds)
      // Perseus does not care about resolution but needs integers
      // -> MicroSeconds should be sufficiently precise
      totalRuntime_ += pipeline_->getExecutionTimeSec() * 1000 * 1000;
      totalTuples_ += pipeline_execution_context->getConsumedTuples();
      totalCalls_++;
    }

    virtual const unsigned long long totalRuntime() const override {
      return totalRuntime_;
    }
    virtual const unsigned long long totalTuples() const override {
      return totalTuples_;
    }
    virtual const unsigned long totalCalls() const override {
      return totalCalls_;
    }
    virtual const double currentRuntimePerTuple() const override {
      return currentRuntimePerTuple_;
    }
    virtual void setCurrentRuntimePerTuple(
        double currentRuntimePerTuple) override {
      currentRuntimePerTuple_ = currentRuntimePerTuple;
    }
    virtual const std::string name() const override { return name_; }
    virtual const Configuration& configuration() const override {
      return *configuration_;
    }
    virtual void reset() override {}

    virtual void waitForLastCall() override {}

    bool replaceInputTables(
        const std::map<TablePtr, TablePtr>& table_replacement) {
      return pipeline_->replaceInputTables(table_replacement);
    }

    const PipelinePtr getPipeline() const { return pipeline_; }
  };

  class MultiStageCodeGeneratorWrapper : public VariantGenerator {
   private:
    query_processing::LogicalQueryPlan& logical_query_plan_;
    std::vector<Variant> variants_;
    std::vector<std::unique_ptr<Feature>> features_;

   public:
    MultiStageCodeGeneratorWrapper(
        query_processing::LogicalQueryPlan& logical_query_plan,
        VariantIterator variant_iterator)
        : logical_query_plan_(logical_query_plan),
          variants_(materializeVariants(variant_iterator)),
          features_(createFeatures()) {}

    virtual ~MultiStageCodeGeneratorWrapper() {}

    virtual std::unique_ptr<perseus::Variant> createVariant(
        std::unique_ptr<Configuration> configuration) const {
      auto variant = mapConfiguration(*configuration);
      setGlobalVariantConfiguration(variant);
      auto pipeline = createPipeline(logical_query_plan_);
      return std::unique_ptr<PipelineVariant>(
          new PipelineVariant(name(), std::move(configuration), pipeline));
    }

    virtual const std::vector<Feature*> features() const {
      return convertUniquePtrElementsToRawPointers(features_);
    }

    virtual const std::string name() const {
      return "MultiStageCodeGeneratorWrapper";
    }

    virtual const bool validateConfiguration(
        const Configuration& configuration) const {
      auto variant = mapConfiguration(configuration);
      return std::find(variants_.begin(), variants_.end(), variant) !=
             variants_.end();
    }

   private:
    void setGlobalVariantConfiguration(const Variant& variant) const {
      VariantConfigurator variant_configurator;
      variant_configurator(variant);
    }

    PipelinePtr createPipeline(
        query_processing::LogicalQueryPlan& logical_query_plan) const {
      ProjectionParam projection_parameters;
      auto code_generator = createCodeGenerator(MULTI_STAGE_CODE_GENERATOR,
                                                projection_parameters);
      auto context = createQueryContext();
      logical_query_plan.getRoot()->produce(code_generator, context);
      auto pipeline = code_generator->compile();
      return pipeline;
    }

    std::vector<std::unique_ptr<Feature>> createFeatures() {
      std::map<std::string, std::set<std::string>> dimensionMap;
      for (auto& variant : variants_) {
        for (auto& feature : variant) {
          auto& name = feature.first;
          auto& value = feature.second;
          dimensionMap[name].insert(value);
        }
      }
      std::vector<std::unique_ptr<Feature>> features;
      for (auto& dimension : dimensionMap) {
        auto& name = dimension.first;
        auto values = std::vector<std::string>(dimension.second.begin(),
                                               dimension.second.end());
        auto feature =
            std::unique_ptr<Feature>(new StringFeature(name, values));
        features.push_back(std::move(feature));
      }
      return features;
    }

    const Variant mapConfiguration(const Configuration& configuration) const {
      Variant variant;
      for (auto feature : configuration.features()) {
        auto name = feature->name();
        if (name == "chunk_size") {
          continue;
        }
        variant[name] = configuration.getValue<std::string>(name);
      }
      return variant;
    }

    const std::vector<Variant> materializeVariants(
        VariantIterator& variant_iterator) {
      std::vector<Variant> variants;
      for (auto& variant : variant_iterator) {
        variants_.push_back(variant);
      }
      return variants;
    }
  };
}
