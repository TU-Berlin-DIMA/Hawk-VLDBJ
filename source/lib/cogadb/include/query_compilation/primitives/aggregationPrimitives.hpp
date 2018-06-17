/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   aggregatePrimitives.hpp
 * Author: dbronesk
 *
 * Created on February 6, 2016, 5:43 PM
 */

#ifndef AGGREGATEPRIMITIVES_HPP
#define AGGREGATEPRIMITIVES_HPP

#include <query_compilation/aggregate_specification.hpp>
#include <query_compilation/primitives/instruction.hpp>

namespace CoGaDB {

  class IncrementResultTupleCounter : public Instruction {
   public:
    IncrementResultTupleCounter();
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;
    bool supportsPredication() const override final;
  };

  class Aggregation : public Instruction {
   public:
    Aggregation(AggregateSpecificationPtr aggregateSpecification);
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

    bool supportsPredication() const override final;
    void setPredicationMode(const PredicationMode& pred_mode) override final;
    PredicationMode getPredicationMode() const override final;

    AggregateSpecificationPtr getAggregateSpecifications() const {
      return agg_spec_;
    }

   private:
    AggregateSpecificationPtr agg_spec_;
  };

  class BitPackedGroupingKey : public Instruction {
   public:
    BitPackedGroupingKey(const GroupingAttributes& groupingAttributes);
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;
    bool supportsPredication() const override final;

   private:
    GroupingAttributes group_attr_;
  };

  class GenericGroupingKey : public Instruction {
   public:
    GenericGroupingKey(const GroupingAttributes& groupingAttributes);
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

   private:
    GroupingAttributes group_attr_;
  };

  class HashGroupAggregate : public Instruction {
   public:
    HashGroupAggregate(const GroupingAttributes& groupingAttributes,
                       const AggregateSpecifications& aggregateSpecifications,
                       const ProjectionParam& projection_param);
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;

    GroupingAttributes getGroupingAttributes() const { return grouping_attrs_; }

    AggregateSpecifications getAggregateSpecifications() const {
      return aggr_specs_;
    }

    ProjectionParam getProjectionParam() const { return projection_param_; }

    bool supportsPredication() const override final;
    void setPredicationMode(const PredicationMode& pred_mode) override final;
    PredicationMode getPredicationMode() const override final;

   private:
    const std::string getAggregationCode(
        const GroupingAttributes& grouping_columns,
        const AggregateSpecifications& aggregation_specs,
        const std::string access_ht_entry_expression);

    GroupingAttributes grouping_attrs_;
    AggregateSpecifications aggr_specs_;
    ProjectionParam projection_param_;
  };
}

#endif /* AGGREGATEPRIMITIVES_HPP */
