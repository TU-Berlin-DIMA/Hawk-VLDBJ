/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   aggregatePrimitives.cpp
 * Author: dbronesk
 *
 * Created on February 6, 2016, 5:43 PM
 */

#include <query_compilation/primitives/aggregationPrimitives.hpp>

#include "query_compilation/code_generators/c_code_generator.hpp"
#include "query_compilation/code_generators/code_generator_utils.hpp"

namespace CoGaDB {

IncrementResultTupleCounter::IncrementResultTupleCounter()
    : Instruction(INCREMENT_INSTR) {}

const GeneratedCodePtr IncrementResultTupleCounter::getCode(
    CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  code->after_for_loop_code_block_ << "++current_result_size;" << std::endl;
  return code;
}

const std::string IncrementResultTupleCounter::toString() const {
  return "INCREMENT_RESULT_TUPLE_COUNTER";
}

std::map<std::string, std::string>
IncrementResultTupleCounter::getInputVariables(CodeGenerationTarget target) {
  return {};
}

std::map<std::string, std::string>
IncrementResultTupleCounter::getOutputVariables(CodeGenerationTarget target) {
  return {};
}

bool IncrementResultTupleCounter::supportsPredication() const { return true; }

Aggregation::Aggregation(AggregateSpecificationPtr aggregateSpecification)
    : Instruction(AGGREGATE_INSTR), agg_spec_(aggregateSpecification) {}

const GeneratedCodePtr Aggregation::getCode(CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());

  code->declare_variables_code_block_
      << agg_spec_->getCodeAggregationDeclareVariables(target) << std::endl;
  code->init_variables_code_block_
      << agg_spec_->getCodeAggregationInitializeVariables(target) << std::endl;
  code->upper_code_block_.push_back(
      agg_spec_->getCodeAggregationComputation(target));
  code->after_for_loop_code_block_
      << agg_spec_->getCodeAggregationWriteResult(target) << std::endl;

  return code;
}

const std::string Aggregation::toString() const {
  std::stringstream str;
  str << "AGGREGATE(" << agg_spec_->toString() << ")";
  return str.str();
}

std::map<std::string, std::string> Aggregation::getInputVariables(
    CodeGenerationTarget target) {
  return agg_spec_->getInputVariables(target);
}

std::map<std::string, std::string> Aggregation::getOutputVariables(
    CodeGenerationTarget target) {
  return agg_spec_->getOutputVariables(target);
}

bool Aggregation::supportsPredication() const {
  return agg_spec_->supportsPredication();
}
void Aggregation::setPredicationMode(const PredicationMode& pred_mode) {
  return agg_spec_->setPredicationMode(pred_mode);
}

PredicationMode Aggregation::getPredicationMode() const {
  return agg_spec_->getPredicationMode();
}

BitPackedGroupingKey::BitPackedGroupingKey(
    const GroupingAttributes& groupingAttributes)
    : Instruction(BITPACKED_GROUPING_KEY_INSTR),
      group_attr_(groupingAttributes) {}

const GeneratedCodePtr BitPackedGroupingKey::getCode(CodeGenerationTarget) {
  GeneratedCodePtr code(new GeneratedCode());
  std::stringstream hash_aggregate;
  hash_aggregate << "TID group_key = "
                 << getComputeGroupIDExpression(group_attr_) << ";"
                 << std::endl;
  code->upper_code_block_.push_back(hash_aggregate.str());
  return code;
}

const std::string BitPackedGroupingKey::toString() const {
  return "AGGREGATE_BITPACKED";
}

std::map<std::string, std::string> BitPackedGroupingKey::getInputVariables(
    CodeGenerationTarget target) {
  return getVariableNameAndTypeFromInputAttributeRefVector(group_attr_);
}

std::map<std::string, std::string> BitPackedGroupingKey::getOutputVariables(
    CodeGenerationTarget target) {
  return {};
}

bool BitPackedGroupingKey::supportsPredication() const {
  /* predication will not change anything in this instruction */
  return true;
}

GenericGroupingKey::GenericGroupingKey(
    const GroupingAttributes& groupingAttributes)
    : Instruction(HASH_AGGREGATE_INSTR), group_attr_(groupingAttributes) {}

const GeneratedCodePtr GenericGroupingKey::getCode(
    CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  std::stringstream lower;
  std::stringstream compute_group_id;
  compute_group_id << "TID group_key = getGroupKey(" << group_attr_.size()
                   << ", ";

  for (size_t col_id = 0; col_id < group_attr_.size(); ++col_id) {
    compute_group_id << group_attr_[col_id].getAttributeType() << ", "
                     << getCompressedElementAccessExpression(
                            group_attr_[col_id]);

    if (col_id + 1 < group_attr_.size()) {
      compute_group_id << ", ";
    }
  }

  compute_group_id << ");" << std::endl;

  code->upper_code_block_.push_back(compute_group_id.str());
  return code;
}

const std::string GenericGroupingKey::toString() const {
  return "GENERIC_GROUPBY";
}

std::map<std::string, std::string> GenericGroupingKey::getInputVariables(
    CodeGenerationTarget target) {
  return getVariableNameAndTypeFromInputAttributeRefVector(group_attr_);
}

std::map<std::string, std::string> GenericGroupingKey::getOutputVariables(
    CodeGenerationTarget target) {
  return {};
}

HashGroupAggregate::HashGroupAggregate(
    const GroupingAttributes& groupingAttributes,
    const AggregateSpecifications& aggregateSpecifications,
    const ProjectionParam& projection_param)
    : Instruction(HASH_AGGREGATE_INSTR),
      grouping_attrs_(groupingAttributes),
      aggr_specs_(aggregateSpecifications),
      projection_param_(projection_param) {}

const std::string HashGroupAggregate::getAggregationCode(
    const GroupingAttributes& grouping_columns,
    const AggregateSpecifications& aggregation_specs,
    const std::string access_ht_entry_expression) {
  return getAggregationCodeGeneric(grouping_columns, aggregation_specs,
                                   access_ht_entry_expression);
}

const GeneratedCodePtr HashGroupAggregate::getCode(
    CodeGenerationTarget target) {
  GeneratedCodePtr code(new GeneratedCode());
  /* generate struct that serves as payload based on aggregation functions */
  code->header_and_types_code_block_ << "struct AggregationPayload {"
                                     << std::endl;

  for (size_t i = 0; i < grouping_attrs_.size(); ++i) {
    AttributeReference attr_ref = grouping_attrs_[i];
    if (!isComputed(attr_ref)) {
      code->header_and_types_code_block_
          << getAggregationGroupTIDPayloadFieldCode(attr_ref) << std::endl;
    } else {
      code->header_and_types_code_block_ << toCType(attr_ref.getAttributeType())
                                         << " " << getVarName(attr_ref) << ";"
                                         << std::endl;
    }
  }

  std::set<AggregateSpecification::AggregationPayloadField> struct_fields;

  for (size_t i = 0; i < aggr_specs_.size(); ++i) {
    std::vector<AggregateSpecification::AggregationPayloadField>
        payload_fields = aggr_specs_[i]->getAggregationPayloadFields();
    struct_fields.insert(payload_fields.begin(), payload_fields.end());
  }

  std::set<AggregateSpecification::AggregationPayloadField>::const_iterator cit;
  for (cit = struct_fields.begin(); cit != struct_fields.end(); ++cit) {
    code->header_and_types_code_block_ << *cit << ";" << std::endl;
  }

  code->header_and_types_code_block_ << "};" << std::endl;
  code->header_and_types_code_block_
      << "typedef struct AggregationPayload AggregationPayload;" << std::endl;

  /* create hash table */
  code->declare_variables_code_block_
      << "C_AggregationHashTable* aggregation_hash_table = "
         "createAggregationHashTable(sizeof(AggregationPayload));"
      << std::endl;
  code->clean_up_code_block_
      << "freeAggregationHashTable(aggregation_hash_table);" << std::endl;

  code->declare_variables_code_block_
      << getCodeDeclareResultMemory(projection_param_) << std::endl;
  code->init_variables_code_block_
      << getCodeMallocResultMemory(projection_param_) << std::endl;

  /* determine the grouping key according to grouping columns*/
  std::stringstream hash_aggregate;
  /* do the usual hash table probe, aggregate using the custom payload for
   * each aggregation function. This performs only one lookup per tuple. */
  hash_aggregate << "AggregationPayload* aggregation_payload = "
                    "(AggregationPayload*)getAggregationHashTablePayload("
                    "aggregation_hash_table, group_key);"
                 << std::endl;
  hash_aggregate << "if(aggregation_payload){" << std::endl;
  hash_aggregate << getAggregationCode(grouping_attrs_, aggr_specs_,
                                       "aggregation_payload->")
                 << std::endl;
  hash_aggregate << "} else {" << std::endl;
  hash_aggregate << "AggregationPayload payload;" << std::endl;
  /* init payload fields */
  for (size_t i = 0; i < aggr_specs_.size(); ++i) {
    hash_aggregate << aggr_specs_[i]->getCodeInitializeAggregationPayloadFields(
        "payload.");
  }
  /* insert new key and payload in hash table */
  hash_aggregate << "aggregation_payload = "
                    "(AggregationPayload*)insertAggregationHashTable("
                    "aggregation_hash_table, group_key, &payload);"
                 << std::endl;
  hash_aggregate << getAggregationCode(grouping_attrs_, aggr_specs_,
                                       "aggregation_payload->")
                 << std::endl;
  hash_aggregate << "}" << std::endl;

  code->upper_code_block_.push_back(hash_aggregate.str());

  /* write result from hash table to output arrays */
  code->after_for_loop_code_block_
      << "if (getAggregationHashTableSize(aggregation_hash_table) >= "
         "allocated_result_elements) {"
      << std::endl;
  code->after_for_loop_code_block_ << "    allocated_result_elements = "
                                      "getAggregationHashTableSize(aggregation_"
                                      "hash_table);"
                                   << std::endl;
  code->after_for_loop_code_block_
      << "    " << getCodeReallocResultMemory(projection_param_, true)
      << std::endl;
  code->after_for_loop_code_block_ << "}" << std::endl;

  code->after_for_loop_code_block_
      << "C_AggregationHashTableIterator* aggr_itr = "
         "createAggregationHashTableIterator(aggregation_hash_table);"
      << std::endl;
  code->after_for_loop_code_block_
      << "for (; hasNextAggregationHashTableIterator(aggr_itr); "
         "nextAggregationHashTableIterator(aggr_itr)) {"
      << std::endl;

  for (size_t i = 0; i < aggr_specs_.size(); ++i) {
    code->after_for_loop_code_block_
        << aggr_specs_[i]->getCodeFetchResultsFromHashTableEntry(
            "((AggregationPayload*)getAggregationHashTableIteratorPayload(aggr_"
            "itr))->");
  }

  for (size_t i = 0; i < grouping_attrs_.size(); ++i) {
    code->after_for_loop_code_block_
        << "    " << getResultArrayVarName(grouping_attrs_[i])
        << "[current_result_size] = ";
    if (!isComputed(grouping_attrs_[i])) {
      code->after_for_loop_code_block_ << getCompressedElementAccessExpression(
          grouping_attrs_[i],
          "((AggregationPayload*)getAggregationHashTableIteratorPayload("
          "aggr_itr))->" +
              getGroupTIDVarName(grouping_attrs_[i]));
    } else {
      /* code path grouping attributes that were computed */
      code->after_for_loop_code_block_
          << "((AggregationPayload*)getAggregationHashTableIteratorPayload("
             "aggr_itr))->" +
                 getVarName(grouping_attrs_[i]);
    }
    code->after_for_loop_code_block_ << ";" << std::endl;
  }

  code->after_for_loop_code_block_ << "    ++current_result_size;" << std::endl;
  code->after_for_loop_code_block_ << "}" << std::endl;
  code->after_for_loop_code_block_
      << "freeAggregationHashTableIterator(aggr_itr);" << std::endl;

  return code;
}

const std::string HashGroupAggregate::toString() const {
  std::stringstream str;
  str << "HASH_GROUP_AGGREGATE("
      << CoGaDB::toString(grouping_attrs_, aggr_specs_) << ")";
  return str.str();
}

std::map<std::string, std::string> HashGroupAggregate::getInputVariables(
    CodeGenerationTarget target) {
  std::map<std::string, std::string> result;

  for (const auto& aggr_spec : aggr_specs_) {
    auto vars = aggr_spec->getInputVariables(target);

    for (const auto& var : vars) {
      result.insert(var);
    }
  }

  return result;
}

std::map<std::string, std::string> HashGroupAggregate::getOutputVariables(
    CodeGenerationTarget target) {
  COGADB_NOT_IMPLEMENTED;
}

bool HashGroupAggregate::supportsPredication() const {
  AggregateSpecifications::iterator it;
  bool ret = true;
  for (size_t i = 0; i < aggr_specs_.size(); ++i) {
    ret = ret && aggr_specs_[i]->supportsPredication();
  }
  return ret;
}
void HashGroupAggregate::setPredicationMode(const PredicationMode& pred_mode) {
  if (supportsPredication()) {
    for (size_t i = 0; i < aggr_specs_.size(); ++i) {
      aggr_specs_[i]->setPredicationMode(pred_mode);
    }
  }
}

PredicationMode HashGroupAggregate::getPredicationMode() const {
  PredicationMode pred_mode = aggr_specs_[0]->getPredicationMode();
  for (size_t i = 1; i < aggr_specs_.size(); ++i) {
    if (aggr_specs_[i]->getPredicationMode() != pred_mode) {
      COGADB_FATAL_ERROR(
          "Inconsistent Aggregation Instruction detected: " << this->toString(),
          "");
    }
  }
  return pred_mode;
}
}
