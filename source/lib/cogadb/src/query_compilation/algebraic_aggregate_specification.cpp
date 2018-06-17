
#include <boost/make_shared.hpp>
#include <query_compilation/algebraic_aggregate_specification.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <util/attribute_reference_handling.hpp>
#include <util/getname.hpp>

namespace CoGaDB {

AlgebraicAggregateSpecification::AlgebraicAggregateSpecification(
    const AttributeReference& _scan_attr,
    const AttributeReference& _result_attr,
    const AggregationFunction& _agg_func)
    : AggregateSpecification(),
      scan_attr_(_scan_attr),
      result_attr_(_result_attr),
      agg_func_(_agg_func) {}

const std::vector<AttributeReferencePtr>
AlgebraicAggregateSpecification::getScannedAttributes() const {
  std::vector<AttributeReferencePtr> result;
  result.push_back(boost::make_shared<AttributeReference>(scan_attr_));
  return result;
}

const std::vector<AttributeReferencePtr>
AlgebraicAggregateSpecification::getComputedAttributes() const {
  std::vector<AttributeReferencePtr> result;
  result.push_back(boost::make_shared<AttributeReference>(result_attr_));
  return result;
}

std::map<std::string, std::string>
AlgebraicAggregateSpecification::getInputVariables(
    CodeGenerationTarget target) {
  std::string pointer = "*";

  if (scan_attr_.getAttributeReferenceType() == COMPUTED_ATTRIBUTE) {
    return std::map<std::string, std::string>();
  }
  return {{getVarName(scan_attr_),
           toCType(scan_attr_.getAttributeType()) + pointer}};
}

std::map<std::string, std::string>
AlgebraicAggregateSpecification::getOutputVariables(
    CodeGenerationTarget target) {
  if (agg_func_ == AVERAGE) {
    return {{getAggregationPayloadFieldVarName(result_attr_, COUNT),
             getAggregationResultCType(result_attr_, COUNT)},
            {getAggregationPayloadFieldVarName(result_attr_, SUM),
             getAggregationResultCType(result_attr_, SUM)}};

  } else {
    return {{getAggregationPayloadFieldVarName(result_attr_, agg_func_),
             getAggregationResultCType(result_attr_, agg_func_)}};
  }
}

void AlgebraicAggregateSpecification::replaceTablePointerInAttributeReferences(
    const ScanParam& scanned_attributes) {
  CoGaDB::replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, this->scan_attr_);
}

const std::string AlgebraicAggregateSpecification::getCodeHashGroupBy(
    const std::string& access_ht_entry_expression) {
  std::stringstream hash_aggregate;
  AttributeReference attr_ref = this->result_attr_;
  if (this->agg_func_ == MIN) {
    assert(getPredicationMode() == BRANCHED_EXECUTION);
    hash_aggregate << getMinFunction() << "_"
                   << getAggregationResultCType(attr_ref, this->agg_func_)
                   << "(" << access_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << "," << getElementAccessExpression(this->scan_attr_)
                   << ");" << std::endl;
  } else if (this->agg_func_ == MAX) {
    hash_aggregate << getMaxFunction() << "_"
                   << getAggregationResultCType(attr_ref, this->agg_func_)
                   << "(" << access_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << ",";
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      hash_aggregate << getElementAccessExpression(this->scan_attr_) << ");";
    } else {
      hash_aggregate << getElementAccessExpression(this->scan_attr_)
                     << "*result_increment);";
    }
  } else if (this->agg_func_ == COUNT) {
    hash_aggregate << "C_SUM"
                   << "_"
                   << getAggregationResultCType(attr_ref, this->agg_func_)
                   << "(" << access_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << ", ";
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      hash_aggregate << " 1);";
    } else {
      hash_aggregate << " result_increment);";
    }
  } else if (this->agg_func_ == SUM) {
    hash_aggregate << "C_SUM"
                   << "_"
                   << getAggregationResultCType(attr_ref, this->agg_func_)
                   << "(" << access_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << ", ";
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      hash_aggregate << getElementAccessExpression(scan_attr_) << ");";
    } else {
      hash_aggregate << "(" << getElementAccessExpression(scan_attr_)
                     << " * result_increment));";
    }
  } else if (this->agg_func_ == AVERAGE) {
    hash_aggregate << "C_SUM"
                   << "_" << getAggregationResultCType(attr_ref, COUNT) << "("
                   << access_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref, COUNT)
                   << ", ";
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      hash_aggregate << " 1);";
    } else {
      hash_aggregate << " result_increment);";
    }

    hash_aggregate << "C_SUM"
                   << "_" << getAggregationResultCType(attr_ref, SUM) << "("
                   << access_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref, SUM) << ", ";
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      hash_aggregate << getElementAccessExpression(scan_attr_) << ");";
    } else {
      hash_aggregate << "(" << getElementAccessExpression(scan_attr_)
                     << " * result_increment));";
    }
  } else {
    COGADB_FATAL_ERROR(
        "Unsupported Aggregation Function: " << util::getName(this->agg_func_),
        "");
  }
  return hash_aggregate.str();
}

const std::string AlgebraicAggregateSpecification::getCodeCopyHashTableEntry(
    const std::string& access_dst_ht_entry_expression,
    const std::string& access_src_ht_entry_expression) {
  std::stringstream hash_aggregate;
  AttributeReference attr_ref = this->result_attr_;
  if (this->agg_func_ == MIN) {
    assert(getPredicationMode() == BRANCHED_EXECUTION);
    hash_aggregate << getMinFunction() << "_"
                   << getAggregationResultCType(attr_ref, this->agg_func_)
                   << "(" << access_dst_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << "," << access_src_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << ");" << std::endl;
  } else if (this->agg_func_ == MAX) {
    hash_aggregate << getMaxFunction() << "_"
                   << getAggregationResultCType(attr_ref, this->agg_func_)
                   << "(" << access_dst_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << "," << access_src_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << ");" << std::endl;
  } else if (this->agg_func_ == COUNT) {
    hash_aggregate << "C_SUM"
                   << "_"
                   << getAggregationResultCType(attr_ref, this->agg_func_)
                   << "(" << access_dst_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << ", " << access_src_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref, agg_func_)
                   << ");" << std::endl;
  } else if (this->agg_func_ == SUM) {
    hash_aggregate << "C_SUM"
                   << "_"
                   << getAggregationResultCType(attr_ref, this->agg_func_)
                   << "(" << access_dst_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref,
                                                        this->agg_func_)
                   << ", " << access_src_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref, agg_func_)
                   << ");" << std::endl;
  } else if (this->agg_func_ == AVERAGE) {
    hash_aggregate << "C_SUM"
                   << "_" << getAggregationResultCType(attr_ref, COUNT) << "("
                   << access_dst_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref, COUNT) << ", "
                   << access_src_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref, COUNT) << ");"
                   << std::endl;

    hash_aggregate << "C_SUM"
                   << "_" << getAggregationResultCType(attr_ref, SUM) << "("
                   << access_dst_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref, SUM) << ", "
                   << access_src_ht_entry_expression
                   << getAggregationPayloadFieldVarName(attr_ref, SUM) << ");"
                   << std::endl;
  } else {
    COGADB_FATAL_ERROR(
        "Unsupported Aggregation Function: " << util::getName(this->agg_func_),
        "");
  }

  return hash_aggregate.str();
}

const std::vector<AggregateSpecification::AggregationPayloadField>
AlgebraicAggregateSpecification::getAggregationPayloadFields() {
  std::vector<AggregationPayloadField> result;
  if (this->agg_func_ == AVERAGE) {
    result.push_back(getAggregationPayloadFieldCode(result_attr_, COUNT));
    result.push_back(getAggregationPayloadFieldCode(result_attr_, SUM));
  } else {
    result.push_back(getAggregationPayloadFieldCode(result_attr_, agg_func_));
  }
  return result;
}

const std::string
AlgebraicAggregateSpecification::getCodeInitializeAggregationPayloadFields(
    const std::string& access_ht_entry_expression) {
  std::stringstream initialization_code;

  /* handle special case of AVERAGE aggregation, this needs sum and count */
  if (this->agg_func_ == AVERAGE) {
    initialization_code << access_ht_entry_expression
                        << getAggregationPayloadFieldVarName(result_attr_,
                                                             COUNT)
                        << " = 0;" << std::endl;
    initialization_code << access_ht_entry_expression
                        << getAggregationPayloadFieldVarName(result_attr_, SUM)
                        << " = 0;" << std::endl;
  } else {
    initialization_code << access_ht_entry_expression
                        << getAggregationPayloadFieldVarName(result_attr_,
                                                             agg_func_)
                        << " = 0;" << std::endl;
  }

  return initialization_code.str();
}

const std::string
AlgebraicAggregateSpecification::getCodeFetchResultsFromHashTableEntry(
    const std::string& access_ht_entry_expression) {
  std::stringstream write_result_expr;
  if (this->agg_func_ == AVERAGE) {
    write_result_expr
        << getResultArrayVarName(this->result_attr_)
        << "[current_result_size] = " << access_ht_entry_expression
        << getAggregationPayloadFieldVarName(this->result_attr_, SUM) << "/"
        << access_ht_entry_expression
        << getAggregationPayloadFieldVarName(this->result_attr_, COUNT) << ";";
  } else {
    write_result_expr << getResultArrayVarName(this->result_attr_)
                      << "[current_result_size] = "
                      << access_ht_entry_expression
                      << getAggregationPayloadFieldVarName(this->result_attr_,
                                                           this->agg_func_)
                      << ";";
  }
  return write_result_expr.str();
}

const std::string
AlgebraicAggregateSpecification::getCodeAggregationDeclareVariables(
    CodeGenerationTarget target) {
  std::stringstream declare_var_expr;
  declare_var_expr << getAggregationPayloadFieldCode(result_attr_, agg_func_)
                   << std::endl;
  declare_var_expr << getCodeDeclareResultMemory(result_attr_);
  return declare_var_expr.str();
}

const std::string
AlgebraicAggregateSpecification::getCodeAggregationInitializeVariables(
    CodeGenerationTarget target) {
  std::stringstream init_var_expr;

  init_var_expr << getAggregationPayloadFieldVarName(result_attr_, agg_func_)
                << " = 0;" << std::endl;

  if (agg_func_ == AVERAGE) {
    init_var_expr << getAggregationPayloadFieldVarName(result_attr_, COUNT)
                  << " = 0;" << std::endl;
  }

  init_var_expr << getCodeMallocResultMemory(result_attr_);

  return init_var_expr.str();
}

const std::string
AlgebraicAggregateSpecification::getCodeAggregationComputation(
    CodeGenerationTarget target) {
  std::stringstream compute_expr;
  AttributeReference computed_attr = result_attr_;

  std::string access = "";

  if (target == OCL_TARGET_CODE) {
    access = "[get_global_id(0)]";
  }

  if (this->agg_func_ == COUNT) {
    compute_expr << getAggregationPayloadFieldVarName(computed_attr, COUNT)
                 << access;
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      compute_expr << "++;";
    } else {
      compute_expr << "+=result_increment;";
    }
  } else if (this->agg_func_ == SUM) {
    compute_expr << getAggregationPayloadFieldVarName(computed_attr, agg_func_)
                 << access;
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      compute_expr << " += " << getElementAccessExpression(scan_attr_) << ";";
    } else {
      compute_expr << " += (" << getElementAccessExpression(scan_attr_)
                   << "*result_increment);";
    }
  } else if (this->agg_func_ == AVERAGE) {
    compute_expr << getAggregationPayloadFieldVarName(computed_attr, agg_func_)
                 << access;
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      compute_expr << " += " << getElementAccessExpression(scan_attr_) << ";";
    } else {
      compute_expr << " += (" << getElementAccessExpression(scan_attr_)
                   << "*result_increment);";
    }
    compute_expr << getAggregationPayloadFieldVarName(computed_attr, COUNT)
                 << access;
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      compute_expr << "++;";
    } else {
      compute_expr << "+=result_increment;";
    }
  } else if (this->agg_func_ == MIN) {
    /* \todo: think of a way to support minimum in presense of predication */
    compute_expr << getAggregationPayloadFieldVarName(computed_attr, agg_func_)
                 << access << " = " << getMinFunction() << "("
                 << getAggregationPayloadFieldVarName(computed_attr, agg_func_)
                 << access << "," << getElementAccessExpression(scan_attr_)
                 << ");";
  } else if (this->agg_func_ == MAX) {
    compute_expr << getAggregationPayloadFieldVarName(computed_attr, agg_func_)
                 << access << " = " << getMaxFunction() << "("
                 << getAggregationPayloadFieldVarName(computed_attr, agg_func_)
                 << access << "," << getElementAccessExpression(scan_attr_);
    if (getPredicationMode() == BRANCHED_EXECUTION) {
      compute_expr << ");";
    } else {
      compute_expr << "*result_increment);";
    }
  } else {
    COGADB_FATAL_ERROR("Unknown Aggregation Function!", "");
  }

  return compute_expr.str();
}

const std::string
AlgebraicAggregateSpecification::getCodeAggregationWriteResult(
    CodeGenerationTarget target) {
  std::stringstream write_result_expr;

  if (this->agg_func_ == AVERAGE) {
    write_result_expr << getResultArrayVarName(result_attr_)
                      << "[current_result_size] = "
                      << getAggregationPayloadFieldVarName(result_attr_,
                                                           AVERAGE)
                      << " / "
                      << getAggregationPayloadFieldVarName(result_attr_, COUNT)
                      << ";";
  } else {
    write_result_expr << getResultArrayVarName(result_attr_)
                      << "[current_result_size] = "
                      << getAggregationPayloadFieldVarName(result_attr_,
                                                           agg_func_)
                      << ";";
  }

  return write_result_expr.str();
}

const std::string AlgebraicAggregateSpecification::toString() const {
  std::stringstream result;
  result << util::getName(agg_func_) << "(" << CoGaDB::toString(scan_attr_)
         << ")"
         << " AS "
         << result_attr_
                .getResultAttributeName();  // CoGaDB::toString(result_attr);
  return result.str();
}

const AggregationFunctionType
AlgebraicAggregateSpecification::getAggregationFunctionType() const {
  return ALGEBRAIC;
}

AggregationFunction AlgebraicAggregateSpecification::getAggregationFunction()
    const {
  return agg_func_;
}

bool AlgebraicAggregateSpecification::supportsPredication() const {
  /* predicating minimum computation currently not supported! */
  if (MIN != agg_func_) {
    return true;
  } else {
    return false;
  }
}

}  // end namespace CoGaDB
