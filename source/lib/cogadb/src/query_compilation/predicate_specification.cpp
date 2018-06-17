
#include <core/attribute_reference.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/predicate_specification.hpp>
#include <util/getname.hpp>
#include <util/iostream.hpp>
#include <util/types.hpp>

#include <compression/dictionary_compressed_column.hpp>
#include <compression/order_preserving_dictionary_compressed_column.hpp>

#include <util/attribute_reference_handling.hpp>
#include "core/vector.hpp"

namespace CoGaDB {

PredicateSpecification::PredicateSpecification(
    PredicateSpecificationType pred_t, const AttributeReferencePtr left_attr,
    const AttributeReferencePtr right_attr, ValueComparator comp)
    : pred_t_(pred_t),
      left_attr_(left_attr),
      right_attr_(right_attr),
      constant_(),
      reg_ex_(),
      comp_(comp) {}

PredicateSpecification::PredicateSpecification(
    PredicateSpecificationType pred_t, const AttributeReferencePtr left_attr,
    const boost::any& right_constant, ValueComparator comp)
    : pred_t_(pred_t),
      left_attr_(left_attr),
      right_attr_(),
      constant_(right_constant),
      reg_ex_(),
      comp_(comp) {}

//        PredicateSpecification::PredicateSpecification(PredicateSpecificationType
//        pred_t,
//                const boost::any& left_constant,
//                const AttributeReferencePtr right_attr,
//                ValueComparator comp)
//        : pred_t_(pred_t), left_attr_(), right_attr_(right_attr),
//        constant_(left_constant), reg_ex_(), comp_(comp)
//        {
//
//        }

PredicateSpecification::PredicateSpecification(
    PredicateSpecificationType pred_t, const AttributeReferencePtr left_attr,
    const std::string& regular_expression, ValueComparator comp)
    : pred_t_(pred_t),
      left_attr_(left_attr),
      right_attr_(),
      constant_(),
      reg_ex_(regular_expression),
      comp_(comp) {}

// we expect SSE 4.2

const std::pair<std::string, std::string>
PredicateSpecification::getSSEExpression(uint32_t& pred_num) const {
  std::stringstream expr;
  std::stringstream header;
  if (comp_ == UNEQUAL) {
    COGADB_FATAL_ERROR("UNEQUAL currently not implemented in SIMD...", "");
  }

  if (this->pred_t_ == ValueValuePredicateSpec) {
    //                COGADB_FATAL_ERROR("No Code Generation for Column Column
    //                Expressions yet!","");
    expr << ::CoGaDB::getSSEExpression(comp_) << "(_mm_load_si128( &((__m128i*)"
         << getElementAccessExpression(*left_attr_) << ")), "

         << "(_mm_load_si128( &((__m128i*)"
         << getElementAccessExpression(*right_attr_) << ")) )";
  } else if (this->pred_t_ == ValueConstantPredicateSpec) {
    if (constant_.type() == typeid(std::string) &&
        this->left_attr_->getAttributeType() == VARCHAR) {
      /* we filter a string column, check whether we can exploit compression */
      if (getColumnType(*this->left_attr_) == DICTIONARY_COMPRESSED) {
        /* we have unordered dictionary compression, so we can use
         it only for equality and inequality predicates */
        if (comp_ == EQUAL) {
          /* retrieve the compressed value and use an integer filter */

          /* WARNING: this call fetches the column from disk,
           * as we need to access the dictionary! */
          ColumnPtr col = left_attr_->getColumn();
          uint32_t filter_id;
          ValueComparator rewritten_value_comparator;
          if (!getDictionaryIDForPredicate(
                  col, comp_, boost::any_cast<std::string>(constant_),
                  filter_id, rewritten_value_comparator)) {
            COGADB_FATAL_ERROR(
                "Failed to retrieve compressed key from dictionary!", "");
          }

          header << "__m128i " << getSSEVarName(pred_num) << "= _mm_set1_epi32("
                 << filter_id << ");\n";
          /* WARNING: this call fetches the column from disk,
           * as we need to access the dictionary! */
          expr << CoGaDB::getSSEExpression(comp_)
               << "(_mm_load_si128( &((__m128i*)"
               << getCompressedElementAccessExpression(*left_attr_) << ")), "
               << getSSEVarName(pred_num) << ")";

        } else {
          COGADB_FATAL_ERROR("Tried to compare strings in SIMD...", "");
        }

      } else if (getColumnType(*this->left_attr_) ==
                 DICTIONARY_COMPRESSED_ORDER_PRESERVING) {
        /* WARNING: this call fetches the column from disk,
         * as we need to access the dictionary! */
        ColumnPtr col = left_attr_->getColumn();
        uint32_t filter_id;
        ValueComparator rewritten_value_comparator;
        if (getDictionaryIDForPredicate(
                col, comp_, boost::any_cast<std::string>(constant_), filter_id,
                rewritten_value_comparator)) {
          header << "__m128i " << getSSEVarName(pred_num) << "= _mm_set1_epi32("
                 << filter_id << ");\n";

          expr << CoGaDB::getSSEExpression(comp_) << "(_mm_load_si128( &("
               << getCompressedElementAccessExpressionSIMD(*left_attr_,
                                                           "(__m128i*)")
               << ")), " << getSSEVarName(pred_num) << ")";
        }
      } else {
        COGADB_FATAL_ERROR("Tried to compare strings in SIMD!", "");
      }

      /* check special case of a DATE comparison value (passed as string value)
       * to filter a column of type DATE
       */
    } else if (constant_.type() == typeid(std::string) &&
               this->left_attr_->getAttributeType() == DATE) {
      uint32_t val = 0;
      if (!convertStringToInternalDateType(
              boost::any_cast<std::string>(constant_), val)) {
        COGADB_FATAL_ERROR("The string '"
                               << boost::any_cast<std::string>(constant_)
                               << "' is not representing a DATE!" << std::endl
                               << "Typecast Failed!",
                           "");
      }
      header << "__m128i " << getSSEVarName(pred_num) << "= _mm_set1_epi32("
             << val << ");\n";
      expr << CoGaDB::getSSEExpression(comp_) << "(_mm_load_si128( &("
           << getCompressedElementAccessExpressionSIMD(*left_attr_,
                                                       "(__m128i*)")
           << ")), " << getSSEVarName(pred_num) << ")";

    } else if (this->left_attr_->getAttributeType() == FLOAT) {
      /* special care for float attributes, because we need to
       * generate float constants for them (by default floating point
       * numbers are double in C++ and comparing the same value in a
       * float and a double results in incorrect results) */

      header << "__m128 " << getSSEVarName(pred_num) << "= _mm_set1_ps(";

      /* note that we might have a float column, but the predicate is
       always converted into a double value first! */
      if (constant_.type() == typeid(float)) {
        header << ::CoGaDB::getConstant(constant_) << ");";

      } else if (constant_.type() == typeid(double)) {
        double f = boost::any_cast<double>(constant_);
        if (ceil(f) == f) {
          /* is integer, add ".0f" as suffix */
          header << ::CoGaDB::getConstant(constant_) << ".0f);";

        } else {
          /* is not integer, add "f" as suffix */
          header << ::CoGaDB::getConstant(constant_) << "f);";
        }
      } else if (constant_.type() == typeid(int32_t)) {
        int32_t i = boost::any_cast<int32_t>(constant_);
        float f = boost::numeric_cast<float>(i);
        if (ceil(f) == f) {
          /* is integer, add ".0f" as suffix */
          header << ::CoGaDB::getConstant(constant_) << ".0f);";
        } else {
          /* is not integer, add "f" as suffix */
          header << ::CoGaDB::getConstant(constant_) << "f);";
        }
      }

      expr << CoGaDB::getSSEExpressionFloat(comp_) << "(_mm_load_ps((float*)&("
           << getElementAccessExpressionSIMD(*left_attr_, "(__m128*)") << ")), "
           << getSSEVarName(pred_num) << ")";

    } else {
      header << "__m128i " << getSSEVarName(pred_num) << "= _mm_set1_epi32("
             << ::CoGaDB::getConstant(constant_) << ");\n";
      expr << CoGaDB::getSSEExpression(comp_) << "(_mm_load_si128( &("
           << getElementAccessExpressionSIMD(*left_attr_, "(__m128i*)")
           << ")), " << getSSEVarName(pred_num) << ")";
    }

  } else if (this->pred_t_ == ValueRegularExpressionPredicateSpec) {
    COGADB_FATAL_ERROR("No Code Generation for Regular Expressions yet!", "");
  } else {
    COGADB_FATAL_ERROR("Invalid PredicateSpecificationType!", "");
  }
  return std::pair<std::string, std::string>(header.str(), expr.str());
}

const std::string PredicateSpecification::getCPPExpression() const {
  std::stringstream expr;

  if (this->pred_t_ == ValueValuePredicateSpec) {
    if (left_attr_->getAttributeType() == VARCHAR &&
        right_attr_->getAttributeType() == VARCHAR) {
      expr << getStringCompareExpression(*left_attr_, *right_attr_, comp_);
    } else {
      expr << getElementAccessExpression(*left_attr_) << " "
           << ::CoGaDB::getExpression(comp_) << " "
           << getElementAccessExpression(*right_attr_);
    }
  } else if (this->pred_t_ == ValueConstantPredicateSpec) {
    if (constant_.type() == typeid(std::string) &&
        this->left_attr_->getAttributeType() == VARCHAR) {
      expr << getStringCompareExpression(
          *left_attr_, boost::any_cast<std::string>(constant_), comp_);
      /* check special case of a DATE comparison value (passed as string value)
      * to filter a column of type DATE
      */
    } else if (constant_.type() == typeid(std::string) &&
               this->left_attr_->getAttributeType() == DATE) {
      uint32_t val = 0;
      if (!convertStringToInternalDateType(
              boost::any_cast<std::string>(constant_), val)) {
        COGADB_FATAL_ERROR("The string '"
                               << boost::any_cast<std::string>(constant_)
                               << "' is not representing a DATE!" << std::endl
                               << "Typecast Failed!",
                           "");
      }
      boost::any new_constant(val);
      expr << getElementAccessExpression(*left_attr_) << " "
           << CoGaDB::getExpression(comp_) << " "
           << CoGaDB::getConstant(new_constant);
    } else if (this->left_attr_->getAttributeType() == FLOAT) {
      /* special care for float attributes, because we need to
       * generate float constants for them (by default floating point
       * numbers are double in C++ and comparing the same value in a
       * float and a double results in incorrect results) */

      /* note that we might have a float column, but the predicate is
       always converted into a double value first! */
      if (constant_.type() == typeid(float)) {
        expr << getElementAccessExpression(*left_attr_) << " "
             << CoGaDB::getExpression(comp_) << " "
             << CoGaDB::getConstant(constant_);
      } else if (constant_.type() == typeid(double)) {
        double f = boost::any_cast<double>(constant_);
        if (ceil(f) == f) {
          /* is integer, add ".0f" as suffix */
          expr << getElementAccessExpression(*left_attr_) << " "
               << CoGaDB::getExpression(comp_) << " "
               << CoGaDB::getConstant(constant_) << ".0f";
        } else {
          /* is not integer, add "f" as suffix */
          expr << getElementAccessExpression(*left_attr_) << " "
               << CoGaDB::getExpression(comp_) << " "
               << CoGaDB::getConstant(constant_) << "f";
        }
      } else if (constant_.type() == typeid(int32_t)) {
        expr << getElementAccessExpression(*left_attr_) << " "
             << CoGaDB::getExpression(comp_) << " ";
        int32_t i = boost::any_cast<int32_t>(constant_);
        float f = boost::numeric_cast<float>(i);
        if (ceil(f) == f) {
          /* is integer, add ".0f" as suffix */
          expr << i << ".0f";
        } else {
          /* is not integer, add "f" as suffix */
          expr << f << "f";
        }
      } else {
        COGADB_FATAL_ERROR("Unhandled Type: " << constant_.type().name(), "");
      }
    } else {
      expr << getElementAccessExpression(*left_attr_) << " "
           << CoGaDB::getExpression(comp_) << " "
           << ::CoGaDB::getConstant(constant_);
    }

  } else if (this->pred_t_ == ValueRegularExpressionPredicateSpec) {
    COGADB_FATAL_ERROR("No Code Generation for Regular Expressions yet!", "");
  } else {
    COGADB_FATAL_ERROR("Invalid PredicateSpecificationType!", "");
  }
  return expr.str();
}

const std::vector<AttributeReferencePtr>
PredicateSpecification::getScannedAttributes() const {
  std::vector<AttributeReferencePtr> ret;

  if (this->pred_t_ == ValueValuePredicateSpec) {
    ret.push_back(left_attr_);
    ret.push_back(right_attr_);
  } else if (this->pred_t_ == ValueConstantPredicateSpec) {
    ret.push_back(left_attr_);
  } else if (this->pred_t_ == ValueRegularExpressionPredicateSpec) {
    ret.push_back(left_attr_);
  } else {
    COGADB_FATAL_ERROR("Invalid PredicateSpecificationType!", "");
  }
  return ret;
}

const std::vector<AttributeReferencePtr>
PredicateSpecification::getColumnsToDecompress() const {
  std::vector<AttributeReferencePtr> ret;

  if ((left_attr_ && left_attr_->getAttributeType() != VARCHAR) ||
      (right_attr_ && right_attr_->getAttributeType() != VARCHAR)) {
    return ret;
  }

  if (pred_t_ == ValueValuePredicateSpec) {
    // if both columns aren't compressed in the same way or if we got dictionary
    // compression and the comparison isn't EQUAL or UNEQUAL
    if (getColumnType(*left_attr_) != getColumnType(*right_attr_) ||
        (getColumnType(*left_attr_) == DICTIONARY_COMPRESSED &&
         comp_ != EQUAL && comp_ != UNEQUAL)) {
      ret.push_back(left_attr_);
      ret.push_back(right_attr_);
    }
  } else if (pred_t_ == ValueConstantPredicateSpec) {
    if (constant_.type() != typeid(std::string) ||
        (getColumnType(*left_attr_) == DICTIONARY_COMPRESSED &&
         comp_ != EQUAL && comp_ != UNEQUAL)) {
      ret.push_back(left_attr_);
    }
  }

  return ret;
}

PredicateSpecificationType
PredicateSpecification::getPredicateSpecificationType() const {
  return this->pred_t_;
}

const AttributeReferencePtr PredicateSpecification::getLeftAttribute() const {
  assert(this->pred_t_ == ValueValuePredicateSpec ||
         this->pred_t_ == ValueConstantPredicateSpec);
  return this->left_attr_;
}

const AttributeReferencePtr PredicateSpecification::getRightAttribute() const {
  assert(this->pred_t_ == ValueValuePredicateSpec);
  return this->right_attr_;
}

const boost::any& PredicateSpecification::getConstant() const {
  //            assert(this->pred_t_==ValueLeftConstantRightPredicateSpec
  //                  || this->pred_t_==ConstantLeftValueRightPredicateSpec);
  return this->constant_;
}

const std::string& PredicateSpecification::getRegularExpression() const {
  assert(this->pred_t_ == ValueRegularExpressionPredicateSpec);
  return this->reg_ex_;
}

ValueComparator PredicateSpecification::getValueComparator() const {
  return this->comp_;
}

void PredicateSpecification::print() const {
  std::cout << this->toString() << std::endl;
}

const std::string PredicateSpecification::toString() const {
  std::stringstream expr;
  if (this->pred_t_ == ValueValuePredicateSpec) {
    expr << CoGaDB::toString(*left_attr_) << " " << util::getName(comp_) << " "
         << CoGaDB::toString(*right_attr_);
  } else if (this->pred_t_ == ValueConstantPredicateSpec) {
    expr << CoGaDB::toString(*left_attr_) << " " << util::getName(comp_) << " "
         << constant_;
  } else if (this->pred_t_ == ValueRegularExpressionPredicateSpec) {
    expr << CoGaDB::toString(*left_attr_);
    if (this->comp_ == EQUAL) {
      expr << " LIKE ";
    } else if (this->comp_ == UNEQUAL) {
      expr << " NOT LIKE ";
    } else {
      COGADB_FATAL_ERROR(
          "Detected invalid parameter combination! ValueComparator may only be "
          "EQUAL or UNEQUAL for ValueRegularExpressionPredicates!",
          "");
    }
    expr << util::getName(comp_);
    expr << constant_;
  } else {
    COGADB_FATAL_ERROR("Invalid PredicateType!", "");
  }

  return expr.str();
}

/*! \brief inverts the order of Value Value Predicates
            e.g., Attr1>Attr2 -> Attr2<Attr1
            this is especially useful for join path optimization*/
void PredicateSpecification::invertOrder() {
  if (this->pred_t_ == ValueValuePredicateSpec) {
    std::swap(this->left_attr_, this->right_attr_);
    if (this->comp_ == LESSER) {
      this->comp_ = GREATER;
    } else if (this->comp_ == GREATER) {
      this->comp_ = LESSER;
    } else if (this->comp_ == LESSER_EQUAL) {
      this->comp_ = GREATER_EQUAL;
    } else if (this->comp_ == GREATER_EQUAL) {
      this->comp_ = LESSER_EQUAL;
    }
  }
}

void PredicateSpecification::replaceTablePointerInAttributeReferences(
    const ScanParam& scanned_attributes) {
  if (this->left_attr_)
    CoGaDB::replaceAttributeTablePointersWithScannedAttributeTablePointers(
        scanned_attributes, *this->left_attr_);
  if (this->right_attr_)
    CoGaDB::replaceAttributeTablePointersWithScannedAttributeTablePointers(
        scanned_attributes, *this->right_attr_);
}

PredicateCombination::PredicateCombination(
    const std::vector<PredicateExpressionPtr>& predicate_expressions,
    LogicalOperation log_op)
    : predicate_expressions_(predicate_expressions), log_op_(log_op) {}

const std::string PredicateCombination::getCPPExpression() const {
  std::stringstream expr;

  expr << "(";
  for (size_t i = 0; i < predicate_expressions_.size(); ++i) {
    expr << predicate_expressions_[i]->getCPPExpression();
    if (i + 1 < predicate_expressions_.size()) {
      expr << " " << ::CoGaDB::getExpression(log_op_) << " ";
    }
  }
  expr << ")";
  return expr.str();
}

const std::pair<std::string, std::string>
PredicateCombination::getSSEExpression(uint32_t& pred_num) const {
  std::pair<std::string, std::string> temp;
  std::stringstream header;
  std::stringstream expr;
  if (predicate_expressions_.size() > 1) {
    expr << ::CoGaDB::getSSEExpression(log_op_) << "(";
  }
  temp = (predicate_expressions_[0])->getSSEExpression(pred_num);
  header << temp.first;
  expr << temp.second;
  for (size_t i = 1; i < predicate_expressions_.size(); ++i) {
    expr << ", ";
    if (i + 1 < predicate_expressions_.size()) {
      expr << ::CoGaDB::getSSEExpression(log_op_) << "(";
    }
    pred_num++;
    temp = (predicate_expressions_[i])->getSSEExpression(pred_num);
    header << temp.first;
    expr << temp.second;
  }

  for (size_t i = 0; i < predicate_expressions_.size() - 1; ++i) {
    expr << ")";
  }
  return std::pair<std::string, std::string>(header.str(), expr.str());
}
/*

    const std::vector<boost::any> PredicateCombination::getConstants() const {
        //            assert(this->pred_t_==ValueLeftConstantRightPredicateSpec
        //                  ||
   this->pred_t_==ConstantLeftValueRightPredicateSpec);
        std::vector<boost::any> result;
        for (size_t i = 0; i < predicate_expressions_.size(); ++i) {
            result.push_back((predicate_expressions_[i]->getConstants())[0]);
        }
        return result;
    }
*/
const std::vector<AttributeReferencePtr>
PredicateCombination::getScannedAttributes() const {
  std::vector<AttributeReferencePtr> ret;

  for (size_t i = 0; i < predicate_expressions_.size(); ++i) {
    std::vector<AttributeReferencePtr> tmp =
        predicate_expressions_[i]->getScannedAttributes();
    ret.insert(ret.end(), tmp.begin(), tmp.end());
  }

  return ret;
}

const std::vector<PredicateExpressionPtr>
PredicateCombination::getPredicateExpression() const {
  return predicate_expressions_;
}

const std::string PredicateCombination::toString() const {
  std::stringstream expr;

  expr << "(";
  for (size_t i = 0; i < predicate_expressions_.size(); ++i) {
    expr << predicate_expressions_[i]->toString();
    if (i + 1 < predicate_expressions_.size()) {
      expr << " " << util::getName(log_op_) << " ";
    }
  }
  expr << ")";
  return expr.str();
}

void PredicateCombination::replaceTablePointerInAttributeReferences(
    const ScanParam& scanned_attributes) {
  for (auto pred_expr : predicate_expressions_) {
    pred_expr->replaceTablePointerInAttributeReferences(scanned_attributes);
  }
}

const std::vector<AttributeReferencePtr>
PredicateCombination::getColumnsToDecompress() const {
  std::vector<AttributeReferencePtr> result, tmp;

  for (std::vector<PredicateExpressionPtr>::const_iterator itr(
           predicate_expressions_.begin());
       itr != predicate_expressions_.end(); ++itr) {
    tmp = (*itr)->getColumnsToDecompress();

    if (!tmp.empty()) {
      result.insert(result.end(), tmp.begin(), tmp.end());
    }
  }

  return result;
}

}  // end namespace CoGaDB
