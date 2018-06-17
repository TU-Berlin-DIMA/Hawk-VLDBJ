
#include <core/attribute_reference.hpp>
#include <core/selection_expression.hpp>
#include <iostream>
#include <sstream>
#include <util/getname.hpp>
#include <util/iostream.hpp>

namespace CoGaDB {
using namespace std;

Predicate::Predicate(const std::string& column1_name,
                     const std::string& column2_name, PredicateType pred_t,
                     ValueComparator comp)
    : pred_t_(pred_t),
      column1_name_(column1_name),
      column2_name_(column2_name),
      constant_(),
      comp_(comp) {
  assert(pred_t == ValueValuePredicate);
  column1_name_ = convertToFullyQualifiedNameIfRequired(column1_name_);
  column2_name_ = convertToFullyQualifiedNameIfRequired(column2_name_);
}

Predicate::Predicate(const std::string& column1_name,
                     const boost::any& constant, PredicateType pred_t,
                     ValueComparator comp)
    : pred_t_(pred_t),
      column1_name_(column1_name),
      column2_name_(),
      constant_(constant),
      comp_(comp) {
  assert(pred_t == ValueConstantPredicate ||
         pred_t == ValueRegularExpressionPredicate);
  column1_name_ = convertToFullyQualifiedNameIfRequired(column1_name_);
}

PredicateType Predicate::getPredicateType() const { return this->pred_t_; }
const std::string& Predicate::getColumn1Name() const {
  return this->column1_name_;
}
const std::string& Predicate::getColumn2Name() const {
  return this->column2_name_;
}
const boost::any& Predicate::getConstant() const { return this->constant_; }
ValueComparator Predicate::getValueComparator() const { return this->comp_; }
void Predicate::print() const { std::cout << this->toString() << std::endl; }
std::string Predicate::toString() const {
  std::string result;

  if (pred_t_ == ValueValuePredicate) {
    result += column1_name_;
    result += util::getName(comp_);
    result += column2_name_;
  } else if (pred_t_ == ValueConstantPredicate) {
    // result+=
    result += column1_name_;
    result += util::getName(comp_);
    std::stringstream ss;
    ss << constant_;
    result += ss.str();
    // result+=")";
  } else if (pred_t_ == ValueRegularExpressionPredicate) {
    result += column1_name_;
    if (this->comp_ == EQUAL) {
      result += " LIKE ";
    } else if (this->comp_ == UNEQUAL) {
      result += " NOT LIKE ";
    } else {
      COGADB_FATAL_ERROR(
          "Detected invalid parameter combination! ValueComparator may only be "
          "EQUAL or UNEQUAL for ValueRegularExpressionPredicates!",
          "");
    }
    result += util::getName(comp_);
    std::stringstream ss;
    ss << constant_;
    result += ss.str();
  } else {
    COGADB_FATAL_ERROR("Invalid PredicateType!", "");
  }

  return result;
}

void Predicate::invertOrder() {
  // enum ValueComparator{LESSER,GREATER,EQUAL,LESSER_EQUAL,GREATER_EQUAL};

  if (this->pred_t_ == ValueValuePredicate) {
    std::swap(this->column1_name_, this->column2_name_);
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

bool Predicate::operator<(const Predicate& p) const {
  if (this->column1_name_ < p.column1_name_) return true;
  if (this->column1_name_ > p.column1_name_) return false;
  if (this->column2_name_ < p.column2_name_) return true;
  if (this->column2_name_ > p.column2_name_) return false;
  if (this->pred_t_ < p.pred_t_) return true;
  if (this->comp_ < p.comp_) return true;
  return false;
}
}