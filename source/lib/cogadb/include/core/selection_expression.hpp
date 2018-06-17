#pragma once

#include <assert.h>
#include <boost/any.hpp>
#include <core/global_definitions.hpp>
#include <string>
#include <vector>

namespace CoGaDB {

  class Predicate {
   public:
    Predicate(const std::string& column1_name, const std::string& column2_name,
              PredicateType pred_t, ValueComparator comp);
    Predicate(const std::string& column1_name, const boost::any& constant,
              PredicateType pred_t, ValueComparator comp);
    PredicateType getPredicateType() const;
    const std::string& getColumn1Name() const;
    const std::string& getColumn2Name() const;
    const boost::any& getConstant() const;
    ValueComparator getValueComparator() const;
    void print() const;
    std::string toString() const;
    /*! \brief inverts the order of Value Value Predicates
                e.g., Attr1>Attr2 -> Attr2<Attr1
                this is especially useful for join path optimization*/
    void invertOrder();
    bool operator<(const Predicate& p) const;

   private:
    PredicateType pred_t_;
    std::string column1_name_;
    std::string column2_name_;
    boost::any constant_;
    ValueComparator comp_;
  };

  struct KNF_Selection_Expression {
    typedef std::vector<Predicate> Disjunction;

    std::vector<Disjunction> disjunctions;
    const std::string toString() const;
    const std::list<std::string> getReferencedColumnNames() const;
  };

  inline const std::string KNF_Selection_Expression::toString() const {
    std::string result;
    result += "(";
    for (unsigned int i = 0; i < disjunctions.size(); ++i) {
      for (unsigned int j = 0; j < disjunctions[i].size(); ++j) {
        result += disjunctions[i][j].toString();
        if (j + 1 < disjunctions[i].size()) result += " OR ";
      }

      if (i + 1 < disjunctions.size()) result += ") AND (";
    }
    result += ")";
    return result;
  }

  inline const std::list<std::string>
  KNF_Selection_Expression::getReferencedColumnNames() const {
    std::list<std::string> result;
    for (unsigned int i = 0; i < disjunctions.size(); ++i) {
      for (unsigned int j = 0; j < disjunctions[i].size(); ++j) {
        if (disjunctions[i][j].getPredicateType() == ValueConstantPredicate ||
            disjunctions[i][j].getPredicateType() ==
                ValueRegularExpressionPredicate) {
          result.push_back(disjunctions[i][j].getColumn1Name());
        } else if (disjunctions[i][j].getPredicateType() ==
                   ValueValuePredicate) {
          result.push_back(disjunctions[i][j].getColumn1Name());
          result.push_back(disjunctions[i][j].getColumn2Name());
        } else {
          COGADB_FATAL_ERROR("Unkown Predicate Type!", "");
        }
      }
    }
    return result;
  }

  typedef KNF_Selection_Expression::Disjunction Disjunction;
}
