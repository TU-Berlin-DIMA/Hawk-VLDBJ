/*
 * File:   regular_expression.hpp
 * Author: sebastian
 *
 * Created on 27. April 2015, 20:58
 */

#ifndef REGULAR_EXPRESSION_HPP
#define REGULAR_EXPRESSION_HPP

#include <boost/make_shared.hpp>
#include <boost/regex.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <core/global_definitions.hpp>
#include <iostream>
#include <string>

namespace CoGaDB {

  template <typename T>
  const ColumnPtr getMatchingIDsFromDictionary(
      const std::string& regular_expression,
      const typename DictionaryCompressedColumn<T>::Dictionary& dict) {
    COGADB_FATAL_ERROR("Called unimplemented method!", "");
    return ColumnPtr();
  }

  template <>
  inline const ColumnPtr getMatchingIDsFromDictionary<std::string>(
      const std::string& regular_expression,
      const typename DictionaryCompressedColumn<std::string>::Dictionary&
          dict) {
    typedef DictionaryCompressedColumn<std::string>::CodeWordType CodeWordType;
    typedef DictionaryCompressedColumn<std::string>::Dictionary Dictionary;

    boost::regex re;
    boost::cmatch matches;

    try {
      // Assignment and construction initialize the FSM used
      // for regexp parsing
      re = regular_expression;
    } catch (boost::regex_error& e) {
      COGADB_FATAL_ERROR("'" << regular_expression
                             << "' is not a valid regular expression: '"
                             << e.what() << "'",
                         "");
      return ColumnPtr();
    }

    boost::shared_ptr<Column<CodeWordType> > column =
        boost::make_shared<Column<CodeWordType> >("", UINT32);
    column->reserve(dict.size());

    Dictionary::const_iterator cit;
    for (cit = dict.begin(); cit != dict.end(); ++cit) {
      if (boost::regex_match(cit->first, re)) {
        column->push_back(cit->second);
      }
    }

    return column;
  }

}  // end namespace CoGaDB

#endif /* REGULAR_EXPRESSION_HPP */
