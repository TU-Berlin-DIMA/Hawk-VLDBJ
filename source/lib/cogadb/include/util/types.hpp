/*
 * File:   types.hpp
 * Author: sebastian
 *
 * Created on 10. Januar 2015, 20:04
 */

#ifndef TYPES_HPP
#define TYPES_HPP

#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include <core/global_definitions.hpp>
#include <typeinfo>

namespace CoGaDB {

  AttributeType getAttributeType(const std::type_info& type);

  bool convertStringToAttributeType(const std::string& attribute_type_str,
                                    AttributeType& attribute_type);

  bool convertStringToInternalDateType(const std::string& value,
                                       uint32_t& result);
  bool convertInternalDateTypeToString(const uint32_t& value,
                                       std::string& result);

  template <typename T>
  bool getValueFromAny(const std::string& name_, const boost::any& value,
                       T& typed_value,
                       const AttributeType& value_database_type) {
    if (value.empty()) return false;
    if (value.type() != typeid(T)) {
      // catch some special cases
      // either a constant is rom type int, double, or std::string
      // if we do not filter a string column, this is a type mismatch
      // in all other cases, we can try and convert it to the type T
      try {
        if (value.type() == typeid(int)) {
          int v = boost::any_cast<int>(value);
          typed_value = boost::lexical_cast<T>(v);
        } else if (value.type() == typeid(double)) {
          double v = boost::any_cast<double>(value);
          typed_value = boost::lexical_cast<T>(v);
        } else if (value.type() == typeid(std::string)) {
          std::string v = boost::any_cast<std::string>(value);
          if (value_database_type == DATE && typeid(T) == typeid(uint32_t)) {
            uint32_t val = 0;
            if (!convertStringToInternalDateType(v, val)) {
              COGADB_FATAL_ERROR("The string '"
                                     << v << "' is not representing a DATE!"
                                     << std::endl
                                     << "Typecast Failed!",
                                 "");
            }
            // this will fail for all non uint32_t types, but this
            // code is only called if T=uint32_t
            typed_value = boost::lexical_cast<T>(val);
          } else {
            typed_value = boost::lexical_cast<T>(v);
          }
        } else {
          COGADB_ERROR("Typemismatch for column "
                           << name_ << std::endl
                           << "Column Type: " << typeid(T).name()
                           << " filter value type: " << value.type().name(),
                       "");
          return false;
        }
      } catch (boost::bad_any_cast& e) {
        COGADB_ERROR("Typemismatch for column "
                         << name_ << std::endl
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: " << value.type().name()
                         << " (bad any cast)",
                     "");
        return false;
      } catch (boost::bad_lexical_cast& e) {
        COGADB_ERROR("Typemismatch for column "
                         << name_ << std::endl
                         << "Column Type: " << typeid(T).name()
                         << " filter value type: " << value.type().name()
                         << " (bad lexical cast)",
                     "");
        return false;
      }
    } else {
      // everything fine, filter value matches type of column
      typed_value = boost::any_cast<T>(value);
    }
    return true;
  }

  template <>
  inline bool getValueFromAny<char*>(const std::string& name_,
                                     const boost::any& value,
                                     char*& typed_value,
                                     const AttributeType& value_database_type) {
    if (value.type() == typeid(char*)) {
      typed_value = boost::any_cast<char*>(value);
      return true;
    } else {
      return false;
    }
  }

}  // end namespace CogaDB

#endif /* TYPES_HPP */
