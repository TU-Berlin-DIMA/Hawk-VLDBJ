#pragma once
#include <boost/any.hpp>
#include <core/global_definitions.hpp>
#include <fstream>
#include <iostream>

namespace CoGaDB {

  inline std::ostream& operator<<(std::ostream& os, const boost::any& v) {
    if (v.empty()) return os;
    // os <<
    if (typeid(int) == v.type()) {
      os << boost::any_cast<int>(v);
    } else if (typeid(unsigned int) == v.type()) {
      os << boost::any_cast<unsigned int>(v);
    } else if (typeid(unsigned long) == v.type()) {
      os << boost::any_cast<unsigned long>(v);
    } else if (typeid(float) == v.type()) {
      os << boost::any_cast<float>(v);
    } else if (typeid(double) == v.type()) {
      os << boost::any_cast<double>(v);
    } else if (typeid(std::string) == v.type()) {
      os << boost::any_cast<std::string>(v);
    } else if (typeid(bool) == v.type()) {
      os << boost::any_cast<bool>(v);
    } else {
      COGADB_FATAL_ERROR(
          "In std::ostream& operator << (std::ostream& os, const boost::any& v)"
              << std::endl
              << "Unkown Type stored in boost::any, cannot print value!",
          "");
    }
    return os;
  }

  /*! \brief A overload of std::vector<T> for operator <<, so that vector
   * objects can be printed in CoGaDB by using operator <<.*/
  template <class T>
  inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end();
         ++ii) {
      os << " " << *ii;
    }
    os << " ]";
    return os;
  }

  /*! \brief Total template specialization for operator <<, so that different
   * semantic can be implemented in contrast to std::vector<T>. This overload is
   * a convinience function, so that Tuple objects can be printed by using
   * operator <<. */
  template <>
  inline std::ostream& operator<<(std::ostream& os, const Tuple& v) {
    // ofs.write((char*)(a.array), 20*sizeof(int));
    os << "Tuple: (";
    for (unsigned int i = 0; i < v.size(); i++) {
      if (v[i].type() == typeid(int)) {
        os << boost::any_cast<int>(v[i]);
      } else if (v[i].type() == typeid(float)) {
        os << boost::any_cast<float>(v[i]);
      } else if (v[i].type() == typeid(std::string)) {
        os << boost::any_cast<std::string>(v[i]);
      } else if (v[i].type() == typeid(bool)) {
        os << boost::any_cast<bool>(v[i]);
      } else {
        COGADB_FATAL_ERROR(
            "Found Invalid Type in Tuple: " << v[i].type().name(), "");
      }
      if (i < v.size() - 1) os << ",";
    }
    os << ")";  // << std::endl;
    return os;
    return os;
  }

}  // end namespace CogaDB
