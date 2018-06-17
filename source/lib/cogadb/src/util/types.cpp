
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <sstream>
#include <util/getname.hpp>
#include <util/types.hpp>
#include <vector>

namespace CoGaDB {

AttributeType getAttributeType(const std::type_info& type) {
  if (type == typeid(int)) {
    return INT;
  } else if (type == typeid(TID)) {
    return OID;
  } else if (type == typeid(uint64_t)) {
    return OID;
  } else if (type == typeid(uint32_t)) {
    return UINT32;
  } else if (type == typeid(float)) {
    return FLOAT;
  } else if (type == typeid(double)) {
    return DOUBLE;
  } else if (type == typeid(char)) {
    return CHAR;
  } else if (type == typeid(std::string)) {
    return VARCHAR;
  } else if (type == typeid(bool)) {
    return BOOLEAN;
  } else {
    COGADB_FATAL_ERROR("INVALID TYPE!", "");
    return INT;
  }
}

bool convertStringToAttributeType(const std::string& attribute_type_str,
                                  AttributeType& attribute_type) {
  if (attribute_type_str == util::getName(INT)) {
    attribute_type = INT;
  } else if (attribute_type_str == util::getName(UINT32)) {
    attribute_type = UINT32;
  } else if (attribute_type_str == util::getName(OID)) {
    attribute_type = OID;
  } else if (attribute_type_str == util::getName(FLOAT)) {
    attribute_type = FLOAT;
  } else if (attribute_type_str == util::getName(DOUBLE)) {
    attribute_type = DOUBLE;
  } else if (attribute_type_str == util::getName(CHAR)) {
    attribute_type = CHAR;
  } else if (attribute_type_str == util::getName(VARCHAR)) {
    attribute_type = VARCHAR;
  } else if (attribute_type_str == util::getName(DATE)) {
    attribute_type = DATE;
  } else if (attribute_type_str == util::getName(BOOLEAN)) {
    attribute_type = BOOLEAN;
  } else {
    return false;
  }
  return true;
}

bool convertStringToInternalDateType(const std::string& value,
                                     uint32_t& result) {
  std::vector<std::string> strs;
  boost::split(strs, value, boost::is_any_of("-"));
  if (strs.size() != 3) {
    return false;
  }
  // we encode a date as integer of the form <year><month><day>
  // e.g., the date '1998-01-05' will be encoded as integer 19980105
  uint32_t res_value = boost::lexical_cast<uint32_t>(strs[2]);
  res_value += boost::lexical_cast<uint32_t>(strs[1]) * 100;
  res_value += boost::lexical_cast<uint32_t>(strs[0]) * 100 * 100;
  result = res_value;
  return true;
}

bool convertInternalDateTypeToString(const uint32_t& value,
                                     std::string& result) {
  uint32_t year = value / (100 * 100);
  uint32_t month = (value / 100) % 100;
  uint32_t day = value % 100;

  // TODO: check date!

  std::stringstream ss;
  ss << year << "-";
  if (month < 10) ss << 0;
  ss << month << "-";
  if (day < 10) ss << 0;
  ss << day;
  result = ss.str();
  return true;
}

}  // end namespace CogaDB
