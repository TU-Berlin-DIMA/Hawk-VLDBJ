
/*
 * File:   variable_manager.cpp
 * Author: sebastian
 *
 * Created on 28. September 2014, 11:03
 */

#include <boost/algorithm/string.hpp>
#include <core/base_table.hpp>
#include <core/table.hpp>
#include <core/variable_manager.hpp>
#include <iostream>
#include <util/getname.hpp>

namespace CoGaDB {

bool checkStringIsBoolean(const std::string& value) {
  std::string new_val = boost::to_lower_copy(value);
  if (new_val == "true" || new_val == "false") {
    return true;
  } else {
    return false;
  }
}
bool checkStringIsInteger(const std::string& value) {
  try {
    boost::lexical_cast<int>(value);
  } catch (const boost::bad_lexical_cast& e) {
    return false;
  }
  return true;
}
bool checkStringIsFloat(const std::string& value) {
  try {
    boost::lexical_cast<float>(value);
  } catch (const boost::bad_lexical_cast& e) {
    return false;
  }
  return true;
}

VariableState::VariableState(const std::string& value, AttributeType type,
                             CheckVariableValueFunction val_check_func,
                             GetVariableValueFunction get_var_func,
                             SetVariableValueFunction set_var_func)
    : value_(value),
      type_(type),
      val_check_func_(val_check_func),
      get_var_func_(get_var_func),
      set_var_func_(set_var_func) {}
bool VariableState::setValue(const std::string& value) {
  if (val_check_func_) {
    if (!(*val_check_func_)(value)) return false;
  }
  if (set_var_func_) {
    return (*set_var_func_)(*this, value);
  } else {
    value_ = value;
    return true;
  }
}
std::string VariableState::getValue() const {
  if (get_var_func_) {
    return (*get_var_func_)(*this);
  } else {
    return value_;
  }
}
AttributeType VariableState::getType() const { return type_; }

VariableManager::VariableManager() : variables_() {}
VariableManager& VariableManager::instance() {
  static VariableManager vm;
  return vm;
}

bool VariableManager::addVariable(const std::string& variable_name,
                                  const VariableState& var_stat) {
  VariableMap::iterator it = variables_.find(variable_name);
  if (it != variables_.end()) {
    // var already exists!
    COGADB_WARNING("Variable Manager: Variable '" << variable_name
                                                  << "' already exists!",
                   "");
    return false;
  } else {
    variables_.insert(Variable(variable_name, var_stat));
    return true;
  }
}
bool VariableManager::setVariableValue(const std::string& variable_name,
                                       const std::string& value) {
  VariableMap::iterator it = variables_.find(variable_name);
  if (it != variables_.end()) {
    return it->second.setValue(value);
  } else {
    return false;
  }
}
std::string VariableManager::getVariableValueString(
    const std::string& variable_name) const {
  VariableMap::const_iterator it = variables_.find(variable_name);
  if (it != variables_.end()) {
    return it->second.getValue();
  }
  COGADB_WARNING("Variable Manager: Variable '" << variable_name
                                                << "' does not exist!",
                 "");
  //        std::stringstream ss;
  //        ss << "Variable Manager: Variable '" << variable_name << "' does not
  //        exist!";
  return "INVALID";  // ss.str();
}
int VariableManager::getVariableValueInteger(
    const std::string& variable_name) const {
  return getVariableValue<int>(variable_name);
}
float VariableManager::getVariableValueFloat(
    const std::string& variable_name) const {
  return getVariableValue<float>(variable_name);
}
bool VariableManager::getVariableValueBoolean(
    const std::string& variable_name) const {
  VariableMap::const_iterator it = variables_.find(variable_name);
  if (it != variables_.end()) {
    std::string val = it->second.getValue();
    if (val == "true") return true;
    if (val == "false") return false;
    COGADB_FATAL_ERROR("Variable Manager: Variable '"
                           << variable_name << "' has invalid value '" << val
                           << "'!",
                       "");
  }
  COGADB_WARNING("Variable Manager: Variable '" << variable_name
                                                << "' does not exist!",
                 "");
  return false;
}

TablePtr VariableManager::getSystemTable() const {
  VariableMap::const_iterator it;
  TableSchema result_schema;
  result_schema.push_back(Attribut(VARCHAR, "VARIABLE_NAME"));
  result_schema.push_back(Attribut(VARCHAR, "VARIABLE_VALUE"));
  result_schema.push_back(Attribut(VARCHAR, "VARIABLE_TYPE"));

  TablePtr result_tab(new Table("SYS_VARIABLES", result_schema));

  for (it = variables_.begin(); it != variables_.end(); ++it) {
    Tuple t;
    t.push_back(it->first);
    t.push_back(it->second.getValue());
    t.push_back(util::getName(it->second.getType()));
    result_tab->insert(t);
  }
  return result_tab;
}

TablePtr getSystemTableVariables() {
  return VariableManager::instance().getSystemTable();
}

}  // end namespace CogaDB
