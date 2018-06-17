/*
 * File:   variable_manager.hpp
 * Author: sebastian
 *
 * Created on 28. September 2014, 11:03
 */

#ifndef VARIABLE_MANAGER_HPP
#define VARIABLE_MANAGER_HPP

#include <boost/lexical_cast.hpp>
#include <core/global_definitions.hpp>
#include <map>
#include <string>

namespace CoGaDB {

  class VariableState;
  class BaseTable;
  typedef boost::shared_ptr<BaseTable> TablePtr;

  typedef bool (*CheckVariableValueFunction)(const std::string&);
  typedef bool (*SetVariableValueFunction)(VariableState&, const std::string&);
  typedef std::string (*GetVariableValueFunction)(const VariableState&);

  bool checkStringIsBoolean(const std::string&);
  bool checkStringIsInteger(const std::string&);
  bool checkStringIsFloat(const std::string&);

  class VariableState {
   public:
    VariableState(const std::string& value, AttributeType type,
                  CheckVariableValueFunction val_check_func = NULL,
                  GetVariableValueFunction get_var_func = NULL,
                  SetVariableValueFunction set_var_func = NULL);
    bool setValue(const std::string& value);
    std::string getValue() const;
    AttributeType getType() const;

   private:
    std::string value_;
    AttributeType type_;
    CheckVariableValueFunction val_check_func_;
    GetVariableValueFunction get_var_func_;
    SetVariableValueFunction set_var_func_;
  };

  class VariableManager {
   public:
    static VariableManager& instance();
    bool addVariable(const std::string& variable_name,
                     const VariableState& var_stat);
    bool setVariableValue(const std::string& variable_name,
                          const std::string& value);
    std::string getVariableValueString(const std::string& variable_name) const;
    int getVariableValueInteger(const std::string& variable_name) const;
    float getVariableValueFloat(const std::string& variable_name) const;
    bool getVariableValueBoolean(const std::string& variable_name) const;
    TablePtr getSystemTable() const;

   private:
    VariableManager();
    VariableManager(const VariableManager&);
    VariableManager& operator=(const VariableManager&);
    template <typename Type>
    Type getVariableValue(const std::string& variable_name) const;
    typedef std::pair<std::string, VariableState> Variable;
    typedef std::map<std::string, VariableState> VariableMap;
    VariableMap variables_;
  };

  TablePtr getSystemTableVariables();

  template <typename Type>
  Type VariableManager::getVariableValue(
      const std::string& variable_name) const {
    VariableMap::const_iterator it = variables_.find(variable_name);
    if (it != variables_.end()) {
      std::string s = it->second.getValue();
      return boost::lexical_cast<Type>(s);
    }
    return Type();
  }

}  // end namespace CogaDB

#endif /* VARIABLE_MANAGER_HPP */
