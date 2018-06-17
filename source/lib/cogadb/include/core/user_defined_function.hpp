/*
 * File:   user_defined_function.hpp
 * Author: sebastian
 *
 * Created on 15. Mai 2015, 09:16
 */

#ifndef USER_DEFINED_FUNCTION_HPP
#define USER_DEFINED_FUNCTION_HPP

#include <boost/any.hpp>
#include <core/base_table.hpp>
#include <core/operator_parameter_types.hpp>
#include <map>

namespace CoGaDB {

  typedef const TablePtr (*UserDefinedFunctionPtr)(
      TablePtr table, const std::string& function_name,
      const std::vector<boost::any>& function_parameters,
      const ProcessorSpecification& proc_spec);

  class UserDefinedFunctions {
   public:
    static UserDefinedFunctions& instance();
    UserDefinedFunctionPtr get(const std::string& function_name);
    bool add(const std::string& function_name, UserDefinedFunctionPtr function);

   private:
    // do not create instances, do not copy construct, do not copy assign
    UserDefinedFunctions();
    UserDefinedFunctions(const UserDefinedFunctions&);
    UserDefinedFunctions& operator=(const UserDefinedFunctions&);
    typedef std::map<std::string, UserDefinedFunctionPtr> UDFMap;
    UDFMap map_;
  };

  const TablePtr limit(TablePtr table, const std::string& function_name,
                       const std::vector<boost::any>& function_parameters,
                       const ProcessorSpecification& proc_spec);

  const TablePtr top_k_sort(TablePtr table, const std::string& function_name,
                            const std::vector<boost::any>& function_parameters,
                            const ProcessorSpecification& proc_spec);

  const TablePtr extract_year(
      TablePtr table, const std::string& function_name,
      const std::vector<boost::any>& function_parameters,
      const ProcessorSpecification& proc_spec);

}  // end namespace CoGaDB

#endif /* USER_DEFINED_FUNCTION_HPP */
