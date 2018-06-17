/*
 * File:   udf_operator.hpp
 * Author: sebastian
 *
 * Created on 14. Mai 2015, 22:14
 */

#ifndef UDF_OPERATOR_HPP
#define UDF_OPERATOR_HPP

#pragma once

#include <core/runtime_configuration.hpp>
#include <query_processing/definitions.hpp>
#include <query_processing/extension/bulk_operator.hpp>

namespace CoGaDB {

  namespace query_processing {

    namespace physical_operator {

      class UDF_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        UDF_Operator(const hype::SchedulingDecision& sched_dec,
                     TypedOperatorPtr child, const std::string& function_name,
                     const std::vector<boost::any>& function_parameters)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              function_name_(function_name),
              function_parameters_(function_parameters) {}

        virtual bool execute();

        virtual ~UDF_Operator() {}

       private:
        std::string function_name_;
        std::vector<boost::any> function_parameters_;
      };

      Physical_Operator_Map_Ptr map_init_function_udf_operator();
      TypedOperatorPtr create_udf_operator(TypedLogicalNode& logical_node,
                                           const hype::SchedulingDecision&,
                                           TypedOperatorPtr left_child,
                                           TypedOperatorPtr right_child);
    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_UDF
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr, physical_operator::map_init_function_udf_operator> {
       public:
        Logical_UDF(const std::string& _function_name,
                    const std::vector<boost::any>& _function_parameters,
                    hype::DeviceConstraint dev_constr =
                        CoGaDB::RuntimeConfiguration::instance()
                            .getGlobalDeviceConstraint());

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const std::string& getFunctionName();

        const std::vector<boost::any>& getFunctionParameters();

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        const std::list<std::string> getNamesOfReferencedColumns() const;

        std::string function_name;
        std::vector<boost::any> function_parameters;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB

#endif /* UDF_OPERATOR_HPP */
