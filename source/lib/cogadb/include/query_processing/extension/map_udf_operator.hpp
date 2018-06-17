/*
 * File:   map_udf_operator.hpp
 * Author: sebastian
 *
 * Created on 22. Dezember 2015, 17:56
 */

#ifndef MAP_UDF_OPERATOR_HPP
#define MAP_UDF_OPERATOR_HPP

#include <query_compilation/aggregate_specification.hpp>
#include <query_processing/definitions.hpp>
#include <query_processing/extension/bulk_operator.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {

    namespace logical_operator {

      class Logical_MapUDF
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                map_init_function_dummy>  // init_function_Groupby_operator>
      {
       public:
        Logical_MapUDF(const Map_UDF_ParamPtr param);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const std::list<std::string> getNamesOfReferencedColumns() const;

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
        Map_UDF_ParamPtr param_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB

#endif /* MAP_UDF_OPERATOR_HPP */
