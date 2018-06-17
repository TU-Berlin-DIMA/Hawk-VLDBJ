/*
 * File:   bulk_operator.hpp
 * Author: sebastian
 *
 * Created on 23. Dezember 2015, 12:46
 */

#ifndef BULK_OPERATOR_HPP
#define BULK_OPERATOR_HPP

#include <query_processing/definitions.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {

    Physical_Operator_Map_Ptr map_init_function_dummy();

    namespace logical_operator {

      class Logical_BulkOperator : public hype::queryprocessing::TypedNode_Impl<
                                       TablePtr, map_init_function_dummy> {
       public:
        Logical_BulkOperator();

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        virtual const TablePtr executeBulkOperator(TablePtr table) const = 0;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CoGaDB

#endif /* BULK_OPERATOR_HPP */
