/*
 * File:   generic_groupby_operator.hpp
 * Author: sebastian
 *
 * Created on 22. Dezember 2015, 17:59
 */

#ifndef GENERIC_GROUPBY_OPERATOR_HPP
#define GENERIC_GROUPBY_OPERATOR_HPP

#include <query_compilation/aggregate_specification.hpp>
#include <query_processing/definitions.hpp>
#include <query_processing/groupby_operator.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {

    namespace logical_operator {

      class Logical_GenericGroupby
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_groupby_operator>  // init_function_Groupby_operator>
      {
       public:
        Logical_GenericGroupby(const GroupByAggregateParam& groupby_param);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const std::list<std::string> getNamesOfReferencedColumns() const;

        const GroupingAttributes getGroupingAttributes();

        const AggregateSpecifications getAggregateSpecifications();

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
        GroupByAggregateParam groupby_param_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB

#endif /* GENERIC_GROUPBY_OPERATOR_HPP */
