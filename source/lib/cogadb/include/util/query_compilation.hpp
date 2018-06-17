#ifndef QUERY_COMPILATION_HPP
#define QUERY_COMPILATION_HPP

#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/definitions.hpp>

#include <boost/function.hpp>

namespace CoGaDB {
  namespace query_processing {

    typedef boost::function<const TablePtr(TablePtr)> BulkOperatorFunction;

    void produceBulkProcessingOperator(
        CodeGeneratorPtr code_gen, QueryContextPtr context, NodePtr left_child,
        BulkOperatorFunction function,
        const std::list<std::string>& accessed_attributes);

  }  // end namespace query_processing

}  // end namespace CoGaDB

#endif  // QUERY_COMPILATION_HPP
