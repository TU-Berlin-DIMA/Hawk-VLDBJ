/*
 * File:   store_table_operator.hpp
 * Author: sebastian
 *
 * Created on 22. Dezember 2015, 17:58
 */

#ifndef STORE_TABLE_OPERATOR_HPP
#define STORE_TABLE_OPERATOR_HPP

#include <query_processing/definitions.hpp>
#include <query_processing/extension/bulk_operator.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace logical_operator {

      class Logical_StoreTable : public Logical_BulkOperator {
       public:
        Logical_StoreTable(const std::string& table_name,
                           const bool persist_to_disk);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        //                void produce_impl(CodeGeneratorPtr code_gen,
        //                QueryContextPtr context);
        //
        //                void consume_impl(CodeGeneratorPtr code_gen,
        //                QueryContextPtr context);

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const TablePtr executeBulkOperator(TablePtr table) const;

        const std::list<std::string> getNamesOfReferencedColumns() const;

       private:
        std::string table_name_;
        bool persist_to_disk_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CoGaDB

#endif /* STORE_TABLE_OPERATOR_HPP */
