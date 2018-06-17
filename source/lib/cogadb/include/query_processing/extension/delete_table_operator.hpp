/*
 * File:   delete_table_operator.hpp
 * Author: sebastian
 *
 * Created on 22. Dezember 2015, 18:10
 */

#ifndef DELETE_TABLE_OPERATOR_HPP
#define DELETE_TABLE_OPERATOR_HPP

#include <query_processing/definitions.hpp>
#include <query_processing/extension/bulk_operator.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace logical_operator {

      class Logical_DeleteTable : public Logical_BulkOperator {
       public:
        Logical_DeleteTable(const std::string& table_name);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const TablePtr executeBulkOperator(TablePtr table) const;

        const std::list<std::string> getNamesOfReferencedColumns() const;

       private:
        std::string table_name_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CoGaDB

#endif /* DELETE_TABLE_OPERATOR_HPP */
