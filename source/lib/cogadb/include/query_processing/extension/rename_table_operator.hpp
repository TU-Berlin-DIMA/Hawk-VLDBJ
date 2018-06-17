/*
 * File:   rename_table_operator.hpp
 * Author: sebastian
 *
 * Created on 6. Januar 2016, 17:52
 */

#ifndef RENAME_TABLE_OPERATOR_HPP
#define RENAME_TABLE_OPERATOR_HPP

#include <query_processing/definitions.hpp>
#include <query_processing/extension/bulk_operator.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace logical_operator {

      class Logical_RenameTable : public Logical_BulkOperator {
       public:
        Logical_RenameTable(const std::string& table_name,
                            const std::string& new_table_name);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const TablePtr executeBulkOperator(TablePtr table) const;

        const std::list<std::string> getNamesOfReferencedColumns() const;

       private:
        std::string table_name_;
        std::string new_table_name_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CoGaDB

#endif /* RENAME_TABLE_OPERATOR_HPP */
