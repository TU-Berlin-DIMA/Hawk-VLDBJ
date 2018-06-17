/*
 * File:   export_into_file.hpp
 * Author: sebastian
 *
 * Created on 23. Dezember 2015, 13:34
 */

#ifndef EXPORT_INTO_FILE_HPP
#define EXPORT_INTO_FILE_HPP

#include <query_processing/definitions.hpp>
#include <query_processing/extension/bulk_operator.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace logical_operator {

      class Logical_ExportTableIntoFile : public Logical_BulkOperator {
       public:
        Logical_ExportTableIntoFile(const std::string& path_to_file,
                                    const std::string& delimiter);

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
        std::string path_to_file_;
        std::string delimiter_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CoGaDB

#endif /* EXPORT_INTO_FILE_HPP */
