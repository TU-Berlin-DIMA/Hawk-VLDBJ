#ifndef LIMIT_OPERATOR_HPP
#define LIMIT_OPERATOR_HPP

#include <query_processing/definitions.hpp>
#include <query_processing/extension/bulk_operator.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace logical_operator {

      class Logical_Limit : public Logical_BulkOperator {
       public:
        Logical_Limit(const uint64_t &max_number_of_rows);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const TablePtr executeBulkOperator(TablePtr table) const;

        const std::list<std::string> getNamesOfReferencedColumns() const;

       private:
        uint64_t max_number_of_rows_;
      };

    }  // end namespace logical_operator
  }    // end namespace query_processing
}  // end namespace CoGaDB

#endif  // LIMIT_OPERATOR_HPP
