/*
 * File:   sink_operator.hpp
 * Author: sebastian
 *
 * Created on 6. Januar 2016, 18:04
 */

#ifndef SINK_OPERATOR_HPP
#define SINK_OPERATOR_HPP

#include <query_processing/definitions.hpp>
#include <query_processing/extension/bulk_operator.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace logical_operator {

      class Logical_Sink : public Logical_BulkOperator {
       public:
        Logical_Sink();

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const TablePtr executeBulkOperator(TablePtr table) const;

        const std::list<std::string> getNamesOfReferencedColumns() const;

       private:
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CoGaDB

#endif /* SINK_OPERATOR_HPP */
