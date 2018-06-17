/*
 * File:   generic_selection_operator.hpp
 * Author: sebastian
 *
 * Created on 22. Dezember 2015, 18:14
 */

#ifndef GENERIC_SELECTION_OPERATOR_HPP
#define GENERIC_SELECTION_OPERATOR_HPP

#include <query_processing/extension/bulk_operator.hpp>

namespace CoGaDB {

  class PredicateExpression;
  typedef boost::shared_ptr<PredicateExpression> PredicateExpressionPtr;

  namespace query_processing {
    namespace logical_operator {

      class Logical_GenericSelection
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                map_init_function_dummy>  // init_function_ComplexSelection_operator>
      {
       public:
        Logical_GenericSelection(PredicateExpressionPtr pred_expr);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const PredicateExpressionPtr getPredicateExpression();

        bool couldNotBePushedDownFurther();

        void couldNotBePushedDownFurther(bool val);

        const std::list<std::string> getNamesOfReferencedColumns() const;

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
        PredicateExpressionPtr pred_expr_;
        /* \brief Bit that shows the optimizer that this selection could not be
         * pushed down more in the query plan*/
        bool could_not_be_pushed_down_further_;
      };

    }  // end namespace logical_operator
  }    // end namespace query_processing
}  // end namespace CoGaDB

#endif /* GENERIC_SELECTION_OPERATOR_HPP */
