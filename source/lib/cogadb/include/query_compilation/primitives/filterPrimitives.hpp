
/*
 * File:   instruction.hpp
 * Author: dbronesk
 *
 * Created on February 1, 2016, 2:56 PM
 */

#ifndef FILTER_HPP
#define FILTER_HPP

#include <sstream>

#include <query_compilation/predicate_expression.hpp>
#include <query_compilation/primitives/instruction.hpp>

namespace CoGaDB {

  class Filter : public Instruction {
   public:
    Filter(const PredicateExpressionPtr pred_expr)
        : Instruction(FILTER_INSTR),
          pred_expr_(pred_expr),
          use_predication_(false),
          result_count_variable_() {}
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override;
    virtual const std::string toString() const override;
    PredicateExpressionPtr getPredicateExpression() { return pred_expr_; }
    virtual std::map<std::string, std::string> getInputVariables(
        CodeGenerationTarget target) override final;
    virtual std::map<std::string, std::string> getOutputVariables(
        CodeGenerationTarget target) override final;
    bool supportsPredication() const override final;

   protected:
    const PredicateExpressionPtr pred_expr_;
    bool use_predication_;
    std::string result_count_variable_;
  };

  typedef boost::shared_ptr<Filter> FilterPtr;

  class SSEFilter : public Filter {
   public:
    SSEFilter(const PredicateExpressionPtr pred_expr)
        : Filter(pred_expr), sse_result_name_("SSE_result") {}
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override;
    virtual const std::string toString() const override;

   protected:
    std::string sse_result_name_;
  };

  typedef boost::shared_ptr<SSEFilter> SSEFilterPtr;

  class SSEMaskFilter : public SSEFilter {
   public:
    SSEMaskFilter(const PredicateExpressionPtr pred_expr)
        : SSEFilter(pred_expr) {}
    SSEMaskFilter(const SSEFilterPtr sseFilter)
        : SSEFilter(sseFilter->getPredicateExpression()) {}
    virtual const GeneratedCodePtr getCode(
        CodeGenerationTarget target) override final;
    virtual const std::string toString() const override final;
  };

  typedef boost::shared_ptr<SSEMaskFilter> SSEMaskFilterPtr;

}  // end namespace CoGaDB

#endif /* INSTRUCTION_HPP */
