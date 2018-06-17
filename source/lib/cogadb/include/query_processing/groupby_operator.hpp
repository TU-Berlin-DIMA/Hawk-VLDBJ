#pragma once

#include <query_processing/definitions.hpp>
#include <util/getname.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class CPU_Groupby_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        CPU_Groupby_Operator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            const std::list<std::string>& grouping_columns,
            const std::list<ColumnAggregation>& aggregation_functions,
            MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              grouping_columns_(grouping_columns),
              aggregation_functions_(aggregation_functions) {}

        virtual bool execute();

        virtual ~CPU_Groupby_Operator() {}

       private:
        std::list<std::string> grouping_columns_;
        std::list<ColumnAggregation> aggregation_functions_;
      };

      class GPU_Groupby_Operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;

        GPU_Groupby_Operator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            const std::list<std::string>& grouping_columns,
            const std::list<ColumnAggregation>& aggregation_functions,
            MaterializationStatus mat_stat = MATERIALIZE)
            : UnaryOperator<TablePtr, TablePtr>(sched_dec, child),
              grouping_columns_(grouping_columns),
              aggregation_functions_(aggregation_functions) {}

        virtual bool execute() { return false; }

        virtual ~GPU_Groupby_Operator() {}

       private:
        std::list<std::string> grouping_columns_;
        std::list<ColumnAggregation> aggregation_functions_;
      };

      Physical_Operator_Map_Ptr map_init_function_groupby_operator();
      TypedOperatorPtr create_CPU_Groupby_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);
      TypedOperatorPtr create_GPU_Groupby_Operator(
          TypedLogicalNode& logical_node, const hype::SchedulingDecision&,
          TypedOperatorPtr left_child, TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_Groupby
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_groupby_operator>  // init_function_Groupby_operator>
      {
       public:
        Logical_Groupby(
            const std::list<std::string>& grouping_columns,
            const std::list<ColumnAggregation>& aggregation_functions,
            MaterializationStatus mat_stat = LOOKUP,
            hype::DeviceConstraint dev_constr =
                CoGaDB::RuntimeConfiguration::instance()
                    .getGlobalDeviceConstraint());

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        std::string toString(bool verbose) const;

        const std::list<std::string> getNamesOfReferencedColumns() const;

        const std::list<std::string>& getGroupingColumns();

        const std::list<ColumnAggregation>& getColumnAggregationFunctions();

        const MaterializationStatus& getMaterializationStatus() const;

        const hype::Tuple getFeatureVector() const;

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
        std::list<std::string> grouping_columns_;
        std::list<ColumnAggregation> aggregation_functions_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
