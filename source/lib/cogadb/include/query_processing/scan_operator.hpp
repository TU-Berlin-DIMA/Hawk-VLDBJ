#pragma once

#include <persistence/storage_manager.hpp>
#include <query_processing/definitions.hpp>

namespace CoGaDB {
  namespace query_processing {
    namespace physical_operator {

      class scan_operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;
        scan_operator(const hype::SchedulingDecision& sched_dec,
                      TablePtr table);
        virtual bool execute();
        virtual void releaseInputData();
        virtual ~scan_operator();

       private:
        TablePtr table_;
      };

      Physical_Operator_Map_Ptr map_init_function_scan_operator();
      TypedOperatorPtr create_scan_operator(TypedLogicalNode& logical_node,
                                            const hype::SchedulingDecision&,
                                            TypedOperatorPtr left_child,
                                            TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_scan_operator;

    // Map_Init_Function
    // init_function_scan_operator=physical_operator::map_init_function_scan_operator;
    // //boost::bind();

    // Map_Init_Function getMap_Init_Function_Scan_Operation();

    namespace logical_operator {

      class Logical_Scan
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_scan_operator>  // init_function_scan_operator>
      // //init_function_scan_operator>
      {
       public:
        Logical_Scan(std::string table_name, uint32_t version = 1);
        Logical_Scan(TablePtr table, uint32_t version = 1);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        virtual std::string toString(bool verbose) const;

        const std::string& getTableName();

        const TablePtr getTablePtr();

        // virtual TypedOperatorPtr getOptimalOperator(TypedOperatorPtr
        // left_child, TypedOperatorPtr right_child, hype::DeviceTypeConstraint
        // dev_constr);
        virtual const hype::Tuple getFeatureVector() const;

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

       private:
        TablePtr table_;
        uint32_t version_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
