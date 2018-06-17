#pragma once

#include <query_processing/definitions.hpp>

namespace CoGaDB {

  namespace query_processing {
    namespace physical_operator {

      class rename_operator
          : public hype::queryprocessing::UnaryOperator<TablePtr, TablePtr> {
       public:
        typedef hype::queryprocessing::OperatorMapper_Helper_Template<
            TablePtr>::TypedOperatorPtr TypedOperatorPtr;
        rename_operator(const hype::SchedulingDecision& sched_dec,
                        TypedOperatorPtr left_child, RenameList rename_list);
        virtual bool execute();
        virtual ~rename_operator();

       private:
        RenameList rename_list_;
      };

      Physical_Operator_Map_Ptr map_init_function_rename_operator();
      TypedOperatorPtr create_rename_operator(TypedLogicalNode& logical_node,
                                              const hype::SchedulingDecision&,
                                              TypedOperatorPtr left_child,
                                              TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    // extern Map_Init_Function init_function_rename_operator;

    // Map_Init_Function
    // init_function_rename_operator=physical_operator::map_init_function_rename_operator;
    // //boost::bind();

    // Map_Init_Function getMap_Init_Function_Scan_Operation();

    namespace logical_operator {

      class Logical_Rename
          : public hype::queryprocessing::TypedNode_Impl<
                TablePtr,
                physical_operator::
                    map_init_function_rename_operator>  // init_function_rename_operator>
      // //init_function_rename_operator>
      {
       public:
        Logical_Rename(const RenameList& rename_list);

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        virtual std::string toString(bool verbose) const;

        const RenameList& getRenameList();

        void produce_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        void consume_impl(CodeGeneratorPtr code_gen, QueryContextPtr context);

        // const TablePtr getTablePtr();

        // virtual TypedOperatorPtr getOptimalOperator(TypedOperatorPtr
        // left_child, TypedOperatorPtr right_child, hype::DeviceTypeConstraint
        // dev_constr);

       private:
        RenameList rename_list_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
