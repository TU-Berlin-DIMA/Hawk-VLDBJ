#pragma once

#include <persistence/storage_manager.hpp>
#include <query_processing/column_processing/column_fetch_join_operator.hpp>
#include <query_processing/column_processing/definitions.hpp>
#include <query_processing/operator_extensions.hpp>

namespace CoGaDB {
  namespace query_processing {

    namespace physical_operator {

      class column_bitmap_fetch_join_operator
          : public hype::queryprocessing::UnaryOperator<ColumnPtr, ColumnPtr>,
            public CoGaDB::query_processing::BitmapOperator {
       public:
        typedef column_processing::cpu::TypedOperatorPtr TypedOperatorPtr;
        column_bitmap_fetch_join_operator(
            const hype::SchedulingDecision& sched_dec, TypedOperatorPtr child,
            PK_FK_Join_Predicate pk_fk_join_pred);
        virtual bool execute();
        virtual ~column_bitmap_fetch_join_operator();

       private:
        PK_FK_Join_Predicate pk_fk_join_pred_;
      };

      //            class gpu_column_bitmap_fetch_join_operator : public
      //            hype::queryprocessing::UnaryOperator<ColumnPtr, ColumnPtr>,
      //            public CoGaDB::query_processing::BitmapOperator{
      //            public:
      //                typedef column_processing::cpu::TypedOperatorPtr
      //                TypedOperatorPtr;
      //                gpu_column_bitmap_fetch_join_operator(const
      //                hype::SchedulingDecision& sched_dec, TypedOperatorPtr
      //                child, PK_FK_Join_Predicate pk_fk_join_pred);
      //                virtual bool execute();
      //                virtual void releaseInputData();
      //                virtual bool isInputDataCachedInGPU();
      //                virtual ~gpu_column_bitmap_fetch_join_operator();
      //            private:
      //                PK_FK_Join_Predicate pk_fk_join_pred_;
      //            };

      column_processing::cpu::Physical_Operator_Map_Ptr
      map_init_function_column_bitmap_fetch_join_operator();
      column_processing::cpu::TypedOperatorPtr
      create_column_bitmap_fetch_join_operator(
          column_processing::cpu::TypedLogicalNode& logical_node,
          const hype::SchedulingDecision&,
          column_processing::cpu::TypedOperatorPtr left_child,
          column_processing::cpu::TypedOperatorPtr right_child);
      //            column_processing::cpu::TypedOperatorPtr
      //            create_gpu_column_bitmap_fetch_join_operator(column_processing::cpu::TypedLogicalNode&
      //            logical_node, const hype::SchedulingDecision&,
      //            column_processing::cpu::TypedOperatorPtr left_child,
      //            column_processing::cpu::TypedOperatorPtr right_child);
    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_Column_Bitmap_Fetch_Join
          : public hype::queryprocessing::TypedNode_Impl<
                ColumnPtr,
                physical_operator::
                    map_init_function_column_bitmap_fetch_join_operator>  // init_function_column_bitmap_fetch_join_operator>
      // //init_function_column_bitmap_fetch_join_operator>
      {
       public:
        Logical_Column_Bitmap_Fetch_Join(
            const PK_FK_Join_Predicate& pk_fk_join_pred_,
            hype::DeviceConstraint dev_constr = hype::DeviceConstraint(
                RuntimeConfiguration::instance().getGlobalDeviceConstraint()));

        virtual unsigned int getOutputResultSize() const;

        virtual double getCalculatedSelectivity() const;

        virtual std::string getOperationName() const;

        const PK_FK_Join_Predicate getPK_FK_Join_Predicate() const;

        std::string toString(bool verbose) const;

        virtual const hype::Tuple getFeatureVector() const;

        virtual bool isInputDataCachedInGPU();

       private:
        PK_FK_Join_Predicate pk_fk_join_pred_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
