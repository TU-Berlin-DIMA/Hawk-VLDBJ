#pragma once

#include <core/bitmap.hpp>
#include <core/lookup_array.hpp>
#include <query_processing/column_processing/definitions.hpp>
#include <query_processing/definitions.hpp>
#include <query_processing/operator_extensions.hpp>
#include <util/getname.hpp>

#include <hardware_optimizations/primitives.hpp>

namespace CoGaDB {
  namespace query_processing {

    namespace physical_operator {

      class CPU_Bitmap_Operator
          : public hype::queryprocessing::BinaryOperator<ColumnPtr, ColumnPtr,
                                                         ColumnPtr>,
            public BitmapOperator {
       public:
        // typedef
        // hype::queryprocessing::OperatorMapper_Helper_Template<ColumnPtr>::TypedOperatorPtr
        // ColumnWise_TypedOperatorPtr;
        typedef column_processing::cpu::TypedOperatorPtr TypedOperatorPtr;

        CPU_Bitmap_Operator(const hype::SchedulingDecision& sched_dec,
                            TypedOperatorPtr left_child,
                            TypedOperatorPtr right_child, BitmapOperation op,
                            MaterializationStatus mat_stat = MATERIALIZE);

        virtual bool execute();

        virtual ~CPU_Bitmap_Operator();

       private:
        BitmapOperation op_;
      };

      column_processing::cpu::Physical_Operator_Map_Ptr
      map_init_function_cpu_bitmap_operator();
      column_processing::cpu::TypedOperatorPtr create_CPU_Bitmap_Operator(
          column_processing::cpu::TypedLogicalNode& logical_node,
          const hype::SchedulingDecision&,
          column_processing::cpu::TypedOperatorPtr left_child,
          column_processing::cpu::TypedOperatorPtr right_child);
      //            column_processing::cpu::TypedOperatorPtr
      //            create_GPU_Bitmap_Operator(column_processing::cpu::TypedLogicalNode&
      //            logical_node, const hype::SchedulingDecision&,
      //            column_processing::cpu::TypedOperatorPtr left_child,
      //            column_processing::cpu::TypedOperatorPtr right_child);

      //			GPU_ColumnWise_TypedOperatorPtr
      // create_GPU_ColumnAlgebraOperator(GPU_ColumnWise_TypedLogicalNode&
      // logical_node, const hype::SchedulingDecision&,
      // GPU_ColumnWise_TypedOperatorPtr left_child,
      // GPU_ColumnWise_TypedOperatorPtr right_child);

    }  // end namespace physical_operator

    namespace logical_operator {

      class Logical_Bitmap_Operator
          : public hype::queryprocessing::TypedNode_Impl<
                ColumnPtr,
                physical_operator::
                    map_init_function_cpu_bitmap_operator>  // init_function_Join_operator>
      {
       public:
        Logical_Bitmap_Operator(BitmapOperation op,
                                MaterializationStatus mat_stat = MATERIALIZE,
                                hype::DeviceConstraint dev_constr =
                                    hype::DeviceConstraint(hype::CPU_ONLY))
            : TypedNode_Impl<
                  ColumnPtr,
                  physical_operator::map_init_function_cpu_bitmap_operator>(
                  false, dev_constr),
              op_(op),
              mat_stat_(mat_stat) {}

        virtual unsigned int getOutputResultSize() const {
          return this->left_->getOutputResultSize();
        }

        virtual double getCalculatedSelectivity() const { return 1; }

        virtual std::string getOperationName() const {
          return "Bitmap_Operator";  // util::getName(op_);
        }

        virtual BitmapOperation getBitmapOperation() const { return op_; }

        const MaterializationStatus& getMaterializationStatus() const {
          return mat_stat_;
        }
        std::string toString(bool verbose) const {
          std::string result = "Bitmap_Operator";
          if (verbose) {
            result += " (";
            result += util::getName(op_);
            result += ")";
          }
          return result;
        }

        const hype::Tuple getFeatureVector() const {
          hype::Tuple t;
          if (this->left_) {  // if left child is valid (has to be by
                              // convention!), add input data size
            // if we already know the correct input data size, because the child
            // node was already executed
            // during query chopping, we use the real cardinality, other wise we
            // call the estimator
            if (this->left_->getPhysicalOperator() &&
                this->right_->getPhysicalOperator()) {
              BitmapOperator* left_bitmap_op = dynamic_cast<BitmapOperator*>(
                  this->left_->getPhysicalOperator().get());
              BitmapOperator* right_bitmap_op = dynamic_cast<BitmapOperator*>(
                  this->right_->getPhysicalOperator().get());
              assert(left_bitmap_op != NULL);
              assert(right_bitmap_op != NULL);
              assert(left_bitmap_op->hasResultBitmap());
              assert(right_bitmap_op->hasResultBitmap());

              // for the learning algorithms, it is helpful to
              // artificially adjust the points in multidimensional space
              // so proper regression models can be build
              // we use as feature vector <size of bitmap in mega byte,number
              // bitmaps cached on GPU>
              if (left_bitmap_op->hasResultBitmap()) {
                t.push_back(left_bitmap_op->getResultBitmap()->size() /
                            (1000 * 1000));
              }

              //                        if(right_bitmap_op->hasResultBitmap()){
              //                            t.push_back(right_bitmap_op->getResultBitmap()->size());
              //                        }else
              //                        if(right_bitmap_op->hasCachedResult_GPU_Bitmap()){
              //                            t.push_back(right_bitmap_op->getResult_GPU_Bitmap()->size());
              //                        }
              double num_of_cached_GPU_bitmaps = 0;
              if (left_bitmap_op->hasResultBitmap() &&
                  CoGaDB::isGPUMemory(
                      left_bitmap_op->getResultBitmap()->getMemoryID()))
                num_of_cached_GPU_bitmaps++;
              if (right_bitmap_op->hasResultBitmap() &&
                  CoGaDB::isGPUMemory(
                      right_bitmap_op->getResultBitmap()->getMemoryID()))
                num_of_cached_GPU_bitmaps++;
              t.push_back(num_of_cached_GPU_bitmaps * 10);
              return t;
              // t.push_back(this->left_->getPhysicalOperator()->getResultSize());
              // // ->result_size_;
            } else {
              return this->Node::getFeatureVector();
              // t.push_back(this->left_->getOutputResultSize());
            }
          } else {
            HYPE_FATAL_ERROR("Invalid Left Child!", std::cout);
          }

          return t;
        }

       private:
        BitmapOperation op_;
        MaterializationStatus mat_stat_;
      };

    }  // end namespace logical_operator

  }  // end namespace query_processing

}  // end namespace CogaDB
