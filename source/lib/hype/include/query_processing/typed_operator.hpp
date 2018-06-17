
#pragma once

//#include <core/scheduling_decision.hpp>

//#include <boost/shared_ptr.hpp>

#include <ostream>
#include <query_processing/operator.hpp>
//#include <util/iostream.hpp>

namespace hype {
  namespace queryprocessing {

    /*! \todo add query id to operator, so it can be easily identified, which
     * query an operator belongs to*/

    template <typename Type>
    class TypedOperator : public hype::queryprocessing::Operator {
     public:
      typedef Type OperatorOutputType;

      TypedOperator(const hype::core::SchedulingDecision& sched_dec)
          : Operator(sched_dec), result_(), out(&std::cout) {
        // result_size_=0;
      }

      virtual ~TypedOperator() {}

      const OperatorOutputType getResult() {
        // this(); //execute Functor
        return result_;
      }

      void setOutputStream(std::ostream& output_stream) {
        this->out = &output_stream;
      }

      void cleanupResult() { result_.reset(); }
      //            virtual void releaseInputData(){
      //                result_.reset();
      //            }

      virtual bool run() = 0;

      virtual void print(std::ostream& out,
                         unsigned int tree_level = 0) const = 0;
      virtual void printResult(std::ostream& out, unsigned int tree_level = 0,
                               bool show_timings = false,
                               bool show_cardinalities = false,
                               bool show_estimation_error = false) = 0;

      //            void setResultSize(double result_size) {
      //                result_size_ = result_size;
      //            }
      //
      //            double getResultSize() const {
      //                return result_size_;
      //            }

      // virtual std::list<TypedOperator&> getOperatorQueue() = 0;

     protected:
      OperatorOutputType result_;
      std::ostream* out;
      //            double result_size_;
    };

  }  // end namespace queryprocessing
}  // end namespace hype
