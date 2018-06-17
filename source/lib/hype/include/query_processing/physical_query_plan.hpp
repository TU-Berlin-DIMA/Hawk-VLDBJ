
#pragma once

//#include <core/scheduling_decision.hpp>

//#include <boost/shared_ptr.hpp>

#include <query_processing/binary_operator.hpp>
#include <query_processing/typed_operator.hpp>
#include <query_processing/unary_operator.hpp>

namespace hype {
  namespace queryprocessing {

    template <typename Type>
    class PhysicalQueryPlan {
     public:
      typedef typename OperatorMapper_Helper_Template<Type>::TypedOperatorPtr
          TypedOperatorPtr;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedNodePtr
          TypedNodePtr;
      typedef
          typename OperatorMapper_Helper_Template<Type>::PhysicalQueryPlanPtr
              PhysicalQueryPlanPtr;
      // typedef typename OperatorMapper_Helper_Template<Type>::TypedOperator
      // TypedOperator;

      PhysicalQueryPlan(TypedOperatorPtr root, std::ostream& output_stream)
          : root_(root), out(&output_stream) {
        timeNeeded = 0;
        timeEstimated = 0;
      }

      bool run() {
        bool retVal = false;
        double begin = double(hype::core::getTimestamp());
        if (root_) {
          retVal = root_->run();
        }
        timeNeeded = double(hype::core::getTimestamp()) - begin;
        return retVal;
      }

      const Type getResult() { return root_->getResult(); }

      double getTotalSchedulingDelay() {
        return root_->getTotalSchedulingDelay();
      }

      double getExpectedExecutionTime() {
        return root_->getRecursiveExecutionTimeInNanoseconds();
      }

      /*
      template <typename CallableType>
      void reverse_level_order(CallableType f){

      }*/

      // copy
      // print

      void print() {
        timeEstimated = root_->getRecursiveEstimationTimeInNanoseconds();
        if (timeNeeded > 0) {
          printResults(false, false, false);
        } else {
          (*out) << "PhysicalQueryPlan:" << std::endl;
          root_->print(*out);
          (*out) << "Estimated Time (total): " << timeEstimated / 1000000
                 << "ms (" << timeEstimated << "ns)" << std::endl;
        }
      }

      double getExecutionTime() const throw() { return this->timeNeeded; }

      void printResults(bool show_timings = false,
                        bool show_cardinalities = false,
                        bool show_estimation_error = false) {
        (*out) << std::string(80, '=') << std::endl;
        (*out) << "Query Execution Plan:" << std::endl;
        root_->printResult(*out, 0, show_timings, show_cardinalities,
                           show_estimation_error);
        if (show_estimation_error) {
          (*out) << "Estimated Query Response Time: " << timeEstimated / 1000000
                 << "ms (" << timeEstimated << "ns)" << std::endl;
          (*out) << "Measured Query Response Time: " << (timeNeeded) / 1000000
                 << "ms (" << timeNeeded << "ns)" << std::endl;
          (*out) << "Estimation Error: "
                 << 1 - (double(timeEstimated) / timeNeeded) << std::endl;
        }
        (*out) << std::string(80, '=') << std::endl;
      }

      void setTimeNeeded(double ns) { timeNeeded = ns; }
      TypedOperatorPtr getRoot() { return root_; }

     private:
      double timeNeeded;
      double timeEstimated;
      TypedOperatorPtr root_;
      std::ostream* out;
    };

    // typedef boost::shared_ptr<PhysicalQueryPlan> PhysicalQueryPlanPtr;

  }  // end namespace queryprocessing
}  // end namespace hype
