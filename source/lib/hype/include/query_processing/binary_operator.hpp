#pragma once

//#include <core/scheduling_decision.hpp>
//#include <boost/shared_ptr.hpp>

#include <config/configuration.hpp>
#include <config/global_definitions.hpp>
#include "node.hpp"
#include "operator.hpp"
#include "typed_operator.hpp"

//#ifndef HYPE_ENABLE_PARALLEL_QUERY_PLAN_EVALUATION
//      #define HYPE_ENABLE_PARALLEL_QUERY_PLAN_EVALUATION
//#endif

#ifdef HYPE_ENABLE_PARALLEL_QUERY_PLAN_EVALUATION
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#endif

namespace hype {
  namespace queryprocessing {

    template <typename OperatorInputTypeLeftChild,
              typename OperatorInputTypeRightChild, typename OperatorOutputType>
    class BinaryOperator
        : public TypedOperator<OperatorOutputType> {  // uses 2D Learning Method
     public:
      // typedef typename
      // OperatorMapper_Helper_Template<OperatorOutputType>::TypedOperator
      // TypedOperator;
      typedef typename OperatorMapper_Helper_Template<
          OperatorOutputType>::TypedOperatorPtr TypedOperatorPtr;
      typedef typename OperatorMapper_Helper_Template<
          OperatorOutputType>::TypedNodePtr TypedNodePtr;

      BinaryOperator(
          const hype::SchedulingDecision& sched_dec,
          boost::shared_ptr<TypedOperator<OperatorInputTypeLeftChild> >
              left_child,
          boost::shared_ptr<TypedOperator<OperatorInputTypeRightChild> >
              right_child)
          : TypedOperator<OperatorOutputType>(sched_dec),
            left_child_(left_child),
            right_child_(right_child) {}

      virtual ~BinaryOperator() {}

      OperatorInputTypeLeftChild getInputDataLeftChild() {
        return this->left_child_->getResult();
      }

      OperatorInputTypeRightChild getInputDataRightChild() {
        return this->right_child_->getResult();
      }

      virtual bool run() {
        if (!hype::core::Runtime_Configuration::instance()
                 .isQueryChoppingEnabled()) {
#ifdef HYPE_ENABLE_PARALLEL_QUERY_PLAN_EVALUATION
          // std::cout << "Binary Operator: launching both childs in parallel"
          // << std::endl;
          //                boost::thread_group threads;
          //                if (left_child_) {
          //                    //left_child_->run();
          //                    threads.add_thread(new
          //                    boost::thread(boost::bind(&TypedOperator<OperatorInputTypeLeftChild>::run,
          //                    left_child_.get())));
          //                }
          //                if (right_child_) {
          //                    //right_child_->run();
          //                    threads.add_thread(new
          //                    boost::thread(boost::bind(&TypedOperator<OperatorInputTypeLeftChild>::run,
          //                    right_child_.get())));
          //                }
          //                threads.join_all();
          boost::thread_group threads;
          if (right_child_) {
            // right_child_->run();
            threads.add_thread(new boost::thread(
                boost::bind(&TypedOperator<OperatorInputTypeLeftChild>::run,
                            right_child_.get())));
          }
          // save one thread creating for one binary operator without loosing
          // parallelism
          if (left_child_) {
            // left_child_->run();
            threads.add_thread(new boost::thread(
                boost::bind(&TypedOperator<OperatorInputTypeLeftChild>::run,
                            left_child_.get())));
          }
          threads.join_all();
#else
          if (left_child_) {
            left_child_->run();
          }
          if (right_child_) {
            right_child_->run();
          }
#endif
        }
        // execute this operator
        bool retVal = (*this)();
        return retVal;
      }

      virtual void releaseInputData() {
        if (this->left_child_) this->left_child_->cleanupResult();
        if (this->right_child_) this->right_child_->cleanupResult();
      }

      virtual void print(std::ostream& out, unsigned int tree_level) const {
        for (unsigned int i = 0; i < tree_level; ++i) {
          out << "\t";
        }
        assert(!this->getFeatureValues().empty());
        out << this->getAlgorithmName() << " ET: "
            << this->getEstimatedExecutionTime().getTimeinNanoseconds()
            << "\tEC: " << this->getFeatureValues()[0]
            << (void*)this->logical_operator_.get()
            << std::endl;  // << " Features: " << this->getFeatureValues() <<
                           // std::endl;
        if (left_child_) left_child_->print(out, tree_level + 1);
        if (right_child_) right_child_->print(out, tree_level + 1);
      }

      //            virtual void printResult(unsigned int tree_level) {
      //                for (unsigned int i = 0; i < tree_level; ++i) {
      //                    out << "\t";
      //                }
      virtual void printResult(std::ostream& out, unsigned int tree_level,
                               bool show_timings, bool show_cardinalities,
                               bool show_estimation_error) {
        for (unsigned int i = 0; i < tree_level; ++i) {
          out << "\t";
        }
        if (this->sched_dec_.getDeviceSpecification().getDeviceType() ==
            hype::CPU) {
          out << "\033[44m";  // highlight following text in blue
        } else if (this->sched_dec_.getDeviceSpecification().getDeviceType() ==
                   hype::GPU) {
          out << "\033[42m";  // highlight following text in green
        }
        if (this->has_aborted_) {
          // color aborted gpu operators red
          out << "\033[41m";
        }
        assert(!this->getFeatureValues().empty());
        //               assert(this->logical_operator_!=NULL);
        //               out << this->logical_operator_->toString(true);
        //               //getAlgorithmName();
        out << this->getAlgorithmName();
        out << " ["
            << util::getName(
                   this->sched_dec_.getDeviceSpecification().getDeviceType())
            << (int)this->sched_dec_.getDeviceSpecification()
                   .getProcessingDeviceID()
            << "]";
        if (this->has_aborted_) {
          out << " [Aborted]";
        }
        out << "\033[0m";
        if (show_timings) {
          if (show_estimation_error) {
            out << "\t[Execution Time: ";
          } else {
            out << "\t[";
          }
          out << this->getTimeNeeded() / (1000 * 1000)
              << "ms]";  // << st::endl;
        }
        out << "\t[SD: " << this->getSchedulingDelay() / (1000 * 1000) << "ms]";
        if (show_estimation_error) {
          out << "\t[Estimated Time: "
              << this->getEstimatedExecutionTime().getTimeinNanoseconds() /
                     (1000 * 1000)
              << "ms]";  // << st::endl;
          out << "\t[Estimation Error: "
              << double(
                     this->getTimeNeeded() -
                     this->getEstimatedExecutionTime().getTimeinNanoseconds()) /
                     this->getTimeNeeded()
              << "ms]";
        }
        if (show_cardinalities) {
          out << ";\t[Result Size: " << this->getResultSize() << "rows]";
          out << ";\t[Estimated Result Size: " << this->getFeatureValues()[0]
              << "rows]";
        }

        assert(!this->getFeatureValues().empty());
        //                out << this->getAlgorithmName() << " ET: " <<
        //                this->getTimeEstimated()
        //                        << "\tMT: " << this->getTimeNeeded()
        //                        << "\tEE: " << 1 - (this->getTimeEstimated() /
        //                        this->getTimeNeeded())
        //                        << "\tEC: " << this->getFeatureValues()[0];
        //                if (this->getResultSize() != -1) {
        //                    out << "\tRR: " << this->getResultSize();
        //                }
        out << std::endl;
        if (left_child_)
          left_child_->printResult(out, tree_level + 1, show_timings,
                                   show_cardinalities, show_estimation_error);
        if (right_child_)
          right_child_->printResult(out, tree_level + 1, show_timings,
                                    show_cardinalities, show_estimation_error);
      }

      virtual double getRecursiveExecutionTimeInNanoseconds() {
        double executiontime = this->getTimeNeeded();
        if (executiontime < 0) {
          executiontime = 0;
        }
        // executiontime=this->getSchedulingDelay();
        //                if
        //                (!hype::core::Runtime_Configuration::instance().isQueryChoppingEnabled())
        //                {
        //                if (left_child_) {
        //                    executiontime +=
        //                    left_child_->getRecursiveExecutionTimeInNanoseconds();
        //                }
        //                if (right_child_) {
        //                    executiontime +=
        //                    right_child_->getRecursiveExecutionTimeInNanoseconds();
        //                }
        //                } else {

        double left_child_delay = 0;
        double right_child_delay = 0;
        if (left_child_) {
          uint64_t start_this_operator_timestamp = this->getBeginTimestamp();
          uint64_t end_child_operator_timestamp =
              left_child_->getEndTimestamp();
          // consider precision jitter of clock in assertion
          //	    		if(start_this_operator_timestamp+1000<end_child_operator_timestamp)
          //	    			HYPE_FATAL_ERROR("Current operator
          // startet
          // before child operator finished!", std::cout);
          // assume no time has passed in jitter interval, otherwise add the
          // scheduling delay
          if (start_this_operator_timestamp > end_child_operator_timestamp)
            left_child_delay = double(start_this_operator_timestamp -
                                      end_child_operator_timestamp);
        }
        if (right_child_) {
          uint64_t start_this_operator_timestamp = this->getBeginTimestamp();
          uint64_t end_child_operator_timestamp =
              right_child_->getEndTimestamp();
          // consider precision jitter of clock in assertion
          //	    		if(start_this_operator_timestamp+1000<end_child_operator_timestamp)
          //	    			HYPE_FATAL_ERROR("Current operator
          // startet
          // before child operator finished!", std::cout);
          // assume no time has passed in jitter interval, otherwise add the
          // scheduling delay
          if (start_this_operator_timestamp > end_child_operator_timestamp)
            right_child_delay = double(start_this_operator_timestamp -
                                       end_child_operator_timestamp);
        }

        if (left_child_ && right_child_) {
          if (left_child_->getRecursiveExecutionTimeInNanoseconds() +
                  left_child_delay >
              right_child_->getRecursiveExecutionTimeInNanoseconds() +
                  right_child_delay) {
            executiontime +=
                left_child_->getRecursiveExecutionTimeInNanoseconds() +
                left_child_delay;
          } else {
            executiontime +=
                right_child_->getRecursiveExecutionTimeInNanoseconds() +
                right_child_delay;
          }
        } else {
          if (left_child_) {
            executiontime +=
                left_child_->getRecursiveExecutionTimeInNanoseconds() +
                left_child_delay;
          }
          if (right_child_) {
            executiontime +=
                right_child_->getRecursiveExecutionTimeInNanoseconds() +
                right_child_delay;
          }
        }
        //}

        return executiontime;
      }

      virtual double getRecursiveEstimationTimeInNanoseconds() {
        double estimationTime = this->getTimeNeeded();
        if (estimationTime < 0) {
          estimationTime = 0;
        }
        if (!hype::core::Runtime_Configuration::instance()
                 .isQueryChoppingEnabled()) {
          if (left_child_) {
            estimationTime +=
                left_child_->getRecursiveEstimationTimeInNanoseconds();
          }
          if (right_child_) {
            estimationTime +=
                right_child_->getRecursiveEstimationTimeInNanoseconds();
          }
        } else {
          if (left_child_ && right_child_) {
            if (left_child_->getRecursiveEstimationTimeInNanoseconds() >
                right_child_->getRecursiveEstimationTimeInNanoseconds()) {
              estimationTime +=
                  left_child_->getRecursiveEstimationTimeInNanoseconds();
            } else {
              estimationTime +=
                  right_child_->getRecursiveEstimationTimeInNanoseconds();
            }
          } else {
            if (left_child_) {
              estimationTime +=
                  left_child_->getRecursiveEstimationTimeInNanoseconds();
            }
            if (right_child_) {
              estimationTime +=
                  right_child_->getRecursiveEstimationTimeInNanoseconds();
            }
          }
        }

        return estimationTime;
      }

      virtual double getTotalSchedulingDelay() {
        double ret = 0;
        double left_child_scheduling_delay = 0;
        double right_child_scheduling_delay = 0;
        if (left_child_) {
          left_child_scheduling_delay = left_child_->getTotalSchedulingDelay();
        }
        if (right_child_) {
          right_child_scheduling_delay =
              right_child_->getTotalSchedulingDelay();
        }
        // return sum of max latencies/delays until now, which is exactly the
        // penalty on the response time of the query
        ret =
            std::max(left_child_scheduling_delay, right_child_scheduling_delay);
        ret += getSchedulingDelay();
        return ret;
      }

      virtual double getSchedulingDelay() {
        double left_child_scheduling_delay = 0;
        double right_child_scheduling_delay = 0;
        double ret = 0;
        if (left_child_) {
          uint64_t start_this_operator_timestamp = this->getBeginTimestamp();
          uint64_t end_child_operator_timestamp =
              left_child_->getEndTimestamp();
          //                        if(start_this_operator_timestamp<=0){
          //                        HYPE_FATAL_ERROR("Invalid begin timestamp!
          //                        for " << this->getAlgorithmName(),
          //                        std::cout); }
          //                        if(end_child_operator_timestamp<=0){
          //                        HYPE_FATAL_ERROR("Invalid end timestamp! for
          //                        " << this->getAlgorithmName(), std::cout); }
          // consider precision jitter of clock in assertion
          //	    		if(start_this_operator_timestamp+1000<end_child_operator_timestamp)
          //	    			HYPE_FATAL_ERROR("Current operator
          // startet
          // before child operator finished!", std::cout);
          // assume no time has passed in jitter interval, otherwise add the
          // scheduling delay
          if (start_this_operator_timestamp > end_child_operator_timestamp)
            left_child_scheduling_delay += double(
                start_this_operator_timestamp - end_child_operator_timestamp);
        }
        if (right_child_) {
          uint64_t start_this_operator_timestamp = this->getBeginTimestamp();
          uint64_t end_child_operator_timestamp =
              right_child_->getEndTimestamp();
          // consider precision jitter of clock in assertion
          //	    		if(start_this_operator_timestamp+1000<end_child_operator_timestamp)
          //	    			HYPE_FATAL_ERROR("Current operator
          // startet
          // before child operator finished!", std::cout);
          //                        if(start_this_operator_timestamp<=0){
          //                        HYPE_FATAL_ERROR("Invalid begin timestamp!
          //                        for " << this->getAlgorithmName(),
          //                        std::cout); }
          //                        if(end_child_operator_timestamp<=0){
          //                        HYPE_FATAL_ERROR("Invalid end timestamp! for
          //                        " << this->getAlgorithmName(), std::cout); }
          // assume no time has passed in jitter interval, otherwise add the
          // scheduling delay
          if (start_this_operator_timestamp > end_child_operator_timestamp)
            right_child_scheduling_delay += double(
                start_this_operator_timestamp - end_child_operator_timestamp);
        }
        // the delay starts after the second child finished!
        ret =
            std::min(left_child_scheduling_delay, right_child_scheduling_delay);
        return ret;
      }

      /*
      virtual std::list<TypedOperator<OperatorOutputType>& > getOperatorQueue(){

              std::list<TypedOperator<OperatorOutputType>& > left_list =
      left_child_->getOperatorQueue();
              left_list.push_front(*this);

              std::list<TypedOperator<OperatorOutputType>& > right_list =
      right_child_->getOperatorQueue();

              left_list.insert(left_list.end(),right_list.begin(),right_list.end());
              return left_list;
              //this->left_child_->
      }*/

     protected:
      boost::shared_ptr<TypedOperator<OperatorInputTypeLeftChild> > left_child_;
      boost::shared_ptr<TypedOperator<OperatorInputTypeRightChild> >
          right_child_;

      // boost::shared_ptr<TypedOperator<OperatorOutputType> > succsessor_;
    };

  }  // end namespace queryprocessing
}  // end namespace hype
