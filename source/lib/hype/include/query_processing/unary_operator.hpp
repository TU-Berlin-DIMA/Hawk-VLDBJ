#pragma once

//#include <core/scheduling_decision.hpp>

//#include <boost/shared_ptr.hpp>
#include <config/configuration.hpp>
#include <config/global_definitions.hpp>
#include <query_processing/typed_operator.hpp>

namespace hype {
  namespace queryprocessing {

    template <typename OperatorInputType,
              typename OperatorOutputType>  //, typename TLogicalOperation>
    class UnaryOperator
        : public TypedOperator<OperatorOutputType> {  // uses 1D Learning Method

     public:
      // typedef typename
      // OperatorMapper_Helper_Template<OperatorOutputType>::TypedOperator
      // TypedOperator;
      typedef typename OperatorMapper_Helper_Template<
          OperatorOutputType>::TypedOperatorPtr TypedOperatorPtr;
      // typedef typename
      // OperatorMapper_Helper_Template<OperatorOutputType>::TypedNodePtr
      // TypedNodePtr;

      // typedef TLogicalOperation LogicalOperationType;

      UnaryOperator(const hype::SchedulingDecision& sched_dec,
                    TypedOperatorPtr child)
          : TypedOperator<OperatorOutputType>(sched_dec), child_(child) {}

      virtual ~UnaryOperator() {}

      OperatorInputType getInputData() {
        if (child_)
          return child_->getResult();
        else
          return OperatorInputType();
      }

      virtual bool run() {
        if (!hype::core::Runtime_Configuration::instance()
                 .isQueryChoppingEnabled() &&
            child_) {
          child_->run();
        }
        // execute this operator
        this->setTimeEstimated(
            double(this->getEstimatedExecutionTime().getTimeinNanoseconds()));
        // double begin = double(hype::core::getTimestamp());
        bool retVal = (*this)();
        // this->setTimeNeeded(double( hype::core::getTimestamp()) - begin);
        return retVal;
      }

      virtual void releaseInputData() {
        if (this->child_) this->child_->cleanupResult();
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
        if (child_) child_->print(out, tree_level + 1);
      }

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

        //                         << " ET: " <<  this->getTimeEstimated()
        //                         << "\tMT: " << this->getTimeNeeded()
        //                         << "\tEE: " << 1 - (this->getTimeEstimated()
        //                         / this->getTimeNeeded())
        //                         << "\tEC: " << this->getFeatureValues()[0];
        //                if (this->getResultSize() != -1) {
        //                    out << "\tRR: " << this->getResultSize();
        //                }
        out << std::endl;
        if (child_)
          child_->printResult(out, tree_level + 1, show_timings,
                              show_cardinalities, show_estimation_error);
      }

      virtual double getRecursiveExecutionTimeInNanoseconds() {
        double executiontime = this->getTimeNeeded();
        if (executiontime < 0) {
          executiontime = 0;
        }
        if (child_) {
          executiontime += child_->getRecursiveExecutionTimeInNanoseconds();
        }
        return executiontime;
      }

      virtual double getRecursiveEstimationTimeInNanoseconds() {
        double estimationTime = this->getTimeEstimated();
        if (estimationTime < 0) {
          estimationTime = 0;
        }
        if (child_) {
          estimationTime += child_->getRecursiveEstimationTimeInNanoseconds();
        }
        return estimationTime;
      }

      virtual double getTotalSchedulingDelay() {
        double ret = 0;
        if (child_) {
          ret = child_->getTotalSchedulingDelay();
          ret += getSchedulingDelay();
        }
        return ret;
      }

      virtual double getSchedulingDelay() {
        double ret = 0;
        if (child_) {
          uint64_t start_this_operator_timestamp = this->getBeginTimestamp();
          uint64_t end_child_operator_timestamp = child_->getEndTimestamp();
          //	    		std::cout << "start_this_operator_timestamp: "
          //<<
          // start_this_operator_timestamp << std::endl;
          //	    		std::cout << "end_child_operator_timestamp: " <<
          // end_child_operator_timestamp << std::endl;
          //                        if(start_this_operator_timestamp<=0){
          //                        HYPE_FATAL_ERROR("Invalid begin timestamp!
          //                        for " << this->getAlgorithmName(),
          //                        std::cout); }
          //                        if(end_child_operator_timestamp<=0){
          //                        HYPE_FATAL_ERROR("Invalid end timestamp! for
          //                        " << this->getAlgorithmName(), std::cout); }

          // consider precision jitter of clock in assertion
          //	    		if(start_this_operator_timestamp+10000<end_child_operator_timestamp)
          //	    			HYPE_FATAL_ERROR("Current operator
          // startet
          // before child operator finished!", std::cout);
          // assume no time has passed in jitter interval, otherwise add the
          // scheduling delay
          if (start_this_operator_timestamp > end_child_operator_timestamp)
            ret += double(start_this_operator_timestamp -
                          end_child_operator_timestamp);
        }
        return ret;
      }

      /*
      virtual std::list<TypedOperator<OperatorOutputType>& > getOperatorQueue(){
              std::list<TypedOperator<OperatorOutputType>& > list =
      child_->getOperatorQueue();
              list.push_front(*this);
              return list;
      }*/
      // private:
     protected:
      boost::shared_ptr<TypedOperator<OperatorInputType> > child_;
      // boost::shared_ptr<TypedOperator<OperatorOutputType> > succsessor_;
    };

  }  // end namespace queryprocessing
}  // end namespace hype
