
#pragma once

#include <core/scheduling_decision.hpp>
#include <core/specification.hpp>
#include <core/time_measurement.hpp>

#include <boost/shared_ptr.hpp>

namespace hype {
  namespace queryprocessing {

    class Node;
    typedef boost::shared_ptr<Node> NodePtr;

    class Operator {
     public:
      bool operator()();
      const hype::core::SchedulingDecision& getSchedulingDecision() const
          throw();
      virtual ~Operator();

     protected:
      Operator(const hype::core::SchedulingDecision&
                   sched_dec);  // : sched_dec_(sched_dec) {}
      Operator(const hype::core::SchedulingDecision& sched_dec,
               NodePtr logical_operator);

     public:
      const core::EstimatedTime getEstimatedExecutionTime() const throw();

      NodePtr getLogicalOperator() throw();

      virtual double getRecursiveEstimationTimeInNanoseconds() = 0;
      virtual double getRecursiveExecutionTimeInNanoseconds() = 0;

      void setTimeNeeded(double timeNeeded);
      void setTimeEstimated(double timeEstimated);
      void setResultSize(double result_size);

      double getTimeNeeded() const;
      double getTimeEstimated() const;
      double getResultSize() const;

      uint64_t getBeginTimestamp() const;
      uint64_t getEndTimestamp() const;

      bool hasAborted() const;

      const std::string getAlgorithmName() const throw();

      const core::Tuple getFeatureValues() const throw();

      const core::DeviceSpecification getDeviceSpecification() const throw();
      void setLogicalOperator(NodePtr logical_operator);

      virtual void setOutputStream(std::ostream& output_stream) = 0;

      virtual void releaseInputData() = 0;
      virtual bool isInputDataCachedInGPU();
      virtual double getTotalSchedulingDelay() = 0;
      virtual double getSchedulingDelay() = 0;

      // void notifyParent();
     private:
      virtual bool execute() = 0;

     protected:
      hype::core::SchedulingDecision sched_dec_;
      NodePtr logical_operator_;
      double timeNeeded;
      double timeEstimated;
      double result_size_;
      bool has_aborted_;
      uint64_t start_timestamp_;
      uint64_t end_timestamp_;
    };

    typedef boost::shared_ptr<Operator> OperatorPtr;

  }  // end namespace queryprocessing
}  // end namespace hype
