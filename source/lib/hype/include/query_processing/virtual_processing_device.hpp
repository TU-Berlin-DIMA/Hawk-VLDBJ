
#pragma once

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <list>

#include <config/global_definitions.hpp>
#include <core/scheduling_decision.hpp>
#include <core/specification.hpp>

namespace hype {
  namespace queryprocessing {

    class VirtualProcessingDevice {
      typedef std::list<core::SchedulingDecision> TaskQueue;

     public:
      explicit VirtualProcessingDevice(
          const core::DeviceSpecification& dev_spec_);

      bool addRunningOperation(
          const core::SchedulingDecision&);  // called by getOptimalAlgorithm

      bool removeFinishedOperation(
          const core::SchedulingDecision&);  // called by addObservation

      unsigned int getNumberOfRunningOperations() const;

      double getEstimatedFinishingTime() const;

      bool isIdle() const;

      const core::DeviceSpecification& getDeviceSpecification() const throw();

      void print() const throw();

     private:
      core::DeviceSpecification dev_spec_;
      std::list<core::SchedulingDecision> scheduled_tasks_;
      mutable boost::mutex mutex_;
    };

    typedef boost::shared_ptr<VirtualProcessingDevice>
        VirtualProcessingDevicePtr;

    // VirtualProcessingDevice& getVirtualProcessingDevice(ProcessingDeviceID);

  }  // end namespace queryprocessing
}  // end namespace hype
