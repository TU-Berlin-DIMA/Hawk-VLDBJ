
#pragma once

#include <list>

#include <config/global_definitions.hpp>

#include <core/specification.hpp>
#include <query_processing/device_operator_queue.hpp>
#include <query_processing/operator.hpp>

// g++ compiler workaround for boost thread!
#ifdef __GNUC__
#pragma GCC visibility push(default)
#endif
#include <boost/bind.hpp>
#include <boost/thread.hpp>
// g++ compiler workaround for boost thread!
#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#define HYPE_ALTERNATE_QUERY_CHOPPING

namespace hype {
  namespace queryprocessing {

    class ProcessingDevice {
     public:
      ProcessingDevice(const core::DeviceSpecification& dev_spec);

      bool addOperator(OperatorPtr op);

      void start();
      void stop();
      void run();  // thread
      bool isIdle();
      /*! \todo Implement function! sum all Estimated Execution times of
       * operators in operator list! OR better updated on add/remove from queue!
       * Finally, add the rest time of current oeprator, which is provided by
       * getEstimatedTimeUntilOperatorCompletion()*/
      double getEstimatedTimeUntilIdle();
      double getTotalProcessingTime() const;
      ProcessingDeviceID getProcessingDeviceID() const throw();

     private:
      ProcessingDevice(const ProcessingDevice& pd);
      ProcessingDevice& operator=(const ProcessingDevice&);
      /*! \todo Implement function! store start timestamp of operator currently
       * running and the Estiamted Execution Time for him -> compute rest time
       * using current timestamp*/
      double getEstimatedTimeUntilOperatorCompletion();

      typedef std::list<OperatorPtr> OperatorList;
      OperatorList operators_;
      boost::condition_variable new_operator_available_;
      boost::condition_variable operator_queue_full_;
      mutable boost::mutex operator_mutex_;
      boost::thread thread_;
      uint64_t start_timestamp_of_currently_executed_operator_;
      double estimated_execution_time_of_currently_executed_operator_;
      bool operator_in_execution_;
      double estimated_execution_time_for_operators_in_queue_to_complete_;
      double
          total_measured_execution_time_for_all_executed_operators_;  // used to
      // determine
      // how
      // processing
      // devices
      // were
      // utilized
      const ProcessingDeviceID proc_dev_id_;
      DeviceOperatorQueuePtr device_operator_queue_;
    };
    typedef boost::shared_ptr<ProcessingDevice> ProcessingDevicePtr;
    /*! */
    ProcessingDevicePtr getProcessingDevice(
        const hype::core::DeviceSpecification&
            dev);  // can later be queried by using
                   // algorithmptr->getDeviceSpecification();

  }  // end namespace queryprocessing
}  // end namespace hype
