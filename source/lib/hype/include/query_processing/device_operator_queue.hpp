/*
 * File:   device_operator_queue.hpp
 * Author: sebastian
 *
 * Created on 21. September 2014, 19:07
 */

#ifndef DEVICE_OPERATOR_QUEUE_HPP
#define DEVICE_OPERATOR_QUEUE_HPP

#include <assert.h>
#include <list>

#include <config/global_definitions.hpp>

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

namespace hype {
  namespace queryprocessing {

    class DeviceOperatorQueue {
     public:
      DeviceOperatorQueue(const ProcessingDeviceMemoryID mem_id);
      OperatorPtr getNextOperator();
      bool addOperator(OperatorPtr op);
      void notify_all();

     private:
      typedef std::list<OperatorPtr> OperatorList;
      OperatorList operators_;
      boost::condition_variable new_operator_available_;
      boost::condition_variable operator_queue_full_;
      boost::mutex operator_mutex_;
      const ProcessingDeviceMemoryID mem_id_;
      double estimated_execution_time_for_operators_in_queue_to_complete_;
    };
    typedef boost::shared_ptr<DeviceOperatorQueue> DeviceOperatorQueuePtr;

    class DeviceOperatorQueues {
     public:
      static DeviceOperatorQueues& instance();
      DeviceOperatorQueuePtr getDeviceOperatorQueue(
          const ProcessingDeviceMemoryID mem_id);

     private:
      DeviceOperatorQueues();
      DeviceOperatorQueues(const DeviceOperatorQueues&);
      DeviceOperatorQueues& operator=(const DeviceOperatorQueues&);
      typedef std::map<ProcessingDeviceMemoryID, DeviceOperatorQueuePtr>
          DeviceOperatorQueueMap;
      DeviceOperatorQueueMap map_;
    };

    //                   DeviceOperatorQueue::DeviceOperatorQueue(const
    //                   ProcessingDeviceMemoryID mem_id)
    //                   : operators_(), new_operator_available_(),
    //                   operator_queue_full_(), operator_mutex_(),
    //                     mem_id_(mem_id),
    //                     estimated_execution_time_for_operators_in_queue_to_complete_(0)
    //                   {
    //
    //                   }
    //                   //Note that this method locks the scoped lock 'lock',
    //                   and if this function returns, 'lock' is locked
    //                   OperatorPtr
    //                   DeviceOperatorQueue::getNextOperator(){//boost::mutex::scoped_lock&
    //                   lock){
    //                       		boost::mutex::scoped_lock
    //                       lock(operator_mutex_);
    //				while(operators_.empty()) {
    //					new_operator_available_.wait(lock);
    //				}
    //                                OperatorPtr op = operators_.front();
    //                                //delete this operator
    //                                operators_.pop_front();
    //                                return op;
    //                   }
    //                   bool DeviceOperatorQueue::addOperator(OperatorPtr op){
    //			assert(op!=NULL);
    //			//boost::mutex::scoped_lock lock(operator_mutex_);
    //			//boost::lock_guard<boost::mutex> lock(operator_mutex_);
    //			boost::mutex::scoped_lock lock(operator_mutex_);
    //			while(operators_.size()>hype::core::Runtime_Configuration::instance().getMaximalReadyQueueLength()){//10
    ////100) {
    //				operator_queue_full_.wait(lock);
    //			}
    //			operators_.push_back(op);
    //			estimated_execution_time_for_operators_in_queue_to_complete_+=std::max(op->getSchedulingDecision().getEstimatedExecutionTimeforAlgorithm().getTimeinNanoseconds(),double(0));
    //			if(!hype::core::quiet){
    //			cout << "new waiting time for Algorithm " <<
    // op->getSchedulingDecision().getNameofChoosenAlgorithm() << ": " <<
    // estimated_execution_time_for_operators_in_queue_to_complete_ << "ns" <<
    // endl;
    //			cout << "number of queued operators: " <<
    // this->operators_.size()
    //<<
    //" for Algorithm " <<
    // op->getSchedulingDecision().getNameofChoosenAlgorithm() << endl;
    //			}
    //			new_operator_available_.notify_all();
    //			return true;
    //                    }
    //
    //                    static DeviceOperatorQueues&
    //                    DeviceOperatorQueues::instance(){
    //                        static DeviceOperatorQueues queues;
    //                        return queues;
    //                    }
    //
    //                    DeviceOperatorQueuePtr
    //                    DeviceOperatorQueues::getDeviceOperatorQueue(const
    //                    ProcessingDeviceMemoryID mem_id){
    //                        DeviceOperatorQueueMap::const_iterator
    //                        cit=map_.find(mem_id);
    //                        if(cit!=map_.end()){
    //                            return cit->second;
    //                        }else{
    //                             DeviceOperatorQueuePtr queue = (new
    //                             DeviceOperatorQueue(mem_id));
    //                             map_[mem_id]=queue;
    //                             return queue;
    //                        }
    //                    }
    //
    //

  }  // end namespace queryprocessing
}  // end namespace hype

#endif /* DEVICE_OPERATOR_QUEUE_HPP */
