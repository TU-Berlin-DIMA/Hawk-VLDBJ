
#include <hype.hpp>
#include <query_processing/operator.hpp>
#include "query_processing/node.hpp"

#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
//    #include <cogadb/include/statistics/statistics_manager.hpp>
//    #include <cogadb/include/util/hardware_detector.hpp>
//    #include <cogadb/include/util/functions.hpp>
#include <statistics/statistics_manager.hpp>
#include <util/functions.hpp>
#include <util/hardware_detector.hpp>
#endif

namespace hype {
namespace queryprocessing {

using namespace core;

bool Operator::operator()() {
  // double begin = double(hype::core::getTimestamp());
  this->start_timestamp_ = hype::core::getTimestamp();
  double begin = double(this->start_timestamp_);

  bool ret = false;
  // try{
  if (!hype::core::quiet && hype::core::verbose && hype::core::debug) {
    std::cout << "[Operator] Executing " << this->getAlgorithmName();
    if (this->logical_operator_) {
      std::cout << " for logical operator: "
                << this->logical_operator_->toString(1);
    }
    std::cout << std::endl;
#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
    double used_memory_cpu_in_gb =
        double(CoGaDB::getUsedMainMemoryInBytes()) / (1024 * 1024 * 1024);
    double free_memory_gpu_in_gb =
        double(CoGaDB::HardwareDetector::instance().getFreeMemorySizeInByte(
            hype::PD_Memory_1)) /
        (1024 * 1024 * 1024);
    double total_memory_gpu_in_gb =
        double(CoGaDB::HardwareDetector::instance().getTotalMemorySizeInByte(
            hype::PD_Memory_1)) /
        (1024 * 1024 * 1024);
    std::cout << "\t Current Memory CPU: " << used_memory_cpu_in_gb << "GiB )"
              << std::endl;
    std::cout << "\t Current Memory GPU: "
              << (total_memory_gpu_in_gb - free_memory_gpu_in_gb) << "GiB )"
              << std::endl;
#endif
    std::cout << std::endl;
  }

  /* Are we on the CPU? */
  if (!util::isCoprocessor(
          this->sched_dec_.getDeviceSpecification().getProcessingDeviceID())) {
    hype::AlgorithmMeasurement alg_measure(sched_dec_);
    // call user defined execution function
    try {
      ret = execute();
    } catch (const std::bad_alloc& e) {
      std::cout << "Catched bad alloc for operator "
                << this->logical_operator_->toString(true) << std::endl;
      ret = false;
    }
    if (ret) {
      alg_measure.afterAlgorithmExecution();
    } else {
      HYPE_FATAL_ERROR(
          "A Non Co-Processor Operator Aborted, which is forbidden!"
              << std::endl
              << "Operator: " << this->logical_operator_->toString(true)
              << std::endl
              << "Physical Operator: " << this->getAlgorithmName(),
          std::cerr);
    }
  } else {
// executed on a co-processor
#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
    using namespace CoGaDB;
    assert(this->logical_operator_ != NULL);
    assert(util::isCoprocessor(this->sched_dec_.getDeviceSpecification()
                                   .getProcessingDeviceID()) == true);
    COGADB_EXECUTE_GPU_OPERATOR(this->logical_operator_->getOperationName());

#endif
    hype::AlgorithmMeasurement alg_measure(sched_dec_);
    // call user defined execution function
    try {
      ret = execute();
    } catch (const std::bad_alloc& e) {
      std::cout << "Catched bad alloc for operator "
                << this->logical_operator_->toString(true) << std::endl;
      ret = false;
    }
    if (ret) {
      alg_measure.afterAlgorithmExecution();
    } else {
#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
      // mark this operator as aborted
      this->has_aborted_ = true;
      COGADB_ABORT_GPU_OPERATOR(this->logical_operator_->getOperationName());
#endif
      // restart operator on CPU
      OperatorSpecification op_spec(this->logical_operator_->getOperationName(),
                                    this->logical_operator_->getFeatureVector(),
                                    hype::PD_Memory_0, hype::PD_Memory_0);

      HYPE_WARNING("Operator '"
                       << this->logical_operator_->toString(true)
                       << "' running on processor "
                       << (int)sched_dec_.getDeviceSpecification()
                              .getProcessingDeviceID()
                       << " aborted: Starting Fallback operator on CPU..."
#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
                       << " (Current Memory: "
                       << double(CoGaDB::getUsedMainMemoryInBytes()) /
                              (1024 * 1024 * 1024)
                       << "GiB )"
#endif
                   ,
                   std::cout);
      SchedulingDecision cpu_sched_dec =
          core::Scheduler::instance().getOptimalAlgorithm(op_spec,
                                                          hype::CPU_ONLY);
      sched_dec_ = cpu_sched_dec;

      ret = this->execute();
    }
  }

  //#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
  //                        using namespace CoGaDB;
  //                        assert(this->logical_operator_!=NULL);
  //                        if(util::isCoprocessor(this->sched_dec_.getDeviceSpecification().getProcessingDeviceID())){
  //                            COGADB_EXECUTE_GPU_OPERATOR(this->logical_operator_->getOperationName());
  //                        }
  //#endif
  //			hype::AlgorithmMeasurement alg_measure(sched_dec_);
  //			//call user defined execution function
  //                        try{
  //                            ret = execute();
  //                        }catch(const std::bad_alloc& e){
  //                            std::cout << "Catched bad alloc for operator "
  //                            << this->logical_operator_->toString(true) <<
  //                            std::endl;
  //                            ret = false;
  //                        }
  //                        if(ret){
  //                            alg_measure.afterAlgorithmExecution();
  //                        }else{
  //#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
  //                            //mark this operator as aborted
  //                            this->has_aborted_=true;
  //                            if(util::isCoprocessor(this->sched_dec_.getDeviceSpecification().getProcessingDeviceID())){
  //                                COGADB_ABORT_GPU_OPERATOR(this->logical_operator_->getOperationName());
  //                            }
  //#endif
  //                            if(!util::isCoprocessor(this->sched_dec_.getDeviceSpecification().getProcessingDeviceID())){
  //                                HYPE_FATAL_ERROR("A Non Co-Processor
  //                                Operator Aborted, which is forbidden!" <<
  //                                std::endl
  //                                              << "Operator: " <<
  //                                              this->logical_operator_->toString(true)
  //                                              << std::endl
  //                                              << "Physical Operator: " <<
  //                                              this->getAlgorithmName(),std::cerr);
  //                            }
  ////
  /// assert(util::isCoprocessor(this->sched_dec_.getDeviceSpecification().getProcessingDeviceID()));
  //                            //restart operator on CPU
  //                            OperatorSpecification
  //                            op_spec(this->logical_operator_->getOperationName(),
  //                            this->logical_operator_->getFeatureVector(),
  //                            hype::PD_Memory_0, hype::PD_Memory_0);
  //
  //                            HYPE_WARNING("Operator '" <<
  //                            this->logical_operator_->toString(true) << "'
  //                            running on processor " <<
  //                            (int)sched_dec_.getDeviceSpecification().getProcessingDeviceID()
  //                            << " aborted: Starting Fallback operator on
  //                            CPU..."
  //#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
  //                                    << " (Current Memory: " <<
  //                                    double(CoGaDB::getUsedMainMemoryInBytes())/(1024*1024*1024)
  //                                    << "GiB )"
  //#endif
  //                                    ,std::cout);
  //                            SchedulingDecision
  //                            cpu_sched_dec=core::Scheduler::instance().getOptimalAlgorithm(op_spec,
  //                            hype::CPU_ONLY);
  //                            sched_dec_=cpu_sched_dec;
  //
  //                            ret = this->execute();
  //
  //                        }
  // free consumed input
  this->releaseInputData();
  this->end_timestamp_ = hype::core::getTimestamp();
  if (logical_operator_) {
    // tell operator that it finished
    logical_operator_->notify();
    // tell Parent node that one child is finished
    if (logical_operator_->getParent())
      logical_operator_->getParent()->notify(logical_operator_);
  }

  if (!hype::core::quiet && hype::core::verbose && hype::core::debug) {
    std::cout << "[Operator] Finished " << this->getAlgorithmName();
    if (this->logical_operator_) {
      std::cout << " for logical operator: "
                << this->logical_operator_->toString(1);
    }
#ifdef HYPE_ENABLE_INTERACTION_WITH_COGADB
    std::cout << " (Current Memory: "
              << double(CoGaDB::getUsedMainMemoryInBytes()) /
                     (1024 * 1024 * 1024)
              << "GiB )" << std::endl;
#endif
    //<< sched_dec_.getNameofChoosenAlgorithm()
    std::cout << std::endl;
  }

  // set logical_operator to NULL, because we do not need it anymore
  logical_operator_ = NodePtr();
  this->end_timestamp_ = hype::core::getTimestamp();
  this->setTimeNeeded(double(this->end_timestamp_) - begin);

  return ret;
}

const hype::core::SchedulingDecision& Operator::getSchedulingDecision() const
    throw() {
  return this->sched_dec_;
}

Operator::~Operator() {
  if (logical_operator_) {
    if (logical_operator_->getLeft()) logical_operator_->setLeft(NodePtr());
    if (logical_operator_->getRight()) logical_operator_->setRight(NodePtr());
    if (logical_operator_->getParent()) logical_operator_->setParent(NodePtr());
    logical_operator_.reset();
  }
}

Operator::Operator(const hype::SchedulingDecision& sched_dec)
    : sched_dec_(sched_dec),
      logical_operator_(),
      timeNeeded(0),
      timeEstimated(0),
      result_size_(0),
      has_aborted_(false),
      start_timestamp_(0),
      end_timestamp_(0) {}

Operator::Operator(const hype::core::SchedulingDecision& sched_dec,
                   NodePtr logical_operator)
    : sched_dec_(sched_dec),
      logical_operator_(logical_operator),
      timeNeeded(0),
      timeEstimated(0),
      result_size_(0),
      has_aborted_(false),
      start_timestamp_(0),
      end_timestamp_(0) {}

const core::EstimatedTime Operator::getEstimatedExecutionTime() const throw() {
  return sched_dec_.getEstimatedExecutionTimeforAlgorithm();
}

NodePtr Operator::getLogicalOperator() throw() {
  return this->logical_operator_;
}

const std::string Operator::getAlgorithmName() const throw() {
  return sched_dec_.getNameofChoosenAlgorithm();
}

const core::Tuple Operator::getFeatureValues() const throw() {
  return sched_dec_.getFeatureValues();
}

//		DeviceSpecification Operator::getComputeDevice() const throw(){
//			return sched_dec_.getComputeDevice();
//		}

const core::DeviceSpecification Operator::getDeviceSpecification() const
    throw() {
  return sched_dec_.getDeviceSpecification();
}

void Operator::setLogicalOperator(NodePtr logical_operator) {
  logical_operator_ = logical_operator;
}

bool Operator::isInputDataCachedInGPU() { return false; }

void Operator::setTimeNeeded(double timeNeeded) {
  this->timeNeeded = timeNeeded;
}

void Operator::setTimeEstimated(double timeEstimated) {
  this->timeEstimated = timeEstimated;
}

void Operator::setResultSize(double result_size) { result_size_ = result_size; }

double Operator::getTimeNeeded() const { return this->timeNeeded; }

double Operator::getTimeEstimated() const { return this->timeEstimated; }

double Operator::getResultSize() const { return result_size_; }

uint64_t Operator::getBeginTimestamp() const { return this->start_timestamp_; }

uint64_t Operator::getEndTimestamp() const { return this->end_timestamp_; }

bool Operator::hasAborted() const { return this->has_aborted_; }

}  // end namespace queryprocessing
}  // end namespace hype
