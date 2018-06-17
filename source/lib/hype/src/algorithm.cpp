
#include <config/configuration.hpp>

#include <core/algorithm.hpp>
#include <core/operation.hpp>

#include "plugins/statistical_methods/knn_regression.hpp"

using namespace std;

namespace hype {
namespace core {

// template<class Archive, BufferType>
// void serialize(Archive & ar, boost::circular_buffer<BufferType> & circ_buf,
// const unsigned int version)
//{
//    ar & circ_buf.size();
//    for(unsigned )
//    ar & g.minutes;
//    ar & g.seconds;
//}

Algorithm::Algorithm(const std::string& name_of_algorithm,
                     const std::string& name_of_statistical_method,
                     const std::string& name_of_recomputation_strategy,
                     Operation& operation, DeviceSpecification comp_dev)
    : name_(name_of_algorithm),
      ptr_statistical_method_(
          getNewStatisticalMethodbyName(name_of_statistical_method)),
      ptr_recomputation_heristic_(
          getNewRecomputationHeuristicbyName(name_of_recomputation_strategy)),
      statistics_(AlgorithmStatisticsManager::instance().getAlgorithmStatistics(
          comp_dev, name_of_algorithm)),
      operation_(operation),
      logical_timestamp_of_last_execution_(0),
      is_in_retraining_phase_(false),
      retraining_length_(0),
      load_change_estimator_(10),
      comp_dev_(comp_dev),
      mem_cost_model_func_ptr_(NULL) {
  if (ptr_statistical_method_ == NULL || ptr_recomputation_heristic_ == NULL)
    std::cout << "Fatal Error!" << std::endl;

  // getNewStatisticalMethodbyName(name_of_statistical_method);

  std::stringstream ss;
  ss << Runtime_Configuration::instance().getOutputDirectory() << "/"
     << this->operation_.getName() << "/" << this->name_
     << ".performance_model";
  std::string path = ss.str();

  if (Runtime_Configuration::instance().getStorePerformanceModels()) {
    // do we have performance measurements stored on prior runs?
    if (this->statistics_->loadFeatureVectors(path)) {
      // ok, found obsevations of prior executions, compute initial
      // approximation function!
      this->ptr_statistical_method_->recomuteApproximationFunction(*this);
      this->is_in_retraining_phase_ = false;
    }
  }
}

Algorithm::~Algorithm() {
  using namespace boost::filesystem;

  if (Runtime_Configuration::instance().printAlgorithmStatistics()) {
    if (!statistics_->writeToDisc(operation_.getName(), name_))
      cout << "Error! Could not write statistics for algorithm" << name_
           << " to disc!" << endl;
  }
  if (Runtime_Configuration::instance().getStorePerformanceModels()) {
    std::string path = Runtime_Configuration::instance().getOutputDirectory();
    if (!exists(path)) {
      create_directory(path);
    }
    path += "/";
    path += this->operation_.getName();
    if (!exists(path)) {
      create_directory(path);
    }
    path += "/";
    path += this->name_;
    path += ".performance_model";
    this->statistics_->storeFeatureVectors(path);
  }
}

bool Algorithm::setStatisticalMethod(
    boost::shared_ptr<StatisticalMethod_Internal> ptr_statistical_method) {
  if (ptr_statistical_method) {
    ptr_statistical_method_.reset();
    ptr_statistical_method_ = ptr_statistical_method;
    // ptr_statistical_method_.reset(ptr_statistical_method);
    return true;
  } else {
    cout << "Error! in setStatisticalMethod(): ptr_statistical_method==NULL !"
         << endl;
    return false;
  }
}

bool Algorithm::inTrainingPhase() const throw() {
  return ptr_statistical_method_->inTrainingPhase();
}

bool Algorithm::inRetrainingPhase() const throw() {
  return is_in_retraining_phase_;
}

bool Algorithm::setRecomputationHeuristic(
    boost::shared_ptr<RecomputationHeuristic_Internal>
        ptr_recomputation_heristic) {
  if (ptr_recomputation_heristic) {
    ptr_recomputation_heristic_.reset();
    ptr_recomputation_heristic_ = ptr_recomputation_heristic;
    // ptr_recomputation_heristic_.reset(ptr_recomputation_heristic);
    return true;
  } else {
    cout << "Error! in setRecomputationHeuristic(): "
            "ptr_recomputation_heristic==NULL !"
         << endl;
    return false;
  }
  return true;
}

const std::string Algorithm::getName() const { return name_; }

bool Algorithm::addMeasurementPair(const MeasurementPair& mp) {
  // logical_timestamp_of_last_execution_=operation_.getNextTimestamp(); //after
  // each execution of algorithm, add new timestamp (logical time form logical
  // clock that is increased with every operation execution! -> Means, that each
  // algorithms increments the timestamp by calling getNextTimestamp())
  if (!this->inTrainingPhase())
    load_change_estimator_.add(mp);  // first we have to learn the current load
                                     // situation properly, afterwards, we can
                                     // compute meaningful Load modifications

  if (ptr_recomputation_heristic_->recompute(*this)) {
    if (!ptr_statistical_method_->recomuteApproximationFunction(*this)) {
      if (!quiet && verbose)
        cout << "Error while recomputing approximation function of algorithm '"
             << name_ << "'" << endl;
    } else {
      // success
      if (!quiet && verbose)
        cout << "Successfully recomputed approximation function of algorithm '"
             << name_ << "'" << endl;
      statistics_->number_of_recomputations_++;  // update statistics
    }
  }
  // if an observation was added, the model obviously decided to execute this
  // algorithm
  // this->statistics_->number_of_decisions_for_this_algorithm_++;

  this->statistics_->number_of_terminated_executions_of_this_algorithm_++;

  this->statistics_->total_execution_time_ +=
      mp.getMeasuredTime().getTimeinNanoseconds();

  double relative_error = -1;
  // only consider relative error in case the estimation is valid (!=0)
  // and the operator needed at least 1ms, because for very small data sets the
  // estimation error is typically very large, but this means no harm at all, we
  // rather
  // have accurate estimates over a large interval!
  if (mp.getEstimatedTime().getTimeinNanoseconds() != -1 &&
      mp.getMeasuredTime().getTimeinNanoseconds() > 1000 * 1000) {
    relative_error = (mp.getMeasuredTime().getTimeinNanoseconds() -
                      mp.getEstimatedTime().getTimeinNanoseconds()) /
                     mp.getEstimatedTime().getTimeinNanoseconds();
  }
  this->statistics_->relative_errors_.push_back(relative_error);

  //				(measured_time.getTimeinNanoseconds()-scheduling_decision_.getEstimatedExecutionTimeforAlgorithm().getTimeinNanoseconds())/scheduling_decision_.getEstimatedExecutionTimeforAlgorithm().getTimeinNanoseconds()
  //<< "%" << endl;

  if (is_in_retraining_phase_) {
    // if(Configuration::maximal_retraining_length>=retraining_length_++){
    if (Runtime_Configuration::instance().getRetrainingLength() >=
            retraining_length_++ ||
        !this->ptr_statistical_method_->inTrainingPhase()) {
      is_in_retraining_phase_ = false;
      if (!quiet) cout << "Finished retraining" << endl;
      // ptr_statistical_method_->recomuteApproximationFunction(*this);
    }
  }

  return statistics_->executionHistory_.addMeasurementPair(mp);
}

std::string Algorithm::toString(unsigned int indent) const {
  std::stringstream ss;
  std::string indent_str = "";
  for (unsigned int i = 0; i < indent; ++i) {
    indent_str.append("\t");
  }
  ss << indent_str << "Algorithm: " << this->name_
     << " (Operation: " << this->operation_.getName() << ")" << std::endl;
  ss << indent_str
     << "StatisticalMethod: " << this->ptr_statistical_method_->getName()
     << std::endl;
  ss << indent_str << "Recomputation Heuristic: "
     << this->ptr_recomputation_heristic_->getName() << std::endl;
  ss << indent_str << this->statistics_->getReport(operation_.getName(), name_);
  return ss.str();
}

const EstimatedTime Algorithm::getEstimatedExecutionTime(
    const Tuple& input_values) {
  //                            if(this->name_=="CPU_Groupby_Algorithm" ||
  //                            this->name_=="GPU_Groupby_Algorithm"){
  //                                std::cout << "GET ESTIMATION FROM KNN
  //                                REGRESSOR!" << std::endl;
  //                                return
  //                                ptr_statistical_method_->computeEstimation(input_values);
  //                            }
  if (ptr_statistical_method_->getName() == "KNN_Regression") {
    boost::shared_ptr<KNN_Regression> ptr =
        boost::dynamic_pointer_cast<KNN_Regression>(ptr_statistical_method_);
    if (ptr) {
      return ptr->computeEstimation(input_values);
    }
  }
  return ptr_statistical_method_->computeEstimation(input_values);
}

double Algorithm::getEstimatedRequiredMemoryCapacity(
    const Tuple& input_values) {
  if (this->mem_cost_model_func_ptr_) {
    return (*this->mem_cost_model_func_ptr_)(input_values);
  } else {
    return 0;
  }
}

unsigned int Algorithm::getNumberOfDecisionsforThisAlgorithm() const throw() {
  return this->statistics_->number_of_decisions_for_this_algorithm_;
}

unsigned int Algorithm::getNumberOfTerminatedExecutions() const throw() {
  return this->statistics_->number_of_terminated_executions_of_this_algorithm_;
}

/*! \brief returns the total time this algorithm spend in execution*/
double Algorithm::getTotalExecutionTime() const throw() {
  return statistics_->total_execution_time_;
}

uint64_t Algorithm::getTimeOfLastExecution() const throw() {
  return logical_timestamp_of_last_execution_;
}

void Algorithm::setTimeOfLastExecution(uint64_t new_timestamp) throw() {
  this->logical_timestamp_of_last_execution_ = new_timestamp;
}

void Algorithm::incrementNumberofDecisionsforThisAlgorithm() throw() {
  this->statistics_->number_of_decisions_for_this_algorithm_++;
}

void Algorithm::retrain() {
  if (!quiet) std::cout << "Retrain algorithm " << name_ << std::endl;
  is_in_retraining_phase_ = true;
  retraining_length_ = 0;
  // ptr_statistical_method_->retrain();

  // this->statistics_.executionHistory_.clear();
}

const LoadChangeEstimator& Algorithm::getLoadChangeEstimator() const throw() {
  return load_change_estimator_;
}

const DeviceSpecification Algorithm::getDeviceSpecification() const throw() {
  return comp_dev_;
}

void Algorithm::setMemoryCostModel(
    MemoryCostModelFuncPtr mem_cost_model_func_ptr) {
  this->mem_cost_model_func_ptr_ = mem_cost_model_func_ptr;
}

}  // end namespace core
}  // end namespace hype
