
#include <core/operation.hpp>
#include <core/plotscriptgenerator.hpp>

namespace hype {
namespace core {

Operation::Operation(const std::string& name)
    : map_algorithmname_to_pointer_(),
      ptr_to_optimization_criterion_(),
      name_(name),
      number_of_right_decisions_(0),
      number_of_total_decisions_(0),
      logical_time_(0) {
  this->setNewOptimizationCriterion("Response Time");
}

Operation::~Operation() {
  std::string title =
      std::string("Execution time curves for Operation ") + name_;
  std::vector<std::string> algorithm_names;

  MapNameToAlgorithm::iterator it;
  for (it = map_algorithmname_to_pointer_.begin();
       it != map_algorithmname_to_pointer_.end(); it++) {
    algorithm_names.push_back(it->second->getName());
  }

  if (!PlotScriptGenerator::create(
          title, "Data size in number of Elements (int)",
          "Execution time in ns", name_, algorithm_names)) {
    std::cout << "Error Creating Gnuplot Skript for Operation" << name_ << "!"
              << std::endl;
  }

  if (!PlotScriptGenerator::createRelativeErrorScript(name_, algorithm_names)) {
    std::cout
        << "Error Creating Gnuplot Skript for relative errors for Operation"
        << name_ << "!" << std::endl;
  }

  if (!PlotScriptGenerator::createAverageRelativeErrorScript(name_,
                                                             algorithm_names)) {
    std::cout << "Error Creating Gnuplot Skript for average relative errors "
                 "for Operation"
              << name_ << "!" << std::endl;
  }

  if (!PlotScriptGenerator::createWindowedAverageRelativeErrorScript(
          name_, algorithm_names)) {
    std::cout << "Error Creating Gnuplot Skript for windowed average relative "
                 "errors for Operation"
              << name_ << "!" << std::endl;
  }
}

/*
void Operation::addAlgorithm(std::string name){



}*/

bool Operation::addAlgorithm(
    const std::string& name_of_algorithm, DeviceSpecification comp_dev,
    const std::string& name_of_statistical_method,
    const std::string& name_of_recomputation_strategy) {
  std::map<std::string, boost::shared_ptr<Algorithm> >::iterator it;
  it = map_algorithmname_to_pointer_.find(name_of_algorithm);
  // map_algorithmname_to_pointer_;
  // boost::shared_ptr<Algorithm> alg;

  if (it == map_algorithmname_to_pointer_.end()) {  // algorithm does not exist
    // alg = boost::shared_ptr<Algorithm>(new
    // Algorithm(name_of_algorithm,name_of_statistical_method,name_of_recomputation_strategy,*this));
    // map_algorithmname_to_pointer_[name_of_algorithm]=alg;
    if (!quiet && verbose && debug)
      std::cout << "add Algorithm '" << name_of_algorithm << "' to operation '"
                << name_ << "'" << std::endl;
    map_algorithmname_to_pointer_[name_of_algorithm] =
        boost::shared_ptr<Algorithm>(
            new Algorithm(name_of_algorithm, name_of_statistical_method,
                          name_of_recomputation_strategy, *this, comp_dev));
    return true;

  } else {
    // alg=it->second; //boost::shared_ptr<Operation> (it->second);
    return false;  // algorithm already exists!
  }
}

void Operation::removeAlgorithm(const std::string& name) {
  std::cout << "Error: Operation::removeAlgorithm() not yet implemented!"
            << std::endl;
  std::cout << name << std::endl;
  std::exit(-1);
}

const AlgorithmPtr Operation::getAlgorithm(
    const std::string& name_of_algorithm) {
  MapNameToAlgorithm::iterator it =
      map_algorithmname_to_pointer_.find(name_of_algorithm);
  if (it == map_algorithmname_to_pointer_.end()) {
    return AlgorithmPtr();  // return NULL Pointer
  }
  return it->second;
}

const std::vector<AlgorithmPtr> Operation::getAlgorithms() {
  std::vector<AlgorithmPtr> result;
  MapNameToAlgorithm::const_iterator it;
  for (it = map_algorithmname_to_pointer_.begin();
       it != map_algorithmname_to_pointer_.end(); it++) {
    result.push_back(it->second);
  }
  return result;
}

bool Operation::setNewOptimizationCriterion(
    const std::string& name_of_optimization_criterion) {
  // calls factory function
  ptr_to_optimization_criterion_ =
      boost::shared_ptr<OptimizationCriterion_Internal>(
          getNewOptimizationCriterionbyName(name_of_optimization_criterion));
  if (!ptr_to_optimization_criterion_) {
    std::cout << "FATEL ERROR! Operation " << name_
              << " has no assigned optimization criterion! Exiting..."
              << std::endl;
    exit(-1);
  }
  return true;
}

// const std::vector< boost::shared_ptr<Algorithm> >
const SchedulingDecision Operation::getOptimalAlgorithm(
    const Tuple& input_values, DeviceTypeConstraint dev_constr) {
  if (!ptr_to_optimization_criterion_) {
    std::cout << "FATEL ERROR! Operation " << name_
              << " has no assigned optimization criterion! Exiting..."
              << std::endl;
    exit(-1);
  }

  return ptr_to_optimization_criterion_->getOptimalAlgorithm(input_values,
                                                             *this, dev_constr);
}

bool Operation::hasAlgorithm(const std::string& name_of_algorithm) {
  // std::map<std::string,boost::shared_ptr<Algorithm> >::iterator it;
  MapNameToAlgorithm::iterator it;
  // std::map<std::string,boost::shared_ptr<Algorithm> >
  // map_algorithmname_to_pointer_;
  // it=map_algorithmname_to_pointer_.begin();
  for (it = map_algorithmname_to_pointer_.begin();
       it != map_algorithmname_to_pointer_.end(); it++) {
    if (it->second->getName() == name_of_algorithm) return true;
  }
  return false;
}

bool Operation::addObservation(const std::string& name_of_algorithm,
                               const MeasurementPair& mp) {
  MapNameToAlgorithm::iterator it;
  for (it = map_algorithmname_to_pointer_.begin();
       it != map_algorithmname_to_pointer_.end(); it++) {
    if (it->second->getName() == name_of_algorithm) {
      return it->second->addMeasurementPair(mp);
    }
  }

  return false;
}

const std::map<double, std::string>
Operation::getEstimatedExecutionTimesforAlgorithms(const Tuple& input_values) {
  std::map<double, std::string> map_execution_times_to_algorithm_name;
  MapNameToAlgorithm::iterator it;
  for (it = map_algorithmname_to_pointer_.begin();
       it != map_algorithmname_to_pointer_.end(); it++) {
    EstimatedTime estimated_time =
        it->second->getEstimatedExecutionTime(input_values);
    std::string name = it->second->getName();
    double time_in_nanosecs = estimated_time.getTimeinNanoseconds();
    map_execution_times_to_algorithm_name[time_in_nanosecs] = name;
    if (!quiet && verbose && debug)
      std::cout
          << "Operation::getEstimatedExecutionTimesforAlgorithms() Algorithm: '"
          << name << "'" << std::endl;
  }
  // std::cout << "Size:" << map_execution_times_to_algorithm_name.size() <<
  // std::endl;
  return map_execution_times_to_algorithm_name;
}

const std::string Operation::getName() const throw() { return name_; }

void Operation::incrementNumberOfRightDecisions() throw() {
  ++number_of_right_decisions_;
}
void Operation::incrementNumberOfTotalDecisions() throw() {
  ++number_of_total_decisions_;
}

uint64_t Operation::getNextTimestamp() throw() { return ++logical_time_; }

uint64_t Operation::getCurrentTimestamp() const throw() {
  return logical_time_;
}
const std::string Operation::toString() {
  std::stringstream ss;
  ss << this->name_ << " (Optimization Criterion: "
     << this->ptr_to_optimization_criterion_->getName() << ")";
  return ss.str();
}

}  // end namespace core
}  // end namespace hype
