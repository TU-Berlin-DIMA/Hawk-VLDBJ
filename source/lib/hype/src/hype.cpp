/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#include <config/global_definitions.hpp>
#include <core/algorithm_measurement.hpp>
#include <core/scheduler.hpp>
#include <core/statistics_gatherer.hpp>
#include <core/time_measurement.hpp>
#include <core/workload_generator.hpp>
#include <hype.hpp>
#include <util/algorithm_name_conversion.hpp>

//#include <boost/thread.hpp>

namespace hype {

// boost::mutex global_mutex;

Scheduler& Scheduler::instance() {
  static Scheduler scheduler;
  // return core::Scheduler::instance();
  return scheduler;
}

//		bool Scheduler::addAlgorithm(const std::string&
// name_of_operation,
//										const
// std::string&
// name_of_algorithm,
//										ComputeDevice
// comp_dev,
//										const
// std::string&
// name_of_statistical_method,
//										const
// std::string&
// name_of_recomputation_strategy)
//		{	//boost::lock_guard<boost::mutex> lock(global_mutex);
//			return
// core::Scheduler::instance().addAlgorithm(name_of_operation,name_of_algorithm,comp_dev,name_of_statistical_method,name_of_recomputation_strategy);
//		}

bool Scheduler::addAlgorithm(
    const AlgorithmSpecification& alg_spec,
    const DeviceSpecification&
        dev_spec) {  // boost::lock_guard<boost::mutex> lock(global_mutex);
  return core::Scheduler::instance().addAlgorithm(alg_spec, dev_spec);
}

bool Scheduler::hasOperation(const std::string& operation_name) const {
  return core::Scheduler::instance().hasOperation(operation_name);
}

bool Scheduler::setOptimizationCriterion(
    const std::string& name_of_operation,
    const std::string&
        name_of_optimization_criterion) {  // boost::lock_guard<boost::mutex>
                                           // lock(global_mutex);
  return core::Scheduler::instance().setOptimizationCriterion(
      name_of_operation, name_of_optimization_criterion);
}

bool Scheduler::setStatisticalMethod(
    const std::string& name_of_algorithm, const DeviceSpecification& dev_spec,
    const std::string&
        name_of_statistical_method) {  // boost::lock_guard<boost::mutex>
                                       // lock(global_mutex);
  return core::Scheduler::instance().setStatisticalMethod(
      hype::util::toInternalAlgName(name_of_algorithm, dev_spec),
      name_of_statistical_method);
}

bool Scheduler::setRecomputationHeuristic(
    const std::string& name_of_algorithm, const DeviceSpecification& dev_spec,
    const std::string&
        name_of_recomputation_strategy) {  // boost::lock_guard<boost::mutex>
                                           // lock(global_mutex);
  // return
  // core::Scheduler::instance().setRecomputationHeuristic(name_of_algorithm,name_of_recomputation_strategy);
  return core::Scheduler::instance().setRecomputationHeuristic(
      hype::util::toInternalAlgName(name_of_algorithm, dev_spec),
      name_of_recomputation_strategy);
}

//		const SchedulingDecision
// Scheduler::getOptimalAlgorithmName(const
// std::string& name_of_operation, const Tuple& input_values,
// DeviceTypeConstraint dev_constr)
//		{	//boost::lock_guard<boost::mutex> lock(global_mutex);
//			return
// core::Scheduler::instance().getOptimalAlgorithmName(name_of_operation,input_values,dev_constr);
//		}

const SchedulingDecision Scheduler::getOptimalAlgorithm(
    const OperatorSpecification& op_spec, const DeviceConstraint& dev_constr) {
  return core::Scheduler::instance().getOptimalAlgorithm(op_spec, dev_constr);
}

/*
bool Scheduler::addObservation(const std::string& name_of_algorithm, const
MeasurementPair& mp)
{
        return core::Scheduler::instance().addObservation(name_of_algorithm,mp);
}*/

bool Scheduler::addObservation(const SchedulingDecision& sched_dec,
                               const double& measured_execution_time) {
  //			return
  // core::Scheduler::instance().addObservation(sched_dec.getNameofChoosenAlgorithm(),
  //																			  core::MeasurementPair(sched_dec.getFeatureValues(),
  //																											core::MeasuredTime(measured_execution_time),
  //																											sched_dec.getEstimatedExecutionTimeforAlgorithm()
  //																											)
  return core::Scheduler::instance().addObservation(sched_dec,
                                                    measured_execution_time);
}

hype::core::EstimatedTime Scheduler::getEstimatedExecutionTime(
    const OperatorSpecification& op_spec, const std::string& alg_name,
    const DeviceSpecification& dev_spec) {
  return core::Scheduler::instance().getEstimatedExecutionTime(
      op_spec, hype::util::toInternalAlgName(alg_name, dev_spec));
  //	return core::Scheduler::instance().getEstimatedExecutionTime(op_spec,
  // alg_name);
}

bool Scheduler::registerMemoryCostModel(
    const AlgorithmSpecification& alg_spec, const DeviceSpecification& dev_spec,
    MemoryCostModelFuncPtr memory_cost_model) {
  return core::Scheduler::instance().registerMemoryCostModel(alg_spec, dev_spec,
                                                             memory_cost_model);
}

void Scheduler::setGlobalLoadAdaptionPolicy(
    RecomputationHeuristic recomp_heur) {
  return core::Scheduler::instance().setGlobalLoadAdaptionPolicy(recomp_heur);
}

const std::list<std::pair<std::string, double> >
Scheduler::getAverageEstimationErrors() const {
  return core::Scheduler::instance().getAverageEstimationErrors();
}

const std::list<std::pair<std::string, double> >
Scheduler::getTotalProcessorTimes() const {
  return core::Scheduler::instance().getTotalProcessorTimes();
}

Scheduler::Scheduler() {}

Scheduler::Scheduler(const Scheduler&) {}

}  // end namespace hype
