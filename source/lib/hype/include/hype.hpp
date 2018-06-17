/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <config/exports.hpp>
#include <config/global_definitions.hpp>
#include <core/specification.hpp>

#include <core/algorithm_measurement.hpp>
#include <core/report.hpp>
#include <core/scheduling_decision.hpp>
#include <core/statistics_gatherer.hpp>
#include <core/time_measurement.hpp>
#include <core/workload_generator.hpp>

namespace hype {

  typedef core::AlgorithmMeasurement AlgorithmMeasurement;
  typedef core::Tuple Tuple;
  typedef core::MemoryCostModelFuncPtr MemoryCostModelFuncPtr;
  typedef core::SchedulingDecision SchedulingDecision;
  typedef core::Offline_Algorithm Offline_Algorithm;
  typedef core::WorkloadGenerator WorkloadGenerator;
  typedef core::StatisticsGatherer StatisticsGatherer;
  typedef core::Report Report;
  typedef core::AlgorithmSpecification AlgorithmSpecification;
  typedef core::OperatorSpecification OperatorSpecification;
  typedef core::DeviceSpecification DeviceSpecification;
  typedef core::DeviceConstraint DeviceConstraint;
  // typedef core::MeasurementPair MeasurementPair;

  /*!
   *  \brief     The Scheduler is the central component for interaction of the
   *application and the library.
   *  \details   The Scheduler provides two main functionalities. First, it
   *provides the service to decide on the optimal algorithm for an operation
   *w.r.t. a user specified optimization criterion.
   *					Second, the Scheduler implements an
   *interface
   *to
   *add
   *new
   *Observations to the executed Algorithm. Hence, it is the central component
   *for interaction of the application and the library.
   *					Since it is not meaningful to have
   *multiple
   *instances
   *of
   *the Scheduler class, it is not possible to create multiple Scheduler
   *instances.
   *					This property is implemented by using
   *the
   *singelton
   *concept. Additionally, the Scheduler enables the user to setup the
   *Operations with their respective Algorithms as well as to configure
   *					for each algorithm a statistical method
   *and
   *a
   *recomputation heuristic and for each operation an optimization criterion.
   *					Note that the statistical method and the
   *recomputation
   *statistic can be exchanged at run-time, because the Algortihm uses the
   *pointer to
   *             implementation technique (or pimpl-idiom). This class is the
   *interface for using stemod. It forwards calls to the Scheduler in
   *stemod::core and implements thread safety.
   *  \author    Sebastian Breß
   *  \version   0.1
   *  \date      2012
   *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
   *http://www.gnu.org/licenses/lgpl-3.0.txt
   */
  class HYPE_EXPORT Scheduler {
   public:
    /*! \brief This method implements the singelton concept for the Scheduler
     * class to avoid multiple instances.
     *  \return Reference to Scheduler instance.
     */
    static Scheduler& instance();
    /* \brief adds an Algorithm to the AlgorithmPool of operation
     * name_of_operation.
     * \details If the specified operation does not exist, it is created.
     * Multiple calls to addAlgorithm
     *  with the same Operation name will add the respective algorithms to the
     * algorithm pool of the specified Operation
     * \param name_of_operation name of the operation, where the algorithm
     * belongs to
     * \param name_of_algorithm Name of Algorithm to be created
     * \param name_of_statistical_method assigns the StatisticalMethod
     * name_of_statistical_method to the new Algorithm
     * \param name_of_recomputation_strategy assigns the RecomputationHeuristic
     * name_of_recomputation_strategy to the new Algorithm
     * \return returns true on success and false otherwise
     */
    //		bool addAlgorithm(const std::string& name_of_operation,
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
    // name_of_recomputation_strategy);

    /*! \brief adds an Algorithm to the AlgorithmPool of an operation defined by
     * alg_spec on the processing device defined by dev_spec.
     * \details If the specified operation does not exist, it is created.
     * Multiple calls to addAlgorithm
     *  with an AlgorithmSpecification having the same Operation name will add
     * the respective algorithms to the algorithm pool of the specified
     * Operation
     * \param alg_spec defines properties of the algorithm, e.g., name, the
     * operation it belongs to, etc.
     * \param dev_spec defines properties of the processing device the algorithm
     * runs on, e.g., device type (CPU or GPU) and the device id
     * \return returns true on success and false otherwise
     */
    bool addAlgorithm(const AlgorithmSpecification& alg_spec,
                      const DeviceSpecification& dev_spec);

    // addAlgorithm(AlgorithmSpecification,DeviceSpecification);
    // getOptAlgorithm(Tuple,std::string operation_name,DeviceConstraint
    // dev_constr);

    bool hasOperation(const std::string& operation_name) const;

    /*! \brief assigns the OptimizationCriterion name_of_optimization_criterion
     * to Operation name_of_operation
     * \param name_of_operation name of the Operation
     * \param name_of_optimization_criterion Name of OptimizationCriterion
     * \return returns true on success and false otherwise
     */
    bool setOptimizationCriterion(
        const std::string& name_of_operation,
        const std::string& name_of_optimization_criterion);
    /*! \brief assigns the StatisticalMethod name_of_statistical_method to an
     * existing Algorithm
     * \param name_of_algorithm Name of Algorithm
     * \param name_of_statistical_method assigns the StatisticalMethod
     * name_of_statistical_method to an existing Algorithm
     * \return returns true on success and false otherwise
     */
    bool setStatisticalMethod(const std::string& name_of_algorithm,
                              const DeviceSpecification& dev_spec,
                              const std::string& name_of_statistical_method);
    /*! \brief assigns the StatisticalMethod name_of_statistical_method to an
     * existing Algorithm
     * \param name_of_algorithm Name of Algorithm
     * \param name_of_recomputation_strategy assigns the RecomputationHeuristic
     * name_of_recomputation_strategy to an existing Algorithm
     * \return returns true on success and false otherwise
     */
    bool setRecomputationHeuristic(
        const std::string& name_of_algorithm,
        const DeviceSpecification& dev_spec,
        const std::string& name_of_recomputation_strategy);

    /* \brief Returns a Scheduling Decision, which contains the name of the
     estimated optimal Algorithm
     *	 w.r.t. the user specified optimization criterion
     * \param name_of_operation name of the operation, where the optimal
     algorithm is requested
     * \param input_values features of the input dataset, which is needed to
     compute estimated execution times
             for all algortihms of the specified operation
     * \return SchedulingDecision, which contains the suggested algortihm for
     the specified information
     */
    // const SchedulingDecision getOptimalAlgorithmName(const std::string&
    // name_of_operation, const Tuple& input_values, DeviceTypeConstraint
    // dev_constr = ANY_DEVICE);

    /*! \brief Returns a Scheduling Decision, which contains the name of the
     *estimated optimal Algorithm
     *	 w.r.t. the user specified optimization criterion
     * \param op_spec OperatorSpecification, contains all available information
     *about the operator to execute
     * \param dev_constr DeviceConstraint, restricting the available algorithms
     *to a subset of the algorithm pool (e.g., allow only CPU algorithms)
     * \return SchedulingDecision, which contains the suggested algortihm for
     *the specified information
     */
    const SchedulingDecision getOptimalAlgorithm(
        const OperatorSpecification& op_spec,
        const DeviceConstraint& dev_constr);

    /* \brief adds an observed MeasurementPair to the algorithm previously
     * choosen by getOptimalAlgorithmName.
     * \param name_of_algorithm name of the algorithm the MeasurementPair
     * belongs to
     * \param mp the observed MeasurementPair
     * \return true on success and false in case an error occured
     */
    // bool addObservation(const std::string& name_of_algorithm, const
    // MeasurementPair& mp);
    /*! \brief adds an observed execution time to the algorithm previously
     * choosen by getOptimalAlgorithmName.
     * \param sched_dec the scheduling decision, this observation belongs to
     * \param measured_execution_time measured execution time, in nanoseconds!!!
     * \return true on success and false in case an error occured
     */
    bool addObservation(const SchedulingDecision& sched_dec,
                        const double& measured_execution_time);
    /* \brief request an Estimated Execution Time from HyPE for an algorithm for
     * a certain operator
     * \param op_spec OperatorSpecification, contains all available information
     * about the operator to execute
     * \param alg_name algorithm name
     * \return estimated execution time
     */
    core::EstimatedTime getEstimatedExecutionTime(
        const OperatorSpecification& op_spec, const std::string& alg_name,
        const DeviceSpecification& dev_spec);
    /* \brief registers a special function, which estimates the memory capacity
     * an algorithm will consume, for a certain algorithm on a specific device
     * \param alg_spec defines properties of the algorithm, e.g., name, the
     * operation it belongs to, etc.
     * \param dev_spec defines properties of the processing device the algorithm
     * runs on, e.g., device type (CPU or GPU) and the device id
     * \param memory_cost_model function pointer to a function that gets a Tuple
     * as input and returns the estimated memory requirements
     */
    bool registerMemoryCostModel(const AlgorithmSpecification& alg_spec,
                                 const DeviceSpecification& dev_spec,
                                 MemoryCostModelFuncPtr memory_cost_model);

    /* \brief sets for all algorithms registered to HyPE one global
     * LoadAdaptionPolicy
     * \param recomp_heur defines the RecomputationHeuristic that has to be used
     * for all algorithms
     */
    void setGlobalLoadAdaptionPolicy(RecomputationHeuristic recomp_heur);

    /*! \brief returns a list that contains all algorithm names with their
     * respective estimation errors*/
    const std::list<std::pair<std::string, double> >
    getAverageEstimationErrors() const;
    /*! \brief returns a list that contains all processor names with their
     * respective total execution time*/
    const std::list<std::pair<std::string, double> > getTotalProcessorTimes()
        const;

   private:
    /*! \brief Constructor is private to avoid multiple instances of Scheduler.
     */
    Scheduler();
    /*! \brief Copy Constructor is private to avoid copying of the single
     * instance of Scheduler.
     *  \param reference to existing Scheduler object
     */
    Scheduler(const Scheduler&);
    /*! \brief Copy assignment operator is private to avoid copying of the
     * single instance of Scheduler.
     *  \param reference to existing Scheduler object
     */
    Scheduler& operator=(const Scheduler& s);
  };

}  // end namespace hype

// namespace hype = hype;
