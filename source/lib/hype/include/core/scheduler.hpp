/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <core/measurementpair.hpp>
#include <core/operation.hpp>
#include <core/specification.hpp>
#include <core/time_measurement.hpp>
#include <iostream>
#include <query_processing/operator.hpp>
#include <query_processing/processing_device.hpp>
#include <query_processing/virtual_processing_device.hpp>
#include <queue>
#include <string>
#include <vector>

namespace hype {
  namespace core {

    typedef std::vector<SchedulingDecision> SchedulingDecisionVector;
    typedef boost::shared_ptr<SchedulingDecisionVector>
        SchedulingDecisionVectorPtr;
    typedef std::vector<std::pair<OperatorSpecification, DeviceConstraint> >
        OperatorSequence;

    struct InternalPhysicalOperator {
      inline InternalPhysicalOperator(AlgorithmPtr alg_ptr_arg,
                                      Tuple feature_vector_arg, double cost_arg)
          : alg_ptr(alg_ptr_arg),
            feature_vector(feature_vector_arg),
            cost(cost_arg) {}
      AlgorithmPtr alg_ptr;
      Tuple feature_vector;
      double cost;
    };
    // typedef std::pair<AlgorithmPtr, double> InternalPhysicalOperator;
    typedef std::vector<InternalPhysicalOperator> InternalPhysicalPlan;
    /*!
     *
     *
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
     *the
     *Scheduler class, it is not possible to create multiple Scheduler
     *instances.
     *					This property is implemented by using
     *the
     *singelton
     *concept.
     *Additionally, the Scheduler enables the user to setup the Operations with
     *their respective Algorithms as well as to configure
     *					for each algorithm a statistical method
     *and
     *a
     *recomputation
     *heuristic and for each operation an optimization criterion.
     *					Note that the statistical method and the
     *recomputation
     *statistic
     *can be exchanged at run-time, because the Algortihm uses the pointer to
     *             implementation technique (or pimpl-idiom).
     *  \author    Sebastian Breß
     *  \version   0.1
     *  \date      2012
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     *http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    class Scheduler {
     public:
      typedef std::map<std::string, boost::shared_ptr<Operation> >
          MapNameToOperation;
      typedef std::queue<queryprocessing::NodePtr> OperatorStream;

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
       * \param name_of_recomputation_strategy assigns the
       * RecomputationHeuristic name_of_recomputation_strategy to the new
       * Algorithm
       * \return returns true on success and false otherwise
       */
      //				bool addAlgorithm(const std::string&
      // name_of_operation,
      //										const
      // std::string& name_of_algorithm,
      //										ComputeDevice
      // comp_dev,
      //										const
      // std::string& name_of_statistical_method,
      //										const
      // std::string& name_of_recomputation_strategy);

      /*! \brief adds an Algorithm to the AlgorithmPool of an operation defined
       * by alg_spec on the processing device defined by dev_spec.
       * \details If the specified operation does not exist, it is created.
       * Multiple calls to addAlgorithm
       *  with an AlgorithmSpecification having the same Operation name will add
       * the respective algorithms to the algorithm pool of the specified
       * Operation
       * \param alg_spec defines properties of the algorithm, e.g., name, the
       * operation it belongs to, etc.
       * \param dev_spec defines properties of the processing device the
       * algorithm runs on, e.g., device type (CPU or GPU) and the device id
       * \return returns true on success and false otherwise
       */
      bool addAlgorithm(const AlgorithmSpecification& alg_spec,
                        const DeviceSpecification& dev_spec);

      bool hasOperation(const std::string& operation_name) const;

      /*! \brief assigns the OptimizationCriterion
       * name_of_optimization_criterion to Operation name_of_operation
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
                                const std::string& name_of_statistical_method);
      /*! \brief assigns the StatisticalMethod name_of_statistical_method to an
       * existing Algorithm
       * \param name_of_algorithm Name of Algorithm
       * \param name_of_recomputation_strategy assigns the
       * RecomputationHeuristic name_of_recomputation_strategy to an existing
       * Algorithm
       * \return returns true on success and false otherwise
       */
      bool setRecomputationHeuristic(
          const std::string& name_of_algorithm,
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
       * \param op_spec OperatorSpecification, contains all available
       *information about the operator to execute
       * \param dev_constr DeviceConstraint, restricting the available
       *algorithms to a subset of the algorithm pool (e.g., allow only CPU
       *algorithms)
       * \return SchedulingDecision, which contains the suggested algortihm for
       *the specified information
       */
      const SchedulingDecision getOptimalAlgorithm(
          const OperatorSpecification& op_spec,
          const DeviceConstraint& dev_constr);

      const SchedulingDecisionVectorPtr getOptimalAlgorithm(
          const OperatorSequence& op_seq,
          const QueryOptimizationHeuristic& heuristic);

      /* \brief adds an observed MeasurementPair to the algorithm previously
       *choosen by getOptimalAlgorithmName.
       * \param name_of_algorithm name of the algorithm the MeasurementPair
       *belongs to
       * \param mp the observed MeasurementPair
       * \return Scheduling Decision, which recommends the application the
       *optimal Algorithm w.r.t. the specified
       *	 operation and features of the input data set
       */
      // bool addObservation(const std::string& name_of_algorithm, const
      // MeasurementPair& mp);

      /*! \brief adds an observed execution time to the algorithm previously
       * choosen by getOptimalAlgorithmName.
       * \param sched_dec the scheduling decision, this observation belongs to
       * \param measured_execution_time measured execution time, in
       * nanoseconds!!!
       * \return true on success and false in case an error occured
       */
      bool addObservation(const SchedulingDecision& sched_dec,
                          const double& measured_execution_time);

      /*! \brief requests a pointer to Algorithm named name_of_algorithm
       * \param name_of_algorithm name of the Algorithm where the pointer is
       * requested
       * \return returns a valid pointer to and Algorithm object on success and
       * a NULL pointer if the requested Algorithm was not found
       */
      const AlgorithmPtr getAlgorithm(const std::string& name_of_algorithm);
      /* \brief request an Estimated Execution Time from HyPE for an algorithm
       * for a certain operator
       * \param op_spec OperatorSpecification, contains all available
       * information about the operator to execute
       * \param alg_name algorithm name
       * \return estimated execution time
       */
      EstimatedTime getEstimatedExecutionTime(
          const OperatorSpecification& op_spec, const std::string& alg_name);

      void print();
      void setGlobalLoadAdaptionPolicy(RecomputationHeuristic recomp_heur);

      const MapNameToOperation& getOperatorMap();
      DeviceSpecification& getDeviceSpecificationforCopyType(
          const std::string& copy_type);

      bool registerMemoryCostModel(const AlgorithmSpecification& alg_spec,
                                   const DeviceSpecification& dev_spec,
                                   MemoryCostModelFuncPtr cost_model);

      void addIntoGlobalOperatorStream(queryprocessing::NodePtr op);
      void addIntoGlobalOperatorStream(
          const std::list<queryprocessing::NodePtr>& operator_list);

      const std::list<std::pair<std::string, double> >
      getAverageEstimationErrors() const;
      const std::list<std::pair<std::string, double> > getTotalProcessorTimes()
          const;

     private:
      void scheduling_thread();
      /*! \brief Constructor is private to avoid multiple instances of
       * Scheduler.
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
      Scheduler& operator=(const Scheduler&);

      ~Scheduler();

      void insertCopyOperationAtBegin(SchedulingDecisionVectorPtr result_plan,
                                      const SchedulingDecision& first_operator);
      void insertCopyOperationAtBegin(
          InternalPhysicalPlan& result_plan,
          const InternalPhysicalOperator& first_operator);
      void insertCopyOperationAtEnd(SchedulingDecisionVectorPtr result_plan,
                                    const SchedulingDecision& last_operator);
      void insertCopyOperationAtEnd(
          InternalPhysicalPlan& result_plan,
          const InternalPhysicalOperator& last_operator);
      void insertCopyOperationInsidePlan(
          SchedulingDecisionVectorPtr result_plan,
          const SchedulingDecision& current_operator);
      void insertCopyOperationInsidePlan(
          InternalPhysicalPlan& result_plan,
          const InternalPhysicalOperator& current_operator);

      // void addOperation(std::string operation_name);

      /*! \brief maps the name of an Operation to a pointer to an Operation
       * object*/
      MapNameToOperation map_operationname_to_operation_;
      /*! \brief maps the name of a StatisticalMethod to a pointer to a
       * StatisticalMethod object*/
      StatisticalMethodMap map_statisticalmethodname_to_statisticalmethod_;
      /*! \brief specification of DMA controller which transfers data from a CPU
       * to a co-processor*/
      DeviceSpecification dma_cpu_cp_;
      /*! \brief specification of DMA controller which transfers data from a
       * co-processor to a CPU*/
      DeviceSpecification dma_cp_cpu_;

     public:
      class ProcessingDevices {
       public:
        typedef std::vector<queryprocessing::ProcessingDevicePtr>
            PhysicalDevices;
        typedef std::pair<queryprocessing::VirtualProcessingDevicePtr,
                          PhysicalDevices>
            LogicalDevice;
        typedef std::map<ProcessingDeviceID, LogicalDevice> Devices;

        //                                        //Virtual Processing Devices
        //                                        are for bookkeeping, their use
        //                                        is mandatory
        //					typedef
        // std::map<ProcessingDeviceID,queryprocessing::VirtualProcessingDevicePtr>
        // Devices;
        //                                        //(Physical) Processing
        //                                        Devices are neccessary in case
        //                                        HyPE is also used as executon
        //                                        engine
        //                                        //(e.g., they start a seperate
        //                                        thread handling the
        //                                        (co-)processors)
        //					typedef
        // std::map<ProcessingDeviceID,queryprocessing::ProcessingDevicePtr>
        // PhysicalDevices;
        ProcessingDevices();
        ~ProcessingDevices();

        queryprocessing::VirtualProcessingDevicePtr getVirtualProcessingDevice(
            ProcessingDeviceID);

        queryprocessing::ProcessingDevicePtr getProcessingDevice(
            ProcessingDeviceID);

        bool addDevice(const DeviceSpecification&);

        bool exists(const DeviceSpecification&) const throw();

        const Devices& getDevices() const throw();
        ProcessingDeviceID getProcessingDeviceID(
            ProcessingDeviceMemoryID mem_id) const;
        const std::list<std::pair<std::string, double> >
        getTotalProcessorTimes() const;

        bool addSchedulingDecision(const SchedulingDecision&);

        void addIntoGlobalOperatorStream(queryprocessing::OperatorPtr op);

        bool removeSchedulingDecision(const SchedulingDecision&);

        void print() const throw();

       private:
        Devices virt_comp_devs_;
        mutable boost::mutex processing_device_mutex_;
        // PhysicalDevices phy_comp_devs_;
      };

      ProcessingDevices& getProcessingDevices();

     private:
      /*! \brief stores the processing devices, which were specified by the
       * user*/
      ProcessingDevices proc_devs_;
      OperatorStream global_operator_stream_;
      boost::mutex operator_stream_mutex_;
      boost::mutex scheduling_thread_mutex_;
      boost::condition_variable scheduling_thread_cond_var_;
      boost::shared_ptr<boost::thread> scheduling_thread_;
      bool terminate_threads_;
    };

  }  // end namespace core
}  // end namespace hype
