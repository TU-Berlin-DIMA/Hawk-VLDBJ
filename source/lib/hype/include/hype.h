/***********************************************************************************************************
Copyright (c) 2013, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
 ***********************************************************************************************************/
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <config/global_definitions.hpp>

//#define NDEBUG //uncomment for release version
#include <assert.h>

#ifdef __cplusplus
using namespace hype;
extern "C" {
#endif

// enum C_ComputeDevice {CPU,GPU,FPGA,NP}; //CPU,GPU,FPGA, Network Processor

/*#############################################################################*/
/************************* Scheduling Decision ******************************/

/*#############################################################################*/

typedef struct C_SchedulingDecision {
  void* ptr;  //=NULL;
} C_SchedulingDecision;

char* hype_SchedulingDecision_getAlgorithmName(C_SchedulingDecision* sched_dec);

ProcessingDeviceID hype_SchedulingDecision_getProcessingDeviceID(
    C_SchedulingDecision* sched_dec);

int hype_deleteSchedulingDecision(C_SchedulingDecision* sched_dec);

/*#############################################################################*/
/************************* Algorithm Specification
 * ******************************/

/*#############################################################################*/

typedef struct C_AlgorithmSpecification { void* ptr; } C_AlgorithmSpecification;

/*! \brief constructs an C_AlgorithmSpecification object by assigning necessary
 * informations to all fields of the object
 * \param alg_name name of the algorithm
 * \param op_name name of the operation the algorithms belongs to
 * \param stat_meth the statistical method used for learning the algorithms
 * behavior (optional)
 * \param recomp_heur the recomputation heuristic used for adapting the
 * algorithms approximation function (optional)
 * \param opt_crit the optimization criterion of the operation the algorithm
 * belongs to (optional)
 */
C_AlgorithmSpecification* hype_createAlgorithmSpecification(
    const char* alg_name, const char* op_name, StatisticalMethod stat_meth,
    RecomputationHeuristic recomp_heur, OptimizationCriterion opt_crit);

int hype_deleteAlgorithmSpecification(C_AlgorithmSpecification* alg_spec);

/*#############################################################################*/
/************************* Device Specification ******************************/

/*#############################################################################*/

typedef struct C_DeviceSpecification { void* ptr; } C_DeviceSpecification;

/*! \brief constructs an C_DeviceSpecification object by assigning necessary
 * informations to all fields of the object
 * \param pd the unique id of the processing device
 * \param pd_t type of the processing device (e.g., CPU or GPU)
 * \param pd_m unique id of the memory the processing device uses
 */
C_DeviceSpecification* hype_createDeviceSpecification(
    ProcessingDeviceID pd, ProcessingDeviceType pd_t,
    ProcessingDeviceMemoryID pd_m);
/*!
  *  \brief returns the processing device's ProcessingDeviceID
  */
ProcessingDeviceID hype_DeviceSpecification_getProcessingDeviceID(
    C_DeviceSpecification* dev_spec);
/*!
  *  \brief returns the processing device's device type
  */
ProcessingDeviceType hype_DeviceSpecification_getDeviceType(
    C_DeviceSpecification* dev_spec);
/*!
  *  \brief returns the processing device's memory id
  */
ProcessingDeviceMemoryID hype_DeviceSpecification_getMemoryID(
    C_DeviceSpecification* dev_spec);
/*!
  *  \brief deletes the device specification
  */
int hype_deleteDeviceSpecification(C_DeviceSpecification* dev_spec);

/*#############################################################################*/
/************************* Operator Specification
 * ******************************/

/*#############################################################################*/

typedef struct C_OperatorSpecification { void* ptr; } C_OperatorSpecification;

/*! \brief constructs an C_OperatorSpecification object by assigning necessary
 * informations to all fields of the object
 * \param operator_name the operations's name
 * \param feature_vector the feature vector of this operator
 * \param feature_vector_length the feature vectors length
 * \param location_of_input_data the memory id where the input data is stored
 * \param location_for_output_data the memory id where the output data is stored
 */
C_OperatorSpecification* hype_create_OperatorSpecification(
    const char* operator_name, double* feature_vector,
    size_t feature_vector_length,
    ProcessingDeviceMemoryID location_of_input_data,
    ProcessingDeviceMemoryID location_for_output_data);

int hype_deleteOperatorSpecification(C_OperatorSpecification* op_spec);

/*#############################################################################*/
/************************* Device Constraints *********************************/

/*#############################################################################*/

typedef struct C_DeviceConstraint { void* ptr; } C_DeviceConstraint;

/*! \brief constructs an C_DeviceConstraint object by assigning necessary
 * informations to all fields of the object
 * \param dev_constr a device type constraint (e.g., CPU_ONLY or ANY_DEVICE for
 * now restriction)
 * \param pd_mem_constr memory id, where the data should be stored when
 * processed (experimental)
 */
C_DeviceConstraint* hype_createDeviceConstraint(
    DeviceTypeConstraint dev_constr, ProcessingDeviceMemoryID pd_mem_constr);

int hype_deleteDeviceConstraint(C_DeviceConstraint* dev_const);

/*#############################################################################*/
/************************* Scheduler functions
 * *********************************/
/*#############################################################################*/

/*! \brief adds an Algorithm to the AlgorithmPool of an operation defined by
 * alg_spec on the processing device defined by dev_spec.
 * \details If the specified operation does not exist, it is created. Multiple
 * calls to addAlgorithm
 *  with an AlgorithmSpecification having the same Operation name will add the
 * respective algorithms to the algorithm pool of the specified Operation
 * \param alg_spec defines properties of the algorithm, e.g., name, the
 * operation it belongs to, etc.
 * \param dev_spec defines properties of the processing device the algorithm
 * runs on, e.g., device type (CPU or GPU) and the device id
 * \return returns true on success and false otherwise
 */
int hype_addAlgorithm(const C_AlgorithmSpecification* alg_spec,
                      const C_DeviceSpecification* dev_spec);

int hype_hasOperation(char* operation_name);

/*! \brief assigns the OptimizationCriterion name_of_optimization_criterion to
 * Operation name_of_operation
 * \param name_of_operation name of the Operation
 * \param name_of_optimization_criterion Name of OptimizationCriterion
 * \return returns true on success and false otherwise
 */
int hype_setOptimizationCriterion(const char* name_of_operation,
                                  const char* name_of_optimization_criterion);
/*! \brief assigns the StatisticalMethod name_of_statistical_method to an
 * existing Algorithm
 * \param name_of_algorithm Name of Algorithm
 * \param name_of_statistical_method assigns the StatisticalMethod
 * name_of_statistical_method to an existing Algorithm
 * \return returns true on success and false otherwise
 */
int hype_setStatisticalMethod(const char* name_of_algorithm,
                              C_DeviceSpecification* dev_spec,
                              const char* name_of_statistical_method);
/*! \brief assigns the StatisticalMethod name_of_statistical_method to an
 * existing Algorithm
 * \param name_of_algorithm Name of Algorithm
 * \param name_of_recomputation_strategy assigns the RecomputationHeuristic
 * name_of_recomputation_strategy to an existing Algorithm
 * \return returns true on success and false otherwise
 */
int hype_setRecomputationHeuristic(const char* name_of_algorithm,
                                   C_DeviceSpecification* dev_spec,
                                   const char* name_of_recomputation_strategy);
/*! \brief Returns a Scheduling Decision, which contains the name of the
 *estimated optimal Algorithm
 *	 w.r.t. the user specified optimization criterion
 * \param op_spec OperatorSpecification, contains all available information
 *about the operator to execute
 * \param dev_constr DeviceConstraint, restricting the available algorithms to a
 *subset of the algorithm pool (e.g., allow only CPU algorithms)
 * \return SchedulingDecision, which contains the suggested algortihm for the
 *specified information
 */
C_SchedulingDecision* hype_getOptimalAlgorithm(
    const C_OperatorSpecification* op_spec,
    const C_DeviceConstraint* dev_constr);
/*! \brief adds an observed execution time to the algorithm previously choosen
 * by getOptimalAlgorithmName.
 * \param sched_dec the scheduling decision, this observation belongs to
 * \param measured_execution_time measured execution time, in nanoseconds!!!
 * \return true on success and false in case an error occured
 */
int hype_addObservation(const C_SchedulingDecision* sched_dec,
                        const double measured_execution_time);
/* \brief request an Estimated Execution Time from HyPE for an algorithm for a
 * certain operator
 * \param op_spec OperatorSpecification, contains all available information
 * about the operator to execute
 * \param alg_name algorithm name
 * \return estimated execution time
 */
double hype_getEstimatedExecutionTime(const C_OperatorSpecification* op_spec,
                                      const char* alg_name,
                                      C_DeviceSpecification* dev_spec);

/*#############################################################################*/
/************************* Util functions *********************************/
/*#############################################################################*/
uint64_t hype_getTimestamp();
void hype_printStatus();

#ifdef __cplusplus
}
#endif
