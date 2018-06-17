/***********************************************************************************************************
Copyright (c) 2013, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
 ***********************************************************************************************************/

#include <hype.h>
#include <config/global_definitions.hpp>
#include <core/scheduler.hpp>
#include <core/time_measurement.hpp>
#include <hype.hpp>
#include <string>

using namespace hype;
using namespace std;

#include <stdio.h>
#include <stdlib.h>

//#define NDEBUG //uncomment for release version
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

/*#############################################################################*/
/************************* Scheduling Decision ******************************/

/*#############################################################################*/

char* hype_SchedulingDecision_getAlgorithmName(
    C_SchedulingDecision* sched_dec) {
  if (!sched_dec) return NULL;
  SchedulingDecision* ptr = static_cast<SchedulingDecision*>(sched_dec->ptr);
  if (!ptr) return NULL;
  std::string result = ptr->getNameofChoosenAlgorithm();
  const char* tmp = result.c_str();
  char* c_result =
      (char*)malloc(result.size() + 1);  // plus one because if zero byte '\0'
  strcpy(c_result, tmp);
  return c_result;
}

ProcessingDeviceID hype_SchedulingDecision_getProcessingDeviceID(
    C_SchedulingDecision* sched_dec) {
  if (!sched_dec) return PD0;
  SchedulingDecision* ptr = static_cast<SchedulingDecision*>(sched_dec->ptr);
  if (!ptr) return PD0;
  return ptr->getDeviceSpecification().getProcessingDeviceID();
}

int hype_deleteSchedulingDecision(C_SchedulingDecision* sched_dec) {
  if (!sched_dec) return int(false);
  SchedulingDecision* ptr = static_cast<SchedulingDecision*>(sched_dec->ptr);
  delete ptr;
  if (sched_dec) free(sched_dec);
  return int(true);
}

/*#############################################################################*/
/************************* Algorithm Specification
 * ******************************/

/*#############################################################################*/

C_AlgorithmSpecification* hype_createAlgorithmSpecification(
    const char* alg_name, const char* op_name, StatisticalMethod stat_meth,
    RecomputationHeuristic recomp_heur, OptimizationCriterion opt_crit) {
  C_AlgorithmSpecification* alg_spec =
      (C_AlgorithmSpecification*)malloc(sizeof(C_AlgorithmSpecification));
  if (!alg_spec) return NULL;
  alg_spec->ptr = new AlgorithmSpecification(alg_name, op_name, stat_meth,
                                             recomp_heur, opt_crit);

  return alg_spec;
}

int hype_deleteAlgorithmSpecification(C_AlgorithmSpecification* alg_spec) {
  if (!alg_spec) return int(false);
  AlgorithmSpecification* ptr =
      static_cast<AlgorithmSpecification*>(alg_spec->ptr);
  delete ptr;
  if (alg_spec) free(alg_spec);
  return int(true);
}

/*#############################################################################*/
/************************* Device Specification ******************************/

/*#############################################################################*/

C_DeviceSpecification* hype_createDeviceSpecification(
    ProcessingDeviceID pd, ProcessingDeviceType pd_t,
    ProcessingDeviceMemoryID pd_m) {
  C_DeviceSpecification* dev_spec =
      (C_DeviceSpecification*)malloc(sizeof(C_DeviceSpecification));
  if (!dev_spec) return NULL;
  dev_spec->ptr = new DeviceSpecification(pd, pd_t, pd_m);

  return dev_spec;
}

/*!
  *  \brief returns the processing device's ProcessingDeviceID
  */
ProcessingDeviceID hype_DeviceSpecification_getProcessingDeviceID(
    C_DeviceSpecification* dev_spec) {
  if (!dev_spec) return PD0;
  DeviceSpecification* ptr = static_cast<DeviceSpecification*>(dev_spec->ptr);
  return ptr->getProcessingDeviceID();
}
/*!
  *  \brief returns the processing device's device type
  */
ProcessingDeviceType hype_DeviceSpecification_getDeviceType(
    C_DeviceSpecification* dev_spec) {
  if (!dev_spec) return CPU;
  DeviceSpecification* ptr = static_cast<DeviceSpecification*>(dev_spec->ptr);
  return ptr->getDeviceType();
}
/*!
  *  \brief returns the processing device's memory id
  */
ProcessingDeviceMemoryID hype_DeviceSpecification_getMemoryID(
    C_DeviceSpecification* dev_spec) {
  if (!dev_spec) return PD_Memory_0;
  DeviceSpecification* ptr = static_cast<DeviceSpecification*>(dev_spec->ptr);
  return ptr->getMemoryID();
}

int hype_deleteDeviceSpecification(C_DeviceSpecification* dev_spec) {
  if (!dev_spec) return int(false);
  DeviceSpecification* ptr = static_cast<DeviceSpecification*>(dev_spec->ptr);
  delete ptr;
  if (dev_spec) free(dev_spec);
  return int(true);
}

/*#############################################################################*/
/************************* Operator Specification
 * ******************************/

/*#############################################################################*/

C_OperatorSpecification* hype_create_OperatorSpecification(
    const char* operator_name, double* feature_vector,
    size_t feature_vector_length,
    ProcessingDeviceMemoryID location_of_input_data,
    ProcessingDeviceMemoryID location_for_output_data) {
  C_OperatorSpecification* op_spec =
      (C_OperatorSpecification*)malloc(sizeof(C_OperatorSpecification));
  if (!op_spec) return NULL;
  op_spec->ptr = new OperatorSpecification(
      operator_name,
      std::vector<double>(feature_vector,
                          feature_vector + feature_vector_length),
      location_of_input_data, location_for_output_data);
  return op_spec;
}

int hype_deleteOperatorSpecification(C_OperatorSpecification* op_spec) {
  if (!op_spec) return int(false);
  if (!op_spec->ptr) return int(false);
  OperatorSpecification* ptr =
      static_cast<OperatorSpecification*>(op_spec->ptr);
  delete ptr;
  if (op_spec) free(op_spec);
  return int(true);
}

/*#############################################################################*/
/************************* Device Constraints *********************************/

/*#############################################################################*/

C_DeviceConstraint* hype_createDeviceConstraint(
    DeviceTypeConstraint dev_type_constr,
    ProcessingDeviceMemoryID pd_mem_constr) {
  C_DeviceConstraint* dev_constr =
      (C_DeviceConstraint*)malloc(sizeof(C_DeviceConstraint));
  if (!dev_constr) return NULL;
  dev_constr->ptr = new DeviceConstraint(dev_type_constr);  //, pd_mem_constr);
  return dev_constr;
}

int hype_deleteDeviceConstraint(C_DeviceConstraint* dev_const) {
  if (!dev_const) return int(false);
  DeviceConstraint* ptr = static_cast<DeviceConstraint*>(dev_const->ptr);
  delete ptr;
  if (dev_const) free(dev_const);
  return int(true);
}

/*#############################################################################*/
/************************* Scheduler functions
 * *********************************/

/*#############################################################################*/

int hype_addAlgorithm(const C_AlgorithmSpecification* alg_spec,
                      const C_DeviceSpecification* dev_spec) {
  if (alg_spec == NULL) return int(false);
  if (dev_spec == NULL) return int(false);
  if (alg_spec->ptr == NULL) return int(false);
  if (dev_spec->ptr == NULL) return int(false);
  AlgorithmSpecification* alg =
      static_cast<AlgorithmSpecification*>(alg_spec->ptr);
  DeviceSpecification* dev = static_cast<DeviceSpecification*>(dev_spec->ptr);
  return int(Scheduler::instance().addAlgorithm(*alg, *dev));
}

int hype_hasOperation(char* operation_name) {
  return int(Scheduler::instance().hasOperation(std::string(operation_name)));
}

int hype_setOptimizationCriterion(const char* name_of_operation,
                                  const char* name_of_optimization_criterion) {
  if (name_of_operation == NULL) return int(false);
  if (name_of_optimization_criterion == NULL) return int(false);
  return int(Scheduler::instance().setOptimizationCriterion(
      name_of_operation, name_of_optimization_criterion));
}

int hype_setStatisticalMethod(const char* name_of_algorithm,
                              C_DeviceSpecification* dev_spec,
                              const char* name_of_statistical_method) {
  if (name_of_algorithm == NULL) return int(false);
  if (name_of_statistical_method == NULL) return int(false);
  if (!dev_spec) return int(false);
  DeviceSpecification* dev_spec_ptr =
      static_cast<DeviceSpecification*>(dev_spec->ptr);
  return int(Scheduler::instance().setStatisticalMethod(
      name_of_algorithm, *dev_spec_ptr, name_of_statistical_method));
}

int hype_setRecomputationHeuristic(const char* name_of_algorithm,
                                   C_DeviceSpecification* dev_spec,
                                   const char* name_of_recomputation_strategy) {
  if (name_of_algorithm == NULL) return int(false);
  if (name_of_recomputation_strategy == NULL) return int(false);
  if (!dev_spec) return int(false);
  DeviceSpecification* dev_spec_ptr =
      static_cast<DeviceSpecification*>(dev_spec->ptr);
  return int(Scheduler::instance().setRecomputationHeuristic(
      name_of_algorithm, *dev_spec_ptr, name_of_recomputation_strategy));
}

C_SchedulingDecision* hype_getOptimalAlgorithm(
    const C_OperatorSpecification* op_spec,
    const C_DeviceConstraint* dev_constr) {
  if (!op_spec) return nullptr;
  if (!op_spec->ptr) return nullptr;
  if (!dev_constr) return nullptr;
  if (!dev_constr->ptr) return nullptr;
  OperatorSpecification* op_ptr =
      static_cast<OperatorSpecification*>(op_spec->ptr);
  DeviceConstraint* dev_ptr = static_cast<DeviceConstraint*>(op_spec->ptr);
  SchedulingDecision sched_dec = Scheduler::instance().getOptimalAlgorithm(
      *op_ptr, *dev_ptr);  //,*dev_ptr);
  SchedulingDecision* sched_dec_ptr = new SchedulingDecision(sched_dec);
  C_SchedulingDecision* c_sched_dec =
      (C_SchedulingDecision*)malloc(sizeof(C_SchedulingDecision));
  if (!c_sched_dec) return NULL;
  c_sched_dec->ptr = sched_dec_ptr;
  return c_sched_dec;
}

int hype_addObservation(const C_SchedulingDecision* sched_dec,
                        const double measured_execution_time) {
  if (!sched_dec) return int(false);
  if (!sched_dec->ptr) return int(false);
  SchedulingDecision* op_ptr = static_cast<SchedulingDecision*>(sched_dec->ptr);
  return int(
      Scheduler::instance().addObservation(*op_ptr, measured_execution_time));
}

double hype_getEstimatedExecutionTime(const C_OperatorSpecification* op_spec,
                                      const char* alg_name,
                                      C_DeviceSpecification* dev_spec) {
  if (!op_spec) return int(false);
  if (!op_spec->ptr) return int(false);
  if (!alg_name) return int(false);
  if (!dev_spec) return int(false);
  DeviceSpecification* dev_spec_ptr =
      static_cast<DeviceSpecification*>(dev_spec->ptr);
  OperatorSpecification* ptr =
      static_cast<OperatorSpecification*>(op_spec->ptr);
  return Scheduler::instance()
      .getEstimatedExecutionTime(*ptr, alg_name, *dev_spec_ptr)
      .getTimeinNanoseconds();
}

/*#############################################################################*/
/************************* Util functions *********************************/
/*#############################################################################*/
uint64_t hype_getTimestamp() { return hype::core::getTimestamp(); }

void hype_printStatus() { core::Scheduler::instance().print(); }

#ifdef __cplusplus
}
#endif

/*
int stemod_operation_addAlgorithm(const char* name_of_operation,
stemod::ComputeDevice comp_dev , const char* name_of_algorithm, const char*
name_of_statistical_method,
                                                                                         const char* name_of_recomputation_strategy){

        return static_cast<int>
(core::Scheduler::instance().addAlgorithm(name_of_operation, name_of_algorithm,
comp_dev, name_of_statistical_method,
                                                                                                  name_of_recomputation_strategy)){
                return 0;
        }

}


int stemod_operation_optimization_criterion_set(const char*
name_of_operation,const char* name_of_optimization_criterion){

        return static_cast<int>
(core::Scheduler::instance().setOptimizationCriterion(name_of_operation,name_of_optimization_criterion)){
                return 0;
        }

}


int stemod_algorithm_statistical_method_set(const char* name_of_algorithm,const
char* name_of_statistical_method){

        return static_cast<int>
(core::Scheduler::instance().setStatisticalMethod(name_of_algorithm,name_of_statistical_method)){
                return 0;
        }

}

int stemod_algorithm_recomputation_heuristic_set(const char*
name_of_algorithm,const char* name_of_recomputation_strategy){

        return static_cast<int>
(core::Scheduler::instance().setRecomputationHeuristic(name_of_algorithm,name_of_recomputation_strategy)){
                return 0;
        }

}


char* stemod_scheduler_OptimalAlgorithmName_get(const char* name_of_operation){

        //TODO: schnittstelle ändern, sodass eingabedaten übergeben werden
können!
        core::Tuple test{
                return 0;
        }
        //SchedulingDecision dec =
core::Scheduler::instance().getOptimalAlgorithmName(name_of_operation,test){
                return 0;
        }

//	char* ret=(char*)malloc(dec.size()){
                return 0;
        }
//
//	for(int i=0{
                return 0;
        }i<name.size(){
                return 0;
        }++i){
//		ret[i]=name[i]{
                return 0;
        }
//	}
//
//	return ret{
                return 0;
        }
        cout << name_of_operation << endl{
                return 0;
        }

        return (char*) 0{
                return 0;
        }
}

void stemod_algorithm_measurement_before_execution(struct
C_AlgorithmMeasurement* algmeas,const char* name_of_algorithm,double*
values,size_t number_of_values){ //starts timer

        algmeas->starting_point_timestamp=0{
                return 0;
        } //getTimestamp(){
                return 0;
        }
        algmeas->is_valid=1{
                return 0;
        }
        cout << name_of_algorithm << values << number_of_values << endl{
                return 0;
        }

}

void stemod_algorithm_measurement_after_execution(struct C_AlgorithmMeasurement*
algmeas){ //stops timer



                assert(algmeas->is_valid==1){
                return 0;
        }
                algmeas->is_valid=0{
                return 0;
        }

}
 */

/*
int hype_add_algorithm_to_operation(const char* name_of_operation, const char*
name_of_algorithm,
                                                                                                ComputeDevice comp_dev, const char* name_of_statistical_method,
                                                                                           const char* name_of_recomputation_strategy){

        return static_cast<int>
(Scheduler::instance().addAlgorithm(name_of_operation, name_of_algorithm,
                                                                                                                                                                        comp_dev, name_of_statistical_method,
                                                                                                                                                                        name_of_recomputation_strategy)){
                return 0;
        }
}

int hype_set_optimization_criterion(const char* name_of_operation,const char*
name_of_optimization_criterion){
        return static_cast<int>
(Scheduler::instance().setOptimizationCriterion(name_of_operation,name_of_optimization_criterion)){
                return 0;
        }
}

int hype_set_statistical_method(const char* name_of_algorithm,const char*
name_of_statistical_method){
        return static_cast<int>
(core::Scheduler::instance().setStatisticalMethod(name_of_algorithm,name_of_statistical_method)){
                return 0;
        }
}


int hype_set_recomputation_heuristic(const char* name_of_algorithm,const char*
name_of_recomputation_strategy){
        return static_cast<int>
(core::Scheduler::instance().setRecomputationHeuristic(name_of_algorithm,name_of_recomputation_strategy)){
                return 0;
        }
}

char* hype_scheduling_decision_get_algorithm_name(C_SchedulingDecision*
scheduling_decision_ptr){
                core::SchedulingDecision* sched_dec_ptr =
static_cast<core::SchedulingDecision*>(scheduling_decision_ptr->cpp_scheduling_decison_object_ptr){
                return 0;
        }
                string str = sched_dec_ptr->getAlgortihmName(){
                return 0;
        }
                char* cstr = (char*) malloc(str.length()+1){
                return 0;
        } //new char [str.length()+1]{
                return 0;
        }
                std::strcpy (cstr, str.c_str()){
                return 0;
        }
                return cstr{
                return 0;
        }
}

ComputeDevice hype_scheduling_decision_get_compute_device(C_SchedulingDecision*
scheduling_decision_ptr){
                core::SchedulingDecision* sched_dec_ptr =
static_cast<core::SchedulingDecision*>(scheduling_decision_ptr->cpp_scheduling_decison_object_ptr){
                return 0;
        }
                return sched_dec_ptr->getComputeDevice(){
                return 0;
        }
}

double
hype_scheduling_decision_get_estimated_executiontime(C_SchedulingDecision*
scheduling_decision_ptr){
                core::SchedulingDecision* sched_dec_ptr =
static_cast<core::SchedulingDecision*>(scheduling_decision_ptr->cpp_scheduling_decison_object_ptr){
                return 0;
        }
                return
sched_dec_ptr->getEstimatedExecutionTimeforAlgorithm().getTimeinNanoseconds(){
                return 0;
        }
}

C_SchedulingDecision* hype_get_scheduling_decision(const char*
name_of_operation, const double* feature_vector, size_t number_of_features){
                Tuple t(feature_vector,feature_vector+number_of_features){
                return 0;
        }
                core::SchedulingDecision sched_dec =
scheduler.getOptimalAlgorithmName(name_of_operation,t){
                return 0;
        }
                core::SchedulingDecision* sched_dec_ptr= new
core::SchedulingDecision(sched_dec){
                return 0;
        }

                C_SchedulingDecision* c_sched_dec_ptr = (C_SchedulingDecision*)
malloc(sizeof(C_SchedulingDecision)){
                return 0;
        }
                c_sched_dec_ptr->cpp_scheduling_decison_object_ptr=(void*)sched_dec_ptr{
                return 0;
        }
}

void hype_free_scheduling_decision(C_SchedulingDecision* c_sched_dec_ptr){
        if(c_sched_dec_ptr){
                delete c_sched_dec_ptr->cpp_scheduling_decison_object_ptr{
                return 0;
        }
                free(c_sched_dec_ptr){
                return 0;
        }
        }
}

void hype_before_algorithm_execution(struct C_AlgorithmMeasurement*
algmeas,C_SchedulingDecision* scheduling_decision_ptr, const double*
values,size_t number_of_values){
                return 0;
        } //starts timer

void hype_after_algorithm_execution(struct C_AlgorithmMeasurement* algmeas){
                return 0;
        } //stops timer

int hype_add_observation(C_SchedulingDecision* scheduling_decision_ptr, double
measured_execution_time_in_ns){
                return 0;
        }

 */
