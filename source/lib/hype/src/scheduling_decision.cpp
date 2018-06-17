
#include <core/algorithm.hpp>
#include <core/scheduling_decision.hpp>
#include <util/algorithm_name_conversion.hpp>

#include <iostream>
#include <typeinfo>

namespace hype {
namespace core {

/*
SchedulingDecision::SchedulingDecision(const std::string& name_of_algorithm,
                                                                                                        const EstimatedTime& estimated_time_for_algorithm,
                                                                                                        const Tuple& feature_values)
                                                                                                        : name_of_algorithm_(name_of_algorithm),
                                                                                                          estimated_time_for_algorithm_(estimated_time_for_algorithm),
                                                                                                          feature_values_(feature_values)
{

}*/

uint64_t getUniqueID() {
  static uint64_t counter = 0;
  return ++counter;
}

SchedulingDecision::SchedulingDecision(
    Algorithm& alg_ref, const EstimatedTime& estimated_time_for_algorithm,
    const Tuple& feature_values,
    MemoryChunkPtr mem_chunk)
    :  // name_of_algorithm_(name_of_algorithm),
      alg_ref_(&alg_ref),
      estimated_time_for_algorithm_(estimated_time_for_algorithm),
      feature_values_(feature_values),
      scheduling_id_(getUniqueID()),
      mem_chunk_(mem_chunk) {
  assert(alg_ref_ != NULL);
  alg_ref_->incrementNumberofDecisionsforThisAlgorithm();
}

SchedulingDecision::~SchedulingDecision() {}

//        SchedulingDecision::SchedulingDecision(const
//        hype::core::SchedulingDecision& other) : alg_ref_(other.alg_ref_),
//                                                                                                estimated_time_for_algorithm_(other.estimated_time_for_algorithm_),
//                                                                                                feature_values_(other.feature_values_),
//                                                                                                scheduling_id_(other.scheduling_id_) {
//
//        }
//
//        SchedulingDecision& SchedulingDecision::operator=(const
//        hype::core::SchedulingDecision& other){
//
//            //prevent self assignment
//            if( this != &other ) {
//                this->alg_ref_ = other.alg_ref_;
//                this->estimated_time_for_algorithm_ =
//                other.estimated_time_for_algorithm_;
//                this->feature_values_ = other.feature_values_;
//                this->scheduling_id_ = other.scheduling_id_;
//            }
//            return *this;
//        }

const std::string SchedulingDecision::getNameofChoosenAlgorithm() const {
  // return name_of_algorithm_;
  // return alg_ref_.getName();
  return hype::util::toExternallAlgName(alg_ref_->getName());
  //                std::string name = alg_ref_.getName();
  //                unsigned int pos = name.find_last_of("_");
  //                return name.substr(0,pos); //return the original algorithm
  //                name, which is independent of the device type
}

const EstimatedTime SchedulingDecision::getEstimatedExecutionTimeforAlgorithm()
    const {
  return estimated_time_for_algorithm_;
}

const Tuple SchedulingDecision::getFeatureValues() const {
  return feature_values_;
}

const DeviceSpecification SchedulingDecision::getDeviceSpecification() const
    throw() {
  return alg_ref_->getDeviceSpecification();
}

MemoryChunkPtr SchedulingDecision::getMemoryChunk() const throw() {
  return this->mem_chunk_;
}

bool SchedulingDecision::operator==(const SchedulingDecision& sched_dec) const {
  // Scheduling Decisions are equal if and only if their (unique) ids are equal
  return this->scheduling_id_ == sched_dec.scheduling_id_;
}

}  // end namespace core
}  // end namespace hype
