
#include <core/measurementpair_ringbuffer.hpp>

namespace hype {
namespace core {

MeasurementPairRingbuffer::MeasurementPairRingbuffer()
    : feature_values_(), measured_times_(), estimated_times_() {
  set_maximal_number_of_measurement_pairs(1000);
}

MeasurementPairRingbuffer::MeasurementPairRingbuffer(size_t size)
    : feature_values_(), measured_times_(), estimated_times_() {
  set_maximal_number_of_measurement_pairs(size);
}

unsigned int MeasurementPairRingbuffer::size() const throw() {
  assert(feature_values_.size() == measured_times_.size());
  assert(feature_values_.size() == estimated_times_.size());
  return feature_values_.size();
}

bool MeasurementPairRingbuffer::store(std::ostream& out) const {
  for (unsigned int i = 0; i < this->size(); i++) {
    MeasurementPair mp(feature_values_[i], measured_times_[i],
                       estimated_times_[i]);
    out << mp << std::endl;
  }
  return true;
}

void MeasurementPairRingbuffer::set_maximal_number_of_measurement_pairs(
    size_t size) {
  feature_values_.set_capacity(size);
  measured_times_.set_capacity(size);
  estimated_times_.set_capacity(size);
}

const std::vector<EstimatedTime>
MeasurementPairRingbuffer::getColumnEstimations() const {
  /*
  //liniarisiert den ringpuffer und gibt zeiger auf das resultierende array
  zurück, für C APIs gedacht
  EstimatedTime* estimatedtimes_array=estimated_times_.linearize();
  // int myints[] = {16,2,77,29};
  //vector<int> fifth (myints, myints + sizeof(myints) / sizeof(int) );
  std::vector<EstimatedTime>
  return_value(estimatedtimes_array,estimatedtimes_array +
  estimated_times_.size() );
  */
  std::vector<EstimatedTime> return_value(estimated_times_.begin(),
                                          estimated_times_.end());
  return return_value;
}

const std::vector<MeasuredTime>
MeasurementPairRingbuffer::getColumnMeasurements() const {
  // MeasuredTime* measured_times_array=measured_times_.linearize();
  // std::vector<MeasuredTime> return_value(measured_times_array,
  // measured_times_array + measured_times_.size() );
  std::vector<MeasuredTime> return_value(measured_times_.begin(),
                                         measured_times_.end());
  return return_value;
}

const std::vector<Tuple> MeasurementPairRingbuffer::getColumnFeatureValues()
    const {
  // Tuple* feature_values_array=feature_values_.linearize();
  // std::vector<Tuple> return_value(feature_values_array, feature_values_array
  // + feature_values_.size() );
  std::vector<Tuple> return_value(feature_values_.begin(),
                                  feature_values_.end());
  return return_value;
}

bool MeasurementPairRingbuffer::addMeasurementPair(const MeasurementPair& mp) {
  feature_values_.push_back(mp.getFeatureValues());
  measured_times_.push_back(mp.getMeasuredTime());
  estimated_times_.push_back(mp.getEstimatedTime());
  return true;
}

void MeasurementPairRingbuffer::clear() throw() {
  feature_values_.clear();
  measured_times_.clear();
  estimated_times_.clear();
}

}  // end namespace core
}  // end namespace hype
