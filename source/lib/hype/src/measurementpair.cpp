
#include <boost/lexical_cast.hpp>
#include <core/measurementpair.hpp>

namespace hype {
namespace core {

MeasurementPair::MeasurementPair()
    : feature_values_(), measured_time_(0), estimated_time_(0) {}

MeasurementPair::MeasurementPair(const Tuple& feature_values,
                                 MeasuredTime measured_time,
                                 EstimatedTime estimated_time)
    : feature_values_(feature_values),
      measured_time_(measured_time),
      estimated_time_(estimated_time) {}

const Tuple& MeasurementPair::getFeatureValues() const {
  return feature_values_;
}
//{ return feature_values.getFeatureValues(); }

const MeasuredTime& MeasurementPair::getMeasuredTime() const {
  return measured_time_;
}

const EstimatedTime& MeasurementPair::getEstimatedTime() const {
  return estimated_time_;
}

const std::string MeasurementPair::toPlainString() const {
  std::string result;
  for (unsigned int i = 0; i < feature_values_.size(); i++) {
    result += boost::lexical_cast<std::string>(feature_values_[i]);
    result += "\t";
    // if(i<feature_values_.size()-1) result+=",";
  }
  result +=
      boost::lexical_cast<std::string>(measured_time_.getTimeinNanoseconds());
  result += "\t";
  result +=
      boost::lexical_cast<std::string>(estimated_time_.getTimeinNanoseconds());
  return result;
}

std::ostream& operator<<(std::ostream& out, MeasurementPair& pair) {
  if (pair.getFeatureValues().size() > 1) {
    // Since operator<< is a friend of the Point class, we can access
    // Point's members directly.
    out << "([";
    for (unsigned int i = 0; i < pair.getFeatureValues().size(); i++) {
      out << pair.getFeatureValues().at(i);
      if (i < pair.getFeatureValues().size() - 1) out << ",";
    }
    out << "], ";

    out << pair.getMeasuredTime().getTimeinNanoseconds() << ", "
        << pair.getEstimatedTime().getTimeinNanoseconds()
        << ")";  // << std::endl;
  } else if (pair.getFeatureValues().size() == 1) {
    // Since operator<< is a friend of the Point class, we can access
    // Point's members directly.
    for (unsigned int i = 0; i < pair.getFeatureValues().size(); i++) {
      out << pair.getFeatureValues().at(i);
      out << "\t";
    }

    out << pair.getMeasuredTime().getTimeinNanoseconds() << "\t"
        << pair.getEstimatedTime()
               .getTimeinNanoseconds();  //<< ")"; // << std::endl;
  } else {
    out << "[[[INVALID MEASUREMENT PAIR]]]";
  }

  return out;
}

}  // end namespace core
}  // end namespace hype
