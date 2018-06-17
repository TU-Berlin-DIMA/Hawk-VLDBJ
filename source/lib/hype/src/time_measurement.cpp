
#include <boost/chrono.hpp>
#include <core/time_measurement.hpp>

using namespace boost::chrono;

namespace hype {
namespace core {

EstimatedTime::EstimatedTime() : value_in_nanoseconds_(-1) {}

EstimatedTime::EstimatedTime(double time_in_nanoseconds)
    : value_in_nanoseconds_(time_in_nanoseconds) {}

double EstimatedTime::getTimeinNanoseconds() const {
  return value_in_nanoseconds_;
}

MeasuredTime::MeasuredTime() : value_in_nanoseconds_(-1) {}

MeasuredTime::MeasuredTime(double time_in_nanoseconds)
    : value_in_nanoseconds_(time_in_nanoseconds) {}

double MeasuredTime::getTimeinNanoseconds() const {
  return value_in_nanoseconds_;
}

uint64_t getTimestamp() {
  high_resolution_clock::time_point tp = high_resolution_clock::now();
  nanoseconds dur = tp.time_since_epoch();

  return (uint64_t)dur.count();
}

}  // end namespace core
}  // end namespace hype
