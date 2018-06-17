#include <util/time_measurement.hpp>

#include <chrono>

namespace CoGaDB {

using NanoSeconds = std::chrono::nanoseconds;
using Clock = std::chrono::high_resolution_clock;

Timestamp getTimestamp() {
  return std::chrono::duration_cast<NanoSeconds>(
             Clock::now().time_since_epoch())
      .count();
}

}  // end namespace CogaDB
