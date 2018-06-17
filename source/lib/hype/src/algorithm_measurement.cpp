
#include <boost/lexical_cast.hpp>
#include <core/algorithm_measurement.hpp>
#include <core/scheduler.hpp>
#include <exception>

using namespace std;

namespace hype {
namespace core {

AlgorithmMeasurement::AlgorithmMeasurement(
    const SchedulingDecision& scheduling_decision)
    : timestamp_begin_(getTimestamp()),
      scheduling_decision_(scheduling_decision) {
  // feature_values_(values),name_of_algorithm_(name_of_algorithm){
}

void AlgorithmMeasurement::afterAlgorithmExecution() {
  uint64_t timestamp_end = getTimestamp();
  if (timestamp_begin_ > timestamp_end) {
    std::cout << "FATAL ERROR: "
              << "STEMOD: measured time duration is negative!!!" << std::endl;
    exit(-1);
  }
  MeasuredTime measured_time(double(timestamp_end - timestamp_begin_));
  if (!quiet) {
    Tuple t = scheduling_decision_.getFeatureValues();
    string input_data_features("(");
    for (unsigned int i = 0; i < t.size(); i++) {
      // cout << t[i] << endl;
      input_data_features += boost::lexical_cast<std::string>(t[i]);
      if (i != t.size() - 1) input_data_features += ", ";
    }
    input_data_features += ")";
    assert(t.size() > 0);

    cout << "Algorithm: '" << scheduling_decision_.getNameofChoosenAlgorithm()
         << "'   Input Data Feature Vector: " << input_data_features
         << "   Estimated Execution Time: "
         << scheduling_decision_.getEstimatedExecutionTimeforAlgorithm()
                .getTimeinNanoseconds()
         << "ns"
         << "   Measured Execution Time: "
         << measured_time.getTimeinNanoseconds() << "ns"
         << "	Relative Error: "
         << (measured_time.getTimeinNanoseconds() -
             scheduling_decision_.getEstimatedExecutionTimeforAlgorithm()
                 .getTimeinNanoseconds()) /
                scheduling_decision_.getEstimatedExecutionTimeforAlgorithm()
                    .getTimeinNanoseconds()
         << "%" << endl;
  }

  // MeasurementPair mp(scheduling_decision_.getFeatureValues(), measured_time,
  // scheduling_decision_.getEstimatedExecutionTimeforAlgorithm());
  if (!core::Scheduler::instance().addObservation(
          scheduling_decision_, measured_time.getTimeinNanoseconds())) {
    string error_message = string("STEMOD: Algorithm '") +
                           scheduling_decision_.getNameofChoosenAlgorithm() +
                           string("' does not exist!");
    cout << error_message << endl;
    throw new invalid_argument(error_message);
  }
}

}  // end namespace core
}  // end namespace hype
