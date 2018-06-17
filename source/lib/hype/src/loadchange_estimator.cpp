

#include <core/loadchange_estimator.hpp>

using namespace std;

namespace hype {
namespace core {

LoadChangeEstimator::LoadChangeEstimator(unsigned int size_of_circular_buffer)
    : last_load_factors_() {
  last_load_factors_.set_capacity(size_of_circular_buffer);
}

double LoadChangeEstimator::getLoadModificator() const throw() {
  if (last_load_factors_.empty()) {
    return 1.0;
  } else {
    // cout << "Load Factors" << endl;
    boost::circular_buffer<double>::const_iterator it;
    double sum = 0;
    for (it = last_load_factors_.begin(); it != last_load_factors_.end();
         ++it) {
      sum += *it;
      // cout << "DEBUG: Load Factor: " << *it << endl;
    }
    // cout << "DEBUG: Result: " << sum/last_load_factors_.size() << endl;
    return sum /
           last_load_factors_.size();  // return average of last_load_factors_
  }
}

void LoadChangeEstimator::add(const MeasurementPair& mp) throw() {
  last_load_factors_.push_back(mp.getMeasuredTime().getTimeinNanoseconds() /
                               mp.getEstimatedTime().getTimeinNanoseconds());
}

}  // end namespace core
}  // end namespace hype
