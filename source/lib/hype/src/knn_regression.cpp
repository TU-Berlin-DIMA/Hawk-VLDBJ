
#include <core/algorithm.hpp>
#include <plugins/statistical_methods/knn_regression.hpp>

#include <core/time_measurement.hpp>
#include <iostream>
#include <util/begin_ptr.hpp>

#include <ap.h>
#include <interpolation.h>
#include <boost/circular_buffer.hpp>
#include <functional>
#include <queue>
#include <vector>

using namespace std;

namespace hype {
namespace core {

template <size_t NUMBER_OF_FEATURES>
class KNN_Regression_Impl : public KNN_Regression_Model {
 public:
  KNN_Regression_Impl();
  const EstimatedTime computeEstimation(const Tuple& input_values);
  bool recomuteApproximationFunction(Algorithm& algorithm);
  bool inTrainingPhase() const throw();
  void retrain();

 private:
  const EstimatedTime getNearestNeighbourEstimation(const Tuple& input_values);
  const EstimatedTime getKNearestNeighbourEstimation(const Tuple& input_values,
                                                     size_t k);
  struct Record {
    Record(const Tuple& input_values, double measured_time);
    double features[NUMBER_OF_FEATURES];
    double measured_time;
    double computeDistance(const Record& rec) const;
    double computeWeightedDistance(const Record& rec,
                                   const std::vector<double>& weights) const;
    bool operator<(const Record& rec) const;
    std::string toString() const;
  };
  std::vector<Record> records_;
  std::vector<double> feature_weights_;
  //            bool operator<(const Record& rec_l, const Record& rec_r);
};

template <size_t NUMBER_OF_FEATURES>
KNN_Regression_Impl<NUMBER_OF_FEATURES>::Record::Record(
    const Tuple& input_values_a, double measured_time_a)
    : features(), measured_time(measured_time_a) {
  assert(input_values_a.size() == NUMBER_OF_FEATURES);
  for (unsigned int i = 0; i < NUMBER_OF_FEATURES; ++i) {
    features[i] = input_values_a[i];
  }
}

template <size_t NUMBER_OF_FEATURES>
double KNN_Regression_Impl<NUMBER_OF_FEATURES>::Record::computeDistance(
    const Record& rec) const {
  double sum = 0;
  double tmp = 0;
  for (unsigned int i = 0; i < NUMBER_OF_FEATURES; ++i) {
    tmp = (this->features[i] - rec.features[i]);
    sum += (tmp * tmp);
  }
  return sqrt(sum);
}

template <size_t NUMBER_OF_FEATURES>
double KNN_Regression_Impl<NUMBER_OF_FEATURES>::Record::computeWeightedDistance(
    const Record& rec, const std::vector<double>& weights) const {
  double sum = 0;
  double tmp = 0;
  for (unsigned int i = 0; i < NUMBER_OF_FEATURES; ++i) {
    tmp = (this->features[i] - rec.features[i]) * weights[i];
    sum += (tmp * tmp);
  }
  return sqrt(sum);
}

template <size_t NUMBER_OF_FEATURES>
bool KNN_Regression_Impl<NUMBER_OF_FEATURES>::Record::operator<(
    const Record& rec) const {
  for (unsigned int i = 0; i < NUMBER_OF_FEATURES; ++i) {
    if (this->features[i] > rec.features[i]) return false;
    if (this->features[i] < rec.features[i]) return true;
  }
  // ok, feature vector is equal, check measured value
  return this->measured_time < rec.measured_time;
}

template <size_t NUMBER_OF_FEATURES>
std::string KNN_Regression_Impl<NUMBER_OF_FEATURES>::Record::toString() const {
  std::stringstream ss;
  ss << "[(";
  for (unsigned int i = 0; i < NUMBER_OF_FEATURES; ++i) {
    ss << this->features[i];
    if ((i + 1) < NUMBER_OF_FEATURES) {
      ss << ", ";
    }
  }
  ss << "), " << this->measured_time << "]";
  return ss.str();
}

template <size_t NUMBER_OF_FEATURES>
KNN_Regression_Impl<NUMBER_OF_FEATURES>::KNN_Regression_Impl()
    : records_(), feature_weights_() {
  // we start with default weights
  for (unsigned int i = 0; i < NUMBER_OF_FEATURES; ++i) {
    feature_weights_.push_back(1.0);
  }
}

template <size_t NUMBER_OF_FEATURES>
const EstimatedTime
KNN_Regression_Impl<NUMBER_OF_FEATURES>::getNearestNeighbourEstimation(
    const Tuple& input_values) {
  double min_dist = std::numeric_limits<double>::max();
  unsigned int nearest_neighbour_index = 0;
  Record rec(input_values, 0);
  if (records_.empty()) return EstimatedTime(-1);
  for (unsigned int i = 0; i < records_.size(); ++i) {
    double distance = records_[i].computeDistance(rec);
    if (min_dist > distance) {
      min_dist = distance;
      nearest_neighbour_index = i;
    }
  }
  return EstimatedTime(records_[nearest_neighbour_index].measured_time);
}

//        template <size_t NUMBER_OF_FEATURES>
//        bool KNN_Regression_Impl<NUMBER_OF_FEATURES>::operator<(const Record&
//        rec_l, const Record& rec_r){
//            return rec_l.operator <(rec_r);
//        }

template <size_t NUMBER_OF_FEATURES>
const EstimatedTime
KNN_Regression_Impl<NUMBER_OF_FEATURES>::getKNearestNeighbourEstimation(
    const Tuple& input_values, size_t k) {
  typedef pair<double, Record> DistanceRecord;
  // Record with largest distance to query point is always at the top of the
  // priority queue
  std::priority_queue<DistanceRecord> nearest_neighbours;
  Record rec(input_values, 0);
  for (unsigned int i = 0; i < records_.size(); ++i) {
    // double distance = records_[i].computeDistance(rec);
    double distance =
        records_[i].computeWeightedDistance(rec, this->feature_weights_);
    if (nearest_neighbours.size() < k) {
      nearest_neighbours.push(std::make_pair(distance, records_[i]));
    } else {
      if (nearest_neighbours.top().first > distance) {
        nearest_neighbours.pop();
        nearest_neighbours.push(std::make_pair(distance, records_[i]));
      }
    }
  }
  size_t number_of_nearest_neighbours = nearest_neighbours.size();
  //            double sum=0;
  //            while(!nearest_neighbours.empty()){
  //                sum+=nearest_neighbours.top().second.measured_time;
  //                if(!quiet && verbose && debug) std::cout << "NN: " <<
  //                nearest_neighbours.top().second.toString() << " (distance: "
  //                << nearest_neighbours.top().first <<  ")" << std::endl;
  //                nearest_neighbours.pop();
  //            }
  //            double average = sum/number_of_nearest_neighbours;

  // perform gamma trimming for k elements and gamma = 1
  // simple and effective outlier removal technique
  // sort the values, then thow away the smallest and the largest element
  // then, compute the average with the remaining values
  std::vector<double> nns;
  while (!nearest_neighbours.empty()) {
    nns.push_back(nearest_neighbours.top().second.measured_time);
    if (!quiet && verbose && debug)
      std::cout << "NN: " << nearest_neighbours.top().second.toString()
                << " (distance: " << nearest_neighbours.top().first << ")"
                << std::endl;
    nearest_neighbours.pop();
  }
  std::sort(nns.begin(), nns.end());
  //            std::cout << "records_.size(): " << records_.size() << " k: " <<
  //            k << " #nn: " << nns.size()  << std::endl;
  if (records_.size() >= k) {
    assert(nns.size() == k);
  }
  double sum = 0;
  double average = 0;

  //            if(nns.size()>k){
  //                nns.erase(nns.begin());
  //                nns.erase(nns.end()-1);
  //////                sum = std::accumulate(++nns.begin(), --nns.end(), 0);
  ////                sum = std::accumulate(nns.begin(), nns.end(), 0);
  ////                average = sum/(number_of_nearest_neighbours-2);
  ////            }else{
  //            }
  sum = std::accumulate(nns.begin(), nns.end(), double(0));
  average = sum / nns.size();
  //            }
  if (!quiet && verbose && debug) {
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Estimation: " << average << std::endl;
  }
  return EstimatedTime(average);
}

template <size_t NUMBER_OF_FEATURES>
const EstimatedTime KNN_Regression_Impl<NUMBER_OF_FEATURES>::computeEstimation(
    const Tuple& input_values) {
  // at the beginning, we have very sparse data, so
  size_t k = 10;
  //            k=std::min(size_t(10),records_.size()/10);
  //            k=std::max(k,size_t(1));
  // uint64_t begin = getTimestamp();
  EstimatedTime est = getKNearestNeighbourEstimation(input_values, k);
  // uint64_t end = getTimestamp();
  // std::cout << "KNN search time:" << double(end-begin)/(1000*1000) << "ms" <<
  // std::endl;
  return est;
  //            double min_dist = std::numeric_limits<double>::max();
  //            unsigned int nearest_neighbour_index=0;
  //            Record rec(input_values, 0);
  //            if(records_.empty()) return EstimatedTime(-1);
  //            for(unsigned int i=0;i<records_.size();++i){
  //                double distance = records_[i].computeDistance(rec);
  //                if(min_dist>distance){
  //                    min_dist=distance;
  //                    nearest_neighbour_index=i;
  //                }
  //            }
  //            return
  //            EstimatedTime(records_[nearest_neighbour_index].measured_time);
}

template <size_t NUMBER_OF_FEATURES>
bool KNN_Regression_Impl<NUMBER_OF_FEATURES>::recomuteApproximationFunction(
    Algorithm& algorithm) {
  std::vector<Tuple> features = algorithm.getAlgorithmStatistics()
                                    .executionHistory_.getColumnFeatureValues();
  std::vector<MeasuredTime> measurements =
      algorithm.getAlgorithmStatistics()
          .executionHistory_.getColumnMeasurements();
  records_.clear();
  for (unsigned int i = 0; i < features.size(); ++i) {
    records_.push_back(
        Record(features[i], measurements[i].getTimeinNanoseconds()));
  }
  // compute maximal value for each feature
  std::vector<double> max_feature_values(NUMBER_OF_FEATURES,
                                         std::numeric_limits<double>::min());

  for (unsigned int i = 0; i < records_.size(); ++i) {
    for (unsigned int k = 0; k < NUMBER_OF_FEATURES; ++k) {
      if (max_feature_values[k] < records_[i].features[k]) {
        max_feature_values[k] = records_[i].features[k];
      }
    }
  }
  // derive weighting vector from maximal values of features
  if (!quiet && verbose && debug) std::cout << "weights: ";
  for (unsigned int i = 0; i < NUMBER_OF_FEATURES; ++i) {
    this->feature_weights_[i] = 1 / max_feature_values[i];
    if (!quiet && verbose && debug) {
      std::cout << feature_weights_[i] << std::endl;
      if (i + 1 < NUMBER_OF_FEATURES) {
        std::cout << ", ";
      } else {
        std::cout << std::endl;
      }
    }
  }

  return true;
}

template <size_t NUMBER_OF_FEATURES>
bool KNN_Regression_Impl<NUMBER_OF_FEATURES>::inTrainingPhase() const throw() {
  return false;
}

template <size_t NUMBER_OF_FEATURES>
void KNN_Regression_Impl<NUMBER_OF_FEATURES>::retrain() {
  return;
}

KNN_Regression::KNN_Regression()
    : StatisticalMethod_Internal("KNN_Regression"),
      polynomial_computed_(false),
      knn_regression_model_() {}

KNN_Regression::~KNN_Regression() {}

const EstimatedTime KNN_Regression::computeEstimation(
    const Tuple& input_values) {
  if (knn_regression_model_ == NULL) {
    if (input_values.size() == 1) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<1>());
    } else if (input_values.size() == 2) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<2>());
    } else if (input_values.size() == 3) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<3>());
    } else if (input_values.size() == 4) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<4>());
    } else if (input_values.size() == 5) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<5>());
    }
  }

  // negative estimation is invalid value, allowing to distinguish between a
  // regular estimation and a dummy value, which is returned if Algorithm is in
  // trainingphase
  // double returnval=-1;
  if (polynomial_computed_) {
    EstimatedTime ret = knn_regression_model_->computeEstimation(input_values);
    assert(ret.getTimeinNanoseconds() >= 0);
    return ret;
  }
  return EstimatedTime(-1);
}

bool KNN_Regression::inTrainingPhase() const throw() {
  return !polynomial_computed_;  // if polynomial is not computed, we are in the
                                 // initial trainingphase
}

void KNN_Regression::retrain() { polynomial_computed_ = false; }

std::string KNN_Regression::getName() const { return "KNN_Regression"; }

bool KNN_Regression::recomuteApproximationFunction(Algorithm& algorithm) {
  polynomial_computed_ = true;
  if (knn_regression_model_ == NULL) {
    assert(!algorithm.getAlgorithmStatistics()
                .executionHistory_.getColumnFeatureValues()
                .empty());
    size_t num_of_features = algorithm.getAlgorithmStatistics()
                                 .executionHistory_.getColumnFeatureValues()
                                 .front()
                                 .size();
    if (num_of_features == 1) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<1>());
    } else if (num_of_features == 2) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<2>());
    } else if (num_of_features == 3) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<3>());
    } else if (num_of_features == 4) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<4>());
    } else if (num_of_features == 5) {
      knn_regression_model_ =
          KNN_Regression_ModelPtr(new KNN_Regression_Impl<5>());
    }
  }
  this->knn_regression_model_->recomuteApproximationFunction(algorithm);
  return true;
}

//	static KNN_Regression* KNN_Regression::create(){
//		return new KNN_Regression();
//	}

}  // end namespace core
}  // end namespace hype
