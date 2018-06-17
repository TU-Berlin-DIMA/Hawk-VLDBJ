
#include <core/algorithm.hpp>
#include <plugins/statistical_methods/multi_dim_least_squares.hpp>

#include <util/begin_ptr.hpp>

#include <iostream>

#include <ap.h>
#include <interpolation.h>

using namespace std;

namespace hype {
namespace core {

Least_Squares_Method_2D::Least_Squares_Method_2D()
    : StatisticalMethod_Internal("Least Squares 2D"),
      degree_of_polynomial_(2),
      polynomial_computed_(false),
      objHeArray_() {}

Least_Squares_Method_2D::~Least_Squares_Method_2D() {}

const EstimatedTime Least_Squares_Method_2D::computeEstimation(
    const Tuple& input_values) {
  if (input_values.size() != 2) {
    std::cout << "FATAL ERROR: Least_Squares_Method_2D cannot handle input "
                 "dimensions other than 2"
              << std::endl;
    std::cout << "Feature Vector Size: " << input_values.size() << std::endl;
    exit(-1);
  }
  // negative estimation is invalid value, allowing to distinguish between a
  // regular estimation and a dummy value, which is returned if Algorithm is in
  // trainingphase
  // double returnval=-1;
  if (polynomial_computed_) {
    double dblEstimation = 0.0;
    // if (isActive_){
    assert((unsigned int)objHeArray_.length() == input_values.size());
    // cout << "Estimation:";
    for (unsigned int i = 0; i < input_values.size(); i++) {
      dblEstimation += objHeArray_[i] * input_values[i];
      // cout << objHeArray_[i] << "*" << input_values[i] << "+";
    }
    // cout << "=" << dblEstimation << endl;
    //}
    return EstimatedTime(dblEstimation);
  }

  return EstimatedTime(-1);

  //		if(polynomial_computed_){
  //			//returnval =
  // alglib::barycentriccalc(timeestimationpolynomial_,input_values[0]);
  //		}
  //		assert(true==false);
  //		return EstimatedTime(returnval);
}

bool Least_Squares_Method_2D::inTrainingPhase() const throw() {
  return !polynomial_computed_;  // if polynomial is not computed, we are in the
                                 // initial trainingphase
}

void Least_Squares_Method_2D::retrain() { polynomial_computed_ = false; }

std::string Least_Squares_Method_2D::getName() const {
  return "Least Squares 2D";
}

bool Least_Squares_Method_2D::recomuteApproximationFunction(
    Algorithm& algorithm) {
  //			int intMax = (int) vecValue->size();
  //			if(intMax == 0 || (int)vecFeature->size() % intMax !=
  // 0){return
  // false;}
  //			int intMaxInner = (int)vecFeature->size() /  intMax;

  polynomial_computed_ = true;

  std::vector<Tuple> features = algorithm.getAlgorithmStatistics()
                                    .executionHistory_.getColumnFeatureValues();

  std::vector<MeasuredTime> measurements =
      algorithm.getAlgorithmStatistics()
          .executionHistory_.getColumnMeasurements();

  assert(!features.empty());

  vector<double> features_array(features.size() * features[0].size());

  // cout << "Matrix:" << endl;
  for (unsigned int i = 0; i < features.size(); i++) {
    Tuple t = features[i];
    for (unsigned int j = 0; j < features[0].size(); j++) {
      features_array[i * features[0].size() + j] = t[j];
      // cout << t[j]  << " "; //<< endl;
    }
    // cout << endl;
  }

  alglib::real_2d_array objArrayMulti;
  objArrayMulti.setlength(features.size(),
                          features[0].size());  //(intMax, intMaxInner);
  objArrayMulti.setcontent(features.size(), features[0].size(),
                           util::begin_ptr(features_array));

  vector<double> measurements_array(measurements.size());
  for (unsigned int i = 0; i < measurements.size(); i++) {
    measurements_array[i] = measurements[i].getTimeinNanoseconds();
  }

  alglib::real_1d_array objArray;
  objArray.setlength(measurements_array.size());
  objArray.setcontent(measurements_array.size(),
                      util::begin_ptr(measurements_array));

  alglib::ae_int_t intInfo;
  alglib::real_1d_array objResult;
  // objResult.setlength(intMaxInner);
  alglib::lsfitreport objReport;
  alglib::lsfitlinear(objArray, objArrayMulti, intInfo, objResult, objReport);

  // std::cout << "Ausgabe" << objResult[0] << "  "<< objResult[1]  << "  " <<
  // objResult[2] << "\n";

  /*! \todo add lock if neccessary*/
  objHeArray_ = objResult;

  // isActive_ = true;
  return true;
}

//	static Least_Squares_Method_2D* Least_Squares_Method_2D::create(){
//		return new Least_Squares_Method_2D();
//	}

}  // end namespace core
}  // end namespace hype
