
#include <core/algorithm.hpp>
#include <plugins/statistical_methods/least_squares.hpp>

#include <iostream>

using namespace std;

namespace hype {
namespace core {

Least_Squares_Method_1D::Least_Squares_Method_1D()
    : StatisticalMethod_Internal("Least Squares 1D"),
      timeestimationpolynomial_(),
      degree_of_polynomial_(2),
      polynomial_computed_(false) {}

Least_Squares_Method_1D::~Least_Squares_Method_1D() {}

const EstimatedTime Least_Squares_Method_1D::computeEstimation(
    const Tuple& input_values) {
  if (input_values.size() != 1) {
    std::cout << "FATAL ERROR: Least_Squares_Method_1D cannot handle input "
                 "dimensions other than 1"
              << std::endl;
    exit(-1);
  }
  // negative estimation is invalid value, allowing to distinguish between a
  // regular estimation and a dummy value, which is returned if Algorithm is in
  // trainingphase
  double returnval = -1;
  if (polynomial_computed_) {
    returnval =
        alglib::barycentriccalc(timeestimationpolynomial_, input_values[0]);
  }
  // cout << "Esitmated Time: " << returnval << endl;
  return EstimatedTime(returnval);
}

bool Least_Squares_Method_1D::inTrainingPhase() const throw() {
  return !polynomial_computed_;  // if polynomial is not computed, we are in the
                                 // initial trainingphase
}

void Least_Squares_Method_1D::retrain() { polynomial_computed_ = false; }

std::string Least_Squares_Method_1D::getName() const {
  return "Least Squares 1D";
}

bool Least_Squares_Method_1D::recomuteApproximationFunction(
    Algorithm& algorithm) {
  polynomial_computed_ = true;  // initial trainingphase is finished

  alglib::real_1d_array x;  //=algorithm.;
  alglib::real_1d_array y;

  if (!quiet) cout << "recomputing Approximation function" << endl;

  std::vector<MeasuredTime> measured_times =
      algorithm.getAlgorithmStatistics()
          .executionHistory_.getColumnMeasurements();
  int number_of_measurement_pairs = measured_times.size();  //=
  // algorithm.getAlgorithmStatistics().executionHistory_.measured_times_.size();
  std::vector<Tuple> tuples = algorithm.getAlgorithmStatistics()
                                  .executionHistory_.getColumnFeatureValues();
  int number_of_feature_values = tuples.size();

  assert(number_of_feature_values ==
         number_of_measurement_pairs);  // number of measured execution times =
                                        // number of processed data sets

  std::vector<double> times;
  for (unsigned int i = 0; i < measured_times.size(); i++)
    times.push_back(measured_times[i].getTimeinNanoseconds());

  y.setcontent(number_of_measurement_pairs, &times[0]);

  vector<double> datasizes;

  for (unsigned int i = 0; i < tuples.size(); i++) {
    datasizes.push_back(tuples[i][0]);
    // cout << tuples[i][0] << "  counter: " << i << endl;
  }
  // assgin feature values to array x
  x.setcontent(datasizes.size(), &datasizes[0]);

  alglib::ae_int_t m = degree_of_polynomial_;
  alglib::ae_int_t info;
  alglib::barycentricinterpolant p;
  alglib::polynomialfitreport rep;
  // double v;

  // cout << "Length Array x: " << x.length() << "   Length Array y: " <<
  // y.length() << endl;
  assert(x.length() == y.length());

  //
  // Fitting without individual weights
  //
  // NOTE: result is returned as barycentricinterpolant structure.
  //       if you want to get representation in the power basis,
  //       you can use barycentricbar2pow() function to convert
  //       from barycentric to power representation (see docs for
  //       POLINT subpackage for more info).
  //
  try {
    alglib::polynomialfit(x, y, m, info, p, rep);

  } catch (alglib::ap_error e) {
    cout << "FATAL ERROR! "
            "Least_Squares_Method_1D::recomuteApproximationFunction() "
         << endl
         << "Failed to compute polynomial!" << endl
         << "File: " << __FILE__ << " Line: " << __LINE__ << endl;
    // cout <<  << endl;
    exit(-1);
    return false;
  }

  timeestimationpolynomial_ = p;  // update field for estimationpolynomial

  return true;
}

//	static Least_Squares_Method_1D* Least_Squares_Method_1D::create(){
//		return new Least_Squares_Method_1D();
//	}

}  // end namespace core
}  // end namespace hype
