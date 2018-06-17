
#pragma once

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <core/statistical_method.hpp>
#include <core/time_measurement.hpp>

// alglib includes
#include <interpolation.h>

namespace hype {
  namespace core {

    class Least_Squares_Method_1D : public StatisticalMethod_Internal {
     public:
      Least_Squares_Method_1D();

      virtual const EstimatedTime computeEstimation(const Tuple& input_values);

      virtual bool recomuteApproximationFunction(Algorithm& algorithm);

      virtual bool inTrainingPhase() const throw();

      virtual void retrain();

      virtual std::string getName() const;

      static StatisticalMethod_Internal* create() {
        return new Least_Squares_Method_1D();
      }

      virtual ~Least_Squares_Method_1D();

     private:
      alglib::barycentricinterpolant timeestimationpolynomial_;
      unsigned int degree_of_polynomial_;
      bool polynomial_computed_;
    };

  }  // end namespace core
}  // end namespace hype
