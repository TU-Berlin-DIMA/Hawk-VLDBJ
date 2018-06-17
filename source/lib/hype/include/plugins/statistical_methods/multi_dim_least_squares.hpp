
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

    class Least_Squares_Method_2D : public StatisticalMethod_Internal {
     public:
      Least_Squares_Method_2D();

      virtual const EstimatedTime computeEstimation(const Tuple& input_values);

      virtual bool recomuteApproximationFunction(Algorithm& algorithm);

      virtual bool inTrainingPhase() const throw();

      virtual void retrain();

      std::string getName() const;

      static StatisticalMethod_Internal* create() {
        return new Least_Squares_Method_2D();
      }

      virtual ~Least_Squares_Method_2D();

     private:
      unsigned int degree_of_polynomial_;
      bool polynomial_computed_;
      alglib::real_1d_array objHeArray_;
    };

  }  // end namespace core
}  // end namespace hype
