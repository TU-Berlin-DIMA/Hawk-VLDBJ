
#pragma once

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <core/statistical_method.hpp>
#include <core/time_measurement.hpp>

// alglib includes
#include <interpolation.h>

#include <boost/shared_ptr.hpp>

namespace hype {
  namespace core {

    class KNN_Regression_Model {
     public:
      virtual const EstimatedTime computeEstimation(
          const Tuple& input_values) = 0;
      virtual bool recomuteApproximationFunction(Algorithm& algorithm) = 0;
      virtual bool inTrainingPhase() const throw() = 0;
      virtual void retrain() = 0;
      virtual ~KNN_Regression_Model() {}
    };
    typedef boost::shared_ptr<KNN_Regression_Model> KNN_Regression_ModelPtr;

    class KNN_Regression : public StatisticalMethod_Internal {
     public:
      KNN_Regression();

      virtual const EstimatedTime computeEstimation(const Tuple& input_values);

      virtual bool recomuteApproximationFunction(Algorithm& algorithm);

      virtual bool inTrainingPhase() const throw();

      virtual void retrain();

      std::string getName() const;

      static StatisticalMethod_Internal* create() {
        return new KNN_Regression();
      }

      virtual ~KNN_Regression();

     private:
      bool polynomial_computed_;
      KNN_Regression_ModelPtr knn_regression_model_;
    };

  }  // end namespace core
}  // end namespace hype
