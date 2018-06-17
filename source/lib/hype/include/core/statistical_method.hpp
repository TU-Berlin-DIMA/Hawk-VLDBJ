/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// hype includes
#include <core/factory.hpp>
#include <core/time_measurement.hpp>

namespace hype {
  namespace core {

    class Algorithm;  // forward declaration

    class StatisticalMethod_Internal {
      /*! \todo move functionality of keeping a approximation function computed
       * bit from derived classes in this base class*/

     public:
      virtual const EstimatedTime computeEstimation(
          const Tuple& input_values) = 0;

      virtual bool recomuteApproximationFunction(Algorithm& algorithm) = 0;

      virtual bool inTrainingPhase() const throw() = 0;

      virtual void retrain() = 0;

      virtual std::string getName() const = 0;

      virtual ~StatisticalMethod_Internal() {}

     protected:
      StatisticalMethod_Internal(const std::string& name);

     private:
      std::string name_;
    };

    // factory function
    const boost::shared_ptr<StatisticalMethod_Internal>
    getNewStatisticalMethodbyName(std::string name);

    typedef core::Factory<StatisticalMethod_Internal, std::string>
        StatisticalMethodFactory;  // aFactory;
    // typedef Loki::Singleton<StatisticalMethodFactory>
    // StatisticalMethodFactorySingleton;

    class StatisticalMethodFactorySingleton {
     public:
      static StatisticalMethodFactory& Instance();
    };

    typedef boost::shared_ptr<StatisticalMethod_Internal> StatisticalMethodPtr;
    typedef std::map<std::string, StatisticalMethodPtr> StatisticalMethodMap;

  }  // end namespace core
}  // end namespace hype
