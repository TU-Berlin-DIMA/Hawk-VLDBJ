/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/

#pragma once

#include <iostream>
#include <string>
#include <vector>
//#include <boost/shared_ptr.hpp>
#include <ostream>

// HyPE includes
//#include <core/recomputation_heuristic.hpp>
#include <core/time_measurement.hpp>

namespace hype {
  namespace core {

    /*!
     *
     *
     *  \brief     This class represents the result of a measurement consisting
     of the features of a data set, the measured execution time of an algorithm
     and an estimated
                                            execution time for an algortihm
     computed by a statistical method.
     *
     *  \author    Sebastian Breß
     *  \version   0.1
     *  \date      2012
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    class MeasurementPair {
     public:
      MeasurementPair();
      MeasurementPair(const Tuple& feature_values, MeasuredTime measured_time,
                      EstimatedTime estimated_time);

      const Tuple& getFeatureValues() const;
      //{ return feature_values.getFeatureValues(); }

      const MeasuredTime& getMeasuredTime() const;

      const EstimatedTime& getEstimatedTime() const;

      const std::string toPlainString() const;

     private:
      Tuple feature_values_;
      MeasuredTime measured_time_;
      EstimatedTime estimated_time_;
    };

    std::ostream& operator<<(std::ostream& out, MeasurementPair& pair);
    //{
    //    // Since operator<< is a friend of the Point class, we can access
    //    // Point's members directly.
    //    out << "([";
    //    for(int i=0;i<pair.getFeatureValues().size();i++){
    //    	out << pair.getFeatureValues().at(i) << ",";
    //    }
    //    out << "]";
    //
    //    out << pair.getMeasuredTime().getTimeinNanoseconds() << ", "
    //    	  << pair.getEstimatedTime().getTimeinNanoseconds() << ")" <<
    //    std::endl;
    //
    //    return out;
    //}

  }  // end namespace core
}  // end namespace hype
