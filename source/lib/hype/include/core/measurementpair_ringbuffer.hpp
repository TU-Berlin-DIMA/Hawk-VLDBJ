/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/

#pragma once

// C++ includes
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
// TR1 includes
#include <boost/shared_ptr.hpp>
// C includes
#include <assert.h>
//#include <stdafx.h>
#include <math.h>

// boost includes
#include <boost/circular_buffer.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

// hype includes
#include <core/measurementpair.hpp>
#include <core/time_measurement.hpp>

// Alglib
//#include <interpolation.h>

namespace hype {
  namespace core {

    class AlgorithmStatistics;

    // stores measurement pairs as columns, not as rows
    class MeasurementPairRingbuffer {
     public:
      MeasurementPairRingbuffer();

      MeasurementPairRingbuffer(size_t size);

      unsigned int size() const throw();

      bool store(std::ostream& out) const;

      void set_maximal_number_of_measurement_pairs(size_t size);

      bool addMeasurementPair(const MeasurementPair& mp);

      // not const functions, since ring buffer has to be linearized to return
      // column
      const std::vector<EstimatedTime> getColumnEstimations() const;

      const std::vector<MeasuredTime> getColumnMeasurements() const;

      const std::vector<Tuple> getColumnFeatureValues() const;

      void clear() throw();
      friend class AlgorithmStatistics;

     private:
      boost::circular_buffer<Tuple> feature_values_;           //(2000);
      boost::circular_buffer<MeasuredTime> measured_times_;    //(2000);
      boost::circular_buffer<EstimatedTime> estimated_times_;  //(2000);
    };

  }  // end namespace core
}  // end namespace hype
