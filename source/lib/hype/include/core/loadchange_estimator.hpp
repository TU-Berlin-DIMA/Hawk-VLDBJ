/***********************************************************************************************************
 Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.
 This program and accompanying materials are made available under the terms of
the
 GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
// default includes
#include <iostream>
// HyPE includes
#include <core/measurementpair.hpp>
// boost includes
#include <boost/circular_buffer.hpp>

namespace hype {
  namespace core {

    class LoadChangeEstimator {
     public:
      LoadChangeEstimator(unsigned int size_of_circular_buffer = 10);

      double getLoadModificator() const throw();

      void add(const MeasurementPair& mp) throw();

     private:
      boost::circular_buffer<double> last_load_factors_;
    };

  }  // end namespace core
}  // end namespace hype
