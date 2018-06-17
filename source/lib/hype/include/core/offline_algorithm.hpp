/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/

#pragma once

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <config/exports.hpp>
#include <config/global_definitions.hpp>
#include <core/measurementpair.hpp>
#include <core/specification.hpp>
#include <core/time_measurement.hpp>

/* temporarily disable warnings because of missing STL DLL interface */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace hype {
  namespace core {

    class HYPE_EXPORT Offline_Algorithm {
     public:
      const std::string& getAlgorithmName() const throw();

      const std::string& getOperationName() const throw();

      DeviceSpecification getDeviceSpecification() const throw();

      unsigned int getNumberOfMeasurementPairs() const throw();

      Offline_Algorithm(DeviceSpecification device, std::string algorithm_name,
                        std::string opname, std::string filepath);

      core::MeasurementPair getNext();

      bool hasNext();

      void reset();

      // the first line of a file determines, how many feature values are
      // expected: for n values, the first n-1 values are considered feature
      // Values and the remaining 1 value is considered the measured execution
      // time
      void loadMeasurementpairsfromFile(std::string filepath);

      void storeInFile(const std::string& file_name);

      void printstoredMeasurementpairs();

      static std::vector<Offline_Algorithm>
      randomize_dataset_of_offline_algorithms(
          std::vector<Offline_Algorithm> offline_algorithms);

     private:
      std::vector<core::MeasurementPair> offline_mesurement_pairs_;
      DeviceSpecification device_;
      std::string name_;
      std::string operation_name_;
      unsigned int current_mesurementpair_index_;
      std::string filepath_;
    };

  }  // end namespace core
}  // end namespace hype

#ifdef _MSC_VER
#pragma warning(pop)
#endif
