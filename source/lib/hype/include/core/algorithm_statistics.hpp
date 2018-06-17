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
#include <string>
#include <vector>

// HyPE includes
#include <core/measurementpair.hpp>
#include <core/measurementpair_ringbuffer.hpp>
#include <core/recomputation_heuristic.hpp>
#include <core/time_measurement.hpp>
// Boost includes
#include <boost/utility.hpp>

#include "specification.hpp"

namespace hype {
  namespace core {

    /*!
     *
     *
     *  \brief     This class represents an information collection of an
     *algorithm. Hence, it is the central place where all statistical
     *information for one algorithm is stored.
     *  \details   The algorithm and the operation the algorithm belongs to are
     *responsible to update the statistical information. Note, that a
     *stemod::core::AlgorithmMeasurement
     *  				object is used for the execution time
     *measurement
     *on
     *the
     *user
     *side and automatically triggers the insertion of a new MeasurementPair as
     *soon as the measurement
     * 				is stopped.
     *
     *					Internally,
     *stemod::core::AlgorithmMeasurement::afterAlgorithmExecution() stops the
     *time and adds the received data to the algorithms statistics.
     *  \author    Sebastian Breß
     *  \version   0.1
     *  \date      2012
     *  \todo		This class is designed to be a member for algorithm that
     *encapsulates statistical information. It is NOT meaningful to create
     * 				an AlgorithmStatistics object on the heap.
     *Therefore,
     *heap
     *based
     *objects should be forbidden by making the operators new and new[] private.
     *Furtheremore,
     * 				the AlgorithmStatistics object should not be
     *copyable,
     *so
     *making
     *the copy cosntructur and copy assignment operator private is necessary as
     *well.
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     *http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    class AlgorithmStatistics {  // : private boost::noncopyable{ //make
                                 // AlgorithmStatistics not copyable

     public:
      // MeasurementPair mp;
      // vector<Measurementpair> >	executionHistory

      AlgorithmStatistics();

      bool writeToDisc(const std::string& operation_name,
                       const std::string& algorithm_name) const;

      MeasurementPairRingbuffer executionHistory_;
      std::vector<double> relative_errors_;
      unsigned int number_of_recomputations_;
      unsigned int number_of_decisions_for_this_algorithm_;
      double average_relative_error_;
      double total_execution_time_;
      unsigned int number_of_terminated_executions_of_this_algorithm_;

      bool storeFeatureVectors(const std::string& path);
      bool loadFeatureVectors(const std::string& path);

      std::string getReport(const std::string& operation_name,
                            const std::string& algorithm_name,
                            const std::string indent_str = "") const;
      double getAverageRelativeError() const;

     private:
      AlgorithmStatistics(const AlgorithmStatistics&);
      AlgorithmStatistics& operator=(const AlgorithmStatistics&);

      //		static void *operator new(size_t size);
      //		static void operator delete(void *ptr);
    };

    typedef boost::shared_ptr<AlgorithmStatistics> AlgorithmStatisticsPtr;

    struct AlgCostModelIdentifier {
      AlgCostModelIdentifier();
      //    : external_alg_name(), pd_t(), pd_m(){
      //
      //    }
      std::string external_alg_name;
      ProcessingDeviceType pd_t;
      ProcessingDeviceMemoryID pd_m;
    };

    bool operator==(AlgCostModelIdentifier a, AlgCostModelIdentifier b);
    bool operator<(AlgCostModelIdentifier a, AlgCostModelIdentifier b);

    class AlgorithmStatisticsManager {
     public:
      static AlgorithmStatisticsManager& instance();

      AlgorithmStatisticsPtr getAlgorithmStatistics(
          const DeviceSpecification& dev_spec, const std::string& alg_name);

     private:
      AlgorithmStatisticsManager();
      AlgorithmStatisticsManager(const AlgorithmStatisticsManager&);
      typedef std::map<AlgCostModelIdentifier, AlgorithmStatisticsPtr>
          AlgorithmStatisticsMap;
      AlgorithmStatisticsMap alg_statistics_;
    };

  }  // end namespace core
}  // end namespace hype
