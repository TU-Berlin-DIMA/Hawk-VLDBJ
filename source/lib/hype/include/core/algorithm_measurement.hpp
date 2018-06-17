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

// HyPE includes
#include <config/exports.hpp>
#include <core/measurementpair.hpp>
#include <core/scheduling_decision.hpp>
#include <core/time_measurement.hpp>

namespace hype {
  namespace core {

    /*!
     *  \brief     This class represents an easy to use interface for time
     *measurement of an algorithms execution time.
     *  \details   The user just have to create an object of
     *AlgorithmMeasurement. Then, the algorithm that is to measure is executed.
     *After that, the user has to call
     *             afterAlgorithmExecution(). The framework takes care of the
     *rest.
     *
     *					Internally, afterAlgorithmExecution()
     *stops
     *the
     *time
     *and
     *adds
     *the received data to the algorithms statistics. <b>It is crucial that the
     *application executes the algortihm specified by the SchedulingDecision or
     *the libraries behaviour is undefined.</b>
     *  \author    Sebastian Breß
     *  \version   0.1
     *  \date      2012
     *  \todo		This class is designed to be a local class, for 2
     *different
     *tiem measurements, two different objects have to be instanciated. It is
     *NOT meaningful to create
     * 				an AlgorithmMeasurement object on the heap.
     *Therefore,
     *heap
     *based
     *objects should be forbidden by making the operators new and new[] private.
     *Furtheremore,
     * 				the object should not be copyable, so making the
     *copy
     *cosntructur
     *and copy assignment operator is necessary as well.
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     *http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    class HYPE_EXPORT AlgorithmMeasurement {
     public:
      /*! \brief constructs an AlgorithmMeasurement object
       * \details The constructor will fetch the current time. Therefore, it
       *starts a timer to measure the execution time of the choosen algorithm.
       *          Therefore, the user should construct the AlgorithmMeasurement
       *object directly before the algorithms call. Directly after the algorthm
       *	         finished execution, afterAlgorithmExecution() has to be
       *called to ensure a precise time measurement.
       * \param cheduling_decision a reference to a SchedulingDecision
       */
      explicit AlgorithmMeasurement(
          const SchedulingDecision& scheduling_decision);  // starts timer
      /*! \brief stops the time of an algorithms execution and adds obtained
       *data to algorithm statistics.
       * \details To ensure a precise time measurement,
       *afterAlgorithmExecution() has to be called directly after the algorthm
       *	         finished execution.
       */
      void afterAlgorithmExecution();

     private:
      /*! \brief timestamp fetched at construction time, used for time
       * measurement*/
      uint64_t timestamp_begin_;
      /*! \brief scheduling decision, where this object belongs to*/
      SchedulingDecision scheduling_decision_;
      /*! \brief flag indicating, whether the automatic measurement feature
       * should be used, */
      // bool manual_management_;
    };

  }  // end namespace core
}  // end namespace hype
