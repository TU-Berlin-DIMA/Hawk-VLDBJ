/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <string>
//#include <vector>
#include <config/exports.hpp>
//#include <core/operation.hpp>
#include <config/global_definitions.hpp>
#include <core/measurementpair.hpp>
#include <core/specification.hpp>
#include <core/time_measurement.hpp>

//#ifdef HYPE_USE_MEMORY_COST_MODELS
#include <core/device_memory.hpp>
//#endif

/* temporarily disable warnings because of missing STL DLL interface */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace hype {

  class Scheduler;  // forward declaration

  namespace core {

    class Algorithm;  // forward declaration

    uint64_t getUniqueID();

    /*!
     *
     *
     *  \brief     This class represents a scheduling decision for an operation
     *w.r.t. a user specified set of features of the input data.
     *  \details   A user has to execute the algortihm suggested by the
     *SchedulingDecision, or the library will not work. The
     *		 			user can determine the algorithm to
     *excute
     *by
     *calling
     *getNameofChoosenAlgorithm(). The user can then measure the execution
             *					time of the algorithm by using
     *an
     *AlgorithmMeasurement
     *object.
     *  \author    Sebastian Breß
     *  \version   0.1
     *  \date      2012
     *  \todo change interface, so it stores a pointer to the choosen algorithm,
     *and update algorithm timestamps in the constructor!
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     *http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    class HYPE_EXPORT SchedulingDecision {
     public:
      /*! \brief constructs a SchedulingDecision object by assigning neccessary
       * informations to all fields of the object
       * \param alg_ref Reference to Algorithm that was choosen
       * \param estimated_time_for_algorithm estimated execution time for
       * algorithm
       * \param feature_values features of the input data set that were
       * previously specified by the user
       */
      SchedulingDecision(Algorithm& alg_ref,
                         const EstimatedTime& estimated_time_for_algorithm,
                         const Tuple& feature_values,
                         MemoryChunkPtr mem_chunk = MemoryChunkPtr());

      ~SchedulingDecision();

      /*! \brief returns the name of the choosen algorithm
       *  \return name of choosen algorithm
       */
      const std::string getNameofChoosenAlgorithm() const;
      /*! \brief returns the estimated execution time of the choosen algorithm
       *  \return estimated execution time of the choosen algorithm
       */
      const EstimatedTime getEstimatedExecutionTimeforAlgorithm() const;
      /*! \brief returns the feature values that were the basis for this
       * decision
       *  \return feature values	of input data set
       */
      const Tuple getFeatureValues() const;
      /*! \brief returns the ComputeDevice the chosen algorithm utilizes
       *  \return ComputeDevice the chosen algorithm utilizes
       */
      const DeviceSpecification getDeviceSpecification() const throw();

      MemoryChunkPtr getMemoryChunk() const throw();

      //                                SchedulingDecision(const
      //                                hype::core::SchedulingDecision&);
      //
      //                                SchedulingDecision& operator=(const
      //                                hype::core::SchedulingDecision&);

      bool operator==(const SchedulingDecision& sched_dec) const;

     private:
      /*! \brief name of choosen algorithm*/
      // std::string name_of_algorithm_;
      /*! \brief reference to algorithm*/
      Algorithm* alg_ref_;
      /*! \brief estimated execution time of the choosen algorithm*/
      EstimatedTime estimated_time_for_algorithm_;
      /*! \brief feature values of input data set*/
      Tuple feature_values_;
      /*! \brief unique id which is unique for each schduling decision */
      uint64_t scheduling_id_;
      //#ifdef HYPE_USE_MEMORY_COST_MODELS
      /*! \brief pointer to memory chunk that is used by this operator */
      MemoryChunkPtr mem_chunk_;
      //#endif
    };

    //                typedef std::vector<SchedulingDecision>
    //                SchedulingDecisionVector;
    //                typedef boost::shared_ptr<SchedulingDecisionVector>
    //                SchedulingDecisionVectorPtr;

  }  // end namespace core
}  // end namespace hype

#ifdef _MSC_VER
#pragma warning(pop)
#endif
