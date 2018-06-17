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
#include <map>
#include <string>
#include <vector>

#include <core/algorithm.hpp>
#include <core/optimization_criterion.hpp>

namespace hype {
  namespace core {

    /*!
     *
     *
     *  \brief     This class represents an operation, which abstracts from
     *algorithms and summarizes them under an abstract task, like sorting, join,
     *selection, projection etc.
     *  \details   An operation gets assigned two things. First is a list of
     *algorithms, which represent the set of algorithms available to process
     *operation O, which is
     *					basically the algorithm pool of the
     *decision
     *model.
     *Second
     *is
     *an
     *optimization_criterion, which is implemented by drived classes of
     *					stemod::core::OptimizationCriterion. The
     *optimization
     *criterion
     *takes the computed estimated execution times of each algorithm and decides
     *on the optimal
     *					algorithm, where the user specifies what
     *is
     *optimal
     *by
     *specifying an OptimizationCriterion.
     *
     *					Note that the
     *stemod::core::OptimizationCriterioc
     *can
     *be
     *exchanged at run-time, because the Operation class uses the pointer to
     *             implementation technique (or pimpl-idiom). This makes it
     *highly adaptable.
     *  \author    Sebastian Breß
     *  \version   0.1
     *  \date      2012
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     *http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    class Operation {
     public:
      typedef std::map<std::string, boost::shared_ptr<Algorithm> >
          MapNameToAlgorithm;

      Operation(const std::string& name);

      ~Operation();

      // void addAlgorithm(std::string name);

      bool addAlgorithm(const std::string& name_of_algorithm,
                        DeviceSpecification comp_dev,
                        const std::string& name_of_statistical_method,
                        const std::string& name_of_recomputation_strategy);

      void removeAlgorithm(const std::string& name);

      // const std::vector< boost::shared_ptr<Algorithm> >
      const SchedulingDecision getOptimalAlgorithm(
          const Tuple& input_values,
          DeviceTypeConstraint dev_constr = ANY_DEVICE);

      /*! \brief returns true if an algrithm named "name_of_algorithm" is
       * registered for this operation and false otherwise*/
      bool hasAlgorithm(const std::string& name_of_algorithm);

      const AlgorithmPtr getAlgorithm(const std::string& name_of_algorithm);

      bool setNewOptimizationCriterion(
          const std::string& name_of_optimization_criterion);

      bool addObservation(const std::string& name_of_algorithm,
                          const MeasurementPair& mp);

      const std::vector<AlgorithmPtr> getAlgorithms();

      const std::map<double, std::string>
      getEstimatedExecutionTimesforAlgorithms(const Tuple& input_values);

      const std::string getName() const throw();

      void incrementNumberOfRightDecisions() throw();
      void incrementNumberOfTotalDecisions() throw();

      uint64_t getNextTimestamp() throw();
      uint64_t getCurrentTimestamp() const throw();

      const std::string toString();

     private:
      // std::map<std::string,boost::shared_ptr<Algorithm> >
      // map_algorithmname_to_pointer_;
      MapNameToAlgorithm map_algorithmname_to_pointer_;
      boost::shared_ptr<OptimizationCriterion_Internal>
          ptr_to_optimization_criterion_;
      // boost::shared_ptr<OptimizationCriterion> optimization_criterion_;
      std::string name_;
      unsigned int number_of_right_decisions_;
      unsigned int number_of_total_decisions_;
      uint64_t logical_time_;
    };

  }  // end namespace core
}  // end namespace hype
