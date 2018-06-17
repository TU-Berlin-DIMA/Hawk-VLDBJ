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
#include <config/global_definitions.hpp>
#include <core/factory.hpp>
#include <core/scheduling_decision.hpp>
#include <core/time_measurement.hpp>

namespace hype {
  namespace core {

    class Operation;  // forward declaration, so that plugins can get pointer to
                      // operation

    /*!
     *
     *
     *  \brief     Base class for all optimization criterions.
     *  \details   OptimizationCriterion enables the user to choose a (likely)
     *optimal algorithm for an operation w.r.t. the features of the dataset to
     *process, which are stored in a Tuple object.
     * 				The internal function
     *getOptimalAlgorithm_internal()
     *has
     *to
     *be
     *implemented by derived classes.
     *  \author    Sebastian Breß
     *  \version   0.1
     *  \date      2012
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     *http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    class OptimizationCriterion_Internal {
     public:
      /*! \brief gets a Scheduling Decision with the algorithm that is (likely
       to be) optimal w.r.t. to an optimization criterion
       * \details calls the pure virtual function getOptimalAlgorithm_internal()
       to ensure functional correctness on one hand (by implementing features
       that would have to be reimplemented by the user)
                                      and extensibility on the othere hand,
       meaning a derived class can implement an arbitraty heuristic that tries
       to optimze for a certain optimization criterion.
       * \param input_values features of the input data set that is to process
       by Operation op
       * \param op a reference to the operation to perfrom on a data et
       * \return
       */
      const SchedulingDecision getOptimalAlgorithm(
          const Tuple& input_values, Operation& op,
          DeviceTypeConstraint dev_constr);
      /*! \brief returns the name of the OptimizationCriterion
       */
      const std::string& getName() const;
      /*! \brief Destructor is virtual, because OptimizationCriterion is
       * intended to be a base class.
       */
      virtual ~OptimizationCriterion_Internal();

     protected:
      /*! \brief constructor of class
       * \details
       * \param name_of_optimization_criterion Name of the optimization
       * criterion
       * \param name_of_operation name of the operation, where this
       * OptimizationCriterion belongs to
       * \return
       */
      OptimizationCriterion_Internal(
          const std::string& name_of_optimization_criterion,
          const std::string& name_of_operation);
      /*! \brief map that keeps track which algorithms were decided for how
       * often
       *  \details through algorithms have a statistic for that, it is not up to
       * date during runtime, since there are algorithms in execution, which
       * have not yet terminated
       *  This can lead to a burst execution of operations with one algorithm.
       * This map helps to avoid this by keeping track on the number of
       * DECITIONS for each algorithm*/
      std::map<std::string, unsigned int>
          map_algorithmname_to_number_of_executions_;

     private:
      /*! \brief this function is called by getOptimalAlgorithm() to ensure an
       * easy extensibility of the library
       * \details has to be implemented by derived classes
       * \param
       * \return
       */
      virtual const SchedulingDecision getOptimalAlgorithm_internal(
          const Tuple& input_values, Operation& op,
          DeviceTypeConstraint dev_constr) = 0;
      /*! \brief stores name of optimization criterion, which is represented by
       * this object*/
      std::string name_of_optimization_criterion_;
      /*! \brief stores the name of the operation, where this
       * OptimizationCriterion belongs to*/
      std::string name_of_operation_;
    };

    typedef core::Factory<OptimizationCriterion_Internal, std::string>
        OptimizationCriterionFactory;  // aFactory;
    // typedef Loki::Singleton<OptimizationCriterionFactory>
    // OptimizationCriterionFactorySingleton;

    class OptimizationCriterionFactorySingleton {
     public:
      static OptimizationCriterionFactory& Instance();
    };

    /*! \brief factory function taht creates OptimizationCriterion objects*/
    boost::shared_ptr<OptimizationCriterion_Internal>
    getNewOptimizationCriterionbyName(
        const std::string& name_of_optimization_criterion);

  }  // end namespace core
}  // end namespace hype
