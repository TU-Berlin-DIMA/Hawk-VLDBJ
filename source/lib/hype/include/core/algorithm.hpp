/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <config/global_definitions.hpp>

#include <boost/shared_ptr.hpp>
#include <core/algorithm_statistics.hpp>
#include <core/loadchange_estimator.hpp>
#include <core/recomputation_heuristic.hpp>
#include <core/specification.hpp>
#include <core/statistical_method.hpp>
#include <core/time_measurement.hpp>
#include <vector>
/*!
* \brief The namespace stemod contains all classes and functions that belongs to
* the framework.
*/
namespace hype {
  /*!
  * \brief The namespace stemod::core contains all classes and functions that
  * belong to the core framework.
  */
  namespace core {

    // forward declaration
    class Operation;

    /*!
     *
     *
     *  \brief     This class represents an algorithm. It abstracts important
     *features and encapsulates information of the algorithm. At the same time,
     *it is highly adaptable.
     *  \details   An algorithm gets assigned three things. First is a
     *statistical method, which dictates an approximation function and takes
     *care of recomputing
     *					the approximation function. Second is a
     *recomputation
     *heuristic,
     *which decides, when to recompute an algorithm. Third is a statistic, which
     *stores past
     * 				observed execution tiems ofthe algorithm
     *together
     *with
     *an
     *abstraction of the input data set in a measurement pair list. This list is
     *implemented by the class
     *					stemod::core::AlgorithmStatistics. The
     *algorithm
     *has
     *to
     *update
     *it's statistics every time, a new observation (an instance of class
     *					stemod::core::MeasurementPair) is added.
     *Hence,
     *the
     *Algorithm
     *class provides the basis for the decision component, which is done by an
     * 				optimization_criterion, which is implemented by
     *drived
     *classes
     *of
     *stemod::core::OptimizationCriterion.
     *
     *					Note that the statistical method and the
     *recomputation
     *statistic
     *can be exchanged at run-time, because the Algortihm uses the pointer to
     *             implementation technique (or pimpl-idiom).
     *  \author    Sebastian Breß
     *  \version   0.1
     *  \date      2012
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     *http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    class Algorithm {
     public:
      /*! \brief Initializes Algorithm according to Parameters {@link
       * name_of_algorithm}, {@link name_of_statistical_method}, {@link
       * name_of_recomputation_strategy}.
       * \param name_of_algorithm name of the algorithm, it has to be unique!
       * \param name_of_statistical_method the name of a supported statistical
       * method
       * \param name_of_recomputation_strategy name of the recomputation
       * strategy
       */
      Algorithm(const std::string& name_of_algorithm,
                const std::string& name_of_statistical_method,
                const std::string& name_of_recomputation_strategy,
                Operation& operation, DeviceSpecification comp_dev);

      /*! \brief Destroys the Algorithm and writes the statistics to disc file
       */
      ~Algorithm();

      bool setStatisticalMethod(
          boost::shared_ptr<StatisticalMethod_Internal> ptr_statistical_method);

      bool setRecomputationHeuristic(
          boost::shared_ptr<RecomputationHeuristic_Internal>
              ptr_recomp_heuristic);

      const std::string getName() const;

      const EstimatedTime getEstimatedExecutionTime(const Tuple& input_values);
      double getEstimatedRequiredMemoryCapacity(const Tuple& input_values);

      unsigned int getNumberOfDecisionsforThisAlgorithm() const throw();
      unsigned int getNumberOfTerminatedExecutions() const throw();
      /*! \brief returns the total time this algorithm spend in execution*/
      double getTotalExecutionTime() const throw();

      bool addMeasurementPair(const MeasurementPair& mp);

      AlgorithmStatistics& getAlgorithmStatistics() { return *statistics_; }

      std::string toString(unsigned int indent = 0) const;
      //                        {
      //                            std::stringstream ss;
      //                            ss << "Algorithm: " << this->name_ << "
      //                            (Operation: " << this->operation_.getName()
      //                            << ")" << std::endl;
      //                            ss << "StatisticalMethod: " <<
      //                            this->ptr_statistical_method_->
      //
      //                        }

      /**
       * @brief
       * @return
       */
      bool inTrainingPhase() const throw();
      /**
       * @brief
       * @return
       */
      bool inRetrainingPhase() const throw();
      uint64_t getTimeOfLastExecution() const throw();
      void setTimeOfLastExecution(uint64_t new_timestamp) throw();
      void incrementNumberofDecisionsforThisAlgorithm() throw();

      void retrain();
      const LoadChangeEstimator& getLoadChangeEstimator() const throw();
      const DeviceSpecification getDeviceSpecification() const throw();
      void setMemoryCostModel(MemoryCostModelFuncPtr mem_cost_model_func_ptr);

     private:
      std::string name_;
      boost::shared_ptr<StatisticalMethod_Internal> ptr_statistical_method_;
      boost::shared_ptr<RecomputationHeuristic_Internal>
          ptr_recomputation_heristic_;
      AlgorithmStatisticsPtr statistics_;
      Operation& operation_;
      uint64_t logical_timestamp_of_last_execution_;
      bool is_in_retraining_phase_;
      unsigned int retraining_length_;
      LoadChangeEstimator load_change_estimator_;
      DeviceSpecification comp_dev_;
      MemoryCostModelFuncPtr mem_cost_model_func_ptr_;
    };

    typedef boost::shared_ptr<Algorithm> AlgorithmPtr;

  }  // end namespace core
}  // end namespace hype
