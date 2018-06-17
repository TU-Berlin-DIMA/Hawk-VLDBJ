
#pragma once

#include <config/configuration.hpp>
#include <config/global_definitions.hpp>
#include <core/time_measurement.hpp>
#include <string>

namespace hype {
  namespace core {

    // ProcessingDeviceMemoryID getMemory(ProcessingDeviceID );
    // std::vector<ProcessingDeviceID>
    // getProcessingDevices(ProcessingDeviceMemoryID);

    /*!
     *  \brief     An AlgorithmSpecification specifies all relevant information
     * about an algorithm,
     * such as the algorithm's name or the name of the operation the algorithms
     * belongs to.
     *  \author    Sebastian Breß
     *  \version   0.2
     *  \date      2013
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     * http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    class HYPE_EXPORT AlgorithmSpecification {
     public:
      /*! \brief constructs an AlgorithmSpecification object by assigning
       * necessary informations to all fields of the object
       * \param alg_name name of the algorithm
       * \param op_name name of the operation the algorithms belongs to
       * \param stat_meth the statistical method used for learning the
       * algorithms behavior (optional)
       * \param recomp_heur the recomputation heuristic used for adapting the
       * algorithms approximation function (optional)
       * \param opt_crit the optimization criterion of the operation the
       * algorithm belongs to (optional)
       */
      AlgorithmSpecification(const std::string& alg_name,
                             const std::string& op_name,
                             StatisticalMethod stat_meth = Least_Squares_1D,
                             RecomputationHeuristic recomp_heur = Periodic,
                             OptimizationCriterion opt_crit =
                                 Runtime_Configuration::instance()
                                     .getDefaultOptimizationCriterion());
      /*!
            *  \brief returns the algorithm's name
            */
      const std::string& getAlgorithmName() const throw();
      /*!
            *  \brief returns the name of the operation the algorithm belongs to
            */
      const std::string& getOperationName() const throw();
      /*!
            *  \brief returns the name of the statistical method that is used
       * for the algorithm
            */
      const std::string getStatisticalMethodName() const throw();
      /*!
            *  \brief returns the name of the recomputation heuristic that is
       * used for the algorithm
            */
      const std::string getRecomputationHeuristicName() const throw();
      /*!
            *  \brief returns the name of the optimization criterion of the
       * operation the algorithm belongs to
            */
      const std::string getOptimizationCriterionName() const throw();

     private:
      /*!  \brief the algorithm's name*/
      std::string alg_name_;
      /*!  \brief name of the operation the algorithm belongs to*/
      std::string op_name_;
      /*!  \brief the statistical method that is used for the algorithm*/
      StatisticalMethod stat_meth_;
      /*!  \brief the recomputation heuristic that is used for the algorithm*/
      RecomputationHeuristic recomp_heur_;
      /*!  \brief the optimization criterion of the operation the algorithm
       * belongs to*/
      OptimizationCriterion opt_crit_;
    };

    /*!
     *  \brief    A OperatorSpecification defines the operator that the user
     * wants to execute.
     *  \details  Hence, it contains the name of the operation and the features
     * of the input data set as well as the
     * 			  two memory ids, where the first identifies where the
     * input
     * data
     * is
     * stored,
     * 			  and the second specifies where the input data is
     * stored.
     * Note
     * that
     * HyPE needs
     * 			  this information to estimate the overhead of data
     * transfers
     * in
     * case
     * the data needs
     * 			  to be copied to use a certain processing device.
     *  \author    Sebastian Breß
     *  \version   0.2
     *  \date      2013
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     * http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    struct HYPE_EXPORT OperatorSpecification {
     public:
      /*! \brief constructs an OperatorSpecification object by assigning
       * necessary informations to all fields of the object
       * \param operator_name the operations's name
       * \param feature_vector the feature vector of this operator
       * \param location_of_input_data the memory id where the input data is
       * stored
       * \param location_for_output_data the memory id where the output data is
       * stored
       */
      OperatorSpecification(const std::string& operator_name,
                            const Tuple& feature_vector,
                            ProcessingDeviceMemoryID location_of_input_data,
                            ProcessingDeviceMemoryID location_for_output_data);
      /*!
            *  \brief returns the operations's name
            */
      const std::string& getOperatorName() const throw();
      /*!
            *  \brief returns the feature vector of this operator
            */
      const Tuple& getFeatureVector() const throw();
      /*!
            *  \brief returns the memory id where the input data is stored
            */
      ProcessingDeviceMemoryID getMemoryLocation() const throw();

     private:
      /*! \brief the operations's name*/
      std::string operator_name_;
      /*! \brief the feature vector of this  operator*/
      Tuple feature_vector_;
      /*! \brief the memory id where the input data is stored*/
      ProcessingDeviceMemoryID location_of_input_data_;
      /*! \brief the memory id where the output data is stored*/
      ProcessingDeviceMemoryID location_for_output_data_;
    };

    typedef size_t (*AvailableMemoryFuncPtr)();

    // member for each processing device
    /*!
     *  \brief    A DeviceSpecification defines a processing device that is
     * available for performing computations.
     *  \details  It consists of a ProcessingDeviceID, which has to be unique, a
     * processing device type (e.g., CPU or GPU)
     * 			  and a memory id, which specifies the memory that the
     * processing
     * devices uses. By convention, the host's CPU has
     * 			  the processing device id of 0, is a processing device
     * from
     * type
     * CPU
     * and the CPU's main memory has memory id 0.
     *  \author    Sebastian Breß
     *  \version   0.2
     *  \date      2013
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     * http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    struct HYPE_EXPORT DeviceSpecification {
     public:
      /*! \brief constructs an DeviceSpecification object by assigning necessary
       * informations to all fields of the object
       * \param pd the unique id of the processing device
       * \param pd_t type of the processing device (e.g., CPU or GPU)
       * \param pd_m unique id of the memory the processing device uses
       */
      DeviceSpecification(
          ProcessingDeviceID pd, ProcessingDeviceType pd_t,
          ProcessingDeviceMemoryID pd_m,
          AvailableMemoryFuncPtr get_avail_memory_func_ptr = NULL,
          size_t total_memory_capacity_in_byte = 0);
      /*!
            *  \brief returns the processing device's ProcessingDeviceID
            */
      ProcessingDeviceID getProcessingDeviceID() const throw();
      /*!
            *  \brief returns the processing device's device type
            */
      ProcessingDeviceType getDeviceType() const throw();
      /*!
            *  \brief returns the processing device's memory id
            */
      ProcessingDeviceMemoryID getMemoryID() const throw();

      size_t getAvailableMemoryCapacity() const;

      size_t getTotalMemoryCapacity() const;

      /*!
            *  \brief implicit conversion to an object of type
       * ProcessingDeviceID
            */
      operator ProcessingDeviceID();
      /*!
            *  \brief implicit conversion to an object of type
       * ProcessingDeviceType
            */
      operator ProcessingDeviceType();
      /*!
            *  \brief implicit conversion to an object of type
       * ProcessingDeviceMemoryID
            */
      operator ProcessingDeviceMemoryID();
      /*!
            *  \brief overload of operator== for this class
            */
      bool operator==(const DeviceSpecification&) const;

     private:
      /*! \brief the processing device's ProcessingDeviceID*/
      ProcessingDeviceID pd_;
      /*! \brief the processing device's device type*/
      ProcessingDeviceType pd_t_;
      /*! \brief the processing device's memory id*/
      ProcessingDeviceMemoryID pd_m_;
      /*! \brief function pointer that returns the available memory of the
       * processing device, ignored if NULL */
      AvailableMemoryFuncPtr get_avail_memory_func_ptr_;
      /*! \brief total memory of the processing device in bytes */
      size_t total_memory_capacity_in_byte_;
    };
    /*!
     *  \brief     A DeviceConstraint restricts the type of processing device,
     * which HyPE may choose to process an operator.
     *  \details   This is especially important if an algorithms does not
     * support a certain data type on a certain processing device
     *  (e.g., no filter operations on an array of strings on the GPU). On
     * default construction, no constraint is defined.
     *  \author    Sebastian Breß
     *  \version   0.2
     *  \date      2013
     *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
     * http://www.gnu.org/licenses/lgpl-3.0.txt
     */
    struct HYPE_EXPORT DeviceConstraint {
     public:
      /*! \brief constructs an DeviceConstraint object by assigning necessary
       * informations to all fields of the object
       * \param dev_constr a device type constraint (e.g., CPU_ONLY or
       * ANY_DEVICE for now restriction)
       * \param pd_mem_constr memory id, where the data should be stored when
       * processed (experimental)
       */
      DeviceConstraint(DeviceTypeConstraint dev_constr = ANY_DEVICE,
                       ProcessingDeviceMemoryID pd_mem_constr = PD_Memory_0);

      /*!
            *  \brief returns the DeviceTypeConstraint
            */
      DeviceTypeConstraint getDeviceTypeConstraint() const;

      /*!
            *  \brief implicit conversion to an object of type
       * DeviceTypeConstraint
            *  \details non-const version
            */
      operator DeviceTypeConstraint();
      /*!
            *  \brief implicit conversion to an object of type
       * ProcessingDeviceMemoryID
            *  \details non-const version
            */
      operator ProcessingDeviceMemoryID();
      /*!
            *  \brief implicit conversion to an object of type
       * DeviceTypeConstraint
            *  \details const version
            */
      operator DeviceTypeConstraint() const;
      /*!
            *  \brief implicit conversion to an object of type
       * ProcessingDeviceMemoryID
            *  \details const version
            */
      operator ProcessingDeviceMemoryID() const;

     private:
      /*! \brief the device type constraint*/
      DeviceTypeConstraint dev_constr_;
      /*! \brief the memory id of the memory, where the data should be
       * processed*/
      ProcessingDeviceMemoryID
          pd_mem_constr_;  // restrict to devices that use a certain memory
    };

  }  // end namespace core
}  // end namespace hype
