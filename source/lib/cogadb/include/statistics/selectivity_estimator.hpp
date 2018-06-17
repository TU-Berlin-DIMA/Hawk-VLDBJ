/*
 * File:   selectivity_estimator.hpp
 * Author: sebastian
 *
 * Created on 20. November 2014, 10:47
 */

#ifndef SELECTIVITY_ESTIMATOR_HPP
#define SELECTIVITY_ESTIMATOR_HPP

#include <core/column_base_typed.hpp>

namespace CoGaDB {

  class SelectivityEstimator;
  typedef boost::shared_ptr<SelectivityEstimator> SelectivityEstimatorPtr;

  class SelectivityEstimator {
   public:
    template <typename T>
    static SelectivityEstimatorPtr createSelectivityEstimator(
        const SelectivityEstimationStrategy& selectivity_est, const T* array,
        size_t array_size);
    /* \brief used for simple predicates, such a x=5, or x<20*/
    virtual double getEstimatedSelectivity(const Predicate& pred) const = 0;
    /* \brief used for ranges, e.g., for x betwen 1 and 10, or x = 6 or x= 10*/
    virtual double getEstimatedSelectivity(
        const Predicate& lower_pred, const Predicate& upper_pred) const = 0;
    virtual std::string getName() const = 0;
    virtual SelectivityEstimationStrategy getSelectivityEstimationStrategy()
        const = 0;
    virtual std::string toString() const = 0;

    virtual ~SelectivityEstimator() {}
  };

  template <typename T>
  class SelectivityEstimatorFactory {
   public:
    static SelectivityEstimatorPtr create(
        const SelectivityEstimationStrategy& selectivity_est, const T* array,
        size_t array_size);
  };

  template <typename T>
  SelectivityEstimatorPtr SelectivityEstimator::createSelectivityEstimator(
      const SelectivityEstimationStrategy& selectivity_est, const T* array,
      size_t array_size) {
    return SelectivityEstimatorFactory<T>::create(selectivity_est, array,
                                                  array_size);
  }

}  // end namespace CoGaDB

#endif /* SELECTIVITY_ESTIMATOR_HPP */
