/*
 * File:   histogram.hpp
 * Author: sebastian
 *
 * Created on 20. November 2014, 10:34
 */

#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <core/column_base_typed.hpp>

#include <statistics/selectivity_estimator.hpp>

namespace CoGaDB {

  template <typename T>
  class EquiHeightHistogram : public SelectivityEstimator {
    typedef std::pair<T, T> Bucket;
    typedef std::vector<Bucket> Buckets;

   public:
    EquiHeightHistogram(ColumnBaseTyped<T>* col,
                        unsigned int number_of_buckets = 100);
    EquiHeightHistogram(const T* array, size_t array_size,
                        unsigned int number_of_buckets = 100);

    size_t countNumberOfExpectedMatches(T value, ValueComparator comp) const;

    virtual double getEstimatedSelectivity(const Predicate& pred) const;

    virtual double getEstimatedSelectivity(const Predicate& lower_pred,
                                           const Predicate& upper_pred) const;
    virtual std::string getName() const;
    virtual SelectivityEstimationStrategy getSelectivityEstimationStrategy()
        const;
    virtual std::string toString() const;

   private:
    Buckets buckets_;
    std::vector<size_t> number_of_distinct_keys_per_bucket_;
    size_t number_of_values_per_bucket_;
    size_t number_of_values_in_column_;
  };

}  // end namespace CoGaDB

#endif /* HISTOGRAM_HPP */
