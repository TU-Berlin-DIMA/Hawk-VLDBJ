/*
 * File:   column_statistics.hpp
 * Author: sebastian
 *
 * Created on 20. November 2014, 08:45
 */

#ifndef COLUMN_STATISTICS_HPP
#define COLUMN_STATISTICS_HPP

#include <boost/any.hpp>
#include <boost/shared_ptr.hpp>

namespace CoGaDB {

  class SelectivityEstimator;
  typedef boost::shared_ptr<SelectivityEstimator> SelectivityEstimatorPtr;

  struct ColumnStatistics {
    ColumnStatistics();

    virtual ~ColumnStatistics();

    virtual bool computeStatistics(const boost::any& array,
                                   size_t array_size) = 0;
    virtual const std::string toString() const = 0;

    bool is_sorted_ascending_;
    bool is_sorted_descending_;
    bool is_dense_value_array_starting_with_zero_;
    bool statistics_up_to_date_;
    SelectivityEstimatorPtr sel_est_;
  };

  template <typename T>
  struct ExtendedColumnStatistics : public ColumnStatistics {
    ExtendedColumnStatistics();

    bool computeStatistics(const boost::any& array, size_t array_size);
    bool computeStatistics(const T* array, size_t array_size);
    virtual const std::string toString() const;

    bool load(const std::string& path_to_column);
    bool store(const std::string& path_to_column);

    T min;
    T max;
    T average;
    double standard_deviation;
  };

}  // end namespace CoGaDB

#endif /* COLUMN_STATISTICS_HPP */
