

#include <algorithm>
#include <functional>
#include <numeric>
#include <sstream>

#include <boost/filesystem.hpp>
#include <core/global_definitions.hpp>
#include <statistics/column_statistics.hpp>
#include <statistics/histogram.hpp>
#include <statistics/selectivity_estimator.hpp>

#pragma GCC diagnostic ignored "-Wunused-variable"

// namespace boost {
// namespace serialization {
//
//    template<class Archive>
//    inline void serialize(
//        Archive & ar,
//        CoGaDB::SelectivityEstimationStrategy t,
//        const unsigned int file_version
//    ){
////        ar & (int) t;
//    }
//
//} // namespace serialization
//} // namespace boost

namespace CoGaDB {

//    COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(ExtendedColumnStatistics);

using namespace std;

/* \brief checks for dense value array startig with 0, such as 0, 1, 2, 3, ...,
 * N-1, N*/
template <typename T>
bool is_dense_value_array_starting_with_zero(T* array, size_t array_size) {
  if (!array) return false;
  if (array_size == 0) return false;
  if (array[0] != 0) return false;

  for (size_t i = 0; i < array_size - 1; ++i) {
    if (array[i + 1] + 1 != array[i]) return false;
  }

  return true;
}

ColumnStatistics::ColumnStatistics()
    : is_sorted_ascending_(false),
      is_sorted_descending_(false),
      is_dense_value_array_starting_with_zero_(false),
      statistics_up_to_date_(false),
      sel_est_() {}

ColumnStatistics::~ColumnStatistics() {}

template <typename T>
ExtendedColumnStatistics<T>::ExtendedColumnStatistics()
    : ColumnStatistics(), min(), max(), average(), standard_deviation() {}

template <typename T>
bool ExtendedColumnStatistics<T>::computeStatistics(const boost::any& array,
                                                    size_t array_size) {
  if (array.empty()) return false;
  if (array.type() != typeid(T*)) return false;
  return computeStatistics(boost::any_cast<T*>(array), array_size);
}

template <typename T>
bool ExtendedColumnStatistics<T>::computeStatistics(const T* array,
                                                    size_t array_size) {
  if (!array) return false;
  if (array_size == 0) return false;

  is_sorted_ascending_ = ::std::is_sorted(array, array + array_size, less<T>());
  is_sorted_descending_ =
      ::std::is_sorted(array, array + array_size, greater<T>());
  is_dense_value_array_starting_with_zero_ =
      is_dense_value_array_starting_with_zero(array, array_size);

  min = *std::min_element(array, array + array_size);
  max = *std::max_element(array, array + array_size);
  average = std::accumulate(array, array + array_size, T()) / array_size;

  double variance = 0;
  for (unsigned int i = 0; i < array_size; ++i) {
    variance += ((array[i] - average) * (array[i] - average));
  }
  standard_deviation = sqrt(variance);
  statistics_up_to_date_ = true;

  sel_est_ = SelectivityEstimator::createSelectivityEstimator(
      EQUI_HEIGHT_HISTOGRAM, array, array_size);
  return true;
}

template <typename T>
const std::string ExtendedColumnStatistics<T>::toString() const {
  std::stringstream ss;
  ss << "Dense Value Column Starting with zero: "
     << is_dense_value_array_starting_with_zero_ << endl;
  ss << "Sorted Ascending: " << is_sorted_ascending_ << endl;
  ss << "Sorted Descending: " << is_sorted_descending_ << endl;
  ss << "Min Value: " << min << endl;
  ss << "Max Value: " << max << endl;
  ss << "Average Value: " << average << endl;
  ss << "Standard Deviation Value: " << standard_deviation << endl;
  ss << "Statistics up to date: " << this->statistics_up_to_date_ << endl;
  if (sel_est_) ss << sel_est_->toString() << endl;
  return ss.str();
}

template <>
const std::string ExtendedColumnStatistics<std::string>::toString() const {
  std::stringstream ss;
  ss << "Sorted Ascending: " << is_sorted_ascending_ << endl;
  ss << "Sorted Descending: " << is_sorted_descending_ << endl;
  ss << "Min Value: " << min << endl;
  ss << "Max Value: " << max << endl;
  ss << "Statistics up to date: " << this->statistics_up_to_date_ << endl;
  if (sel_est_) ss << sel_est_->toString() << endl;
  return ss.str();
}

// in case of string columns, we can only provide min and max values
template <>
bool ExtendedColumnStatistics<char*>::computeStatistics(const C_String* array,
                                                        size_t array_size) {
  COGADB_FATAL_ERROR("Called unimplemented function!", "");
  return false;
}

// in case of string columns, we can onkly provide min and max values
template <>
bool ExtendedColumnStatistics<std::string>::computeStatistics(
    const std::string* array, size_t array_size) {
  if (!array) return false;
  if (array_size == 0) return false;

  min = *std::min_element(array, array + array_size);
  max = *std::max_element(array, array + array_size);
  statistics_up_to_date_ = true;

  sel_est_ = SelectivityEstimatorPtr(
      new EquiHeightHistogram<std::string>(array, array_size));
  return true;
}

template <>
const std::string ExtendedColumnStatistics<char*>::toString() const {
  std::stringstream ss;
  ss << "Sorted Ascending: " << is_sorted_ascending_ << endl;
  ss << "Sorted Descending: " << is_sorted_descending_ << endl;
  ss << "Min Value: " << min << endl;
  ss << "Max Value: " << max << endl;
  ss << "Statistics up to date: " << this->statistics_up_to_date_ << endl;
  if (sel_est_) ss << sel_est_->toString() << endl;
  return ss.str();
}

template <typename T>
bool ExtendedColumnStatistics<T>::load(const std::string& path_to_column) {
  std::string path_to_statistics(path_to_column);
  path_to_statistics += ".statistics";
  std::ifstream infile(path_to_statistics.c_str(),
                       std::ios_base::binary | std::ios_base::in);
  if (!infile.good()) return false;
  boost::archive::binary_iarchive ia(infile);

  ia >> this->average;
  ia >> this->is_dense_value_array_starting_with_zero_;
  ia >> this->is_sorted_ascending_;
  ia >> this->is_sorted_descending_;
  ia >> this->max;
  ia >> this->min;
  ia >> this->standard_deviation;
  ia >> this->statistics_up_to_date_;
  //            ia >> sel_est_strat;
  /* \todo Create Selectivity Estimator and load it's data from disk. */

  infile.close();
  return true;
}

template <typename T>
bool ExtendedColumnStatistics<T>::store(const std::string& path_to_column) {
  std::string path_to_statistics(path_to_column);
  path_to_statistics += ".statistics";

  if (boost::filesystem::exists(path_to_statistics)) {
    bool ret = boost::filesystem::remove(path_to_statistics);
    assert(ret == true);
  }
  std::ofstream outfile(path_to_statistics.c_str(), std::ios_base::binary |
                                                        std::ios_base::out |
                                                        std::ios_base::trunc);
  if (!outfile.good()) return false;
  boost::archive::binary_oarchive oa(outfile);
  oa << this->average;
  oa << this->is_dense_value_array_starting_with_zero_;
  oa << this->is_sorted_ascending_;
  oa << this->is_sorted_descending_;
  oa << this->max;
  oa << this->min;
  oa << this->standard_deviation;
  oa << this->statistics_up_to_date_;
  outfile.flush();
  outfile.close();
  return true;
}

template <>
bool ExtendedColumnStatistics<char*>::load(const std::string& path_to_column) {
  COGADB_WARNING(
      "Called unimplemented function! Will not load Column statistics for "
      "Column<char*>!",
      "");
  return false;
}

template <>
bool ExtendedColumnStatistics<char*>::store(const std::string& path_to_column) {
  COGADB_WARNING(
      "Called unimplemented function! Will not store Column statistics for "
      "Column<char*>",
      "");
  return false;
}

COGADB_INSTANTIATE_STRUCT_TEMPLATE_FOR_SUPPORTED_TYPES(ExtendedColumnStatistics)

}  // end namespace CoGaDB
