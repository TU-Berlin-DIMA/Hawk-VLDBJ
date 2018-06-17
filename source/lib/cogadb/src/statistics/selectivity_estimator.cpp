
#include <statistics/histogram.hpp>

namespace CoGaDB {
template <typename T>
SelectivityEstimatorPtr SelectivityEstimatorFactory<T>::create(
    const SelectivityEstimationStrategy& selectivity_est, const T* array,
    size_t array_size) {
  SelectivityEstimatorPtr sel_est;
  std::vector<T> array_copy(array, array + array_size);
  std::sort(array_copy.begin(), array_copy.end());
  if (selectivity_est == EQUI_HEIGHT_HISTOGRAM) {
    sel_est = SelectivityEstimatorPtr(
        new EquiHeightHistogram<T>(array_copy.data(), array_size));
  } else {
    COGADB_FATAL_ERROR(
        "Unkown selectivity estimation strategy: " << (int)selectivity_est, "");
  }
  return sel_est;
}
COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(
    SelectivityEstimatorFactory)
}  // end namespace CoGaDB
