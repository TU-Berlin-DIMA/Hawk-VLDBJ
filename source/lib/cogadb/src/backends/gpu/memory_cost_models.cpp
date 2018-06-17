
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <backends/gpu/memory_cost_models.hpp>
#include <persistence/storage_manager.hpp>

namespace CoGaDB {
namespace gpu {

size_t GPU_Operators_Memory_Cost_Models::columnFetchJoin(
    const hype::Tuple& feature_vector) {
  if (feature_vector.size() != 2) {
    COGADB_FATAL_ERROR("Invalid Feature Vector!", "");
  }
  const double join_selectivity = 1.0;
  // assume that fact (foreign key) table is the larger table
  size_t fact_table_size =
      static_cast<size_t>(std::max(feature_vector[0], feature_vector[1]));
  size_t input_dimension_table =
      static_cast<size_t>(std::min(feature_vector[0], feature_vector[1]));
  size_t size_join_index = fact_table_size * 2;
  // flag (counter) array has size of char
  size_t size_of_counter_array = fact_table_size * 0.25;
  size_t size_of_prefix_sum = fact_table_size;
  // result is a TID list qualifying matching rows of the fact table
  // hence, the maximal result size is the number of ros in the fact table
  size_t size_of_maximal_result = fact_table_size;
  size_t result = sizeof(TID) * (input_dimension_table + size_join_index +
                                 size_of_counter_array + size_of_prefix_sum +
                                 join_selectivity * size_of_maximal_result);
  return result;
}

size_t GPU_Operators_Memory_Cost_Models::tableFetchJoin(
    const hype::Tuple& feature_vector) {
  if (feature_vector.size() != 2) {
    COGADB_FATAL_ERROR("Invalid Feature Vector!", "");
  }
  const double join_selectivity = 1.0;
  // assume that fact (foreign key) table is the larger table
  size_t fact_table_size =
      static_cast<size_t>(std::max(feature_vector[0], feature_vector[1]));
  size_t input_dimension_table =
      static_cast<size_t>(std::min(feature_vector[0], feature_vector[1]));
  size_t size_join_index = fact_table_size * 2;
  // flag (counter) array has size of char
  size_t size_of_counter_array = fact_table_size * 0.25;
  size_t size_of_prefix_sum = fact_table_size;
  // result is a TID list qualifying matching rows of the fact table
  // hence, the maximal result size is the number of ros in the fact table
  size_t size_of_maximal_result = fact_table_size;
  size_t result = sizeof(TID) * (input_dimension_table + size_join_index +
                                 size_of_counter_array + size_of_prefix_sum +
                                 join_selectivity * size_of_maximal_result * 2);
  return result;
}
size_t getMaximalTableSize() {
  // workaround: we use bitmaps only after the fetch join phase in the invisible
  // join,
  // so we know that the bitmaps are as large as the rows of the fact table
  // devided by 8
  // furthermore, we assume that the fact table is the largest table
  std::vector<TablePtr>& tables = getGlobalTableList();
  size_t largest_table_size = 0;
  for (unsigned int i = 0; i < tables.size(); ++i) {
    if (largest_table_size < tables[i]->getNumberofRows()) {
      largest_table_size = tables[i]->getNumberofRows();
    }
  }
  return largest_table_size;
}

size_t GPU_Operators_Memory_Cost_Models::positionlistToBitmap(
    const hype::Tuple& feature_vector) {
  size_t result = 0;

  size_t size_of_position_list = feature_vector[0];
  // result is a TID list qualifying matching rows of the fact table
  // hence, the maximal result size is the number of ros in the fact table

  // workaround: we use bitmaps only after the fetch join phase in the invisible
  // join,
  // so we know that the bitmaps are as large as the rows of the fact table
  // devided by 8
  // furthermore, we assume that the fact table is the largest table
  // std::vector<TablePtr>& tables = getGlobalTableList();
  size_t largest_table_size = getMaximalTableSize();

  size_t flag_array = largest_table_size;  //(bytes)
  size_t size_of_maximal_result =
      largest_table_size / 8;  // size of bitmap in bytes

  result = size_of_position_list + flag_array + size_of_maximal_result;

  return result;
}
size_t GPU_Operators_Memory_Cost_Models::bitmapToPositionList(
    const hype::Tuple& feature_vector) {
  // assume bitmap is as large as fact table
  size_t largest_table_size = getMaximalTableSize();
  size_t size_of_input_bitmap = (largest_table_size + 7) / 8;  // bytes
  size_t result = 0;
  // flag (counter) array has size of char
  size_t size_of_counter_array =
      largest_table_size;  // one byte for bit in flag array
  size_t size_of_prefix_sum =
      largest_table_size * sizeof(unsigned int);  // bytes
  // result is a TID list qualifying matching rows of the fact table
  // hence, the maximal result size is the number of rows in the fact table
  // size_t size_of_maximal_result = largest_table_size;

  // FIXME: We omit the memory footprint of the result tids, since this operator
  // is mainly called
  // at the end of an invisible join, which produces mostly small amounts of
  // TIDs

  result = size_of_input_bitmap + size_of_counter_array + size_of_prefix_sum;

  return result;
}
size_t GPU_Operators_Memory_Cost_Models::bitwiseAND(
    const hype::Tuple& feature_vector) {
  size_t result = 0;
  size_t largest_table_size = getMaximalTableSize();
  size_t size_of_input_bitmap = (largest_table_size + 7) / 8;  // bytes
  result = 3 * size_of_input_bitmap;
  return result;
}
size_t GPU_Operators_Memory_Cost_Models::columnSelection(
    const hype::Tuple& feature_vector) {
  size_t result = 0;

  size_t input_size = feature_vector[0];
  size_t size_of_counter_array = input_size * 0.25;
  size_t size_of_prefix_sum = input_size;
  // optimize worst case estimation error
  size_t output_size = 0.5 * input_size;

  result =
      (input_size + size_of_counter_array + size_of_prefix_sum + output_size) *
      sizeof(TID);

  return result;
}

}  // end namespace gpu
}  // end namespace CogaDB