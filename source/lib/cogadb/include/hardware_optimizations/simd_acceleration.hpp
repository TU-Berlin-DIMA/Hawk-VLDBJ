#ifndef SIMD_ACCELERATION_HPP_
#define SIMD_ACCELERATION_HPP_

#include <core/column.hpp>
#include <core/global_definitions.hpp>

namespace CoGaDB {

#ifdef ENABLE_SIMD_ACCELERATION
  void simd_selection_int_thread(int* array, unsigned int array_size,
                                 int comparison_value,
                                 const ValueComparator comp,
                                 unsigned int* result_array,
                                 unsigned int* result_array_size,
                                 unsigned int result_tid_offset = 0);

  void simd_selection_float_thread(float* array, unsigned int array_size,
                                   float comparison_value,
                                   const ValueComparator comp,
                                   unsigned int* result_array,
                                   unsigned int* result_array_size,
                                   unsigned int result_tid_offset = 0);
  /* \brief  scans an integer column with SIMD isntructions
   * \details result_tid_offset is a fixed value which is added to all matching
   *          tid values, allowing to use the SIMD Scan in Multithreaded
   * algorithms
  */
  const PositionListPtr simd_selection_int(int* array, unsigned int array_size,
                                           int comparison_value,
                                           const ValueComparator comp,
                                           unsigned int result_tid_offset = 0);
  /* \brief  scans a float column with SIMD isntructions
   * \details result_tid_offset is a fixed value which is added to all matching
   *          tid values, allowing to use the SIMD Scan in Multithreaded
   * algorithms
  */
  const PositionListPtr simd_selection_float(
      float* array, unsigned int array_size, float comparison_value,
      const ValueComparator comp, unsigned int result_tid_offset = 0);

  void bf_simd_selection_int_thread(int* array, unsigned int array_size,
                                    int comparison_value,
                                    const ValueComparator comp,
                                    unsigned int* result_array,
                                    unsigned int* result_array_size,
                                    unsigned int result_tid_offset = 0);

  void bf_simd_selection_float_thread(float* array, unsigned int array_size,
                                      float comparison_value,
                                      const ValueComparator comp,
                                      unsigned int* result_array,
                                      unsigned int* result_array_size,
                                      unsigned int result_tid_offset = 0);
  /* \brief  scans an integer column with SIMD isntructions
   * \details result_tid_offset is a fixed value which is added to all matching
   *          tid values, allowing to use the SIMD Scan in Multithreaded
   * algorithms
   */
  const PositionListPtr bf_simd_selection_int(
      int* array, unsigned int array_size, int comparison_value,
      const ValueComparator comp, unsigned int result_tid_offset = 0);
  /* \brief  scans a float column with SIMD isntructions
   * \details result_tid_offset is a fixed value which is added to all matching
   *          tid values, allowing to use the SIMD Scan in Multithreaded
   * algorithms
   */
  const PositionListPtr bf_simd_selection_float(
      float* array, unsigned int array_size, float comparison_value,
      const ValueComparator comp, unsigned int result_tid_offset = 0);

  namespace CDK {
    namespace selection {
      namespace variants {

        void unrolled_simd_selection_int_thread(
            int* array, unsigned int array_size, int comparison_value,
            const ValueComparator comp, unsigned int* result_array,
            unsigned int* result_array_size, unsigned int result_tid_offset);
        void unrolled_simd_selection_float_thread(
            float* array, unsigned int array_size, float comparison_value,
            const ValueComparator comp, unsigned int* result_array,
            unsigned int* result_array_size, unsigned int result_tid_offset);

        void bf_unrolled_simd_selection_int_thread(
            int* array, unsigned int array_size, int comparison_value,
            const ValueComparator comp, unsigned int* result_array,
            unsigned int* result_array_size, unsigned int result_tid_offset);
        void bf_unrolled_simd_selection_float_thread(
            float* array, unsigned int array_size, float comparison_value,
            const ValueComparator comp, unsigned int* result_array,
            unsigned int* result_array_size, unsigned int result_tid_offset);
      };
    };
  };

#endif

}  // end namespace CoGaDB

#endif  //#ifndef SIMD_ACCELERATION_HPP_
