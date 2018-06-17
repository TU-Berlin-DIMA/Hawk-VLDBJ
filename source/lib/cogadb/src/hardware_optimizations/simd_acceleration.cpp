
#include <core/global_definitions.hpp>
#include <hardware_optimizations/simd_acceleration.hpp>

namespace CoGaDB {

//#define ENABLE_SIMD_ACCELERATION

#ifdef ENABLE_SIMD_ACCELERATION

#include <stdint.h>

// SSE compiler intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif

// For SSE2:
#ifdef __SSE2__
extern "C" {
#include <emmintrin.h>
#include <mmintrin.h>
}
#endif

// For SSE3:
#ifdef __SSE3__
extern "C" {
#include <immintrin.h>  // (Meta-header, for GCC only)
#include <pmmintrin.h>
}
#endif

// For SSE4: (WITHOUT extern "C")
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#define SIMD_DEBUG_MODE false

/**
 * TYPE = C data type (int,float)
 * SIMD_TYPE = corresponding SSE Type (__m128i, __m128 respectively)
 * SIMD_LOAD = SSE load (from array) instruction for the given SIMD_TYPE
 * SIMD_SET = SSE set (mostly comparison value, initialization) instruction for
*the given SIMD_TYPE
 * COMPARISON_OPERATOR = C comparison operator in {<,<=,>,>=,==}
 * SIMD_COMPARISON_FUNCTION = SSE comparison function for the given SIMD_TYPE
*and COMPARISON_OPERATOR
**/
#define COGADB_SIMD_SCAN(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, SIMD_SET, \
                         array, array_size, result_array, comparison_value,    \
                         SIMD_COMPARISON_FUNCTION, COMPARISON_OPERATOR,        \
                         result_size, result_tid_offset)                       \
  SIMD_TYPE* sse_array = reinterpret_cast<SIMD_TYPE*>(array);                  \
  assert(sse_array != NULL);                                                   \
  int alignment_offset = ((intptr_t)sse_array) % sizeof(SIMD_TYPE);            \
  const int sse_array_length =                                                 \
      (array_size - alignment_offset) * sizeof(TYPE) / sizeof(SIMD_TYPE);      \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "SSE Array Length: " << sse_array_length << std::endl;        \
  char* tmp_array = (char*)sse_array;                                          \
  tmp_array += alignment_offset;                                               \
  sse_array = reinterpret_cast<SIMD_TYPE*>(tmp_array);                         \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "array adress: " << (void*)array                              \
              << "sse array: " << (void*)sse_array << std::endl;               \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "First SSE Array Element: " << ((int*)sse_array)[0]           \
              << std::endl;                                                    \
  unsigned int pos = 0;                                                        \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "alignment_offset " << alignment_offset << std::endl;         \
  if (alignment_offset != 0) {                                                 \
    if (SIMD_DEBUG_MODE)                                                       \
      std::cout << "process first unaligned data chunk: index 0 to "           \
                << alignment_offset << std::endl;                              \
    for (unsigned int i = 0; i < alignment_offset / sizeof(TYPE); i++) {       \
      if (SIMD_DEBUG_MODE) {                                                   \
        std::cout << "index " << i << std::endl;                               \
        std::cout << "value " << array[i] << " match:"                         \
                  << (array[i] COMPARISON_OPERATOR comparison_value)           \
                  << std::endl;                                                \
      }                                                                        \
      if (array[i] COMPARISON_OPERATOR comparison_value) {                     \
        result_array[pos++] = i + result_tid_offset;                           \
      }                                                                        \
    }                                                                          \
  }                                                                            \
  SIMD_TYPE comp_val = SIMD_SET(comparison_value);                             \
  SIMD_TYPE read_value = SIMD_SET(0);                                          \
  SIMD_TYPE comp_result = SIMD_SET(0);                                         \
  SIMD_TYPE result_tids;                                                       \
  int mask;                                                                    \
  int count = 0;                                                               \
  int j;                                                                       \
  /*char* simd_res_array = (char*) (result_array+pos);*/                       \
  unsigned int basetid;                                                        \
  unsigned int offsets = result_tid_offset + (alignment_offset / sizeof(int)); \
  for (unsigned int i = 0; i < sse_array_length; i++) {                        \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i,        \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,     \
              mask, basetid, result_array, pos);                               \
  }                                                                            \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "Remaining offsets: "                                         \
              << (sse_array_length * sizeof(SIMD_TYPE) / sizeof(TYPE)) +       \
                     alignment_offset                                          \
              << " to " << array_size << std::endl;                            \
  for (unsigned int i =                                                        \
           (sse_array_length * sizeof(SIMD_TYPE) / sizeof(TYPE)) +             \
           (alignment_offset / sizeof(TYPE));                                  \
       i < array_size; i++) {                                                  \
    if (array[i] COMPARISON_OPERATOR comparison_value) {                       \
      result_array[pos++] = i + result_tid_offset;                             \
    }                                                                          \
  }                                                                            \
  result_size = pos;

// first variant
#define SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, I, \
                  SIMD_COMPARISON_FUNCTION, SIMD_COMP, SIMD_READ, COMP_RES, \
                  MASK, BASE_TID, result_array, pos)                        \
  assert(((intptr_t)sse_array) % sizeof(SIMD_TYPE) == 0);                   \
  SIMD_READ = SIMD_LOAD((SIMD_LOAD_TYPE*)&sse_array[I]);                    \
  if (SIMD_DEBUG_MODE) {                                                    \
    std::cout << "index: " << I << std::endl;                               \
  }                                                                         \
  COMP_RES = SIMD_COMPARISON_FUNCTION(SIMD_READ, SIMD_COMP);                \
  MASK = _mm_movemask_ps((__m128)COMP_RES);                                 \
  BASE_TID = (I) * (sizeof(SIMD_TYPE) / sizeof(TYPE)) + offsets;            \
  if (SIMD_DEBUG_MODE)                                                      \
    std::cout << "Mask: " << std::hex << mask << std::dec << std::endl;     \
  if (MASK) {                                                               \
    if (SIMD_DEBUG_MODE) std::cout << "at least one match!" << std::endl;   \
    for (unsigned j = 0; j < sizeof(SIMD_TYPE) / sizeof(TYPE); ++j) {       \
      if (SIMD_DEBUG_MODE)                                                  \
        std::cout << "sub index: " << j << " value: " << ((mask >> j) & 1)  \
                  << std::endl;                                             \
      if ((MASK >> j) & 1) result_array[pos++] = BASE_TID + j;              \
      if (SIMD_DEBUG_MODE)                                                  \
        std::cout << "base_tid: " << BASE_TID + j << std::endl;             \
    }                                                                       \
  }

#define COGADB_Unrolled_SIMD_SCAN(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, \
                                  SIMD_SET, array, array_size, result_array,  \
                                  comparison_value, SIMD_COMPARISON_FUNCTION, \
                                  COMPARISON_OPERATOR, result_size,           \
                                  result_tid_offset)                          \
  SIMD_TYPE* sse_array = reinterpret_cast<SIMD_TYPE*>(array);                 \
  assert(sse_array != NULL);                                                  \
  int alignment_offset = ((intptr_t)sse_array) % sizeof(SIMD_TYPE);           \
  const int sse_array_length =                                                \
      (array_size - alignment_offset) * sizeof(TYPE) / sizeof(SIMD_TYPE);     \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "SSE Array Length: " << sse_array_length << std::endl;       \
  char* tmp_array = (char*)sse_array;                                         \
  tmp_array += alignment_offset;                                              \
  sse_array = reinterpret_cast<SIMD_TYPE*>(tmp_array);                        \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "array adress: " << (void*)array                             \
              << "sse array: " << (void*)sse_array << std::endl;              \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "First SSE Array Element: " << ((TYPE*)sse_array)[0]         \
              << std::endl;                                                   \
  unsigned int pos = 0;                                                       \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "alignment_offset " << alignment_offset << std::endl;        \
  if (alignment_offset != 0) {                                                \
    if (SIMD_DEBUG_MODE)                                                      \
      std::cout << "process first unaligned data chunk: index 0 to "          \
                << alignment_offset << std::endl;                             \
    for (unsigned int i = 0; i < alignment_offset / sizeof(TYPE); i++) {      \
      if (SIMD_DEBUG_MODE) {                                                  \
        std::cout << "index " << i << std::endl;                              \
        std::cout << "value " << array[i] << " match:"                        \
                  << (array[i] COMPARISON_OPERATOR comparison_value)          \
                  << std::endl;                                               \
      }                                                                       \
      if (array[i] COMPARISON_OPERATOR comparison_value) {                    \
        result_array[pos++] = i + result_tid_offset;                          \
      }                                                                       \
    }                                                                         \
  }                                                                           \
  SIMD_TYPE comp_val = SIMD_SET(comparison_value);                            \
  SIMD_TYPE read_value = SIMD_SET(0);                                         \
  SIMD_TYPE comp_result = SIMD_SET(0);                                        \
  SIMD_TYPE result_tids;                                                      \
  int mask;                                                                   \
  int count = 0;                                                              \
  int j;                                                                      \
  char* simd_res_array = (char*)(result_array + pos);                         \
  unsigned int basetid =                                                      \
      (alignment_offset / sizeof(TYPE)) + result_tid_offset;                  \
  unsigned int i;                                                             \
  unsigned int offsets =                                                      \
      result_tid_offset + (alignment_offset / sizeof(TYPE));                  \
  for (i = 0; i + 7 < sse_array_length; i += 8) {                             \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i,       \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,    \
              mask, basetid, result_array, pos);                              \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 1,   \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,    \
              mask, basetid, result_array, pos);                              \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 2,   \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,    \
              mask, basetid, result_array, pos);                              \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 3,   \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,    \
              mask, basetid, result_array, pos);                              \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 4,   \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,    \
              mask, basetid, result_array, pos);                              \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 5,   \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,    \
              mask, basetid, result_array, pos);                              \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 6,   \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,    \
              mask, basetid, result_array, pos);                              \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 7,   \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,    \
              mask, basetid, result_array, pos);                              \
  }                                                                           \
  for (; i < sse_array_length; i++) {                                         \
    SIMD_LOOP(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i,       \
              SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,    \
              mask, basetid, result_array, pos);                              \
  }                                                                           \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "Remaining offsets: "                                        \
              << (sse_array_length * sizeof(SIMD_TYPE) / sizeof(TYPE)) +      \
                     alignment_offset                                         \
              << " to " << array_size << std::endl;                           \
  for (i = (sse_array_length * sizeof(SIMD_TYPE) / sizeof(TYPE)) +            \
           (alignment_offset / sizeof(TYPE));                                 \
       i < array_size; i++) {                                                 \
    if (array[i] COMPARISON_OPERATOR comparison_value) {                      \
      result_array[pos++] = i + result_tid_offset;                            \
    }                                                                         \
  }                                                                           \
  result_size = pos;

#define COGADB_BF_SIMD_SCAN(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE,       \
                            SIMD_SET, array, array_size, result_array,        \
                            comparison_value, SIMD_COMPARISON_FUNCTION,       \
                            COMPARISON_OPERATOR, result_size,                 \
                            result_tid_offset)                                \
  SIMD_TYPE* sse_array = reinterpret_cast<SIMD_TYPE*>(array);                 \
  assert(sse_array != NULL);                                                  \
  int alignment_offset = ((intptr_t)sse_array) % sizeof(SIMD_TYPE);           \
  const int sse_array_length =                                                \
      (array_size - alignment_offset) * sizeof(TYPE) / sizeof(SIMD_TYPE);     \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "SSE Array Length: " << sse_array_length << std::endl;       \
  char* tmp_array = (char*)sse_array;                                         \
  tmp_array += alignment_offset;                                              \
  sse_array = reinterpret_cast<SIMD_TYPE*>(tmp_array);                        \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "array adress: " << (void*)array                             \
              << "sse array: " << (void*)sse_array << std::endl;              \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "First SSE Array Element: " << ((TYPE*)sse_array)[0]         \
              << std::endl;                                                   \
  unsigned int pos = 0;                                                       \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "alignment_offset " << alignment_offset << std::endl;        \
  if (alignment_offset != 0) {                                                \
    if (SIMD_DEBUG_MODE)                                                      \
      std::cout << "process first unaligned data chunk: index 0 to "          \
                << alignment_offset << std::endl;                             \
    for (unsigned int i = 0; i < alignment_offset / sizeof(TYPE); i++) {      \
      if (SIMD_DEBUG_MODE) {                                                  \
        std::cout << "index " << i << std::endl;                              \
        std::cout << "value " << array[i]                                     \
                  << " match:" << (array[i] < comparison_value) << std::endl; \
      }                                                                       \
      result_array[pos] = i + result_tid_offset;                              \
      pos += (array[i] COMPARISON_OPERATOR comparison_value);                 \
    }                                                                         \
  }                                                                           \
  SIMD_TYPE comp_val = SIMD_SET(comparison_value);                            \
  SIMD_TYPE read_value = SIMD_SET(0);                                         \
  SIMD_TYPE comp_result = SIMD_SET(0);                                        \
  SIMD_TYPE result_tids;                                                      \
  int mask;                                                                   \
  int count = 0;                                                              \
  char* simd_res_array = (char*)(result_array + pos);                         \
  unsigned int basetid =                                                      \
      (alignment_offset / sizeof(TYPE)) + result_tid_offset;                  \
  for (unsigned int i = 0; i < sse_array_length; i++) {                       \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i,    \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result, \
                 mask, basetid, result_array, pos);                           \
  }                                                                           \
  if (SIMD_DEBUG_MODE)                                                        \
    std::cout << "Remaining offsets: "                                        \
              << (sse_array_length * sizeof(SIMD_TYPE) / sizeof(TYPE)) +      \
                     alignment_offset                                         \
              << " to " << array_size << std::endl;                           \
  for (unsigned int i =                                                       \
           (sse_array_length * sizeof(SIMD_TYPE) / sizeof(TYPE)) +            \
           (alignment_offset / sizeof(TYPE));                                 \
       i < array_size; i++) {                                                 \
    result_array[pos] = i + result_tid_offset;                                \
    pos += (array[i] COMPARISON_OPERATOR comparison_value);                   \
  }                                                                           \
  result_size = pos;

#define SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, I, \
                     SIMD_COMPARISON_FUNCTION, SIMD_COMP, SIMD_READ, COMP_RES, \
                     MASK, BASE_TID, result_array, pos)                        \
  assert(((intptr_t)sse_array) % sizeof(SIMD_TYPE) == 0);                      \
  SIMD_READ = SIMD_LOAD((SIMD_LOAD_TYPE*)&sse_array[I]);                       \
  if (SIMD_DEBUG_MODE) {                                                       \
    std::cout << "index: " << I << std::endl;                                  \
  }                                                                            \
  COMP_RES = SIMD_COMPARISON_FUNCTION(SIMD_READ, SIMD_COMP);                   \
  MASK = _mm_movemask_ps((__m128)COMP_RES);                                    \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "Mask: " << std::hex << mask << std::dec << std::endl;        \
  result_array[pos] = BASE_TID++;                                              \
  pos += ((mask)&1);                                                           \
  result_array[pos] = BASE_TID++;                                              \
  pos += ((mask >> 1) & 1);                                                    \
  result_array[pos] = BASE_TID++;                                              \
  pos += ((mask >> 2) & 1);                                                    \
  result_array[pos] = BASE_TID++;                                              \
  pos += ((mask >> 3) & 1);

#define COGADB_BF_Unrolled_SIMD_SCAN(                                          \
    TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, SIMD_SET, array, array_size,   \
    result_array, comparison_value, SIMD_COMPARISON_FUNCTION,                  \
    COMPARISON_OPERATOR, result_size, result_tid_offset)                       \
  SIMD_TYPE* sse_array = reinterpret_cast<SIMD_TYPE*>(array);                  \
  assert(sse_array != NULL);                                                   \
  int alignment_offset = ((intptr_t)sse_array) % sizeof(SIMD_TYPE);            \
  const int sse_array_length =                                                 \
      (array_size - alignment_offset) * sizeof(TYPE) / sizeof(SIMD_TYPE);      \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "SSE Array Length: " << sse_array_length << std::endl;        \
  char* tmp_array = (char*)sse_array;                                          \
  tmp_array += alignment_offset;                                               \
  sse_array = reinterpret_cast<SIMD_TYPE*>(tmp_array);                         \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "array adress: " << (void*)array                              \
              << "sse array: " << (void*)sse_array << std::endl;               \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "First SSE Array Element: " << ((TYPE*)sse_array)[0]          \
              << std::endl;                                                    \
  unsigned int pos = 0;                                                        \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "alignment_offset " << alignment_offset << std::endl;         \
  if (alignment_offset != 0) {                                                 \
    if (SIMD_DEBUG_MODE)                                                       \
      std::cout << "process first unaligned data chunk: index 0 to "           \
                << alignment_offset << std::endl;                              \
    for (unsigned int i = 0; i < alignment_offset / sizeof(TYPE); i++) {       \
      if (SIMD_DEBUG_MODE) {                                                   \
        std::cout << "index " << i << std::endl;                               \
        std::cout << "value " << array[i] << " match:"                         \
                  << (array[i] COMPARISON_OPERATOR comparison_value)           \
                  << std::endl;                                                \
      }                                                                        \
      result_array[pos] = i + result_tid_offset;                               \
      pos += (array[i] COMPARISON_OPERATOR comparison_value);                  \
    }                                                                          \
  }                                                                            \
  unsigned int offsets =                                                       \
      result_tid_offset + (alignment_offset / sizeof(TYPE));                   \
  SIMD_TYPE comp_val = SIMD_SET(comparison_value);                             \
  SIMD_TYPE read_value = SIMD_SET(0);                                          \
  SIMD_TYPE comp_result = SIMD_SET(0);                                         \
  int mask;                                                                    \
  unsigned int basetid =                                                       \
      (alignment_offset / sizeof(TYPE)) + result_tid_offset;                   \
  unsigned int i;                                                              \
  for (i = 0; i + 7 < sse_array_length; i += 8) {                              \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i,     \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,  \
                 mask, basetid, result_array, pos);                            \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 1, \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,  \
                 mask, basetid, result_array, pos);                            \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 2, \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,  \
                 mask, basetid, result_array, pos);                            \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 3, \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,  \
                 mask, basetid, result_array, pos);                            \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 4, \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,  \
                 mask, basetid, result_array, pos);                            \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 5, \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,  \
                 mask, basetid, result_array, pos);                            \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 6, \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,  \
                 mask, basetid, result_array, pos);                            \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i + 7, \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,  \
                 mask, basetid, result_array, pos);                            \
  }                                                                            \
  for (; i < sse_array_length; i++) {                                          \
    SIMD_LOOP_BF(TYPE, SIMD_TYPE, SIMD_LOAD, SIMD_LOAD_TYPE, sse_array, i,     \
                 SIMD_COMPARISON_FUNCTION, comp_val, read_value, comp_result,  \
                 mask, basetid, result_array, pos);                            \
  }                                                                            \
  if (SIMD_DEBUG_MODE)                                                         \
    std::cout << "Remaining offsets: "                                         \
              << (sse_array_length * sizeof(SIMD_TYPE) / sizeof(TYPE)) +       \
                     alignment_offset                                          \
              << " to " << array_size << std::endl;                            \
  for (i = (sse_array_length * sizeof(SIMD_TYPE) / sizeof(TYPE)) +             \
           (alignment_offset / sizeof(TYPE));                                  \
       i < array_size; i++) {                                                  \
    result_array[pos] = i + result_tid_offset;                                 \
    pos += (array[i] COMPARISON_OPERATOR comparison_value);                    \
  }                                                                            \
  result_size = pos;

#define COGADB_SIMD_SCAN_FLOAT(array, array_size, result_array,              \
                               comparison_value, SIMD_COMPARISON_FUNCTION,   \
                               COMPARISON_OPERATOR, result_size,             \
                               result_tid_offset)                            \
  __m128* sse_array = reinterpret_cast<__m128*>(array);                      \
  assert(sse_array != NULL);                                                 \
  int alignment_offset = ((intptr_t)sse_array) % sizeof(__m128);             \
  const int sse_array_length =                                               \
      (array_size - alignment_offset) * sizeof(int) / sizeof(__m128);        \
  if (SIMD_DEBUG_MODE)                                                       \
    std::cout << "SSE Array Length: " << sse_array_length << std::endl;      \
  char* tmp_array = (char*)sse_array;                                        \
  tmp_array += alignment_offset;                                             \
  sse_array = reinterpret_cast<__m128*>(tmp_array);                          \
  if (SIMD_DEBUG_MODE)                                                       \
    std::cout << "array adress: " << (void*)array                            \
              << "sse array: " << (void*)sse_array << std::endl;             \
  if (SIMD_DEBUG_MODE)                                                       \
    std::cout << "First SSE Array Element: " << ((int*)sse_array)[0]         \
              << std::endl;                                                  \
  unsigned int pos = 0;                                                      \
  __m128 comp_val = _mm_set1_ps(comparison_value);                           \
  __m128 read_value = _mm_set1_ps(0);                                        \
  if (SIMD_DEBUG_MODE)                                                       \
    std::cout << "alignment_offset " << alignment_offset << std::endl;       \
  if (alignment_offset != 0) {                                               \
    if (SIMD_DEBUG_MODE)                                                     \
      std::cout << "process first unaligned data chunk: index 0 to "         \
                << alignment_offset << std::endl;                            \
    for (unsigned int i = 0; i < alignment_offset / sizeof(int); i++) {      \
      if (SIMD_DEBUG_MODE) {                                                 \
        std::cout << "index " << i << std::endl;                             \
        std::cout << "value " << array[i] << " match:"                       \
                  << (array[i] COMPARISON_OPERATOR comparison_value)         \
                  << std::endl;                                              \
      }                                                                      \
      if (array[i] COMPARISON_OPERATOR comparison_value) {                   \
        result_array[pos++] = i + result_tid_offset;                         \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  for (unsigned int i = 0; i < sse_array_length; i++) {                      \
    assert(((intptr_t)sse_array) % sizeof(__m128) == 0);                     \
    read_value = _mm_load_ps((float*)&sse_array[i]);                         \
    if (SIMD_DEBUG_MODE) {                                                   \
      std::cout << "index: " << i << std::endl;                              \
    }                                                                        \
    __m128 comp_result = SIMD_COMPARISON_FUNCTION(read_value, comp_val);     \
    int mask = _mm_movemask_ps(comp_result);                                 \
    if (SIMD_DEBUG_MODE)                                                     \
      std::cout << "Mask: " << std::hex << mask << std::dec << std::endl;    \
    if (mask) {                                                              \
      if (SIMD_DEBUG_MODE) std::cout << "at least one match!" << std::endl;  \
      for (unsigned j = 0; j < sizeof(__m128) / sizeof(int); ++j) {          \
        if (SIMD_DEBUG_MODE)                                                 \
          std::cout << "sub index: " << j << " value: " << ((mask >> j) & 1) \
                    << std::endl;                                            \
        int tmp = ((mask >> j) & 1);                                         \
        result_array[pos] = i * (sizeof(__m128) / sizeof(int)) + j +         \
                            (alignment_offset / sizeof(int)) +               \
                            result_tid_offset;                               \
        pos += tmp;                                                          \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  if (SIMD_DEBUG_MODE)                                                       \
    std::cout << "Remaining offsets: "                                       \
              << (sse_array_length * sizeof(__m128) / sizeof(int)) +         \
                     alignment_offset                                        \
              << " to " << array_size << std::endl;                          \
  for (unsigned int i = (sse_array_length * sizeof(__m128) / sizeof(int)) +  \
                        (alignment_offset / sizeof(int));                    \
       i < array_size; i++) {                                                \
    if (array[i] COMPARISON_OPERATOR comparison_value) {                     \
      result_array[pos++] = i + result_tid_offset;                           \
    }                                                                        \
  }                                                                          \
  result_size = pos;

/*
//
//                if(mask){ \
//                    if(SIMD_DEBUG_MODE) std::cout << "at least one match!" <<
std::endl; \
//                    for(unsigned j=0;j<sizeof(__m128i)/sizeof(int);++j){ \
//                        if(SIMD_DEBUG_MODE) std::cout << "sub index: " << j <<
" value: " << ((mask >> j) & 1) << std::endl; \
//                        if((mask >> j) & 1){ \
//
result_array[pos++]=i*(sizeof(__m128i)/sizeof(int))+j+(alignment_offset/sizeof(int));
\
//                        } \
//                    } \
//                } */

namespace CDK {
namespace selection {
namespace variants {

void unrolled_simd_selection_int_thread(int* array, unsigned int array_size,
                                        int comparison_value,
                                        const ValueComparator comp,
                                        unsigned int* result_array,
                                        unsigned int* result_array_size,
                                        unsigned int result_tid_offset) {
  // unsigned int* result_array = new unsigned int[array_size];
  assert(result_array != NULL);

  unsigned int result_size = 0;

  if (comp == EQUAL) {
    COGADB_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value,
        _mm_cmpeq_epi32, ==, result_size, result_tid_offset);
  } else if (comp == LESSER) {
    COGADB_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value,
        _mm_cmplt_epi32, <, result_size, result_tid_offset);
  } else if (comp == LESSER_EQUAL) {
    // add one to comparison value to compare for less equal
    COGADB_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value + 1,
        _mm_cmplt_epi32, <, result_size, result_tid_offset);
  } else if (comp == GREATER) {
    COGADB_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value,
        _mm_cmpgt_epi32, >, result_size, result_tid_offset);
  } else if (comp == GREATER_EQUAL) {
    // substract one of comparison value to compare for greater equal
    COGADB_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value - 1,
        _mm_cmpgt_epi32, >, result_size, result_tid_offset);
  } else {
    COGADB_FATAL_ERROR("Invalid ValueComparator!", "");
  }
  *result_array_size = result_size;
}

void unrolled_simd_selection_float_thread(float* array, unsigned int array_size,
                                          float comparison_value,
                                          const ValueComparator comp,
                                          unsigned int* result_array,
                                          unsigned int* result_array_size,
                                          unsigned int result_tid_offset) {
  // unsigned int* result_array = new unsigned int[array_size];
  assert(result_array != NULL);

  unsigned int result_size = 0;

  if (comp == EQUAL) {
    COGADB_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmpeq_ps,
        ==, result_size, result_tid_offset);
  } else if (comp == LESSER) {
    COGADB_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmplt_ps,
        <, result_size, result_tid_offset);
  } else if (comp == LESSER_EQUAL) {
    // add one to comparison value to compare for less equal
    COGADB_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmple_ps,
        <=, result_size, result_tid_offset);
  } else if (comp == GREATER) {
    COGADB_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmpgt_ps,
        >, result_size, result_tid_offset);
  } else if (comp == GREATER_EQUAL) {
    // substract one of comparison value to compare for greater equal
    COGADB_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmpge_ps,
        >=, result_size, result_tid_offset);
  } else {
    COGADB_FATAL_ERROR("Invalid ValueComparator!", "");
  }
  *result_array_size = result_size;
}

void bf_unrolled_simd_selection_int_thread(int* array, unsigned int array_size,
                                           int comparison_value,
                                           const ValueComparator comp,
                                           unsigned int* result_array,
                                           unsigned int* result_array_size,
                                           unsigned int result_tid_offset) {
  // unsigned int* result_array = new unsigned int[array_size];
  assert(result_array != NULL);

  unsigned int result_size = 0;

  if (comp == EQUAL) {
    COGADB_BF_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value,
        _mm_cmpeq_epi32, ==, result_size, result_tid_offset);
  } else if (comp == LESSER) {
    COGADB_BF_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value,
        _mm_cmplt_epi32, <, result_size, result_tid_offset);
  } else if (comp == LESSER_EQUAL) {
    // add one to comparison value to compare for less equal
    COGADB_BF_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value + 1,
        _mm_cmplt_epi32, <, result_size, result_tid_offset);
  } else if (comp == GREATER) {
    COGADB_BF_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value,
        _mm_cmpgt_epi32, >, result_size, result_tid_offset);
  } else if (comp == GREATER_EQUAL) {
    // substract one of comparison value to compare for greater equal
    COGADB_BF_Unrolled_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value - 1,
        _mm_cmpgt_epi32, >, result_size, result_tid_offset);
  } else {
    COGADB_FATAL_ERROR("Invalid ValueComparator!", "");
  }
  *result_array_size = result_size;
}

void bf_unrolled_simd_selection_float_thread(
    float* array, unsigned int array_size, float comparison_value,
    const ValueComparator comp, unsigned int* result_array,
    unsigned int* result_array_size, unsigned int result_tid_offset) {
  // unsigned int* result_array = new unsigned int[array_size];
  assert(result_array != NULL);

  unsigned int result_size = 0;

  if (comp == EQUAL) {
    COGADB_BF_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmpeq_ps,
        ==, result_size, result_tid_offset);
  } else if (comp == LESSER) {
    COGADB_BF_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmplt_ps,
        <, result_size, result_tid_offset);
  } else if (comp == LESSER_EQUAL) {
    // add one to comparison value to compare for less equal
    COGADB_BF_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmple_ps,
        <=, result_size, result_tid_offset);
  } else if (comp == GREATER) {
    COGADB_BF_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmpgt_ps,
        >, result_size, result_tid_offset);
  } else if (comp == GREATER_EQUAL) {
    // substract one of comparison value to compare for greater equal
    COGADB_BF_Unrolled_SIMD_SCAN(
        float, __m128, _mm_load_ps, float, _mm_set1_ps, array, array_size,
        (result_array + (*result_array_size)), comparison_value, _mm_cmpge_ps,
        >=, result_size, result_tid_offset);
  } else {
    COGADB_FATAL_ERROR("Invalid ValueComparator!", "");
  }
  *result_array_size = result_size;
}
};
};
};

#endif

#ifdef ENABLE_SIMD_ACCELERATION

void simd_selection_int_thread(int* array, unsigned int array_size,
                               int comparison_value, const ValueComparator comp,
                               unsigned int* result_array,
                               unsigned int* result_array_size,
                               unsigned int result_tid_offset) {
  // unsigned int* result_array = new unsigned int[array_size];
  assert(result_array != NULL);

  unsigned int result_size = 0;

  if (comp == EQUAL) {
    COGADB_SIMD_SCAN(int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32,
                     array, array_size, (result_array + (*result_array_size)),
                     comparison_value, _mm_cmpeq_epi32, ==, result_size,
                     result_tid_offset);
  } else if (comp == LESSER) {
    COGADB_SIMD_SCAN(int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32,
                     array, array_size, (result_array + (*result_array_size)),
                     comparison_value, _mm_cmplt_epi32, <, result_size,
                     result_tid_offset);
  } else if (comp == LESSER_EQUAL) {
    // add one to comparison value to compare for less equal
    COGADB_SIMD_SCAN(int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32,
                     array, array_size, (result_array + (*result_array_size)),
                     comparison_value + 1, _mm_cmplt_epi32, <, result_size,
                     result_tid_offset);
  } else if (comp == GREATER) {
    COGADB_SIMD_SCAN(int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32,
                     array, array_size, (result_array + (*result_array_size)),
                     comparison_value, _mm_cmpgt_epi32, >, result_size,
                     result_tid_offset);
  } else if (comp == GREATER_EQUAL) {
    // substract one of comparison value to compare for greater equal
    COGADB_SIMD_SCAN(int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32,
                     array, array_size, (result_array + (*result_array_size)),
                     comparison_value - 1, _mm_cmpgt_epi32, >, result_size,
                     result_tid_offset);
  } else {
    COGADB_FATAL_ERROR("Invalid ValueComparator!", "");
  }
  // write result size
  *result_array_size = result_size;
}

void simd_selection_float_thread(float* array, unsigned int array_size,
                                 float comparison_value,
                                 const ValueComparator comp,
                                 unsigned int* result_array,
                                 unsigned int* result_array_size,
                                 unsigned int result_tid_offset) {
  // unsigned int* result_array = new unsigned int[array_size];
  assert(result_array != NULL);

  // will be set by the makro
  unsigned int result_size = 0;
  if (comp == EQUAL) {
    COGADB_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                     array_size, (result_array + (*result_array_size)),
                     comparison_value, _mm_cmpeq_ps, ==, result_size,
                     result_tid_offset);
  } else if (comp == LESSER) {
    COGADB_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                     array_size, (result_array + (*result_array_size)),
                     comparison_value, _mm_cmplt_ps, <, result_size,
                     result_tid_offset);
  } else if (comp == LESSER_EQUAL) {
    // add one to comparison value to compare for less equal
    COGADB_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                     array_size, (result_array + (*result_array_size)),
                     comparison_value, _mm_cmple_ps, <=, result_size,
                     result_tid_offset);
  } else if (comp == GREATER) {
    COGADB_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                     array_size, (result_array + (*result_array_size)),
                     comparison_value, _mm_cmpgt_ps, >, result_size,
                     result_tid_offset);
  } else if (comp == GREATER_EQUAL) {
    // substract one of comparison value to compare for greater equal
    COGADB_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                     array_size, (result_array + (*result_array_size)),
                     comparison_value, _mm_cmpge_ps, >=, result_size,
                     result_tid_offset);
  } else {
    COGADB_FATAL_ERROR("Invalid ValueComparator!", "");
  }

  // write result size
  *result_array_size = result_size;
}

void bf_simd_selection_int_thread(int* array, unsigned int array_size,
                                  int comparison_value,
                                  const ValueComparator comp,
                                  unsigned int* result_array,
                                  unsigned int* result_array_size,
                                  unsigned int result_tid_offset) {
  // unsigned int* result_array = new unsigned int[array_size];
  assert(result_array != NULL);

  unsigned int result_size = 0;

  if (comp == EQUAL) {
    COGADB_BF_SIMD_SCAN(int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32,
                        array, array_size,
                        (result_array + (*result_array_size)), comparison_value,
                        _mm_cmpeq_epi32, ==, result_size, result_tid_offset);
  } else if (comp == LESSER) {
    COGADB_BF_SIMD_SCAN(int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32,
                        array, array_size,
                        (result_array + (*result_array_size)), comparison_value,
                        _mm_cmplt_epi32, <, result_size, result_tid_offset);
  } else if (comp == LESSER_EQUAL) {
    // add one to comparison value to compare for less equal
    COGADB_BF_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value + 1,
        _mm_cmplt_epi32, <, result_size, result_tid_offset);
  } else if (comp == GREATER) {
    COGADB_BF_SIMD_SCAN(int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32,
                        array, array_size,
                        (result_array + (*result_array_size)), comparison_value,
                        _mm_cmpgt_epi32, >, result_size, result_tid_offset);
  } else if (comp == GREATER_EQUAL) {
    // substract one of comparison value to compare for greater equal
    COGADB_BF_SIMD_SCAN(
        int, __m128i, _mm_load_si128, __m128i, _mm_set1_epi32, array,
        array_size, (result_array + (*result_array_size)), comparison_value - 1,
        _mm_cmpgt_epi32, >, result_size, result_tid_offset);
  } else {
    COGADB_FATAL_ERROR("Invalid ValueComparator!", "");
  }
  // write result size
  *result_array_size = result_size;
}

void bf_simd_selection_float_thread(float* array, unsigned int array_size,
                                    float comparison_value,
                                    const ValueComparator comp,
                                    unsigned int* result_array,
                                    unsigned int* result_array_size,
                                    unsigned int result_tid_offset) {
  // unsigned int* result_array = new unsigned int[array_size];
  assert(result_array != NULL);

  // will be set by the makro
  unsigned int result_size = 0;
  if (comp == EQUAL) {
    COGADB_BF_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                        array_size, (result_array + (*result_array_size)),
                        comparison_value, _mm_cmpeq_ps, ==, result_size,
                        result_tid_offset);
  } else if (comp == LESSER) {
    COGADB_BF_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                        array_size, (result_array + (*result_array_size)),
                        comparison_value, _mm_cmplt_ps, <, result_size,
                        result_tid_offset);
  } else if (comp == LESSER_EQUAL) {
    // add one to comparison value to compare for less equal
    COGADB_BF_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                        array_size, (result_array + (*result_array_size)),
                        comparison_value, _mm_cmple_ps, <=, result_size,
                        result_tid_offset);
  } else if (comp == GREATER) {
    COGADB_BF_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                        array_size, (result_array + (*result_array_size)),
                        comparison_value, _mm_cmpgt_ps, >, result_size,
                        result_tid_offset);
  } else if (comp == GREATER_EQUAL) {
    // substract one of comparison value to compare for greater equal
    COGADB_BF_SIMD_SCAN(float, __m128, _mm_load_ps, float, _mm_set1_ps, array,
                        array_size, (result_array + (*result_array_size)),
                        comparison_value, _mm_cmpge_ps, >=, result_size,
                        result_tid_offset);
  } else {
    COGADB_FATAL_ERROR("Invalid ValueComparator!", "");
  }

  // write result size
  *result_array_size = result_size;
}

// scans an integer column with SIMD isntructions, result_tid_offset is a fixed
// value which is added to all matching tid value, allowing to use the SIMD Scan
// in Multithreaded algorithms
const PositionListPtr simd_selection_int(int* array, unsigned int array_size,
                                         int comparison_value,
                                         const ValueComparator comp,
                                         unsigned int result_tid_offset) {
  PositionListPtr result_tids = createPositionList();
  unsigned int result_size = 0;

  simd_selection_int_thread(array, array_size, comparison_value, comp,
                            result_tids->data(), &result_size,
                            result_tid_offset);

  resize(*result_tids.get(), result_size);
  return result_tids;
}
#endif

#ifdef ENABLE_SIMD_ACCELERATION
//        template<>
//        inline const PositionListPtr Column<float>::selection(const
//        boost::any& value_for_comparison, const ValueComparator comp){
//            //std::cout << "SIMD SCAN" << std::endl;
//            float value;
//
//            if(value_for_comparison.type()!=typeid(float)){
//					 //allow comparison with itnegers as
// well
//                if(value_for_comparison.type()==typeid(int)){
//                    value = boost::any_cast<int>(value_for_comparison);
//                }else{
//                    COGADB_FATAL_ERROR(std::string("Typemismatch for
//                    column")+this->name_
//                                       +std::string(" Column Type: ") +
//                                       typeid(float).name()
//                                       +std::string(" filter value type: ") +
//                                       value_for_comparison.type().name(),"");
//					 }
//            }else{
//                //everything fine, filter value matches type of column
//                value = boost::any_cast<float>(value_for_comparison);
//            }
//            unsigned int array_size = this->size();
//            float* array=hype::util::begin_ptr(values_);

const PositionListPtr simd_selection_float(float* array,
                                           unsigned int array_size,
                                           float comparison_value,
                                           const ValueComparator comp,
                                           unsigned int result_tid_offset) {
  PositionListPtr result_tids = createPositionList();
  unsigned int result_size = 0;

  simd_selection_float_thread(array, array_size, comparison_value, comp,
                              result_tids->data(), &result_size,
                              result_tid_offset);

  resize(*result_tids.get(), result_size);
  return result_tids;
}

const PositionListPtr bf_simd_selection_int(int* array, unsigned int array_size,
                                            int comparison_value,
                                            const ValueComparator comp,
                                            unsigned int result_tid_offset) {
  PositionListPtr result_tids = createPositionList();
  unsigned int result_size = 0;

  bf_simd_selection_int_thread(array, array_size, comparison_value, comp,
                               result_tids->data(), &result_size,
                               result_tid_offset);

  resize(*result_tids.get(), result_size);
  return result_tids;
}

const PositionListPtr bf_simd_selection_float(float* array,
                                              unsigned int array_size,
                                              float comparison_value,
                                              const ValueComparator comp,
                                              unsigned int result_tid_offset) {
  PositionListPtr result_tids = createPositionList();
  unsigned int result_size = 0;

  bf_simd_selection_float_thread(array, array_size, comparison_value, comp,
                                 result_tids->data(), &result_size,
                                 result_tid_offset);

  resize(*result_tids.get(), result_size);
  return result_tids;
}

#endif

}  // end namespace CoGaDB
