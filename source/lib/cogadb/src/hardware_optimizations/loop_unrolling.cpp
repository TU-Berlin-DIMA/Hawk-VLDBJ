
#include <boost/chrono.hpp>
#include <boost/thread.hpp>
#include <core/global_definitions.hpp>

#include <stdint.h>
#include <algorithm>

//#define __SSE__
//#define __SSE2__
//#define __SSE3__
//#define __SSE4_1__
//#define __MXX__

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

namespace CoGaDB {
using namespace boost::chrono;
uint64_t getTimestamp() {
  high_resolution_clock::time_point tp = high_resolution_clock::now();
  nanoseconds dur = tp.time_since_epoch();

  return (uint64_t)dur.count();
}

#define COGADB_INSTANTIATE_FUNCTION_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES( \
    TEMPLATE)                                                           \
  template class TEMPLATE<int>;                                         \
  template class TEMPLATE<foat>;                                        \
  template class TEMPLATE<std::string>;

// class template
// The explicit instantiation part
// template class myTemplate<int>;

#define COGADB_INSTANTIATE_FUNCTION_TEMPLATE_FOR_SUPPORTED_TYPES(                                                                                     \
    RETURNTYPE, FUNCTION_TEMPLATE_NAME, FUNCTION_TEMPLATE_PARAMETERS) template RETURNTYPE FUNCTION_TEMPLATE_NAME<int>(FUNCTION_TEMPLATE_PARAMETERS)); \
        template RETURNTYPE FUNCTION_TEMPLATE_NAME<float>(FUNCTION_TEMPLATE_PARAMETERS));                                                             \
        template RETURNTYPE FUNCTION_TEMPLATE_NAME<std::string>(FUNCTION_TEMPLATE_PARAMETERS));

// function:
// template void func<int>(int param); // explicit instantiation.

#define COGADB_SELECTION_BODY_WRITE_TID_BRANCH(array, value, i, array_tids, \
                                               pos, COMPARATOR)             \
  if (array[i] COMPARATOR value) {                                          \
    array_tids[pos] = i;                                                    \
  }
#define COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array, value, i, array_tids, \
                                                 pos, COMPARATOR)             \
  array_tids[pos] = i;                                                        \
  pos += (array[i] COMPARATOR value);

#define COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i, pos, \
                                               result_bitmap)        \
  result_bitmap[pos++] = (array[i] == value);

#define COGADB_SELECTION_BODY COGADB_SELECTION_BODY_WRITE_TID_BRANCH

#define SIMD_DEBUG_MODE 0

#define LOOP_BODY(i) result_array[pos++] = i + result_tid_offset;

#define COGADB_SIMD_SCAN_INT(array, array_size, result_array,                \
                             comparison_value, SIMD_COMPARISON_FUNCTION,     \
                             COMPARISON_OPERATOR, result_size,               \
                             result_tid_offset)                              \
  __m128i* sse_array = reinterpret_cast<__m128i*>(array);                    \
  assert(sse_array != NULL);                                                 \
  int alignment_offset = ((intptr_t)sse_array) % sizeof(__m128i);            \
  const int sse_array_length =                                               \
      (array_size - alignment_offset) * sizeof(int) / sizeof(__m128i);       \
  if (SIMD_DEBUG_MODE)                                                       \
    std::cout << "SSE Array Length: " << sse_array_length << std::endl;      \
  char* tmp_array = (char*)sse_array;                                        \
  tmp_array += alignment_offset;                                             \
  sse_array = reinterpret_cast<__m128i*>(tmp_array);                         \
  if (SIMD_DEBUG_MODE)                                                       \
    std::cout << "array adress: " << (void*)array                            \
              << "sse array: " << (void*)sse_array << std::endl;             \
  if (SIMD_DEBUG_MODE)                                                       \
    std::cout << "First SSE Array Element: " << ((int*)sse_array)[0]         \
              << std::endl;                                                  \
  unsigned int pos = 0;                                                      \
  __m128i comp_val = _mm_set1_epi32(comparison_value);                       \
  __m128i read_value = _mm_set1_epi32(0);                                    \
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
    assert(((intptr_t)sse_array) % sizeof(__m128i) == 0);                    \
    read_value = _mm_load_si128(&sse_array[i]);                              \
    if (SIMD_DEBUG_MODE) {                                                   \
      std::cout << "index: " << i << std::endl;                              \
    }                                                                        \
    __m128 comp_result =                                                     \
        (__m128)SIMD_COMPARISON_FUNCTION(read_value, comp_val);              \
    int mask = _mm_movemask_ps(comp_result);                                 \
    if (SIMD_DEBUG_MODE)                                                     \
      std::cout << "Mask: " << std::hex << mask << std::dec << std::endl;    \
    if (mask) {                                                              \
      if (SIMD_DEBUG_MODE) std::cout << "at least one match!" << std::endl;  \
      for (unsigned j = 0; j < sizeof(__m128i) / sizeof(int); ++j) {         \
        if (SIMD_DEBUG_MODE)                                                 \
          std::cout << "sub index: " << j << " value: " << ((mask >> j) & 1) \
                    << std::endl;                                            \
        int tmp = ((mask >> j) & 1);                                         \
        result_array[pos] = i * (sizeof(__m128i) / sizeof(int)) + j +        \
                            (alignment_offset / sizeof(int)) +               \
                            result_tid_offset;                               \
        pos += tmp;                                                          \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  if (SIMD_DEBUG_MODE)                                                       \
    std::cout << "Remaining offsets: "                                       \
              << (sse_array_length * sizeof(__m128i) / sizeof(int)) +        \
                     alignment_offset                                        \
              << " to " << array_size << std::endl;                          \
  for (unsigned int i = (sse_array_length * sizeof(__m128i) / sizeof(int)) + \
                        (alignment_offset / sizeof(int));                    \
       i < array_size; i++) {                                                \
    if (array[i] COMPARISON_OPERATOR comparison_value) {                     \
      result_array[pos++] = i + result_tid_offset;                           \
    }                                                                        \
  }                                                                          \
  result_size = pos;

//#define COGADB_SIMD_LOOP_BODY __m128 comp_result = (__m128) SIMD_COMPARISON_FUNCTION(read_value,comp_val); \
//                int mask= _mm_movemask_ps(comp_result); \
//                if(SIMD_DEBUG_MODE) std::cout << "Mask: " << std::hex << mask << std::dec << std::endl; \
//                if(mask){ \
//                    if(SIMD_DEBUG_MODE) std::cout << "at least one match!" << std::endl; \
//                    for(unsigned j=0;j<sizeof(__m128i)/sizeof(int);++j){ \
//                        if(SIMD_DEBUG_MODE) std::cout << "sub index: " << j << " value: " << ((mask >> j) & 1) << std::endl; \
//                        int tmp =((mask >> j) & 1); \
//                        result_array[pos]=i*(sizeof(__m128i)/sizeof(int))+j+(alignment_offset/sizeof(int))+result_tid_offset; \
//                        pos+=tmp; \
//                    } \
//                } \


//         #define SIMD_PROLOG __m128i comp_val=_mm_set1_epi32(comparison_value); \
//                                              __m128i read_value=_mm_set1_epi32 (0);

/*
#define COGADB_SIMD_ACCELERATED_PRIMITIVE(array, array_size, result_array,
comparison_value,SIMD_PROLOG,LOOP_BODY, SIMD_LOOP_BODY, result_size,
result_tid_offset) \
        __m128i* sse_array = reinterpret_cast<__m128i*>(array); \
        assert(sse_array!=NULL); \
        int alignment_offset = ((intptr_t)sse_array)%sizeof(__m128i); \
        const int sse_array_length =
(array_size-alignment_offset)*sizeof(int)/sizeof(__m128i); \
        if(SIMD_DEBUG_MODE) std::cout << "SSE Array Length: " <<
sse_array_length << std::endl; \
        char* tmp_array = (char*) sse_array; \
        tmp_array+=alignment_offset; \
        sse_array=reinterpret_cast<__m128i*>(tmp_array); \
        if(SIMD_DEBUG_MODE)  std::cout << "array adress: "  << (void*)array <<
"sse array: " << (void*)sse_array << std::endl; \
        if(SIMD_DEBUG_MODE)  std::cout << "First SSE Array Element: " <<
((int*)sse_array)[0] << std::endl; \
        unsigned int pos=0; \
        SIMD_PROLOG
        if(SIMD_DEBUG_MODE)  std::cout << "alignment_offset " <<
alignment_offset << std::endl; \
        if(alignment_offset!=0){ \
            if(SIMD_DEBUG_MODE) std::cout << "process first unaligned data
chunk: index 0 to " << alignment_offset << std::endl; \
            for(unsigned int i=0;i<alignment_offset/sizeof(int);i++){ \
                if(SIMD_DEBUG_MODE){ \
                    std::cout << "index "<< i << std::endl; \
                    std::cout << "value " << array[i] << " match:" << (array[i]
COMPARISON_OPERATOR comparison_value) << std::endl; \
                } \
                if(array[i] COMPARISON_OPERATOR comparison_value){ \
                    LOOP_BODY;
                } \
            } \
        } \
        for(unsigned int i=0;i<sse_array_length;i++){ \
                assert(((intptr_t)sse_array)%sizeof(__m128i)==0); \
                read_value=_mm_load_si128(&sse_array[i]); \
                if(SIMD_DEBUG_MODE){ \
                    std::cout << "index: " << i << std::endl; \
                } \
                SIMD_LOOP_BODY
        } \
        if(SIMD_DEBUG_MODE) std::cout << "Remaining offsets: " <<
(sse_array_length*sizeof(__m128i)/sizeof(int))+alignment_offset  << " to " <<
array_size << std::endl; \
        for(unsigned int
i=(sse_array_length*sizeof(__m128i)/sizeof(int))+(alignment_offset/sizeof(int));i<array_size;i++){
\
            if(array[i] COMPARISON_OPERATOR comparison_value){ \
                LOOP_BODY(i);
            } \
        } \
        result_size=pos;
*/

//
// template <typename T>
// void scan_column_equal(T* array, TID begin_index, TID end_index, const T&
// value, TID* array_tids, size_t& result_size, TID pos){
//                TID i=begin_index;
//                for(;i+8<end_index;i+=8){
//                    COGADB_SELECTION_BODY_BRANCH(array,value,i,array_tids,pos,==);
//                    COGADB_SELECTION_BODY_BRANCH(array,value,i+1,array_tids,pos,==);
//                    COGADB_SELECTION_BODY_BRANCH(array,value,i+2,array_tids,pos,==);
//                    COGADB_SELECTION_BODY_BRANCH(array,value,i+3,array_tids,pos,==);
//                    COGADB_SELECTION_BODY_BRANCH(array,value,i+4,array_tids,pos,==);
//                    COGADB_SELECTION_BODY_BRANCH(array,value,i+5,array_tids,pos,==);
//                    COGADB_SELECTION_BODY_BRANCH(array,value,i+6,array_tids,pos,==);
//                    COGADB_SELECTION_BODY_BRANCH(array,value,i+7,array_tids,pos,==);
//                }
//                for(;i<end_index;++i){
//                    COGADB_SELECTION_BODY_BRANCH(array,value,i,array_tids,pos,==);
//                }
//}

// template <typename T>
// void scan_column_equal(T* array, TID begin_index, TID end_index, const T&
// value, TID* array_tids, size_t& result_size, TID pos);
//

// void scan_simd_test(){
//
//    int* array, result_array;
//    size_t array_size;
//
//    TID begin_index, TID end_inde
//    COGADB_SIMD_ACCELERATED_PRIMITIVE();
//
//}

template <typename T>
void gather_thread(T* __restrict__ array, TID* __restrict__ tid_array,
                   const TID& begin_index, const TID& end_index,
                   T* __restrict__ result_array) {
  TID pos = begin_index;
  const size_t chunk_size = end_index - begin_index;
  for (int i = 0; i < chunk_size; ++i) {
    result_array[pos] = array[tid_array[pos]];
    pos++;
  }
}

template <typename T>
void parallel_gather(T* __restrict__ array, TID* __restrict__ tid_array,
                     const size_t& array_size, T* __restrict__ result_array,
                     unsigned int number_of_threads) {
  std::vector<unsigned int> result_sizes(number_of_threads);
  boost::thread_group threads;

  for (unsigned int thread_id = 0; thread_id < number_of_threads; ++thread_id) {
    // number of elements per thread
    unsigned int chunk_size = array_size / number_of_threads;
    TID begin_index = chunk_size * thread_id;
    TID end_index;
    if (thread_id + 1 == number_of_threads) {
      // process until end of input array
      end_index = array_size;
    } else {
      end_index = (chunk_size) * (thread_id + 1);
    }

    // gather_thread(array, tid_array, begin_index, end_index, result_array);
    // create a gather thread
    threads.add_thread(
        new boost::thread(boost::bind(&gather_thread<T>, array, tid_array,
                                      begin_index, end_index, result_array)));
  }
  threads.join_all();
}

template <typename T>
void scan_column_equal_bitmap(T* __restrict__ array, const TID& begin_index,
                              const TID& end_index, const T& value,
                              char* __restrict__ result_bitmap,
                              size_t& result_size) {
  //                TID pos=begin_index;
  //                const size_t chunk_size=end_index-begin_index;
  //                for(int i=0;i<chunk_size;++i){
  //                    {COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i,
  //                    pos, result_bitmap);}
  //                }

  TID pos = begin_index;
  const size_t chunk_size = end_index - begin_index;
  int i = 0;
  for (; i < chunk_size; i += 8) {
    COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i, pos, result_bitmap);
    COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i + 1, pos,
                                           result_bitmap);
    COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i + 2, pos,
                                           result_bitmap);
    COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i + 3, pos,
                                           result_bitmap);
    COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i + 4, pos,
                                           result_bitmap);
    COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i + 5, pos,
                                           result_bitmap);
    COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i + 6, pos,
                                           result_bitmap);
    COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i + 7, pos,
                                           result_bitmap);
  }

  for (; i < chunk_size; ++i) {
    COGADB_SELECTION_BODY_SET_BIT_NOBRANCH(array, value, i, pos, result_bitmap);
  }

  result_size = pos;
}

template <typename T>
void scan_column_equal_bitmap_parallel(T* __restrict__ array,
                                       const size_t& array_size, const T& value,
                                       char* __restrict__ result_bitmap,
                                       size_t& result_size,
                                       unsigned int number_of_threads) {
  std::vector<unsigned int> result_sizes(number_of_threads);
  boost::thread_group threads;

  for (unsigned int thread_id = 0; thread_id < number_of_threads; ++thread_id) {
    // number of elements per thread
    unsigned int chunk_size = array_size / number_of_threads;
    TID begin_index = chunk_size * thread_id;
    TID end_index;
    if (thread_id + 1 == number_of_threads) {
      // process until end of input array
      end_index = array_size;
    } else {
      end_index = (chunk_size) * (thread_id + 1);
    }

    // scan_column_equal_bitmap(array, begin_index, end_index, value,
    // result_bitmap, result_size);
    // create a selection thread
    threads.add_thread(new boost::thread(
        boost::bind(&scan_column_equal_bitmap<int>, array, begin_index,
                    end_index, value, result_bitmap, result_size)));
  }
  threads.join_all();

  std::vector<unsigned int> prefix_sum(number_of_threads + 1);
  prefix_sum[0] = 0;
  for (unsigned int i = 1; i < number_of_threads + 1; i++) {
    prefix_sum[i] = prefix_sum[i - 1] + result_sizes[i - 1];
  }
  result_size = prefix_sum[number_of_threads];
}

template <typename T>
void scan_column_equal_tid(T* __restrict__ array, const TID& begin_index,
                           const TID& end_index, const T& value,
                           TID* __restrict__ array_tids, size_t& result_size,
                           const TID& write_index) {
  //                 TID pos=write_index;
  //                for(TID i=begin_index;i<end_index;++i){
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i,array_tids,pos,==);
  //                }

  //                TID pos=write_index;
  //                TID i;
  //                for(i=begin_index;i+7<end_index;i+=8){
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i,array_tids,pos,==);
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i+1,array_tids,pos,==);
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i+2,array_tids,pos,==);
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i+3,array_tids,pos,==);
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i+4,array_tids,pos,==);
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i+5,array_tids,pos,==);
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i+6,array_tids,pos,==);
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i+7,array_tids,pos,==);
  //                }
  //
  //                for(;i<end_index;++i){
  //                    COGADB_SELECTION_BODY_WRITE_TID_NOBRANCH(array,value,i,array_tids,pos,==);
  //                }

  size_t array_size = end_index - begin_index;
  COGADB_SIMD_SCAN_INT(array, array_size, array_tids, value, _mm_cmpeq_epi32,
                       ==, result_size, 0);

  //                TID pos=write_index;
  //                for(TID i=begin_index;i<end_index;++i){
  //                    array_tids[pos]=i; pos+=(array[i] == value);
  //                    //if(array[i] == value) std::cout << i << std::endl;
  //                }
  //                result_size=pos;

  //                TID mypos=0;
  //                size_t size=end_index-begin_index;
  //                for(TID i=0;i<size;++i){
  //////                    array_tids[mypos]=i; mypos++; //=(array[i] ==
  /// value);
  ////                    bool tmp=(array[i] == value);
  ////                    array_tids[mypos]=i; mypos+=tmp; //mypos++;
  //                    if(array[i] == value){ array_tids[mypos]=i; mypos++;}
  //                }

  // TID pos=write_index;
  //                TID my_pos=0;
  //                size_t size=end_index-begin_index;
  //                for(TID i=0;i<size;++i){
  //                    array_tids[my_pos]=i;
  //                    bool tmp = (array[i] == value);
  //                    my_pos+=tmp;
  //                }
  //
  //                int k=6;
  //                int i=0;
  //                for(;i<end_index;++i){
  //                    array_tids[i]=array[i]*5;
  //                }

  //    int a[256], b[256], c[256];
  //  int i;
  //
  //  for (i=0; i<256; i++){
  //    a[i] = b[i] + c[i];
  //  }

  //  int a[256], b[256], c[256];
  //  int size=100;
  //
  //
  //   int i;
  //
  //   /* feature: support for unknown loop bound  */
  //   /* feature: support for loop invariants  */
  //   for (i=0; i<size; i++){
  //      b[i] = value;
  //   }
}

/*
template<class T>
void selection_thread(unsigned int thread_id, unsigned int number_of_threads,
const T& value, const ValueComparator comp, ColumnBaseTyped<T>* col, unsigned
int* result_tids, unsigned int* result_size){
        //std::cout << "Hi I'm thread" << thread_id << std::endl;
        if(!quiet) std::cout << "Using CPU for Selection (parallel mode)..." <<
std::endl;
            ColumnBaseTyped<T>& column_ref = dynamic_cast< ColumnBaseTyped<T>&
>(*col);
            //number of elements per thread
            unsigned int chunk_size = column_ref.size()/number_of_threads;
            unsigned int begin_index=chunk_size*thread_id;
            unsigned int end_index;
            if(thread_id+1 == number_of_threads){
                //process until end of input array
                end_index=column_ref.size();
            }else{
                end_index=(chunk_size)*(thread_id+1);
            }
            //cout << "Thread " << thread_id << " begin index: " <<  begin_index
<< " end index: " << end_index << endl;
            unsigned int pos=0;
            //size_t result_size=0;
            if(comp==EQUAL){
                scan_column_equal(array, begin_index, end_index, array_tids,
pos);
            }else if(comp==LESSER){
                for(TID i=begin_index;i<end_index;++i){
                    if(column_ref[i]<value){
                            result_tids[pos+begin_index]=i;
                            pos++;
                    }
                }
            }else if(comp==LESSER_EQUAL){
                for(TID i=begin_index;i<end_index;++i){
                    if(column_ref[i]<=value){
                            result_tids[pos+begin_index]=i;
                            pos++;
                    }
                }
            }else if(comp==GREATER){
                for(TID i=begin_index;i<end_index;++i){
                    if(column_ref[i]>value){
                            result_tids[pos+begin_index]=i;
                            pos++;
                    }
                }
            }else if(comp==GREATER_EQUAL){
                for(TID i=begin_index;i<end_index;++i){
                    if(column_ref[i]>=value){
                            result_tids[pos+begin_index]=i;
                            pos++;
                    }
                }
            }else{
                COGADB_FATAL_ERROR("Unsupported Filter Predicate!","");
            }
                //}

            //write result size to array
            *result_size=pos;
}
*/
};

using namespace CoGaDB;

int main(int argc, char* argv[]) {
  //  const int NUMBER_OF_ELEMENTS=100*1000*1000;
  const int NUMBER_OF_ELEMENTS = 6 * 1000 * 1000;
//#define ENABLE_SCAN_TEST
#define ENABLE_GATHER_TEST

// COGADB_INSTANTIATE_FUNCTION_TEMPLATE_FOR_SUPPORTED_TYPES(void,parallel_gather,T*
// __restrict__ array, TID* __restrict__ tid_array, const size_t& array_size,
// T* __restrict__ result_array, unsigned int number_of_threads);

#ifdef ENABLE_SCAN_TEST
  {
    int* array = (int*)malloc(sizeof(int) * NUMBER_OF_ELEMENTS);

    unsigned int number_of_threads = 6;
    std::cout << "Enter Number of Threads: " << std::endl;
    // std::cin >> number_of_threads;

    std::memset(array, 0, sizeof(int) * NUMBER_OF_ELEMENTS);

    for (unsigned int i = 0; i < NUMBER_OF_ELEMENTS; ++i) {
      array[i] = rand() % 100;
    }
    int comparison_value = 42;

    void* p = (void*)&scan_column_equal_tid<int>;
    std::cout << p << std::endl;

    for (unsigned int i = 0; i < 20 * 10; ++i) {
      uint64_t begin = getTimestamp();
      TID* result_buffer = (TID*)malloc(sizeof(TID) * NUMBER_OF_ELEMENTS);
      size_t result_size = 0;
      // std::cout << "Iteration: " << i << std::endl;
      scan_column_equal_tid(array, 0, NUMBER_OF_ELEMENTS, comparison_value,
                            result_buffer, result_size, TID(0));
      if (result_buffer) free(result_buffer);
      uint64_t end = getTimestamp();
      assert(end > begin);
      uint64_t second = 1000 * 1000 * 1000;
      double rel_time = second / (end - begin);  /// 1000*1000*1000
      std::cout << "Time TID Scan: " << double(end - begin) / (1000 * 1000)
                << "ms ("
                << (double(sizeof(int) * NUMBER_OF_ELEMENTS) * rel_time) /
                       (1024 * 1024)
                << "MB/s)" << std::endl;
    }

    // begin = getTimestamp();
    for (unsigned int i = 0; i < 20 * 10; ++i) {
      uint64_t begin = getTimestamp();
      bool* result_bitmap = (bool*)malloc(sizeof(bool) * NUMBER_OF_ELEMENTS);
      size_t result_size = 0;
      // std::cout << "Iteration: " << i << std::endl;
      // scan_column_equal_bitmap(array,0,NUMBER_OF_ELEMENTS,comparison_value,(char*)result_bitmap,result_size);
      scan_column_equal_bitmap_parallel(array, NUMBER_OF_ELEMENTS,
                                        comparison_value, (char*)result_bitmap,
                                        result_size, number_of_threads);
      if (result_bitmap) free(result_bitmap);
      uint64_t end = getTimestamp();
      assert(end > begin);
      uint64_t second = 1000 * 1000 * 1000;
      double rel_time = second / (end - begin);  /// 1000*1000*1000
      std::cout << "Time Bitmap Scan: " << double(end - begin) / (1000 * 1000)
                << "ms ("
                << (double(sizeof(int) * NUMBER_OF_ELEMENTS) * rel_time) /
                       (1024 * 1024)
                << "MB/s)" << std::endl;
      // std::cout << "Time Bitmap Scan: " << double(end-begin)/(1000*1000) <<
      // "ms" << std::endl;
    }

    // end = getTimestamp();
    //    assert(end>begin);
    //    std::cout << "Time Bitmap Scan: " << double(end-begin)/(1000*1000) <<
    //    "ms" << std::endl;

    //    std::cout << "Result Size: " <<  result_size << std::endl;
    //    for(unsigned int i=0;i<result_size;++i){
    //        std::cout << "tid: " << result_buffer[i]  << " val: " <<
    //        array[result_buffer[i]] << std::endl;
    //
    //    }

    // if(result_buffer) free(result_buffer);
    if (array) free(array);
  }
#endif

#ifdef ENABLE_GATHER_TEST
  {
    int* array = (int*)malloc(sizeof(int) * NUMBER_OF_ELEMENTS);
    TID* tid_array = (TID*)malloc(sizeof(TID) * NUMBER_OF_ELEMENTS);
    unsigned int number_of_threads = 8;
    std::cout << "Enter Number of Threads: " << std::endl;
    //        std::cin >> number_of_threads;

    for (unsigned int i = 0; i < NUMBER_OF_ELEMENTS; ++i) {
      array[i] = rand() % 100;
      tid_array[i] =
          rand() % NUMBER_OF_ELEMENTS;  // generate valid tids w.r.t. array
    }

    std::sort(tid_array, tid_array + NUMBER_OF_ELEMENTS);

    for (unsigned int i = 0; i < 20; ++i) {
      uint64_t begin = getTimestamp();
      int* result_array = (int*)malloc(sizeof(int) * NUMBER_OF_ELEMENTS);
      parallel_gather(array, tid_array, NUMBER_OF_ELEMENTS, result_array,
                      number_of_threads);
      uint64_t end = getTimestamp();
      if (result_array) free(result_array);
      assert(end > begin);
      uint64_t second = 1000 * 1000 * 1000;
      double rel_time = second / (end - begin);  /// 1000*1000*1000
      std::cout << "Time Gather Operation: "
                << double(end - begin) / (1000 * 1000) << "ms ("
                << (double(sizeof(int) * NUMBER_OF_ELEMENTS) * rel_time) /
                       (1024 * 1024)
                << "MB/s)" << std::endl;
    }

    if (array) free(array);
    if (tid_array) free(tid_array);
  }

#endif

  return 0;
}
