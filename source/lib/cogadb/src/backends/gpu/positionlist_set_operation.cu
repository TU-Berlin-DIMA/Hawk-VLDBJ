
#include <assert.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <thrust/set_operations.h>
#include <backends/gpu/positionlist_set_operation.hpp>
#include <backends/gpu/stream_manager.hpp>
#include <backends/gpu/util.hpp>
#include <cstdlib>
#include <statistics/statistics_manager.hpp>
#include <util/time_measurement.hpp>

//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, char *file, int line, bool abort =
// true) {
//    if (code != cudaSuccess) {
//        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
//        file, line);
//        if (abort) exit(code);
//    }
//}

#define gpuErrorMessage(ans) \
  { gpuAssert((ans), __FILE__, __LINE__, false); }

namespace CoGaDB {

const PositionListPtr computePositionListUnion(
    PositionListPtr tids1, PositionListPtr tids2,
    const ProcessorSpecification& proc_spec) {
  if (!tids1 || !tids2) return PositionListPtr();

  PositionListPtr result_tids =
      createPositionList(getSize(*tids1) + getSize(*tids2), proc_spec);
  if (!result_tids) return PositionListPtr();

  //        std::cout << "Left: Memory ID: " << (int) getMemoryID(*tids1) << "
  //        Size: " << getSize(*tids1) << std::endl;
  //        std::cout << "Right: Memory ID: " << (int) getMemoryID(*tids2) << "
  //        Size: " << getSize(*tids2) << std::endl;
  //        std::cout << "Result: Memory ID: " << (int)
  //        getMemoryID(*result_tids) << " Size: " << getSize(*result_tids) <<
  //        std::endl;

  try {
    thrust::device_ptr<TID> begin_tids1 =
        thrust::device_pointer_cast(getPointer(*tids1));
    thrust::device_ptr<TID> end_tids1 = thrust::device_pointer_cast(
        getPointer(*tids1) +
        getSize(*tids1));  // begin_tids1 + getSize(*tids1);
    thrust::device_ptr<TID> begin_tids2 =
        thrust::device_pointer_cast(getPointer(*tids2));
    thrust::device_ptr<TID> end_tids2 =
        thrust::device_pointer_cast(getPointer(*tids2) + getSize(*tids2));
    thrust::device_ptr<TID> result_tids_ptr =
        thrust::device_pointer_cast(getPointer(*result_tids));

    thrust::device_ptr<TID> result_end = thrust::set_union(
        begin_tids1, end_tids1, begin_tids2, end_tids2, result_tids_ptr);

    size_t result_size = result_end - result_tids_ptr;
    if (!resize(*result_tids, result_size)) return PositionListPtr();

  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during computePositionListUnion!"
              << std::endl;
    return PositionListPtr();
  }

  return result_tids;
}

const PositionListPtr computePositionListIntersection(
    PositionListPtr tids1, PositionListPtr tids2,
    const ProcessorSpecification& proc_spec) {
  if (!tids1 || !tids2) return PositionListPtr();

  PositionListPtr result_tids =
      createPositionList(getSize(*tids1) + getSize(*tids2), proc_spec);
  if (!result_tids) return PositionListPtr();

  try {
    thrust::device_ptr<TID> begin_tids1 =
        thrust::device_pointer_cast(getPointer(*tids1));
    thrust::device_ptr<TID> end_tids1 = thrust::device_pointer_cast(
        getPointer(*tids1) +
        getSize(*tids1));  // begin_tids1 + getSize(*tids1);
    thrust::device_ptr<TID> begin_tids2 =
        thrust::device_pointer_cast(getPointer(*tids2));
    thrust::device_ptr<TID> end_tids2 =
        thrust::device_pointer_cast(getPointer(*tids2) + getSize(*tids2));
    thrust::device_ptr<TID> result_tids_ptr =
        thrust::device_pointer_cast(getPointer(*result_tids));

    thrust::device_ptr<TID> result_end = thrust::set_intersection(
        begin_tids1, end_tids1, begin_tids2, end_tids2, result_tids_ptr);

    size_t result_size = result_end - result_tids_ptr;
    if (!resize(*result_tids, result_size)) return PositionListPtr();

  } catch (std::bad_alloc& e) {
    std::cerr << "Ran out of memory during computePositionListIntersection!"
              << std::endl;
    return PositionListPtr();
  }

  return result_tids;
}

const PositionListPtr
GPU_PositionListSetOperation::computePositionListSetOperation(
    PositionListPtr left, PositionListPtr right,
    const SetOperationParam& param) {
  if (param.set_op == UNION) {
    return computePositionListUnion(left, right, param.proc_spec);
  } else if (param.set_op == INTERSECT) {
    return computePositionListIntersection(left, right, param.proc_spec);
  } else {
    COGADB_FATAL_ERROR("Unknown Set Operation!", "");
    return PositionListPtr();
  }
}

};  // end namespace CoGaDB
