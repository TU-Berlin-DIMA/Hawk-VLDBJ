
#include <core/global_definitions.hpp>
#include <core/memory_allocator.hpp>
#include <map>
#include <utility>

#include <boost/shared_ptr.hpp>

#include <hype.hpp>
#ifdef ENABLE_GPU_ACCELERATION
#include <cuda_runtime.h>
#include <driver_types.h>
#endif

#include <boost/make_shared.hpp>
#include <core/copy_function_factory.hpp>
#include <util/hardware_detector.hpp>

namespace CoGaDB {

template <typename T>
class CPUMemoryAllocator_C : public MemoryAllocator<T> {
 public:
  CPUMemoryAllocator_C(hype::ProcessingDeviceMemoryID mem_id)
      : MemoryAllocator<T>(mem_id, "CPU Plain C Allocator") {}

  T* allocate(size_t number_of_elements) {
    T* ret = (T*)malloc(sizeof(T) * number_of_elements);
    return ret;
  }
  T* reallocate(T* memory_block, size_t new_number_of_elements) {
    T* ret = (T*)realloc(memory_block, new_number_of_elements * sizeof(T));
    return ret;
  }
  void deallocate(T*& memory_block) { free(memory_block); }
};

template <typename T>
class CPUMemoryAllocator_CPP : public MemoryAllocator<T> {
 public:
  CPUMemoryAllocator_CPP(hype::ProcessingDeviceMemoryID mem_id)
      : MemoryAllocator<T>(mem_id, "CPU C++ Allocator"), alloc_map_() {}

  T* allocate(size_t number_of_elements) {
    T* ret = new T[number_of_elements];
    alloc_map_.insert(std::make_pair(ret, number_of_elements));
    return ret;
  }
  T* reallocate(T* memory_block, size_t new_number_of_elements) {
    T* ret = NULL;
    typename AllocationMap::iterator it = alloc_map_.find(memory_block);
    if (memory_block) {
      if (it == alloc_map_.end()) {
        COGADB_FATAL_ERROR("Allocated Memory Block not in AllocationMap!", "");
      }
      // is the allocated chunk larger than what is requested?
      // if yes, then do nothing
      if (it->second >= new_number_of_elements) return memory_block;
    }
    // std::cout << "Called Realloc: " << (void*) memory_block << ": " <<
    // new_number_of_elements << std::endl;
    ret = allocate(new_number_of_elements);
    // in case memory block is not NULL, copy data to new buffer
    // in case memory block is NULL, allocate new buffer
    if (memory_block) {
      // copy objects to new buffer
      std::copy(memory_block, memory_block + it->second, ret);
      // delete old entry from map
      alloc_map_.erase(it);
      // delete old memory block
      delete[] memory_block;
    }
    return ret;
  }
  void deallocate(T*& memory_block) {
    typename AllocationMap::iterator it = alloc_map_.find(memory_block);
    if (it == alloc_map_.end()) {
      COGADB_FATAL_ERROR("Deallocated Memory Block not in AllocationMap!", "");
    }
    // delete entry from map
    alloc_map_.erase(it);
    delete[] memory_block;
  }
  typedef std::map<T*, size_t> AllocationMap;
  AllocationMap alloc_map_;
};

#ifdef ENABLE_GPU_ACCELERATION
template <typename T>
class GPUMemoryAllocator : public MemoryAllocator<T> {
 public:
  GPUMemoryAllocator(hype::ProcessingDeviceMemoryID mem_id, int gpu_id)
      : MemoryAllocator<T>(mem_id, "GPU Allocator"),
        alloc_map_(),
        gpu_id_(gpu_id) {}

  T* allocate(size_t number_of_elements) {
    T* ret;  // = new T[number_of_elements];

    cudaError_t err = cudaMalloc((void**)&ret, number_of_elements * sizeof(T));
    if (err == cudaSuccess) {
      alloc_map_.insert(std::make_pair(ret, number_of_elements));
      return ret;
    } else {
      // throw exception
      return NULL;
    }
  }

  T* reallocate(T* memory_block, size_t new_number_of_elements) {
    T* ret = NULL;
    typename AllocationMap::iterator it = alloc_map_.find(memory_block);
    if (memory_block) {
      if (it == alloc_map_.end()) {
        COGADB_FATAL_ERROR("Allocated Memory Block not in AllocationMap!", "");
      }
      // is the allocated chunk larger than what is requested?
      // if yes, then do nothing
      if (it->second >= new_number_of_elements) return memory_block;
    }
    // std::cout << "Called CUDA Realloc: " << (void*) memory_block << ": " <<
    // new_number_of_elements << std::endl;
    ret = allocate(new_number_of_elements);
    // in case memory block is not NULL, copy data to new buffer
    // in case memory block is NULL, return allocated new buffer
    if (memory_block && ret) {
      // copy objects to new buffer
      // thrust::copy(memory_block, memory_block+it->second, ret);
      typedef typename CopyFunctionFactory<T>::CopyFunctionPtr CopyFunctionPtr;
      CopyFunctionPtr func = CopyFunctionFactory<T>::getCopyFunction(
          this->getMemoryID(), this->getMemoryID());
      assert(func != NULL);
      bool func_ret = (*func)(ret, memory_block, it->second * sizeof(T));
      assert(func_ret == true);

      deallocate(memory_block);
    }
    return ret;
  }
  void deallocate(T*& memory_block) {
    typename AllocationMap::iterator it = alloc_map_.find(memory_block);
    if (it == alloc_map_.end()) {
      COGADB_FATAL_ERROR("Deallocated Memory Block not in AllocationMap!", "");
    }
    // delete entry from map
    alloc_map_.erase(it);
    // delete old memory block
    cudaFree(memory_block);
  }
  typedef std::map<T*, size_t> AllocationMap;
  AllocationMap alloc_map_;
  int gpu_id_;
};
#endif

template <typename T>
class DummyMemoryAllocator : public MemoryAllocator<T> {
 public:
  DummyMemoryAllocator(hype::ProcessingDeviceMemoryID mem_id)
      : MemoryAllocator<T>(mem_id, "Dummy Allocator") {}

  T* allocate(size_t number_of_elements) { return NULL; }
  T* reallocate(T* memory_block, size_t new_number_of_elements) { return NULL; }
  void deallocate(T*& memory_block) {}
};

template <typename T>
const typename MemoryAllocator<T>::MemoryAllocatorPtr
MemoryAllocator<T>::getMemoryAllocator(hype::ProcessingDeviceMemoryID mem_id) {
  typedef MemoryAllocator<T>::MemoryAllocatorPtr MemoryAllocatorPtr;
  MemoryAllocatorPtr allocator;
  if (mem_id == hype::PD_Memory_0) {
    // create CPU allocators, plain C allocators for elemtary types, and a c++
    // allocator for object types
    if (typeid(T) == typeid(std::string)) {
      allocator = MemoryAllocatorPtr(new CPUMemoryAllocator_CPP<T>(mem_id));
    } else {
      allocator = MemoryAllocatorPtr(new CPUMemoryAllocator_C<T>(mem_id));
    }
  } else if (mem_id == hype::PD_Memory_1) {
    // create GPU allocator for elemtary types, and a dummy allocator for object
    // types
    if (typeid(T) == typeid(std::string)) {
      allocator = MemoryAllocatorPtr(new DummyMemoryAllocator<T>(mem_id));
    } else {
#ifdef ENABLE_GPU_ACCELERATION
      // only allocate memory of GPU with id 0
      allocator = MemoryAllocatorPtr(new GPUMemoryAllocator<T>(mem_id, 0));
#else
      COGADB_FATAL_ERROR("Tried to create GPU memory allocator, "
                             << "but GPU acceleration disabled!",
                         "");
#endif
    }
  } else {
    COGADB_FATAL_ERROR("Unsupported Memory ID: " << mem_id, "");
  }
  return allocator;
}

#ifdef ENABLE_GPU_ACCELERATION
template <typename T>
typename MemoryAllocator<T>::MemoryAllocatorPtr getGPUMemoryAllocator_Cuda(
    hype::ProcessingDeviceMemoryID& mem_id) {
  static typename MemoryAllocator<T>::MemoryAllocatorPtr ptr;
  if (!ptr) {
    ptr = boost::make_shared<GPUMemoryAllocator<T> >(mem_id, getIDOfFirstGPU());
  }
  return ptr;
}
#else
template <typename T>
typename MemoryAllocator<T>::MemoryAllocatorPtr getGPUMemoryAllocator_Cuda(
    hype::ProcessingDeviceMemoryID& mem_id) {
  COGADB_FATAL_ERROR("Tried to create GPU memory allocator, "
                         << "but GPU acceleration disabled!",
                     "");
  typename MemoryAllocator<T>::MemoryAllocatorPtr ptr;
  return ptr;
}
#endif
template <typename T>
typename MemoryAllocator<T>::MemoryAllocatorPtr getCPUMemoryAllocator_CPP(
    hype::ProcessingDeviceMemoryID& mem_id) {
  static typename MemoryAllocator<T>::MemoryAllocatorPtr ptr;
  if (!ptr) {
    ptr = boost::make_shared<CPUMemoryAllocator_CPP<T> >(mem_id);
  }
  return ptr;
}

template <typename T>
typename MemoryAllocator<T>::MemoryAllocatorPtr getCPUMemoryAllocator_C(
    hype::ProcessingDeviceMemoryID& mem_id) {
  static typename MemoryAllocator<T>::MemoryAllocatorPtr ptr;
  if (!ptr) {
    ptr = boost::make_shared<CPUMemoryAllocator_C<T> >(mem_id);
  }
  return ptr;
}

template <typename T>
T* AllocationManager<T>::malloc(hype::ProcessingDeviceMemoryID mem_id,
                                size_t number_of_elements) {
  typedef typename MemoryAllocator<T>::MemoryAllocatorPtr MemoryAllocatorPtr;
  MemoryAllocatorPtr allocator;
  if (mem_id == hype::PD_Memory_0) {
    // create CPU allocators, plain C allocators for elemtary types, and a c++
    // allocator for object types
    if (typeid(T) == typeid(std::string)) {
      allocator = getCPUMemoryAllocator_CPP<T>(mem_id);
    } else {
      allocator = getCPUMemoryAllocator_C<T>(mem_id);
    }
  } else if (mem_id == hype::PD_Memory_1) {
    // create GPU allocator for elemtary types, and a dummy allocator for object
    // types
    if (typeid(T) == typeid(std::string)) {
      COGADB_FATAL_ERROR("no string support on gpu", "");
    } else {
      // only allocate memory of GPU with id 0
      allocator = getGPUMemoryAllocator_Cuda<T>(mem_id);
    }
  } else {
    COGADB_FATAL_ERROR("Unsupported Memory ID: " << mem_id, "");
  }
  return allocator->allocate(number_of_elements);
}

template <typename T>
void AllocationManager<T>::free(hype::ProcessingDeviceMemoryID mem_id,
                                T*& data) {
  typedef typename MemoryAllocator<T>::MemoryAllocatorPtr MemoryAllocatorPtr;
  MemoryAllocatorPtr allocator;
  if (mem_id == hype::PD_Memory_0) {
    // create CPU allocators, plain C allocators for elemtary types, and a c++
    // allocator for object types
    if (typeid(T) == typeid(std::string)) {
      allocator = getCPUMemoryAllocator_CPP<T>(mem_id);
    } else {
      allocator = getCPUMemoryAllocator_C<T>(mem_id);
    }
  } else if (mem_id == hype::PD_Memory_1) {
    // create GPU allocator for elemtary types, and a dummy allocator for object
    // types
    if (typeid(T) == typeid(std::string)) {
      COGADB_FATAL_ERROR("no string support on gpu", "");
    } else {
      // only allocate memory of GPU with id 0
      allocator = getGPUMemoryAllocator_Cuda<T>(mem_id);
    }
  } else {
    COGADB_FATAL_ERROR("Unsupported Memory ID: " << mem_id, "");
  }
  allocator->deallocate(data);
}

template <typename T>
T* AllocationManager<T>::realloc(hype::ProcessingDeviceMemoryID mem_id, T* data,
                                 size_t number_of_elements) {
  typedef typename MemoryAllocator<T>::MemoryAllocatorPtr MemoryAllocatorPtr;
  MemoryAllocatorPtr allocator;
  if (mem_id == hype::PD_Memory_0) {
    // create CPU allocators, plain C allocators for elemtary types, and a c++
    // allocator for object types
    if (typeid(T) == typeid(std::string)) {
      allocator = getCPUMemoryAllocator_CPP<T>(mem_id);
    } else {
      allocator = getCPUMemoryAllocator_C<T>(mem_id);
    }
  } else if (mem_id == hype::PD_Memory_1) {
    // create GPU allocator for elemtary types, and a dummy allocator for object
    // types
    if (typeid(T) == typeid(std::string)) {
      COGADB_FATAL_ERROR("no string support on gpu", "");
    } else {
      // only allocate memory of GPU with id 0
      allocator = getGPUMemoryAllocator_Cuda<T>(mem_id);
    }
  } else {
    COGADB_FATAL_ERROR("Unsupported Memory ID: " << mem_id, "");
  }
  return allocator->reallocate(data, number_of_elements);
}

COGADB_INSTANTIATE_STRUCT_TEMPLATE_FOR_SUPPORTED_TYPES(AllocationManager)
COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(MemoryAllocator)
COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(CPUMemoryAllocator_C)
COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(CPUMemoryAllocator_CPP)

}  // end namespace CoGaDB
