
#include <core/global_definitions.hpp>
#include <query_compilation/ocl_data_cache.hpp>

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
//#include <boost/thread.hpp>
//#include <util/opencl_runtime.hpp>
#include <util/time_measurement.hpp>

#include <query_compilation/ocl_api.hpp>
#include <sstream>

namespace CoGaDB {

//#define OCL_DATA_CACHE_TRACE_CACHE_OPERATIONS

class CachedChunk;
typedef boost::shared_ptr<CachedChunk> CachedChunkPtr;

class OCL_DataCache;
typedef boost::shared_ptr<OCL_DataCache> OCL_DataCachePtr;

class OCL_DataCaches;
typedef boost::shared_ptr<OCL_DataCaches> OCL_DataCachesPtr;

class CachedChunk {
 public:
  CachedChunk(uint64_t num_bytes, cl_mem data_buf);
  ~CachedChunk();
  uint64_t num_bytes_;
  cl_mem data_buf_;
  uint64_t last_access_;
  uint64_t num_access_;
  uint32_t clock_counter_;

 private:
  CachedChunk(const CachedChunk&);
  CachedChunk& operator=(const CachedChunk&);
};

CachedChunk::CachedChunk(uint64_t num_bytes, cl_mem data_buf)
    : num_bytes_(num_bytes),
      data_buf_(data_buf),
      last_access_(getTimestamp()),
      num_access_(0),
      clock_counter_(1) {}

CachedChunk::~CachedChunk() {
#ifdef OCL_DATA_CACHE_TRACE_CACHE_OPERATIONS
  std::cout << "Release MemObject: " << data_buf_ << std::endl;
#endif
  if (data_buf_) clReleaseMemObject(data_buf_);
}

class OCL_DataCache {
 public:
  OCL_DataCache(cl_device_id dev_id, size_t buffer_size_in_byte);
  cl_mem getMemBuffer(const void* data, size_t num_bytes);
  void uncacheMemoryArea(const void* data_begin, size_t num_bytes);

 private:
  typedef std::map<const void*, CachedChunkPtr> MemoryMap;
  void evictMemory(size_t num_bytes);
  MemoryMap::iterator freeMemory(const MemoryMap::const_iterator& itr);
  void checkMemMapEmpty(uint64_t required_bytes);
  MemoryMap mem_map_;
  cl_device_id dev_id_;
  size_t buffer_size_;     /* in byte */
  size_t max_buffer_size_; /* in byte */
  boost::mutex mutex_;
  MemoryMap::iterator clock_pointer_;
};

OCL_DataCache::OCL_DataCache(cl_device_id dev_id, size_t buffer_size_in_byte)
    : mem_map_(),
      dev_id_(dev_id),
      buffer_size_(0),
      max_buffer_size_(buffer_size_in_byte),
      mutex_(),
      clock_pointer_(mem_map_.end()) {
#ifdef OCL_DATA_CACHE_TRACE_CACHE_OPERATIONS
  std::cout << "Created Cache of size "
            << double(max_buffer_size_) / (1024 * 1024)
            << "MB for device: " << std::endl;
  ocl_print_device(dev_id_);
#endif
}

cl_mem OCL_DataCache::getMemBuffer(const void* data, size_t num_bytes) {
  boost::lock_guard<boost::mutex> lock(mutex_);
#ifdef OCL_DATA_CACHE_TRACE_CACHE_OPERATIONS
  std::cout << "Request Memory Chunk: " << data << " of size " << num_bytes
            << "bytes" << std::endl;
#endif
  if (num_bytes > max_buffer_size_) {
    COGADB_FATAL_ERROR(
        "Request " << num_bytes
                   << "bytes memory, but overall data cache size is just "
                   << max_buffer_size_ << "bytes!",
        "");
  }
  MemoryMap::iterator it = mem_map_.find(data);
  if (it != mem_map_.end()) {
    if (num_bytes == it->second->num_bytes_) {
#ifdef OCL_DATA_CACHE_TRACE_CACHE_OPERATIONS
      std::cout << "Hit: cl_mem: " << it->second->data_buf_ << std::endl;
#endif
      it->second->last_access_ = getTimestamp();
      it->second->num_access_++;
      it->second->clock_counter_ = 1;
      return it->second->data_buf_;
    } else {
      COGADB_FATAL_ERROR("Requested Cached Memory Chunk with different size!",
                         "");
      return NULL;
    }
  } else {
#ifdef OCL_DATA_CACHE_TRACE_CACHE_OPERATIONS
    std::cout << "Miss" << std::endl;
#endif
    if (buffer_size_ + num_bytes > max_buffer_size_) {
      uint64_t free_memory = max_buffer_size_ - buffer_size_;
      uint64_t required_memory = num_bytes - free_memory;
      evictMemory(required_memory);
    }
    /* copy data to device and update cache */
    cl_int err = CL_SUCCESS;
    OCL_Runtime::OCL_DeviceStructuresPtr dev_structs =
        OCL_Runtime::instance().getDeviceStructures(dev_id_);
    cl_mem cl_input_mem = clCreateBuffer(dev_structs->context, CL_MEM_READ_ONLY,
                                         num_bytes, NULL, &err);
    if (err != CL_SUCCESS) {
      COGADB_FATAL_ERROR("Create buffer failed!", "");
      return NULL;
    }

    err = oclCopyHostToDevice(
        data, num_bytes, cl_input_mem, dev_structs->context,
        dev_structs->copy_host_to_device_queues[0], true, NULL);

    if (err != CL_SUCCESS) {
      COGADB_FATAL_ERROR("Copy Host to Device failed!", "");
      return NULL;
    }

    CachedChunkPtr chunk(new CachedChunk(num_bytes, cl_input_mem));
    std::pair<MemoryMap::iterator, bool> ret =
        mem_map_.insert(std::make_pair(data, chunk));
    assert(ret.second);
    if (clock_pointer_ == mem_map_.end()) {
      clock_pointer_ = ret.first;
    }
    buffer_size_ += num_bytes;
#ifdef OCL_DATA_CACHE_TRACE_CACHE_OPERATIONS
    std::cout << "OCL Mem: " << (void*)cl_input_mem << std::endl;
#endif
    return cl_input_mem;
  }
}

void OCL_DataCache::checkMemMapEmpty(uint64_t required_bytes) {
  if (mem_map_.empty()) {
    COGADB_FATAL_ERROR(
        "OCL_DataCache: Nothing cached but not enough memory! (num_bytes "
        "required "
            << required_bytes << ")",
        "");
  }
}

void OCL_DataCache::evictMemory(size_t num_bytes) {
  assert(num_bytes <= max_buffer_size_);
  assert(num_bytes <= buffer_size_);

  size_t cleanedup_memory_bytes = 0;
  while (cleanedup_memory_bytes < num_bytes) {
    checkMemMapEmpty(num_bytes - cleanedup_memory_bytes);

    if (clock_pointer_->second->clock_counter_ == 1) {
      clock_pointer_->second->clock_counter_ = 0;
    } else {
      /* evict mememory region */
      cleanedup_memory_bytes += clock_pointer_->second->num_bytes_;
      std::cout << "Evict cached data for memory area: "
                << "begin=" << clock_pointer_->first << "end="
                << static_cast<const void*>(
                       static_cast<const uint8_t*>(clock_pointer_->first) +
                       clock_pointer_->second->num_bytes_)
                << std::endl;

      clock_pointer_ = freeMemory(clock_pointer_);

      if (clock_pointer_ == mem_map_.end()) {
        clock_pointer_ = mem_map_.begin();
      }
    }
  }
}

void OCL_DataCache::uncacheMemoryArea(const void* data_begin,
                                      size_t num_bytes) {
  boost::lock_guard<boost::mutex> lock(mutex_);

  const uint8_t* data_end = static_cast<const uint8_t*>(data_begin) + num_bytes;

#ifdef OCL_DATA_CACHE_TRACE_CACHE_OPERATIONS
  if (begin_it != end_it) {
    std::cout << "Uncache memory area: begin=" << data_begin
              << " end=" << (const void*)data_end << std::endl;
  }
#endif

  auto begin = mem_map_.lower_bound(data_begin);
  auto end = mem_map_.upper_bound(static_cast<const void*>(data_end));

  for (auto itr = begin; itr != end;) {
    if (clock_pointer_ == itr) {
      clock_pointer_ = itr = freeMemory(itr);
    } else {
      itr = freeMemory(itr);
    }
  }

  if (!mem_map_.empty() && clock_pointer_ == mem_map_.end()) {
    clock_pointer_ = mem_map_.begin();
  }
}

OCL_DataCache::MemoryMap::iterator OCL_DataCache::freeMemory(
    const MemoryMap::const_iterator& itr) {
  if (itr->second->num_bytes_ > buffer_size_) {
    buffer_size_ = 0;
  } else {
    buffer_size_ -= itr->second->num_bytes_;
  }

  return mem_map_.erase(itr);
}

OCL_DataCaches::OCL_DataCaches() : caches_() {
  const auto cache_size_of_memory = 0.7;
  std::vector<cl_device_id> devices =
      OCL_Runtime::instance().getDevicesWithDedicatedMemory();

  for (auto device : devices) {
    auto global_memory = boost::compute::device(device).global_memory_size();

    auto cache_size =
        static_cast<uint64_t>(cache_size_of_memory * global_memory);
    OCL_DataCachePtr data_cache =
        boost::make_shared<OCL_DataCache>(device, cache_size);
    caches_.insert(std::make_pair(device, data_cache));
  }
}

cl_mem OCL_DataCaches::getMemBuffer(cl_device_id dev_id, const void* data,
                                    size_t num_bytes) {
  Caches::const_iterator it = caches_.find(dev_id);
  if (it != caches_.end()) {
    return it->second->getMemBuffer(data, num_bytes);
  }
  COGADB_FATAL_ERROR("Unknown opencl device: " << dev_id, "");
  return NULL;
}

void OCL_DataCaches::uncacheMemoryArea(const void* data_begin,
                                       size_t num_bytes) {
  Caches::const_iterator it;
  for (it = caches_.begin(); it != caches_.end(); ++it) {
    it->second->uncacheMemoryArea(data_begin, num_bytes);
  }
}

OCL_DataCaches& OCL_DataCaches::instance() {
  static OCL_DataCaches caches;
  return caches;
}

}  // end namespace CoGaDB
