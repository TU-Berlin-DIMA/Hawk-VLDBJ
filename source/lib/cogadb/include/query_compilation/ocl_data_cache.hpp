#ifndef OCL_DATA_CACHE_HPP
#define OCL_DATA_CACHE_HPP

#include <core/global_definitions.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/thread.hpp>
#include <util/opencl_runtime.hpp>

namespace CoGaDB {

  class OCL_DataCache;
  typedef boost::shared_ptr<OCL_DataCache> OCL_DataCachePtr;

  class OCL_DataCaches;
  typedef boost::shared_ptr<OCL_DataCaches> OCL_DataCachesPtr;

  class OCL_DataCaches {
   public:
    cl_mem getMemBuffer(cl_device_id dev_id, const void* data,
                        size_t num_bytes);
    void uncacheMemoryArea(const void* data_begin, size_t num_bytes);
    static OCL_DataCaches& instance();

   private:
    OCL_DataCaches();
    OCL_DataCaches(const OCL_DataCaches&);
    typedef std::map<cl_device_id, OCL_DataCachePtr> Caches;
    Caches caches_;
  };

}  // end namespace CoGaDB

#endif  // OCL_DATA_CACHE_HPP
