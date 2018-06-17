
#include <boost/thread.hpp>
#include <core/data_dictionary.hpp>
#include <core/gpu_column_cache.hpp>
#include <core/runtime_configuration.hpp>
#include <lookup_table/join_index.hpp>
#include <statistics/statistics_manager.hpp>
#include <util/time_measurement.hpp>

namespace CoGaDB {
using namespace std;
boost::mutex global_gpu_cache_mutex;

GPU_Column_Cache::GPU_Column_Cache(unsigned int max_gpu_buffer_size_in_byte)
    : map_(),
      join_index_map_(),
      join_index_accesses_(),
      join_index_least_recently_accessed_(),
      column_accesses_(),
      column_least_recently_accessed_(),
      max_gpu_buffer_size_in_byte_(max_gpu_buffer_size_in_byte),
      caching_enabled_(true),
      pin_join_indexes_(false),
      pin_columns_(false) {}

GPU_Column_Cache& GPU_Column_Cache::instance() {
  static GPU_Column_Cache cache(
      getTotalGPUMemorySizeInByte() *
      0.5);  // use half of the total device memory as buffer for columns
  // static GPU_Column_Cache cache(getTotalGPUMemorySizeInByte()*0.4); //use
  // half of the total device memory as buffer for columns
  return cache;
}

void GPU_Column_Cache::printStatus(std::ostream& out) const throw() {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  out << "GPU_Column_Cache Status:" << endl;
  out << "GPU Buffer Size: " << max_gpu_buffer_size_in_byte_ / (1024 * 1024)
      << "MB" << endl;
  out << "Available GPU Buffer: "
      << this->getAvailableGPUBufferSize_internal() / (1024 * 1024) << "MB"
      << endl;
  out << "Total GPU Memory: " << getTotalGPUMemorySizeInByte() / (1024 * 1024)
      << "MB" << endl;
  out << "Free GPU Memory: " << getFreeGPUMemorySizeInByte() / (1024 * 1024)
      << "MB" << endl;
  out << "Cached Columns:" << endl;
  Map::const_iterator cit;
  for (cit = map_.begin(); cit != map_.end(); ++cit) {
    out << cit->second->getName() << " of Size "
        << double(cit->first->getSizeinBytes()) / (1024 * 1024) << "MB ("
        << cit->second->size()
        << " elements), currently in use: " << !cit->second.unique() << endl;
    if (this->column_accesses_.find(cit->first) != this->column_accesses_.end())
      out << "\tReferenced: "
          << this->column_accesses_.find(cit->first)->second;
    if (this->column_least_recently_accessed_.find(cit->first) !=
        this->column_least_recently_accessed_.end())
      out << "\tLast Used Timestamp: "
          << this->column_least_recently_accessed_.find(cit->first)->second
          << std::endl;
  }
  out << "Cached Join Indexes:" << endl;
  JoinIndexMap::const_iterator cit_indx;
  for (cit_indx = join_index_map_.begin(); cit_indx != join_index_map_.end();
       ++cit_indx) {
    out << toString(cit_indx->second->cpu_join_index) << ": "
        << getSizeInBytes(cit_indx->second->cpu_join_index) / (1024 * 1024)
        << "MB";
    if (this->join_index_accesses_.find(cit_indx->second->cpu_join_index) !=
        this->join_index_accesses_.end())
      out << "\tReferenced: "
          << this->join_index_accesses_.find(cit_indx->second->cpu_join_index)
                 ->second;
    if (this->join_index_least_recently_accessed_.find(
            cit_indx->second->cpu_join_index) !=
        this->join_index_least_recently_accessed_.end())
      out << "\tLast Used Timestamp: "
          << this->join_index_least_recently_accessed_
                 .find(cit_indx->second->cpu_join_index)
                 ->second
          << std::endl;
  }
}

//                bool isPersistentColumn(ColumnPtr column_ptr){
//
//                    std::list<std::pair<ColumnPtr,TablePtr> > columns =
//                    DataDictionary::instance().getColumnsforColumnName(column_ptr->getName());
//                    //we assume unique column names
//                    assert(columns.size()<=1);
//                    if(columns.size()==1){
//                        if(columns.front().first==column_ptr){
//                            return true;
//                        }else{
//                            return false;
//                        }
//                    }else{
//                        return false;
//                    }
//                }
//
//                bool isIntermediateResultColumn(ColumnPtr column_ptr){
//                    return !isPersistentColumn(column_ptr);
//                }

const gpu::GPU_Base_ColumnPtr GPU_Column_Cache::getGPUColumn(ColumnPtr col) {
  gpu::GPU_Base_ColumnPtr result;
  if (!col) return result;
  if (col->getSizeinBytes() > max_gpu_buffer_size_in_byte_) {
    return result;
  }
  // while(result==NULL){
  result = this->getGPUColumn_internal(col);
  // if(result) return result;
  // boost::this_thread::sleep(boost::posix_time::microseconds(1000));
  //}
  return result;
}
const gpu::GPU_JoinIndexPtr GPU_Column_Cache::getGPUJoinIndex(
    JoinIndexPtr join_index) {
  gpu::GPU_JoinIndexPtr result;
  if (!join_index) return result;
  if (getSizeInBytes(join_index) > max_gpu_buffer_size_in_byte_) {
    return result;
  }
  // while(result==NULL){
  result = this->getGPUJoinIndex_internal(join_index);
  // if(result) return result;
  // boost::this_thread::sleep(boost::posix_time::microseconds(1000));
  //}
  return result;
}

bool GPU_Column_Cache::isCached(ColumnPtr col) {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  if (!col) return false;
  Map::iterator it = map_.find(col);
  if (it != map_.end()) {
    return true;
  } else {
    return false;
  }
}

bool GPU_Column_Cache::isCached(JoinIndexPtr join_index) {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  if (!join_index) return false;
  JoinIndexMap::iterator it = join_index_map_.find(join_index);
  if (it != join_index_map_.end()) {
    return true;
  } else {
    return false;
  }
}

bool GPU_Column_Cache::isCachingEnabled() { return this->caching_enabled_; }

void GPU_Column_Cache::setCacheEnabledStatus(bool status) {
  this->caching_enabled_ = status;
}

void GPU_Column_Cache::pinColumnsOnGPU(bool value) {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  this->pin_columns_ = value;
}

void GPU_Column_Cache::pinJoinIndexesOnGPU(bool value) {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  this->pin_join_indexes_ = value;
}

bool GPU_Column_Cache::haveColumnsPinned() const {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  return this->pin_columns_;
}
bool GPU_Column_Cache::haveJoinIndexesPinned() const {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  return this->pin_join_indexes_;
}

size_t GPU_Column_Cache::getAvailableGPUBufferSize() const {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  return getAvailableGPUBufferSize_internal();
}

size_t GPU_Column_Cache::getGPUBufferSize() const {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  return this->max_gpu_buffer_size_in_byte_;
}

// do not call this function during query processing!
bool GPU_Column_Cache::setGPUBufferSizeInByte(size_t size_in_bytes) {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  if (size_in_bytes < getTotalGPUMemorySizeInByte()) {
    this->max_gpu_buffer_size_in_byte_ = size_in_bytes;
    return true;
  } else {
    return false;
  }
}

size_t GPU_Column_Cache::getAvailableGPUBufferSize_internal() const {
  //                    size_t gpu_used_bytes=0;
  //                    JoinIndexMap::const_iterator ji_it;
  //                    Map::const_iterator col_it;
  //                    for(col_it=map_.begin();col_it!=map_.end();++col_it){
  //                        gpu_used_bytes+=(*col_it).first->getSizeinBytes();
  //                    }
  //                    for(ji_it=join_index_map_.begin();ji_it!=join_index_map_.end();++ji_it){
  //                        gpu_used_bytes+=getSizeInBytes((*ji_it).second->cpu_join_index);
  //                    }
  //                    //assert(this->max_gpu_buffer_size_in_byte_>=gpu_used_bytes);
  //                    if(this->max_gpu_buffer_size_in_byte_<gpu_used_bytes)
  //                    return 0;
  //
  //                    return
  //                    this->max_gpu_buffer_size_in_byte_-gpu_used_bytes;
  size_t gpu_used_bytes = this->getUsedGPUBufferSize_internal();
  if (gpu_used_bytes > max_gpu_buffer_size_in_byte_) {
    return 0;
  } else {
    return this->max_gpu_buffer_size_in_byte_ - gpu_used_bytes;
  }
}

size_t GPU_Column_Cache::getUsedGPUBufferSize_internal() const {
  size_t gpu_used_bytes = 0;
  JoinIndexMap::const_iterator ji_it;
  Map::const_iterator col_it;
  for (col_it = map_.begin(); col_it != map_.end(); ++col_it) {
    gpu_used_bytes += (*col_it).first->getSizeinBytes();
  }
  for (ji_it = join_index_map_.begin(); ji_it != join_index_map_.end();
       ++ji_it) {
    gpu_used_bytes += getSizeInBytes((*ji_it).second->cpu_join_index);
  }
  return gpu_used_bytes;
}

bool GPU_Column_Cache::cleanupGPUMemory(size_t number_of_bytes_to_cleanup) {
  size_t initial_used_gpu_memory_in_bytes =
      getTotalGPUMemorySizeInByte() - getFreeGPUMemorySizeInByte();

  if (number_of_bytes_to_cleanup > getTotalGPUMemorySizeInByte()) {
    COGADB_ERROR(
        "number_of_bytes_to_cleanup is greater than total memory capacity of "
        "GPU!",
        "");
    // return true;
  }
  // assert(number_of_bytes_to_cleanup<getTotalGPUMemorySizeInByte());
  ////assert(number_of_bytes_to_cleanup<=max_gpu_buffer_size_in_byte_);
  ////assert(number_of_bytes_to_cleanup<=initial_used_gpu_memory_in_bytes);

  if (number_of_bytes_to_cleanup > initial_used_gpu_memory_in_bytes) {
    // ok, we have not yet a central allocation/deallocation mechanism for
    // device memory
    // that means, during the time we computed the number_of_bytes to cleanup
    // a GPU operator might terminate, leaving initial_used_gpu_memory_in_bytes
    // smaller than
    // the number_of_bytes_to_cleanup.
    // for now, we will return here, because this is a very rare corner case
    COGADB_WARNING(
        "number_of_bytes_to_cleanup is greater than "
        "initial_used_gpu_memory_in_bytes!",
        "");
    return true;
  }

  if (!this->pin_columns_) {
    //                        while(!map_.empty() &&
    //                        (double(initial_used_gpu_memory_in_bytes)-(getTotalGPUMemorySizeInByte()-getFreeGPUMemorySizeInByte()))
    //                        <= number_of_bytes_to_cleanup){
    //                            Map::iterator cit;
    //                            Map::iterator
    //                            cit_col_with_largest_memory=map_.begin();
    //
    //                            for(cit=map_.begin();cit!=map_.end();++cit){
    //                                    if(cit->first->getSizeinBytes()>cit_col_with_largest_memory->first->getSizeinBytes()){
    //                                        cit_col_with_largest_memory=cit;
    //                                    }
    //                            }
    //                            //delete largest column
    //                            //cit_col_with_largest_memory->first->getSizeinBytes();
    //                            std::cout << "Remove Column '" <<
    //                            cit_col_with_largest_memory->first->getName()
    //                            << "' from GPU Cache" << std::endl;
    //                            map_.erase(cit_col_with_largest_memory);
    //
    //                        }
    GPUBufferManagementStrategy buffer_strategy =
        RuntimeConfiguration::instance().getGPUBufferManagementStrategy();
    if (buffer_strategy == LEAST_FREQUENTLY_USED) {
      while (!map_.empty() &&
             initial_used_gpu_memory_in_bytes -
                     (getTotalGPUMemorySizeInByte() -
                      getFreeGPUMemorySizeInByte()) >=
                 number_of_bytes_to_cleanup) {
        Map::iterator cit;
        size_t least_number_of_accesses = std::numeric_limits<size_t>::max();
        Map::iterator cit_frequently_used_column = map_.begin();
        // JoinIndexAccesses::iterator jia_it = join_index_accesses_.begin();
        for (cit = map_.begin(); cit != map_.end(); ++cit) {
          if (this->column_accesses_[cit->first] < least_number_of_accesses) {
            cit_frequently_used_column = cit;
            least_number_of_accesses = this->column_accesses_[cit->first];
          }
        }
        // delete largest join index
        std::cout << "Remove Column '"
                  << cit_frequently_used_column->first->getName()
                  << "' from GPU Cache" << std::endl;
        map_.erase(cit_frequently_used_column);
      }
    } else if (buffer_strategy == LEAST_RECENTLY_USED) {
      while (!map_.empty() &&
             initial_used_gpu_memory_in_bytes -
                     (getTotalGPUMemorySizeInByte() -
                      getFreeGPUMemorySizeInByte()) >=
                 number_of_bytes_to_cleanup) {
        Map::iterator cit;
        uint64_t least_recent_access = std::numeric_limits<uint64_t>::max();
        Map::iterator cit_least_recent_accessed_column = map_.begin();
        // JoinIndexAccesses::iterator jia_it = join_index_accesses_.begin();
        for (cit = map_.begin(); cit != map_.end(); ++cit) {
          if (this->column_least_recently_accessed_[cit->first] <
              least_recent_access) {
            cit_least_recent_accessed_column = cit;
            least_recent_access =
                this->column_least_recently_accessed_[cit->first];
          }
        }
        // delete largest join index
        std::cout << "Remove Column '"
                  << cit_least_recent_accessed_column->first->getName()
                  << "' from GPU Cache" << std::endl;
        map_.erase(cit_least_recent_accessed_column);
      }
    }
  }
  if (!this->pin_join_indexes_) {
    //                        while(!join_index_map_.empty() &&
    //                        initial_used_gpu_memory_in_bytes-(getTotalGPUMemorySizeInByte()-getFreeGPUMemorySizeInByte())
    //                        >= number_of_bytes_to_cleanup){
    //                            JoinIndexMap::iterator cit_indx;
    //                            JoinIndexMap::iterator
    //                            cit_join_index_with_largest_memory=join_index_map_.begin();
    //                            for(cit_indx=join_index_map_.begin();cit_indx!=join_index_map_.end();++cit_indx){
    //                                    if(getSizeInBytes(cit_indx->second->cpu_join_index)>
    //                                    getSizeInBytes(cit_join_index_with_largest_memory->second->cpu_join_index)){
    //                                        cit_join_index_with_largest_memory=cit_indx;
    //                                    }
    //                            }
    //                            //delete largest join index
    //                            std::cout << "Remove Join Index '" <<
    //                            toString(cit_join_index_with_largest_memory->first)
    //                            << "' from GPU Cache" << std::endl;
    //                            join_index_map_.erase(cit_join_index_with_largest_memory);
    //                        }
    GPUBufferManagementStrategy buffer_strategy =
        RuntimeConfiguration::instance().getGPUBufferManagementStrategy();
    if (buffer_strategy == LEAST_FREQUENTLY_USED) {
      while (!join_index_map_.empty() &&
             initial_used_gpu_memory_in_bytes -
                     (getTotalGPUMemorySizeInByte() -
                      getFreeGPUMemorySizeInByte()) >=
                 number_of_bytes_to_cleanup) {
        JoinIndexMap::iterator cit_indx;
        size_t least_number_of_accesses = std::numeric_limits<size_t>::max();
        JoinIndexMap::iterator cit_frequently_used_join_index =
            join_index_map_.begin();
        // JoinIndexAccesses::iterator jia_it = join_index_accesses_.begin();
        for (cit_indx = join_index_map_.begin();
             cit_indx != join_index_map_.end(); ++cit_indx) {
          if (this->join_index_accesses_[cit_indx->first] <
              least_number_of_accesses) {
            cit_frequently_used_join_index = cit_indx;
            least_number_of_accesses =
                this->join_index_accesses_[cit_indx->first];
          }
        }
        // delete largest join index
        std::cout << "Remove Join Index '"
                  << toString(cit_frequently_used_join_index->first)
                  << "' from GPU Cache" << std::endl;
        join_index_map_.erase(cit_frequently_used_join_index);
      }
    } else if (buffer_strategy == LEAST_RECENTLY_USED) {
      while (!join_index_map_.empty() &&
             initial_used_gpu_memory_in_bytes -
                     (getTotalGPUMemorySizeInByte() -
                      getFreeGPUMemorySizeInByte()) >=
                 number_of_bytes_to_cleanup) {
        JoinIndexMap::iterator cit_indx;
        uint64_t least_recent_access = std::numeric_limits<uint64_t>::max();
        JoinIndexMap::iterator cit_least_recent_accessed_join_index =
            join_index_map_.begin();
        // JoinIndexAccesses::iterator jia_it = join_index_accesses_.begin();
        for (cit_indx = join_index_map_.begin();
             cit_indx != join_index_map_.end(); ++cit_indx) {
          if (this->join_index_least_recently_accessed_[cit_indx->first] <
              least_recent_access) {
            cit_least_recent_accessed_join_index = cit_indx;
            least_recent_access =
                this->join_index_least_recently_accessed_[cit_indx->first];
          }
        }
        // delete largest join index
        std::cout << "Remove Join Index '"
                  << toString(cit_least_recent_accessed_join_index->first)
                  << "' from GPU Cache" << std::endl;
        join_index_map_.erase(cit_least_recent_accessed_join_index);
      }
    } else {
      COGADB_FATAL_ERROR("Invalid GPU Buffer Management Strategy!", "");
    }
  }
  //                    while(!join_index_map_.empty() &&
  //                    initial_used_gpu_memory_in_bytes-(getTotalGPUMemorySizeInByte()-getFreeGPUMemorySizeInByte())
  //                    >= number_of_bytes_to_cleanup){
  //                        JoinIndexMap::iterator cit_indx;
  //			JoinIndexMap::iterator
  // cit_join_index_with_largest_memory=join_index_map_.begin();
  //                        for(cit_indx=join_index_map_.begin();cit_indx!=join_index_map_.end();++cit_indx){
  //				if(getSizeInBytes(cit_indx->second->cpu_join_index)>
  // getSizeInBytes(cit_join_index_with_largest_memory->second->cpu_join_index)){
  //                                    cit_join_index_with_largest_memory=cit_indx;
  //                                }
  //                        }
  //                        //delete largest join index
  //                        std::cout << "Remove Join Index '" <<
  //                        toString(cit_join_index_with_largest_memory->first)
  //                        << "' from GPU Cache" << std::endl;
  //                        join_index_map_.erase(cit_join_index_with_largest_memory);
  //                    }
  // check whether we were successful
  if (initial_used_gpu_memory_in_bytes -
          (getTotalGPUMemorySizeInByte() - getFreeGPUMemorySizeInByte()) <=
      number_of_bytes_to_cleanup) {
    std::cout << "Successfully cleaned up "
              << double(number_of_bytes_to_cleanup) / (1024 * 1024)
              << "MB from GPU Cache" << std::endl;
    return true;
  } else {
    std::cout << "Failed to cleaned up "
              << double(number_of_bytes_to_cleanup) / (1024 * 1024)
              << "MB from GPU Cache!" << std::endl;
    return false;
  }
}

const gpu::GPU_Base_ColumnPtr GPU_Column_Cache::getGPUColumn_internal(
    ColumnPtr column_ptr) {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);

  column_accesses_[column_ptr]++;
  column_least_recently_accessed_[column_ptr] = getTimestamp();

  // StatisticsManager::instance().addToValue("GPU_CACHE_ACCESS");
  StatisticsManager::instance().addToValue("GPU_COLUMN_CACHE_ACCESS", 1);

  gpu::GPU_Base_ColumnPtr result_ptr;
  if (!column_ptr) return gpu::GPU_Base_ColumnPtr();
  // do not cache if caching is disabled or column is intermediate result
  if (!caching_enabled_ || isIntermediateResultColumn(column_ptr)) {
    result_ptr = gpu::copy_column_host_to_device(column_ptr);
    StatisticsManager::instance().addToValue("GPU_COLUMN_CACHE_MISS", 1);
    return result_ptr;
  }

  Map::iterator it = map_.find(column_ptr);
  if (it == map_.end()) {
    if (!quiet) cout << "Cache Miss: " << column_ptr->getName() << endl;
    StatisticsManager::instance().addToValue("GPU_COLUMN_CACHE_MISS", 1);
    // unsigned int
    // used_memory_in_bytes=getTotalGPUMemorySizeInByte()-getFreeGPUMemorySizeInByte();
    size_t used_memory_in_bytes = this->getUsedGPUBufferSize_internal();

    // can we possibly fit the requested column in the GPU Cache?
    if (column_ptr->getSizeinBytes() > max_gpu_buffer_size_in_byte_)
      return gpu::GPU_Base_ColumnPtr();

    if (column_ptr->getSizeinBytes() + used_memory_in_bytes >
        max_gpu_buffer_size_in_byte_) {
      cout << "Warning: GPU_Column_Cache::getGPUColumn(): Unsufficient Device "
              "Memory"
           << endl;
      size_t number_of_bytes_to_cleanup =
          (column_ptr->getSizeinBytes() + used_memory_in_bytes) -
          max_gpu_buffer_size_in_byte_;
      // try to cleanup the Cache so we have enough available memory on GPU
      // if it does not work, return a null pointer, otherwise, continue with
      // the function
      if (!cleanupGPUMemory(number_of_bytes_to_cleanup)) {
        return gpu::GPU_Base_ColumnPtr();
      }
      //					//apply column replacement
      // strategy!
      //                                        //simple greedy strategy that
      //                                        deletes the largest column from
      //                                        the cache
      //                                        Map::iterator cit;
      //                                        Map::iterator
      //                                        cit_column_with_largest_size=map_.begin();
      //					for(cit=map_.begin();cit!=map_.end();++cit){
      //                                            //consider only unused
      //                                            columns for removal from
      //                                            cache
      //                                            if(cit_column_with_largest_size->second->size()<cit->second->size()
      //                                            &&
      //                                            cit_column_with_largest_size->second.unique()){
      //						cit_column_with_largest_size=cit->second->size();
      //                                            }
      //					}
      //                                        //delete column by pointer reset
      //                                        cit_column_with_largest_size->second.reset();

      //					return
      // gpu::GPU_Base_ColumnPtr();
    }

    result_ptr = gpu::copy_column_host_to_device(column_ptr);
    if (!result_ptr) {
      cout << "Warning: GPU_Column_Cache::getGPUColumn(): Unsufficient Device "
              "Memory"
           << endl;
      return gpu::GPU_Base_ColumnPtr();
    }
    // if(caching_enabled_){
    map_.insert(std::make_pair(column_ptr, result_ptr));
    //}
    return result_ptr;
  } else {
    if (!quiet) cout << "Cache Hit: " << column_ptr->getName() << endl;
    StatisticsManager::instance().addToValue("GPU_COLUMN_CACHE_HIT", 1);
    return it->second;  // return pointer to GPU memory
  }
}

const gpu::GPU_JoinIndexPtr GPU_Column_Cache::getGPUJoinIndex_internal(
    JoinIndexPtr join_index) {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);

  join_index_accesses_[join_index]++;
  join_index_least_recently_accessed_[join_index] = getTimestamp();
  StatisticsManager::instance().addToValue("GPU_JOIN_INDEX_CACHE_ACCESS", 1);
  gpu::GPU_JoinIndexPtr result_ptr;
  if (!join_index) return gpu::GPU_JoinIndexPtr();
  if (!caching_enabled_) {
    StatisticsManager::instance().addToValue("GPU_JOIN_INDEX_CACHE_MISS", 1);
    result_ptr = gpu::copy_JoinIndex_host_to_device(join_index);
    return result_ptr;
  }

  JoinIndexMap::iterator it = join_index_map_.find(join_index);
  if (it == join_index_map_.end()) {
    if (!quiet) cout << "Cache Miss: JOIN Index" << endl;
    StatisticsManager::instance().addToValue("GPU_JOIN_INDEX_CACHE_MISS", 1);
    // unsigned int
    // used_memory_in_bytes=getTotalGPUMemorySizeInByte()-getFreeGPUMemorySizeInByte();
    size_t used_memory_in_bytes = this->getUsedGPUBufferSize_internal();

    size_t size_of_joinindex = getSizeInBytes(
        join_index);  //(join_index->first->getPositionList()->size()+join_index->second->getPositionList()->size())*sizeof(PositionList::value_type);
    if (size_of_joinindex + used_memory_in_bytes >
        max_gpu_buffer_size_in_byte_) {
      COGADB_WARNING(
          "GPU_Column_Cache::getGPUJoinIndex(): Unsufficient Device Memory"
              << std::endl
              << "\tSizeof JoinIndex: " << size_of_joinindex / (1024 * 1024)
              << "MB" << std::endl
              << "\tMemory in Use: " << used_memory_in_bytes / (1024 * 1024)
              << "MB" << std::endl
              << "\tGPU Buffer Size: "
              << max_gpu_buffer_size_in_byte_ / (1024 * 1024) << "MB",
          "");
      size_t number_of_bytes_to_cleanup =
          (size_of_joinindex + used_memory_in_bytes) -
          max_gpu_buffer_size_in_byte_;
      // try to cleanup the cache so we have enough available memory on GPU
      // if it does not work, return a null pointer, otherwise, continue with
      // the function
      if (!cleanupGPUMemory(number_of_bytes_to_cleanup)) {
        return gpu::GPU_JoinIndexPtr();
      }
      used_memory_in_bytes = this->getUsedGPUBufferSize_internal();
      // apply column replacement strategy!
      //					return gpu::GPU_JoinIndexPtr();
    }

    // if we now have enough memory, copy the Join Index to the GPU
    if (size_of_joinindex + used_memory_in_bytes <
        max_gpu_buffer_size_in_byte_) {
      result_ptr = gpu::copy_JoinIndex_host_to_device(join_index);
    }

    if (!result_ptr) {
      COGADB_WARNING(
          "GPU_Column_Cache::getGPUJoinIndex(): Unsufficient Device Memory",
          "");
      return gpu::GPU_JoinIndexPtr();
    }
    // if(caching_enabled_){
    join_index_map_.insert(std::make_pair(join_index, result_ptr));
    //}
    return result_ptr;
  } else {
    if (!quiet) cout << "Cache Hit: JOIN INDEX" << endl;
    StatisticsManager::instance().addToValue("GPU_JOIN_INDEX_CACHE_HIT", 1);
    return it->second;  // return pointer to GPU memory
  }

  return gpu::GPU_JoinIndexPtr();
}

bool GPU_Column_Cache::removeGPUColumn(gpu::GPU_Base_ColumnPtr dev_col) {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  Map::iterator it;
  for (it = map_.begin(); it != map_.end(); ++it) {
    if (it->second == dev_col) {
      map_.erase(it);
      return true;
    }
  }
  return true;
}

bool GPU_Column_Cache::removeGPUJoinIndex(gpu::GPU_JoinIndexPtr join_index) {
  boost::lock_guard<boost::mutex> lock(global_gpu_cache_mutex);
  JoinIndexMap::iterator it;
  for (it = join_index_map_.begin(); it != join_index_map_.end(); ++it) {
    if (it->second == join_index) {
      join_index_map_.erase(it);
      return true;
    }
  }
  return false;
}

}  // end namespace CoGaDB
