
//#include <boost/thread.hpp>

#include <core/column.hpp>
#include <core/data_dictionary.hpp>
#include <core/processor_data_cache.hpp>

#include <core/runtime_configuration.hpp>
#include <lookup_table/join_index.hpp>
#include <statistics/statistics_manager.hpp>
#include <util/time_measurement.hpp>
//#include <gpu/gpu_algorithms.hpp>

#include <util/hardware_detector.hpp>
#include <util/utility_functions.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem.hpp>
#include <core/variable_manager.hpp>
#include <persistence/storage_manager.hpp>

namespace CoGaDB {
using namespace std;
//		boost::mutex cache_mutex_;

DataCache::DataCache(size_t max_buffer_size_in_byte,
                     const hype::ProcessingDeviceMemoryID& mem_id)
    : map_(),
      join_index_map_(),
      join_index_accesses_(),
      join_index_least_recently_accessed_(),
      column_accesses_(),
      column_least_recently_accessed_(),
      max_buffer_size_in_byte_(max_buffer_size_in_byte),
      caching_enabled_(true),
      pin_join_indexes_(false),
      pin_columns_(false),
      mem_id_(mem_id),
      cache_mutex_(),
      data_placement_thread_(new boost::thread(
          boost::bind(&DataCache::data_placement_thread, this))) {}

DataCache::~DataCache() { stop_data_placement_thread(); }

//		DataCache& DataCache::instance(){
//			static DataCache
// cache(HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_)*0.5);
////use half of the total device memory as buffer for columns
//                        //static DataCache
//                        cache(HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_)*0.4);
//                        //use half of the total device memory as buffer for
//                        columns
//			return cache;
//		}

void DataCache::printStatus(std::ostream& out) const throw() {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  out << "DataCache Status:" << endl;
  out << " MemoryID: " << (int)mem_id_ << endl;
  out << " Buffer Size: " << max_buffer_size_in_byte_ / (1024 * 1024) << "MB"
      << endl;
  out << "Available  Buffer: "
      << this->getAvailableBufferSize_internal() / (1024 * 1024) << "MB"
      << endl;
  out << "Total  Memory: "
      << HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_) /
             (1024 * 1024)
      << "MB" << endl;
  out << "Free  Memory: "
      << HardwareDetector::instance().getFreeMemorySizeInByte(mem_id_) /
             (1024 * 1024)
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
    out << toString(cit_indx->second) << ": "
        << getSizeInBytes(cit_indx->second) / (1024 * 1024) << "MB"
        << std::endl;
    if (this->join_index_accesses_.find(cit_indx->second) !=
        this->join_index_accesses_.end())
      out << "\tReferenced: "
          << this->join_index_accesses_.find(cit_indx->second)->second;
    if (this->join_index_least_recently_accessed_.find(cit_indx->second) !=
        this->join_index_least_recently_accessed_.end())
      out << "\tLast Used Timestamp: "
          << this->join_index_least_recently_accessed_.find(cit_indx->second)
                 ->second
          << std::endl;
  }
}

bool isPersistentColumn(ColumnPtr column_ptr) {
  // handle reverse join indexes uniform to other columns, but here we need to
  // know because we never cache intermediate results for longer than the
  // queries life time
  // but we do for indeces
  if (JoinIndexes::instance().isReverseJoinIndex(
          boost::dynamic_pointer_cast<PositionList>(column_ptr)))
    return true;

  std::list<std::pair<ColumnPtr, TablePtr> > columns =
      DataDictionary::instance().getColumnsforColumnName(column_ptr->getName());
  // we assume unique column names
  assert(columns.size() <= 1);
  if (columns.size() == 1) {
    if (columns.front().first == column_ptr) {
      //                std::cout << "ColumnPtr " << column_ptr.get() <<  " to "
      //                << column_ptr->getName() << " is Persistent Column!" <<
      //                std::endl;
      return true;
    } else {
      //                std::cout << "ColumnPtr " << column_ptr.get() <<  " to "
      //                << column_ptr->getName() << " is Intermediate Column!"
      //                << std::endl;
      return false;
    }
  } else {
    //            std::cout << "ColumnPtr " << column_ptr.get() <<  " to " <<
    //            column_ptr->getName() << " is Intermediate Column!" <<
    //            std::endl;
    return false;
  }
}

bool isIntermediateResultColumn(ColumnPtr column_ptr) {
  return !isPersistentColumn(column_ptr);
}

const ColumnPtr DataCache::getColumn(ColumnPtr col) {
  ColumnPtr result;
  if (!col) return result;
  if (col->getSizeinBytes() > max_buffer_size_in_byte_) {
    return result;
  }
  // while(result==NULL){
  result = this->getColumn_internal(col);
  // if(result) return result;
  // boost::this_thread::sleep(boost::posix_time::microseconds(1000));
  //}
  return result;
}

const ColumnPtr DataCache::getHostColumn(ColumnPtr col) {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  ColumnPtr result;
  if (!col) return result;
  assert(col->getMemoryID() == this->mem_id_);
  Map::const_iterator col_it;
  for (col_it = map_.begin(); col_it != map_.end(); ++col_it) {
    if ((*col_it).second == col) {
      return (*col_it).first;
    }
  }
  // not found
  return result;
}

const JoinIndexPtr DataCache::getJoinIndex(JoinIndexPtr join_index) {
  JoinIndexPtr result;
  if (!join_index) return result;
  if (getSizeInBytes(join_index) > max_buffer_size_in_byte_) {
    return result;
  }
  // while(result==NULL){
  result = this->getJoinIndex_internal(join_index);
  // if(result) return result;
  // boost::this_thread::sleep(boost::posix_time::microseconds(1000));
  //}
  return result;
}

bool DataCache::isCached(ColumnPtr col) {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  if (!col) return false;
  Map::iterator it = map_.find(col);
  if (it != map_.end()) {
    return true;
  } else {
    return false;
  }
}

bool DataCache::isCached(JoinIndexPtr join_index) {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  if (!join_index) return false;
  JoinIndexMap::iterator it = join_index_map_.find(join_index);
  if (it != join_index_map_.end()) {
    return true;
  } else {
    return false;
  }
}

bool DataCache::isCachingEnabled() { return this->caching_enabled_; }

void DataCache::setCacheEnabledStatus(bool status) {
  this->caching_enabled_ = status;
}

void DataCache::pinColumns(bool value) {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  this->pin_columns_ = value;
}

void DataCache::pinJoinIndexes(bool value) {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  this->pin_join_indexes_ = value;
}

bool DataCache::haveColumnsPinned() const {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  return this->pin_columns_;
}

bool DataCache::haveJoinIndexesPinned() const {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  return this->pin_join_indexes_;
}

size_t DataCache::getAvailableBufferSize() const {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  return getAvailableBufferSize_internal();
}

size_t DataCache::getBufferSize() const {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  return this->max_buffer_size_in_byte_;
}

// do not call this function during query processing!

bool DataCache::setBufferSizeInByte(size_t size_in_bytes) {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  if (size_in_bytes <
      HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_)) {
    this->max_buffer_size_in_byte_ = size_in_bytes;
    return true;
  } else {
    return false;
  }
}

size_t DataCache::getAvailableBufferSize_internal() const {
  size_t used_bytes = this->getUsedBufferSize_internal();
  if (used_bytes > max_buffer_size_in_byte_) {
    return 0;
  } else {
    return this->max_buffer_size_in_byte_ - used_bytes;
  }
}

size_t DataCache::getUsedBufferSize_internal() const {
  size_t used_bytes = 0;
  JoinIndexMap::const_iterator ji_it;
  Map::const_iterator col_it;
  for (col_it = map_.begin(); col_it != map_.end(); ++col_it) {
    used_bytes += (*col_it).first->getSizeinBytes();
  }
  for (ji_it = join_index_map_.begin(); ji_it != join_index_map_.end();
       ++ji_it) {
    used_bytes += getSizeInBytes((*ji_it).second);
  }
  return used_bytes;
}

bool DataCache::cleanupMemory(size_t number_of_bytes_to_cleanup) {
  size_t initial_used_memory_in_bytes =
      HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_) -
      HardwareDetector::instance().getFreeMemorySizeInByte(mem_id_);

  if (number_of_bytes_to_cleanup >
      HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_)) {
    COGADB_ERROR(
        "number_of_bytes_to_cleanup is greater than total memory capacity of !",
        "");
    // return true;
  }
  // assert(number_of_bytes_to_cleanup<HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_));
  ////assert(number_of_bytes_to_cleanup<=max_buffer_size_in_byte_);
  ////assert(number_of_bytes_to_cleanup<=initial_used_memory_in_bytes);

  if (number_of_bytes_to_cleanup > initial_used_memory_in_bytes) {
    // ok, we have not yet a central allocation/deallocation mechanism for
    // device memory
    // that means, during the time we computed the number_of_bytes to cleanup
    // a  operator might terminate, leaving initial_used_memory_in_bytes smaller
    // than
    // the number_of_bytes_to_cleanup.
    // for now, we will return here, because this is a very rare corner case
    COGADB_WARNING(
        "number_of_bytes_to_cleanup is greater than "
        "initial_used_memory_in_bytes!",
        "");
    return true;
  }

  if (!this->pin_columns_) {
    BufferManagementStrategy buffer_strategy =
        RuntimeConfiguration::instance().getBufferManagementStrategy();
    if (buffer_strategy == LEAST_FREQUENTLY_USED) {
      while (!map_.empty() && (number_of_bytes_to_cleanup >
                               this->getAvailableBufferSize_internal())) {
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
                  << "' from  Cache" << std::endl;
        map_.erase(cit_frequently_used_column);
        StatisticsManager::instance().addToValue("TOTAL_COLUMN_EVICTIONS", 1);
      }
    } else if (buffer_strategy == LEAST_RECENTLY_USED) {
      while (!map_.empty() && (number_of_bytes_to_cleanup >
                               this->getAvailableBufferSize_internal())) {
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
                  << "' from  Cache" << std::endl;
        map_.erase(cit_least_recent_accessed_column);
        StatisticsManager::instance().addToValue("TOTAL_COLUMN_EVICTIONS", 1);
      }
    }
  }
  if (!this->pin_join_indexes_) {
    BufferManagementStrategy buffer_strategy =
        RuntimeConfiguration::instance().getBufferManagementStrategy();
    if (buffer_strategy == LEAST_FREQUENTLY_USED) {
      while (!join_index_map_.empty() &&
             (number_of_bytes_to_cleanup >
              this->getAvailableBufferSize_internal())) {
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
                  << "' from  Cache" << std::endl;
        join_index_map_.erase(cit_frequently_used_join_index);
        StatisticsManager::instance().addToValue("TOTAL_JOIN_INDEX_EVICTIONS",
                                                 1);
      }
    } else if (buffer_strategy == LEAST_RECENTLY_USED) {
      while (!join_index_map_.empty() &&
             (number_of_bytes_to_cleanup >
              this->getAvailableBufferSize_internal())) {
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
                  << "' from  Cache" << std::endl;
        join_index_map_.erase(cit_least_recent_accessed_join_index);
        StatisticsManager::instance().addToValue("TOTAL_JOIN_INDEX_EVICTIONS",
                                                 1);
      }
    } else {
      COGADB_FATAL_ERROR("Invalid  Buffer Management Strategy!", "");
    }
  }

  // check whether we were successful
  // if (initial_used_memory_in_bytes -
  // (HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_) -
  // HardwareDetector::instance().getFreeMemorySizeInByte(mem_id_)) <=
  // number_of_bytes_to_cleanup) {
  if (number_of_bytes_to_cleanup < this->getAvailableBufferSize_internal()) {
    std::cout << "Successfully cleaned up "
              << double(number_of_bytes_to_cleanup) / (1024 * 1024)
              << "MB from  Cache" << std::endl;
    return true;
  } else {
    std::cout << "Failed to cleaned up "
              << double(number_of_bytes_to_cleanup) / (1024 * 1024)
              << "MB from  Cache!" << std::endl;
    return false;
  }
}

const ColumnPtr DataCache::getColumn_internal(ColumnPtr column_ptr) {
  if (!column_ptr) return ColumnPtr();
  boost::mutex::scoped_lock lock(cache_mutex_);
  lock.unlock();

  // StatisticsManager::instance().addToValue("CACHE_ACCESS");
  StatisticsManager::instance().addToValue("COLUMN_CACHE_ACCESS", 1);

  ColumnPtr result_ptr;
  // do not cache if caching is disabled or column is intermediate result
  if (!caching_enabled_ || isIntermediateResultColumn(column_ptr)) {
    //            cache_mutex_.unlock();
    result_ptr = copy(column_ptr, mem_id_);
    StatisticsManager::instance().addToValue("COLUMN_CACHE_MISS", 1);
    return result_ptr;
  }

  lock.lock();
  Map::iterator it = map_.find(column_ptr);
  if (it == map_.end()) {
    if (!quiet) cout << "Cache Miss: " << column_ptr->getName() << endl;
    StatisticsManager::instance().addToValue("COLUMN_CACHE_MISS", 1);
    // unsigned int
    // used_memory_in_bytes=HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_)-HardwareDetector::instance().getFreeMemorySizeInByte(mem_id_);
    size_t used_memory_in_bytes = this->getUsedBufferSize_internal();

    // can we possibly fit the requested column in the  Cache?
    if (column_ptr->getSizeinBytes() > max_buffer_size_in_byte_)
      return ColumnPtr();

    if (column_ptr->getSizeinBytes() + used_memory_in_bytes >
        max_buffer_size_in_byte_) {
      cout << "Warning: DataCache::getColumn(): Unsufficient Device Memory"
           << endl;
      size_t number_of_bytes_to_cleanup =
          (column_ptr->getSizeinBytes() + used_memory_in_bytes) -
          max_buffer_size_in_byte_;
      // try to cleanup the Cache so we have enough available memory on
      // if it does not work, return a null pointer, otherwise, continue with
      // the function
      if (!cleanupMemory(number_of_bytes_to_cleanup)) {
        return ColumnPtr();
      }
    }
    // in general columns can use other columns to store their data,
    // e.g., compression methods or LookupArrays
    // to avoid a deadlock here, we unlock the mutex for the duration of the
    // copy
    // operation
    lock.unlock();
    result_ptr = copy(column_ptr, mem_id_);
    lock.lock();
    if (!result_ptr) {
      cout << "Warning: DataCache::getColumn(): Unsufficient Device Memory"
           << endl;
      return ColumnPtr();
    }
    map_.insert(std::make_pair(column_ptr, result_ptr));
    // update statistics maps
    column_accesses_[column_ptr]++;
    column_least_recently_accessed_[column_ptr] = getTimestamp();
    return result_ptr;
  } else {
    if (!quiet) cout << "Cache Hit: " << column_ptr->getName() << endl;
    StatisticsManager::instance().addToValue("COLUMN_CACHE_HIT", 1);
    // update statistics maps
    column_accesses_[column_ptr]++;
    column_least_recently_accessed_[column_ptr] = getTimestamp();
    return it->second;  // return pointer to  memory
  }
}

const JoinIndexPtr DataCache::getJoinIndex_internal(JoinIndexPtr join_index) {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);

  StatisticsManager::instance().addToValue("JOIN_INDEX_CACHE_ACCESS", 1);
  JoinIndexPtr result_ptr;
  if (!join_index) return JoinIndexPtr();
  if (!caching_enabled_) {
    StatisticsManager::instance().addToValue("JOIN_INDEX_CACHE_MISS", 1);
    result_ptr = copy(join_index, mem_id_);
    return result_ptr;
  }

  JoinIndexMap::iterator it = join_index_map_.find(join_index);
  if (it == join_index_map_.end()) {
    if (!quiet) cout << "Cache Miss: JOIN Index" << endl;
    StatisticsManager::instance().addToValue("JOIN_INDEX_CACHE_MISS", 1);
    // unsigned int
    // used_memory_in_bytes=HardwareDetector::instance().getTotalMemorySizeInByte(mem_id_)-HardwareDetector::instance().getFreeMemorySizeInByte(mem_id_);
    size_t used_memory_in_bytes = this->getUsedBufferSize_internal();

    size_t size_of_joinindex = getSizeInBytes(
        join_index);  //(join_index->first->getPositionList()->size()+join_index->second->getPositionList()->size())*sizeof(PositionList::value_type);
    if (size_of_joinindex + used_memory_in_bytes > max_buffer_size_in_byte_) {
      COGADB_WARNING(
          "DataCache::getJoinIndex(): Unsufficient Device Memory"
              << std::endl
              << "\tSizeof JoinIndex: " << size_of_joinindex / (1024 * 1024)
              << "MB" << std::endl
              << "\tMemory in Use: " << used_memory_in_bytes / (1024 * 1024)
              << "MB" << std::endl
              << "\t Buffer Size: " << max_buffer_size_in_byte_ / (1024 * 1024)
              << "MB",
          "");
      size_t number_of_bytes_to_cleanup =
          (size_of_joinindex + used_memory_in_bytes) - max_buffer_size_in_byte_;
      // try to cleanup the cache so we have enough available memory on
      // if it does not work, return a null pointer, otherwise, continue with
      // the function
      if (!cleanupMemory(number_of_bytes_to_cleanup)) {
        return JoinIndexPtr();
      }
      used_memory_in_bytes = this->getUsedBufferSize_internal();
      // apply column replacement strategy!
      //					return JoinIndexPtr();
    }

    // if we now have enough memory, copy the Join Index to the
    if (size_of_joinindex + used_memory_in_bytes < max_buffer_size_in_byte_) {
      result_ptr = copy(join_index, mem_id_);
    }

    if (!result_ptr) {
      COGADB_WARNING("DataCache::getJoinIndex(): Unsufficient Device Memory",
                     "");
      return JoinIndexPtr();
    }
    join_index_map_.insert(std::make_pair(join_index, result_ptr));
    join_index_accesses_[join_index]++;
    join_index_least_recently_accessed_[join_index] = getTimestamp();
    return result_ptr;
  } else {
    if (!quiet) cout << "Cache Hit: JOIN INDEX" << endl;
    StatisticsManager::instance().addToValue("JOIN_INDEX_CACHE_HIT", 1);
    join_index_accesses_[join_index]++;
    join_index_least_recently_accessed_[join_index] = getTimestamp();
    return it->second;  // return pointer to  memory
  }

  return JoinIndexPtr();
}

bool DataCache::removeColumn(ColumnPtr dev_col) {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  Map::iterator it;
  for (it = map_.begin(); it != map_.end(); ++it) {
    if (it->second == dev_col) {
      map_.erase(it);
      return true;
    }
  }
  return true;
}

bool DataCache::removeJoinIndex(JoinIndexPtr join_index) {
  boost::lock_guard<boost::mutex> lock(cache_mutex_);
  JoinIndexMap::iterator it;
  for (it = join_index_map_.begin(); it != join_index_map_.end(); ++it) {
    if (it->second == join_index) {
      join_index_map_.erase(it);
      return true;
    }
  }
  return false;
}

boost::mutex global_mutex_place_frequently_used_columns;

/* We had an issue when CoGaDB exited and the global table list was
 * cleaned up and afterwards, the data placement thread
 * called placeMostFrequentlyUsedColumns(). This is a race condition,
 * and we fix this by calling this function before return in main.
 * Then, the data placement thread will notice this when he wakes up during
 * the exit process of calling global destructors and quits execution. */
void OnExitCloseDataPlacementThread() {
  DataCacheManager::instance().stopDataPlacementThreads();
  if (!quiet)
    std::cout << "Calling exit handler OnExitCloseDataPlacementThread!"
              << std::endl;
}

void DataCache::stop_data_placement_thread() {
  this->data_placement_thread_->interrupt();
  this->data_placement_thread_->join();
  //        std::cout << "Stopped data placement thread" << std::endl;
}

bool DataCache::placeMostFrequentlyUsedColumns(std::ostream& out) {
  boost::lock_guard<boost::mutex> lock(
      global_mutex_place_frequently_used_columns);

  size_t current_buffer_size = max_buffer_size_in_byte_;
  std::vector<TablePtr> tables = getGlobalTableList();
  typedef std::pair<std::string, std::string> FullColumnName;
  typedef std::multimap<uint64_t, FullColumnName> ColumnAccessFrequencies;
  ColumnAccessFrequencies access_frequency_map;

  for (size_t i = 0; i < tables.size(); ++i) {
    boost::shared_ptr<Table> tab =
        boost::dynamic_pointer_cast<Table>(tables[i]);
    assert(tab != NULL);
    ColumnAccessStatisticsVector stats = tab->getColumnAccessStatistics();
    for (size_t j = 0; j < stats.size(); ++j) {
      access_frequency_map.insert(std::make_pair(
          stats[j].statistics.number_of_accesses,
          FullColumnName(stats[j].table_name, stats[j].column_name)));
    }
  }

  size_t used_buffer_space_in_byte = 0;
  ColumnAccessFrequencies::reverse_iterator cit;
  std::set<ColumnPtr> columns_to_place;
  for (cit = access_frequency_map.rbegin(); cit != access_frequency_map.rend();
       ++cit) {
    TablePtr tab = getTablebyName(cit->second.first);
    ColumnPtr col = tab->getColumnbyName(cit->second.second);
    assert(col != NULL);
    if (used_buffer_space_in_byte + col->getSizeinBytes() <
            current_buffer_size &&
        cit->first > 0) {
      used_buffer_space_in_byte += col->getSizeinBytes();
      columns_to_place.insert(col);
    }
  }

  {
    boost::lock_guard<boost::mutex> lock(cache_mutex_);
    Map::const_iterator col_it;
    std::vector<ColumnPtr> columns_to_evict;
    // determine columns to evict from cache
    for (col_it = map_.begin(); col_it != map_.end(); ++col_it) {
      if (columns_to_place.find((*col_it).first) == columns_to_place.end()) {
        columns_to_evict.push_back((*col_it).first);
      }
    }
    // evict columns
    for (size_t i = 0; i < columns_to_evict.size(); ++i) {
      out << "[data placement thread] evict column "
          << columns_to_evict[i]->getName() << std::endl;
      map_.erase(columns_to_evict[i]);
    }
  }

  // put columns in buffer
  {
    std::set<ColumnPtr>::const_iterator cit;
    for (cit = columns_to_place.begin(); cit != columns_to_place.end(); ++cit) {
      out << "[data placement thread] Cache Column: " << (*cit)->getName()
          << std::endl;
      ColumnPtr col = getColumn_internal(*cit);
    }
  }

  return true;
}

void DataCache::data_placement_thread() {
  boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
  using namespace boost::filesystem;

  uint64_t number_of_iterations = 0;
  string dir_path(RuntimeConfiguration::instance().getPathToDatabase() +
                  "/logfiles");
  string logfile_path = dir_path + "/data_placement_manager.log";

  /* in case no database is set, we just omit the logging */
  if (RuntimeConfiguration::instance().getPathToDatabase() != "" &&
      RuntimeConfiguration::instance().getPathToDatabase() != "./data") {
    try {
      // create directory if is does not exist
      if (!exists(dir_path)) create_directory(dir_path);
      std::ofstream logfile(logfile_path.c_str(), ios::trunc);
      logfile.close();
    } catch (boost::filesystem::filesystem_error& e) {
      COGADB_ERROR("Could not create new logfile! Path: " << logfile_path, "");
    }
  }

  while (true) {
    boost::this_thread::sleep(boost::posix_time::milliseconds(10000));

    if (!VariableManager::instance().getVariableValueBoolean(
            "enable_automatic_data_placement")) {
      continue;
    }
    std::ofstream logfile(logfile_path.c_str(), ios::app);

    std::stringstream ss;
    ss << "This is the data placement thread!" << std::endl;
    placeMostFrequentlyUsedColumns(ss);
    if (logfile.good()) {
      logfile << ss.str();
    } else {
      COGADB_ERROR(
          "Could not write log message to file: '" << logfile_path << "'", "");
    }
    logfile.close();
    number_of_iterations++;
  }
}

DataCacheManager::DataCacheManager() : caches_() {
  const CoGaDB::DeviceSpecifications& dev_specs =
      CoGaDB::HardwareDetector::instance().getDeviceSpecifications();

  /* set statistical method */
  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    // create caches for all non CPUs
    if (dev_specs[i].getDeviceType() != hype::CPU) {
      // if we have no data cache for a memory ID, we create one
      if (caches_.find(dev_specs[i].getMemoryID()) == caches_.end()) {
        // use half of the total device memory as buffer for access structures
        DataCachePtr cache(
            new DataCache(dev_specs[i].getTotalMemoryCapacity() * 0.5,
                          dev_specs[i].getMemoryID()));
        assert(cache != NULL);
        caches_.insert(std::make_pair(dev_specs[i].getMemoryID(), cache));
      }
    }
  }
  if (caches_.empty()) {
    // add default device so calls to DataCacheManager::getDataCache() always
    // return a data cache
    DataCachePtr cache(new DataCache(0, hype::PD_Memory_1));
    assert(cache != NULL);
    caches_.insert(std::make_pair(hype::PD_Memory_1, cache));
  }
}

boost::mutex data_cache_manager_mutex;

DataCacheManager& DataCacheManager::instance() {
  data_cache_manager_mutex.lock();
  static DataCacheManager cache_manager;
  data_cache_manager_mutex.unlock();

  return cache_manager;
}

DataCache& DataCacheManager::getDataCache(
    const hype::ProcessingDeviceMemoryID mem_id) const {
  CacheMap::const_iterator cit = caches_.find(mem_id);
  assert(cit != caches_.end());
  return *cit->second;
  //                return DataCache(0, hype::PD_Memory_0);
}

DataCache& DataCacheManager::getDataCache(
    const ProcessorSpecification& proc_spec) const {
  return getDataCache(hype::util::getMemoryID(proc_spec.proc_id));
}

void DataCacheManager::print(std::ostream& out) const {
  CacheMap::const_iterator cit;
  for (cit = caches_.begin(); cit != caches_.end(); ++cit) {
    cit->second->printStatus(out);
  }
}

void DataCacheManager::stopDataPlacementThreads() {
  CacheMap::const_iterator cit;
  for (cit = caches_.begin(); cit != caches_.end(); ++cit) {
    cit->second->stop_data_placement_thread();
  }
}

}  // end namespace CoGaDB
