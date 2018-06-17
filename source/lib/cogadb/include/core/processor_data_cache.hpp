/*
 * File:   processor_data_cache.hpp
 * Author: sebastian
 *
 * Created on 3. Januar 2015, 18:37
 */

#ifndef PROCESSOR_DATA_CACHE_HPP
#define PROCESSOR_DATA_CACHE_HPP

#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <core/base_column.hpp>
#include <map>
#include <utility>

namespace CoGaDB {

  class DataCacheManager;

  void OnExitCloseDataPlacementThread();

  class DataCache {
    friend class DataCacheManager;

   public:
    typedef std::map<ColumnPtr, ColumnPtr> Map;
    typedef std::map<JoinIndexPtr, JoinIndexPtr> JoinIndexMap;

    ~DataCache();

    /* \brief if olumn already exists, then return pointer to it, otherwise
       create new column by transferring the data the (co-)processor memory
            \detail if the Buffer is full, then delete least recently used
       Column*/
    const ColumnPtr getColumn(ColumnPtr);
    /* \brief maps a column in co-proessor memory to a host column
       \details this is important to fetch the host version of cached columns*/
    const ColumnPtr getHostColumn(ColumnPtr);
    const JoinIndexPtr getJoinIndex(JoinIndexPtr);

    void pinColumns(bool value);
    void pinJoinIndexes(bool value);
    bool haveColumnsPinned() const;
    bool haveJoinIndexesPinned() const;
    size_t getAvailableBufferSize() const;
    size_t getBufferSize() const;
    bool setBufferSizeInByte(size_t);

    bool isCached(ColumnPtr);
    bool isCached(JoinIndexPtr);

    bool isCachingEnabled();
    void setCacheEnabledStatus(bool status);

    static DataCache& instance();

    void printStatus(std::ostream& out) const throw();

    bool removeColumn(ColumnPtr);
    bool removeJoinIndex(JoinIndexPtr);

    bool placeMostFrequentlyUsedColumns(std::ostream& out);

    void data_placement_thread();
    void stop_data_placement_thread();

   private:
    DataCache(size_t max_buffer_size_in_byte,
              const hype::ProcessingDeviceMemoryID&
                  mem_id);        // no constructor call outside of this class
    DataCache(const DataCache&);  // no copy constructor
    DataCache& operator=(const DataCache&);  // nocopy assignment
    bool cleanupMemory(size_t number_of_bytes_to_cleanup);
    const ColumnPtr getColumn_internal(ColumnPtr);
    const JoinIndexPtr getJoinIndex_internal(JoinIndexPtr);
    size_t getAvailableBufferSize_internal() const;
    size_t getUsedBufferSize_internal() const;
    Map map_;
    JoinIndexMap join_index_map_;
    /* join index replacement statistics*/
    typedef std::map<JoinIndexPtr, size_t> JoinIndexAccesses;
    JoinIndexAccesses join_index_accesses_;
    typedef std::map<JoinIndexPtr, uint64_t> JoinIndexLRUAccesses;
    JoinIndexLRUAccesses join_index_least_recently_accessed_;
    /* column replacement statistics*/
    typedef std::map<ColumnPtr, size_t> ColumnAccesses;
    ColumnAccesses column_accesses_;
    typedef std::map<ColumnPtr, uint64_t> ColumnLRUAccesses;
    ColumnLRUAccesses column_least_recently_accessed_;
    size_t max_buffer_size_in_byte_;
    bool caching_enabled_;
    /*! \brief if this is true, join indexes are never evicted*/
    bool pin_join_indexes_;
    /*! \brief if this is true, columns are never evicted*/
    bool pin_columns_;
    /*! \brief ID of memory managed by this cache*/
    hype::ProcessingDeviceMemoryID mem_id_;
    /*! \brief mutex to protect data stuctures in case of concurrent accesses
     *  \details mutex needs to be locked even in cosnt functions, so we make it
     * mutable
     */
    mutable boost::mutex cache_mutex_;
    boost::shared_ptr<boost::thread> data_placement_thread_;
  };

  typedef boost::shared_ptr<DataCache> DataCachePtr;

  class DataCacheManager {
   public:
    static DataCacheManager& instance();
    DataCache& getDataCache(const hype::ProcessingDeviceMemoryID mem_id) const;
    DataCache& getDataCache(const ProcessorSpecification& proc_spec) const;
    void print(std::ostream& out) const;
    void stopDataPlacementThreads();

   private:
    DataCacheManager();
    typedef std::map<hype::ProcessingDeviceMemoryID, DataCachePtr> CacheMap;
    CacheMap caches_;
    //            boost::mutex mutex_;
  };

}  // end namespace CoGaDB

#endif /* PROCESSOR_DATA_CACHE_HPP */
