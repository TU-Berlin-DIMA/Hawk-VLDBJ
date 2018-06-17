#pragma once
#include <core/base_column.hpp>
#include <lookup_table/join_index.hpp>
#include <map>
#include <utility>

namespace CoGaDB {

  //    class JoinIndex;
  //    typedef boost::shared_ptr<JoinIndex> JoinIndexPtr;

  //	class GPU_Column_Cache{
  //		public:
  //		typedef std::map<ColumnPtr,gpu::GPU_Base_ColumnPtr> Map;
  //                typedef std::map<JoinIndexPtr,gpu::GPU_JoinIndexPtr>
  //                JoinIndexMap;
  //
  //
  //		/* \brief if GPU column already exists, then return pointer to
  // it,
  // otherwise create new GPU column by transferring the data the GPU memory
  //			\detail if the GPU Buffer is full, then delete least
  // recently
  // used Column*/
  //		const gpu::GPU_Base_ColumnPtr getGPUColumn(ColumnPtr);
  //		const gpu::GPU_JoinIndexPtr getGPUJoinIndex(JoinIndexPtr);
  //
  //                void pinColumnsOnGPU(bool value);
  //                void pinJoinIndexesOnGPU(bool value);
  //                bool haveColumnsPinned() const;
  //                bool haveJoinIndexesPinned() const;
  //                size_t getAvailableGPUBufferSize() const;
  //                size_t getGPUBufferSize() const;
  //                bool setGPUBufferSizeInByte(size_t);
  //
  //                bool isCached(ColumnPtr);
  //                bool isCached(JoinIndexPtr);
  //
  //                bool isCachingEnabled();
  //                void setCacheEnabledStatus(bool status);
  //
  //		static GPU_Column_Cache& instance();
  //
  //		void printStatus(std::ostream& out) const throw();
  //
  //		bool removeGPUColumn(gpu::GPU_Base_ColumnPtr);
  //		bool removeGPUJoinIndex(gpu::GPU_JoinIndexPtr);
  //
  //		private:
  //		GPU_Column_Cache(unsigned int max_gpu_buffer_size_in_byte); //no
  // constructor call outside of this class
  //		GPU_Column_Cache(const GPU_Column_Cache&); //no copy constructor
  //		GPU_Column_Cache& operator=(const GPU_Column_Cache&); //nocopy
  // assignment
  //                bool cleanupGPUMemory(size_t number_of_bytes_to_cleanup);
  //                const gpu::GPU_Base_ColumnPtr
  //                getGPUColumn_internal(ColumnPtr);
  //		const gpu::GPU_JoinIndexPtr
  // getGPUJoinIndex_internal(JoinIndexPtr);
  //                size_t getAvailableGPUBufferSize_internal() const;
  //                size_t getUsedGPUBufferSize_internal() const;
  //		Map map_;
  //                JoinIndexMap join_index_map_;
  //                /* join index replacement statistics*/
  //                typedef std::map<JoinIndexPtr,size_t> JoinIndexAccesses;
  //                JoinIndexAccesses join_index_accesses_;
  //                typedef std::map<JoinIndexPtr,uint64_t>
  //                JoinIndexLRUAccesses;
  //                JoinIndexLRUAccesses join_index_least_recently_accessed_;
  //                /* column replacement statistics*/
  //                typedef std::map<ColumnPtr,size_t> ColumnAccesses;
  //		ColumnAccesses column_accesses_;
  //                typedef std::map<ColumnPtr,uint64_t> ColumnLRUAccesses;
  //		ColumnLRUAccesses column_least_recently_accessed_;
  //		unsigned int max_gpu_buffer_size_in_byte_;
  //		bool caching_enabled_;
  //                /*! \brief if this is true, join indexes are never evicted*/
  //                bool pin_join_indexes_;
  //                /*! \brief if this is true, columns are never evicted*/
  //                bool pin_columns_;
  //	};

}  // end namespace CoGaDB
