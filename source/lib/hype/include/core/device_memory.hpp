/***********************************************************************************************************
Copyright (c) 2014, Sebastian Breß, TU Dortmund University, Germany. All rights
reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <stdint.h>
#include <boost/shared_ptr.hpp>
#include <config/exports.hpp>
#include <config/global_definitions.hpp>
#include <list>
#include <map>
#include <string>

/* temporarily disable warnings because of missing STL DLL interface */
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4251)
#endif

namespace hype {

  namespace core {

    /*! \brief This type is the basic unit for memory book keeping.*/
    struct MemoryChunk {
      MemoryChunk(size_t estimated_memory_capacity,
                  ProcessingDeviceMemoryID mem_id, uint64_t allocation_id);
      size_t estimated_memory_capacity;
      ProcessingDeviceMemoryID mem_id;
      uint64_t allocation_id;
    };

    typedef boost::shared_ptr<MemoryChunk> MemoryChunkPtr;

    class DeviceMemory {
     public:
      DeviceMemory(size_t total_memory_in_bytes,
                   ProcessingDeviceMemoryID mem_id);
      uint64_t getUniqueMemoryAllocationID() throw();

      size_t getTotalMemoryInBytes() const throw();
      size_t getEstimatedFreeMemoryInBytes() const throw();
      size_t getEstimatedUsedMemoryInBytes() const throw();

      MemoryChunkPtr allocateMemory(size_t number_of_bytes);
      bool releaseMemory(MemoryChunkPtr);

     private:
      typedef std::list<MemoryChunkPtr> MemoryChunks;
      MemoryChunks allocated_memory_chunks;
      size_t total_memory_in_bytes;
      uint64_t unique_allocation_id_;
      ProcessingDeviceMemoryID mem_id_;
    };

    typedef boost::shared_ptr<DeviceMemory> DeviceMemoryPtr;

    // make this a singleton
    class DeviceMemories {
     public:
      static DeviceMemories& instance();
      DeviceMemoryPtr getDeviceMemory(ProcessingDeviceMemoryID mem_id);
      bool existDeviceMemory(ProcessingDeviceMemoryID mem_id) const;
      bool addDeviceMemory(ProcessingDeviceMemoryID mem_id,
                           size_t total_memory_in_bytes);

     private:
      DeviceMemories();
      DeviceMemories(const DeviceMemories&);
      DeviceMemories& operator=(const DeviceMemories&);
      typedef std::map<ProcessingDeviceMemoryID, DeviceMemoryPtr> Memories;
      Memories device_memories;
    };

    //		DeviceMemoryPtr getDeviceMemory(ProcessingDeviceMemoryID
    // mem_id);

  }  // end namespace core
}  // end namespace hype

#ifdef _MSC_VER
#pragma warning(pop)
#endif
