
#include <core/device_memory.hpp>
#include <iostream>

namespace hype {

namespace core {

MemoryChunk::MemoryChunk(size_t estimated_memory_capacity_a,
                         ProcessingDeviceMemoryID mem_id_a,
                         uint64_t allocation_id_a)
    : estimated_memory_capacity(estimated_memory_capacity_a),
      mem_id(mem_id_a),
      allocation_id(allocation_id_a) {}

DeviceMemory::DeviceMemory(size_t total_memory_in_bytes_a,
                           ProcessingDeviceMemoryID mem_id)
    : allocated_memory_chunks(),
      total_memory_in_bytes(total_memory_in_bytes_a),
      unique_allocation_id_(0),
      mem_id_(mem_id) {}

uint64_t DeviceMemory::getUniqueMemoryAllocationID() throw() {
  return unique_allocation_id_++;
}

size_t DeviceMemory::getTotalMemoryInBytes() const throw() {
  return this->total_memory_in_bytes;
}
size_t DeviceMemory::getEstimatedFreeMemoryInBytes() const throw() {
  return this->total_memory_in_bytes - getEstimatedUsedMemoryInBytes();
}
size_t DeviceMemory::getEstimatedUsedMemoryInBytes() const throw() {
  size_t estimated_allocated_memory = 0;
  MemoryChunks::const_iterator cit;
  for (cit = this->allocated_memory_chunks.begin();
       cit != this->allocated_memory_chunks.end(); ++cit) {
    estimated_allocated_memory += (*cit)->estimated_memory_capacity;
  }
  return estimated_allocated_memory;
}

MemoryChunkPtr DeviceMemory::allocateMemory(size_t number_of_bytes) {
  size_t free_mem = getEstimatedFreeMemoryInBytes();
  if (number_of_bytes <= free_mem) {
    MemoryChunkPtr mem_chunk(new MemoryChunk(number_of_bytes, mem_id_,
                                             getUniqueMemoryAllocationID()));
    this->allocated_memory_chunks.push_back(mem_chunk);
    return mem_chunk;
  } else {
    // allocation fails because, more memory is
    // requested than we have available according
    // to our book keeping
    return MemoryChunkPtr();
  }
}
bool DeviceMemory::releaseMemory(MemoryChunkPtr mem_chunk) {
  if (mem_chunk->mem_id != this->mem_id_) {
    HYPE_FATAL_ERROR("Release Memory Chunk in Wrong Device Memory!", std::cerr);
  }
  MemoryChunks::iterator cit;
  for (cit = this->allocated_memory_chunks.begin();
       cit != this->allocated_memory_chunks.end(); ++cit) {
    if ((*cit)->allocation_id == mem_chunk->allocation_id) {
      this->allocated_memory_chunks.erase(cit);
      return true;
    }
  }
  return false;
}

DeviceMemories& DeviceMemories::instance() {
  static DeviceMemories dev_mems;
  return dev_mems;
}
DeviceMemoryPtr DeviceMemories::getDeviceMemory(
    ProcessingDeviceMemoryID mem_id) {
  DeviceMemories::Memories::const_iterator it;
  it = this->device_memories.find(mem_id);
  if (it != this->device_memories.end()) {
    return it->second;
  } else {
    // return NULL
    return DeviceMemoryPtr();
  }
}
bool DeviceMemories::existDeviceMemory(ProcessingDeviceMemoryID mem_id) const {
  DeviceMemories::Memories::const_iterator it;
  it = this->device_memories.find(mem_id);
  if (it != this->device_memories.end()) {
    return true;
  } else {
    return false;
  }
}
bool DeviceMemories::addDeviceMemory(ProcessingDeviceMemoryID mem_id,
                                     size_t total_memory_in_bytes) {
  if (!existDeviceMemory(mem_id)) {
    DeviceMemoryPtr dev_mem(new DeviceMemory(total_memory_in_bytes, mem_id));
    this->device_memories.insert(std::make_pair(mem_id, dev_mem));
    return true;
  } else {
    return false;
  }
}
DeviceMemories::DeviceMemories() : device_memories() {}

}  // end namespace core
}  // end namespace hype