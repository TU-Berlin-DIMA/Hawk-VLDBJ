#pragma once
//#include <core/base_column.hpp>
#include <map>
#include <utility>

#include <boost/shared_ptr.hpp>

#include <hype.hpp>

namespace CoGaDB {

  template <typename T>
  class MemoryAllocator {
   public:
    typedef boost::shared_ptr<MemoryAllocator<T> > MemoryAllocatorPtr;
    MemoryAllocator() : mem_id_(hype::PD_Memory_0), name_() {}
    MemoryAllocator(hype::ProcessingDeviceMemoryID mem_id,
                    const std::string name)
        : mem_id_(mem_id), name_(name) {}
    virtual ~MemoryAllocator() {}

    virtual T* allocate(size_t number_of_elements) = 0;
    virtual T* reallocate(T* memory_block, size_t new_number_of_elements) = 0;
    virtual void deallocate(T*& memory_block) = 0;

    hype::ProcessingDeviceMemoryID getMemoryID() const { return mem_id_; }

    std::string getName() const { return name_; }

    static const MemoryAllocatorPtr getMemoryAllocator(
        hype::ProcessingDeviceMemoryID mem_id);

   private:
    hype::ProcessingDeviceMemoryID mem_id_;
    std::string name_;
  };

  template <typename T>
  struct AllocationManager {
   public:
    static T* malloc(hype::ProcessingDeviceMemoryID mem_id,
                     size_t number_of_elements);
    static void free(hype::ProcessingDeviceMemoryID mem_id, T*& data);
    static T* realloc(hype::ProcessingDeviceMemoryID mem_id, T* data,
                      size_t number_of_elements);
  };

  template <typename T>
  T* customMalloc(hype::ProcessingDeviceMemoryID mem_id,
                  size_t number_of_elements) {
    return AllocationManager<T>::malloc(mem_id, number_of_elements);
  }

  template <typename T>
  void customFree(hype::ProcessingDeviceMemoryID mem_id, T*& data) {
    AllocationManager<T>::free(mem_id, data);
  }

  template <typename T>
  T* customRealloc(hype::ProcessingDeviceMemoryID mem_id, T* data,
                   size_t number_of_elements) {
    return AllocationManager<T>::realloc(mem_id, data, number_of_elements);
  }

  //	template <typename T>
  //	class CPUMemoryAllocator_C : public MemoryAllocator<T>{
  //		public:
  //		CPUMemoryAllocator_C() : MemoryAllocator<T>(){}
  //
  //		T* allocate(size_t number_of_elements){
  //			T* ret = (T*) malloc(sizeof(T)*number_of_elements);
  //			return ret;
  //		}
  //		T* reallocate(T* memory_block, size_t new_number_of_elements){
  //			T* ret = (T*) realloc(memory_block,
  // new_number_of_elements*sizeof(T));
  //			return ret;
  //		}
  //		void deallocate(T*& memory_block){
  //			free(memory_block);
  //		}
  //	};
  //
  //	template <typename T>
  //	class CPUMemoryAllocator_CPP : public MemoryAllocator<T>{
  //		public:
  //		CPUMemoryAllocator_CPP() : MemoryAllocator<T>(), alloc_map_(){}
  //
  //		T* allocate(size_t number_of_elements){
  //			T* ret = new T[number_of_elements];
  //                        alloc_map_.insert(std::make_pair(ret,
  //                        number_of_elements));
  //			return ret;
  //		}
  //		T* reallocate(T* memory_block, size_t new_number_of_elements){
  //                    	T* ret=NULL;
  //                        typename AllocationMap::iterator
  //                        it=alloc_map_.find(memory_block);
  //                        if(memory_block){
  //                            if(it==alloc_map_.end()){
  //                                COGADB_FATAL_ERROR("Allocated Memory Block
  //                                not in AllocationMap!","");
  //                            }
  //                            //is the allocated chunk larger than what is
  //                            requested?
  //                            //if yes, then do nothing
  //                            if(it->second>=new_number_of_elements) return
  //                            memory_block;
  //                        }
  //                        std::cout << "Called Realloc: " << (void*)
  //                        memory_block << ": " << new_number_of_elements <<
  //                        std::endl;
  //                        ret = allocate(new_number_of_elements);
  //                        //in case memory block is not NULL, copy data to new
  //                        buffer
  //                        //in case memory block is NULL, allocate new buffer
  //                        if(memory_block){
  //                            //copy objects to new buffer
  //                            std::copy(memory_block, memory_block+it->second,
  //                            ret);
  //                            //delete old entry from map
  //                            alloc_map_.erase(it);
  //                            //delete old memory block
  //                            delete[] memory_block;
  //                        }
  //			return ret;
  //		}
  //		void deallocate(T*& memory_block){
  //                        typename AllocationMap::iterator
  //                        it=alloc_map_.find(memory_block);
  //                        if(it==alloc_map_.end()){
  //                            COGADB_FATAL_ERROR("Deallocated Memory Block not
  //                            in AllocationMap!","");
  //                        }
  //                        //delete entry from map
  //                        alloc_map_.erase(it);
  //			delete[] memory_block;
  //		}
  //                typedef std::map<T*,size_t> AllocationMap;
  //                AllocationMap alloc_map_;
  //	};

  //	template <typename T>
  //	const typename MemoryAllocator<T>::MemoryAllocatorPtr
  // MemoryAllocator<T>::getMemoryAllocator(hype::ProcessingDeviceMemoryID
  // mem_id){
  //		typedef MemoryAllocator<T>::MemoryAllocatorPtr
  // MemoryAllocatorPtr;
  //		MemoryAllocatorPtr allocator;
  //		//FIXME: for now we only create CPU allocators, plain C
  // allocators
  // for elemtary types, and a c++ allocator for obejct types
  //		if(typeid(T)==typeid(std::string)){
  //			allocator=MemoryAllocatorPtr(new
  // CPUMemoryAllocator_CPP<T>());
  //		}else{
  //			allocator=MemoryAllocatorPtr(new
  // CPUMemoryAllocator_C<T>());
  //		}
  //		return allocator;
  //	}

}  // end namespace CoGaDB
