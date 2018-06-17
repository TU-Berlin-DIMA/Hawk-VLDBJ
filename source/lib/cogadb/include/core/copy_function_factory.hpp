/*
 * File:   copy_function_factory.hpp
 * Author: sebastian
 *
 * Created on 31. Dezember 2014, 16:50
 */

#ifndef COPY_FUNCTION_FACTORY_HPP
#define COPY_FUNCTION_FACTORY_HPP

#include <hype.hpp>

namespace CoGaDB {

  bool isCPUMemory(const hype::ProcessingDeviceMemoryID& mem_id);
  bool isGPUMemory(const hype::ProcessingDeviceMemoryID& mem_id);

  template <typename T>
  class CopyFunctionFactory {
   public:
    typedef bool (*CopyFunctionPtr)(T* dest, const T* source,
                                    size_t number_of_bytes);

    static CopyFunctionPtr getCopyFunction(
        const hype::ProcessingDeviceMemoryID& mem_id_dest,
        const hype::ProcessingDeviceMemoryID& mem_id_source);

   private:
    typedef std::pair<bool, bool> MemoryLocationPair;
    typedef std::map<MemoryLocationPair, CopyFunctionPtr> CopyFunctionMap;

    CopyFunctionMap map;
    /* do not copy */
    CopyFunctionFactory();
    CopyFunctionFactory(const CopyFunctionFactory&);
    CopyFunctionFactory& operator=(const CopyFunctionFactory&);

    CopyFunctionPtr getCopyFunction_internal(
        const hype::ProcessingDeviceMemoryID& mem_id_dest,
        const hype::ProcessingDeviceMemoryID& mem_id_source);

    static CopyFunctionFactory<T>& instance();
    /* COPY OPERATIONS */
    static bool copyCPU2CPU(T* dest, const T* source, size_t number_of_bytes);
    static bool copyCPU2GPU(T* dest, const T* source, size_t number_of_bytes);
    static bool copyGPU2CPU(T* dest, const T* source, size_t number_of_bytes);
    static bool copyGPU2GPU(T* dest, const T* source, size_t number_of_bytes);
    /* END COPY OPERATIONS */
  };

  class MemsetFunctionFactory {
   public:
    typedef bool (*MemsetFunctionPtr)(void* array, int value,
                                      size_t number_of_bytes);
    static MemsetFunctionPtr getMemsetFunction(
        const hype::ProcessingDeviceMemoryID& mem_id);

   private:
    static bool memsetCPU(void* array, int value, size_t number_of_bytes);
    static bool memsetGPU(void* array, int value, size_t number_of_bytes);
  };
  typedef MemsetFunctionFactory::MemsetFunctionPtr MemsetFunctionPtr;

}  // end namespae CoGaDB

#endif /* COPY_FUNCTION_FACTORY_HPP */
