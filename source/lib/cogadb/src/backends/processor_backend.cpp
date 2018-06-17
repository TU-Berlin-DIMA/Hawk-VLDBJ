

/*
 * File:   processor_backend.cpp
 * Author: sebastian
 *
 * Created on 27. Dezember 2014, 1:13
 */

#include <backends/processor_backend.hpp>

#include <backends/cpu/cpu_backend.hpp>
#ifdef ENABLE_GPU_ACCELERATION
#include <backends/gpu/gpu_backend.hpp>
#endif

namespace CoGaDB {

template <typename T>
ProcessorBackend<T>* ProcessorBackend<T>::get(
    hype::ProcessingDeviceID proc_dev_id) {
  static CPU_Backend<T> cpu;
#ifdef ENABLE_GPU_ACCELERATION
  static GPU_Backend<T> gpu;
#endif
  // std::cout << "fetch Backend with id: " << (int) proc_dev_id << std::endl;

  if (proc_dev_id == hype::PD0) {
    return &cpu;
#ifdef ENABLE_GPU_ACCELERATION
  } else if (proc_dev_id == hype::PD1) {
    return &gpu;
#endif
  } else {
    COGADB_FATAL_ERROR("Unsupported Device: " << (int)proc_dev_id, "");
  }

  return NULL;
}

COGADB_INSTANTIATE_CLASS_TEMPLATE_FOR_SUPPORTED_TYPES(ProcessorBackend)

}  // end namespace CogaDB
