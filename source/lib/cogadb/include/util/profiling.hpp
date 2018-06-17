/*
 * File:   profiling.hpp
 * Author: sebastian
 *
 * Created on 25. Mai 2015, 18:28
 */

#ifndef PROFILING_HPP
#define PROFILING_HPP

#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

#include <core/global_definitions.hpp>
#include <util/getname.hpp>
#include <util/time_measurement.hpp>
#include <util/types.hpp>

#undef COGADB_USE_INTEL_PERFORMANCE_COUNTER

#ifdef COGADB_USE_INTEL_PERFORMANCE_COUNTER
//    #include <IntelPerformanceCounterMonitorV2.7/cpucounters.h>
//    #include <IntelPerformanceCounterMonitorV2.7/utils.h>
#include <cpucounters.h>
#include <utils.h>

void print_pcm_measurements(PCM* m,
                            const std::vector<CoreCounterState>& cstates1,
                            const std::vector<CoreCounterState>& cstates2,
                            const std::vector<SocketCounterState>& sktstate1,
                            const std::vector<SocketCounterState>& sktstate2,
                            const SystemCounterState& sstate1,
                            const SystemCounterState& sstate2,
                            const int cpu_model, const bool show_core_output,
                            const bool show_socket_output,
                            const bool show_system_output, std::ostream& out);

#define COGADB_PCM_START_PROFILING(name, outstream)                            \
  CoGaDB::Timestamp begin = CoGaDB::getTimestamp();                            \
  PCM* pcm = PCM::getInstance();                                               \
  outstream << "[PROFILER]: Start Profiling '" << name << "'..." << std::endl; \
  std::vector<CoreCounterState> cstates1, cstates2;                            \
  std::vector<SocketCounterState> sktstate1, sktstate2;                        \
  SystemCounterState sstate1, sstate2;                                         \
  const int cpu_model = pcm->getCPUModel();                                    \
  pcm->getAllCounterStates(sstate1, sktstate1, cstates1);

#define COGADB_PCM_STOP_PROFILING(name, outstream, num_elements,              \
                                  size_of_element, show_core_output,          \
                                  show_socket_output, show_system_output)     \
  CoGaDB::Timestamp end = CoGaDB::getTimestamp();                             \
  pcm->getAllCounterStates(sstate2, sktstate2, cstates2);                     \
  outstream << "[PROFILER]: Stop Profiling '" << name << "'..." << std::endl; \
  print_pcm_measurements(pcm, cstates1, cstates2, sktstate1, sktstate2,       \
                         sstate1, sstate2, cpu_model, show_core_output,       \
                         show_socket_output, show_system_output, outstream);  \
  outstream << "[PROFILER]: Time for '" << name << "' for " << num_elements   \
            << " elements: " << double(end - begin) / (1000 * 1000) << "ms"   \
            << std::endl;                                                     \
  outstream << "[PROFILER]: Bandwidth: "                                      \
            << (double(num_elements * size_of_element) /                      \
                (1024 * 1024 * 1024)) /                                       \
                   (double(end - begin) / (1000 * 1000 * 1000))               \
            << "GB/s" << std::endl;

#else

#define COGADB_PCM_START_PROFILING(name, outstream) \
  {}

#define COGADB_PCM_STOP_PROFILING(name, outstream, num_elements,          \
                                  size_of_element, show_core_output,      \
                                  show_socket_output, show_system_output) \
  {}

#endif

#endif /* PROFILING_HPP */
