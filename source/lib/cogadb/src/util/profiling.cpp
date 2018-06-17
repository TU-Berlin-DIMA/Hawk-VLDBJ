

#include <util/profiling.hpp>

#ifdef COGADB_USE_INTEL_PERFORMANCE_COUNTER
//    #include <IntelPerformanceCounterMonitorV2.7/cpucounters.h>
//    #include <IntelPerformanceCounterMonitorV2.7/utils.h>
#include <cpucounters.h>
#include <utils.h>

/* This functions were taken from
 * "gpudbms/external_libraries/IntelPerformanceCounterMonitorV2.7/pcm.cpp"
 * for pretty printing the measurements of Intel's Performance Counter Monitor.
 */
std::string temp_format(int32 t) {
  char buffer[1024];
  if (t == PCM_INVALID_THERMAL_HEADROOM) return "N/A";

  sprintf(buffer, "%2d", t);
  return buffer;
}

void print_pcm_measurements(PCM* m,
                            const std::vector<CoreCounterState>& cstates1,
                            const std::vector<CoreCounterState>& cstates2,
                            const std::vector<SocketCounterState>& sktstate1,
                            const std::vector<SocketCounterState>& sktstate2,
                            const SystemCounterState& sstate1,
                            const SystemCounterState& sstate2,
                            const int cpu_model, const bool show_core_output,
                            const bool show_socket_output,
                            const bool show_system_output, std::ostream& out) {
  out << "\n";
  out << " EXEC  : instructions per nominal CPU cycle"
      << "\n";
  out << " IPC   : instructions per CPU cycle"
      << "\n";
  out << " FREQ  : relation to nominal CPU frequency='unhalted clock "
         "ticks'/'invariant timer ticks' (includes Intel Turbo Boost)"
      << "\n";
  if (cpu_model != PCM::ATOM)
    out << " AFREQ : relation to nominal CPU frequency while in active state "
           "(not in power-saving C state)='unhalted clock ticks'/'invariant "
           "timer ticks while in C0-state'  (includes Intel Turbo Boost)"
        << "\n";
  if (cpu_model != PCM::ATOM)
    out << " L3MISS: L3 cache misses "
        << "\n";
  if (cpu_model == PCM::ATOM)
    out << " L2MISS: L2 cache misses "
        << "\n";
  else
    out << " L2MISS: L2 cache misses (including other core's L2 cache *hits*) "
        << "\n";
  if (cpu_model != PCM::ATOM)
    out << " L3HIT : L3 cache hit ratio (0.00-1.00)"
        << "\n";
  out << " L2HIT : L2 cache hit ratio (0.00-1.00)"
      << "\n";
  if (cpu_model != PCM::ATOM)
    out << " L3CLK : ratio of CPU cycles lost due to L3 cache misses "
           "(0.00-1.00), in some cases could be >1.0 due to a higher memory "
           "latency"
        << "\n";
  if (cpu_model != PCM::ATOM)
    out << " L2CLK : ratio of CPU cycles lost due to missing L2 cache but "
           "still hitting L3 cache (0.00-1.00)"
        << "\n";
  if (cpu_model != PCM::ATOM)
    out << " READ  : bytes read from memory controller (in GBytes)"
        << "\n";
  if (cpu_model != PCM::ATOM)
    out << " WRITE : bytes written to memory controller (in GBytes)"
        << "\n";
  if (m->memoryIOTrafficMetricAvailable())
    out << " IO    : bytes read/written due to IO requests to memory "
           "controller (in GBytes); this may be an over estimate due to "
           "same-cache-line partial requests"
        << "\n";
  out << " TEMP  : Temperature reading in 1 degree Celsius relative to the "
         "TjMax temperature (thermal headroom): 0 corresponds to the max "
         "temperature"
      << "\n";
  out << "\n";
  out << "\n";
  out.precision(2);
  out << std::fixed;
  if (cpu_model == PCM::ATOM)
    out << " Core (SKT) | EXEC | IPC  | FREQ | L2MISS | L2HIT | TEMP"
        << "\n"
        << "\n";
  else {
    if (m->memoryIOTrafficMetricAvailable())
      out << " Core (SKT) | EXEC | IPC  | FREQ  | AFREQ | L3MISS | L2MISS | "
             "L3HIT | L2HIT | L3CLK | L2CLK  | READ  | WRITE |  IO   | TEMP"
          << "\n"
          << "\n";
    else
      out << " Core (SKT) | EXEC | IPC  | FREQ  | AFREQ | L3MISS | L2MISS | "
             "L3HIT | L2HIT | L3CLK | L2CLK  | READ  | WRITE | TEMP"
          << "\n"
          << "\n";
  }

  if (show_core_output) {
    for (uint32 i = 0; i < m->getNumCores(); ++i) {
      if (m->isCoreOnline(i) == false) continue;

      if (cpu_model != PCM::ATOM) {
        out << " " << std::setw(3) << i << "   " << std::setw(2)
            << m->getSocketId(i) << "     "
            << getExecUsage(cstates1[i], cstates2[i]) << "   "
            << getIPC(cstates1[i], cstates2[i]) << "   "
            << getRelativeFrequency(cstates1[i], cstates2[i]) << "    "
            << getActiveRelativeFrequency(cstates1[i], cstates2[i]) << "    "
            << unit_format(getL3CacheMisses(cstates1[i], cstates2[i])) << "   "
            << unit_format(getL2CacheMisses(cstates1[i], cstates2[i])) << "    "
            << getL3CacheHitRatio(cstates1[i], cstates2[i]) << "    "
            << getL2CacheHitRatio(cstates1[i], cstates2[i]) << "    "
            << getCyclesLostDueL3CacheMisses(cstates1[i], cstates2[i]) << "    "
            << getCyclesLostDueL2CacheMisses(cstates1[i], cstates2[i]);
        if (m->memoryIOTrafficMetricAvailable())
          out << "     N/A     N/A     N/A";
        else
          out << "     N/A     N/A";
        out << "     " << temp_format(cstates2[i].getThermalHeadroom()) << "\n";
      } else
        out << " " << std::setw(3) << i << "   " << std::setw(2)
            << m->getSocketId(i) << "     "
            << getExecUsage(cstates1[i], cstates2[i]) << "   "
            << getIPC(cstates1[i], cstates2[i]) << "   "
            << getRelativeFrequency(cstates1[i], cstates2[i]) << "   "
            << unit_format(getL2CacheMisses(cstates1[i], cstates2[i])) << "    "
            << getL2CacheHitRatio(cstates1[i], cstates2[i]) << "     "
            << temp_format(cstates2[i].getThermalHeadroom()) << "\n";
    }
  }
  if (show_socket_output) {
    if (!(m->getNumSockets() == 1 && cpu_model == PCM::ATOM)) {
      out << "-----------------------------------------------------------------"
             "------------------------------------------------------------"
          << "\n";
      for (uint32 i = 0; i < m->getNumSockets(); ++i) {
        out << " SKT   " << std::setw(2) << i << "     "
            << getExecUsage(sktstate1[i], sktstate2[i]) << "   "
            << getIPC(sktstate1[i], sktstate2[i]) << "   "
            << getRelativeFrequency(sktstate1[i], sktstate2[i]) << "    "
            << getActiveRelativeFrequency(sktstate1[i], sktstate2[i]) << "    "
            << unit_format(getL3CacheMisses(sktstate1[i], sktstate2[i]))
            << "   "
            << unit_format(getL2CacheMisses(sktstate1[i], sktstate2[i]))
            << "    " << getL3CacheHitRatio(sktstate1[i], sktstate2[i])
            << "    " << getL2CacheHitRatio(sktstate1[i], sktstate2[i])
            << "    "
            << getCyclesLostDueL3CacheMisses(sktstate1[i], sktstate2[i])
            << "    "
            << getCyclesLostDueL2CacheMisses(sktstate1[i], sktstate2[i]);
        if (m->memoryTrafficMetricsAvailable())
          out << "    "
              << getBytesReadFromMC(sktstate1[i], sktstate2[i]) /
                     double(1024ULL * 1024ULL * 1024ULL)
              << "    "
              << getBytesWrittenToMC(sktstate1[i], sktstate2[i]) /
                     double(1024ULL * 1024ULL * 1024ULL);
        else
          out << "     N/A     N/A";

        if (m->memoryIOTrafficMetricAvailable())
          out << "    "
              << getIORequestBytesFromMC(sktstate1[i], sktstate2[i]) /
                     double(1024ULL * 1024ULL * 1024ULL);

        out << "     " << temp_format(sktstate2[i].getThermalHeadroom())
            << "\n";
      }
    }
  }
  out << "---------------------------------------------------------------------"
         "--------------------------------------------------------"
      << "\n";

  if (show_system_output) {
    if (cpu_model != PCM::ATOM) {
      out << " TOTAL  *     " << getExecUsage(sstate1, sstate2) << "   "
          << getIPC(sstate1, sstate2) << "   "
          << getRelativeFrequency(sstate1, sstate2) << "    "
          << getActiveRelativeFrequency(sstate1, sstate2) << "    "
          << unit_format(getL3CacheMisses(sstate1, sstate2)) << "   "
          << unit_format(getL2CacheMisses(sstate1, sstate2)) << "    "
          << getL3CacheHitRatio(sstate1, sstate2) << "    "
          << getL2CacheHitRatio(sstate1, sstate2) << "    "
          << getCyclesLostDueL3CacheMisses(sstate1, sstate2) << "    "
          << getCyclesLostDueL2CacheMisses(sstate1, sstate2);
      if (m->memoryTrafficMetricsAvailable())
        out << "    "
            << getBytesReadFromMC(sstate1, sstate2) /
                   double(1024ULL * 1024ULL * 1024ULL)
            << "    "
            << getBytesWrittenToMC(sstate1, sstate2) /
                   double(1024ULL * 1024ULL * 1024ULL);
      else
        out << "     N/A     N/A";

      if (m->memoryIOTrafficMetricAvailable())
        out << "    "
            << getIORequestBytesFromMC(sstate1, sstate2) /
                   double(1024ULL * 1024ULL * 1024ULL);

      out << "     N/A\n";
    } else
      out << " TOTAL  *     " << getExecUsage(sstate1, sstate2) << "   "
          << getIPC(sstate1, sstate2) << "   "
          << getRelativeFrequency(sstate1, sstate2) << "   "
          << unit_format(getL2CacheMisses(sstate1, sstate2)) << "    "
          << getL2CacheHitRatio(sstate1, sstate2) << "     N/A\n";
  }

  if (show_system_output) {
    out << "\n"
        << " Instructions retired: "
        << unit_format(getInstructionsRetired(sstate1, sstate2))
        << " ; Active cycles: " << unit_format(getCycles(sstate1, sstate2))
        << " ; Time (TSC): "
        << unit_format(getInvariantTSC(cstates1[0], cstates2[0]))
        << "ticks ; C0 (active,non-halted) core residency: "
        << (getCoreCStateResidency(0, sstate1, sstate2) * 100.) << " %\n";
    out << "\n";
    for (int s = 1; s <= PCM::MAX_C_STATE; ++s)
      if (m->isCoreCStateResidencySupported(s))
        out << " C" << s << " core residency: "
            << (getCoreCStateResidency(s, sstate1, sstate2) * 100.) << " %;";
    out << "\n";
    for (int s = 0; s <= PCM::MAX_C_STATE; ++s)
      if (m->isPackageCStateResidencySupported(s))
        out << " C" << s << " package residency: "
            << (getPackageCStateResidency(s, sstate1, sstate2) * 100.) << " %;";
    out << "\n";
    if (m->getNumCores() == m->getNumOnlineCores()) {
      out << "\n"
          << " PHYSICAL CORE IPC                 : "
          << getCoreIPC(sstate1, sstate2) << " => corresponds to "
          << 100. * (getCoreIPC(sstate1, sstate2) / double(m->getMaxIPC()))
          << " % utilization for cores in active state";
      out << "\n"
          << " Instructions per nominal CPU cycle: "
          << getTotalExecUsage(sstate1, sstate2) << " => corresponds to "
          << 100. *
                 (getTotalExecUsage(sstate1, sstate2) / double(m->getMaxIPC()))
          << " % core utilization over time interval"
          << "\n";
    }
  }

  if (show_socket_output) {
    if (m->getNumSockets() > 1)  // QPI info only for multi socket systems
    {
      out << "\n"
          << "Intel(r) QPI data traffic estimation in bytes (data traffic "
             "coming to CPU/socket through QPI links):"
          << "\n"
          << "\n";

      const uint32 qpiLinks = (uint32)m->getQPILinksPerSocket();

      out << "              ";
      for (uint32 i = 0; i < qpiLinks; ++i) out << " QPI" << i << "    ";

      if (m->qpiUtilizationMetricsAvailable()) {
        out << "| ";
        for (uint32 i = 0; i < qpiLinks; ++i) out << " QPI" << i << "  ";
      }

      out << "\n"
          << "-----------------------------------------------------------------"
             "-----------------------------"
          << "\n";

      for (uint32 i = 0; i < m->getNumSockets(); ++i) {
        out << " SKT   " << std::setw(2) << i << "     ";
        for (uint32 l = 0; l < qpiLinks; ++l)
          out << unit_format(getIncomingQPILinkBytes(i, l, sstate1, sstate2))
              << "   ";

        if (m->qpiUtilizationMetricsAvailable()) {
          out << "|  ";
          for (uint32 l = 0; l < qpiLinks; ++l)
            out << std::setw(3) << std::dec
                << int(100. *
                       getIncomingQPILinkUtilization(i, l, sstate1, sstate2))
                << "%   ";
        }

        out << "\n";
      }
    }
  }

  if (show_system_output) {
    out << "-------------------------------------------------------------------"
           "---------------------------"
        << "\n";

    if (m->getNumSockets() > 1)  // QPI info only for multi socket systems
      out << "Total QPI incoming data traffic: "
          << unit_format(getAllIncomingQPILinkBytes(sstate1, sstate2))
          << "     QPI data traffic/Memory controller traffic: "
          << getQPItoMCTrafficRatio(sstate1, sstate2) << "\n";
  }

  if (show_socket_output) {
    if (m->getNumSockets() > 1 &&
        (m->outgoingQPITrafficMetricsAvailable()))  // QPI info only for multi
                                                    // socket systems
    {
      out << "\n"
          << "Intel(r) QPI traffic estimation in bytes (data and non-data "
             "traffic outgoing from CPU/socket through QPI links):"
          << "\n"
          << "\n";

      const uint32 qpiLinks = (uint32)m->getQPILinksPerSocket();

      out << "              ";
      for (uint32 i = 0; i < qpiLinks; ++i) out << " QPI" << i << "    ";

      out << "| ";
      for (uint32 i = 0; i < qpiLinks; ++i) out << " QPI" << i << "  ";

      out << "\n"
          << "-----------------------------------------------------------------"
             "-----------------------------"
          << "\n";

      for (uint32 i = 0; i < m->getNumSockets(); ++i) {
        out << " SKT   " << std::setw(2) << i << "     ";
        for (uint32 l = 0; l < qpiLinks; ++l)
          out << unit_format(getOutgoingQPILinkBytes(i, l, sstate1, sstate2))
              << "   ";

        out << "|  ";
        for (uint32 l = 0; l < qpiLinks; ++l)
          out << std::setw(3) << std::dec
              << int(100. *
                     getOutgoingQPILinkUtilization(i, l, sstate1, sstate2))
              << "%   ";

        out << "\n";
      }

      out << "-----------------------------------------------------------------"
             "-----------------------------"
          << "\n";
      out << "Total QPI outgoing data and non-data traffic: "
          << unit_format(getAllOutgoingQPILinkBytes(sstate1, sstate2)) << "\n";
    }
  }
  if (show_socket_output) {
    if (m->packageEnergyMetricsAvailable()) {
      out << "\n";
      out << "-----------------------------------------------------------------"
             "-----------------------------"
          << "\n";
      for (uint32 i = 0; i < m->getNumSockets(); ++i) {
        out << " SKT   " << std::setw(2) << i << " package consumed "
            << getConsumedJoules(sktstate1[i], sktstate2[i]) << " Joules\n";
      }
      out << "-----------------------------------------------------------------"
             "-----------------------------"
          << "\n";
      out << " TOTAL:                    "
          << getConsumedJoules(sstate1, sstate2) << " Joules\n";
    }
    if (m->dramEnergyMetricsAvailable()) {
      out << "\n";
      out << "-----------------------------------------------------------------"
             "-----------------------------"
          << "\n";
      for (uint32 i = 0; i < m->getNumSockets(); ++i) {
        out << " SKT   " << std::setw(2) << i << " DIMMs consumed "
            << getDRAMConsumedJoules(sktstate1[i], sktstate2[i]) << " Joules\n";
      }
      out << "-----------------------------------------------------------------"
             "-----------------------------"
          << "\n";
      out << " TOTAL:                  "
          << getDRAMConsumedJoules(sstate1, sstate2) << " Joules\n";
    }
  }
}

#endif
