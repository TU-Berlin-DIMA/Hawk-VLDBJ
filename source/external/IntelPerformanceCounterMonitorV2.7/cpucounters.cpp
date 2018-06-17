/*
Copyright (c) 2009-2014, Intel Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// written by Roman Dementiev
//            Otto Bruggeman
//            Thomas Willhalm
//            Pat Fay
//            Austen Ott
//            Jim Harris (FreeBSD)

/*!     \file cpucounters.cpp
        \brief The bulk of Intel PCM implementation
  */

//#define PCM_TEST_FALLBACK_TO_ATOM

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#ifdef INTELPCM_EXPORTS
// Intelpcm.h includes cpucounters.h
#include "Intelpcm.dll\Intelpcm.h"
#else
#include "cpucounters.h"
#endif
#include "msr.h"
#include "pci.h"
#include "types.h"
#include "utils.h"

#ifdef _MSC_VER
#include <intrin.h>
#include <windows.h>
#include <comdef.h>
#include <tchar.h>
#include "winring0/OlsApiInit.h"
#include "PCM_Win/windriver.h"
#else
#include <pthread.h>
#include <errno.h>
#include <sys/time.h>
#endif

#include <string.h>
#include <limits>
#include <map>
#include <fstream>
#include <algorithm>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/sem.h>

// convertUnknownToInt is used in the safe sysctl call to convert an unkown size to an int
int convertUnknownToInt(size_t size, char* value);

#endif

#if defined (__FreeBSD__)
#include <sys/param.h>
#include <sys/module.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <sys/sem.h>
#include <sys/ioccom.h>
#include <sys/cpuctl.h>
#include <machine/cpufunc.h>
#endif

#undef PCM_UNCORE_PMON_BOX_CHECK_STATUS // debug only

// FreeBSD is much more restrictive about names for semaphores
#if defined (__FreeBSD__)
#define PCM_INSTANCE_LOCK_SEMAPHORE_NAME "/Intel_PCM_inst_lock"
#define PCM_NUM_INSTANCES_SEMAPHORE_NAME "/Intel_num_PCM_inst"
#else
#define PCM_INSTANCE_LOCK_SEMAPHORE_NAME "Intel(r) PCM inst lock"
#define PCM_NUM_INSTANCES_SEMAPHORE_NAME "Num Intel(r) PCM insts"
#endif

#ifdef _MSC_VER

HMODULE hOpenLibSys = NULL;

bool PCM::initWinRing0Lib()
{
	const BOOL result = InitOpenLibSys(&hOpenLibSys);
	
	if(result == FALSE) hOpenLibSys = NULL;

	return result==TRUE;
}

class InstanceLock
{
    HANDLE Mutex;

    InstanceLock();
public:
    InstanceLock(const bool global)
    {
        Mutex = CreateMutex(NULL, FALSE,
            global?(L"Global\\Intel(r) Performance Counter Monitor instance create/destroy lock"):(L"Local\\Intel(r) Performance Counter Monitor instance create/destroy lock"));
        // lock
        WaitForSingleObject(Mutex, INFINITE);
    }
    ~InstanceLock()
    {
        // unlock
        ReleaseMutex(Mutex);
    }
};
#else // Linux or Apple

pthread_mutex_t processIntanceMutex = PTHREAD_MUTEX_INITIALIZER;

class InstanceLock
{
    const char * globalSemaphoreName;
    sem_t * globalSemaphore;
    bool global;

    InstanceLock();
public:
    InstanceLock(const bool global_) : globalSemaphoreName(PCM_INSTANCE_LOCK_SEMAPHORE_NAME), globalSemaphore(NULL), global(global_)
    {
        if(!global)
        {
            pthread_mutex_lock(&processIntanceMutex);
            return;
        }
        umask(0);
        while (1)
        {
            //sem_unlink(globalSemaphoreName); // temporary
            globalSemaphore = sem_open(globalSemaphoreName, O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO, 1);
            if (SEM_FAILED == globalSemaphore)
            {
              if (EACCES == errno)
                {
                    std::cerr << "PCM Error, do not have permissions to open semaphores in /dev/shm/. Waiting one second and retrying..." << std::endl;
                    sleep(1);
                }
            }
            else
            {
				/*
				if (sem_post(globalSemaphore)) {
					perror("sem_post error");
				}
				*/
                break;         // success
            }
        }
        if (sem_wait(globalSemaphore)) {
			perror("sem_wait error");
		}
    }
    ~InstanceLock()
    {
        if(!global)
        {
            pthread_mutex_unlock(&processIntanceMutex);
            return;
        }
        if (sem_post(globalSemaphore)) {
			perror("sem_post error");
		}
    }
}; 
#endif // end of _MSC_VER else

PCM * PCM::instance = NULL;

int bitCount(uint64 n)
{
    int count = 0;
    while (n)
    {
        count += (int)(n & 0x00000001);
        n >>= 1;
    }
    return count;
}

PCM * PCM::getInstance()
{
    // no lock here
    if (instance) return instance;

    InstanceLock lock(false);
    if (instance) return instance;

    return instance = new PCM();
}

uint32 build_bit_ui(int beg, int end)
{
    uint32 myll = 0;
    if (end == 31)
    {
        myll = (uint32)(-1);
    }
    else
    {
        myll = (1 << (end + 1)) - 1;
    }
    myll = myll >> beg;
    return myll;
}

uint32 extract_bits_ui(uint32 myin, uint32 beg, uint32 end)
{
    uint32 myll = 0;
    uint32 beg1, end1;

    // Let the user reverse the order of beg & end.
    if (beg <= end)
    {
        beg1 = beg;
        end1 = end;
    }
    else
    {
        beg1 = end;
        end1 = beg;
    }
    myll = myin >> beg1;
    myll = myll & build_bit_ui(beg1, end1);
    return myll;
}

uint64 build_bit(uint32 beg, uint32 end)
{
    uint64 myll = 0;
    if (end == 63)
    {
        myll = (uint64)(-1);
    }
    else
    {
        myll = (1LL << (end + 1)) - 1;
    }
    myll = myll >> beg;
    return myll;
}

uint64 extract_bits(uint64 myin, uint32 beg, uint32 end)
{
    uint64 myll = 0;
    uint32 beg1, end1;

    // Let the user reverse the order of beg & end.
    if (beg <= end)
    {
        beg1 = beg;
        end1 = end;
    }
    else
    {
        beg1 = end;
        end1 = beg;
    }
    myll = myin >> beg1;
    myll = myll & build_bit(beg1, end1);
    return myll;
}

uint64 PCM::extractCoreGenCounterValue(uint64 val)
{
    if(core_gen_counter_width) 
        return extract_bits(val, 0, core_gen_counter_width-1);
    
    return val;
}

uint64 PCM::extractCoreFixedCounterValue(uint64 val)
{
    if(core_fixed_counter_width) 
        return extract_bits(val, 0, core_fixed_counter_width-1);
    
    return val;
}

uint64 PCM::extractUncoreGenCounterValue(uint64 val)
{
    if(uncore_gen_counter_width) 
        return extract_bits(val, 0, uncore_gen_counter_width-1);
    
    return val;
}

uint64 PCM::extractUncoreFixedCounterValue(uint64 val)
{
    if(uncore_fixed_counter_width) 
        return extract_bits(val, 0, uncore_fixed_counter_width-1);
    
    return val;
}

int32 extractThermalHeadroom(uint64 val)
{
    if(val & (1ULL<<31ULL))
    {  // valid reading
       return (int32)extract_bits(val,16,22);
    }

    // invalid reading
    return PCM_INVALID_THERMAL_HEADROOM;
}

uint64 get_frequency_from_cpuid();

union PCM_CPUID_INFO
{
	int array[4];
        struct { int eax,ebx,ecx,edx; } reg ;
};

void pcm_cpuid(int leaf, PCM_CPUID_INFO & info)
{
    #ifdef _MSC_VER
    // version for Windows
    __cpuid(info.array, leaf);
    #else
    __asm__ __volatile__ ("cpuid" : \
                          "=a" (info.reg.eax), "=b" (info.reg.ebx), "=c" (info.reg.ecx), "=d" (info.reg.edx) : "a" (leaf));
    #endif	
}

bool PCM::detectModel()
{
    char buffer[1024];
    union {
        char cbuf[16];
        int  ibuf[16/sizeof(int)];
    } buf;
    PCM_CPUID_INFO cpuinfo;
    int max_cpuid;
    pcm_cpuid(0, cpuinfo);
    memset(buffer, 0, 1024);
    memset(buf.cbuf, 0, 16);
    buf.ibuf[0] = cpuinfo.array[1];
    buf.ibuf[1] = cpuinfo.array[3];
    buf.ibuf[2] = cpuinfo.array[2];
    if (strncmp(buf.cbuf, "GenuineIntel", 4 * 3) != 0)
    {
        std::cerr << getUnsupportedMessage() << std::endl;
        return false;
    }
    max_cpuid = cpuinfo.array[0];

    pcm_cpuid(1, cpuinfo);
    cpu_family = (((cpuinfo.array[0]) >> 8) & 0xf) | ((cpuinfo.array[0] & 0xf00000) >> 16);
    cpu_model = original_cpu_model = (((cpuinfo.array[0]) & 0xf0) >> 4) | ((cpuinfo.array[0] & 0xf0000) >> 12);

    if (max_cpuid >= 0xa)
    {
        // get counter related info
        pcm_cpuid(0xa, cpuinfo);
        perfmon_version = extract_bits_ui(cpuinfo.array[0], 0, 7);
        core_gen_counter_num_max = extract_bits_ui(cpuinfo.array[0], 8, 15);
        core_gen_counter_width = extract_bits_ui(cpuinfo.array[0], 16, 23);
        if (perfmon_version > 1)
        {
            core_fixed_counter_num_max = extract_bits_ui(cpuinfo.array[3], 0, 4);
            core_fixed_counter_width = extract_bits_ui(cpuinfo.array[3], 5, 12);
        }
    }

    if (cpu_family != 6)
    {
        std::cerr << getUnsupportedMessage() << " CPU Family: " << cpu_family << std::endl;
        return false;
    }

    return true;
}

void PCM::initCStateSupportTables()
{
#define PCM_PARAM_PROTECT(...) __VA_ARGS__
#define PCM_CSTATE_ARRAY(array_ , val ) \
    { \
        static uint64 tmp[] = val; \
        PCM_COMPILE_ASSERT( sizeof(tmp)/sizeof(uint64) == MAX_C_STATE + 1); \
        array_ = tmp; \
        break; \
    }

    // fill package C state array
    switch(original_cpu_model)
    {
        case ATOM:
        case ATOM_2:
        case ATOM_CENTERTON:
        case ATOM_AVOTON:
        case ATOM_BAYTRAIL:
            PCM_CSTATE_ARRAY(pkgCStateMsr, PCM_PARAM_PROTECT({0,    0,  0x3F8,      0,  0x3F9,  0,  0x3FA,  0,      0,  0,  0 }) );
        case NEHALEM_EP:
        case NEHALEM:
        case CLARKDALE:
        case WESTMERE_EP:
        case NEHALEM_EX:
        case WESTMERE_EX:
            PCM_CSTATE_ARRAY(pkgCStateMsr, PCM_PARAM_PROTECT({0,    0,      0,  0x3F8,      0,  0,  0x3F9,  0x3FA,  0,  0,  0}) );
        case SANDY_BRIDGE:
        case JAKETOWN:
        case IVY_BRIDGE:
        case IVYTOWN:
            PCM_CSTATE_ARRAY(pkgCStateMsr, PCM_PARAM_PROTECT({0,    0,  0x60D,  0x3F8,      0,  0,  0x3F9,  0x3FA,  0,  0,  0}) );
        case HASWELL:
        case HASWELL_2:
        case HASWELLX:
            PCM_CSTATE_ARRAY(pkgCStateMsr, PCM_PARAM_PROTECT({0,    0,  0x60D,  0x3F8,      0,  0,  0x3F9,  0x3FA,  0,  0,  0}) );
        case HASWELL_ULT:
            PCM_CSTATE_ARRAY(pkgCStateMsr, PCM_PARAM_PROTECT({0,    0,  0x60D,  0x3F8,      0,  0,  0x3F9,  0x3FA,  0x630,  0x631,  0x632}) );

        default:
            std::cerr << "PCM error: package C-states support array is not initialized. Package C-states metrics will not be shown." << std::endl;
            PCM_CSTATE_ARRAY(pkgCStateMsr, PCM_PARAM_PROTECT({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }) );
    };

    // fill core C state array
    switch(original_cpu_model)
    {
        case ATOM:
        case ATOM_2:
        case ATOM_CENTERTON:
            PCM_CSTATE_ARRAY(coreCStateMsr, PCM_PARAM_PROTECT({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }) );
        case NEHALEM_EP:
        case NEHALEM:
        case CLARKDALE:
        case WESTMERE_EP:
        case NEHALEM_EX:
        case WESTMERE_EX:
	        PCM_CSTATE_ARRAY(coreCStateMsr, PCM_PARAM_PROTECT({0,	0,	0,	0x3FC,	0,	0,	0x3FD,      0,  0,	0,	0}) );
        case SANDY_BRIDGE:
        case JAKETOWN:
        case IVY_BRIDGE:
        case IVYTOWN:
        case HASWELL:
        case HASWELL_2:
        case HASWELL_ULT:
        case HASWELLX:
        case ATOM_BAYTRAIL:
        case ATOM_AVOTON:
            PCM_CSTATE_ARRAY(coreCStateMsr, PCM_PARAM_PROTECT({0,	0,	0,	0x3FC,	0,	0,	0x3FD,	0x3FE,	0,	0,	0}) );
        default:
            std::cerr << "PCM error: core C-states support array is not initialized. Core C-states metrics will not be shown." << std::endl;
            PCM_CSTATE_ARRAY(coreCStateMsr, PCM_PARAM_PROTECT({ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }) );
    };
}

bool PCM::discoverSystemTopology()
{
    
#ifdef _MSC_VER
// version for Windows

#ifdef COMPILE_FOR_WINDOWS_7
    DWORD GroupStart[5];     // at most 4 groups on Windows 7
    GroupStart[0] = 0;
    GroupStart[1] = GetActiveProcessorCount(0);
    GroupStart[2] = GroupStart[1] + GetActiveProcessorCount(1);
    GroupStart[3] = GroupStart[2] + GetActiveProcessorCount(2);
    GroupStart[4] = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (GroupStart[3] + GetActiveProcessorCount(3) != GetActiveProcessorCount(ALL_PROCESSOR_GROUPS))
    {
        std::cerr << "Error in processor group size counting (1)" << std::endl;
        std::cerr << "Make sure your binary is compiled for 64-bit: using 'x64' platform configuration." << std::endl;
        return false;
    }
    char * slpi = new char[sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)];
    DWORD len = sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX);
    DWORD res = GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)slpi, &len);

    while (res == FALSE)
    {
        delete[] slpi;

        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
        {
            slpi = new char[len];
            res = GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)slpi, &len);
        }
        else
        {
            _com_error error(GetLastError());
            std::wcerr << "Error in Windows function 'GetLogicalProcessorInformationEx': " <<
                GetLastError() <<  " " << error.ErrorMessage() << std::endl;
            return false;
        }
    }

    char * base_slpi = slpi;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX pi = NULL;

    for ( ; slpi < base_slpi + len; slpi += pi->Size)
    {
        pi = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)slpi;
        if (pi->Relationship == RelationProcessorCore)
        {
            threads_per_core = (pi->Processor.Flags == LTP_PC_SMT) ? 2 : 1;
            // std::cout << "thr per core: "<< threads_per_core << std::endl;
            num_cores += threads_per_core;
        }
    }


    if (num_cores != GetActiveProcessorCount(ALL_PROCESSOR_GROUPS))
    {
        std::cerr << "Error in processor group size counting: " << num_cores << "!=" << GetActiveProcessorCount(ALL_PROCESSOR_GROUPS) << std::endl;
        std::cerr << "Make sure your binary is compiled for 64-bit: using 'x64' platform configuration." << std::endl;
        return false;
    }

    topology.resize(num_cores);

    slpi = base_slpi;
    pi = NULL;

    for ( ; slpi < base_slpi + len; slpi += pi->Size)
    {
        pi = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)slpi;
        if (pi->Relationship == RelationNumaNode)
        {
            ++num_sockets;
            for (unsigned int c = 0; c < (unsigned int)num_cores; ++c)
            {
                // std::cout << "c:"<<c<<" GroupStart[slpi->NumaNode.GroupMask.Group]: "<<GroupStart[slpi->NumaNode.GroupMask.Group]<<std::endl;
                if (c < GroupStart[pi->NumaNode.GroupMask.Group] || c >= GroupStart[(pi->NumaNode.GroupMask.Group) + 1])
                {
                    //std::cout <<"core "<<c<<" is not in group "<< slpi->NumaNode.GroupMask.Group << std::endl;
                    continue;
                }
                if ((1LL << (c - GroupStart[pi->NumaNode.GroupMask.Group])) & pi->NumaNode.GroupMask.Mask)
                {
                    topology[c].core_id = c;
                    topology[c].os_id = c;
                    topology[c].socket = pi->NumaNode.NodeNumber;
                    // std::cout << "Core "<< c <<" is in NUMA node "<< topology[c].socket << " and belongs to processor group " << slpi->NumaNode.GroupMask.Group <<std::endl;
                }
            }
        }
    }

    delete[] base_slpi;

#else // windows, not windows 7
	int32 size = 1;
	SYSTEM_LOGICAL_PROCESSOR_INFORMATION * slpi = new SYSTEM_LOGICAL_PROCESSOR_INFORMATION[size];
    DWORD len = sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    DWORD res = GetLogicalProcessorInformation(slpi, &len);

    while (res == FALSE)
    {
        delete[] slpi;

        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
        {
			size = (int32) len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            slpi = new SYSTEM_LOGICAL_PROCESSOR_INFORMATION[size];
            res = GetLogicalProcessorInformation(slpi, &len);
        }
        else
        {
            _com_error error(GetLastError());
            std::wcerr << "Error in Windows function 'GetLogicalProcessorInformation': " <<
                GetLastError() << " " << error.ErrorMessage() << std::endl;
            return false;
        }
    }

    for (i = 0; i < size; ++i)
    {
        if (slpi[i].Relationship == RelationProcessorCore)
        {
            //std::cout << "Physical core found, mask: "<<slpi[i].ProcessorMask<< std::endl;
            threads_per_core = bitCount(slpi[i].ProcessorMask);
            num_cores += threads_per_core;
        }
    }
    topology.resize(num_cores);

    for (i = 0; i < size; ++i)
    {
        if (slpi[i].Relationship == RelationNumaNode)
        {
            //std::cout << "NUMA node "<<slpi[i].NumaNode.NodeNumber<<" cores: "<<slpi[i].ProcessorMask<< std::endl;
            ++num_sockets;
            for (int c = 0; c < num_cores; ++c)
            {
                if ((1LL << c) & slpi[i].ProcessorMask)
                {
                    topology[c].core_id = c;
                    topology[c].os_id = c;
                    topology[c].socket = slpi[i].NumaNode.NodeNumber;
                    //std::cout << "Core "<< c <<" is in NUMA node "<< topology[c].socket << std::endl;
                }
            }
        }
    }

    delete[] slpi;

#endif // end of COMPILE_FOR_WINDOWS_7 
#else
    // for Linux and Mac OS
    
    TopologyEntry entry;
    typedef std::map<uint32, uint32> socketIdMap_type;
    socketIdMap_type socketIdMap;

#ifdef __linux__

    FILE * f_presentcpus = fopen("/sys/devices/system/cpu/present", "r");
    if (!f_presentcpus)
    {
        std::cerr << "Can not open /sys/devices/system/cpu/present file." << std::endl;
        return false;
    }
    char buffer[1024];
    if(NULL == fgets(buffer, 1024, f_presentcpus))
    {
        std::cerr << "Can not read /sys/devices/system/cpu/present." << std::endl;
        return false;
    }
    fclose(f_presentcpus);
    num_cores = -1;
    sscanf(buffer, "0-%d", &num_cores);
    if(num_cores == -1)
    {
       sscanf(buffer, "%d", &num_cores);
    }
    if(num_cores == -1)
    {
      std::cerr << "Can not read number of present cores" << std::endl;
      return false;
    }
    ++num_cores;

    // open /proc/cpuinfo
    FILE * f_cpuinfo = fopen("/proc/cpuinfo", "r");
    if (!f_cpuinfo)
    {
        std::cerr << "Can not open /proc/cpuinfo file." << std::endl;
        return false;
    }

    topology.resize(num_cores);
    while (0 != fgets(buffer, 1024, f_cpuinfo))
    {
        if (strncmp(buffer, "processor", sizeof("processor") - 1) == 0)
        {
            if (entry.os_id >= 0)
            {
                topology[entry.os_id] = entry;
                ++num_online_cores;
                if (entry.socket == 0 && entry.core_id == 0) ++threads_per_core;
            }
            sscanf(buffer, "processor\t: %d", &entry.os_id);
            //std::cout << "os_core_id: "<<entry.os_id<< std::endl;
            continue;
        }
        if (strncmp(buffer, "physical id", sizeof("physical id") - 1) == 0)
        {
            sscanf(buffer, "physical id\t: %d", &entry.socket);
            //std::cout << "physical id: "<<entry.socket<< std::endl;
            socketIdMap[entry.socket] = 0;
            continue;
        }
        if (strncmp(buffer, "core id", sizeof("core id") - 1) == 0)
        {
            sscanf(buffer, "core id\t: %d", &entry.core_id);
            //std::cout << "core id: "<<entry.core_id<< std::endl;
            continue;
        }
        if (num_phys_cores_per_socket == 0 && strncmp(buffer, "cpu cores", sizeof("cpu cores") - 1) == 0)
        {
            sscanf(buffer, "cpu cores\t: %d", &num_phys_cores_per_socket);
            //std::cout << "cpu cores: " << num_phys_cores_per_socket << std::endl;
            continue;
        }
    }
    if (entry.os_id >= 0)
    {
        topology[entry.os_id] = entry;
        ++num_online_cores;
        if (entry.socket == 0 && entry.core_id == 0) ++threads_per_core;
    }
    fclose(f_cpuinfo);

#elif defined(__FreeBSD__) 

    size_t size = sizeof(num_cores);
    cpuctl_cpuid_args_t cpuid_args;
    int fd, apic_ids_per_package, apic_ids_per_core;

    if(0 != sysctlbyname("kern.smp.cpus", &num_cores, &size, NULL, 0))
    {
        std::cerr << "Unable to get kern.smp.cpus from sysctl." << std::endl;
        return false;
    }

    if (modfind("cpuctl") == -1)
    {
        std::cout << "cpuctl(4) not loaded." << std::endl;
        return false;
    }
 
    do_cpuid(1, cpuid_args.data);

    apic_ids_per_package = (cpuid_args.data[1] & 0x00FF0000) >> 16;

    if (apic_ids_per_package == 0)
    {
        std::cout << "apic_ids_per_package == 0" << std::endl;
        return false;
    }

    cpuid_count(0xb, 0x0, cpuid_args.data);

    if ((cpuid_args.data[2] & 0xFF00) == 0x100)
        apic_ids_per_core = cpuid_args.data[1] & 0xFFFF;
    else
        apic_ids_per_core = 1;

    if (apic_ids_per_core == 0)
    {
        std::cout << "apic_ids_per_core == 0" << std::endl;
        return false;
    }

    for (int i = 0; i < num_cores; i++)
    {
        char cpuctl_name[64];
        int apic_id;

        sprintf(cpuctl_name, "/dev/cpuctl%d", i);
        fd = ::open(cpuctl_name, O_RDWR);

        cpuid_args.level = 0xb;

        ::ioctl(fd, CPUCTL_CPUID, &cpuid_args);

        apic_id = cpuid_args.data[3];

        entry.os_id = i;
        entry.socket = apic_id / apic_ids_per_package;
        entry.core_id = (apic_id % apic_ids_per_package) / apic_ids_per_core;

        if (entry.socket == 0 && entry.core_id == 0) ++threads_per_core;

        topology.push_back(entry);
        socketIdMap[entry.socket] = 0;
    }

#else // Getting processor info for Mac OS
#define SAFE_SYSCTLBYNAME(message, ret_value)                                                              \
    {                                                                                                      \
        size_t size;                                                                                       \
        char *pParam;                                                                                      \
        if(0 != sysctlbyname(message, NULL, &size, NULL, 0))                                               \
        {                                                                                                  \
            std::cerr << "Unable to determine size of " << message << " sysctl return type." << std::endl; \
            return false;                                                                                  \
        }                                                                                                  \
        if(NULL == (pParam = (char *)malloc(size)))                                                        \
        {                                                                                                  \
            std::cerr << "Unable to allocate memory for " << message << std::endl;                         \
            return false;                                                                                  \
        }                                                                                                  \
        if(0 != sysctlbyname(message, (void*)pParam, &size, NULL, 0))                                      \
        {                                                                                                  \
            std::cerr << "Unable to get " << message << " from sysctl." << std::endl;                      \
            return false;                                                                                  \
        }                                                                                                  \
        ret_value = convertUnknownToInt(size, pParam);                                                     \
        free(pParam);                                                                                      \
    }
// End SAFE_SYSCTLBYNAME

    // Using OSXs sysctl to get the number of CPUs right away
    SAFE_SYSCTLBYNAME("hw.logicalcpu", num_cores) 
    
#undef SAFE_SYSCTLBYNAME
    
    // The OSX version needs the MSR handle earlier so that it can build the CPU topology.
    // This topology functionality should potentially go into a different KEXT
    MSR = new SafeMsrHandle *[num_cores];
    for(int i = 0; i < num_cores; i++)
    {
        MSR[i] = new SafeMsrHandle(i);
    } 

    TopologyEntry *entries = new TopologyEntry[num_cores];
    MSR[0]->buildTopology(num_cores, entries);
    for(int i = 0; i < num_cores; i++){
        socketIdMap[entries[i].socket] = 0;
        if(entries[i].os_id >= 0)
        {
            if(entries[i].core_id == 0 && entries[i].socket == 0) ++threads_per_core;
            topology.push_back(entries[i]);
        }
    }
	delete entries;
// End of OSX specific code
#endif // end of ifndef __APPLE__

    num_cores = topology.size();
    num_sockets = (std::max)(socketIdMap.size(), (size_t)1);


    socketIdMap_type::iterator s = socketIdMap.begin();
    for (uint sid = 0; s != socketIdMap.end(); ++s)
    {
        s->second = sid++;
    }

    for (int i = 0; i < num_cores; ++i)
    {
        if(isCoreOnline(i))
          topology[i].socket = socketIdMap[topology[i].socket];
    }

#if 0
    std::cerr << "Number of socket ids: " << socketIdMap.size() << "\n";
    std::cerr << "Topology:\nsocket os_id core_id\n";
    for (int i = 0; i < num_cores; ++i)
    {
        std::cerr << topology[i].socket << " " << topology[i].os_id << " " << topology[i].core_id << std::endl;
    }
#endif

#endif //end of ifdef _MSC_VER

    if(threads_per_core == 0)
    {
        for (int i = 0; i < num_cores; ++i)
        {
            if(topology[i].socket == topology[0].socket && topology[i].core_id == topology[0].core_id)
                ++threads_per_core;
        }
    }
    if(num_phys_cores_per_socket == 0) num_phys_cores_per_socket = num_cores / num_sockets / threads_per_core;
    if(num_online_cores == 0) num_online_cores = num_cores;

    int32 i = 0;

    socketRefCore.resize(num_sockets);
    for(i = 0; i < num_cores; ++i)
    {
        if(isCoreOnline(i))
        {
            socketRefCore[topology[i].socket] = i;
        }
    }

#if 0
    for(int32 i=0; i< num_sockets;++i)
    {
      std::cout << "socketRefCore["<< i << "]=" << socketRefCore[i] << std::endl;
    }
#endif    

    return true;
}

void PCM::printSystemTopology() const
{
    if(num_cores == num_online_cores)
    {
      std::cerr << "Number of physical cores: " << (num_cores/threads_per_core) << std::endl;
    }

    std::cerr << "Number of logical cores: " << num_cores << std::endl;
    std::cerr << "Number of online logical cores: " << num_online_cores << std::endl;

    if(num_cores == num_online_cores)
    {
      std::cerr << "Threads (logical cores) per physical core: " << threads_per_core << std::endl;
    }
    else
    {
        std::cerr << "Offlined cores: ";
        for (int i = 0; i < num_cores; ++i)
            if(isCoreOnline(i) == false)
                std::cerr << i << " ";
        std::cerr << std::endl;
    }
    std::cerr << "Num sockets: " << num_sockets << std::endl;
    std::cerr << "Physical cores per socket: " << num_phys_cores_per_socket << std::endl;
    std::cerr << "Core PMU (perfmon) version: " << perfmon_version << std::endl;
    std::cerr << "Number of core PMU generic (programmable) counters: " << core_gen_counter_num_max << std::endl;
    std::cerr << "Width of generic (programmable) counters: " << core_gen_counter_width << " bits" << std::endl;
    if (perfmon_version > 1)
    {
        std::cerr << "Number of core PMU fixed counters: " << core_fixed_counter_num_max << std::endl;
        std::cerr << "Width of fixed counters: " << core_fixed_counter_width << " bits" << std::endl;
    }
}

void PCM::initMSR()
{
    int32 i = 0;

#ifndef __APPLE__
    MSR = new SafeMsrHandle *[num_cores];
    try
    {
        for (i = 0; i < num_cores; ++i)
        {
            MSR[i] = isCoreOnline(i) ? (new SafeMsrHandle(i)):(new SafeMsrHandle()); // the core is offlined, assign an invalid MSR handle
        }
    }
    catch (...)
    {
        // failed
        for (int j = 0; j < i; j++)
        {
            if(MSR[j])
              delete MSR[j];
        }
        delete[] MSR;
        MSR = NULL;

        std::cerr << "Can not access CPUs Model Specific Registers (MSRs)." << std::endl;
#ifdef _MSC_VER
        std::cerr << "You must have signed msr.sys driver in your current directory and have administrator rights to run this program." << std::endl;
#elif defined(__linux__)
        std::cerr << "Try to execute 'modprobe msr' as root user and then" << std::endl;
        std::cerr << "you also must have read and write permissions for /dev/cpu/*/msr devices (/dev/msr* for Android). The 'chown' command can help." << std::endl;
#elif defined(__FreeBSD__)
        std::cerr << "Ensure cpuctl module is loaded and that you have read and write" << std::endl;
        std::cerr << "permissions for /dev/cpuctl* devices (the 'chown' command can help)." << std::endl;
#endif

    }
#endif
}

bool PCM::detectNominalFrequency()
{
    if (MSR)
    {
        uint64 freq = 0;
        MSR[socketRefCore[0]]->read(PLATFORM_INFO_ADDR, &freq);
        const uint64 bus_freq = (
                  cpu_model == SANDY_BRIDGE 
               || cpu_model == JAKETOWN 
               || cpu_model == IVYTOWN
               || cpu_model == HASWELLX
               || cpu_model == IVY_BRIDGE
               || cpu_model == HASWELL
               || original_cpu_model == ATOM_AVOTON
               ) ? (100000000ULL) : (133333333ULL);

        nominal_frequency = ((freq >> 8) & 255) * bus_freq;

        if(!nominal_frequency) 
        	nominal_frequency = get_frequency_from_cpuid();

        if(!nominal_frequency)
        {
            std::cerr << "Error: Can not detect core frequency." << std::endl;
            destroyMSR();
		    return false;
        }

        std::cerr << "Nominal core frequency: " << nominal_frequency << " Hz" << std::endl;
    }

    return true;
}

void PCM::initEnergyMonitoring()
{
    if(packageEnergyMetricsAvailable() && MSR)
    {
        uint64 rapl_power_unit = 0;
        MSR[socketRefCore[0]]->read(MSR_RAPL_POWER_UNIT,&rapl_power_unit);
        uint64 energy_status_unit = extract_bits(rapl_power_unit,8,12);
        joulesPerEnergyUnit = 1./double(1ULL<<energy_status_unit); // (1/2)^energy_status_unit
	//std::cout << "MSR_RAPL_POWER_UNIT: "<<energy_status_unit<<"; Joules/unit "<< joulesPerEnergyUnit << std::endl;
        uint64 power_unit = extract_bits(rapl_power_unit,0,3);
        double wattsPerPowerUnit = 1./double(1ULL<<power_unit);

        uint64 package_power_info = 0;
        MSR[socketRefCore[0]]->read(MSR_PKG_POWER_INFO,&package_power_info);
        pkgThermalSpecPower = (uint32) (double(extract_bits(package_power_info, 0, 14))*wattsPerPowerUnit);
        pkgMinimumPower = (uint32) (double(extract_bits(package_power_info, 16, 30))*wattsPerPowerUnit);
        pkgMaximumPower = (uint32) (double(extract_bits(package_power_info, 32, 46))*wattsPerPowerUnit);

        std::cerr << "Package thermal spec power: "<< pkgThermalSpecPower << " Watt; ";
        std::cerr << "Package minimum power: "<< pkgMinimumPower << " Watt; ";
        std::cerr << "Package maximum power: "<< pkgMaximumPower << " Watt; " << std::endl;

        int i = 0;

        if(snb_energy_status.empty())
	        for (i = 0; i < num_sockets; ++i)
		        snb_energy_status.push_back(new CounterWidthExtender(new CounterWidthExtender::MsrHandleCounter(MSR[socketRefCore[i]],MSR_PKG_ENERGY_STATUS)) );

        if(dramEnergyMetricsAvailable() && jkt_dram_energy_status.empty())
            for (i = 0; i < num_sockets; ++i)
                jkt_dram_energy_status.push_back(new CounterWidthExtender(new CounterWidthExtender::MsrHandleCounter(MSR[socketRefCore[i]],MSR_DRAM_ENERGY_STATUS)));
    }
}

void PCM::initUncoreObjects()
{
    if (hasPCICFGUncore() && MSR != NULL)
    {
        server_pcicfg_uncore = new ServerPCICFGUncore *[num_sockets];
        int i = 0;

        try
        {
            for (i = 0; i < num_sockets; ++i)
            {
                server_pcicfg_uncore[i] = new ServerPCICFGUncore(i, this);
            }
        }
        catch (...)
        {
            // failed
            for (int j = 0; j < i; j++)
                delete server_pcicfg_uncore[j];
            delete[] server_pcicfg_uncore;

            server_pcicfg_uncore = NULL;

            std::cerr << "Can not access Jaketown/Ivytown PCI configuration space. Access to uncore counters (memory and QPI bandwidth) is disabled." << std::endl;
                #ifdef _MSC_VER
            std::cerr << "You must have signed msr.sys driver in your current directory and have administrator rights to run this program." << std::endl;
                #else
            //std::cerr << "you must have read and write permissions for /proc/bus/pci/7f/10.* and /proc/bus/pci/ff/10.* devices (the 'chown' command can help)." << std::endl;
            //std::cerr << "you must have read and write permissions for /dev/mem device (the 'chown' command can help)."<< std::endl;
            //std::cerr << "you must have read permission for /sys/firmware/acpi/tables/MCFG device (the 'chmod' command can help)."<< std::endl;
            std::cerr << "You must be root to access these Jaketown/Ivytown counters in PCM. " << std::endl;
                #endif
        }
    } else if((cpu_model == SANDY_BRIDGE || cpu_model == IVY_BRIDGE || cpu_model == HASWELL) && MSR != NULL)
    {
       // initialize memory bandwidth counting
       try
       {
           clientBW = new ClientBW();
           clientImcReads = new CounterWidthExtender(new CounterWidthExtender::ClientImcReadsCounter(clientBW));
           clientImcWrites = new CounterWidthExtender(new CounterWidthExtender::ClientImcWritesCounter(clientBW));
           clientIoRequests = new CounterWidthExtender(new CounterWidthExtender::ClientIoRequestsCounter(clientBW));

       } catch(...)
       {
           std::cerr << "Can not read memory controller counter information from PCI configuration space. Access to memory bandwidth counters is not possible." << std::endl;
           #ifdef _MSC_VER
           // TODO: add message here
           #endif
           #ifdef __linux__
           std::cerr << "You must be root to access these SandyBridge/IvyBridge/Haswell counters in PCM. " << std::endl;
           #endif 
       }
    }

    switch(cpu_model)
    {
        case IVYTOWN:
        case JAKETOWN:
            PCU_MSR_PMON_BOX_CTL_ADDR = JKTIVT_PCU_MSR_PMON_BOX_CTL_ADDR;
            PCU_MSR_PMON_CTRX_ADDR[0] = JKTIVT_PCU_MSR_PMON_CTR0_ADDR;
            PCU_MSR_PMON_CTRX_ADDR[1] = JKTIVT_PCU_MSR_PMON_CTR1_ADDR;
            PCU_MSR_PMON_CTRX_ADDR[2] = JKTIVT_PCU_MSR_PMON_CTR2_ADDR;
            PCU_MSR_PMON_CTRX_ADDR[3] = JKTIVT_PCU_MSR_PMON_CTR3_ADDR;
            break;
        case HASWELLX:
            PCU_MSR_PMON_BOX_CTL_ADDR = HSX_PCU_MSR_PMON_BOX_CTL_ADDR;
            PCU_MSR_PMON_CTRX_ADDR[0] = HSX_PCU_MSR_PMON_CTR0_ADDR;
            PCU_MSR_PMON_CTRX_ADDR[1] = HSX_PCU_MSR_PMON_CTR1_ADDR;
            PCU_MSR_PMON_CTRX_ADDR[2] = HSX_PCU_MSR_PMON_CTR2_ADDR;
            PCU_MSR_PMON_CTRX_ADDR[3] = HSX_PCU_MSR_PMON_CTR3_ADDR;
            break;
        default:
            PCU_MSR_PMON_BOX_CTL_ADDR = 0;
            PCU_MSR_PMON_CTRX_ADDR[0] = 0;
            PCU_MSR_PMON_CTRX_ADDR[1] = 0;
            PCU_MSR_PMON_CTRX_ADDR[2] = 0;
            PCU_MSR_PMON_CTRX_ADDR[3] = 0;
    }


}

#ifdef __linux__

bool isNMIWatchdogEnabled()
{
    FILE * f = fopen("/proc/sys/kernel/nmi_watchdog", "r");
    if (!f)
    {
        return false;
    }

    char buffer[1024];
    if(NULL == fgets(buffer, 1024, f))
    {
        std::cerr << "Can not read /proc/sys/kernel/nmi_watchdog ." << std::endl;
        fclose(f);
        return true;
    }
    int enabled = 1;
    sscanf(buffer, "%d", &enabled);
    fclose(f);

    if(enabled == 1)
    {
       std::cerr << "Error: NMI watchdog is enabled. This consumes one hw-PMU counter" << std::endl;
       std::cerr << " to disable NMI watchdog please run under root: echo 0 > /proc/sys/kernel/nmi_watchdog"<< std::endl;
       std::cerr << " or to disable it permanently: echo 'kernel.nmi_watchdog=0' >> /etc/sysctl.conf "<< std::endl;
    }

    return (enabled == 1);
}

#endif

PCM::PCM() :
    cpu_family(-1),
    cpu_model(-1),
    original_cpu_model(-1),
    threads_per_core(0),
    num_cores(0),
    num_sockets(0),
    num_phys_cores_per_socket(0),
    num_online_cores(0),
    core_gen_counter_num_max(0),
    core_gen_counter_num_used(0), // 0 means no core gen counters used
    core_gen_counter_width(0),
    core_fixed_counter_num_max(0),
    core_fixed_counter_num_used(0),
    core_fixed_counter_width(0),
    uncore_gen_counter_num_max(8),
    uncore_gen_counter_num_used(0),
    uncore_gen_counter_width(48),
    uncore_fixed_counter_num_max(1),
    uncore_fixed_counter_num_used(0),
    uncore_fixed_counter_width(48),
    perfmon_version(0),
    perfmon_config_anythread(1),
    nominal_frequency(0),
    max_qpi_speed(0),
    pkgThermalSpecPower(-1),
    pkgMinimumPower(-1), 
    pkgMaximumPower(-1),
    allow_multiple_instances(false),
    programmed_pmu(false),
    MSR(NULL),
    server_pcicfg_uncore(NULL),
    clientBW(NULL),
    clientImcReads(NULL),
    clientImcWrites(NULL),
    clientIoRequests(NULL),
    disable_JKT_workaround(false),
    blocked(false),
    coreCStateMsr(NULL),
    pkgCStateMsr(NULL),
    mode(INVALID_MODE),
    numInstancesSemaphore(NULL),
    canUsePerf(false),
    outfile(NULL),
    backup_ofile(NULL),
    run_state(1)
{

#ifdef _MSC_VER
    TCHAR driverPath[1040]; // length for current directory + "\\msr.sys"
    GetCurrentDirectory(1024, driverPath);
    wcscat_s(driverPath, 1040, L"\\msr.sys");
    // WARNING: This driver code (msr.sys) is only for testing purposes, not for production use
    Driver drv;
    // drv.stop();     // restart driver (usually not needed)
    if (!drv.start(driverPath))
    {
        std::cerr << "Cannot access CPU counters" << std::endl;
        std::cerr << "You must have signed msr.sys driver in your current directory and have administrator rights to run this program" << std::endl;
		return;
    }
#endif

#ifdef __linux__
    if(isNMIWatchdogEnabled()) return;
#endif

    if(!detectModel()) return;

    if(!checkModel()) return;

    initCStateSupportTables();

    if(!discoverSystemTopology()) return;

    printSystemTopology();

    initMSR();

    if(!detectNominalFrequency()) return;

    initEnergyMonitoring();

    initUncoreObjects();

#ifdef PCM_USE_PERF
    canUsePerf = true;
    std::vector<int> dummy(PERF_MAX_COUNTERS, -1);
    perfEventHandle.resize(num_cores, dummy);
#endif
}

void PCM::enableJKTWorkaround(bool enable)
{
    if(disable_JKT_workaround) return;
    std::cerr << "Using PCM on your system might have a performance impact as per http://software.intel.com/en-us/articles/performance-impact-when-sampling-certain-llc-events-on-snb-ep-with-vtune" << std::endl;
    std::cerr << "You can avoid the performance impact by using the option --noJKTWA, however the cache metrics might be wrong then." << std::endl;
    if(MSR)
    {
        for(int32 i = 0; i < num_cores; ++i)
        {
            uint64 val64 = 0;
            MSR[i]->read(0x39C, &val64);
            if(enable)
                val64 |= 1ULL;
            else
                val64 &= (~1ULL);
            MSR[i]->write(0x39C, val64);
        }
    }
    if(server_pcicfg_uncore)
    {
        for (int32 i = 0; i < num_sockets; ++i)
        {
                if(server_pcicfg_uncore[i]) server_pcicfg_uncore[i]->enableJKTWorkaround(enable);
        }
    }
}

bool PCM::isCoreOnline(int32 os_core_id) const
{
    return (topology[os_core_id].os_id != -1) && (topology[os_core_id].core_id != -1) && (topology[os_core_id].socket != -1);
}

bool PCM::isCPUModelSupported(int model_)
{
    return (   model_ == NEHALEM_EP
            || model_ == NEHALEM_EX
            || model_ == WESTMERE_EP
            || model_ == WESTMERE_EX
            || model_ == ATOM
            || model_ == CLARKDALE
            || model_ == SANDY_BRIDGE
            || model_ == JAKETOWN
            || model_ == IVY_BRIDGE
            || model_ == HASWELL
            || model_ == IVYTOWN
            || model_ == HASWELLX
           );
}

bool PCM::checkModel()
{
    if (cpu_model == NEHALEM) cpu_model = NEHALEM_EP;
    if (cpu_model == ATOM_2 || cpu_model == ATOM_CENTERTON || cpu_model == ATOM_BAYTRAIL || cpu_model == ATOM_AVOTON) cpu_model = ATOM;
    if (cpu_model == HASWELL_ULT || cpu_model == HASWELL_2) cpu_model = HASWELL;

    if(!isCPUModelSupported(cpu_model))
    {
        std::cerr << getUnsupportedMessage() << " CPU model number: " << cpu_model << " Brand: \"" << getCPUBrandString().c_str() <<"\""<< std::endl;
/* FOR TESTING PURPOSES ONLY */
#ifdef PCM_TEST_FALLBACK_TO_ATOM
        std::cerr << "Fall back to ATOM functionality." << std::endl;
        cpu_model = ATOM;
        return true;
#endif
        return false;
    }
    return true;
}

void PCM::destroyMSR()
{
        if (MSR)
        {
            for (int i = 0; i < num_cores; ++i)
                if (MSR[i]) delete MSR[i];
            delete[] MSR;
            MSR = NULL;
        }
}

PCM::~PCM()
{
    InstanceLock lock(allow_multiple_instances);
    if (instance)
    {
        destroyMSR();

        if (server_pcicfg_uncore)
        {
            for (int i = 0; i < num_sockets; ++i)
                if (server_pcicfg_uncore[i]) delete server_pcicfg_uncore[i];
            delete[] server_pcicfg_uncore;
        }
	for (uint32 i = 0; i < snb_energy_status.size(); ++i)
	{
	    delete  snb_energy_status[i];
	}
        for (uint32 i = 0; i < jkt_dram_energy_status.size(); ++i)
            delete jkt_dram_energy_status[i];
	
        instance = NULL;
        
        if(clientImcReads) delete clientImcReads;
        clientImcReads = NULL;
        if(clientImcWrites) delete clientImcWrites;
        clientImcWrites = NULL;
        if(clientIoRequests) delete clientIoRequests;
        clientIoRequests = NULL;
        if(clientBW) delete clientBW;
        clientBW = NULL;
    }
}

bool PCM::good()
{
    return MSR != NULL;
}

class TemporalThreadAffinity  // speedup trick for Linux
{
#ifdef __linux__
    cpu_set_t old_affinity;
    TemporalThreadAffinity(); // forbiden

public:
    TemporalThreadAffinity(uint32 core_id)
    {
        pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &old_affinity);

        cpu_set_t new_affinity;
        CPU_ZERO(&new_affinity);
        CPU_SET(core_id, &new_affinity);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &new_affinity);
    }
    ~TemporalThreadAffinity()
    {
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &old_affinity);
    }
#else // not implemented for windows or os x
    TemporalThreadAffinity(); // forbiden

public:
    TemporalThreadAffinity(uint32) { }
#endif
};

#ifdef PCM_USE_PERF
perf_event_attr PCM_init_perf_event_attr()
{
    perf_event_attr e;
    bzero(&e,sizeof(perf_event_attr));
    e.type = -1; // must be set up later
    e.size = sizeof(e);
    e.config = -1; // must be set up later
    e.sample_period = 0;
    e.sample_type = 0;
    e.read_format = PERF_FORMAT_GROUP; /* PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING |
                          PERF_FORMAT_ID | PERF_FORMAT_GROUP ; */
    e.disabled = 0;
    e.inherit = 0;
    e.pinned = 1;
    e.exclusive = 0;
    e.exclude_user = 0;
    e.exclude_kernel = 0;
    e.exclude_hv = 0;
    e.exclude_idle = 0;
    e.mmap = 0;
    e.comm = 0;
    e.freq = 0;
    e.inherit_stat = 0;
    e.enable_on_exec = 0;
    e.task = 0;
    e.watermark = 0;
    e.wakeup_events = 0;
    return e;
}               
#endif

PCM::ErrorCode PCM::program(const PCM::ProgramMode mode_, const void * parameter_)
{
    if(allow_multiple_instances && (EXT_CUSTOM_CORE_EVENTS == mode_ || CUSTOM_CORE_EVENTS == mode_))
    {
        allow_multiple_instances = false;
        std::cerr << "Warning: multiple PCM instance mode is not allowed with custom events." << std::endl;
    }

    InstanceLock lock(allow_multiple_instances);
    if (!MSR) return PCM::MSRAccessDenied;
    
    ExtendedCustomCoreEventDescription * pExtDesc = (ExtendedCustomCoreEventDescription *)parameter_;

#ifdef PCM_USE_PERF
    std::cerr << "Trying to use Linux perf events..." << std::endl;
    if(PERF_COUNT_HW_MAX <= PCM_PERF_COUNT_HW_REF_CPU_CYCLES)
    {
        canUsePerf = false;
        std::cerr << "Can not use Linux perf because your Linux kernel does not support PERF_COUNT_HW_REF_CPU_CYCLES event. Falling-back to direct PMU programming." << std::endl;
    }
    if(EXT_CUSTOM_CORE_EVENTS == mode_ && pExtDesc && pExtDesc->fixedCfg)
    {
        canUsePerf = false;
        std::cerr << "Can not use Linux perf because non-standard fixed counter configuration requested. Falling-back to direct PMU programming." << std::endl;
    }
    if(EXT_CUSTOM_CORE_EVENTS == mode_ && pExtDesc && (pExtDesc->OffcoreResponseMsrValue[0] || pExtDesc->OffcoreResponseMsrValue[1]))
    {
        canUsePerf = false;
        std::cerr << "Can not use Linux perf because OffcoreResponse counter usage requested. Falling-back to direct PMU programming." << std::endl;
    }
#endif

    if(allow_multiple_instances)
    {
        //std::cerr << "Checking for other instances of PCM..." << std::endl;
    #ifdef _MSC_VER

        numInstancesSemaphore = CreateSemaphore(NULL, 0, 1 << 20, L"Global\\Number of running Intel Processor Counter Monitor instances");
        if (!numInstancesSemaphore)
        {
            _com_error error(GetLastError());
            std::wcerr << "Error in Windows function 'CreateSemaphore': " << GetLastError() << " " << error.ErrorMessage() << std::endl;
            return PCM::UnknownError;
        }
        LONG prevValue = 0;
        if (!ReleaseSemaphore(numInstancesSemaphore, 1, &prevValue))
        {
            _com_error error(GetLastError());
            std::wcerr << "Error in Windows function 'ReleaseSemaphore': " << GetLastError() << " " << error.ErrorMessage() << std::endl;
            return PCM::UnknownError;
        }
        if (prevValue > 0)  // already programmed since another instance exists
        {
            std::cerr << "Number of PCM instances: " << (prevValue + 1) << std::endl;
            if (hasPCICFGUncore() && server_pcicfg_uncore && server_pcicfg_uncore[0] && max_qpi_speed==0)
               for (int i = 0; i < num_sockets; ++i)
                  max_qpi_speed = (std::max)(server_pcicfg_uncore[i]->computeQPISpeed(socketRefCore[i],cpu_model), max_qpi_speed); // parenthesis to avoid macro expansion on Windows

            reportQPISpeed();
            return PCM::Success;
        }

    #else // if linux or apple
        numInstancesSemaphore = sem_open(PCM_NUM_INSTANCES_SEMAPHORE_NAME, O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO, 0);
        if (SEM_FAILED == numInstancesSemaphore)
        {
            if (EACCES == errno)
                std::cerr << "PCM Error, do not have permissions to open semaphores in /dev/shm/. Clean up them." << std::endl;
            return PCM::UnknownError;
        }
    #ifndef __APPLE__
        sem_post(numInstancesSemaphore);
        int curValue = 0;
        sem_getvalue(numInstancesSemaphore, &curValue);
    #else //if it is apple
        uint32 curValue = PCM::incrementNumInstances();
        sem_post(numInstancesSemaphore);
    #endif // end ifndef __APPLE__

        if (curValue > 1)  // already programmed since another instance exists
        {
            std::cerr << "Number of PCM instances: " << curValue << std::endl;
            if (hasPCICFGUncore() && server_pcicfg_uncore && server_pcicfg_uncore[0] && max_qpi_speed==0)
			    for (int i = 0; i < num_sockets; ++i) {
                    max_qpi_speed = std::max(server_pcicfg_uncore[i]->computeQPISpeed(socketRefCore[i],cpu_model), max_qpi_speed);
		    reportQPISpeed();
            }
            if(!canUsePerf) return PCM::Success;
        }

    #endif // end ifdef _MSC_VER

    #ifdef PCM_USE_PERF
    /* 
    numInst>1 &&  canUsePerf==false -> not reachable, already PMU programmed in another PCM instance
    numInst>1 &&  canUsePerf==true  -> perf programmed in different PCM, is not allowed 
    numInst<=1 && canUsePerf==false -> we are first, perf cannot be used, *check* if PMU busy
    numInst<=1 && canUsePerf==true -> we are first, perf will be used, *dont check*, this is now perf business
    */
        if(curValue > 1 && (canUsePerf == true))
        {
            std::cerr << "Running several clients using the same counters is not posible with Linux perf. Recompile PCM without Linux Perf support to allow such usage. " << std::endl;
            decrementInstanceSemaphore();
            return PCM::UnknownError;
        }

        if((curValue <= 1) && (canUsePerf == false) && PMUinUse()) 
        {
            decrementInstanceSemaphore();
            return PCM::PMUBusy;
        }
    #else
        if (PMUinUse())
        {
            decrementInstanceSemaphore();
            return PCM::PMUBusy;
        }
    #endif
    }
    else
    {
        if((canUsePerf == false) && PMUinUse())
        {
            return PCM::PMUBusy;
        }
    }

    mode = mode_;
    
    // copy custom event descriptions
    if (mode == CUSTOM_CORE_EVENTS)
    {
        if (!parameter_)
		{
            std::cerr << "PCM Internal Error: data structure for custom event not initialized" << std::endl;
            return PCM::UnknownError;
		}
        CustomCoreEventDescription * pDesc = (CustomCoreEventDescription *)parameter_;
        coreEventDesc[0] = pDesc[0];
        coreEventDesc[1] = pDesc[1];
        if (cpu_model != ATOM)
        {
            coreEventDesc[2] = pDesc[2];
            coreEventDesc[3] = pDesc[3];
            core_gen_counter_num_used = 4;
        }
        else
            core_gen_counter_num_used = 2;
    }
    else
    {
        if (cpu_model == ATOM)
        {
            coreEventDesc[0].event_number = ARCH_LLC_MISS_EVTNR;
            coreEventDesc[0].umask_value = ARCH_LLC_MISS_UMASK;
            coreEventDesc[1].event_number = ARCH_LLC_REFERENCE_EVTNR;
            coreEventDesc[1].umask_value = ARCH_LLC_REFERENCE_UMASK;
            core_gen_counter_num_used = 2;
        }
        else if (
               SANDY_BRIDGE == cpu_model 
            || JAKETOWN == cpu_model 
            || IVYTOWN == cpu_model
            || IVY_BRIDGE == cpu_model
            || HASWELL == cpu_model
            || HASWELLX == cpu_model
                )
        {
            coreEventDesc[0].event_number = ARCH_LLC_MISS_EVTNR;
            coreEventDesc[0].umask_value = ARCH_LLC_MISS_UMASK;
            coreEventDesc[1].event_number = MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_NONE_EVTNR;
            coreEventDesc[1].umask_value = MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_NONE_UMASK;
            coreEventDesc[2].event_number = MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_EVTNR;
            coreEventDesc[2].umask_value = MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_UMASK;
            coreEventDesc[3].event_number = MEM_LOAD_UOPS_RETIRED_L2_HIT_EVTNR;
            coreEventDesc[3].umask_value = MEM_LOAD_UOPS_RETIRED_L2_HIT_UMASK;
            core_gen_counter_num_used = 4;
        }
        else
        {   // Nehalem or Westmere
	    if(
               NEHALEM_EP == cpu_model 
            || WESTMERE_EP == cpu_model 
            || CLARKDALE == cpu_model
            )
	    {
		coreEventDesc[0].event_number = MEM_LOAD_RETIRED_L3_MISS_EVTNR;
		coreEventDesc[0].umask_value = MEM_LOAD_RETIRED_L3_MISS_UMASK;
	    }
	    else
	    {
		coreEventDesc[0].event_number = ARCH_LLC_MISS_EVTNR;
		coreEventDesc[0].umask_value = ARCH_LLC_MISS_UMASK;
	    }
            coreEventDesc[1].event_number = MEM_LOAD_RETIRED_L3_UNSHAREDHIT_EVTNR;
            coreEventDesc[1].umask_value = MEM_LOAD_RETIRED_L3_UNSHAREDHIT_UMASK;
            coreEventDesc[2].event_number = MEM_LOAD_RETIRED_L2_HITM_EVTNR;
            coreEventDesc[2].umask_value = MEM_LOAD_RETIRED_L2_HITM_UMASK;
            coreEventDesc[3].event_number = MEM_LOAD_RETIRED_L2_HIT_EVTNR;
            coreEventDesc[3].umask_value = MEM_LOAD_RETIRED_L2_HIT_UMASK;
            core_gen_counter_num_used = 4;
        }
    }

    core_fixed_counter_num_used = 3;
    
    if(EXT_CUSTOM_CORE_EVENTS == mode_ && pExtDesc && pExtDesc->gpCounterCfg)
    {
        core_gen_counter_num_used = (std::min)(core_gen_counter_num_used,pExtDesc->nGPCounters);
    }

    if(cpu_model == JAKETOWN)
    {
        bool enableWA = false;
        for(uint32 i = 0; i< core_gen_counter_num_used; ++i)
        {
            if(coreEventDesc[i].event_number == MEM_LOAD_UOPS_LLC_HIT_RETIRED_XSNP_EVTNR)
                enableWA = true;
        }
        enableJKTWorkaround(enableWA); // this has a performance penalty on memory access
    }

    programmed_pmu = true;

    // Version for linux/windows
    for (int i = 0; i < num_cores; ++i)
    {
        // program core counters

        TemporalThreadAffinity tempThreadAffinity(i); // speedup trick for Linux

      FixedEventControlRegister ctrl_reg;
#ifdef PCM_USE_PERF      
      int leader_counter = -1; 
      perf_event_attr e = PCM_init_perf_event_attr();
      if(canUsePerf)
      {
        e.type = PERF_TYPE_HARDWARE;
        e.config = (((long long unsigned)PERF_TYPE_HARDWARE)<<(64ULL-8ULL)) + PERF_COUNT_HW_INSTRUCTIONS;
        if((perfEventHandle[i][PERF_INST_RETIRED_ANY_POS] = syscall(SYS_perf_event_open, &e, -1,
                   i /* core id */, leader_counter /* group leader */ ,0 )) <= 0)
        {
          std::cerr <<"Linux Perf: Error on programming INST_RETIRED_ANY: "<<strerror(errno)<< std::endl;
          decrementInstanceSemaphore();
          return PCM::UnknownError;
        }
        leader_counter = perfEventHandle[i][PERF_INST_RETIRED_ANY_POS];
        e.pinned = 0; // all following counter are not leaders, thus need not be pinned explicitly
        e.config = (((long long unsigned)PERF_TYPE_HARDWARE)<<(64ULL-8ULL)) + PERF_COUNT_HW_CPU_CYCLES;
        if( (perfEventHandle[i][PERF_CPU_CLK_UNHALTED_THREAD_POS] = syscall(SYS_perf_event_open, &e, -1,
                                                i /* core id */, leader_counter /* group leader */ ,0 )) <= 0)
        {
          std::cerr <<"Linux Perf: Error on programming CPU_CLK_UNHALTED_THREAD: "<<strerror(errno)<< std::endl;
          decrementInstanceSemaphore();
          return PCM::UnknownError;
        }
        e.config = (((long long unsigned)PERF_TYPE_HARDWARE)<<(64ULL-8ULL)) + PCM_PERF_COUNT_HW_REF_CPU_CYCLES;
        if((perfEventHandle[i][PERF_CPU_CLK_UNHALTED_REF_POS] = syscall(SYS_perf_event_open, &e, -1,
                         i /* core id */, leader_counter /* group leader */ ,0 )) <= 0)
        {
          std::cerr <<"Linux Perf: Error on programming CPU_CLK_UNHALTED_REF: "<<strerror(errno)<< std::endl;
          decrementInstanceSemaphore();
          return PCM::UnknownError;
        }
      }
      else
#endif        
      {
        // disable counters while programming
        MSR[i]->write(IA32_CR_PERF_GLOBAL_CTRL, 0);
        MSR[i]->read(IA32_CR_FIXED_CTR_CTRL, &ctrl_reg.value);

	
	if(EXT_CUSTOM_CORE_EVENTS == mode_ && pExtDesc && pExtDesc->fixedCfg)
	{
	  ctrl_reg = *(pExtDesc->fixedCfg);
	}
	else
	{
	  ctrl_reg.fields.os0 = 1;
	  ctrl_reg.fields.usr0 = 1;
	  ctrl_reg.fields.any_thread0 = 0;
	  ctrl_reg.fields.enable_pmi0 = 0;

	  ctrl_reg.fields.os1 = 1;
	  ctrl_reg.fields.usr1 = 1;
	  ctrl_reg.fields.any_thread1 = 0;
	  ctrl_reg.fields.enable_pmi1 = 0;

	  ctrl_reg.fields.os2 = 1;
	  ctrl_reg.fields.usr2 = 1;
	  ctrl_reg.fields.any_thread2 = 0;
	  ctrl_reg.fields.enable_pmi2 = 0;	

	  ctrl_reg.fields.reserved1 = 0;
	}

        MSR[i]->write(IA32_CR_FIXED_CTR_CTRL, ctrl_reg.value); 
      }

        if(EXT_CUSTOM_CORE_EVENTS == mode_ && pExtDesc)
        {
            if(pExtDesc->OffcoreResponseMsrValue[0])
                MSR[i]->write(MSR_OFFCORE_RSP0, pExtDesc->OffcoreResponseMsrValue[0]);
            if(pExtDesc->OffcoreResponseMsrValue[1])
                MSR[i]->write(MSR_OFFCORE_RSP1, pExtDesc->OffcoreResponseMsrValue[1]);
        }

        EventSelectRegister event_select_reg;

        for (uint32 j = 0; j < core_gen_counter_num_used; ++j)
        {
            if(EXT_CUSTOM_CORE_EVENTS == mode_ && pExtDesc && pExtDesc->gpCounterCfg)
            {
              event_select_reg = pExtDesc->gpCounterCfg[j];
            }
            else
            {
              MSR[i]->read(IA32_PERFEVTSEL0_ADDR + j, &event_select_reg.value); // read-only also safe for perf
              
              event_select_reg.fields.event_select = coreEventDesc[j].event_number;
              event_select_reg.fields.umask = coreEventDesc[j].umask_value;
              event_select_reg.fields.usr = 1;
              event_select_reg.fields.os = 1;
              event_select_reg.fields.edge = 0;
              event_select_reg.fields.pin_control = 0;
              event_select_reg.fields.apic_int = 0;
              event_select_reg.fields.any_thread = 0;
              event_select_reg.fields.enable = 1;
              event_select_reg.fields.invert = 0;
              event_select_reg.fields.cmask = 0;
              event_select_reg.fields.in_tx = 0;
              event_select_reg.fields.in_txcp = 0;
            }
#ifdef PCM_USE_PERF            
            if(canUsePerf)
            {
                e.type = PERF_TYPE_RAW;
                e.config = (1ULL<<63ULL) + (((long long unsigned)PERF_TYPE_RAW)<<(64ULL-8ULL)) + event_select_reg.value;
                if((perfEventHandle[i][PERF_GEN_EVENT_0_POS + j] = syscall(SYS_perf_event_open, &e, -1,
                                                i /* core id */, leader_counter /* group leader */ ,0 )) <= 0)
                {
                        std::cerr <<"Linux Perf: Error on programming generic event #"<< i <<" error: "<<strerror(errno)<< std::endl;
                        decrementInstanceSemaphore();
                        return PCM::UnknownError;
                }
            }
            else
#endif              
            {
              MSR[i]->write(IA32_PMC0 + j, 0);
              MSR[i]->write(IA32_PERFEVTSEL0_ADDR + j, event_select_reg.value);
            }
        }

        if(!canUsePerf)
        {
          // start counting, enable all (4 programmable + 3 fixed) counters
          uint64 value = (1ULL << 0) + (1ULL << 1) + (1ULL << 2) + (1ULL << 3) + (1ULL << 32) + (1ULL << 33) + (1ULL << 34);

          if (cpu_model == ATOM)       // Atom has only 2 programmable counters
              value = (1ULL << 0) + (1ULL << 1) + (1ULL << 32) + (1ULL << 33) + (1ULL << 34);

          MSR[i]->write(IA32_CR_PERF_GLOBAL_CTRL, value);
        }

        // program uncore counters

        if (cpu_model == NEHALEM_EP || cpu_model == WESTMERE_EP || cpu_model == CLARKDALE)
        {
            programNehalemEPUncore(i);
        }
        else if (hasBecktonUncore())
        {
            programBecktonUncore(i);
        }
    }
    
    if(canUsePerf)
    {
      std::cerr << "Successfully programmed on-core PMU using Linux perf"<<std::endl;
    }

    if (hasPCICFGUncore() && server_pcicfg_uncore)
    {
        for (int i = 0; i < num_sockets; ++i)
        {
            server_pcicfg_uncore[i]->program();
            max_qpi_speed = (std::max)(server_pcicfg_uncore[i]->computeQPISpeed(socketRefCore[i],cpu_model), max_qpi_speed); // parenthesis to avoid macro expansion on Windows
        }
    }

    reportQPISpeed();

    return PCM::Success;
}

void PCM::reportQPISpeed() const
{
    if (!max_qpi_speed) return;
 
    if (hasPCICFGUncore() && server_pcicfg_uncore && server_pcicfg_uncore[0]) {
       for (int i = 0; i < num_sockets; ++i)
       {
          std::cerr << "Socket " << i << std::endl;
          server_pcicfg_uncore[i]->reportQPISpeed();
       }
    } else {
          std::cerr << "Max QPI speed: " << max_qpi_speed / (1e9) << " GBytes/second (" << max_qpi_speed / (2e9) << " GT/second)" << std::endl;
    }

}

void PCM::programNehalemEPUncore(int32 core)
{

#define CPUCNT_INIT_THE_REST_OF_EVTCNT \
    unc_event_select_reg.fields.occ_ctr_rst = 1; \
    unc_event_select_reg.fields.edge = 0; \
    unc_event_select_reg.fields.enable_pmi = 0; \
    unc_event_select_reg.fields.enable = 1; \
    unc_event_select_reg.fields.invert = 0; \
    unc_event_select_reg.fields.cmask = 0;

            uncore_gen_counter_num_used = 8;

            UncoreEventSelectRegister unc_event_select_reg;

            MSR[core]->read(MSR_UNCORE_PERFEVTSEL0_ADDR, &unc_event_select_reg.value);

            unc_event_select_reg.fields.event_select = UNC_QMC_WRITES_FULL_ANY_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QMC_WRITES_FULL_ANY_UMASK;

            CPUCNT_INIT_THE_REST_OF_EVTCNT

                MSR[core]->write(MSR_UNCORE_PERFEVTSEL0_ADDR, unc_event_select_reg.value);


            MSR[core]->read(MSR_UNCORE_PERFEVTSEL1_ADDR, &unc_event_select_reg.value);

            unc_event_select_reg.fields.event_select = UNC_QMC_NORMAL_READS_ANY_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QMC_NORMAL_READS_ANY_UMASK;

            CPUCNT_INIT_THE_REST_OF_EVTCNT

                MSR[core]->write(MSR_UNCORE_PERFEVTSEL1_ADDR, unc_event_select_reg.value);


            MSR[core]->read(MSR_UNCORE_PERFEVTSEL2_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_IOH_READS_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[core]->write(MSR_UNCORE_PERFEVTSEL2_ADDR, unc_event_select_reg.value);

            MSR[core]->read(MSR_UNCORE_PERFEVTSEL3_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_IOH_WRITES_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[core]->write(MSR_UNCORE_PERFEVTSEL3_ADDR, unc_event_select_reg.value);

            MSR[core]->read(MSR_UNCORE_PERFEVTSEL4_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_REMOTE_READS_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[core]->write(MSR_UNCORE_PERFEVTSEL4_ADDR, unc_event_select_reg.value);

            MSR[core]->read(MSR_UNCORE_PERFEVTSEL5_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_REMOTE_WRITES_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[core]->write(MSR_UNCORE_PERFEVTSEL5_ADDR, unc_event_select_reg.value);

            MSR[core]->read(MSR_UNCORE_PERFEVTSEL6_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_LOCAL_READS_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[core]->write(MSR_UNCORE_PERFEVTSEL6_ADDR, unc_event_select_reg.value);

            MSR[core]->read(MSR_UNCORE_PERFEVTSEL7_ADDR, &unc_event_select_reg.value);
            unc_event_select_reg.fields.event_select = UNC_QHL_REQUESTS_EVTNR;
            unc_event_select_reg.fields.umask = UNC_QHL_REQUESTS_LOCAL_WRITES_UMASK;
            CPUCNT_INIT_THE_REST_OF_EVTCNT
                MSR[core]->write(MSR_UNCORE_PERFEVTSEL7_ADDR, unc_event_select_reg.value);


#undef CPUCNT_INIT_THE_REST_OF_EVTCNT

            // start uncore counting
            uint64 value = 255 + (1ULL << 32);           // enable all counters
            MSR[core]->write(MSR_UNCORE_PERF_GLOBAL_CTRL_ADDR, value);

            // synchronise counters
            MSR[core]->write(MSR_UNCORE_PMC0, 0);
            MSR[core]->write(MSR_UNCORE_PMC1, 0);
            MSR[core]->write(MSR_UNCORE_PMC2, 0);
            MSR[core]->write(MSR_UNCORE_PMC3, 0);
            MSR[core]->write(MSR_UNCORE_PMC4, 0);
            MSR[core]->write(MSR_UNCORE_PMC5, 0);
            MSR[core]->write(MSR_UNCORE_PMC6, 0);
            MSR[core]->write(MSR_UNCORE_PMC7, 0);
}

void PCM::programBecktonUncore(int32 core)
{
            // program Beckton uncore
            if (core == socketRefCore[0]) computeQPISpeedBeckton(core);

            uint64 value = 1 << 29ULL;           // reset all counters
            MSR[core]->write(U_MSR_PMON_GLOBAL_CTL, value);

            BecktonUncorePMUZDPCTLFVCRegister FVCreg;
            FVCreg.value = 0;
            if (cpu_model == NEHALEM_EX)
            {
                FVCreg.fields.bcmd = 0;             // rd_bcmd
                FVCreg.fields.resp = 0;             // ack_resp
                FVCreg.fields.evnt0 = 5;            // bcmd_match
                FVCreg.fields.evnt1 = 6;            // resp_match
                FVCreg.fields.pbox_init_err = 0;
            }
            else
            {
                FVCreg.fields_wsm.bcmd = 0;             // rd_bcmd
                FVCreg.fields_wsm.resp = 0;             // ack_resp
                FVCreg.fields_wsm.evnt0 = 5;            // bcmd_match
                FVCreg.fields_wsm.evnt1 = 6;            // resp_match
                FVCreg.fields_wsm.pbox_init_err = 0;
            }
            MSR[core]->write(MB0_MSR_PMU_ZDP_CTL_FVC, FVCreg.value);
            MSR[core]->write(MB1_MSR_PMU_ZDP_CTL_FVC, FVCreg.value);

            BecktonUncorePMUCNTCTLRegister CNTCTLreg;
            CNTCTLreg.value = 0;
            CNTCTLreg.fields.en = 1;
            CNTCTLreg.fields.pmi_en = 0;
            CNTCTLreg.fields.count_mode = 0;
            CNTCTLreg.fields.storage_mode = 0;
            CNTCTLreg.fields.wrap_mode = 1;
            CNTCTLreg.fields.flag_mode = 0;
            CNTCTLreg.fields.inc_sel = 0x0d;           // FVC_EV0
            MSR[core]->write(MB0_MSR_PMU_CNT_CTL_0, CNTCTLreg.value);
            MSR[core]->write(MB1_MSR_PMU_CNT_CTL_0, CNTCTLreg.value);
            CNTCTLreg.fields.inc_sel = 0x0e;           // FVC_EV1
            MSR[core]->write(MB0_MSR_PMU_CNT_CTL_1, CNTCTLreg.value);
            MSR[core]->write(MB1_MSR_PMU_CNT_CTL_1, CNTCTLreg.value);

            value = 1 + ((0x0C) << 1ULL);              // enable bit + (event select IMT_INSERTS_WR)
            MSR[core]->write(BB0_MSR_PERF_CNT_CTL_1, value);
            MSR[core]->write(BB1_MSR_PERF_CNT_CTL_1, value);

            MSR[core]->write(MB0_MSR_PERF_GLOBAL_CTL, 3); // enable two counters
            MSR[core]->write(MB1_MSR_PERF_GLOBAL_CTL, 3); // enable two counters

            MSR[core]->write(BB0_MSR_PERF_GLOBAL_CTL, 2); // enable second counter
            MSR[core]->write(BB1_MSR_PERF_GLOBAL_CTL, 2); // enable second counter

            // program R-Box to monitor QPI traffic

            // enable counting on all counters on the left side (port 0-3)
            MSR[core]->write(R_MSR_PMON_GLOBAL_CTL_7_0, 255);
            // ... on the right side (port 4-7)
            MSR[core]->write(R_MSR_PMON_GLOBAL_CTL_15_8, 255);

            // pick the event
            value = (1 << 7ULL) + (1 << 6ULL) + (1 << 2ULL); // count any (incoming) data responses
            MSR[core]->write(R_MSR_PORT0_IPERF_CFG0, value);
            MSR[core]->write(R_MSR_PORT1_IPERF_CFG0, value);
            MSR[core]->write(R_MSR_PORT4_IPERF_CFG0, value);
            MSR[core]->write(R_MSR_PORT5_IPERF_CFG0, value);

            // pick the event
            value = (1ULL << 30ULL); // count null idle flits sent
            MSR[core]->write(R_MSR_PORT0_IPERF_CFG1, value);
            MSR[core]->write(R_MSR_PORT1_IPERF_CFG1, value);
            MSR[core]->write(R_MSR_PORT4_IPERF_CFG1, value);
            MSR[core]->write(R_MSR_PORT5_IPERF_CFG1, value);

            // choose counter 0 to monitor R_MSR_PORT0_IPERF_CFG0
            MSR[core]->write(R_MSR_PMON_CTL0, 1 + 2 * (0));
            // choose counter 1 to monitor R_MSR_PORT1_IPERF_CFG0
            MSR[core]->write(R_MSR_PMON_CTL1, 1 + 2 * (6));
            // choose counter 8 to monitor R_MSR_PORT4_IPERF_CFG0
            MSR[core]->write(R_MSR_PMON_CTL8, 1 + 2 * (0));
            // choose counter 9 to monitor R_MSR_PORT5_IPERF_CFG0
            MSR[core]->write(R_MSR_PMON_CTL9, 1 + 2 * (6));

            // choose counter 2 to monitor R_MSR_PORT0_IPERF_CFG1
            MSR[core]->write(R_MSR_PMON_CTL2, 1 + 2 * (1));
            // choose counter 3 to monitor R_MSR_PORT1_IPERF_CFG1
            MSR[core]->write(R_MSR_PMON_CTL3, 1 + 2 * (7));
            // choose counter 10 to monitor R_MSR_PORT4_IPERF_CFG1
            MSR[core]->write(R_MSR_PMON_CTL10, 1 + 2 * (1));
            // choose counter 11 to monitor R_MSR_PORT5_IPERF_CFG1
            MSR[core]->write(R_MSR_PMON_CTL11, 1 + 2 * (7));

            // enable uncore TSC counter (fixed one)
            MSR[core]->write(W_MSR_PMON_GLOBAL_CTL, 1ULL << 31ULL);
            MSR[core]->write(W_MSR_PMON_FIXED_CTR_CTL, 1ULL);

            value = (1 << 28ULL) + 1;                  // enable all counters
            MSR[core]->write(U_MSR_PMON_GLOBAL_CTL, value);
}

uint64 RDTSC();

void PCM::computeNominalFrequency()
{
   const int ref_core = 0;
   uint64 before = 0, after = 0;
   MSR[ref_core]->read(IA32_TIME_STAMP_COUNTER, &before);
// sleep fo 100 ms
#ifdef _MSC_VER
        Sleep(1000);
#else
        usleep(1000*1000);
#endif
   MSR[ref_core]->read(IA32_TIME_STAMP_COUNTER, &after);
   nominal_frequency = after-before; 
}
std::string PCM::getCPUBrandString()
{
	char buffer[sizeof(int)*4*3+1];
        PCM_CPUID_INFO * info = (PCM_CPUID_INFO *) buffer;
        pcm_cpuid(0x80000002, *info);
        ++info;
        pcm_cpuid(0x80000003, *info);       
        ++info;
        pcm_cpuid(0x80000004, *info); 
        buffer[sizeof(int)*4*3] = 0;
        std::string result(buffer);
        while(result[0]==' ') result.erase(0,1);
        size_t i = std::string::npos;
        while((i=result.find("  "))!=std::string::npos) result.replace(i,2," "); // remove duplicate spaces
        return result;
}

uint64 get_frequency_from_cpuid() // from Pat Fay (Intel)
{
                double speed=0;
                std::string brand = PCM::getCPUBrandString();
                if(brand.length() > 0)
                {
                                size_t unitsg = brand.find("GHz");
                                if(unitsg != std::string::npos)
                                {
                                        size_t atsign = brand.rfind(' ', unitsg);
                                        if(atsign != std::string::npos)
                                        {
                                                std::istringstream(brand.substr(atsign)) >> speed;
												speed *= 1000;
                                        }
                                }
                                else
                                {
	                                size_t unitsg = brand.find("MHz");
		                            if(unitsg != std::string::npos)
			                        {
				                            size_t atsign = brand.rfind(' ', unitsg);
					                        if(atsign != std::string::npos)
						                    {
								                    std::istringstream(brand.substr(atsign)) >> speed;
										    }
									}
                                }
                }
                return (uint64)speed * 1000ULL * 1000ULL;
}

std::string PCM::getSupportedUarchCodenames() const
{
    std::ostringstream ostr;
    for(int32 i=0; i < PCM::END_OF_MODEL_LIST ; ++i)
        if(isCPUModelSupported(i))
            ostr << getUArchCodename(i) << ", ";
    return ostr.str().substr(0, ostr.str().length() - 2);
}

std::string PCM::getUnsupportedMessage() const
{
    std::ostringstream ostr;
    ostr << "Error: unsupported processor. Only Intel(R) processors are supported (Atom(R) and microarchitecture codename "<< getSupportedUarchCodenames() <<").";
    return ostr.str();
}

void PCM::computeQPISpeedBeckton(int core_nr)
{
    uint64 startFlits = 0;
    // reset all counters
    MSR[core_nr]->write(U_MSR_PMON_GLOBAL_CTL, 1 << 29ULL);

    // enable counting on all counters on the left side (port 0-3)
    MSR[core_nr]->write(R_MSR_PMON_GLOBAL_CTL_7_0, 255);
    // disable on the right side (port 4-7)
    MSR[core_nr]->write(R_MSR_PMON_GLOBAL_CTL_15_8, 0);

    // count flits sent
    MSR[core_nr]->write(R_MSR_PORT0_IPERF_CFG0, 1ULL << 31ULL);

    // choose counter 0 to monitor R_MSR_PORT0_IPERF_CFG0
    MSR[core_nr]->write(R_MSR_PMON_CTL0, 1 + 2 * (0));

    // enable all counters
    MSR[core_nr]->write(U_MSR_PMON_GLOBAL_CTL, (1 << 28ULL) + 1);

    MSR[core_nr]->read(R_MSR_PMON_CTR0, &startFlits);

    const uint64 timerGranularity = 1000000ULL; // mks
    uint64 startTSC = getTickCount(timerGranularity, core_nr);
    uint64 endTSC;
    do
    {
        endTSC = getTickCount(timerGranularity, core_nr);
    } while (endTSC - startTSC < 200000ULL); // spin for 200 ms

    uint64 endFlits = 0;
    MSR[core_nr]->read(R_MSR_PMON_CTR0, &endFlits);
    max_qpi_speed = (endFlits - startFlits) * 8ULL * timerGranularity / (endTSC - startTSC);
    
}

bool PCM::PMUinUse()
{
    // follow the "Performance Monitoring Unit Sharing Guide" by P. Irelan and Sh. Kuo
    for (int i = 0; i < num_cores; ++i)
    {
        //std::cout << "Core "<<i<<" exemine registers"<< std::endl;
        uint64 value;
        MSR[i]->read(IA32_CR_PERF_GLOBAL_CTRL, &value);
        // std::cout << "Core "<<i<<" IA32_CR_PERF_GLOBAL_CTRL is "<< std::hex << value << std::dec << std::endl;

        EventSelectRegister event_select_reg;
		event_select_reg.value = 0xFFFFFFFFFFFFFFFF;

        for (uint32 j = 0; j < core_gen_counter_num_max; ++j)
        {
            MSR[i]->read(IA32_PERFEVTSEL0_ADDR + j, &event_select_reg.value);

            if (event_select_reg.fields.event_select != 0 || event_select_reg.fields.apic_int != 0)
            {
                std::cerr << "WARNING: Core "<<i<<" IA32_PERFEVTSEL0_ADDR are not zeroed "<< event_select_reg.value << std::endl;
                return true;
            }
        }

        FixedEventControlRegister ctrl_reg;
		ctrl_reg.value = 0xffffffffffffffff;

        MSR[i]->read(IA32_CR_FIXED_CTR_CTRL, &ctrl_reg.value);

        // Check if someone has installed pmi handler on counter overflow.
        // If so, that agent might potentially need to change counter value 
        // for the "sample after"-mode messing up PCM measurements
        if(ctrl_reg.fields.enable_pmi0 || ctrl_reg.fields.enable_pmi1 || ctrl_reg.fields.enable_pmi2)
        {
            std::cerr << "WARNING: Core "<<i<<" fixed ctrl:"<< ctrl_reg.value << std::endl;
            return true;
        }
        // either os=0,usr=0 (not running) or os=1,usr=1 (fits PCM modus) are ok, other combinations are not
        if(ctrl_reg.fields.os0 != ctrl_reg.fields.usr0 || 
           ctrl_reg.fields.os1 != ctrl_reg.fields.usr1 || 
           ctrl_reg.fields.os2 != ctrl_reg.fields.usr2)
        {
           std::cerr << "WARNING: Core "<<i<<" fixed ctrl:"<< ctrl_reg.value << std::endl;
           return true;
        }
    }
    return false;
}

const char * PCM::getUArchCodename(int32 cpu_model_) const
{
    if(cpu_model_ < 0)
        cpu_model_ = this->cpu_model ;

    switch(cpu_model_)
    {
        case NEHALEM_EP:
        case NEHALEM:
            return "Nehalem/Nehalem-EP";
        case ATOM:
            return "Atom(tm)";
        case CLARKDALE:
            return "Westmere/Clarkdale";
        case WESTMERE_EP:
            return "Westmere-EP";
        case NEHALEM_EX:
            return "Nehalem-EX";
        case WESTMERE_EX:
            return "Westmere-EX";
        case SANDY_BRIDGE:
            return "Sandy Bridge";
        case JAKETOWN:
            return "Sandy Bridge-EP/Jaketown";
        case IVYTOWN:
            return "Ivy Bridge-EP/EN/EX/Ivytown";
        case HASWELLX:
            return "Haswell-EP/EN/EX";
        case IVY_BRIDGE:
            return "Ivy Bridge";
        case HASWELL:
            return "Haswell";
    }
    return "unknown";
}

void PCM::cleanupPMU()
{
#ifdef PCM_USE_PERF  
    if(canUsePerf)
    {
      for (int i = 0; i < num_cores; ++i)
        for(int c = 0; c < PERF_MAX_COUNTERS; ++c)
            ::close(perfEventHandle[i][c]);
      
      return;
    }
#endif    

    // follow the "Performance Monitoring Unit Sharing Guide" by P. Irelan and Sh. Kuo
    for (int i = 0; i < num_cores; ++i)
    {
        // disable generic counters and continue free running counting for fixed counters
        MSR[i]->write(IA32_CR_PERF_GLOBAL_CTRL, (1ULL << 32) + (1ULL << 33) + (1ULL << 34));

        for (uint32 j = 0; j < core_gen_counter_num_max; ++j)
        {
            MSR[i]->write(IA32_PERFEVTSEL0_ADDR + j, 0);
        }
    }

    if(cpu_model == JAKETOWN)
        enableJKTWorkaround(false);

    std::cerr << " Zeroed PMU registers" << std::endl;
}

void PCM::resetPMU()
{
    for (int i = 0; i < num_cores; ++i)
    {
        // disable all counters
        MSR[i]->write(IA32_CR_PERF_GLOBAL_CTRL, 0);

        for (uint32 j = 0; j < core_gen_counter_num_max; ++j)
        {
            MSR[i]->write(IA32_PERFEVTSEL0_ADDR + j, 0);
        }


        FixedEventControlRegister ctrl_reg;
		ctrl_reg.value = 0xffffffffffffffff;

        MSR[i]->read(IA32_CR_FIXED_CTR_CTRL, &ctrl_reg.value);
        if ((ctrl_reg.fields.os0 ||
             ctrl_reg.fields.usr0 ||
             ctrl_reg.fields.enable_pmi0 ||
             ctrl_reg.fields.os1 ||
             ctrl_reg.fields.usr1 ||
             ctrl_reg.fields.enable_pmi1 ||
             ctrl_reg.fields.os2 ||
             ctrl_reg.fields.usr2 ||
             ctrl_reg.fields.enable_pmi2)
            != 0)
            MSR[i]->write(IA32_CR_FIXED_CTR_CTRL, 0);
    }

    std::cerr << " Zeroed PMU registers" << std::endl;
}

void PCM::setOutput(const std::string filename)
{
     outfile = new std::ofstream(filename.c_str());
     backup_ofile = std::cout.rdbuf();
     std::cout.rdbuf(outfile->rdbuf());
}

void PCM::restoreOutput()
{
    // restore cout back to what it was originally
    if(backup_ofile)
        std::cout.rdbuf(backup_ofile);

// close output file
    if(outfile)
        outfile->close();
}

void PCM::cleanup()
{
    InstanceLock lock(allow_multiple_instances);
    
    if (!MSR) return;
    
    std::cerr << "Cleaning up" << std::endl;

    if (decrementInstanceSemaphore())
        cleanupPMU();
}

#ifdef __APPLE__

uint32 PCM::getNumInstances()
{
    return MSR[0]->getNumInstances();
}


uint32 PCM::incrementNumInstances()
{
    return MSR[0]->incrementNumInstances();
}

uint32 PCM::decrementNumInstances()
{
    return MSR[0]->decrementNumInstances();;
}

int convertUnknownToInt(size_t size, char* value){
    if(sizeof(int) == size)	
    {
        return *(int*)value;
    }
    else if(sizeof(long) == size)
    {
        return *(long *)value;
    }
    else if(sizeof(long long) == size)
    {
        return *(long long *)value;
    }
    else 
    {
        // In this case, we don't know what it is so we guess int
        return *(int *)value;
    }
}

#endif

bool PCM::decrementInstanceSemaphore()
{
    if(allow_multiple_instances == false)
    {
        return programmed_pmu;
    }
    bool isLastInstance = false;
    // when decrement was called before program() the numInstancesSemaphore
    // may not be initialized, causing SIGSEGV. This fixes it.
    if(numInstancesSemaphore == NULL)
        return true;

                #ifdef _MSC_VER
    WaitForSingleObject(numInstancesSemaphore, 0);

    DWORD res = WaitForSingleObject(numInstancesSemaphore, 0);
    if (res == WAIT_TIMEOUT)
    {
        // I have the last instance of monitor

        isLastInstance = true;

        CloseHandle(numInstancesSemaphore);
    }
    else if (res == WAIT_OBJECT_0)
    {
        ReleaseSemaphore(numInstancesSemaphore, 1, NULL);

        // std::cerr << "Someone else is running monitor instance, no cleanup needed"<< std::endl;
    }
    else
    {
        // unknown error
        std::cerr << "ERROR: Bad semaphore. Performed cleanup twice?" << std::endl;
    }
        

        #elif __APPLE__
    sem_wait(numInstancesSemaphore);
    uint32 oldValue = PCM::getNumInstances();
    sem_post(numInstancesSemaphore);
    if(oldValue == 0)
    {
	// see same case for linux
	return false;
    }
    sem_wait(numInstancesSemaphore);
    uint32 currValue = PCM::decrementNumInstances();
    sem_post(numInstancesSemaphore);
    if(currValue == 0){
	isLastInstance = true;
    }

	#else // if linux
    int oldValue = -1;
    sem_getvalue(numInstancesSemaphore, &oldValue);
    if(oldValue == 0)
    {
       // the current value is already zero, somewhere the semaphore has been already decremented (and thus the clean up has been done if needed)
       // that means logically we are do not own the last instance anymore, thus returning false
       return false;
    }
    sem_wait(numInstancesSemaphore);
    int curValue = -1;
    sem_getvalue(numInstancesSemaphore, &curValue);
    if (curValue == 0)
    {
        // I have the last instance of monitor

        isLastInstance = true;

        // std::cerr << "I am the last one"<< std::endl;
    }
        #endif // end ifdef _MSC_VER

    return isLastInstance;
}

uint64 PCM::getTickCount(uint64 multiplier, uint32 core)
{
    return (multiplier * getInvariantTSC(CoreCounterState(), getCoreCounterState(core))) / getNominalFrequency();
}

uint64 RDTSC()
{
        uint64 result = 0;
#ifdef _MSC_VER
        // Windows
        #if _MSC_VER>= 1600
        result = __rdtsc();
        #endif
#else
        // Linux
        uint32 high = 0, low = 0;
        asm volatile("rdtsc" : "=a" (low), "=d" (high));
        result = low + (uint64(high)<<32ULL);
#endif
        return result;

}

uint64 RDTSCP()
{
	uint64 result = 0;
#ifdef _MSC_VER
        // Windows
        #if _MSC_VER>= 1600
        unsigned int Aux;
        result = __rdtscp(&Aux);
        #endif
#else
	// Linux and OS X
        uint32 high = 0, low = 0;
        asm volatile (
           "rdtscp\n\t"
           "mov %%edx, %0\n\t"
           "mov %%eax, %1\n\t":
           "=r" (high), "=r" (low) :: "%rax", "%rcx", "%rdx");
        result = low + (uint64(high)<<32ULL);
#endif
	return result;
}


uint64 PCM::getTickCountRDTSCP(uint64 multiplier)
{
	return (multiplier*RDTSCP())/getNominalFrequency();
}

SystemCounterState getSystemCounterState()
{
    PCM * inst = PCM::getInstance();
    SystemCounterState result;
    if (inst) result = inst->getSystemCounterState();
    return result;
}

SocketCounterState getSocketCounterState(uint32 socket)
{
    PCM * inst = PCM::getInstance();
    SocketCounterState result;
    if (inst) result = inst->getSocketCounterState(socket);
    return result;
}

CoreCounterState getCoreCounterState(uint32 core)
{
    PCM * inst = PCM::getInstance();
    CoreCounterState result;
    if (inst) result = inst->getCoreCounterState(core);
    return result;
}

#ifdef PCM_USE_PERF
void PCM::readPerfData(uint32 core, std::vector<uint64> & outData)
{
    uint64 data[1 + PERF_MAX_COUNTERS];
    const int32 bytes2read =  sizeof(uint64)*(1 + core_fixed_counter_num_used + core_gen_counter_num_used);
    int result = ::read(perfEventHandle[core][PERF_GROUP_LEADER_COUNTER], data, bytes2read );
    // data layout: nr counters; counter 0, counter 1, counter 2,...    
    if(result != bytes2read)
    {
       std::cerr << "Error while reading perf data. Result is "<< result << std::endl;
       std::cerr << "Check if you run other competing Linux perf clients." << std::endl;
    } else if(data[0] != core_fixed_counter_num_used + core_gen_counter_num_used)
    {
       std::cerr << "Number of counters read from perf is wrong. Elements read: "<< data[0] << std::endl;
    }
    else
    {  // copy all counters, they start from position 1 in data
       std::copy((data + 1), (data + 1) + data[0], outData.begin());
    }
}
#endif

void BasicCounterState::readAndAggregate(SafeMsrHandle * msr)
{
    uint64 cInstRetiredAny = 0, cCpuClkUnhaltedThread = 0, cCpuClkUnhaltedRef = 0;
    uint64 cL3Miss = 0;
    uint64 cL3UnsharedHit = 0;
    uint64 cL2HitM = 0;
    uint64 cL2Hit = 0;
    uint64 cInvariantTSC = 0;
    uint64 cCStateResidency[PCM::MAX_C_STATE + 1];
    memset(&(cCStateResidency[0]), 0, sizeof(cCStateResidency));
    uint64 thermStatus = 0;
    TemporalThreadAffinity tempThreadAffinity(msr->getCoreId()); // speedup trick for Linux

    PCM * m = PCM::getInstance();
    uint32 cpu_model = m->getCPUModel();

  // reading core PMU counters
#ifdef PCM_USE_PERF
  if(m->canUsePerf)
  {
    std::vector<uint64> perfData(PERF_MAX_COUNTERS, 0ULL);
    m->readPerfData(msr->getCoreId(), perfData);
    cInstRetiredAny =       perfData[PCM::PERF_INST_RETIRED_ANY_POS];
    cCpuClkUnhaltedThread = perfData[PCM::PERF_CPU_CLK_UNHALTED_THREAD_POS];
    cCpuClkUnhaltedRef =    perfData[PCM::PERF_CPU_CLK_UNHALTED_REF_POS];
    cL3Miss =               perfData[PCM::PERF_GEN_EVENT_0_POS];
    cL3UnsharedHit =        perfData[PCM::PERF_GEN_EVENT_1_POS];
    cL2HitM =               perfData[PCM::PERF_GEN_EVENT_2_POS];
    cL2Hit =                perfData[PCM::PERF_GEN_EVENT_3_POS];
  }
  else
#endif
  {
    msr->read(INST_RETIRED_ANY_ADDR, &cInstRetiredAny);
    msr->read(CPU_CLK_UNHALTED_THREAD_ADDR, &cCpuClkUnhaltedThread);
    msr->read(CPU_CLK_UNHALTED_REF_ADDR, &cCpuClkUnhaltedRef);
    switch (cpu_model)
    {
    case PCM::WESTMERE_EP:
    case PCM::NEHALEM_EP:
    case PCM::NEHALEM_EX:
    case PCM::WESTMERE_EX:
    case PCM::CLARKDALE:
    case PCM::SANDY_BRIDGE:
    case PCM::JAKETOWN:
    case PCM::IVYTOWN:
    case PCM::HASWELLX:
	case PCM::IVY_BRIDGE:
    case PCM::HASWELL:
        msr->read(IA32_PMC0, &cL3Miss);
        msr->read(IA32_PMC1, &cL3UnsharedHit);
        msr->read(IA32_PMC2, &cL2HitM);
        msr->read(IA32_PMC3, &cL2Hit);
        break;
    case PCM::ATOM:
        msr->read(IA32_PMC0, &cL3Miss);         // for Atom mapped to ArchLLCMiss field
        msr->read(IA32_PMC1, &cL3UnsharedHit);  // for Atom mapped to ArchLLCRef field
        break;
    }
  }

    // std::cout << "DEBUG1: "<< msr->getCoreId() << " " << cInstRetiredAny<< " "<< std::endl;

    if(cpu_model != PCM::ATOM ||  m->getOriginalCPUModel() == PCM::ATOM_AVOTON) msr->read(IA32_TIME_STAMP_COUNTER, &cInvariantTSC);
    else
    {
#ifdef _MSC_VER
        cInvariantTSC = ((uint64)(GetTickCount()/1000))*m->getNominalFrequency();
#else
        struct timeval tp;
        gettimeofday(&tp, NULL);
        cInvariantTSC = (double(tp.tv_sec) + tp.tv_usec / 1000000.)*m->getNominalFrequency();
#endif
    }

    // reading core C state counters
    for(int i=0; i <= PCM::MAX_C_STATE ;++i)
        if(m->coreCStateMsr && m->coreCStateMsr[i])
                msr->read(m->coreCStateMsr[i], &(cCStateResidency[i]));

    // reading temperature
    msr->read(MSR_IA32_THERM_STATUS, &thermStatus);

    InstRetiredAny += m->extractCoreFixedCounterValue(cInstRetiredAny);
    CpuClkUnhaltedThread += m->extractCoreFixedCounterValue(cCpuClkUnhaltedThread);
    CpuClkUnhaltedRef += m->extractCoreFixedCounterValue(cCpuClkUnhaltedRef);
    L3Miss += m->extractCoreGenCounterValue(cL3Miss);
    L3UnsharedHit += m->extractCoreGenCounterValue(cL3UnsharedHit);
    L2HitM += m->extractCoreGenCounterValue(cL2HitM);
    L2Hit += m->extractCoreGenCounterValue(cL2Hit);
    InvariantTSC += cInvariantTSC;
    for(int i=0; i <= PCM::MAX_C_STATE ;++i)
        CStateResidency[i] += cCStateResidency[i];
    ThermalHeadroom = extractThermalHeadroom(thermStatus);
}

PCM::ErrorCode PCM::programServerUncorePowerMetrics(int mc_profile, int pcu_profile, int * freq_bands)
{
    if(MSR == NULL || server_pcicfg_uncore == NULL)  return PCM::MSRAccessDenied;

	uint32 PCUCntConf[4] = {0,0,0,0};

    PCUCntConf[0] = PCU_MSR_PMON_CTL_EVENT(0); // clock ticks

    switch(pcu_profile)
    {
	case 0:
         PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0xB); // FREQ_BAND0_CYCLES
         PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0xC); // FREQ_BAND1_CYCLES
         PCUCntConf[3] =  PCU_MSR_PMON_CTL_EVENT(0xD); // FREQ_BAND2_CYCLES
         break;
    case 1:
         PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0x80) + PCU_MSR_PMON_CTL_OCC_SEL(1); // POWER_STATE_OCCUPANCY.C0 using CLOCKTICKS + 8th-bit
         PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0x80) + PCU_MSR_PMON_CTL_OCC_SEL(2); // POWER_STATE_OCCUPANCY.C3 using CLOCKTICKS + 8th-bit
         PCUCntConf[3] =  PCU_MSR_PMON_CTL_EVENT(0x80) + PCU_MSR_PMON_CTL_OCC_SEL(3); // POWER_STATE_OCCUPANCY.C6 using CLOCKTICKS + 8th-bit
         break;
	case 2:
         PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0x09); // PROCHOT_INTERNAL_CYCLES
         PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0x0A); // PROCHOT_EXTERNAL_CYCLES
         PCUCntConf[3] =  PCU_MSR_PMON_CTL_EVENT(0x04); // Thermal frequency limit cycles: FREQ_MAX_LIMIT_THERMAL_CYCLES
         break;
	case 3:
         PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0x04); // Thermal frequency limit cycles: FREQ_MAX_LIMIT_THERMAL_CYCLES
         PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0x05); // Power frequency limit cycles: FREQ_MAX_POWER_CYCLES
         PCUCntConf[3] =  PCU_MSR_PMON_CTL_EVENT(0x07); // Clipped frequency limit cycles: FREQ_MAX_CURRENT_CYCLES
         break;
	case 4:
         PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0x06); // OS frequency limit cycles: FREQ_MAX_OS_CYCLES
         PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0x05); // Power frequency limit cycles: FREQ_MAX_POWER_CYCLES
         PCUCntConf[3] =  PCU_MSR_PMON_CTL_EVENT(0x07); // Clipped frequency limit cycles: FREQ_MAX_CURRENT_CYCLES
         break;
	case 5:
         if(JAKETOWN == cpu_model)
         {
             PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0) + PCU_MSR_PMON_CTL_EXTRA_SEL + PCU_MSR_PMON_CTL_EDGE_DET ; // number of frequency transitions
             PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0) + PCU_MSR_PMON_CTL_EXTRA_SEL ; // cycles spent changing frequency
         } else if (IVYTOWN == cpu_model )
         {
             PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0x60) + PCU_MSR_PMON_CTL_EDGE_DET ; // number of frequency transitions
             PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0x60) ; // cycles spent changing frequency: FREQ_TRANS_CYCLES
         } else if (HASWELLX == cpu_model )
         {
             PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0x74) + PCU_MSR_PMON_CTL_EDGE_DET ; // number of frequency transitions
             PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0x74) ; // cycles spent changing frequency: FREQ_TRANS_CYCLES
             PCUCntConf[3] =  PCU_MSR_PMON_CTL_EVENT(0x79) + PCU_MSR_PMON_CTL_EDGE_DET ; // number of UFS transitions
             PCUCntConf[0] =  PCU_MSR_PMON_CTL_EVENT(0x79)                             ; // UFS transition cyclesss
         } else
         {
             std::cerr << "ERROR: no frequency transition events defined for CPU model "<< cpu_model << std::endl;
         }
         break;
    case 6:
         if (IVYTOWN == cpu_model )
         {
             PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0x2B) + PCU_MSR_PMON_CTL_EDGE_DET ; // PC2 transitions
             PCUCntConf[3] =  PCU_MSR_PMON_CTL_EVENT(0x2D) + PCU_MSR_PMON_CTL_EDGE_DET ; // PC6 transitions
         } else if (HASWELLX == cpu_model )
         {
             PCUCntConf[0] =  PCU_MSR_PMON_CTL_EVENT(0x4E)                             ; // PC1e residencys
             PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0x4E) + PCU_MSR_PMON_CTL_EDGE_DET ; // PC1 transitions
             PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0x2B) + PCU_MSR_PMON_CTL_EDGE_DET ; // PC2 transitions
             PCUCntConf[3] =  PCU_MSR_PMON_CTL_EVENT(0x2D) + PCU_MSR_PMON_CTL_EDGE_DET ; // PC6 transitions
         } else
         {
             std::cerr << "ERROR: no package C-state transition events defined for CPU model "<< cpu_model << std::endl;
         }
         break;
     case 7:
         if (HASWELLX == cpu_model )
         {
             PCUCntConf[0] =  PCU_MSR_PMON_CTL_EVENT(0x7E) ; // UFS_TRANSITIONS_PERF_P_LIMIT
             PCUCntConf[1] =  PCU_MSR_PMON_CTL_EVENT(0x7D) ; // UFS_TRANSITIONS_IO_P_LIMIT
             PCUCntConf[2] =  PCU_MSR_PMON_CTL_EVENT(0x7A) ; // UFS_TRANSITIONS_UP_RING_TRAFFIC
             PCUCntConf[3] =  PCU_MSR_PMON_CTL_EVENT(0x7B) ; // UFS_TRANSITIONS_UP_STALL_CYCLES
         } else
         {
             std::cerr << "ERROR: no UFS transition events defined for CPU model "<< cpu_model << std::endl;
         }
         break;
    case 8:
         if (HASWELLX == cpu_model )
         {
             PCUCntConf[0] =  PCU_MSR_PMON_CTL_EVENT(0x7C) ; // UFS_TRANSITIONS_DOWN
         } else
         {
             std::cerr << "ERROR: no UFS transition events defined for CPU model "<< cpu_model << std::endl;
         }
         break;
    default:
         std::cerr << "ERROR: unsupported PCU profile "<< pcu_profile << std::endl;
    }

	uint32 PCU_MSR_PMON_BOX_FILTER_ADDR, PCU_MSR_PMON_CTLX_ADDR[4] = { 0, 0, 0, 0 };

    switch(cpu_model)
    {
        case IVYTOWN:
        case JAKETOWN:
            PCU_MSR_PMON_BOX_FILTER_ADDR = JKTIVT_PCU_MSR_PMON_BOX_FILTER_ADDR;
            PCU_MSR_PMON_CTLX_ADDR[0] = JKTIVT_PCU_MSR_PMON_CTL0_ADDR;
            PCU_MSR_PMON_CTLX_ADDR[1] = JKTIVT_PCU_MSR_PMON_CTL1_ADDR;
            PCU_MSR_PMON_CTLX_ADDR[2] = JKTIVT_PCU_MSR_PMON_CTL2_ADDR;
            PCU_MSR_PMON_CTLX_ADDR[3] = JKTIVT_PCU_MSR_PMON_CTL3_ADDR;
            break;
        case HASWELLX:
            PCU_MSR_PMON_BOX_FILTER_ADDR = HSX_PCU_MSR_PMON_BOX_FILTER_ADDR;
            PCU_MSR_PMON_CTLX_ADDR[0] = HSX_PCU_MSR_PMON_CTL0_ADDR;
            PCU_MSR_PMON_CTLX_ADDR[1] = HSX_PCU_MSR_PMON_CTL1_ADDR;
            PCU_MSR_PMON_CTLX_ADDR[2] = HSX_PCU_MSR_PMON_CTL2_ADDR;
            PCU_MSR_PMON_CTLX_ADDR[3] = HSX_PCU_MSR_PMON_CTL3_ADDR;
            break;
        default:
            std::cerr << "Error: unsupported uncore. cpu model is "<< cpu_model << std::endl;
            return PCM::UnknownError;
    }

    for (int i = 0; (i < num_sockets) && server_pcicfg_uncore && MSR; ++i)
    {
      server_pcicfg_uncore[i]->program_power_metrics(mc_profile);

      uint32 refCore = socketRefCore[i];
      TemporalThreadAffinity tempThreadAffinity(refCore); // speedup trick for Linux

       // freeze enable
      MSR[refCore]->write(PCU_MSR_PMON_BOX_CTL_ADDR, PCU_MSR_PMON_BOX_CTL_FRZ_EN);
        // freeze
      MSR[refCore]->write(PCU_MSR_PMON_BOX_CTL_ADDR, PCU_MSR_PMON_BOX_CTL_FRZ_EN + PCU_MSR_PMON_BOX_CTL_FRZ);

#ifdef PCM_UNCORE_PMON_BOX_CHECK_STATUS
      uint64 val = 0;
      MSR[refCore]->read(PCU_MSR_PMON_BOX_CTL_ADDR,&val);
      if ((val & UNCORE_PMON_BOX_CTL_VALID_BITS_MASK) != (PCU_MSR_PMON_BOX_CTL_FRZ_EN + PCU_MSR_PMON_BOX_CTL_FRZ))
            std::cerr << "ERROR: PCU counter programming seems not to work. PCU_MSR_PMON_BOX_CTL=0x" << std::hex << val << " needs to be =0x"<< (PCU_MSR_PMON_BOX_CTL_FRZ_EN + PCU_MSR_PMON_BOX_CTL_FRZ) << std::endl;
#endif

	  if(freq_bands == NULL)
		MSR[refCore]->write(PCU_MSR_PMON_BOX_FILTER_ADDR,
                PCU_MSR_PMON_BOX_FILTER_BAND_0(10) + // 1000 MHz
                PCU_MSR_PMON_BOX_FILTER_BAND_1(20) + // 2000 MHz
                PCU_MSR_PMON_BOX_FILTER_BAND_2(30)   // 3000 MHz
			);
	  else
		MSR[refCore]->write(PCU_MSR_PMON_BOX_FILTER_ADDR,
                PCU_MSR_PMON_BOX_FILTER_BAND_0(freq_bands[0]) + 
                PCU_MSR_PMON_BOX_FILTER_BAND_1(freq_bands[1]) + 
                PCU_MSR_PMON_BOX_FILTER_BAND_2(freq_bands[2])   
			);

      for(int ctr = 0; ctr < 4; ++ctr)
      {
        MSR[refCore]->write(PCU_MSR_PMON_CTLX_ADDR[ctr], PCU_MSR_PMON_CTL_EN);
        MSR[refCore]->write(PCU_MSR_PMON_CTLX_ADDR[ctr], PCU_MSR_PMON_CTL_EN + PCUCntConf[ctr]);
      }

      // reset counter values
      MSR[refCore]->write(PCU_MSR_PMON_BOX_CTL_ADDR, PCU_MSR_PMON_BOX_CTL_FRZ_EN + PCU_MSR_PMON_BOX_CTL_FRZ + PCU_MSR_PMON_BOX_CTL_RST_COUNTERS);

      // unfreeze counters
      MSR[refCore]->write(PCU_MSR_PMON_BOX_CTL_ADDR, PCU_MSR_PMON_BOX_CTL_FRZ_EN);
 
    }
    return PCM::Success;
}

void PCM::freezeServerUncoreCounters()
{
	for (int i = 0; (i < num_sockets) && server_pcicfg_uncore && MSR; ++i)
    {
      server_pcicfg_uncore[i]->freezeCounters();
	  MSR[socketRefCore[i]]->write(PCU_MSR_PMON_BOX_CTL_ADDR, PCU_MSR_PMON_BOX_CTL_FRZ_EN + PCU_MSR_PMON_BOX_CTL_FRZ);
	}
}
void PCM::unfreezeServerUncoreCounters()
{
	for (int i = 0; (i < num_sockets) && server_pcicfg_uncore && MSR; ++i)
    {
      server_pcicfg_uncore[i]->unfreezeCounters();
	  MSR[socketRefCore[i]]->write(PCU_MSR_PMON_BOX_CTL_ADDR, PCU_MSR_PMON_BOX_CTL_FRZ_EN);
	}
}
void UncoreCounterState::readAndAggregate(SafeMsrHandle * msr)
{
    TemporalThreadAffinity tempThreadAffinity(msr->getCoreId()); // speedup trick for Linux

    PCM::getInstance()->readAndAggregatePackageCStateResidencies(msr, *this);
}


SystemCounterState PCM::getSystemCounterState()
{
    SystemCounterState result;
    if (MSR)
    {
        // read core and uncore counter state
        for (int32 core = 0; core < num_cores; ++core)
            result.readAndAggregate(MSR[core]);

        for (int32 s=0; s < num_sockets; s++)
        {
            readAndAggregateUncoreMCCounters(s, result);
            readAndAggregateEnergyCounters(s, result);
        }

        readQPICounters(result);

        result.ThermalHeadroom = PCM_INVALID_THERMAL_HEADROOM; // not available for system
    }
    return result;
}

template <class CounterStateType>
void PCM::readAndAggregateUncoreMCCounters(const uint32 socket, CounterStateType & result)
{
    if (hasPCICFGUncore())
    {
        if (server_pcicfg_uncore)
        {
            server_pcicfg_uncore[socket]->freezeCounters();
            result.UncMCNormalReads += server_pcicfg_uncore[socket]->getImcReads();
            result.UncMCFullWrites += server_pcicfg_uncore[socket]->getImcWrites();
            server_pcicfg_uncore[socket]->unfreezeCounters();
        }
    }
    else if(clientBW && socket == 0)
    {
        result.UncMCNormalReads += clientImcReads->read();
        result.UncMCFullWrites += clientImcWrites->read();
        result.UncMCIORequests += clientIoRequests->read();
    }
    else
    {
        SafeMsrHandle * msr = MSR[socketRefCore[socket]];
        TemporalThreadAffinity tempThreadAffinity(socketRefCore[socket]); // speedup trick for Linux
        switch (cpu_model)
        {
            case PCM::WESTMERE_EP:
            case PCM::NEHALEM_EP:
            {
                uint64 cUncMCFullWrites = 0;
                uint64 cUncMCNormalReads = 0;
                msr->read(MSR_UNCORE_PMC0, &cUncMCFullWrites);
                msr->read(MSR_UNCORE_PMC1, &cUncMCNormalReads);
                result.UncMCFullWrites += extractUncoreGenCounterValue(cUncMCFullWrites);
                result.UncMCNormalReads += extractUncoreGenCounterValue(cUncMCNormalReads);
            }
            break;
            case PCM::NEHALEM_EX:
            case PCM::WESTMERE_EX:
            {
                uint64 cUncMCNormalReads = 0;
                msr->read(MB0_MSR_PMU_CNT_0, &cUncMCNormalReads);
                result.UncMCNormalReads += extractUncoreGenCounterValue(cUncMCNormalReads);
                msr->read(MB1_MSR_PMU_CNT_0, &cUncMCNormalReads);
                result.UncMCNormalReads += extractUncoreGenCounterValue(cUncMCNormalReads);

                uint64 cUncMCFullWrites = 0;                         // really good approximation of
                msr->read(BB0_MSR_PERF_CNT_1, &cUncMCFullWrites);
                result.UncMCFullWrites += extractUncoreGenCounterValue(cUncMCFullWrites);
                msr->read(BB1_MSR_PERF_CNT_1, &cUncMCFullWrites);
                result.UncMCFullWrites += extractUncoreGenCounterValue(cUncMCFullWrites);
            }
            break;

            default:;
        }
    }
}

template <class CounterStateType>
void PCM::readAndAggregateEnergyCounters(const uint32 socket, CounterStateType & result)
{
    if(socket < snb_energy_status.size())
        result.PackageEnergyStatus += snb_energy_status[socket]->read();

    if(socket < jkt_dram_energy_status.size())
        result.DRAMEnergyStatus += jkt_dram_energy_status[socket]->read();
}


template <class CounterStateType>
void PCM::readAndAggregatePackageCStateResidencies(SafeMsrHandle * msr, CounterStateType & result)
{
    // reading package C state counters
    uint64 cCStateResidency[PCM::MAX_C_STATE + 1];
    memset(&(cCStateResidency[0]), 0, sizeof(cCStateResidency));

    for(int i=0; i <= PCM::MAX_C_STATE ;++i)
        if(pkgCStateMsr && pkgCStateMsr[i])
                msr->read(pkgCStateMsr[i], &(cCStateResidency[i]));

    for(int i=0; i <= PCM::MAX_C_STATE ;++i)
      result.CStateResidency[i] += cCStateResidency[i];
}

void PCM::readQPICounters(SystemCounterState & result)
{
        // read QPI counters
        std::vector<bool> SocketProcessed(num_sockets, false);
        if (cpu_model == PCM::NEHALEM_EX || cpu_model == PCM::WESTMERE_EX)
        {
            for (int32 core = 0; core < num_cores; ++core)
            {
                if(isCoreOnline(core) == false) continue;

                if(core == socketRefCore[0]) MSR[core]->read(W_MSR_PMON_FIXED_CTR, &(result.uncoreTSC));

                uint32 s = topology[core].socket;

                if (!SocketProcessed[s])
                {
                    TemporalThreadAffinity tempThreadAffinity(core); // speedup trick for Linux

                    // incoming data responses from QPI link 0
                    MSR[core]->read(R_MSR_PMON_CTR1, &(result.incomingQPIPackets[s][0]));
                    // incoming data responses from QPI link 1 (yes, from CTR0)
                    MSR[core]->read(R_MSR_PMON_CTR0, &(result.incomingQPIPackets[s][1]));
                    // incoming data responses from QPI link 2
                    MSR[core]->read(R_MSR_PMON_CTR8, &(result.incomingQPIPackets[s][2]));
                    // incoming data responses from QPI link 3
                    MSR[core]->read(R_MSR_PMON_CTR9, &(result.incomingQPIPackets[s][3]));

                    // outgoing idle flits from QPI link 0
                    MSR[core]->read(R_MSR_PMON_CTR3, &(result.outgoingQPIIdleFlits[s][0]));
                    // outgoing idle flits from QPI link 1 (yes, from CTR0)
                    MSR[core]->read(R_MSR_PMON_CTR2, &(result.outgoingQPIIdleFlits[s][1]));
                    // outgoing idle flits from QPI link 2
                    MSR[core]->read(R_MSR_PMON_CTR10, &(result.outgoingQPIIdleFlits[s][2]));
                    // outgoing idle flits from QPI link 3
                    MSR[core]->read(R_MSR_PMON_CTR11, &(result.outgoingQPIIdleFlits[s][3]));

                    SocketProcessed[s] = true;
                }
            }
        }
        else if ((cpu_model == PCM::NEHALEM_EP || cpu_model == PCM::WESTMERE_EP))
        {
            if (num_sockets == 2)
            {
                uint32 SCore[2] = { 0, 0 };
                uint64 Total_Reads[2] = { 0, 0 };
                uint64 Total_Writes[2] = { 0, 0 };
                uint64 IOH_Reads[2] = { 0, 0 };
                uint64 IOH_Writes[2] = { 0, 0 };
                uint64 Remote_Reads[2] = { 0, 0 };
                uint64 Remote_Writes[2] = { 0, 0 };
                uint64 Local_Reads[2] = { 0, 0 };
                uint64 Local_Writes[2] = { 0, 0 };

                while (topology[SCore[0]].socket != 0) ++(SCore[0]);
                while (topology[SCore[1]].socket != 1) ++(SCore[1]);
                for (int s = 0; s < 2; ++s)
                {
                    TemporalThreadAffinity tempThreadAffinity(SCore[s]); // speedup trick for Linux

                    MSR[SCore[s]]->read(MSR_UNCORE_PMC0, &Total_Writes[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC1, &Total_Reads[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC2, &IOH_Reads[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC3, &IOH_Writes[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC4, &Remote_Reads[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC5, &Remote_Writes[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC6, &Local_Reads[s]);
                    MSR[SCore[s]]->read(MSR_UNCORE_PMC7, &Local_Writes[s]);
                }

#if 1
                // compute Remote_Reads differently
                for (int s = 0; s < 2; ++s)
                {
                    uint64 total = Total_Writes[s] + Total_Reads[s];
                    uint64 rem = IOH_Reads[s]
                                 + IOH_Writes[s]
                                 + Local_Reads[s]
                                 + Local_Writes[s]
                                 + Remote_Writes[s];
                    Remote_Reads[s] = (total > rem) ? (total - rem) : 0;
                }
#endif


                // only an estimation (lower bound) - does not count NT stores correctly
                result.incomingQPIPackets[0][0] = Remote_Reads[1] + Remote_Writes[0];
                result.incomingQPIPackets[0][1] = IOH_Reads[0];
                result.incomingQPIPackets[1][0] = Remote_Reads[0] + Remote_Writes[1];
                result.incomingQPIPackets[1][1] = IOH_Reads[1];
            }
            else
            {
                // for a single socket systems no information is available
                result.incomingQPIPackets[0][0] = 0;
            }
        }
        else if (hasPCICFGUncore())
        {
            if (server_pcicfg_uncore)
                for (int32 s = 0; (s < num_sockets); ++s)
                {
                    server_pcicfg_uncore[s]->freezeCounters();
                    for (uint32 port = 0; port < getQPILinksPerSocket(); ++port)
		            {
                        result.incomingQPIPackets[s][port] = server_pcicfg_uncore[s]->getIncomingDataFlits(port) / 8;
			            result.outgoingQPIDataNonDataFlits[s][port] = server_pcicfg_uncore[s]->getOutgoingDataNonDataFlits(port);
		            }
                    server_pcicfg_uncore[s]->unfreezeCounters();
                }
        }
        // end of reading QPI counters
}

template <class CounterStateType>
void PCM::readPackageThermalHeadroom(const uint32 socket, CounterStateType & result)
{
    if(packageThermalMetricsAvailable())
    {
        uint64 val = 0;
        MSR[socketRefCore[socket]]->read(MSR_PACKAGE_THERM_STATUS,&val);
        result.ThermalHeadroom = extractThermalHeadroom(val);
    }
    else
        result.ThermalHeadroom = PCM_INVALID_THERMAL_HEADROOM; // not available
}

SocketCounterState PCM::getSocketCounterState(uint32 socket)
{
    SocketCounterState result;
    if (MSR)
    {
        // reading core and uncore counter states
        for (int32 core = 0; core < num_cores; ++core)
            if (isCoreOnline(core) && (topology[core].socket == int32(socket)))
                result.readAndAggregate(MSR[core]);

        readAndAggregateUncoreMCCounters(socket, result);

        readAndAggregateEnergyCounters(socket, result);

        readPackageThermalHeadroom(socket, result);

    }
    return result;
}

void PCM::getAllCounterStates(SystemCounterState & systemState, std::vector<SocketCounterState> & socketStates, std::vector<CoreCounterState> & coreStates)
{
    // clear and zero-initialize all inputs
    systemState = SystemCounterState();
    socketStates.clear();
    socketStates.resize(num_sockets);
    coreStates.clear();
    coreStates.resize(num_cores);

    for (int32 core = 0; core < num_cores; ++core)
    {
        // read core counters
        if(isCoreOnline(core))
        {
          coreStates[core].readAndAggregate(MSR[core]);
          socketStates[topology[core].socket].UncoreCounterState::readAndAggregate(MSR[core]); // read package C state counters
        }
        // std::cout << "DEBUG2: "<< core<< " "<< coreStates[core].InstRetiredAny << " "<< std::endl;
    }

    for (int32 s=0; s < num_sockets; ++s)
    {
        readAndAggregateUncoreMCCounters(s, socketStates[s]);
        readAndAggregateEnergyCounters(s, socketStates[s]);
        readPackageThermalHeadroom(s, socketStates[s]);
    }

    readQPICounters(systemState);

    for (int32 core = 0; core < num_cores; ++core)
    {   // aggregate core counters into sockets
        if(isCoreOnline(core))
          socketStates[topology[core].socket].accumulateCoreState(coreStates[core]);
    }

    for (int32 s = 0; s < num_sockets; ++s)
    {   // aggregate core counters from sockets into system state and
        // aggregate socket uncore iMC, energy and package C state counters into system
        systemState.accumulateSocketState(socketStates[s]);
    }
}

CoreCounterState PCM::getCoreCounterState(uint32 core)
{
    CoreCounterState result;
    if (MSR) result.readAndAggregate(MSR[core]);
    return result;
}

uint32 PCM::getNumCores()
{
    return num_cores;
}

uint32 PCM::getNumOnlineCores()
{
    return num_online_cores;
}

uint32 PCM::getNumSockets()
{
    return num_sockets;
}

uint32 PCM::getThreadsPerCore()
{
    return threads_per_core;
}

bool PCM::getSMT()
{
    return threads_per_core > 1;
}

uint64 PCM::getNominalFrequency()
{
    return nominal_frequency;
}

ServerUncorePowerState PCM::getServerUncorePowerState(uint32 socket)
{
  ServerUncorePowerState result;
  if(server_pcicfg_uncore && server_pcicfg_uncore[socket])
  {
    server_pcicfg_uncore[socket]->freezeCounters();
    for(uint32 port=0;port < server_pcicfg_uncore[socket]->getNumQPIPorts();++port)
    {
      result.QPIClocks[port] = server_pcicfg_uncore[socket]->getQPIClocks(port);
      result.QPIL0pTxCycles[port] = server_pcicfg_uncore[socket]->getQPIL0pTxCycles(port);
      result.QPIL1Cycles[port] = server_pcicfg_uncore[socket]->getQPIL1Cycles(port);
    }
    for(uint32 channel=0;channel<server_pcicfg_uncore[socket]->getNumMCChannels();++channel)
    {
      result.DRAMClocks[channel] = server_pcicfg_uncore[socket]->getDRAMClocks(channel);
      for(uint32 cnt=0;cnt<4;++cnt)
	    result.MCCounter[channel][cnt] = server_pcicfg_uncore[socket]->getMCCounter(channel,cnt);
    }
    server_pcicfg_uncore[socket]->unfreezeCounters();
  }
  if(MSR)
  {
    uint32 refCore = socketRefCore[socket];
    TemporalThreadAffinity tempThreadAffinity(refCore);
    for(int i=0; i<4; ++i)
        MSR[refCore]->read(PCU_MSR_PMON_CTRX_ADDR[i],&(result.PCUCounter[i]));
    // std::cout<< "values read: " << result.PCUCounter[0]<<" "<<result.PCUCounter[1] << " " << result.PCUCounter[2] << " " << result.PCUCounter[3] << std::endl; 
    uint64 val=0;
    //MSR[refCore]->read(MSR_PKG_ENERGY_STATUS,&val);
    //std::cout << "Energy status: "<< val << std::endl;
    MSR[refCore]->read(MSR_PACKAGE_THERM_STATUS,&val);
    result.PackageThermalHeadroom = extractThermalHeadroom(val);
    MSR[refCore]->read(IA32_TIME_STAMP_COUNTER, &result.InvariantTSC);
    readAndAggregatePackageCStateResidencies(MSR[refCore], result);
  }

  readAndAggregateEnergyCounters(socket, result);

  return result;
}

#ifndef _MSC_VER
void print_mcfg(const char * path)
{
    int mcfg_handle = ::open(path, O_RDONLY);

    if (mcfg_handle < 0)
    {
       std::cerr << "PCM Error: Cannot open " << path << std::endl;
       throw std::exception();
    }

    MCFGHeader header;

    ssize_t read_bytes = ::read(mcfg_handle, (void *)&header, sizeof(MCFGHeader));

    if(read_bytes == 0)
    {
       std::cerr << "PCM Error: Cannot read " << path << std::endl;
       throw std::exception();
    }

    const unsigned segments = header.nrecords();
    header.print();
    std::cout << "Segments: "<<segments<<std::endl;

    for(unsigned int i=0; i<segments;++i)
    {
        MCFGRecord record;
        read_bytes = ::read(mcfg_handle, (void *)&record, sizeof(MCFGRecord));
        if(read_bytes == 0)
        {
              std::cerr << "PCM Error: Cannot read " << path << " (2)" << std::endl;
              throw std::exception();
        }
        std::cout << "Segment " <<std::dec <<  i<< " ";
        record.print();
    }

    ::close(mcfg_handle);
}
#endif


static const uint32 IMC_DEV_IDS[] = {
        0x03cb0,
        0x03cb1,
        0x03cb4,
        0x03cb5,
        0x0EB4,
        0x0EB5,
        0x0EB0,
        0x0EB1,
        0x0EF4,
        0x0EF5,
        0x0EF0,
        0x0EF1,
        0x2fb0,
        0x2fb1,
        0x2fb4,
        0x2fb5,
        0x2fd0,
        0x2fd1,
        0x2fd4,
        0x2fd5
};

std::vector<std::pair<uint32,uint32> > ServerPCICFGUncore::socket2bus;

void ServerPCICFGUncore::initSocket2Bus()
{
    if(!socket2bus.empty()) return;

    #ifdef __linux__
    const std::vector<MCFGRecord> & mcfg = PciHandleMM::getMCFGRecords();
    #else
    std::vector<MCFGRecord> mcfg;
    MCFGRecord segment;
    segment.PCISegmentGroupNumber = 0;
    segment.startBusNumber = 0;
    segment.endBusNumber = 0xff;
    mcfg.push_back(segment);
    #endif


    for(uint32 s = 0; s < mcfg.size(); ++s)
    for(uint32 bus = mcfg[s].startBusNumber; bus <= mcfg[s].endBusNumber; ++bus)
    {
        uint32 value = 0;
        try
        {
            PciHandleM h(mcfg[s].PCISegmentGroupNumber, bus, MCX_CHY_REGISTER_DEV_ADDR[0][0], MCX_CHY_REGISTER_FUNC_ADDR[0][0]);
            h.read32(0, &value);

        } catch(...)
        {
            // reached invalid bus
            return;
        }
        const uint32 vendor_id = value & 0xffff;
        const uint32 device_id = (value >> 16) & 0xffff;
        if (vendor_id != PCM_INTEL_PCI_VENDOR_ID)
           continue;

        for(uint32 i = 0; i< sizeof(IMC_DEV_IDS)/sizeof(IMC_DEV_IDS[0]) ; ++i)
        {
           // match
           if(IMC_DEV_IDS[i] == device_id)
           {
               // std::cout << "DEBUG: found bus "<<std::hex << bus << std::dec << std::endl; 
               socket2bus.push_back(std::make_pair(mcfg[s].PCISegmentGroupNumber,bus));
               break;
           }
        } 
    }
}


int getBusFromSocket(const uint32 socket)
{
    int cur_bus = 0;
    uint32 cur_socket = 0;
    // std::cout << "socket: "<< socket << std::endl;
    while(cur_socket <= socket)
    {
        // std::cout << "reading from bus 0x"<< std::hex << cur_bus << std::dec << " "; 
        PciHandleM h(0, cur_bus, 5, 0);
        uint32 cpubusno = 0;
        h.read32(0x108, &cpubusno); // CPUBUSNO register
        cur_bus = (cpubusno >> 8)& 0x0ff;
        // std::cout << "socket: "<< cur_socket<< std::hex << " cpubusno: 0x"<< std::hex << cpubusno << " "<<cur_bus<< std::dec << std::endl;
        if(socket == cur_socket)
            return cur_bus;
        ++cur_socket;
        ++cur_bus;
        if(cur_bus > 0x0ff)
           return -1;
    }

    return -1;
}

PciHandleM * ServerPCICFGUncore::createIntelPerfMonDevice(uint32 groupnr_, uint32 bus_, uint32 dev_, uint32 func_, bool checkVendor)
{
        if (PciHandleM::exists(bus_, dev_, func_))
        {
            PciHandleM * handle = new PciHandleM(groupnr_, bus_, dev_, func_);

            if(!checkVendor) return handle;
            
            uint32 vendor_id = 0;
            handle->read32(PCM_PCI_VENDOR_ID_OFFSET,&vendor_id);
            vendor_id &= 0x0ffff;

            if(vendor_id == PCM_INTEL_PCI_VENDOR_ID) return handle;

            delete handle;
        }
        return NULL;
}

ServerPCICFGUncore::ServerPCICFGUncore(uint32 socket_, PCM * pcm) : 
     bus(-1)
   , groupnr(0)
   , imcHandles(NULL)
   , num_imc_channels(0)
   , qpiLLHandles(NULL)
   , num_qpi_ports(0)
   , qpi_speed(0)
   , num_imc(0)
{

#define PCM_PCICFG_MC_INIT(controller, channel, arch) \
    MCX_CHY_REGISTER_DEV_ADDR[controller][channel] = arch##_MC##controller##_CH##channel##_REGISTER_DEV_ADDR; \
    MCX_CHY_REGISTER_FUNC_ADDR[controller][channel] = arch##_MC##controller##_CH##channel##_REGISTER_FUNC_ADDR;

    const uint32 cpu_model = pcm->getCPUModel();

    if(cpu_model == PCM::JAKETOWN || cpu_model == PCM::IVYTOWN)
    {
        PCM_PCICFG_MC_INIT(0, 0, JKTIVT)
        PCM_PCICFG_MC_INIT(0, 1, JKTIVT)
        PCM_PCICFG_MC_INIT(0, 2, JKTIVT)
        PCM_PCICFG_MC_INIT(0, 3, JKTIVT)
        PCM_PCICFG_MC_INIT(1, 0, JKTIVT)
        PCM_PCICFG_MC_INIT(1, 1, JKTIVT)
        PCM_PCICFG_MC_INIT(1, 2, JKTIVT)
        PCM_PCICFG_MC_INIT(1, 3, JKTIVT)
    }
    else if(cpu_model == PCM::HASWELLX)
    {
        PCM_PCICFG_MC_INIT(0, 0, HSX)
        PCM_PCICFG_MC_INIT(0, 1, HSX)
        PCM_PCICFG_MC_INIT(0, 2, HSX)
        PCM_PCICFG_MC_INIT(0, 3, HSX)
        PCM_PCICFG_MC_INIT(1, 0, HSX)
        PCM_PCICFG_MC_INIT(1, 1, HSX)
        PCM_PCICFG_MC_INIT(1, 2, HSX)
        PCM_PCICFG_MC_INIT(1, 3, HSX)
    }
    else
    {
        std::cout << "Error: Uncore PMU for processor with model id "<< cpu_model << " is not supported."<< std::endl;
        throw std::exception();
    }

#undef PCM_PCICFG_MC_INIT

    initSocket2Bus();
    const uint32 total_sockets_ = pcm->getNumSockets();

    if(total_sockets_ == socket2bus.size())
    {
      groupnr = socket2bus[socket_].first;
      bus = socket2bus[socket_].second;

    } else if(total_sockets_ <= 4)
    {
        bus = getBusFromSocket(socket_);
        if(bus < 0)
        {
            std::cerr << "Cannot find bus for socket "<< socket_ <<" on system with "<< total_sockets_ << " sockets."<< std::endl;
            throw std::exception();
        }
        else
        {
            std::cerr << "PCM Warning: the bus for socket "<< socket_ <<" on system with "<< total_sockets_ << " sockets could not find via PCI bus scan. Using cpubusno register. Bus = "<< bus << std::endl;
        }
    }
    else
    {
        std::cerr << "Cannot find bus for socket "<< socket_ <<" on system with "<< total_sockets_ << " sockets."<< std::endl;
        throw std::exception();
    }

    imcHandles = new PciHandleM *[8];

    {
#define PCM_PCICFG_SETUP_MC_HANDLE(controller,channel)                                 \
        {                                                                           \
            PciHandleM * handle = createIntelPerfMonDevice(groupnr, bus,            \
                MCX_CHY_REGISTER_DEV_ADDR[controller][channel], MCX_CHY_REGISTER_FUNC_ADDR[controller][channel], true); \
            if(handle) imcHandles[num_imc_channels++] = handle;                     \
        }

        PCM_PCICFG_SETUP_MC_HANDLE(0,0)
        PCM_PCICFG_SETUP_MC_HANDLE(0,1)
        PCM_PCICFG_SETUP_MC_HANDLE(0,2)
        PCM_PCICFG_SETUP_MC_HANDLE(0,3)

        if(num_imc_channels > 0) ++num_imc; // at least one memory controller
        const uint32 num_imc_channels1 = num_imc_channels;

        PCM_PCICFG_SETUP_MC_HANDLE(1,0)
        PCM_PCICFG_SETUP_MC_HANDLE(1,1)
        PCM_PCICFG_SETUP_MC_HANDLE(1,2)
        PCM_PCICFG_SETUP_MC_HANDLE(1,3)

        if(num_imc_channels > num_imc_channels1 ) ++num_imc; // another memory controller found

#undef PCM_PCICFG_SETUP_MC_HANDLE        
    }

    if (num_imc_channels == 0)
    {
        delete [] imcHandles;
        imcHandles = NULL;
        std::cerr << "PCM error: no memory controllers found." << std::endl;
        throw std::exception();
    }

    if (num_imc_channels < 3)
    {
        std::cerr << "Intel PCM: warning only " << num_imc_channels << " memory channels detected, must be >= 3." << std::endl;
    }

    if (total_sockets_ == 1) {
        /*
         * For single socket systems, do not worry at all about QPI ports.  This
         *  eliminates QPI LL programming error messages on single socket systems
         *  with BIOS that hides QPI performance counting PCI functions.  It also
         *  eliminates register programming that is not needed since no QPI traffic
         *  is possible with single socket systems.
         */
        num_qpi_ports = 0;
        std::cerr << num_imc<<" memory controllers detected with total number of "<< num_imc_channels <<" channels. " << std::endl;
        return;
    }

#ifdef PCM_NOQPI
    num_qpi_ports = 0;
    std::cerr << num_imc<<" memory controllers detected with total number of "<< num_imc_channels <<" channels. " << std::endl;
    return;
#else
    qpiLLHandles = new PciHandleM *[3];
    for(uint32 i=0; i<3; ++i)
       qpiLLHandles[i] = NULL;

	uint32 QPI_PORTX_REGISTER_DEV_ADDR[3] = { 0, 0, 0 };
	uint32 QPI_PORTX_REGISTER_FUNC_ADDR[3] = { 0, 0, 0 };

    #define PCM_PCICFG_QPI_INIT(port, arch) \
        QPI_PORTX_REGISTER_DEV_ADDR[port] = arch##_QPI_PORT##port##_REGISTER_DEV_ADDR; \
        QPI_PORTX_REGISTER_FUNC_ADDR[port] = arch##_QPI_PORT##port##_REGISTER_FUNC_ADDR;

    if(cpu_model == PCM::JAKETOWN || cpu_model == PCM::IVYTOWN)
    {
        PCM_PCICFG_QPI_INIT(0, JKTIVT);
        PCM_PCICFG_QPI_INIT(1, JKTIVT);
        PCM_PCICFG_QPI_INIT(2, JKTIVT);
    }
    else if(cpu_model == PCM::HASWELLX)
    {
        PCM_PCICFG_QPI_INIT(0, HSX);
        PCM_PCICFG_QPI_INIT(1, HSX);
        PCM_PCICFG_QPI_INIT(2, HSX);
    }
    else
    {
        std::cout << "Error: Uncore PMU for processor with model id "<< cpu_model << " is not supported."<< std::endl;
        throw std::exception();
    }

    #undef PCM_PCICFG_QPI_INIT

    try
    {
        {
            PciHandleM * handle = createIntelPerfMonDevice(groupnr, bus, QPI_PORTX_REGISTER_DEV_ADDR[0], QPI_PORTX_REGISTER_FUNC_ADDR[0], true);
            if (handle)
                qpiLLHandles[num_qpi_ports++] = handle;
            else
                std::cerr << "ERROR: QPI LL monitoring device ("<< groupnr<<":"<<bus<<":"<< QPI_PORTX_REGISTER_DEV_ADDR[0] <<":"<<  QPI_PORTX_REGISTER_FUNC_ADDR[0] << ") is missing. The QPI statistics will be incomplete or missing."<< std::endl;
        }
        {
            PciHandleM * handle = createIntelPerfMonDevice(groupnr, bus, QPI_PORTX_REGISTER_DEV_ADDR[1], QPI_PORTX_REGISTER_FUNC_ADDR[1], true);
            if (handle)
                qpiLLHandles[num_qpi_ports++] = handle;
            else
                std::cerr << "ERROR: QPI LL monitoring device ("<< groupnr<<":"<<bus<<":"<< QPI_PORTX_REGISTER_DEV_ADDR[1] <<":"<<  QPI_PORTX_REGISTER_FUNC_ADDR[1] << ") is missing. The QPI statistics will be incomplete or missing."<< std::endl;
        }
        {
            PciHandleM * handle = createIntelPerfMonDevice(groupnr, bus, QPI_PORTX_REGISTER_DEV_ADDR[2], QPI_PORTX_REGISTER_FUNC_ADDR[2], true);
            if (handle) qpiLLHandles[num_qpi_ports++] = handle;
        }
    }
    catch (...)
    {
        std::cerr << "PCM Error: can not create QPI LL handles." <<std::endl;
        for(uint32 i=0; i<num_qpi_ports; ++i)
            if (qpiLLHandles[i]) delete qpiLLHandles[i];
        delete[] qpiLLHandles;
        qpiLLHandles = NULL;
        throw std::exception();
    }
#endif
    std::cerr << "Socket "<<socket_<<": "<<
        num_imc<<" memory controllers detected with total number of "<< num_imc_channels <<" channels. "<<
        num_qpi_ports<< " QPI ports detected."<<std::endl;
}

ServerPCICFGUncore::~ServerPCICFGUncore()
{
    if (imcHandles)
    {
        for (uint32 i = 0; i < num_imc_channels; ++i)
            if (imcHandles[i]) delete imcHandles[i];
        delete[] imcHandles;
        imcHandles = NULL;
    }

    if (qpiLLHandles)
    {
        for(uint32 i=0; i<num_qpi_ports; ++i)
            if (qpiLLHandles[i]) delete qpiLLHandles[i];
        delete[] qpiLLHandles;
        qpiLLHandles = NULL;
    }
}


void ServerPCICFGUncore::program()
{
    for (uint32 i = 0; i < num_imc_channels; ++i)
    {
        // imc PMU

        // freeze enable
        imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN);
        // freeze
        imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN + MC_CH_PCI_PMON_BOX_CTL_FRZ);

#ifdef PCM_UNCORE_PMON_BOX_CHECK_STATUS
        uint32 val = 0;
        imcHandles[i]->read32(MC_CH_PCI_PMON_BOX_CTL_ADDR, &val);
        if ((val & UNCORE_PMON_BOX_CTL_VALID_BITS_MASK) != (MC_CH_PCI_PMON_BOX_CTL_FRZ_EN + MC_CH_PCI_PMON_BOX_CTL_FRZ))
		{
            std::cerr << "ERROR: IMC counter programming seems not to work. MC_CH" << i << "_PCI_PMON_BOX_CTL=0x" << std::hex << val << std::endl;
			std::cerr << "       Please see BIOS options to enable the export of performance monitoring devices." << std::endl;
		}
#endif

        // enable counter 0
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL0_ADDR, MC_CH_PCI_PMON_CTL_EN);

        // monitor reads on counter 0: CAS_COUNT.RD
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL0_ADDR, MC_CH_PCI_PMON_CTL_EN + MC_CH_PCI_PMON_CTL_EVENT(0x04) + MC_CH_PCI_PMON_CTL_UMASK(3));

        // enable counter 1
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL1_ADDR, MC_CH_PCI_PMON_CTL_EN);

        // monitor writes on counter 1: CAS_COUNT.WR
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL1_ADDR, MC_CH_PCI_PMON_CTL_EN + MC_CH_PCI_PMON_CTL_EVENT(0x04) + MC_CH_PCI_PMON_CTL_UMASK(12));

        // enable counter 2
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL2_ADDR, MC_CH_PCI_PMON_CTL_EN);

        // monitor partial writes on counter 2: CAS_COUNT.RD_UNDERFILL
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL2_ADDR, MC_CH_PCI_PMON_CTL_EN + MC_CH_PCI_PMON_CTL_EVENT(0x04) + MC_CH_PCI_PMON_CTL_UMASK(2));

        // enable fixed counter (DRAM clocks)
        imcHandles[i]->write32(MC_CH_PCI_PMON_FIXED_CTL_ADDR, MC_CH_PCI_PMON_FIXED_CTL_EN);

        // reset it
        imcHandles[i]->write32(MC_CH_PCI_PMON_FIXED_CTL_ADDR, MC_CH_PCI_PMON_FIXED_CTL_EN + MC_CH_PCI_PMON_FIXED_CTL_RST);

        // reset counters values
        imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN + MC_CH_PCI_PMON_BOX_CTL_FRZ + MC_CH_PCI_PMON_BOX_CTL_RST_COUNTERS);

        // unfreeze counters
        imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN);
    }

    for (uint32 i = 0; i < num_qpi_ports; ++i)
    {
        // QPI LL PMU

        // freeze enable
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN);
        // freeze
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN + Q_P_PCI_PMON_BOX_CTL_RST_FRZ);

#ifdef PCM_UNCORE_PMON_BOX_CHECK_STATUS
        uint32 val = 0;
        qpiLLHandles[i]->read32(Q_P_PCI_PMON_BOX_CTL_ADDR, &val);
        if ((val & UNCORE_PMON_BOX_CTL_VALID_BITS_MASK) != (Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN + Q_P_PCI_PMON_BOX_CTL_RST_FRZ))
		{
            std::cerr << "ERROR: QPI LL counter programming seems not to work. Q_P" << i << "_PCI_PMON_BOX_CTL=0x" << std::hex << val << std::endl;
			std::cerr << "       Please see BIOS options to enable the export of performance monitoring devices (devices 8 and 9: function 2)." << std::endl;
		}
#endif

        // enable counter 0
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL0_ADDR, Q_P_PCI_PMON_CTL_EN);

        // monitor DRS data received on counter 0: RxL_FLITS_G1.DRS_DATA
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL0_ADDR, Q_P_PCI_PMON_CTL_EN + Q_P_PCI_PMON_CTL_EVENT(0x02) + Q_P_PCI_PMON_CTL_EVENT_EXT + Q_P_PCI_PMON_CTL_UMASK(8));

        // enable counter 1
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL1_ADDR, Q_P_PCI_PMON_CTL_EN);

        // monitor NCB data received on counter 1: RxL_FLITS_G2.NCB_DATA
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL1_ADDR, Q_P_PCI_PMON_CTL_EN + Q_P_PCI_PMON_CTL_EVENT(0x03) + Q_P_PCI_PMON_CTL_EVENT_EXT + Q_P_PCI_PMON_CTL_UMASK(4));

        // enable counter 2
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL2_ADDR, Q_P_PCI_PMON_CTL_EN);
	
	    // monitor outgoing data+nondata flits on counter 2: TxL_FLITS_G0.DATA + TxL_FLITS_G0.NON_DATA
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL2_ADDR, Q_P_PCI_PMON_CTL_EN + Q_P_PCI_PMON_CTL_EVENT(0x00) + Q_P_PCI_PMON_CTL_UMASK(6));

	// enable counter 3
	qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL3_ADDR, Q_P_PCI_PMON_CTL_EN);
	
	// monitor QPI clocks
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL3_ADDR, Q_P_PCI_PMON_CTL_EN + Q_P_PCI_PMON_CTL_EVENT(0x14)); // QPI clocks (CLOCKTICKS)
	
        // reset counters values
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN + Q_P_PCI_PMON_BOX_CTL_RST_FRZ + Q_P_PCI_PMON_BOX_CTL_RST_COUNTERS);

        // unfreeze counters
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN);
    }
}

uint64 ServerPCICFGUncore::getImcReads()
{
    uint64 result = 0;

    for (uint32 i = 0; i < num_imc_channels; ++i)
    {
        uint64 value = 0;
        imcHandles[i]->read64(MC_CH_PCI_PMON_CTR0_ADDR, &value);
        result += value;
    }

    return result;
}

uint64 ServerPCICFGUncore::getImcWrites()
{
    uint64 result = 0;

    for (uint32 i = 0; i < num_imc_channels; ++i)
    {
        uint64 value = 0;
        imcHandles[i]->read64(MC_CH_PCI_PMON_CTR1_ADDR, &value);
        result += value;
    }

    return result;
}

uint64 ServerPCICFGUncore::getIncomingDataFlits(uint32 port)
{
    uint64 drs = 0, ncb = 0;

    if (port >= num_qpi_ports)
        return 0;

    qpiLLHandles[port]->read64(Q_P_PCI_PMON_CTR0_ADDR, &drs);
    qpiLLHandles[port]->read64(Q_P_PCI_PMON_CTR1_ADDR, &ncb);
    
    return drs + ncb;
}

uint64 ServerPCICFGUncore::getOutgoingDataNonDataFlits(uint32 port)
{
    return getQPILLCounter(port,2);
}

void ServerPCICFGUncore::program_power_metrics(int mc_profile)
{
    for (uint32 i = 0; i < num_qpi_ports; ++i)
    {
        // QPI LL PMU

        // freeze enable
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN);
        // freeze
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN + Q_P_PCI_PMON_BOX_CTL_RST_FRZ);

#ifdef PCM_UNCORE_PMON_BOX_CHECK_STATUS
        uint32 val = 0;
        qpiLLHandles[i]->read32(Q_P_PCI_PMON_BOX_CTL_ADDR, &val);
        if ((val & UNCORE_PMON_BOX_CTL_VALID_BITS_MASK) != (Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN + Q_P_PCI_PMON_BOX_CTL_RST_FRZ))
		{
            std::cerr << "ERROR: QPI LL counter programming seems not to work. Q_P" << i << "_PCI_PMON_BOX_CTL=0x" << std::hex << val << std::endl;
	    std::cerr << "       Please see BIOS options to enable the export of performance monitoring devices." << std::endl;
		}
#endif

        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL3_ADDR, Q_P_PCI_PMON_CTL_EN);
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL3_ADDR, Q_P_PCI_PMON_CTL_EN + Q_P_PCI_PMON_CTL_EVENT(0x14)); // QPI clocks (CLOCKTICKS)

        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL0_ADDR, Q_P_PCI_PMON_CTL_EN);
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL0_ADDR, Q_P_PCI_PMON_CTL_EN + Q_P_PCI_PMON_CTL_EVENT(0x0D)); // L0p Tx Cycles (TxL0P_POWER_CYCLES)
	
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL2_ADDR, Q_P_PCI_PMON_CTL_EN);
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_CTL2_ADDR, Q_P_PCI_PMON_CTL_EN + Q_P_PCI_PMON_CTL_EVENT(0x12)); // L1 Cycles (L1_POWER_CYCLES)
	
        // reset counters values
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN + Q_P_PCI_PMON_BOX_CTL_RST_FRZ + Q_P_PCI_PMON_BOX_CTL_RST_COUNTERS);

        // unfreeze counters
        qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN);
    }
    
	uint32 MCCntConfig[4] = {0,0,0,0};
    switch(mc_profile)
    {
      case 0: // POWER_CKE_CYCLES.RANK0 and POWER_CKE_CYCLES.RANK1
	MCCntConfig[0] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(1) + MC_CH_PCI_PMON_CTL_INVERT + MC_CH_PCI_PMON_CTL_THRESH(1);
	MCCntConfig[1] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(1) + MC_CH_PCI_PMON_CTL_THRESH(1) + MC_CH_PCI_PMON_CTL_EDGE_DET;
	MCCntConfig[2] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(2) + MC_CH_PCI_PMON_CTL_INVERT + MC_CH_PCI_PMON_CTL_THRESH(1);
	MCCntConfig[3] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(2) + MC_CH_PCI_PMON_CTL_THRESH(1) + MC_CH_PCI_PMON_CTL_EDGE_DET;
	break;
      case  1: // POWER_CKE_CYCLES.RANK2 and POWER_CKE_CYCLES.RANK3
	MCCntConfig[0] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(4) + MC_CH_PCI_PMON_CTL_INVERT + MC_CH_PCI_PMON_CTL_THRESH(1);
	MCCntConfig[1] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(4) + MC_CH_PCI_PMON_CTL_THRESH(1) + MC_CH_PCI_PMON_CTL_EDGE_DET;
	MCCntConfig[2] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(8) + MC_CH_PCI_PMON_CTL_INVERT + MC_CH_PCI_PMON_CTL_THRESH(1);
	MCCntConfig[3] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(8) + MC_CH_PCI_PMON_CTL_THRESH(1) + MC_CH_PCI_PMON_CTL_EDGE_DET;
	break;
      case 2: // POWER_CKE_CYCLES.RANK4 and POWER_CKE_CYCLES.RANK5
	MCCntConfig[0] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(0x10) + MC_CH_PCI_PMON_CTL_INVERT + MC_CH_PCI_PMON_CTL_THRESH(1);
	MCCntConfig[1] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(0x10) + MC_CH_PCI_PMON_CTL_THRESH(1) + MC_CH_PCI_PMON_CTL_EDGE_DET;
	MCCntConfig[2] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(0x20) + MC_CH_PCI_PMON_CTL_INVERT + MC_CH_PCI_PMON_CTL_THRESH(1);
	MCCntConfig[3] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(0x20) + MC_CH_PCI_PMON_CTL_THRESH(1) + MC_CH_PCI_PMON_CTL_EDGE_DET;
	break;
      case 3: // POWER_CKE_CYCLES.RANK6 and POWER_CKE_CYCLES.RANK7
	MCCntConfig[0] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(0x40) + MC_CH_PCI_PMON_CTL_INVERT + MC_CH_PCI_PMON_CTL_THRESH(1);
	MCCntConfig[1] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(0x40) + MC_CH_PCI_PMON_CTL_THRESH(1) + MC_CH_PCI_PMON_CTL_EDGE_DET;
	MCCntConfig[2] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(0x80) + MC_CH_PCI_PMON_CTL_INVERT + MC_CH_PCI_PMON_CTL_THRESH(1);
	MCCntConfig[3] = MC_CH_PCI_PMON_CTL_EVENT(0x83) + MC_CH_PCI_PMON_CTL_UMASK(0x80) + MC_CH_PCI_PMON_CTL_THRESH(1) + MC_CH_PCI_PMON_CTL_EDGE_DET;
	break;
      case 4: // POWER_SELF_REFRESH
	MCCntConfig[0] = MC_CH_PCI_PMON_CTL_EVENT(0x43);
	MCCntConfig[1] = MC_CH_PCI_PMON_CTL_EVENT(0x43) + MC_CH_PCI_PMON_CTL_THRESH(1) + MC_CH_PCI_PMON_CTL_EDGE_DET;
        MCCntConfig[2] = MC_CH_PCI_PMON_CTL_EVENT(0x85);
	break;
    }

    for (uint32 i = 0; i < num_imc_channels; ++i)
    {
        // imc PMU

        // freeze enable
        imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN);
        // freeze
        imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN + MC_CH_PCI_PMON_BOX_CTL_FRZ);

#ifdef PCM_UNCORE_PMON_BOX_CHECK_STATUS
        uint32 val = 0;
        imcHandles[i]->read32(MC_CH_PCI_PMON_BOX_CTL_ADDR, &val);
        if ((val & UNCORE_PMON_BOX_CTL_VALID_BITS_MASK) != (MC_CH_PCI_PMON_BOX_CTL_FRZ_EN + MC_CH_PCI_PMON_BOX_CTL_FRZ))
		{
            std::cerr << "ERROR: IMC counter programming seems not to work. MC_CH" << i << "_PCI_PMON_BOX_CTL=0x" << std::hex << val << std::endl;
	    std::cerr << "       Please see BIOS options to enable the export of performance monitoring devices." << std::endl;
		}
#endif

        // enable fixed counter (DRAM clocks)
        imcHandles[i]->write32(MC_CH_PCI_PMON_FIXED_CTL_ADDR, MC_CH_PCI_PMON_FIXED_CTL_EN);

        // reset it
        imcHandles[i]->write32(MC_CH_PCI_PMON_FIXED_CTL_ADDR, MC_CH_PCI_PMON_FIXED_CTL_EN + MC_CH_PCI_PMON_FIXED_CTL_RST);

	
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL0_ADDR, MC_CH_PCI_PMON_CTL_EN);
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL0_ADDR, MC_CH_PCI_PMON_CTL_EN + MCCntConfig[0]);
	
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL1_ADDR, MC_CH_PCI_PMON_CTL_EN);
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL1_ADDR, MC_CH_PCI_PMON_CTL_EN + MCCntConfig[1]);
	
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL2_ADDR, MC_CH_PCI_PMON_CTL_EN);
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL2_ADDR, MC_CH_PCI_PMON_CTL_EN + MCCntConfig[2]);
	
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL3_ADDR, MC_CH_PCI_PMON_CTL_EN);
        imcHandles[i]->write32(MC_CH_PCI_PMON_CTL3_ADDR, MC_CH_PCI_PMON_CTL_EN + MCCntConfig[3]);

        // reset counters values
        imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN + MC_CH_PCI_PMON_BOX_CTL_FRZ + MC_CH_PCI_PMON_BOX_CTL_RST_COUNTERS);

        // unfreeze counters
        imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN);
    }
}

void ServerPCICFGUncore::freezeCounters()
{
    for (uint32 i = 0; i < num_qpi_ports; ++i)
    {
		qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN + Q_P_PCI_PMON_BOX_CTL_RST_FRZ);
	}
    for (uint32 i = 0; i < num_imc_channels; ++i)
    {
		imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN + MC_CH_PCI_PMON_BOX_CTL_FRZ);
	}
}

void ServerPCICFGUncore::unfreezeCounters()
{
    for (uint32 i = 0; i < num_qpi_ports; ++i)
    {
		qpiLLHandles[i]->write32(Q_P_PCI_PMON_BOX_CTL_ADDR, Q_P_PCI_PMON_BOX_CTL_RST_FRZ_EN);
	}
	for (uint32 i = 0; i < num_imc_channels; ++i)
    {
		imcHandles[i]->write32(MC_CH_PCI_PMON_BOX_CTL_ADDR, MC_CH_PCI_PMON_BOX_CTL_FRZ_EN);
	}
}

uint64 ServerPCICFGUncore::getQPIClocks(uint32 port)
{
    uint64 res = 0;

    if (port >= num_qpi_ports)
        return 0;

    qpiLLHandles[port]->read64(Q_P_PCI_PMON_CTR3_ADDR, &res);
    
    return res;
}

uint64 ServerPCICFGUncore::getQPIL0pTxCycles(uint32 port)
{
    uint64 res = 0;

    if (port >= num_qpi_ports)
        return 0;

    qpiLLHandles[port]->read64(Q_P_PCI_PMON_CTR0_ADDR, &res);

    return res;
}

uint64 ServerPCICFGUncore::getQPIL1Cycles(uint32 port)
{
    uint64 res = 0;

    if (port >= num_qpi_ports)
        return 0;

    qpiLLHandles[port]->read64(Q_P_PCI_PMON_CTR2_ADDR, &res);

    return res;
}

uint64 ServerPCICFGUncore::getDRAMClocks(uint32 channel)
{
    uint64 result = 0;

    if(channel < num_imc_channels) 
      imcHandles[channel]->read64(MC_CH_PCI_PMON_FIXED_CTR_ADDR, &result);

    return result;
}

uint64 ServerPCICFGUncore::getMCCounter(uint32 channel, uint32 counter)
{
    uint64 result = 0;
    
    if(channel < num_imc_channels) 
    {
      switch(counter)
      {
	case 0:
	  imcHandles[channel]->read64(MC_CH_PCI_PMON_CTR0_ADDR, &result);
	  break;
	case 1:
	  imcHandles[channel]->read64(MC_CH_PCI_PMON_CTR1_ADDR, &result);
	  break;
	case 2:
	  imcHandles[channel]->read64(MC_CH_PCI_PMON_CTR2_ADDR, &result);
	  break;
	case 3:
	  imcHandles[channel]->read64(MC_CH_PCI_PMON_CTR3_ADDR, &result);
	  break;
      }
    }
    
    return result;
}

uint64 ServerPCICFGUncore::getQPILLCounter(uint32 port, uint32 counter)
{
    uint64 result = 0;
    
    if(port < num_qpi_ports) 
    {
      switch(counter)
      {
	case 0:
	  qpiLLHandles[port]->read64(Q_P_PCI_PMON_CTR0_ADDR, &result);
	  break;
	case 1:
	  qpiLLHandles[port]->read64(Q_P_PCI_PMON_CTR1_ADDR, &result);
	  break;
	case 2:
	  qpiLLHandles[port]->read64(Q_P_PCI_PMON_CTR2_ADDR, &result);
	  break;
	case 3:
	  qpiLLHandles[port]->read64(Q_P_PCI_PMON_CTR3_ADDR, &result);
	  break;
      }
    }
    
    return result;
}

void ServerPCICFGUncore::enableJKTWorkaround(bool enable)
{
    {
        PciHandleM reg(groupnr,bus,14,0);
        uint32 value = 0;
        reg.read32(0x84, &value);
        if(enable)
            value |= 2;
        else
            value &= (~2);
        reg.write32(0x84, value);
    }
    {
        PciHandleM reg(groupnr,bus,8,0);
        uint32 value = 0;
        reg.read32(0x80, &value);
        if(enable)
            value |= 2;
        else
            value &= (~2);
        reg.write32(0x80, value);
    }
    {
        PciHandleM reg(groupnr,bus,9,0);
        uint32 value = 0;
        reg.read32(0x80, &value);
        if(enable)
            value |= 2;
        else
            value &= (~2);
        reg.write32(0x80, value);
    }
}


uint64 ServerPCICFGUncore::computeQPISpeed(const uint32 core_nr, const int cpumodel)
{
    if(qpi_speed.size()==0)
    {
	qpi_speed.resize(num_qpi_ports);
        for (uint32 i=0; i<num_qpi_ports; ++i) {
           uint32 addr = QPI_PORT0_MISC_REGISTER_DEV_ADDR;
           if (i==1) addr = QPI_PORT1_MISC_REGISTER_DEV_ADDR;
           if (i==2) addr = QPI_PORT2_MISC_REGISTER_DEV_ADDR;
           PciHandleM reg(groupnr,bus,addr,QPI_PORT0_MISC_REGISTER_FUNC_ADDR);
           uint32 value = 0;
           reg.read32(QPI_RATE_STATUS_ADDR, &value);
           value &= 7; // extract lower 3 bits
           if(value) qpi_speed[i] = (4000000000ULL + ((uint64)value)*800000000ULL)*2ULL;
           if(qpi_speed[i]==0)
           {
               std::cerr << "Warning: QPI_RATE_STATUS register is not available on port "<< i <<". Computing QPI speed using a measurement loop." << std::endl;
 
               // compute qpi speed
               const uint64 timerGranularity = 1000000ULL; // mks
    
               PCM * pcm = PCM::getInstance();
               uint64 startClocks = getQPIClocks(i);
               uint64 startTSC = pcm->getTickCount(timerGranularity, core_nr);
               uint64 endTSC;
               do
               {
                    endTSC = pcm->getTickCount(timerGranularity, core_nr);
               } while (endTSC - startTSC < 200000ULL); // spin for 200 ms

               uint64 endClocks = getQPIClocks(i);

               qpi_speed[i] = ((std::max)((endClocks - startClocks) * 16ULL * timerGranularity / (endTSC - startTSC),0ULL));
               if(cpumodel == PCM::HASWELLX) {
                  qpi_speed[i] /=2; // HSX runs QPI clocks with doubled speed
               }
           }
         }
    }
    if(!qpi_speed.empty())
    {
        return *std::max_element(qpi_speed.begin(),qpi_speed.end());
    }
    else
    {
        return 0;
    }
}

#ifdef _MSC_VER
DWORD WINAPI WatchDogProc(LPVOID state)
#else
void * WatchDogProc(void * state)
#endif
{
    CounterWidthExtender * ext = (CounterWidthExtender * ) state;
    while(1)
    {
#ifdef _MSC_VER
		Sleep(10000);
#else
        sleep(10);
#endif
        /* uint64 dummy = */ ext->read();
    }
    return NULL;
}

uint32 PCM::CX_MSR_PMON_CTRY(uint32 Cbo, uint32 Ctr) const
{
    if(JAKETOWN == cpu_model || IVYTOWN == cpu_model)
    {
        return JKT_C0_MSR_PMON_CTR0 + ((JKTIVT_CBO_MSR_STEP)*Cbo) + Ctr;

    } else if(HASWELLX == cpu_model)
    {
        return HSX_C0_MSR_PMON_CTR0 + ((HSX_CBO_MSR_STEP)*Cbo) + Ctr;
    }
    return 0;
}

uint32 PCM::CX_MSR_PMON_BOX_FILTER(uint32 Cbo) const
{
    if(JAKETOWN == cpu_model || IVYTOWN == cpu_model)
    {
        return JKT_C0_MSR_PMON_BOX_FILTER + ((JKTIVT_CBO_MSR_STEP)*Cbo);

    } else if(HASWELLX == cpu_model)
    {
        return HSX_C0_MSR_PMON_BOX_FILTER + ((HSX_CBO_MSR_STEP)*Cbo);
    }

    return 0;
}

uint32 PCM::CX_MSR_PMON_BOX_FILTER1(uint32 Cbo) const
{
    if(IVYTOWN == cpu_model)
    {
        return IVT_C0_MSR_PMON_BOX_FILTER1 + ((JKTIVT_CBO_MSR_STEP)*Cbo);

    } else if(HASWELLX == cpu_model)
    {
        return HSX_C0_MSR_PMON_BOX_FILTER1 + ((HSX_CBO_MSR_STEP)*Cbo);
    }
    return 0;
}

uint32 PCM::CX_MSR_PMON_CTLY(uint32 Cbo, uint32 Ctl) const
{
    if(JAKETOWN == cpu_model || IVYTOWN == cpu_model)
    {
        return JKT_C0_MSR_PMON_CTL0 + ((JKTIVT_CBO_MSR_STEP)*Cbo) + Ctl;

    } else if(HASWELLX == cpu_model)
    {
        return HSX_C0_MSR_PMON_CTL0 + ((HSX_CBO_MSR_STEP)*Cbo) + Ctl;
    }
    return 0;
}

uint32 PCM::CX_MSR_PMON_BOX_CTL(uint32 Cbo) const
{
    if(JAKETOWN == cpu_model || IVYTOWN == cpu_model)
    {
        return JKT_C0_MSR_PMON_BOX_CTL + ((JKTIVT_CBO_MSR_STEP)*Cbo);

    } else if(HASWELLX == cpu_model)
    {
        return HSX_C0_MSR_PMON_BOX_CTL + ((HSX_CBO_MSR_STEP)*Cbo);
    }
    return 0;
}

uint32 PCM::getMaxNumOfCBoxes() const
{
    if(1)
    {
        /*
         * There is one CBox per physical core.  This calculation will get us
         *  the number of physical cores per socket which is the expected
         *  value to be returned.
         */
        return num_phys_cores_per_socket;
    }
    return 0;
}

void PCM::programCboOpcodeFilter(const uint32 opc, const uint32 cbo, SafeMsrHandle * msr)
{
    if(JAKETOWN == cpu_model)
    {
        msr->write(CX_MSR_PMON_BOX_FILTER(cbo), JKT_CBO_MSR_PMON_BOX_FILTER_OPC(opc));

    } else if(IVYTOWN == cpu_model || HASWELLX == cpu_model)
    {
        msr->write(CX_MSR_PMON_BOX_FILTER1(cbo), IVTHSX_CBO_MSR_PMON_BOX_FILTER1_OPC(opc));
    }
}

void PCM::programPCIeMissCounters(const PCM::PCIeEventCode event_, const uint32 tid_)
{
    programPCIeCounters(event_,tid_,1);
}

void PCM::programPCIeCounters(const PCM::PCIeEventCode event_, const uint32 tid_, const uint32 miss_)
{
    for (int32 i = 0; (i < num_sockets) && MSR; ++i)
    {
        uint32 refCore = socketRefCore[i];
        TemporalThreadAffinity tempThreadAffinity(refCore); // speedup trick for Linux

        for(uint32 cbo = 0; cbo < getMaxNumOfCBoxes(); ++cbo)
        {
            // freeze enable
            MSR[refCore]->write(CX_MSR_PMON_BOX_CTL(cbo), CBO_MSR_PMON_BOX_CTL_FRZ_EN);
            // freeze
            MSR[refCore]->write(CX_MSR_PMON_BOX_CTL(cbo), CBO_MSR_PMON_BOX_CTL_FRZ_EN + CBO_MSR_PMON_BOX_CTL_FRZ);

#ifdef PCM_UNCORE_PMON_BOX_CHECK_STATUS
            uint64 val = 0;
            MSR[refCore]->read(CX_MSR_PMON_BOX_CTL(cbo), &val);
            if ((val & UNCORE_PMON_BOX_CTL_VALID_BITS_MASK) != (CBO_MSR_PMON_BOX_CTL_FRZ_EN + CBO_MSR_PMON_BOX_CTL_FRZ))
            {
                std::cerr << "ERROR: CBO counter programming seems not to work. ";
                std::cerr << "C" << std::dec << cbo << "_MSR_PMON_BOX_CTL=0x" << std::hex << val << std::endl;
            }
#endif

            programCboOpcodeFilter(event_, cbo, MSR[refCore]);

            if(HASWELLX == cpu_model && tid_ != 0)
                MSR[refCore]->write(CX_MSR_PMON_BOX_FILTER(cbo), tid_);

            MSR[refCore]->write(CX_MSR_PMON_CTLY(cbo, 0), CBO_MSR_PMON_CTL_EN);
            // TOR_INSERTS.OPCODE event
            MSR[refCore]->write(CX_MSR_PMON_CTLY(cbo, 0), CBO_MSR_PMON_CTL_EN + CBO_MSR_PMON_CTL_EVENT(0x35) + (CBO_MSR_PMON_CTL_UMASK(1) | (miss_?CBO_MSR_PMON_CTL_UMASK(0x3):0ULL)) + (tid_?CBO_MSR_PMON_CTL_TID_EN:0ULL));

            // reset counter values
            MSR[refCore]->write(CX_MSR_PMON_BOX_CTL(cbo), CBO_MSR_PMON_BOX_CTL_FRZ_EN + CBO_MSR_PMON_BOX_CTL_FRZ + CBO_MSR_PMON_BOX_CTL_RST_COUNTERS);

            // unfreeze counters
            MSR[refCore]->write(CX_MSR_PMON_BOX_CTL(cbo), CBO_MSR_PMON_BOX_CTL_FRZ_EN);
        }
    }
}

PCIeCounterState PCM::getPCIeCounterState(const uint32 socket_)
{
    PCIeCounterState result;

    uint32 refCore = socketRefCore[socket_];
    TemporalThreadAffinity tempThreadAffinity(refCore); // speedup trick for Linux

    for(uint32 cbo=0; cbo < getMaxNumOfCBoxes(); ++cbo)
    {
        uint64 ctrVal = 0;
        MSR[refCore]->read(CX_MSR_PMON_CTRY(cbo, 0), &ctrVal);
        result.data += ctrVal;
    }
    return result;
}
