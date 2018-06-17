#pragma once

#include <core/global_definitions.hpp>
#include <hype.hpp>
#include <string>

namespace CoGaDB {

  class RuntimeConfiguration {
   public:
    static RuntimeConfiguration& instance();
    void setPathToDatabase(const std::string&);
    const std::string& getPathToDatabase();
    void setOptimizer(const std::string&);
    const std::string& getOptimizer();
    void setQueryOptimizationHeuristic(hype::QueryOptimizationHeuristic);
    hype::QueryOptimizationHeuristic getQueryOptimizationHeuristic() const;
    void setParallelizationModeForTwoPhasePhysicalOptimizationQueryPlans(
        ParallelizationMode);
    ParallelizationMode
    getParallelizationModeForTwoPhasePhysicalOptimizationQueryPlans() const;
    GPUBufferManagementStrategy getGPUBufferManagementStrategy() const;
    void setGPUBufferManagementStrategy(GPUBufferManagementStrategy);
    BufferManagementStrategy getBufferManagementStrategy() const;
    void setBufferManagementStrategy(BufferManagementStrategy);

    TableLoaderMode getTableLoaderMode() const;
    void setTableLoaderMode(TableLoaderMode);

    bool getPrintQueryPlan() const;
    void setPrintQueryPlan(bool);
    bool getProfileQueries() const;
    void setProfileQueries(bool);
    void setQueryChoppingEnabled(bool);
    bool isQueryChoppingEnabled() const;
    void setGlobalDeviceConstraint(const hype::DeviceConstraint&);
    hype::DeviceConstraint getGlobalDeviceConstraint() const;

   private:
    RuntimeConfiguration();
    RuntimeConfiguration(const RuntimeConfiguration&);
    RuntimeConfiguration& operator=(RuntimeConfiguration&);
    std::string path_to_database_;
    std::string optimizer_name_;
    bool print_query_plan_;
    bool enable_profiling_;
    bool enable_query_chopping_;
    hype::DeviceConstraint global_device_constraint_;
    ParallelizationMode two_phase_physical_optimization_parallelization_mode;
    hype::QueryOptimizationHeuristic hybrid_optimizer_heuristic;
    GPUBufferManagementStrategy gpu_buffer_management_strategy_;
    TableLoaderMode table_loader_mode_;
  };

}  // end namespace CogaDB
