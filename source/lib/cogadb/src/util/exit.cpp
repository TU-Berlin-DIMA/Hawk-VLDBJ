
#include <core/global_definitions.hpp>
#include <core/processor_data_cache.hpp>

#include <stdlib.h>
#include <core/variable_manager.hpp>
#include <util/filesystem.hpp>

namespace CoGaDB {

void callGlobalCleanupRoutines() {
  if (VariableManager::instance().getVariableValueBoolean("createPIDfile")) {
    CoGaDB::deletePIDFile();
  }
  OnExitCloseDataPlacementThread();
}

void exit(int status) {
  callGlobalCleanupRoutines();
  if (status != EXIT_SUCCESS) {
#ifndef __APPLE__
    quick_exit(status);
#else
    abort();
#endif
  }

  std::exit(status);
}
}
