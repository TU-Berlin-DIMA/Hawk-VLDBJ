

#include <util/init.hpp>

#include <core/global_definitions.hpp>
#include <core/processor_data_cache.hpp>

#include <stdlib.h>
#include <core/variable_manager.hpp>
#include <util/filesystem.hpp>

#include <core/runtime_configuration.hpp>
//#include <util/reduce_by_keys.hpp>

//#include <core/gpu_column_cache.hpp>

//#include <query_processing/query_processor.hpp>
//#include <parser/commandline_interpreter.hpp>

#include <boost/lexical_cast.hpp>
//#include <boost/random.hpp>
//#include <boost/generator_iterator.hpp>

#include <utility>
//#include <parser/generated/Parser.h>

#include <core/processor_data_cache.hpp>
#include <query_compilation/ocl_data_cache.hpp>
#include <util/filesystem.hpp>
#include <util/hardware_detector.hpp>
#include <util/opencl/prefix_sum.hpp>
#include <util/opencl_runtime.hpp>

namespace CoGaDB {

void initSingletons() {
  /* init old hardware detector */
  CoGaDB::HardwareDetector::instance();
  /* init variable manager */
  CoGaDB::VariableManager::instance();
  /* init OpenCL runtime at startup */
  CoGaDB::OCL_Runtime::instance();
  /* init OpenCL data caches*/
  CoGaDB::OCL_DataCaches::instance();
  /* init OpenCL kernel cache */
  CoGaDB::OCL_Kernels::instance();
}

void setUpVariableManager(char* argv[]) {
  // check if argument one is startup.coga
  std::string argument(argv[1]);
  std::string databaseFarmDir("/.cogadb/database_farm/");
  std::size_t hasCogaDir = argument.find(databaseFarmDir);
  std::string startupFile("startup.coga");
  std::size_t indexOfStartupFile = argument.find(startupFile);
  std::cout << "startupFile is at " << indexOfStartupFile << " and size is "
            << argument.size() << std::endl;
  if (hasCogaDir == std::string::npos ||
      indexOfStartupFile == std::string::npos)
    return;

  std::cout << "databasefarm dir is at "
            << argument.substr(0, hasCogaDir + databaseFarmDir.size())
            << std::endl;

  std::cout << "database name is at (" << hasCogaDir + databaseFarmDir.size()
            << " " << indexOfStartupFile - 1 << "):"
            << argument.substr(
                   hasCogaDir + databaseFarmDir.size(),
                   indexOfStartupFile - 1 - hasCogaDir - databaseFarmDir.size())
            << std::endl;

  VariableManager::instance().addVariable(
      "path_to_database_farm",
      VariableState(argument.substr(0, hasCogaDir + databaseFarmDir.size()),
                    VARCHAR));

  if (argument.size() <= hasCogaDir + databaseFarmDir.size() + 1) return;

  VariableManager::instance().addVariable(
      "name_of_current_databse",
      VariableState(argument.substr(hasCogaDir + databaseFarmDir.size(),
                                    indexOfStartupFile - hasCogaDir -
                                        databaseFarmDir.size() - 1),
                    VARCHAR));

  VariableManager::instance().addVariable("createPIDfile",
                                          VariableState("true", BOOLEAN));
}

void init() { initSingletons(); }
}
