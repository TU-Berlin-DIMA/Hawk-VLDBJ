
#include <core/variable_manager.hpp>
#include <util/shared_library.hpp>

#include <assert.h>
#include <dlfcn.h>
#include <iostream>

#include <boost/smart_ptr.hpp>

namespace CoGaDB {

SharedLibrary::SharedLibrary(void* _shared_lib) : shared_lib_(_shared_lib) {
  assert(shared_lib_ != NULL);
}

SharedLibrary::~SharedLibrary() {
  if (!VariableManager::instance().getVariableValueBoolean(
          "profiling.keep_shared_libraries_loaded")) {
    dlclose(shared_lib_);
  }
}

void* SharedLibrary::getSymbol(const std::string& mangeled_symbol_name) const {
  auto symbol = dlsym(shared_lib_, mangeled_symbol_name.c_str());
  auto error = dlerror();

  if (error) {
    COGADB_FATAL_ERROR("Could not load symbol: " << mangeled_symbol_name
                                                 << std::endl
                                                 << "Error:" << std::endl
                                                 << error,
                       "");
  }

  return symbol;
}

SharedLibraryPtr SharedLibrary::load(const std::string& file_path) {
  auto myso = dlopen(file_path.c_str(), RTLD_NOW);

  auto error = dlerror();
  if (error) {
    COGADB_FATAL_ERROR(
        "Could not load shared library: " << file_path << std::endl
                                          << "Error:" << std::endl
                                          << error,
        "");
  } else if (!myso) {
    COGADB_FATAL_ERROR("Could not load shared library: " << file_path
                                                         << std::endl
                                                         << "Error unknown!",
                       "");
  }

  return SharedLibraryPtr(new SharedLibrary(myso));
}

}  // end namespace CoGaDB
