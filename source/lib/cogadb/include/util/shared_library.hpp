/*
 * File:   shared_library.hpp
 * Author: sebastian
 *
 * Created on 21. August 2015, 16:52
 */

#ifndef SHARED_LIBRARY_HPP
#define SHARED_LIBRARY_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

#include <boost/shared_ptr.hpp>
#include <string>

namespace CoGaDB {

  class SharedLibrary;
  typedef boost::shared_ptr<SharedLibrary> SharedLibraryPtr;

  class SharedLibrary {
   public:
    ~SharedLibrary();

    static SharedLibraryPtr load(const std::string& file_path);
    void* getSymbol(const std::string& mangeled_symbol_name) const;

    template <typename Function>
    Function getFunction(const std::string& mangeled_symbol_name) const {
      return reinterpret_cast<Function>(getSymbol(mangeled_symbol_name));
    }

   private:
    SharedLibrary(void* shared_lib);
    void* shared_lib_;
  };

}  // end namespace CoGaDB

#pragma GCC diagnostic pop

#endif /* SHARED_LIBRARY_HPP */
