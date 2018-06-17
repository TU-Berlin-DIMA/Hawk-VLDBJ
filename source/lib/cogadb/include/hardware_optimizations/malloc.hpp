/*
 * File:   malloc.hpp
 * Author: dave
 *
 * Created on February 24, 2015, 6:17 AM
 */

#ifndef MALLOC_HPP
#define MALLOC_HPP

#include <core/global_definitions.hpp>

// allocates 4 byte-aligned memory
//#define malloc(size) CDK_malloc(size)

namespace CoGaDB {
  namespace CDK {

    void* CDK_malloc(size_t size);
  }
}

#endif /* MALLOC_HPP */
