
//
//  malloc.cpp
//  CoGaDB
//
//  Created by David Broneske on 15.08.14.
//  Copyright (c) 2014 David Broneske. All rights reserved.
//

#include "hardware_optimizations/malloc.hpp"

namespace CoGaDB {
namespace CDK {

void* CDK_malloc(size_t size) {
  void* memptr;
  if (posix_memalign(&memptr, 16, size) == 0) {
    return memptr;
  } else {
    COGADB_FATAL_ERROR("Unable to allocate memory!", "CDK_malloc")
  }
}
}
}
