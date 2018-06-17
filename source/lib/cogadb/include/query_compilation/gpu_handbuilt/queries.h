/*
 * File:   tpch.h
 * Author: henning
 *
 * Created on 29. Juli 2015, 10:43
 */

#ifndef TPCH_H
#define TPCH_H

#include <query_compilation/minimal_api.hpp>

#define CUDA_CHECK_ERROR_RETURN(errorMessage)                             \
  {                                                                       \
    cudaError_t err = cudaGetLastError();                                 \
    if (cudaSuccess != err) {                                             \
      fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",   \
              errorMessage, __FILE__, __LINE__, cudaGetErrorString(err)); \
      CoGaDB::exit(EXIT_FAILURE);                                         \
    }                                                                     \
  }

class ClientPtr;

namespace CoGaDB {

#define HOST_MEMORY hype::PD_Memory_0
#define DEVICE_MEMORY hype::PD_Memory_1

  bool tpch6_hand_compiled(ClientPtr client);

  bool tpch6_hand_compiled_kernel(ClientPtr client);
  bool tpch6_hand_compiled_holistic_kernel(ClientPtr client);

  // bool ssb4_hand_compiled_holistic_kernel(ClientPtr client);

  // bool tpch4_holistic_kernel(ClientPtr client);

  bool tpch3_holistic_kernel(ClientPtr client);

  // bool tpch5_holistic_kernel(ClientPtr client);
}

#endif /* TPCH_H */
