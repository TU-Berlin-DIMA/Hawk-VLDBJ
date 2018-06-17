#pragma once

#include <cuda_runtime_api.h>

dim3 computeGridDim(unsigned int n, unsigned int gridSize, unsigned blockSize) {
  // Round up in order to make sure all items are hashed in.
  dim3 grid((n + blockSize - 1) / blockSize);
  if (grid.x > gridSize) {
    grid.y = (grid.x + gridSize - 1) / gridSize;
    grid.x = gridSize;
  }
  if (grid.x == 0) grid.x = 1;
  return grid;
}
