#include <cub/cub.cuh>


const int kGridSize = 16384;
const int kBuildBlockSize = 64;
const int kProbeBlockSize = 128;
//const int kReduceBlockSize = 256;
