#include "utils.h"
#include <ctime>
#include <iomanip>
#include <iostream>
#include <random>

namespace perseus {

int verbosity = 3;
static std::random_device rd;
std::mt19937 global_rnd(rd());
};
