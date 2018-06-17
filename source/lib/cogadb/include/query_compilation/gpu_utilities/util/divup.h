#pragma once

template <typename T>
inline T divUp(T dividend, T divisor) {
  return ((dividend + divisor - 1) / divisor);
}
