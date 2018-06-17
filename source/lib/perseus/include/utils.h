#ifndef PERSEUS_UTILS_H
#define PERSEUS_UTILS_H

#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#define BOOST_LOG_DYN_LINK
#undef rethrow
#include <boost/log/trivial.hpp>

// from http://stackoverflow.com/a/26221725/2560133
template <typename... Args>
std::string string_format(const std::string& format, Args... args) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
  size_t size = snprintf(nullptr, 0, format.c_str(), args...) +
                1;  // Extra space for '\0'
  std::unique_ptr<char[]> buf(new char[size]);
  snprintf(buf.get(), size, format.c_str(), args...);
#pragma GCC diagnostic pop
  return std::string(buf.get(),
                     buf.get() + size - 1);  // We don't want the '\0' inside
}

// logging
#define PERSEUS_TIMING if (perseus::verbosity >= 2)
#define PERSEUS_TRACE if (perseus::verbosity >= 3)

namespace perseus {

  extern int verbosity;

  extern std::mt19937 global_rnd;

  template <typename T, typename S>
  std::vector<T*> convertUniquePtrElementsToTypedRawPointers(
      const std::vector<std::unique_ptr<S>>& source) {
    std::vector<T*> result;
    result.reserve(source.size());
    for (auto& element : source) {
      result.push_back(element.get());
    }
    return result;
  }

  template <typename T>
  std::vector<T*> convertUniquePtrElementsToRawPointers(
      const std::vector<std::unique_ptr<T>>& source) {
    return convertUniquePtrElementsToTypedRawPointers<T>(source);
  }

  template <typename T>
  std::string join(const std::vector<T>& v,
                   const std::string& separator = " ") {
    std::stringstream ss;
    for (auto i = 0u; i < v.size(); ++i) {
      if (i != 0) {
        ss << separator;
      }
      ss << v[i];
    }
    auto s = ss.str();
    return s;
  }

  template <typename K, typename V>
  std::vector<K> keys(std::unordered_map<K, V>& m) {
    std::vector<K> keys;
    for (auto& e : m) {
      keys.push_back(e.first);
    }
    return keys;
  }
}

#endif  // PERSEUS_UTILS_H
