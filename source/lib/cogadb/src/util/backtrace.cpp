

#include <assert.h>
#include <cxxabi.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <vector>

namespace CoGaDB {

std::string demangleFunctionName(const std::string &line) {
  int status = -1;

  std::vector<std::string> strs;
  boost::split(strs, line, boost::is_any_of("()+"));
  std::string function_name;
  std::string ret;
  if (strs.size() == 4) {
    function_name = strs[1];
    char *demangledName =
        abi::__cxa_demangle(function_name.c_str(), NULL, NULL, &status);
    if (status != 0) return function_name;
    ret = demangledName;
    ret += "+";
    ret += strs[2];
    free(demangledName);
  }
  return ret;
}

void printStackTrace(std::ostream &out) {
  const int MAXIMAL_NUMBER_OF_STACKFRAMES = 500;

  void *buffer[MAXIMAL_NUMBER_OF_STACKFRAMES];
  char **strings;

  const int number_of_stack_frames =
      backtrace(buffer, MAXIMAL_NUMBER_OF_STACKFRAMES);
  assert(number_of_stack_frames < MAXIMAL_NUMBER_OF_STACKFRAMES);

  strings = backtrace_symbols(buffer, number_of_stack_frames);
  if (strings == NULL) {
    out << "Could not produce backtrace: backtrace_symbols failed!"
        << std::endl;
    return;
  }

  for (int i = 0; i < number_of_stack_frames; ++i) {
    out << demangleFunctionName(strings[i]) << std::endl;
  }
  free(strings);
}

}  // end namespace CogaDB
