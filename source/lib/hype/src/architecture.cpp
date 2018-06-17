
#include <util/architecture.hpp>

namespace hype {
namespace util {

Architecture getArchitecture() {
#ifdef __LP64__
  // 64-bit Intel or PPC
  //#warning "Compiling for 64 Bit"
  return Architecture_64Bit;
#else
  // 32-bit Intel, PPC or ARM
  //#warning "Compiling for 32 Bit"
  return Architecture_32Bit;
#endif
}

}  // end namespace util
}  // end namespace hype
