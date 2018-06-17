#include "Variant.hpp"
#include "Configuration.hpp"

namespace perseus {

std::ostream& operator<<(std::ostream& ostream, const Variant& variant) {
  ostream << "Variant(";
  ostream << variant.name();
  ostream << ", ";
  ostream << (variant.configuration());
  ostream << ")";
  return ostream;
}
}