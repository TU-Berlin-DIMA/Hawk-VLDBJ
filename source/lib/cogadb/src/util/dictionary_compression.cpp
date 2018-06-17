
#include <util/dictionary_compression.hpp>

namespace CoGaDB {

boost::shared_ptr<Column<char*> > createPointerArrayToValues(
    const uint32_t* compressed_values, const size_t num_elements,
    const std::string* reverse_lookup_vector) {
  boost::shared_ptr<Column<char*> > result(new Column<char*>("", VARCHAR));
  result->resize(num_elements);
  const char** array = (const char**)result->data();
  for (size_t i = 0; i < num_elements; ++i) {
    array[i] = reverse_lookup_vector[compressed_values[i]].c_str();
  }

  return result;
}

}  // end namespace CoGaDB
