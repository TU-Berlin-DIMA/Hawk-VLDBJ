/*
 * File:   dictionary_compression.hpp
 * Author: sebastian
 *
 * Created on 20. Dezember 2014, 20:30
 */

#ifndef DICTIONARY_COMPRESSION_HPP
#define DICTIONARY_COMPRESSION_HPP

#include <string>
#include <utility>

#include <boost/shared_ptr.hpp>
#include <core/column.hpp>

namespace CoGaDB {

  boost::shared_ptr<Column<char*> > createPointerArrayToValues(
      const uint32_t* compressed_values, const size_t num_elements,
      const std::string* reverse_lookup_vector);

  //    boost::shared_ptr<Column<char*> > createPointerArrayToValues(const
  //    uint32_t* compressed_values,
  //            const size_t num_elements,
  //            const std::string* reverse_lookup_vector){
  //
  //        boost::shared_ptr<Column<char*> > result(new
  //        Column<char*>("",VARCHAR));
  //        result->resize(num_elements);
  //        char** array = result->data();
  //        for(size_t i=0;i<num_elements;++i){
  //            array[i]=reverse_lookup_vector[compressed_values[i]].c_str();
  //        }
  //
  //        return result;
  //    }

}  // end namespace CoGaDB

#endif /* DICTIONARY_COMPRESSION_HPP */
