/*
 * File:   result_output_format.hpp
 * Author: sebastian
 *
 * Created on 5. März 2015, 15:06
 */

#ifndef RESULT_OUTPUT_FORMAT_HPP
#define RESULT_OUTPUT_FORMAT_HPP

#include <core/base_column.hpp>

namespace CoGaDB {

  class ResultFormat {
   public:
    ResultFormat() {}
    virtual ~ResultFormat() {}

    virtual const std::string getHeader(
        const std::string& table_name, const std::vector<ColumnPtr>& columns,
        const std::vector<unsigned int>& max_column_widths) = 0;
    virtual const std::string getRows(
        const std::vector<std::vector<std::string> >& string_columns,
        const std::vector<unsigned int>& max_column_widths) = 0;
    virtual const std::string getFooter(
        const std::vector<ColumnPtr>& columns,
        const std::vector<unsigned int>& max_column_widths) = 0;

    const std::string getResult(const std::string& table_name,
                                const std::vector<ColumnPtr>& columns,
                                bool include_header);
  };

  typedef boost::shared_ptr<ResultFormat> ResultFormatPtr;

  ResultFormatPtr getResultFormatter(const std::string& formatter_name);

}  // end namespace CoGaDB

#endif /* RESULT_OUTPUT_FORMAT_HPP */
