/*
 * File:   pipeline_selectivity_estimates.hpp
 * Author: henning
 *
 * Created on 30. August 2016
 */
#ifndef PIPELINE_SELECTIVITY_ESTIMATES_HPP
#define PIPELINE_SELECTIVITY_ESTIMATES_HPP

#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/any.hpp>
#include <core/global_definitions.hpp>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace CoGaDB {

  class PipelineSelectivityTable {
   public:
    static PipelineSelectivityTable& instance();
    void updateSelectivity(std::string table_name, double selectivity);
    double getSelectivity(std::string table_name);
    void dropSelectivities();

   private:
    std::map<std::string, double> s_table;
    PipelineSelectivityTable(){};
    PipelineSelectivityTable(const PipelineSelectivityTable&);
    PipelineSelectivityTable& operator=(const PipelineSelectivityTable&);
    std::map<std::string, double> table;
  };

}  // end namespace CoGaDB

#endif
