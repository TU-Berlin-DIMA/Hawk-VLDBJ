/*
 * File:   pipeline_selectivity_estimate.cpp
 * Author: henning
 *
 * Created on 30. August 2016
 */
#include <query_compilation/pipeline_selectivity_estimates.hpp>

namespace CoGaDB {

PipelineSelectivityTable& PipelineSelectivityTable::instance() {
  static PipelineSelectivityTable vm;
  return vm;
}

void PipelineSelectivityTable::updateSelectivity(std::string table_name,
                                                 double selectivity) {
  table[table_name] = selectivity;
}

double PipelineSelectivityTable::getSelectivity(std::string table_name) {
  if (table.find(table_name) == table.end()) {
    return 1.0;
  } else {
    return table[table_name];
  }
}

void PipelineSelectivityTable::dropSelectivities() { table.clear(); }

}  // end namespace CoGaDB
