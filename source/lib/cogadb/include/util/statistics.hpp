/*
 * File:   statistics.hpp
 * Author: sebastian
 *
 * Created on 30. Mai 2015, 12:00
 */

#ifndef STATISTICS_HPP
#define STATISTICS_HPP

#include <core/table.hpp>
#include <parser/client.hpp>

namespace CoGaDB {

  bool computeStatisticsOnTable(const std::string& table_name,
                                ClientPtr client);

}  // end namespace CoGaDB

#endif /* STATISTICS_HPP */
