/*
 * File:   join_order_optimization.hpp
 * Author: sebastian
 *
 * Created on 26. Oktober 2014, 09:02
 */

#ifndef JOIN_ORDER_OPTIMIZATION_HPP
#define JOIN_ORDER_OPTIMIZATION_HPP

//#include <optimizer/optimizer.hpp>
#include <boost/function.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <core/data_dictionary.hpp>
#include <iostream>
#include <map>
#include <set>

#include "query_processing/pk_fk_join_operator.hpp"

namespace CoGaDB {

  namespace optimizer {

    typedef std::pair<size_t, std::string> JoinTable;
    typedef std::multimap<size_t, std::string> JoinTables;
    typedef std::set<Predicate> JoinConditions;
    typedef std::pair<std::string, Predicate> PartialJoinSpecification;
    typedef std::pair<std::string, PartialJoinSpecification> JoinSpecification;
    typedef std::multimap<std::string, PartialJoinSpecification> JoinableTables;

    typedef std::pair<std::string, std::list<PartialJoinSpecification> >
        JoinPath;

    Predicate invertPredicate(const Predicate& pred);
    bool isJoinable(const std::string& left_table,
                    const std::string& right_table, const Predicate& pred);
    void printJoinConditions(const JoinConditions& conditions);

    std::list<PartialJoinSpecification> createJoinPath(
        const std::string first_table, const JoinConditions& conditions,
        const JoinableTables& joinable_tables);

    std::pair<std::string, std::list<PartialJoinSpecification> > getJoinPath(
        const JoinTables& join_tables, const JoinConditions& conditions);

  } /* namespace optimizer */
} /* namespace CoGaDB */

#endif /* JOIN_ORDER_OPTIMIZATION_HPP */
