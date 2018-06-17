
#include <boost/function.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <core/data_dictionary.hpp>
#include <iostream>
#include <map>
#include <optimizer/optimizer.hpp>
#include <set>

#include "query_processing/pk_fk_join_operator.hpp"

#include <optimizer/join_order_optimization.hpp>

namespace CoGaDB {

namespace optimizer {

//        typedef std::pair<size_t,std::string> JoinTable;
//        typedef std::multimap<size_t,std::string> JoinTables;
//        typedef std::list<Predicate> JoinConditions;
//        typedef std::pair<std::string,Predicate> PartialJoinSpecification;
//        typedef std::pair<std::string,PartialJoinSpecification>
//        JoinSpecification;
//        typedef std::multimap<std::string,PartialJoinSpecification>
//        JoinableTables;
//
//        Predicate invertPredicate(const Predicate& pred);
//        bool isJoinable(const std::string& left_table, const std::string&
//        right_table, const Predicate& pred);

Predicate invertPredicate(const Predicate& pred) {
  Predicate p(pred);
  p.invertOrder();
  return p;
}
bool isJoinable(const std::string& left_table, const std::string& right_table,
                const Predicate& pred) {
  TablePtr left_table_ptr = getTablebyName(left_table);
  TablePtr right_table_ptr = getTablebyName(right_table);
  if (!left_table_ptr || !right_table_ptr) return false;

  // if we find the tables and in both tables the column to join, we can
  if (left_table_ptr->getColumnbyName(pred.getColumn1Name()) != NULL &&
      right_table_ptr->getColumnbyName(pred.getColumn2Name()) != NULL) {
    return true;
  }

  return false;
}

void printJoinConditions(const JoinConditions& conditions) {
  JoinConditions::const_iterator cit_cond;
  std::cout << "Join Conditions: " << std::endl;
  for (cit_cond = conditions.begin(); cit_cond != conditions.end();
       ++cit_cond) {
    std::cout << "\t" << cit_cond->toString() << std::endl;
  }
}

JoinableTables createJoinableTables(const JoinTables& join_tables,
                                    const JoinConditions& conditions) {
  JoinableTables joinable_tables;

  JoinTables::const_iterator cit_tab1;
  JoinTables::const_iterator cit_tab2;
  JoinConditions::const_iterator cit_cond;
  //            std::cout << join_tables.size() << std::endl;

  for (cit_tab1 = join_tables.begin(); cit_tab1 != join_tables.end();
       ++cit_tab1) {
    for (cit_tab2 = join_tables.begin(); cit_tab2 != join_tables.end();
         ++cit_tab2) {
      // std::cout << cit_tab1->second << "  " << cit_tab2->second << std::endl;
      if (cit_tab1->second != cit_tab2->second)
        for (cit_cond = conditions.begin(); cit_cond != conditions.end();
             ++cit_cond) {
          //                        PartialJoinSpecification(cit_tab2->second,
          //                        *cit_cond);
          //                        JoinSpecification spec(cit_tab1->second,
          //                        PartialJoinSpecification(cit_tab2->second,
          //                        *cit_cond));
          //                        joinable_tables.insert(spec);
          //                        std::cout << "Is Joinable: " <<
          //                        cit_tab1->second << " JOIN " <<
          //                        cit_tab2->second  << " USING " <<
          //                        cit_cond->toString() << std::endl;
          //                        std::cout << "Is Joinable: " <<
          //                        cit_tab1->second << " JOIN " <<
          //                        cit_tab2->second  << " USING " <<
          //                        invertPredicate(*cit_cond).toString() <<
          //                        std::endl;
          if (isJoinable(cit_tab1->second, cit_tab2->second, *cit_cond)) {
            joinable_tables.insert(JoinSpecification(
                cit_tab1->second,
                PartialJoinSpecification(cit_tab2->second, *cit_cond)));
            joinable_tables.insert(JoinSpecification(
                cit_tab2->second,
                PartialJoinSpecification(cit_tab1->second,
                                         invertPredicate(*cit_cond))));
            if (!quiet)
              std::cout << "Joinable Tables: " << cit_tab1->second << " JOIN "
                        << cit_tab2->second << " ON " << cit_cond->toString()
                        << std::endl;
            break;
          }

          //                        if(isJoinable(cit_tab1->second,cit_tab2->second,
          //                        invertPredicate(*cit_cond))){
          //                           joinable_tables.insert(JoinSpecification(cit_tab1->second,
          //                           PartialJoinSpecification(cit_tab2->second,
          //                           invertPredicate(*cit_cond))));
          //                           joinable_tables.insert(JoinSpecification(cit_tab2->second,
          //                           PartialJoinSpecification(cit_tab1->second,
          //                           *cit_cond)));
          //                           std::cout << "Joinable Tables: " <<
          //                           cit_tab1->second << " JOIN " <<
          //                           cit_tab2->second << " ON " <<
          //                           invertPredicate(*cit_cond).toString() <<
          //                           std::endl;
          //                           break;
          //                        }
        }
    }
  }

  return joinable_tables;
}

std::list<PartialJoinSpecification> createJoinPath(
    const std::string first_table, const JoinConditions& conditions,
    const JoinableTables& joinable_tables) {
  if (!quiet) printJoinConditions(conditions);

  std::list<PartialJoinSpecification> join_path;
  std::string current_table = first_table;
  std::set<std::string> tables_in_join_path;
  std::set<std::string>::iterator jp_it;
  JoinableTables::const_iterator cit;

  typedef std::pair<JoinableTables::const_iterator,
                    JoinableTables::const_iterator>
      SuccessorTables;

  tables_in_join_path.insert(current_table);
  //            while(join_path.size()+1<conditions.size()){
  while (join_path.size() < conditions.size()) {
    std::set<std::string>::iterator current_table_it;
    for (current_table_it = tables_in_join_path.begin();
         current_table_it != tables_in_join_path.end(); ++current_table_it) {
      SuccessorTables successor_tables =
          joinable_tables.equal_range(*current_table_it);
      if (!quiet)
        std::cout << "Successor for table " << *current_table_it << ": "
                  << std::endl;
      for (cit = successor_tables.first; cit != successor_tables.second;
           ++cit) {
        if (!quiet) std::cout << cit->second.first << std::endl;
        // look whether table is already in join path
        jp_it = tables_in_join_path.find(cit->second.first);
        if (jp_it == tables_in_join_path.end()) {
          // ok, add table to the join path and continue with next table
          // tables_in_join_path.insert(current_table);
          //                            *current_table_it=cit->second.first;
          tables_in_join_path.insert(cit->second.first);
          join_path.push_back(cit->second);
          break;
        }
      }
    }
  }

  return join_path;
}

std::pair<std::string, std::list<PartialJoinSpecification> > getJoinPath(
    const JoinTables& join_tables, const JoinConditions& conditions) {
  if (join_tables.empty())
    return std::pair<std::string, std::list<PartialJoinSpecification> >(
        "", std::list<PartialJoinSpecification>());
  JoinableTables joinable_tables =
      createJoinableTables(join_tables, conditions);
  if (!quiet) {
    JoinableTables::iterator it;
    for (it = joinable_tables.begin(); it != joinable_tables.end(); ++it) {
      std::cout << it->first << " JOINABLE WITH " << it->second.first
                << " ON PREDICATE " << it->second.second.toString()
                << std::endl;
    }
  }

  std::list<PartialJoinSpecification> join_path =
      createJoinPath(join_tables.begin()->second, conditions, joinable_tables);
  if (!quiet) {
    std::cout << join_tables.begin()->second << " ";
    std::list<PartialJoinSpecification>::iterator it;
    for (it = join_path.begin(); it != join_path.end(); ++it) {
      std::cout << "JOIN " << it->first << " ON " << it->second.toString()
                << " ";
    }
    std::cout << std::endl;
  }

  return std::pair<std::string, std::list<PartialJoinSpecification> >(
      join_tables.begin()->second, join_path);
}

bool joo_test() {
  JoinTables join_tables;
  join_tables.insert(JoinTable(6000000, "LINEORDER"));
  join_tables.insert(JoinTable(2556, "DATES"));
  join_tables.insert(JoinTable(40000, "SUPPLIER"));
  join_tables.insert(JoinTable(3000, "PART"));
  join_tables.insert(JoinTable(50000, "CUSTOMER"));

  //			JOIN (LO_PARTKEY=P_PARTKEY)
  //				JOIN (LO_SUPPKEY=S_SUPPKEY)
  //					JOIN (LO_CUSTKEY=C_CUSTKEY)
  //						JOIN (LO_ORDERDATE=D_DATEKEY)

  JoinConditions conditions;
  conditions.insert(Predicate(std::string("LO_PARTKEY"),
                              std::string("P_PARTKEY"), ValueValuePredicate,
                              EQUAL));
  conditions.insert(Predicate(std::string("LO_SUPPKEY"),
                              std::string("S_SUPPKEY"), ValueValuePredicate,
                              EQUAL));
  conditions.insert(Predicate(std::string("LO_CUSTKEY"),
                              std::string("C_CUSTKEY"), ValueValuePredicate,
                              EQUAL));
  conditions.insert(Predicate(std::string("LO_ORDERDATE"),
                              std::string("D_DATEKEY"), ValueValuePredicate,
                              EQUAL));

  getJoinPath(join_tables, conditions);

  return true;
}

namespace optimizer_rules {
//            bool join_order_optimization(query_processing::LogicalQueryPlanPtr
//            log_plan){
//                return false;
//            }
} /* namespace optimizer_rules */

//        bool optimizeJoinOrder(){
//            return false;
//        }

} /* namespace optimizer */

bool joo_test() { return optimizer::joo_test(); }

} /* namespace CoGaDB */
