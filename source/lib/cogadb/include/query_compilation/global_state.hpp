/*
 * File:   global_state.hpp
 * Author: sebastian
 *
 * Created on 29. Dezember 2015, 23:37
 */

#ifndef GLOBAL_STATE_HPP
#define GLOBAL_STATE_HPP

#include <boost/shared_ptr.hpp>

namespace CoGaDB {

  class State;
  typedef boost::shared_ptr<State> StatePtr;

  class BaseTable;
  typedef boost::shared_ptr<BaseTable> TablePtr;

  class HashTable;
  typedef boost::shared_ptr<HashTable> HashTablePtr;

  const HashTablePtr getHashtable(StatePtr);

}  // end namespace CoGaDB

#endif /* GLOBAL_STATE_HPP */
