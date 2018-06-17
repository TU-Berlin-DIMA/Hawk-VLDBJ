/*
 * File:   hash_table.hpp
 * Author: sebastian
 *
 * Created on 2. August 2015, 18:11
 */

#ifndef HASH_TABLE_HPP
#define HASH_TABLE_HPP

#include <boost/shared_ptr.hpp>
#include <string>
namespace CoGaDB {

  typedef void (*HashTableCleanupFunctionPtr)(void*);

  class HashTable {
   public:
    HashTable(void* _ptr, HashTableCleanupFunctionPtr _cleanup_handler,
              const std::string& HTidentifier);
    void* ptr;
    HashTableCleanupFunctionPtr cleanup_handler;
    const std::string identifier;
    ~HashTable();
  };

  inline HashTable::HashTable(void* _ptr,
                              HashTableCleanupFunctionPtr _cleanup_handler,
                              const std::string& HTidentifier)
      : ptr(_ptr),
        cleanup_handler(_cleanup_handler),
        identifier(HTidentifier) {}

  inline HashTable::~HashTable() { (*cleanup_handler)(ptr); }

}  // end namespace CoGaDB

#endif /* HASH_TABLE_HPP */
