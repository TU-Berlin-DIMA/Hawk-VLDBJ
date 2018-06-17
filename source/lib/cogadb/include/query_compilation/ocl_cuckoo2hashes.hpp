/*
 * File:   ocl_cuckoo2hashes.hpp
 * Author: henning
 *
 * Created on 25. August 2016
 */

#ifndef OCL_CUCKOO2HASHES_HPP
#define OCL_CUCKOO2HASHES_HPP

#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/any.hpp>
#include <core/attribute_reference.hpp>
#include <core/global_definitions.hpp>
#include <query_compilation/hash_table_generator.hpp>
#include <sstream>
#include <string>
#include <util/code_generation.hpp>
#include <vector>

namespace CoGaDB {

  class OCLCuckoo2HashesHashTableGenerator : public HashTableGenerator {
   public:
    std::string generateCodeDeclareHashTable();

    std::string generateCodeInitHashTable(bool create_new,
                                          const std::string &num_elements = "");

    std::string generateCodeCleanupHashTable() {
      COGADB_FATAL_ERROR("IMPLEMENT", "");
    }

    std::string generateIdentifierCleanupHandler();

    std::string generateCodeInsertIntoHashTable(uint32_t loopVar = 1);

    ProbeHashTableCode generateCodeProbeHashTable(
        const AttributeReference &probe_attr, uint32_t loopVar = 1);

    std::string generateCodeHTEntryAccess();

    std::string generateCodeForHeaderAndTypesBlock();

    std::string getHashTableCType();

    std::map<std::string, std::string> getBuildExtraInputVariables();

    std::map<std::string, std::string> getProbeExtraInputVariables();

    std::map<std::string, std::string> getBuildExtraOutputVariables();

    std::string getHTVarNameForSystemHT() const;

    static void generateHashConstants();

    OCLCuckoo2HashesHashTableGenerator(
        const AttributeReference &build_attr,
        std::vector<StructFieldPtr> &payload_fields,
        const std::string &identifier);

    OCLCuckoo2HashesHashTableGenerator(const AttributeReference &build_attr,
                                       const std::string &identifier);

   private:
    std::string getStructVarName() const;
    std::string getHTVarName() const;

   public:
    static bool generated;
    // hash 1
    static uint32_t constA1;
    static uint32_t constB1;
    // hash 2
    static uint32_t constA2;
    static uint32_t constB2;
    // stash hash
    static uint32_t constA3;
    static uint32_t constB3;
    // all
    static uint32_t primeDiv;
  };

}  // end namespace CoGaDB

#endif
