/*
 * File:   ocl_cuckoo2hashes.cpp
 * Author: henning
 *
 * Created on 25. August 2016
 */

#include <chrono>
#include <iostream>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/hash_table_generator.hpp>
#include <query_compilation/ocl_cuckoo2hashes.hpp>
#include <query_compilation/pipeline_selectivity_estimates.hpp>
#include <random>
#include <sstream>
#include <string>
#include "core/variable_manager.hpp"
#include "schema.h"

namespace CoGaDB {

bool OCLCuckoo2HashesHashTableGenerator::generated = false;
uint32_t OCLCuckoo2HashesHashTableGenerator::constA1;
uint32_t OCLCuckoo2HashesHashTableGenerator::constB1;
uint32_t OCLCuckoo2HashesHashTableGenerator::constA2;
uint32_t OCLCuckoo2HashesHashTableGenerator::constB2;
uint32_t OCLCuckoo2HashesHashTableGenerator::constA3;
uint32_t OCLCuckoo2HashesHashTableGenerator::constB3;
uint32_t OCLCuckoo2HashesHashTableGenerator::primeDiv;

OCLCuckoo2HashesHashTableGenerator::OCLCuckoo2HashesHashTableGenerator(
    const AttributeReference &build_attr,
    std::vector<StructFieldPtr> &payload_fields, const std::string &identifier)
    : HashTableGenerator(build_attr, payload_fields, identifier) {
  if (!generated) {
    generateHashConstants();
    generated = true;
  }
}

OCLCuckoo2HashesHashTableGenerator::OCLCuckoo2HashesHashTableGenerator(
    const AttributeReference &build_attr, const std::string &identifier)
    : HashTableGenerator(build_attr, identifier) {
  if (!generated) {
    generateHashConstants();
    generated = true;
  }
}

void OCLCuckoo2HashesHashTableGenerator::generateHashConstants() {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(
      seed);  // mt19937 is a standard mersenne_twister_engine
  constA1 = generator();
  constB1 = generator();
  constA2 = generator();
  constB2 = generator();
  constA3 = generator();
  constB3 = generator();
  primeDiv = 4294967291u;
}

std::string OCLCuckoo2HashesHashTableGenerator::generateCodeDeclareHashTable() {
  std::stringstream expr;
  /* add code for hash table creation before for loop */
  expr << "OCLCuckoo2HashesHT* " << getStructVarName() << " = NULL;"
       << std::endl;

  expr << "uint64_t* " << getHTVarName() << " = NULL;" << std::endl;
  expr << "uint64_t " << getHTVarName() << "_length = 0;" << std::endl;

  return expr.str();
}

std::string OCLCuckoo2HashesHashTableGenerator::generateCodeInitHashTable(
    bool create_new, const std::string &num_elements) {
  std::stringstream expr;

  if (create_new) {
    expr << getStructVarName() << " = malloc(sizeof(OCLCuckoo2HashesHT));"
         << std::endl;

    expr << getHTVarName() << "_length = (double)(" << num_elements
         << ") * 2.5 + 101;" << std::endl
         << getStructVarName() << "->hash_table_size = " << getHTVarName()
         << "_length;" << std::endl;

    // malloc and init
    auto ht_bytesize =
        "sizeof(uint64_t) * " + getStructVarName() + "->hash_table_size";
    expr << getStructVarName() << "->hash_table = " << getHTVarName()
         << " = (uint64_t*)malloc(" << ht_bytesize << ");" << std::endl;
    expr << "memset(" << getHTVarName() << ", 0xFF, " << ht_bytesize << ");"
         << std::endl;

    return expr.str();

  } else {
    expr << getStructVarName()
         << " = (OCLCuckoo2HashesHT*)getHashTableFromSystemHashTable("
         << "generic_hashtable_" << getHashTableVarName(build_attr_) << ");"
         << std::endl;

    expr << getHTVarName() << " = " << getStructVarName() << "->hash_table;"
         << std::endl;
    expr << getHTVarName() << "_length = " << getStructVarName()
         << "->hash_table_size;" << std::endl;
  }

  return expr.str();
}

std::string
OCLCuckoo2HashesHashTableGenerator::generateIdentifierCleanupHandler() {
  return "freeOCLCuckoo2HashesHT";
}

std::string OCLCuckoo2HashesHashTableGenerator::generateCodeInsertIntoHashTable(
    uint32_t loopVar) {
  std::stringstream expr;

  expr << "hashBuildCuckoo_" << getHTVarName() << "(" << getHTVarName() << ", "
       << getHTVarName() << "_length, "
       << "(uint32_t)" << getElementAccessExpression(build_attr_) << ", "
       << "(uint32_t)write_pos);" << std::endl;

  return expr.str();
}

ProbeHashTableCode
OCLCuckoo2HashesHashTableGenerator::generateCodeProbeHashTable(
    const AttributeReference &probe_attr, uint32_t loopVar) {
  /* code for hash table probes */
  std::stringstream hash_probe;
  /* closing brackets and pointer chasing */
  std::stringstream hash_probe_lower;

  hash_probe << "uint32_t " << getHTVarName() << "_payload = "
             << "hashProbeCuckoo_" << getHTVarName() << "(" << getHTVarName()
             << ", " << getHTVarName() << "_length, "
             << "(uint32_t)" << getElementAccessExpression(probe_attr) << ");"
             << std::endl
             << "if(" << getHTVarName() << "_payload != 0xffffffff) {"
             << std::endl;

  hash_probe << getTupleIDVarName(build_attr_) << " = " << getHTVarName()
             << "_payload;" << std::endl;

  hash_probe_lower << "}" << std::endl;

  return ProbeHashTableCode(hash_probe.str(), hash_probe_lower.str());
}

std::string OCLCuckoo2HashesHashTableGenerator::generateCodeHTEntryAccess() {
  COGADB_NOT_IMPLEMENTED;
}

std::string OCLCuckoo2HashesHashTableGenerator::getHashTableCType() {
  return "uint64_t*";
}

std::map<std::string, std::string>
OCLCuckoo2HashesHashTableGenerator::getBuildExtraInputVariables() {
  return {{getHTVarName() + "_length", "uint64_t"}};
}

std::map<std::string, std::string>
OCLCuckoo2HashesHashTableGenerator::getBuildExtraOutputVariables() {
  return {};
}

std::map<std::string, std::string>
OCLCuckoo2HashesHashTableGenerator::getProbeExtraInputVariables() {
  return {{getHTVarName() + "_length", "uint64_t"}};
}

std::string
OCLCuckoo2HashesHashTableGenerator::generateCodeForHeaderAndTypesBlock() {
  std::stringstream expr;

  expr << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable"
       << std::endl;

  expr << "uint32_t h1_" << getHTVarName() << "(uint32_t key) {" << std::endl
       << "  uint32_t v = ((" << constA1 << "u ^ key) + " << constB1 << "u) % "
       << primeDiv << "u;" << std::endl
       << "  return v;" << std::endl
       << "}" << std::endl;

  expr << "uint32_t h2_" << getHTVarName() << "(uint32_t key) {" << std::endl
       << "  uint32_t v = ((" << constA2 << "u ^ key) + " << constB2 << "u) % "
       << primeDiv << "u;" << std::endl
       << "  return v;" << std::endl
       << "}" << std::endl;

  expr << "uint32_t h_stash_" << getHTVarName() << "(uint32_t key) {"
       << std::endl
       << "  uint32_t v = ((" << constA3 << "u ^ key) + " << constB3 << "u) %"
       << primeDiv << "u;" << std::endl
       << "  return v;" << std::endl
       << "}" << std::endl;

  expr << "uint32_t hashProbeCuckoo_" << getHTVarName()
       << "(__global uint64_t* hash_table, uint64_t hash_table_size, uint32_t "
          "key) {"
       << std::endl
       << "  uint32_t loc1,loc2;" << std::endl
       << "  uint64_t ht_size = hash_table_size - 101;" << std::endl
       << "  loc1 = h1_" << getHTVarName() << "(key) % ht_size;" << std::endl
       << "  loc2 = h2_" << getHTVarName() << "(key) % ht_size;" << std::endl
       << "  uint64_t entry = hash_table[loc1];" << std::endl
       << "  uint32_t probeKey = (uint32_t) entry;" << std::endl
       << "  if(probeKey != key) {" << std::endl
       << "    entry = hash_table[loc2];" << std::endl
       << "    probeKey = (uint32_t) entry;" << std::endl
       << "  }" << std::endl
       << "  if(probeKey != key) {" << std::endl
       << "    uint32_t slot = h_stash_" << getHTVarName() << "(key) % 101;"
       << std::endl
       << "    entry = hash_table[ht_size + slot];" << std::endl
       << "    probeKey = (uint32_t) entry;" << std::endl
       << "  }" << std::endl
       << "  if(probeKey == key) {" << std::endl
       << "    return (uint32_t)(entry >> 32);" << std::endl
       << "  } else {" << std::endl
       << "    return 0xffffffff;" << std::endl
       << "  }" << std::endl
       << "}" << std::endl;

  expr << "uint64_t make_entry_" << getHTVarName()
       << "(uint32_t key, uint32_t value) {" << std::endl
       << "  uint64_t entry = (((uint64_t)value) << 32) + key;" << std::endl
       << "  return entry;" << std::endl
       << "}" << std::endl;

  expr << "uint32_t determine_next_location_" << getHTVarName()
       << "(uint64_t hash_table, uint64_t ht_size, uint32_t key, uint32_t "
          "location) {"
       << std::endl
       << "  uint32_t loc1 = h1_" << getHTVarName() << "(key) % ht_size;"
       << std::endl
       << "  uint32_t loc2 = h2_" << getHTVarName() << "(key) % ht_size;"
       << std::endl
       << "  return location == loc1 ? loc2 : loc1;" << std::endl
       << "}" << std::endl;

  expr << "void hashBuildCuckoo_" << getHTVarName() << "("
       << "__global uint64_t* hash_table, uint64_t hash_table_size, uint32_t "
          "key, uint32_t value) {"
       << std::endl
       << "  uint64_t ht_size = hash_table_size - 101;" << std::endl
       << "  uint64_t entry = make_entry_" << getHTVarName() << "(key, value);"
       << std::endl
       << "  uint32_t location = h1_" << getHTVarName() << "(key) % ht_size;"
       << std::endl
       << "  uint32_t replaced_key;" << std::endl
       << "  if(hashProbeCuckoo_" << getHTVarName()
       << "(hash_table, hash_table_size, key) != 0xffffffff) return;"
       << std::endl
       << "  for(int its = 1; its <= 100; its++) {" << std::endl
       << "    entry = atom_xchg(&hash_table[location], entry);" << std::endl
       << "    replaced_key = (uint32_t) entry;" << std::endl
       << "    if(replaced_key == 0xffffffff) break;" << std::endl
       << "    if(replaced_key == key) return;" << std::endl
       << "    location = determine_next_location_" << getHTVarName()
       << "(hash_table, ht_size, replaced_key, location);" << std::endl
       << "  }" << std::endl
       << "  if(replaced_key != 0xffffffff) {" << std::endl
       << "    uint32_t slot = h_stash_" << getHTVarName() << "(key) % 101;"
       << std::endl
       << "    uint64_t replaced_entry = atom_cmpxchg(&hash_table[ht_size + "
          "slot], 0xffffffffffffffff, entry);"
       << std::endl
       << "    replaced_key = (uint32_t) replaced_entry;" << std::endl
       << "    if(replaced_key != 0xffffffff) {" << std::endl
       << "      printf(\"hash build failure: stash entry occupied\\n\");"
       << std::endl
       << "    }" << std::endl
       << "  }" << std::endl
       << "}" << std::endl;

  return expr.str();
}

std::string OCLCuckoo2HashesHashTableGenerator::getStructVarName() const {
  return getHTVarName() + "_struct";
}

std::string OCLCuckoo2HashesHashTableGenerator::getHTVarName() const {
  return getHashTableVarName(build_attr_);
}

std::string OCLCuckoo2HashesHashTableGenerator::getHTVarNameForSystemHT()
    const {
  return getStructVarName();
}

}  // end namespace CoGaDB
