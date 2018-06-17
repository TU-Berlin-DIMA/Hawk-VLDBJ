/*
 * File:   ocl_lin_genhash.cpp
 * Author: henning
 *
 * Created on 26. August 2016
 */
#include <chrono>
#include <iostream>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/hash_table_generator.hpp>
#include <query_compilation/ocl_lin_genhash.hpp>
#include <query_compilation/pipeline_selectivity_estimates.hpp>
#include <random>
#include <sstream>
#include <string>
#include "core/variable_manager.hpp"
#include "schema.h"

namespace CoGaDB {

bool OCLSeededLinearProbingHashTableGenerator::generated = false;
uint32_t OCLSeededLinearProbingHashTableGenerator::constA1;
uint32_t OCLSeededLinearProbingHashTableGenerator::constB1;
uint32_t OCLSeededLinearProbingHashTableGenerator::constA2;
uint32_t OCLSeededLinearProbingHashTableGenerator::constB2;
uint32_t OCLSeededLinearProbingHashTableGenerator::constA3;
uint32_t OCLSeededLinearProbingHashTableGenerator::constB3;
uint32_t OCLSeededLinearProbingHashTableGenerator::primeDiv;

OCLSeededLinearProbingHashTableGenerator::
    OCLSeededLinearProbingHashTableGenerator(
        const AttributeReference &build_attr,
        std::vector<StructFieldPtr> &payload_fields,
        const std::string &identifier)
    : HashTableGenerator(build_attr, payload_fields, identifier) {
  if (!generated) {
    generateHashConstants();
    generated = true;
  }
}

OCLSeededLinearProbingHashTableGenerator::
    OCLSeededLinearProbingHashTableGenerator(
        const AttributeReference &build_attr, const std::string &identifier)
    : HashTableGenerator(build_attr, identifier) {
  if (!generated) {
    generateHashConstants();
    generated = true;
  }
}

void OCLSeededLinearProbingHashTableGenerator::generateHashConstants() {
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

  bool use_tested_constants =
      VariableManager::instance().getVariableValueBoolean(
          "code_gen.use_tested_hash_constants");
  if (use_tested_constants) {
    constA1 = 2578908015u;  // 3837550874u;
    constB1 = 3672160021u;  // 1983097511u;
  }
}

std::string
OCLSeededLinearProbingHashTableGenerator::generateCodeDeclareHashTable() {
  std::stringstream expr;
  /* add code for hash table creation before for loop */
  expr << "OCLCuckoo2HashesHT* " << getStructVarName() << " = NULL;"
       << std::endl;

  expr << "uint64_t* " << getHTVarName() << " = NULL;" << std::endl;
  expr << "uint64_t " << getHTVarName() << "_length = 0;" << std::endl;

  return expr.str();
}

std::string OCLSeededLinearProbingHashTableGenerator::generateCodeInitHashTable(
    bool create_new, const std::string &num_elements) {
  std::stringstream expr;

  if (create_new) {
    expr << getStructVarName() << " = malloc(sizeof(OCLCuckoo2HashesHT));"
         << std::endl;
    expr << "uint64_t length_min = (double)(" << num_elements << ") * 10.0;"
         << std::endl;
    expr << getHTVarName() << "_length = 1;" << std::endl;
    expr << "while(" << getHTVarName() << "_length < length_min) {" << std::endl
         << "  " << getHTVarName() << "_length *= 2;" << std::endl
         << "}" << std::endl;
    expr << getHTVarName() << "_length += 101;" << std::endl;
    expr << getStructVarName() << "->hash_table_size = " << getHTVarName()
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
OCLSeededLinearProbingHashTableGenerator::generateIdentifierCleanupHandler() {
  return "freeOCLCuckoo2HashesHT";
}

std::string
OCLSeededLinearProbingHashTableGenerator::generateCodeInsertIntoHashTable(
    uint32_t loopVar) {
  std::stringstream expr;

  expr << "hashBuildLinearProbing_" << getHTVarName() << "(" << getHTVarName()
       << ", " << getHTVarName() << "_length, "
       << "(uint32_t)" << getElementAccessExpression(build_attr_) << ", "
       << "(uint32_t)write_pos);" << std::endl;

  return expr.str();
}

ProbeHashTableCode
OCLSeededLinearProbingHashTableGenerator::generateCodeProbeHashTable(
    const AttributeReference &probe_attr, uint32_t loopVar) {
  /* code for hash table probes */
  std::stringstream hash_probe;
  /* closing brackets and pointer chasing */
  std::stringstream hash_probe_lower;

  hash_probe << "uint32_t " << getHTVarName() << "_payload = "
             << "hashProbeLinearProbing_" << getHTVarName() << "("
             << getHTVarName() << ", " << getHTVarName() << "_length, "
             << "(uint32_t)" << getElementAccessExpression(probe_attr) << ");"
             << std::endl
             << "if(" << getHTVarName() << "_payload != 0xffffffff) {"
             << std::endl;

  hash_probe << getTupleIDVarName(build_attr_) << " = " << getHTVarName()
             << "_payload;" << std::endl;

  hash_probe_lower << "}" << std::endl;

  return ProbeHashTableCode(hash_probe.str(), hash_probe_lower.str());
}

std::string
OCLSeededLinearProbingHashTableGenerator::generateCodeHTEntryAccess() {
  COGADB_NOT_IMPLEMENTED;
}

std::string OCLSeededLinearProbingHashTableGenerator::getHashTableCType() {
  return "uint64_t*";
}

std::map<std::string, std::string>
OCLSeededLinearProbingHashTableGenerator::getBuildExtraInputVariables() {
  return {{getHTVarName() + "_length", "uint64_t"}};
}

std::map<std::string, std::string>
OCLSeededLinearProbingHashTableGenerator::getBuildExtraOutputVariables() {
  return {};
}

std::map<std::string, std::string>
OCLSeededLinearProbingHashTableGenerator::getProbeExtraInputVariables() {
  return {{getHTVarName() + "_length", "uint64_t"}};
}

std::string
OCLSeededLinearProbingHashTableGenerator::generateCodeForHeaderAndTypesBlock() {
  std::stringstream expr;

  expr << "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable"
       << std::endl;

  expr << "uint32_t h1_" << getHTVarName() << "(uint32_t key) {" << std::endl
       << "  uint32_t v = ((" << constA1 << "u ^ key) + " << constB1 << "u) % "
       << primeDiv << "u;" << std::endl
       << "  return v;" << std::endl
       << "}" << std::endl;

  expr << "uint32_t hashProbeLinearProbing_" << getHTVarName()
       << "(__global uint64_t* hash_table, uint64_t hash_table_size, uint32_t "
          "key) {"
       << std::endl
       << "  uint64_t ht_size = hash_table_size - 101;" << std::endl
       << "  uint32_t location = h1_" << getHTVarName() << "(key) % ht_size;"
       << std::endl
       << "  uint64_t probeEntry;" << std::endl
       << "  uint32_t probeKey;" << std::endl
       << "  for (int its = 1; its <= 100000000; its++) {" << std::endl
       << "    probeEntry = hash_table[location];" << std::endl
       << "    probeKey = (uint32_t)probeEntry;" << std::endl
       << "    if(probeKey == key) {" << std::endl
       << "      return (uint32_t)(probeEntry >> 32);" << std::endl
       << "    }" << std::endl
       << "    if(probeKey == 0xffffffff) {" << std::endl
       << "      return 0xffffffff;" << std::endl
       << "    }" << std::endl
       << "    location = (location + 1) % ht_size;" << std::endl
       << "  }" << std::endl
       << "  return 0xffffffff;" << std::endl
       << "}" << std::endl;

  expr << "uint64_t make_entry_" << getHTVarName()
       << "(uint32_t key, uint32_t value) {" << std::endl
       << "  uint64_t entry = (((uint64_t)value) << 32) + key;" << std::endl
       << "  return entry;" << std::endl
       << "}" << std::endl;

  expr << "uint32_t hashBuildLinearProbing_" << getHTVarName() << "("
       << "__global uint64_t* hash_table, uint64_t hash_table_size, uint32_t "
          "key, uint32_t value) {"
       << std::endl
       << "  uint64_t ht_size = hash_table_size - 101;" << std::endl
       << "  uint64_t entry = make_entry_" << getHTVarName() << "(key, value);"
       << std::endl
       << "  uint32_t org_location = h1_" << getHTVarName()
       << "(key) % ht_size;" << std::endl
       << "  uint32_t location = org_location;" << std::endl
       << "  uint64_t replaced_entry = 0;" << std::endl
       << "  for(int its=1; its <= 100000000; its++) {" << std::endl
       << "    replaced_entry = atom_cmpxchg(&hash_table[location], "
          "0xffffffffffffffff, entry);"
       << std::endl
       << "    if(replaced_entry == 0xffffffffffffffff) break;"
       << "    location = (location + 1) % ht_size;" << std::endl
       << "    if(location == org_location) {" << std::endl
       << "      return 0xffffffff;" << std::endl
       //<< "      printf(\"hash build failure: hash table full\\n\");" <<
       // std::endl
       << "    }" << std::endl
       << "  }" << std::endl
       << "}" << std::endl;

  return expr.str();
}

std::string OCLSeededLinearProbingHashTableGenerator::getStructVarName() const {
  return getHTVarName() + "_struct";
}

std::string OCLSeededLinearProbingHashTableGenerator::getHTVarName() const {
  return getHashTableVarName(build_attr_);
}

std::string OCLSeededLinearProbingHashTableGenerator::getHTVarNameForSystemHT()
    const {
  return getStructVarName();
}

}  // end namespace CoGaDB
