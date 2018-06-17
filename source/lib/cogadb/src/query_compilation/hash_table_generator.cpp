#include <iostream>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/hash_table_generator.hpp>
#include <query_compilation/ocl_cuckoo2hashes.hpp>
#include <query_compilation/ocl_lin_genhash.hpp>
#include <sstream>
#include <string>
#include "core/variable_manager.hpp"
#include "schema.h"

namespace CoGaDB {

const HashTableGeneratorPtr createHashTableGenerator(
    const std::string& identifier, const AttributeReference& build_attr) {
  std::vector<std::string> strs;
  boost::split(strs, identifier, boost::is_any_of(";"));
  const std::string ht_type = strs[0];
  if (ht_type == "bucketchained") {
    return HashTableGeneratorPtr(
        new BucketChainedHashTableGenerator(build_attr, identifier));
  } else if (ht_type == "cuckoo") {
    return HashTableGeneratorPtr(new CuckooHashTableGenerator(
        build_attr, static_cast<uint8_t>(atoi(strs[1].c_str())),
        static_cast<uint32_t>(atoi(strs[2].c_str())), identifier));
  } else if (ht_type == "linear_probing") {
    return HashTableGeneratorPtr(
        new LinearProbeHashTableGenerator(build_attr, identifier));
  } else if (ht_type == "ocl_hash_table") {
    return HashTableGeneratorPtr(
        new OCLHashTableGenerator(build_attr, identifier));
  } else if (ht_type == "ocl_cuckoo") {
    return HashTableGeneratorPtr(
        new OCLCuckooHashTableGenerator(build_attr, identifier));
  } else if (ht_type == "ocl_linear_probing") {
    return HashTableGeneratorPtr(
        new OCLLinearProbingHashTableGenerator(build_attr, identifier));
  } else if (ht_type == "ocl_cuckoo2hashes") {
    return HashTableGeneratorPtr(
        new OCLCuckoo2HashesHashTableGenerator(build_attr, identifier));
  } else if (ht_type == "ocl_seeded_linear_probing") {
    return HashTableGeneratorPtr(
        new OCLSeededLinearProbingHashTableGenerator(build_attr, identifier));
  } else {
    COGADB_FATAL_ERROR("Hash Table Type not found!", "");
  }
}

const HashTableGeneratorPtr createHashTableGenerator(
    const AttributeReference& build_attr) {
  if (VariableManager::instance().getVariableValueString(
          "default_hash_table") == "cuckoo") {
    std::stringstream ss;
    ss << "cuckoo;"
       << VariableManager::instance().getVariableValueInteger(
              "default_num_hash_tables")
       << ";"
       << VariableManager::instance().getVariableValueInteger(
              "default_cuckoo_seed");
    const std::string id = ss.str();
    return createHashTableGenerator(id, build_attr);
  } else {
    return createHashTableGenerator(
        VariableManager::instance().getVariableValueString(
            "default_hash_table"),
        build_attr);
  }
}

std::string HashTableGenerator::getHTVarNameForSystemHT() const {
  return getHashTableVarName(build_attr_);
}

std::string BucketChainedHashTableGenerator::generateCodeDeclareHashTable() {
  std::stringstream expr;

  /* add code for hash table creation before for loop */
  expr << "hashtable_t* " << getHashTableVarName(build_attr_) << " = NULL;"
       << std::endl;

  return expr.str();
}

std::string BucketChainedHashTableGenerator::generateCodeInsertIntoHashTable(
    uint32_t loopVar) {
  std::stringstream hash_table;
  /* insert values into hash table */
  hash_table << "tuple_t t_" << getHashTableVarName(build_attr_) << " = {"
             << getInputArrayVarName(build_attr_) << "["
             << getTupleIDVarName(build_attr_) << "], "
             << "current_result_size};" << std::endl;
  hash_table << "hash_put(" << getHashTableVarName(build_attr_);
  hash_table << ", t_" << getHashTableVarName(build_attr_) << ");" << std::endl;

  return hash_table.str();
}

std::string BucketChainedHashTableGenerator::generateCodeInitHashTable(
    bool create_new, const std::string& num_elements) {
  std::stringstream expr;
  if (create_new) {
    expr << getHashTableVarName(build_attr_) << "=hash_new (" << num_elements
         << ");" << std::endl;
  } else {
    expr
        << getHashTableVarName(build_attr_)
        << "= (hashtable_t*) getHashTableFromSystemHashTable(generic_hashtable_"
        << getHashTableVarName(build_attr_) << ");" << std::endl;
  }

  return expr.str();
}

std::string
BucketChainedHashTableGenerator::generateIdentifierCleanupHandler() {
  return "hash_release";
}

std::string BucketChainedHashTableGenerator::getHashTableCType() {
  return "hashtable_t*";
}

std::map<std::string, std::string>
BucketChainedHashTableGenerator::getBuildExtraInputVariables() {
  return {};
}

std::map<std::string, std::string>
BucketChainedHashTableGenerator::getProbeExtraInputVariables() {
  return {};
}

ProbeHashTableCode BucketChainedHashTableGenerator::generateCodeProbeHashTable(
    const AttributeReference& probe_attr, uint32_t loopVar) {
  /* code for hash table probes */
  std::stringstream hash_probe;
  /* closing brackets and pointer chasing */
  std::stringstream hash_probe_lower;
  hash_probe << "unsigned long hash_" << getHashTableVarName(build_attr_)
             << " = HASH("
             << getElementAccessExpression(
                    probe_attr,
                    getTupleIDVarName(probe_attr.getTable(), loopVar))
             << ") & " << getHashTableVarName(build_attr_) << "->mask;"
             << std::endl;
  hash_probe << "hash_bucket_t *bucket_" << getHashTableVarName(build_attr_)
             << " = &" << getHashTableVarName(build_attr_) << "->buckets[hash_"
             << getHashTableVarName(build_attr_) << "];" << std::endl;

  hash_probe << "while (bucket_" << getHashTableVarName(build_attr_) << ") {"
             << std::endl;
  hash_probe << "  for (size_t bucket_tid_" << getHashTableVarName(build_attr_)
             << " = 0; "
             << "bucket_tid_" << getHashTableVarName(build_attr_)
             << " < bucket_" << getHashTableVarName(build_attr_) << "->count; "
             << "bucket_tid_" << getHashTableVarName(build_attr_) << "++) {"
             << std::endl;
  hash_probe << "     if (bucket_" << getHashTableVarName(build_attr_)
             << "->tuples[bucket_tid_" << getHashTableVarName(build_attr_)
             << "].key == "
             << getElementAccessExpression(
                    probe_attr,
                    getTupleIDVarName(probe_attr.getTable(), loopVar))
             << ") {" << std::endl;
  hash_probe << "          " << getTupleIDVarName(build_attr_) << "="
             << "bucket_" << getHashTableVarName(build_attr_)
             << "->tuples[bucket_tid_" << getHashTableVarName(build_attr_)
             << "].value;" << std::endl;

  hash_probe_lower << "            }" << std::endl;
  hash_probe_lower << "        }" << std::endl;
  hash_probe_lower << "        bucket_" << getHashTableVarName(build_attr_)
                   << " = bucket_" << getHashTableVarName(build_attr_)
                   << "->next;" << std::endl;
  hash_probe_lower << "    }" << std::endl;

  return ProbeHashTableCode(hash_probe.str(), hash_probe_lower.str());
}

std::string BucketChainedHashTableGenerator::generateCodeHTEntryAccess() {
  std::stringstream expr;
  expr << "bucket_" << getHashTableVarName(build_attr_)
       << "->tuples[bucket_tid_" << getHashTableVarName(build_attr_)
       << "].payload";
  return expr.str();
}

std::string CuckooHashTableGenerator::generateCodeDeclareHashTable() {
  std::stringstream expr;

  /* add code for hash table creation before for loop */
  expr << "cuckoo_hashtable_t* " << getHashTableVarName(build_attr_)
       << "= NULL;" << std::endl;
  return expr.str();
}

std::string CuckooHashTableGenerator::generateCodeInsertIntoHashTable(
    uint32_t loopVar) {
  std::stringstream hash_table;
  /* insert values into hash table */
  hash_table << "tuple_t t_" << getHashTableVarName(build_attr_) << " = {"
             << getInputArrayVarName(build_attr_) << "["
             << getTupleIDVarName(build_attr_.getTable(), loopVar) << "], "
             << "current_result_size};" << std::endl;
  hash_table << "cuckoo_hash_put(" << getHashTableVarName(build_attr_);
  hash_table << ", t_" << getHashTableVarName(build_attr_) << ");" << std::endl;

  return hash_table.str();
}

ProbeHashTableCode CuckooHashTableGenerator::generateCodeProbeHashTable(
    const AttributeReference& probe_attr, uint32_t loopVar) {
  /* thoughts
  ht_value_t* result_vals;
  int num_of_results = cuckoo_hash_get(hashtableXYZ,attr.key,vals);
  for(int i = 0; i < num_of_results; i++){
      miau ...
  }
  */
  /* code for hash table probes */
  std::stringstream hash_probe;
  /* closing brackets and pointer chasing */
  std::stringstream hash_probe_lower;
  hash_probe << "ht_value_t* result_vals_" << getHashTableVarName(build_attr_)
             << ";" << std::endl;
  hash_probe << "unsigned int num_of_results_"
             << getHashTableVarName(build_attr_)
             << " = cuckoo_hash_get(hashtable_"
             << getHashTableVarName(build_attr_) << ","
             << getElementAccessExpression(probe_attr) << ", result_vals_"
             << getHashTableVarName(build_attr_) << ");" << std::endl;

  hash_probe << "  for (size_t bucket_tid_" << getHashTableVarName(build_attr_)
             << " = 0; "
             << "bucket_tid_" << getHashTableVarName(build_attr_)
             << " < num_of_results_" << getHashTableVarName(build_attr_) << "; "
             << "bucket_tid_" << getHashTableVarName(build_attr_) << "++) {"
             << std::endl;

  hash_probe_lower << "            }" << std::endl;

  return ProbeHashTableCode(hash_probe.str(), hash_probe_lower.str());
}

std::string CuckooHashTableGenerator::getHashTableCType() {
  return "cuckoo_hashtable_t*";
}

std::map<std::string, std::string>
CuckooHashTableGenerator::getBuildExtraInputVariables() {
  return {};
}

std::map<std::string, std::string>
CuckooHashTableGenerator::getProbeExtraInputVariables() {
  return {};
}

std::string CuckooHashTableGenerator::generateCodeHTEntryAccess() {
  std::stringstream expr;
  expr << "bucket_" << getHashTableVarName(build_attr_)
       << "->tuples[bucket_tid_" << getHashTableVarName(build_attr_)
       << "].payload";
  return expr.str();
}

std::string CuckooHashTableGenerator::generateCodeInitHashTable(
    bool create_new, const std::string& num_elements) {
  std::stringstream expr;

  if (create_new) {
    expr << getHashTableVarName(build_attr_) << "=cuckoo_hash_new ("
         << num_elements << "," << (uint32_t)numHTs << "," << seed << ");"
         << std::endl;
  } else {
    expr << getHashTableVarName(build_attr_)
         << " = (cuckoo_hashtable_t*) "
            "getHashTableFromSystemHashTable(generic_hashtable_"
         << getHashTableVarName(build_attr_) << ");" << std::endl;
  }
  return expr.str();
}

std::string CuckooHashTableGenerator::generateIdentifierCleanupHandler() {
  return "cuckoo_hash_release";
}

std::string LinearProbeHashTableGenerator::generateCodeDeclareHashTable() {
  std::stringstream expr;
  /* add code for hash table creation before for loop */
  expr << "C_UniqueKeyHashTable* " << getHashTableVarName(build_attr_)
       << " = NULL;" << std::endl;

  return expr.str();
}

std::string LinearProbeHashTableGenerator::generateCodeInitHashTable(
    bool create_new, const std::string& num_elements) {
  std::stringstream expr;

  if (create_new) {
    expr << getHashTableVarName(build_attr_)
         << " = createUniqueKeyJoinHashTable(" << num_elements << ");"
         << std::endl;
  } else {
    expr << getHashTableVarName(build_attr_)
         << " = (C_UniqueKeyHashTable*) "
            "getHashTableFromSystemHashTable(generic_hashtable_"
         << getHashTableVarName(build_attr_) << ");" << std::endl;
  }
  return expr.str();
}

std::string LinearProbeHashTableGenerator::generateIdentifierCleanupHandler() {
  return "freeUniqueKeyJoinHashTable";
}

std::string LinearProbeHashTableGenerator::generateCodeInsertIntoHashTable(
    uint32_t loopVar) {
  std::stringstream expr;
  expr << "insertUniqueKeyJoinHashTable(" << getHashTableVarName(build_attr_)
       << "," << getElementAccessExpression(build_attr_) << ", "
       << "current_result_size);" << std::endl;
  return expr.str();
}

ProbeHashTableCode LinearProbeHashTableGenerator::generateCodeProbeHashTable(
    const AttributeReference& probe_attr, uint32_t loopVar) {
  /* code for hash table probes */
  std::stringstream hash_probe;
  /* closing brackets and pointer chasing */
  std::stringstream hash_probe_lower;
  hash_probe << "uint64_t* ht_payload_" << getHashTableVarName(build_attr_)
             << " = getUniqueKeyJoinHashTablePayload("
             << getHashTableVarName(build_attr_) << ","
             << getElementAccessExpression(probe_attr) << ");" << std::endl;
  hash_probe << "if(ht_payload_" << getHashTableVarName(build_attr_) << "){";
  hash_probe << getTupleIDVarName(build_attr_) << " = *ht_payload_"
             << getHashTableVarName(build_attr_) << ";";
  hash_probe_lower << "}" << std::endl;
  return ProbeHashTableCode(hash_probe.str(), hash_probe_lower.str());
}

std::string LinearProbeHashTableGenerator::generateCodeHTEntryAccess() {
  std::stringstream expr;
  expr << "(*ht_payload_" << getHashTableVarName(build_attr_) << ")"
       << std::endl;
  return expr.str();
}

std::string LinearProbeHashTableGenerator::getHashTableCType() {
  return "C_UniqueKeyHashTable*";
}

std::map<std::string, std::string>
LinearProbeHashTableGenerator::getBuildExtraInputVariables() {
  return {};
}

std::map<std::string, std::string>
LinearProbeHashTableGenerator::getProbeExtraInputVariables() {
  return {};
}

std::string OCLHashTableGenerator::generateCodeDeclareHashTable() {
  std::stringstream expr;
  /* add code for hash table creation before for loop */
  expr << "uint64_t* " << getHashTableVarName(build_attr_) << " = NULL;"
       << std::endl;

  return expr.str();
}

std::string OCLHashTableGenerator::generateCodeInitHashTable(
    bool create_new, const std::string& num_elements) {
  std::stringstream expr;

  if (create_new) {
    expr << getHashTableVarName(build_attr_)
         << " = (uint64_t*)malloc(sizeof(uint64_t) * 4 * " << num_elements
         << " + 1);" << std::endl;
    expr << "memset(" << getHashTableVarName(build_attr_)
         << ", 0xFF, sizeof(uint64_t) * 4 * " << num_elements << " + 1);"
         << std::endl;
    expr << getHashTableVarName(build_attr_) << "[0] = " << num_elements
         << " * 2;" << std::endl;
  } else {
    expr << getHashTableVarName(build_attr_)
         << " = (uint64_t*) getHashTableFromSystemHashTable(generic_hashtable_"
         << getHashTableVarName(build_attr_) << ");" << std::endl;
  }
  return expr.str();
}

std::string OCLHashTableGenerator::generateIdentifierCleanupHandler() {
  return "free";
}

std::string OCLHashTableGenerator::generateCodeInsertIntoHashTable(
    uint32_t loopVar) {
  std::stringstream expr;
  auto bucket_name = getBucketName();
  std::string table_size = getHashTableVarName(build_attr_) + "[0]";
  auto key_access = getHashTableVarName(build_attr_) + "[" + bucket_name +
                    " % " + table_size + " + " + table_size + " + 1]";
  std::vector<std::string> hashes = {getHash1(),
                                     getHash2(),
                                     getHash3(),
                                     getHash1("^ 0x348c3def"),
                                     getHash2("^ 0x348c3def"),
                                     getHash3("^ 0x348c3def")};

  for (auto i = 0u; i < 10; ++i) {
    hashes.push_back(getHashLinearProbing());
  }

  expr << "uint64_t " << bucket_name
       << "_key = " << getElementAccessExpression(build_attr_) << ";"
       << std::endl;
  expr << "uint64_t " << bucket_name << " = " << bucket_name << "_key;"
       << std::endl;
  expr << "uint64_t " << bucket_name << "_found = " << bucket_name << "_key;"
       << std::endl;
  expr << "char " << bucket_name << "_found_bucket = 0;" << std::endl;

  for (const auto& hash : hashes) {
    expr << hash << std::endl;

    expr << "if (" << bucket_name << "_found_bucket == 0 && (" << key_access
         << " == 0xFFFFFFFFFFFFFFFF || " << key_access << " == " << bucket_name
         << "_key)) {" << std::endl
         << "  " << bucket_name << "_found_bucket = 1;" << std::endl
         << "  " << bucket_name << "_found = " << bucket_name << ";"
         << std::endl
         << "}" << std::endl;
  }

  // TODO, allocated_result_elements we don't know if the variable exists and
  // write_pos, bucket_name_found_bucket == 0 is ignored
  expr << "if (" << bucket_name << "_found_bucket == 1) {" << std::endl;
  expr << "  " << getHashTableVarName(build_attr_) << "[" << bucket_name
       << "_found % " << table_size << " + 1]"
       << " = write_pos;" << std::endl;
  expr << "  " << getHashTableVarName(build_attr_) << "[" << bucket_name
       << "_found % " << table_size << " + " << table_size << " + 1]"
       << " = " << bucket_name << "_key;" << std::endl;
  expr << "};" << std::endl;

  return expr.str();
}

ProbeHashTableCode OCLHashTableGenerator::generateCodeProbeHashTable(
    const AttributeReference& probe_attr, uint32_t loopVar) {
  /* code for hash table probes */
  std::stringstream hash_probe;
  /* closing brackets and pointer chasing */
  std::stringstream hash_probe_lower;
  auto bucket_name = getBucketName();
  std::string table_size = getHashTableVarName(build_attr_) + "[0]";

  auto key_access = getHashTableVarName(build_attr_) + "[" + bucket_name +
                    " % " + table_size + " + " + table_size + " + 1]";
  std::vector<std::string> hashes = {getHash1(),
                                     getHash2(),
                                     getHash3(),
                                     getHash1("^ 0x348c3def"),
                                     getHash2("^ 0x348c3def"),
                                     getHash3("^ 0x348c3def")};

  for (auto i = 0u; i < 10; ++i) {
    hashes.push_back(getHashLinearProbing());
  }

  hash_probe << "uint64_t " << bucket_name
             << "_key = " << getElementAccessExpression(probe_attr) << ";"
             << std::endl;
  hash_probe << "uint64_t " << bucket_name << " = " << bucket_name << "_key;"
             << std::endl;
  hash_probe << "uint64_t " << bucket_name << "_found = " << bucket_name
             << "_key;" << std::endl;
  hash_probe << "char " << bucket_name << "_found_key = 0;" << std::endl;

  for (const auto& hash : hashes) {
    hash_probe << hash << std::endl;

    hash_probe << "if (" << bucket_name << "_found_key == 0 && " << key_access
               << " == " << bucket_name << "_key) {" << std::endl
               << "  " << bucket_name << "_found_key = 1;" << std::endl
               << "  " << bucket_name << "_found = " << bucket_name << ";"
               << std::endl
               << "}" << std::endl;
  }

  hash_probe << "uint64_t ht_payload_" << getHashTableVarName(build_attr_)
             << " = " << getHashTableVarName(build_attr_) << "[" << bucket_name
             << "_found % " << table_size << " + 1];" << std::endl;
  hash_probe << "if (" << bucket_name << "_found_key == 1) {" << std::endl;
  hash_probe << getTupleIDVarName(build_attr_) << " = ht_payload_"
             << getHashTableVarName(build_attr_) << ";" << std::endl;

  hash_probe_lower << "}" << std::endl;
  return ProbeHashTableCode(hash_probe.str(), hash_probe_lower.str());
}

std::string OCLHashTableGenerator::generateCodeHTEntryAccess() {
  COGADB_NOT_IMPLEMENTED;
}

// source:
// https://bitbucket.org/msaecker/monetdb-opencl/src/0afb0d013e8c5c1d330e0599f1a65bfcd2c8beec/monetdb5/extras/ocelot/clKernel/hash.h?at=simple_mem_manager&fileviewer=file-view-default
std::string OCLHashTableGenerator::getHash1(const std::string& key_modifier) {
  std::stringstream expr;
  auto bucket = getBucketName();

  expr << bucket << " = " << bucket << "_key " << key_modifier << ";"
       << std::endl;
  expr << bucket << " = (" << bucket << " + 0x7ed55d16) + (" << bucket
       << " << 12);" << std::endl;
  expr << bucket << " = (" << bucket << " ^ 0xc761c23c) ^ (" << bucket
       << " >> 19);" << std::endl;
  expr << bucket << " = (" << bucket << " + 0x165667b1) + (" << bucket
       << " << 5);" << std::endl;
  expr << bucket << " = (" << bucket << " + 0xd3a2646c) ^ (" << bucket
       << " << 9);" << std::endl;
  expr << bucket << " = (" << bucket << " + 0xfd7046c5) + (" << bucket
       << " << 3);" << std::endl;
  expr << bucket << " = (" << bucket << " ^ 0xb55a4f09) ^ (" << bucket
       << " >> 16);" << std::endl;

  return expr.str();
}

std::string OCLHashTableGenerator::getHash2(const std::string& key_modifier) {
  std::stringstream expr;

  auto bucket = getBucketName();

  expr << bucket << " = " << bucket << "_key " << key_modifier << ";"
       << std::endl;
  expr << bucket << " = (" << bucket << " + 0x7fb9b1ee) + (" << bucket
       << " << 12);" << std::endl;
  expr << bucket << " = (" << bucket << " ^ 0xab35dd63) ^ (" << bucket
       << " >> 19);" << std::endl;
  expr << bucket << " = (" << bucket << " + 0x41ed960d) + (" << bucket
       << " << 5);" << std::endl;
  expr << bucket << " = (" << bucket << " + 0xc7d0125e) ^ (" << bucket
       << " << 9);" << std::endl;
  expr << bucket << " = (" << bucket << " + 0x071f9f8f) + (" << bucket
       << " << 3);" << std::endl;
  expr << bucket << " = (" << bucket << " ^ 0x55ab55b9) ^ (" << bucket
       << " >> 16);" << std::endl;

  return expr.str();
}

std::string OCLHashTableGenerator::getHash3(const std::string& key_modifier) {
  std::stringstream expr;

  auto bucket = getBucketName();

  expr << bucket << " = " << bucket << "_key" << key_modifier << ";"
       << std::endl;
  expr << bucket << " = (" << bucket << " ^ 61) ^ (" << bucket << " >> 16);"
       << std::endl;
  expr << bucket << " = (" << bucket << ")      + (" << bucket << " << 3);"
       << std::endl;
  expr << bucket << " = (" << bucket << ")      ^ (" << bucket << " >> 4);"
       << std::endl;
  expr << bucket << " = (" << bucket << ")      * (0x27d4eb2d);" << std::endl;
  expr << bucket << " = (" << bucket << ")      ^ (" << bucket << " >> 15);"
       << std::endl;

  return expr.str();
}

std::string OCLHashTableGenerator::getHashLinearProbing() {
  std::stringstream expr;

  auto bucket = getBucketName();

  expr << bucket << " = (" << bucket << " + 1);" << std::endl;

  return expr.str();
}

std::string OCLHashTableGenerator::getHashTableCType() { return "uint64_t*"; }

std::string OCLHashTableGenerator::getBucketName() {
  return "bucket_" + getHashTableVarName(build_attr_);
}

std::map<std::string, std::string>
OCLHashTableGenerator::getBuildExtraInputVariables() {
  return {};
}

std::map<std::string, std::string>
OCLHashTableGenerator::getProbeExtraInputVariables() {
  return {};
}

namespace CuckooHT {

std::string createAndInitHT(const std::string& struct_name,
                            const std::string& member_name,
                            const std::string& var_name,
                            const std::string ctype = "uint64_t") {
  std::stringstream expr;

  auto ht_bytesize =
      "sizeof(" + ctype + ") * " + struct_name + "->" + member_name + "_size";

  expr << struct_name << "->" << member_name << " = " << var_name << " = ("
       << ctype << "*)malloc(" << ht_bytesize << ");" << std::endl;
  expr << "memset(" << var_name << ", 0xFF, " << ht_bytesize << ");"
       << std::endl;

  return expr.str();
}

std::string declareHT(const std::string& ht_var_name,
                      const std::string ctype = "uint64_t") {
  std::stringstream ss;

  ss << ctype << "* " + ht_var_name + " = NULL;" << std::endl;
  ss << ctype << " " << ht_var_name << "_length = 0;" << std::endl;

  return ss.str();
}

std::string getKeyVarName(bool use_evicted, bool dereference = true) {
  std::string ptr = dereference ? "*" : "";

  return use_evicted ? ptr + "evicted_key" : "key";
}

std::string getPayloadVarName(bool use_evicted, bool dereference = true) {
  std::string ptr = dereference ? "*" : "";

  return use_evicted ? ptr + "evicted_payload" : "payload";
}

std::string getCodeCallInsertKeyAndPayload(
    const std::string& ht_name, const std::string& ht_mask,
    const std::string& func_name_postfix) {
  std::stringstream expr;

  expr << "  if (insertKeyAndPayload" << func_name_postfix << "(" << ht_name
       << ", " << ht_mask << ", evicted_key, evicted_payload, "
       << "&evicted_key, &evicted_payload)) {" << std::endl
       << "    return;" << std::endl
       << "  }" << std::endl;

  return expr.str();
}

std::string getCodePostProcessIndex() {
  std::stringstream ss;

  ss << "  index &= hash_table_mask; " << std::endl
     // even indices are used for the key, uneven for the payload
     << "  index &= ~1ul; " << std::endl;

  return ss.str();
}

std::string getCodeInsertKeyAndPayloadForHash(const std::string& hash_code,
                                              bool use_evicted) {
  std::stringstream expr;

  auto invalid_key = "0xFFFFFFFFFFFFFFFF";
  auto key_var_name = getKeyVarName(use_evicted);
  auto payload_var_name = getPayloadVarName(use_evicted);

  expr << "  index = key;" << std::endl
       << hash_code << std::endl
       << getCodePostProcessIndex() << std::endl
       << "  tmp_key = hash_table[index];" << std::endl
       << "  tmp_payload = hash_table[index + 1];" << std::endl
       << "  hash_table[index] = " << key_var_name << ";" << std::endl
       << "  hash_table[index + 1] = " << payload_var_name << ";" << std::endl
       << "  if (tmp_key == " << invalid_key
       << " || tmp_key == " << key_var_name << ") {" << std::endl
       << "    return true;" << std::endl
       << "  }" << std::endl
       << "  *evicted_payload = tmp_payload;" << std::endl
       << "  *evicted_key = tmp_key;" << std::endl;

  return expr.str();
}

std::string getCodeCallFindKeyImpl(const std::string& ht_name,
                                   const std::string& ht_mask,
                                   const std::string& func_name_postfix) {
  std::stringstream expr;

  expr << "  if (findKeyCuckooHTImpl" << func_name_postfix << "(" << ht_name
       << ", " << ht_mask << ", key, found_key, found_payload, "
       << "accept_invalid_key)) {" << std::endl
       << "    return true;" << std::endl
       << "  }" << std::endl;

  return expr.str();
}

std::string getCodeCheckForKeyWithHash(const std::string& hash_code) {
  std::stringstream expr;

  auto invalid_key = "0xFFFFFFFFFFFFFFFF";

  expr << "  index = key;" << std::endl
       << hash_code << std::endl
       << getCodePostProcessIndex() << std::endl
       << "  if (hash_table[index] == key || (accept_invalid_key && "
       << "hash_table[index] == " << invalid_key << ")) {" << std::endl
       << "    *found_payload = &hash_table[index + 1];" << std::endl
       << "    *found_key = &hash_table[index];" << std::endl
       << "    return true;" << std::endl
       << "  }" << std::endl;

  return expr.str();
}

std::string getCodeMakeSizeMultipleOfTwo(const std::string& result_name,
                                         const std::string& minimum_size) {
  std::stringstream expr;

  expr << "uint64_t " << result_name << " = 1;" << std::endl
       << "  while (" << result_name << " < " << minimum_size << ") {"
       << std::endl
       << result_name << " <<= 1;" << std::endl
       << "}" << std::endl;

  return expr.str();
}
}

std::string OCLCuckooHashTableGenerator::generateCodeDeclareHashTable() {
  std::stringstream expr;
  /* add code for hash table creation before for loop */
  expr << "OCLCuckooHT* " << getStructVarName() << " = NULL;" << std::endl;

  expr << CuckooHT::declareHT(getHTVarName()) << std::endl;
  expr << CuckooHT::declareHT(getOverflowVarName()) << std::endl;

  return expr.str();
}

std::string OCLCuckooHashTableGenerator::generateCodeInitHashTable(
    bool create_new, const std::string& num_elements) {
  std::stringstream expr;

  if (create_new) {
    expr << getStructVarName() << " = malloc(sizeof(OCLCuckooHT));"
         << std::endl;

    expr << CuckooHT::getCodeMakeSizeMultipleOfTwo(
        getHTVarName() + "_length_tmp", "2 * " + num_elements);
    expr << getHTVarName() << "_length = " << getStructVarName()
         << "->hash_table_size = " << getHTVarName() << "_length_tmp;"
         << std::endl;

    expr << CuckooHT::createAndInitHT(getStructVarName(), "hash_table",
                                      getHTVarName())
         << std::endl;

    expr << CuckooHT::getCodeMakeSizeMultipleOfTwo(
        getOverflowVarName() + "_length_tmp",
        "0.1 * " + getStructVarName() + "->hash_table_size");
    expr << getOverflowVarName() << "_length = " << getStructVarName()
         << "->overflow_size = " << getOverflowVarName() << "_length_tmp;"
         << std::endl;

    expr << CuckooHT::createAndInitHT(getStructVarName(), "overflow",
                                      getOverflowVarName())
         << std::endl;
  } else {
    expr << getStructVarName()
         << " = (OCLCuckooHT*)getHashTableFromSystemHashTable("
         << "generic_hashtable_" << getHashTableVarName(build_attr_) << ");"
         << std::endl;

    expr << getHTVarName() << " = " << getStructVarName() << "->hash_table;"
         << std::endl;
    expr << getHTVarName() << "_length = " << getStructVarName()
         << "->hash_table_size;" << std::endl;

    expr << getOverflowVarName() << " = " << getStructVarName() << "->overflow;"
         << std::endl;

    expr << getOverflowVarName() << "_length = " << getStructVarName()
         << "->overflow_size;" << std::endl;
  }

  return expr.str();
}

std::string OCLCuckooHashTableGenerator::generateIdentifierCleanupHandler() {
  return "freeOCLCuckooHT";
}

std::string OCLCuckooHashTableGenerator::generateCodeInsertIntoHashTable(
    uint32_t loopVar) {
  std::stringstream expr;

  expr << "insertIntoCuckooHT" << getHTVarName() << "(" << getHTVarName()
       << ", " << getHTVarName() << "_length - 1, " << getOverflowVarName()
       << ", " << getOverflowVarName() << "_length - 1, "
       << getElementAccessExpression(build_attr_) << ", write_pos);"
       << std::endl;

  return expr.str();
}

ProbeHashTableCode OCLCuckooHashTableGenerator::generateCodeProbeHashTable(
    const AttributeReference& probe_attr, uint32_t loopVar) {
  /* code for hash table probes */
  std::stringstream hash_probe;
  /* closing brackets and pointer chasing */
  std::stringstream hash_probe_lower;

  hash_probe << "__global uint64_t const* " << getHTVarName() << "_key = NULL;"
             << std::endl;
  hash_probe << "__global uint64_t const* " << getHTVarName()
             << "_payload = NULL;" << std::endl;
  hash_probe << "if (findKeyCuckooHT" << getHTVarName() << "(" << getHTVarName()
             << ", " << getHTVarName() << "_length - 1, "
             << getOverflowVarName() << ", " << getOverflowVarName()
             << "_length - 1, " << getElementAccessExpression(probe_attr)
             << ", &" << getHTVarName() << "_key, &" << getHTVarName()
             << "_payload, false)) {" << std::endl;

  hash_probe << getTupleIDVarName(build_attr_) << " = *" << getHTVarName()
             << "_payload;" << std::endl;

  hash_probe_lower << "}" << std::endl;
  return ProbeHashTableCode(hash_probe.str(), hash_probe_lower.str());
}

std::string OCLCuckooHashTableGenerator::generateCodeHTEntryAccess() {
  COGADB_NOT_IMPLEMENTED;
}

std::string OCLCuckooHashTableGenerator::getHashTableCType() {
  return "uint64_t*";
}

std::map<std::string, std::string>
OCLCuckooHashTableGenerator::getBuildExtraInputVariables() {
  return {{getHTVarName() + "_length", "uint64_t"},
          {getOverflowVarName() + "_length", "uint64_t"}};
}

std::map<std::string, std::string>
OCLCuckooHashTableGenerator::getBuildExtraOutputVariables() {
  return {{getOverflowVarName(), "uint64_t*"}};
}

std::map<std::string, std::string>
OCLCuckooHashTableGenerator::getProbeExtraInputVariables() {
  return {{getOverflowVarName(), "uint64_t*"},
          {getHTVarName() + "_length", "uint64_t"},
          {getOverflowVarName() + "_length", "uint64_t"}};
}

std::string OCLCuckooHashTableGenerator::generateCodeForHeaderAndTypesBlock() {
  std::stringstream expr;

  expr << "bool findKeyCuckooHTImpl" << getHTVarName()
       << "(const __global uint64_t* const hash_table, "
       << "uint64_t hash_table_mask, uint64_t key, "
       << "__global uint64_t const** found_key, "
       << "__global uint64_t const** found_payload, bool accept_invalid_key) {"
       << std::endl;

  expr << "  uint64_t index = 0;" << std::endl;

  expr << CuckooHT::getCodeCheckForKeyWithHash(getCodeMurmur3Hashing())
       << std::endl;

  expr << CuckooHT::getCodeCheckForKeyWithHash(getCodeMultiplyShift())
       << std::endl;

  expr << CuckooHT::getCodeCheckForKeyWithHash(getCodeMultiplyAddShift())
       << std::endl;

  expr << "  return false;" << std::endl << "}" << std::endl;

  expr << "bool findKeyCuckooHT" << getHTVarName()
       << "(const __global uint64_t* const hash_table, "
       << "uint64_t hash_table_mask, const __global uint64_t* const overflow, "
       << "uint64_t overflow_mask, uint64_t key, "
       << "__global uint64_t const** found_key, "
       << "__global uint64_t const** found_payload, bool accept_invalid_key) {"
       << std::endl;

  expr << CuckooHT::getCodeCallFindKeyImpl("hash_table", "hash_table_mask",
                                           getHTVarName())
       << std::endl;

  expr << CuckooHT::getCodeCallFindKeyImpl("overflow", "overflow_mask",
                                           getHTVarName())
       << std::endl;

  expr << "  return false;" << std::endl << "}" << std::endl;

  expr << "bool insertKeyAndPayload" << getHTVarName()
       << "(__global uint64_t* hash_table, uint64_t hash_table_mask, "
       << "uint64_t key, uint64_t payload, uint64_t* evicted_key, "
       << "uint64_t* evicted_payload) { " << std::endl;

  expr << "  uint64_t index = 0, tmp_key = 0, tmp_payload = 0;" << std::endl;

  expr << CuckooHT::getCodeInsertKeyAndPayloadForHash(getCodeMurmur3Hashing(),
                                                      false)
       << std::endl;

  expr << CuckooHT::getCodeInsertKeyAndPayloadForHash(getCodeMultiplyShift(),
                                                      true)
       << std::endl;

  expr << CuckooHT::getCodeInsertKeyAndPayloadForHash(getCodeMultiplyAddShift(),
                                                      true)
       << std::endl;

  expr << "  return false;" << std::endl << "}" << std::endl;

  expr << "void insertIntoCuckooHT" << getHTVarName()
       << "(__global uint64_t* hash_table, uint64_t hash_table_mask, "
       << "__global uint64_t* overflow, uint64_t overflow_mask, uint64_t key,"
       << " uint64_t payload) {" << std::endl;

  expr << "  __global uint64_t* const found_key = NULL;" << std::endl
       << "  __global uint64_t* const found_payload = NULL;" << std::endl
       << "  if (findKeyCuckooHT" << getHTVarName() << "(hash_table, "
       << "hash_table_mask, overflow, overflow_mask, key, &found_key, "
       << "&found_payload, true)) {" << std::endl
       << "    *found_key = key;" << std::endl
       << "    *found_payload = payload;" << std::endl
       << "    return;" << std::endl
       << "  }" << std::endl;

  const unsigned int max_iterations = 10;

  expr << "  uint64_t evicted_key = key, evicted_payload = payload;"
       << std::endl;

  expr << "  for (unsigned int i = 0; i < " << max_iterations << "; ++i) {"
       << std::endl;

  expr << CuckooHT::getCodeCallInsertKeyAndPayload(
      "hash_table", "hash_table_mask", getHTVarName());

  expr << CuckooHT::getCodeCallInsertKeyAndPayload("overflow", "overflow_mask",
                                                   getHTVarName());

  expr << "  }" << std::endl;

  // TODO, handle the case that we could not insert the key!

  expr << "}" << std::endl;

  return expr.str();
}

std::string OCLCuckooHashTableGenerator::getStructVarName() const {
  return getHTVarName() + "_struct";
}

std::string OCLCuckooHashTableGenerator::getOverflowVarName() const {
  return getHTVarName() + "_overflow";
}

std::string OCLCuckooHashTableGenerator::getHTVarName() const {
  return getHashTableVarName(build_attr_);
}

std::string OCLCuckooHashTableGenerator::getHTVarNameForSystemHT() const {
  return getStructVarName();
}

namespace LinearProbingHT {

std::string getCodeInsertKeyAndPayloadForHash(const std::string& hash_code) {
  std::stringstream expr;

  auto invalid_key = "0xFFFFFFFF";

  expr << "  index = key;" << std::endl
       << hash_code << std::endl
       << "  for (unsigned int i = 0; i < hash_table_mask + 1; ++i, index += 2)"
       << " {" << std::endl
       << CuckooHT::getCodePostProcessIndex() << std::endl
       << "    unsigned int old = hash_table[index];"
       << "    if (old == key) {" << std::endl
       << "      return;" << std::endl
       << "    } else if (old == " << invalid_key << ") {" << std::endl
       << "      old = "
       << "atomic_cmpxchg(&hash_table[index], " << invalid_key << ", key);"
       << std::endl
       << "      if (old == " << invalid_key << " || old == key) {" << std::endl
       << "        hash_table[index + 1] = payload;" << std::endl
       << "        return;"
       << "      }" << std::endl
       << "    }" << std::endl
       << "  }" << std::endl;

  return expr.str();
}

std::string getCodeCheckForKeyWithHash(const std::string& hash_code) {
  std::stringstream expr;

  auto invalid_key = "0xFFFFFFFF";

  expr
      << "  index = key;" << std::endl
      << hash_code << std::endl
      << " for (unsigned int i = 0; i < hash_table_mask + 1; ++i, index += 2) {"
      << std::endl
      << CuckooHT::getCodePostProcessIndex() << std::endl
      << "    if (hash_table[index] == key) {" << std::endl
      << "      *found_payload = &hash_table[index + 1];" << std::endl
      << "      return true;" << std::endl
      << "    } else if (hash_table[index] == " << invalid_key << ") {"
      << std::endl
      << "      return false;" << std::endl
      << "    }" << std::endl
      << "  }" << std::endl;

  return expr.str();
}
}

std::string OCLLinearProbingHashTableGenerator::generateCodeDeclareHashTable() {
  std::stringstream expr;
  /* add code for hash table creation before for loop */
  expr << "OCLLinearProbingHT* " << getStructVarName() << " = NULL;"
       << std::endl;

  expr << CuckooHT::declareHT(getHTVarName(), "uint32_t") << std::endl;

  return expr.str();
}

std::string OCLLinearProbingHashTableGenerator::generateCodeInitHashTable(
    bool create_new, const std::string& num_elements) {
  std::stringstream expr;

  if (create_new) {
    expr << getStructVarName() << " = malloc(sizeof(OCLLinearProbingHT));"
         << std::endl;

    expr << CuckooHT::getCodeMakeSizeMultipleOfTwo(
        getHTVarName() + "_length_tmp", "2 * " + num_elements);
    expr << getHTVarName() << "_length = " << getStructVarName()
         << "->hash_table_size = " << getHTVarName() << "_length_tmp;"
         << std::endl;

    expr << CuckooHT::createAndInitHT(getStructVarName(), "hash_table",
                                      getHTVarName(), "uint32_t")
         << std::endl;
  } else {
    expr << getStructVarName()
         << " = (OCLLinearProbingHT*)getHashTableFromSystemHashTable("
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
OCLLinearProbingHashTableGenerator::generateIdentifierCleanupHandler() {
  return "freeOCLLinearProbingHT";
}

std::string OCLLinearProbingHashTableGenerator::generateCodeInsertIntoHashTable(
    uint32_t loopVar) {
  std::stringstream expr;

  expr << "insertIntoLinearProbingHT" << getHTVarName() << "(" << getHTVarName()
       << ", " << getHTVarName() << "_length - 1, "
       << getElementAccessExpression(build_attr_) << ", write_pos);"
       << std::endl;

  return expr.str();
}

ProbeHashTableCode
OCLLinearProbingHashTableGenerator::generateCodeProbeHashTable(
    const AttributeReference& probe_attr, uint32_t loopVar) {
  /* code for hash table probes */
  std::stringstream hash_probe;
  /* closing brackets and pointer chasing */
  std::stringstream hash_probe_lower;

  hash_probe << "__global uint32_t const* " << getHTVarName()
             << "_payload = NULL;" << std::endl;
  hash_probe << "if (findKeyLinearProbingHT" << getHTVarName() << "("
             << getHTVarName() << ", " << getHTVarName() << "_length - 1, "
             << getElementAccessExpression(probe_attr) << ", &"
             << getHTVarName() << "_payload)) {" << std::endl;

  hash_probe << getTupleIDVarName(build_attr_) << " = *" << getHTVarName()
             << "_payload;" << std::endl;

  hash_probe_lower << "}" << std::endl;
  return ProbeHashTableCode(hash_probe.str(), hash_probe_lower.str());
}

std::string OCLLinearProbingHashTableGenerator::generateCodeHTEntryAccess() {
  COGADB_NOT_IMPLEMENTED;
}

std::string OCLLinearProbingHashTableGenerator::getHashTableCType() {
  return "uint32_t*";
}

std::map<std::string, std::string>
OCLLinearProbingHashTableGenerator::getBuildExtraInputVariables() {
  return {{getHTVarName() + "_length", "uint32_t"}};
}

std::map<std::string, std::string>
OCLLinearProbingHashTableGenerator::getBuildExtraOutputVariables() {
  return {};
}

std::map<std::string, std::string>
OCLLinearProbingHashTableGenerator::getProbeExtraInputVariables() {
  return {{getHTVarName() + "_length", "uint32_t"}};
}

std::string
OCLLinearProbingHashTableGenerator::generateCodeForHeaderAndTypesBlock() {
  std::stringstream expr;

  expr << "bool findKeyLinearProbingHT" << getHTVarName()
       << "(const __global uint32_t* const hash_table, "
       << "uint32_t hash_table_mask, uint32_t key, "
       << "__global uint32_t const** found_payload) {" << std::endl;

  expr << "  uint32_t index = 0;" << std::endl;

  expr << LinearProbingHT::getCodeCheckForKeyWithHash(getCodeMultiplyShift())
       << std::endl;

  expr << "  return false;" << std::endl << "}" << std::endl;

  expr << "void insertIntoLinearProbingHT" << getHTVarName()
       << "(__global uint32_t* hash_table, uint32_t hash_table_mask, "
       << "uint32_t key, uint32_t payload) {" << std::endl;

  expr << "  uint32_t index = 0;" << std::endl;

  expr << LinearProbingHT::getCodeInsertKeyAndPayloadForHash(
              getCodeMultiplyShift())
       << std::endl;

  // TODO, handle the case that we could not insert the key!

  expr << "}" << std::endl;

  return expr.str();
}

std::string OCLLinearProbingHashTableGenerator::getStructVarName() const {
  return getHTVarName() + "_struct";
}

std::string OCLLinearProbingHashTableGenerator::getHTVarName() const {
  return getHashTableVarName(build_attr_);
}

std::string OCLLinearProbingHashTableGenerator::getHTVarNameForSystemHT()
    const {
  return getStructVarName();
}
}
