/*
 * File:   hash_table_generator.hpp
 * Author: sebastian
 *
 * Created on 2. Januar 2016, 11:58
 */

#ifndef HASH_TABLE_GENERATOR_HPP
#define HASH_TABLE_GENERATOR_HPP

#include <assert.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/any.hpp>
#include <core/global_definitions.hpp>
#include <sstream>
#include <string>
#include <vector>

#include <core/attribute_reference.hpp>
#include <util/code_generation.hpp>

namespace CoGaDB {
  // class BucketChainedHashTableGenerator;
  class HashTableGenerator;
  typedef boost::shared_ptr<HashTableGenerator> HashTableGeneratorPtr;

  class AttributeReference;
  typedef boost::shared_ptr<AttributeReference> AttributeReferencePtr;

  typedef std::pair<std::string, std::string> ProbeHashTableCode;

  const HashTableGeneratorPtr createHashTableGenerator(
      const std::string& identifier, const AttributeReference& build_attr);
  const HashTableGeneratorPtr createHashTableGenerator(
      const AttributeReference& build_attr);

  class HashTableGenerator {
   public:
    virtual ~HashTableGenerator() {}
    const AttributeReference build_attr_;
    std::vector<StructFieldPtr> payload_fields_;
    const std::string identifier_;
    virtual std::string generateCodeDeclareHashTable() = 0;
    virtual std::string generateCodeInitHashTable(
        bool create_new, const std::string& num_elements = "") = 0;
    virtual std::string generateCodeCleanupHashTable() = 0;
    virtual std::string generateIdentifierCleanupHandler() = 0;

    virtual std::string generateCodeInsertIntoHashTable(uint32_t loopVar) = 0;
    virtual ProbeHashTableCode generateCodeProbeHashTable(
        const AttributeReference& attr, uint32_t loopVar) = 0;
    virtual std::string generateCodeForHeaderAndTypesBlock() { return ""; }

    virtual std::string generateCodeHTEntryAccess() = 0;
    virtual std::map<std::string, std::string>
    getBuildExtraInputVariables() = 0;
    virtual std::map<std::string, std::string> getBuildExtraOutputVariables() {
      return {};
    };
    virtual std::map<std::string, std::string>
    getProbeExtraInputVariables() = 0;
    virtual std::string getHashTableCType() = 0;
    virtual std::string getHTVarNameForSystemHT() const;

    friend const HashTableGeneratorPtr createHashTableGenerator(
        const std::string& identifier, const AttributeReference& build_attr);
    friend const HashTableGeneratorPtr createHashTableGenerator(
        const AttributeReference& build_attr);

   protected:
    HashTableGenerator(const AttributeReference& build_attr,
                       const std::vector<StructFieldPtr>& payload_fields,
                       const std::string& identifier)
        : build_attr_(build_attr),
          payload_fields_(payload_fields),
          identifier_(identifier) {}
    HashTableGenerator(const AttributeReference& build_attr,
                       const std::string& identifier)
        : build_attr_(build_attr), identifier_(identifier) {}
  };

  class BucketChainedHashTableGenerator : public HashTableGenerator {
   public:
    std::string generateCodeDeclareHashTable();
    std::string generateCodeInitHashTable(bool create_new,
                                          const std::string& num_elements = "");
    std::string generateCodeCleanupHashTable() {
      COGADB_FATAL_ERROR("IMPLEMENT", "");
    }
    std::string generateIdentifierCleanupHandler();

    std::string generateCodeInsertIntoHashTable(uint32_t loopVar = 1);
    ProbeHashTableCode generateCodeProbeHashTable(
        const AttributeReference& probe_attr, uint32_t loopVar = 1);

    std::string generateCodeHTEntryAccess();

    std::string getHashTableCType();
    std::map<std::string, std::string> getBuildExtraInputVariables();
    std::map<std::string, std::string> getProbeExtraInputVariables();
    // createprivate:
    BucketChainedHashTableGenerator(const AttributeReference& build_attr,
                                    std::vector<StructFieldPtr>& payload_fields,
                                    const std::string& identifier)
        : HashTableGenerator(build_attr, payload_fields, identifier) {}
    BucketChainedHashTableGenerator(const AttributeReference& build_attr,
                                    const std::string& identifier)
        : HashTableGenerator(build_attr, identifier) {}
  };

  class CuckooHashTableGenerator : public HashTableGenerator {
   public:
    const uint8_t numHTs;
    const uint32_t seed;
    std::string generateCodeDeclareHashTable();
    std::string generateCodeInitHashTable(bool create_new,
                                          const std::string& num_elements = "");
    std::string generateCodeCleanupHashTable() {
      COGADB_FATAL_ERROR("IMPLEMENT", "");
    }
    std::string generateIdentifierCleanupHandler();

    std::string generateCodeInsertIntoHashTable(uint32_t loopVar = 1);
    ProbeHashTableCode generateCodeProbeHashTable(
        const AttributeReference& probe_attr, uint32_t loopVar = 1);

    std::string generateCodeHTEntryAccess();
    std::string getHashTableCType();
    std::map<std::string, std::string> getBuildExtraInputVariables();
    std::map<std::string, std::string> getProbeExtraInputVariables();
    // createprivate:
    CuckooHashTableGenerator(const AttributeReference& build_attr,
                             std::vector<StructFieldPtr>& payload_fields,
                             uint8_t number_of_hashtables,
                             uint32_t seed_for_hashf,
                             const std::string& identifier)
        : HashTableGenerator(build_attr, payload_fields, identifier),
          numHTs(number_of_hashtables),
          seed(seed_for_hashf) {}
    CuckooHashTableGenerator(const AttributeReference& build_attr,
                             uint8_t number_of_hashtables,
                             uint32_t seed_for_hashf,
                             const std::string& identifier)
        : HashTableGenerator(build_attr, identifier),
          numHTs(number_of_hashtables),
          seed(seed_for_hashf) {}
  };

  class LinearProbeHashTableGenerator : public HashTableGenerator {
   public:
    std::string generateCodeDeclareHashTable();
    std::string generateCodeInitHashTable(bool create_new,
                                          const std::string& num_elements = "");
    std::string generateCodeCleanupHashTable() {
      COGADB_FATAL_ERROR("IMPLEMENT", "");
    }
    std::string generateIdentifierCleanupHandler();

    std::string generateCodeInsertIntoHashTable(uint32_t loopVar = 1);
    ProbeHashTableCode generateCodeProbeHashTable(
        const AttributeReference& probe_attr, uint32_t loopVar = 1);

    std::string generateCodeHTEntryAccess();
    std::string getHashTableCType();
    std::map<std::string, std::string> getBuildExtraInputVariables();
    std::map<std::string, std::string> getProbeExtraInputVariables();
    // createprivate:
    LinearProbeHashTableGenerator(const AttributeReference& build_attr,
                                  std::vector<StructFieldPtr>& payload_fields,
                                  const std::string& identifier)
        : HashTableGenerator(build_attr, payload_fields, identifier) {}
    LinearProbeHashTableGenerator(const AttributeReference& build_attr,
                                  const std::string& identifier)
        : HashTableGenerator(build_attr, identifier) {}
  };

  class OCLHashTableGenerator : public HashTableGenerator {
   public:
    std::string generateCodeDeclareHashTable();
    std::string generateCodeInitHashTable(bool create_new,
                                          const std::string& num_elements = "");
    std::string generateCodeCleanupHashTable() {
      COGADB_FATAL_ERROR("IMPLEMENT", "");
    }
    std::string generateIdentifierCleanupHandler();

    std::string generateCodeInsertIntoHashTable(uint32_t loopVar = 1);
    ProbeHashTableCode generateCodeProbeHashTable(
        const AttributeReference& probe_attr, uint32_t loopVar = 1);

    std::string generateCodeHTEntryAccess();
    std::string getHashTableCType();
    std::map<std::string, std::string> getBuildExtraInputVariables();
    std::map<std::string, std::string> getProbeExtraInputVariables();
    // createprivate:
    OCLHashTableGenerator(const AttributeReference& build_attr,
                          std::vector<StructFieldPtr>& payload_fields,
                          const std::string& identifier)
        : HashTableGenerator(build_attr, payload_fields, identifier) {}
    OCLHashTableGenerator(const AttributeReference& build_attr,
                          const std::string& identifier)
        : HashTableGenerator(build_attr, identifier) {}

   private:
    std::string getHash1(const std::string& key_modifier = "");
    std::string getHash2(const std::string& key_modifier = "");
    std::string getHash3(const std::string& key_modifier = "");
    std::string getHashLinearProbing();
    std::string getBucketName();
  };

  class OCLCuckooHashTableGenerator : public HashTableGenerator {
   public:
    std::string generateCodeDeclareHashTable();

    std::string generateCodeInitHashTable(bool create_new,
                                          const std::string& num_elements = "");

    std::string generateCodeCleanupHashTable() {
      COGADB_FATAL_ERROR("IMPLEMENT", "");
    }

    std::string generateIdentifierCleanupHandler();

    std::string generateCodeInsertIntoHashTable(uint32_t loopVar = 1);

    ProbeHashTableCode generateCodeProbeHashTable(
        const AttributeReference& probe_attr, uint32_t loopVar = 1);

    std::string generateCodeHTEntryAccess();

    std::string generateCodeForHeaderAndTypesBlock();

    std::string getHashTableCType();

    std::map<std::string, std::string> getBuildExtraInputVariables();

    std::map<std::string, std::string> getProbeExtraInputVariables();

    std::map<std::string, std::string> getBuildExtraOutputVariables();

    std::string getHTVarNameForSystemHT() const;

    // createprivate:
    OCLCuckooHashTableGenerator(const AttributeReference& build_attr,
                                std::vector<StructFieldPtr>& payload_fields,
                                const std::string& identifier)
        : HashTableGenerator(build_attr, payload_fields, identifier) {}

    OCLCuckooHashTableGenerator(const AttributeReference& build_attr,
                                const std::string& identifier)
        : HashTableGenerator(build_attr, identifier) {}

   private:
    std::string getStructVarName() const;
    std::string getOverflowVarName() const;
    std::string getHTVarName() const;
  };

  class OCLLinearProbingHashTableGenerator : public HashTableGenerator {
   public:
    std::string generateCodeDeclareHashTable();

    std::string generateCodeInitHashTable(bool create_new,
                                          const std::string& num_elements = "");

    std::string generateCodeCleanupHashTable() {
      COGADB_FATAL_ERROR("IMPLEMENT", "");
    }

    std::string generateIdentifierCleanupHandler();

    std::string generateCodeInsertIntoHashTable(uint32_t loopVar = 1);

    ProbeHashTableCode generateCodeProbeHashTable(
        const AttributeReference& probe_attr, uint32_t loopVar = 1);

    std::string generateCodeHTEntryAccess();

    std::string generateCodeForHeaderAndTypesBlock();

    std::string getHashTableCType();

    std::map<std::string, std::string> getBuildExtraInputVariables();

    std::map<std::string, std::string> getProbeExtraInputVariables();

    std::map<std::string, std::string> getBuildExtraOutputVariables();

    std::string getHTVarNameForSystemHT() const;

    // createprivate:
    OCLLinearProbingHashTableGenerator(
        const AttributeReference& build_attr,
        std::vector<StructFieldPtr>& payload_fields,
        const std::string& identifier)
        : HashTableGenerator(build_attr, payload_fields, identifier) {}

    OCLLinearProbingHashTableGenerator(const AttributeReference& build_attr,
                                       const std::string& identifier)
        : HashTableGenerator(build_attr, identifier) {}

   private:
    std::string getStructVarName() const;
    std::string getHTVarName() const;
  };
}

#endif /* HASH_TABLE_GENERATOR_HPP */
