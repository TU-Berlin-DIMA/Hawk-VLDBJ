/*
 * File:   json_parser.hpp
 * Author: sebastian
 *
 * Created on 2. Januar 2016, 19:22
 */

#ifndef JSON_PARSER_HPP
#define JSON_PARSER_HPP

#include <query_processing/definitions.hpp>

#include <rapidjson/document.h>

#ifdef JSON_VALIDATION_ENABLED
#include <rapidjson/schema.h>
#endif

#include <stdexcept>

namespace CoGaDB {

  class Client;
  typedef boost::shared_ptr<Client> ClientPtr;

  const std::string readFileContent(const std::string& path_to_file);
  const query_processing::LogicalQueryPlanPtr import_query_from_json(
      const std::string& path_to_file, ClientPtr client);
  const std::pair<bool, TablePtr> load_and_execute_query_from_json(
      const std::string& path_to_file, ClientPtr client);

  struct OutputAttribute {
    OutputAttribute(const AttributeType& field_type,
                    const std::string& field_name,
                    const std::string& attribute_name);
    /* the internal variable name */
    std::string field_name;
    /* the internal variable type */
    AttributeType field_type;
    /* the name in the result table */
    std::string attribute_name;
  };
  typedef boost::shared_ptr<OutputAttribute> OutputAttributePtr;

  //
  // Exceptions
  //
  struct json_file_not_found_exception : std::runtime_error {
    using runtime_error::runtime_error;
  };

  struct json_parsing_exception : std::runtime_error {
    using runtime_error::runtime_error;
  };

  struct json_plan_import_exception : std::runtime_error {
    using runtime_error::runtime_error;
  };

  struct json_invalid_plan_document_exception : std::runtime_error {
    using runtime_error::runtime_error;
  };

  //
  // Logical query plan import from JSON files
  //
  /**
   * \brief Load and parse JSON file from the filesystem
   * \throws json_file_not_found_exception
   * \throws json_parsing_exception
   */
  ::rapidjson::Document load_json_from_file(const std::string& path_to_file);

#ifdef JSON_VALIDATION_ENABLED
  /**
   * \brief Load a JSON schema document from the filesystem
   */
  ::rapidjson::SchemaDocument load_schema_from_file(
      const std::string& path_to_file);

  /**
   * \brief Validate a JSON document using a given schema
   */
  void validate_jsondocument(const ::rapidjson::Document& document,
                             const ::rapidjson::SchemaDocument& schema);
#endif  // JSON_VALIDATION_ENABLED

}  // end namespace CoGaDB

#endif /* JSON_PARSER_HPP */
