
#include <query_compilation/user_defined_code.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/make_shared.hpp>
#include <parser/json_parser.hpp>
#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generators/code_generator_utils.hpp>
#include <util/attribute_reference_handling.hpp>
#include <util/code_generation.hpp>

namespace CoGaDB {

Map_UDF_Result::Map_UDF_Result(
    bool _return_code, std::string _generated_code,
    std::string _declared_variables,
    std::vector<AttributeReferencePtr> _computed_attributes,
    std::vector<AttributeReferencePtr> _scanned_attributes)
    : return_code(_return_code),
      generated_code(_generated_code),
      declared_variables(_declared_variables),
      computed_attributes(_computed_attributes),
      scanned_attributes(_scanned_attributes) {}

Map_UDF::Map_UDF(Map_UDF_Type _udf_type, const ScanParam& scanned_attrs,
                 const ProjectionParam& projected_attrs,
                 const std::vector<StructFieldPtr>& decl_vars)
    : udf_type(_udf_type),
      scanned_attributes(scanned_attrs),
      projected_attributes(projected_attrs),
      declared_variables(decl_vars) {}

const std::string Map_UDF::getDeclaredVariables() const {
  std::stringstream declare_variables_code;
  for (size_t i = 0; i < this->declared_variables.size(); ++i) {
    declare_variables_code << toCPPType(this->declared_variables[i]->field_type)
                           << " "
                           << "computed_var_"
                           << this->declared_variables[i]->field_name << ";"
                           << std::endl;
  }
  return declare_variables_code.str();
}

const ScanParam Map_UDF::getScannedAttributes() const {
  return scanned_attributes;
}

const ProjectionParam Map_UDF::getProjectedAttributes() const {
  return projected_attributes;
}

const Map_UDF_Type Map_UDF::getMap_UDF_Type() const { return udf_type; }

void Map_UDF::replaceTablePointerInAttributeReferences(
    const ScanParam& scanned_attributes) {
  CoGaDB::replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, this->scanned_attributes);
}

Map_UDF_Param::Map_UDF_Param(Map_UDFPtr udf) : map_udf(udf) {}

const ScanParam Map_UDF_Param::getScannedAttributes() const {
  return map_udf->getScannedAttributes();
}

const ProjectionParam Map_UDF_Param::getProjectedAttributes() const {
  return map_udf->getProjectedAttributes();
}

bool parseUDFCode_impl(const std::vector<std::string>& code_lines,
                       const std::vector<OutputAttributePtr>& output_attributes,
                       const UDF_Type udf_type,
                       const ScanParam& pipeline_scan_attributes,
                       std::string& udf_code,
                       std::vector<AttributeReferencePtr>& scanned_attributes,
                       std::vector<AttributeReferencePtr>& result_attributes) {
  std::stringstream code;

  for (size_t i = 0; i < code_lines.size(); i++) {
    std::string code_line = code_lines[i];  // member->value[i].GetString();

    std::vector<std::string> tokens;
    boost::split(tokens, code_line, boost::is_any_of("#"));
    /* n tokens == n-1 '#' characters -> if number of '#' characters should be
     even, we need an uneven number of tokens! */
    if (tokens.size() % 2 == 0) {
      COGADB_FATAL_ERROR(
          "Parse Error in UDF: uneven number of '#' characters in line: "
              << "'" << code_line << "'",
          "");
    }

    std::stringstream transformed_code_line;
    for (size_t j = 0; j < tokens.size(); ++j) {
      if (j % 2 == 1) {
        std::cout << "Attribute Reference: " << tokens[j] << std::endl;

        std::string attribute_type_identifier;
        std::string attribute_name;
        uint32_t version = 1;
        if (!parseColumnIndentifierName(tokens[j], attribute_type_identifier,
                                        attribute_name, version)) {
          COGADB_FATAL_ERROR("Invalid attribute reference in UDF Code: '"
                                 << tokens[j] << "'",
                             "");
        }
        std::cout << "Attribute Type Identifier: " << attribute_type_identifier
                  << std::endl;
        std::cout << "Attribute Name: " << attribute_name << std::endl;
        std::cout << "Attribute Version: " << version << std::endl;

        if (attribute_type_identifier == "<HASH_ENTRY>") {
          // do nothing
          transformed_code_line << boost::to_upper_copy(tokens[j]);
        } else if (attribute_type_identifier == "<OUT>") {
          OutputAttributePtr output_attr;
          for (size_t k = 0; k < output_attributes.size(); ++k) {
            if (output_attributes[i]->attribute_name == attribute_name) {
              output_attr = output_attributes[i];
            }
          }
          assert(output_attr != NULL);
          AttributeReferencePtr attr;
          attr = boost::make_shared<AttributeReference>(
              output_attr->field_name,  // output_attr->attribute_name,
              output_attr->field_type, output_attr->attribute_name);
          result_attributes.push_back(attr);

          if (udf_type == MAP_UDF) {
            transformed_code_line << getElementAccessExpression(*attr);
          } else if (udf_type == REDUCE_UDF) {
            transformed_code_line << getResultArrayVarName(*attr)
                                  << "[current_result_size]";
          } else {
            COGADB_FATAL_ERROR("", "");
          }

        } else if (attribute_type_identifier == "<COMPUTED>") {
          COGADB_FATAL_ERROR("Unhandled case!", "");
        } else {
          attribute_name =
              attribute_type_identifier + std::string(".") + attribute_name;
          TablePtr table = getTablebyName(attribute_type_identifier);
          assert(table != NULL);
          assert(table->hasColumn(attribute_name));
          AttributeReferencePtr attr;
          attr = boost::make_shared<AttributeReference>(
              table, attribute_name, attribute_name, version);
          /* update pointer to input table in attribute reference
           * using scanned attributes from pipeline */
          CoGaDB::
              replaceAttributeTablePointersWithScannedAttributeTablePointers(
                  pipeline_scan_attributes, *attr);
          scanned_attributes.push_back(attr);
          transformed_code_line << getElementAccessExpression(*attr);
        }
      } else {
        transformed_code_line << tokens[j];
      }
    }
    code << transformed_code_line.str();
    if (i + 1 < code_lines.size()) code << std::endl;
  }
  udf_code = code.str();
  return true;
}

const UDF_CodePtr parseUDFCode(
    const std::vector<std::string>& code_lines,
    const std::vector<OutputAttributePtr>& output_attributes,
    UDF_Type udf_type) {
  std::string udf_code;
  std::vector<AttributeReferencePtr> scanned_attributes;
  std::vector<AttributeReferencePtr> result_attributes;
  /* we do not have this information yet, but we need to
   * pass a  reference as parameter
   */
  ScanParam placeholder;

  bool ret;
  ret = parseUDFCode_impl(code_lines, output_attributes, udf_type, placeholder,
                          udf_code, scanned_attributes, result_attributes);
  if (!ret) {
    COGADB_FATAL_ERROR("Parsing Failed!", "");
    return UDF_CodePtr();
  }
  UDF_CodePtr udf_code_ptr(new UDF_Code("", scanned_attributes,
                                        result_attributes, code_lines,
                                        output_attributes, udf_type));
  return udf_code_ptr;
}

const std::string UDF_Code::getCode(const ScanParam& scan_attributes) {
  scanned_attributes.clear();
  result_attributes.clear();
  std::string udf_code;
  bool ret;
  ret = parseUDFCode_impl(code_lines, output_attributes, udf_type,
                          scan_attributes, udf_code, scanned_attributes,
                          result_attributes);
  if (ret) {
    return udf_code;
  } else {
    return std::string();
  }
}

Map_EuclidianDistance::Map_EuclidianDistance(
    std::vector<std::pair<AttributeReferencePtr, AttributeReferencePtr> >
        pair_wise_vector_elements)
    : Map_UDF(MAP_UDF_EUCLIDIAN_DISTANCE,
              toScanParam(pair_wise_vector_elements),
              toProjectionParam(pair_wise_vector_elements),
              std::vector<StructFieldPtr>()),
      pair_wise_vector_elements_(pair_wise_vector_elements) {}

const AttributeReferencePtr createComputedAttribute(
    Map_UDF_Type map_udf_type,
    std::vector<std::pair<AttributeReferencePtr, AttributeReferencePtr> >
        pair_wise_vector_elements) {
  std::stringstream new_name;
  new_name << "euclidian_distance_";
  //        getVersionedAttributeName()
  for (size_t i = 0; i < pair_wise_vector_elements.size(); ++i) {
    new_name
        << pair_wise_vector_elements[i].first->getVersionedAttributeName()
        << "_"
        << pair_wise_vector_elements[i].second->getVersionedAttributeName();
    //                     << std::endl;
  }

  return boost::make_shared<AttributeReference>(new_name.str(), DOUBLE,
                                                new_name.str(), 1);
}

const Map_UDF_Result Map_EuclidianDistance::generateCode(
    const ScanParam& scan_attributes,
    const ProjectionParam& project_attributes) {
  AttributeReferencePtr distance_computed_attr =
      createComputedAttribute(udf_type, pair_wise_vector_elements_);
  std::stringstream code;

  code << "double " << getElementAccessExpression(*distance_computed_attr)
       << " = sqrt(";
  for (size_t i = 0; i < pair_wise_vector_elements_.size(); ++i) {
    code << "pow("
         << getElementAccessExpression(*pair_wise_vector_elements_[i].first)
         << "-"
         << getElementAccessExpression(*pair_wise_vector_elements_[i].second)
         << ",2)";
    if (i + 1 < pair_wise_vector_elements_.size()) {
      code << "+";
    }
  }
  code << ");";
  //        code  << std::endl << "std::cout << " <<
  //        getElementAccessExpression(*distance_computed_attr) << " <<
  //        std::endl;" << std::endl;
  std::vector<AttributeReferencePtr> result;
  result.push_back(distance_computed_attr);

  std::vector<AttributeReferencePtr> scanned_attributes;
  auto input_attributes = this->getScannedAttributes();
  for (auto attr : input_attributes) {
    scanned_attributes.push_back(
        AttributeReferencePtr(new AttributeReference(attr)));
  }

  return Map_UDF_Result(true, code.str(), getDeclaredVariables(), result,
                        scanned_attributes);
}

const ScanParam Map_EuclidianDistance::toScanParam(
    const std::vector<std::pair<AttributeReferencePtr, AttributeReferencePtr> >&
        pair_wise_vector_elements) {
  //        COGADB_FATAL_ERROR("Called unimplemented function!","");

  ScanParam param;
  for (size_t i = 0; i < pair_wise_vector_elements.size(); ++i) {
    param.push_back(*pair_wise_vector_elements[i].first);
    param.push_back(*pair_wise_vector_elements[i].second);
  }
  return param;
}

const ProjectionParam Map_EuclidianDistance::toProjectionParam(
    const std::vector<std::pair<AttributeReferencePtr, AttributeReferencePtr> >&
        pair_wise_vector_elements) {
  //        COGADB_FATAL_ERROR("Called unimplemented function!","");
  ProjectionParam param;
  //        param.push_back();
  return param;
}

Generic_Map_UDF::Generic_Map_UDF(UDF_CodePtr _udf_code,
                                 const std::vector<StructFieldPtr>& decl_vars)
    : Map_UDF(MAP_UDF_CUSTOM, ScanParam(), ProjectionParam(), decl_vars),
      udf_code(_udf_code) {
  for (size_t i = 0; i < udf_code->scanned_attributes.size(); ++i) {
    this->scanned_attributes.push_back(*udf_code->scanned_attributes[i]);
  }
  for (size_t i = 0; i < udf_code->result_attributes.size(); ++i) {
    this->projected_attributes.push_back(*udf_code->result_attributes[i]);
  }
}

const Map_UDF_Result Generic_Map_UDF::generateCode(
    const ScanParam& scan_attributes,
    const ProjectionParam& project_attributes) {
  return Map_UDF_Result(true, udf_code->getCode(scan_attributes),
                        getDeclaredVariables(), udf_code->result_attributes,
                        udf_code->scanned_attributes);
}

void Generic_Map_UDF::replaceTablePointerInAttributeReferences(
    const ScanParam& scanned_attributes) {
  CoGaDB::replaceAttributeTablePointersWithScannedAttributeTablePointers(
      scanned_attributes, udf_code->scanned_attributes);
}

UDF_Code::UDF_Code(
    const std::string& _code,
    const std::vector<AttributeReferencePtr>& _scanned_attributes,
    const std::vector<AttributeReferencePtr>& _result_attributes,
    const std::vector<std::string>& _code_lines,
    const std::vector<OutputAttributePtr>& _output_attributes,
    UDF_Type _udf_type)
    : scanned_attributes(_scanned_attributes),
      result_attributes(_result_attributes),
      code(_code),
      code_lines(_code_lines),
      output_attributes(_output_attributes),
      udf_type(_udf_type) {}

}  // end namespace CoGaDB
