/*
 * File:   user_defined_code.hpp
 * Author: sebastian
 *
 * Created on 12. Dezember 2015, 20:54
 */

#ifndef USER_DEFINED_CODE_HPP
#define USER_DEFINED_CODE_HPP

#include <core/attribute_reference.hpp>
#include <string>
#include <vector>

namespace CoGaDB {

  class OutputAttribute;
  typedef boost::shared_ptr<OutputAttribute> OutputAttributePtr;

  class PredicateExpression;
  typedef boost::shared_ptr<PredicateExpression> PredicateExpressionPtr;

  typedef std::vector<AttributeReference> ProjectionParam;
  typedef std::vector<AttributeReference> ScanParam;

  class Map_UDF;
  typedef boost::shared_ptr<Map_UDF> Map_UDFPtr;

  struct Map_UDF_Param;
  typedef boost::shared_ptr<Map_UDF_Param> Map_UDF_ParamPtr;

  struct UDF_Code;
  typedef boost::shared_ptr<UDF_Code> UDF_CodePtr;

  struct StructField;
  typedef boost::shared_ptr<StructField> StructFieldPtr;

  enum UDF_Type { MAP_UDF, REDUCE_UDF };

  enum Map_UDF_Type { MAP_UDF_EUCLIDIAN_DISTANCE, MAP_UDF_CUSTOM };

  class Map_UDF_Result {
   public:
    Map_UDF_Result(bool _error, std::string _generated_code,
                   std::string declared_variables,
                   std::vector<AttributeReferencePtr> _computed_attributes,
                   std::vector<AttributeReferencePtr> _scanned_attributes);
    bool return_code;
    std::string generated_code;
    std::string declared_variables;
    std::vector<AttributeReferencePtr> computed_attributes;
    std::vector<AttributeReferencePtr> scanned_attributes;
  };

  class Map_UDF {
   public:
    Map_UDF(Map_UDF_Type udf_type, const ScanParam& scanned_attrs,
            const ProjectionParam& projected_attrs,
            const std::vector<StructFieldPtr>& decl_vars);

    const std::string getDeclaredVariables() const;
    virtual const Map_UDF_Result generateCode(
        const ScanParam& scan_attributes,
        const ProjectionParam& project_attributes) = 0;
    const ScanParam getScannedAttributes() const;
    const ProjectionParam getProjectedAttributes() const;
    const Map_UDF_Type getMap_UDF_Type() const;
    virtual void replaceTablePointerInAttributeReferences(
        const ScanParam& scanned_attributes);

   protected:
    Map_UDF_Type udf_type;
    ScanParam scanned_attributes;
    ProjectionParam projected_attributes;
    std::vector<StructFieldPtr> declared_variables;
  };

  class Map_EuclidianDistance : public Map_UDF {
   public:
    Map_EuclidianDistance(
        std::vector<std::pair<AttributeReferencePtr, AttributeReferencePtr> >
            pair_wise_vector_elements);

    const Map_UDF_Result generateCode(
        const ScanParam& scan_attributes,
        const ProjectionParam& project_attributes);

   private:
    const ScanParam toScanParam(
        const std::vector<
            std::pair<AttributeReferencePtr, AttributeReferencePtr> >&);
    const ProjectionParam toProjectionParam(
        const std::vector<
            std::pair<AttributeReferencePtr, AttributeReferencePtr> >&);
    std::vector<std::pair<AttributeReferencePtr, AttributeReferencePtr> >
        pair_wise_vector_elements_;
  };

  class Generic_Map_UDF : public Map_UDF {
   public:
    Generic_Map_UDF(UDF_CodePtr udf_code,
                    const std::vector<StructFieldPtr>& decl_vars);
    const Map_UDF_Result generateCode(
        const ScanParam& scan_attributes,
        const ProjectionParam& project_attributes);
    void replaceTablePointerInAttributeReferences(
        const ScanParam& scanned_attributes);

   private:
    UDF_CodePtr udf_code;
  };

  struct Map_UDF_Param {
    Map_UDF_Param(Map_UDFPtr);
    const ScanParam getScannedAttributes() const;
    const ProjectionParam getProjectedAttributes() const;
    Map_UDFPtr map_udf;
  };

  struct UDF_Code {
    UDF_Code(const std::string& code,
             const std::vector<AttributeReferencePtr>& scanned_attributes,
             const std::vector<AttributeReferencePtr>& result_attributes,
             const std::vector<std::string>& _code_lines,
             const std::vector<OutputAttributePtr>& _output_attributes,
             UDF_Type _udf_type);
    const std::string getCode(const ScanParam& scan_attributes = ScanParam());
    std::vector<AttributeReferencePtr> scanned_attributes;
    std::vector<AttributeReferencePtr> result_attributes;

   private:
    std::string code;
    std::vector<std::string> code_lines;
    std::vector<OutputAttributePtr> output_attributes;
    UDF_Type udf_type;
  };

  const UDF_CodePtr parseUDFCode(
      const std::vector<std::string>& code_lines,
      const std::vector<OutputAttributePtr>& output_attributes,
      UDF_Type udf_type);

}  // end namespace CoGaDB

#endif /* USER_DEFINED_CODE_HPP */
