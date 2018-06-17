
#include <persistence/storage_manager.hpp>
#include <query_compilation/aggregate_specification.hpp>
#include <query_compilation/predicate_expression.hpp>
#include <util/attribute_reference_handling.hpp>

namespace CoGaDB {

void replaceAttributeTablePointersWithScannedAttributeTablePointers(
    const ScanParam& scanned_attributes, AttributeReference& attr) {
  for (auto scanned_attr : scanned_attributes) {
    if (CoGaDB::toString(attr) == CoGaDB::toString(scanned_attr)) {
      /* update attribute references in predicate expression,
     to refer to new intermediate result table */
      attr.setTable(scanned_attr.getTable());
    }
  }
}

void replaceAttributeTablePointersWithScannedAttributeTablePointers(
    const ScanParam& scanned_attributes,
    std::vector<AttributeReference>& attribute_references) {
  for (auto& attr : attribute_references) {
    //   std::cout << "[REPLACER]: Check " << CoGaDB::toString(attr) <<
    //   std::endl;
    for (auto& scanned_attr : scanned_attributes) {
      //    std::cout << "   [REPLACER]: Scanned Attribute: " <<
      //    CoGaDB::toString(scanned_attr); // << std::endl;
      //    if(isPersistent(scanned_attr.getTable() )){
      //      std::cout << " (Persistent)"  << std::endl;
      //    }else{
      //      std::cout <<  " (Intermediate Result TablePtr: " <<
      //      scanned_attr.getTable() << ")" << std::endl;
      //    }
      if (CoGaDB::toString(attr) == CoGaDB::toString(scanned_attr)) {
        //        std::cout << "[REPLACER]: Replace Table Pointer for Attribute
        //        "
        //                  << CoGaDB::toString(attr) << " "
        //                  << (void*) attr.getTable().get() << "->"
        //                  << (void*) scanned_attr.getTable().get() <<
        //                  std::endl;
        /* update attribute references in predicate expression,
       to refer to new intermediate result table */
        attr.setTable(scanned_attr.getTable());
      }
    }
  }
}

void replaceAttributeTablePointersWithScannedAttributeTablePointers(
    const ScanParam& scanned_attributes,
    std::vector<AttributeReferencePtr>& attribute_references) {
  for (auto attr : attribute_references) {
    for (auto scanned_attr : scanned_attributes) {
      if (CoGaDB::toString(*attr) == CoGaDB::toString(scanned_attr)) {
        /* update attribute references in predicate expression,
       to refer to new intermediate result table */
        attr->setTable(scanned_attr.getTable());
      }
    }
  }
}

void replaceAttributeTablePointersWithScannedAttributeTablePointers(
    const ScanParam& scanned_attributes, PredicateExpressionPtr pred_expr) {
  if (pred_expr)
    pred_expr->replaceTablePointerInAttributeReferences(scanned_attributes);
}

void replaceAttributeTablePointersWithScannedAttributeTablePointers(
    const ScanParam& scanned_attributes, AggregateSpecifications& agg_specs) {
  for (auto& agg_spec : agg_specs) {
    agg_spec->replaceTablePointerInAttributeReferences(scanned_attributes);
  }
}

}  // end namespace CoGaDB
