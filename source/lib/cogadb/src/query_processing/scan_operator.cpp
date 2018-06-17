
#include <query_processing/scan_operator.hpp>
#include <util/hardware_detector.hpp>

#include <core/attribute_reference.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>

namespace CoGaDB {

namespace query_processing {
// Map_Init_Function
// init_function_scan_operator=physical_operator::map_init_function_scan_operator;

namespace physical_operator {

TypedOperatorPtr create_scan_operator(TypedLogicalNode& logical_node,
                                      const hype::SchedulingDecision& sched_dec,
                                      TypedOperatorPtr, TypedOperatorPtr) {
  logical_operator::Logical_Scan& log_sort_ref =
      static_cast<logical_operator::Logical_Scan&>(logical_node);
  return TypedOperatorPtr(
      new scan_operator(sched_dec, log_sort_ref.getTablePtr()));
}

Physical_Operator_Map_Ptr map_init_function_scan_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function! (SCAN OPERATION)" << std::endl;
// hype::Scheduler::instance().addAlgorithm("SCAN","TABLE_SCAN",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "TABLE_SCAN", "SCAN", hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "TABLE_SCAN", "SCAN", hype::Least_Squares_1D, hype::Periodic);
#endif

  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu,
                                               dev_specs[i]);
    }
  }

  map["TABLE_SCAN"] = create_scan_operator;
  // map["GPU_Algorithm"]=create_GPU_SORT_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

scan_operator::scan_operator(const hype::SchedulingDecision& sched_dec,
                             TablePtr table)
    : UnaryOperator<TablePtr, TablePtr>(sched_dec, TypedOperatorPtr()),
      table_(table)

{
  this->result_ = table_;
}
bool scan_operator::execute() {
  if (!quiet && verbose && debug) std::cout << "Execute Scan" << std::endl;
  // const TablePtr sort(TablePtr table, const std::string& column_name,
  // SortOrder order=ASCENDING, MaterializationStatus mat_stat=MATERIALIZE,
  // ComputeDevice comp_dev=CPU);
  this->result_ = table_;
  if (this->result_) {
    setResultSize(this->result_->getNumberofRows());
    return true;
  } else
    return false;
}

void scan_operator::releaseInputData() {
  // do nothing
}

scan_operator::~scan_operator() {}

}  // end namespace physical_operator

namespace logical_operator {
Logical_Scan::Logical_Scan(std::string table_name, uint32_t version)
    : TypedNode_Impl<TablePtr,
                     physical_operator::map_init_function_scan_operator>(),
      table_(getTablebyName(table_name)),
      version_(version) {
  if (table_ == NULL) {
    COGADB_FATAL_ERROR(
        std::string("Table ") + table_name + " not found in global Table List!",
        "");
  }
}
Logical_Scan::Logical_Scan(TablePtr table, uint32_t version)
    : TypedNode_Impl<TablePtr,
                     physical_operator::map_init_function_scan_operator>(),
      table_(table),
      version_(version) {
  assert(table_ != NULL);
}
unsigned int Logical_Scan::getOutputResultSize() const {
  // TablePtr table_ptr = getTablebyName(table_name_);
  // assert(table_ptr != NULL);

  return table_->getNumberofRows();
}

double Logical_Scan::getCalculatedSelectivity() const { return 1.0; }
std::string Logical_Scan::getOperationName() const { return "SCAN"; }

std::string Logical_Scan::toString(bool verbose) const {
  std::stringstream str;
  str << "SCAN " << table_->getName() << " v" << version_;
  return str.str();
}

const std::string& Logical_Scan::getTableName() { return table_->getName(); }

const TablePtr Logical_Scan::getTablePtr() { return table_; }

const hype::Tuple Logical_Scan::getFeatureVector() const {
  hype::Tuple t;
  t.push_back(getOutputResultSize());
  return t;
}

//                        void
//                        retrieveScannedAndProjectedAttributesFromScannedTable(CodeGeneratorPtr
//                        code_gen,
//                                QueryContextPtr context,
//                                TablePtr scanned_table,
//                                uint32_t version){
//
//                            bool debug_code_generator =
//                            VariableManager::instance()
//                                .getVariableValueBoolean("debug_code_generator");
//
//                            std::map<std::string,std::string> rename_map =
//                            context->getRenameMap();
//                            std::map<std::string,std::string>::const_iterator
//                            rename_cit;
//
//                            std::vector<AttributeReferencePtr> proj_cols =
//                            context->getProjectionList();
//                            if(!proj_cols.empty()){
//                                for(size_t i=0;i<proj_cols.size();++i){
//                                    std::string new_name =
//                                    createFullyQualifiedColumnIdentifier(proj_cols[i]);
//                                    if(scanned_table->hasColumn(new_name)){
//                                        AttributeReference attr(scanned_table,
//                                                proj_cols[i]->getUnversionedAttributeName(),
//                                                proj_cols[i]->getResultAttributeName(),
//                                                version);
//                                        code_gen->addAttributeProjection(attr);
//                                        if(debug_code_generator){
//                                            std::cout << "[SCAN]: Add
//                                            projection attribute: "
//                                                      <<
//                                                      attr.getVersionedTableName()
//                                                      << "."
//                                                      <<
//                                                      attr.getVersionedAttributeName()
//                                                      << "->"
//                                                      <<
//                                                      attr.getResultAttributeName()
//                                                      << std::endl;
//                                        }
//                                    }else{
//                                        if(debug_code_generator){
//                                            std::cout << "[SCAN]: Could not
//                                            resolve "
//                                                    << "projection attribute
//                                                    '"
//                                                    <<
//                                                    createFullyQualifiedColumnIdentifier(proj_cols[i])
//                                                    << "'" << std::endl;
//                                        }
//                                        context->addUnresolvedProjectionAttribute(createFullyQualifiedColumnIdentifier(proj_cols[i]));
//                                    }
//                                }
//                            }else{
//                                /* the projection list is empty, which means
//                                that
//                                 * we have no projection as parent in the
//                                 pipeline.
//                                 * Thus we have to add all attributes in the
//                                 input
//                                 * table to the projection list.
//                                 */
//                                std::vector<ColumnProperties> col_props =
//                                scanned_table->getPropertiesOfColumns();
//                                for(size_t i=0;i<col_props.size();++i){
////                                    std::stringstream new_name;
////                                    new_name << this->table_->getName() <<
///"." << col_props[i].name << "." << this->version_;
//                                    std::string new_name =
//                                    createFullyQualifiedColumnIdentifier(scanned_table->getName(),
//                                            col_props[i].name, version);
//                                    rename_cit=rename_map.find(col_props[i].name);
//                                    if(rename_cit!=rename_map.end()){
//                                        new_name = rename_cit->second;
//                                        if(debug_code_generator){
//                                            std::cout << "[SCAN]: Rename
//                                            attribute: "
//                                                      << col_props[i].name <<
//                                                      "->"
//                                                      << new_name
//                                                      << std::endl;
//                                        }
//                                    }
//                                    AttributeReference attr(scanned_table,
//                                    col_props[i].name, new_name, version);
//                                    code_gen->addAttributeProjection(attr);
//                                }
//                            }
//                            /* add all accessed columns we have found so far
//                            to the pipeline,
//                             * if they originate from here */
//                            std::vector<std::string> accessed_columns
//                                    = context->getAccessedColumns();
//                            for(size_t i=0;i<accessed_columns.size();++i){
//                                if(scanned_table->hasColumn(accessed_columns[i])){
//                                    std::string new_name =
//                                    accessed_columns[i];
//                                    rename_cit=rename_map.find(accessed_columns[i]);
//                                    if(rename_cit!=rename_map.end()){
//                                        new_name = rename_cit->second;
//                                        if(debug_code_generator){
//                                            std::cout << "[SCAN]: Rename
//                                            attribute: "
//                                                      << accessed_columns[i]
//                                                      << "->"
//                                                      << new_name
//                                                      << std::endl;
//                                        }
//                                    }
//                                    AttributeReferencePtr attr =
//                                    createInputAttribute(
//                                            scanned_table,
//                                            accessed_columns[i], new_name,
//                                            version);
//                                    assert(attr!=NULL);
//                                    if(debug_code_generator){
//                                        std::cout << "[SCAN]: Add accessed
//                                        attribute: "
//                                                  <<
//                                                  attr->getVersionedTableName()
//                                                  << "."
//                                                  <<
//                                                  attr->getVersionedAttributeName()
//                                                  << std::endl;
//                                    }
//                                    code_gen->addToScannedAttributes(*attr);
//                                }
//                            }
//                        }
//

void Logical_Scan::produce_impl(CodeGeneratorPtr code_gen,
                                QueryContextPtr context) {
  //                            bool debug_code_generator =
  //                            VariableManager::instance()
  //                                .getVariableValueBoolean("debug_code_generator");
  //
  //                            std::vector<std::string> proj_cols =
  //                            context->getProjectionList();
  //                            if(!proj_cols.empty()){
  //                                for(size_t i=0;i<proj_cols.size();++i){
  //                                    if(this->table_->hasColumn(proj_cols[i])){
  //                                        AttributeReference
  //                                        attr(this->table_, proj_cols[i],
  //                                        proj_cols[i], this->version_);
  //                                        code_gen->addAttributeProjection(attr);
  //                                        if(debug_code_generator){
  //                                            std::cout << "[SCAN]: Add
  //                                            projection attribute: "
  //                                                      <<
  //                                                      attr.getVersionedTableName()
  //                                                      << "."
  //                                                      <<
  //                                                      attr.getVersionedAttributeName()
  //                                                      << std::endl;
  //                                        }
  //                                    }else{
  //                                        if(debug_code_generator){
  //                                            std::cout << "[SCAN]: Could not
  //                                            resolve "
  //                                                    << "projection attribute
  //                                                    '"
  //                                                    <<  proj_cols[i] << "'"
  //                                                    << std::endl;
  //                                        }
  //                                        context->addUnresolvedProjectionAttribute(proj_cols[i]);
  //                                    }
  //                                }
  //                            }else{
  //                                /* the projection list is empty, which means
  //                                that
  //                                 * we have no projection as parent in the
  //                                 pipeline.
  //                                 * Thus we have to add all attributes in the
  //                                 input
  //                                 * table to the projection list.
  //                                 */
  //                                std::vector<ColumnProperties> col_props =
  //                                this->table_->getPropertiesOfColumns();
  //                                for(size_t i=0;i<col_props.size();++i){
  ////                                    std::stringstream new_name;
  ////                                    new_name << this->table_->getName() <<
  ///"." << col_props[i].name << "." << this->version_;
  //                                    std::string new_name =
  //                                    createFullyQualifiedColumnIdentifier(this->table_->getName(),
  //                                            col_props[i].name,
  //                                            this->version_);
  //                                    AttributeReference attr(this->table_,
  //                                    col_props[i].name, new_name,
  //                                    this->version_);
  //                                    code_gen->addAttributeProjection(attr);
  //                                }
  //                            }
  //                            /* add all accessed columns we have found so far
  //                            to the pipeline,
  //                             * if they originate from here */
  //                            std::vector<std::string> accessed_columns
  //                                    = context->getAccessedColumns();
  //                            for(size_t i=0;i<accessed_columns.size();++i){
  //                                if(this->table_->hasColumn(accessed_columns[i])){
  //                                    AttributeReferencePtr attr =
  //                                    createInputAttribute(
  //                                            this->table_,
  //                                            accessed_columns[i],
  //                                            accessed_columns[i],
  //                                            this->version_);
  //                                    assert(attr!=NULL);
  //                                    if(debug_code_generator){
  //                                        std::cout << "[SCAN]: Add accessed
  //                                        attribute: "
  //                                                  <<
  //                                                  attr->getVersionedTableName()
  //                                                  << "."
  //                                                  <<
  //                                                  attr->getVersionedAttributeName()
  //                                                  << std::endl;
  //                                    }
  //                                    code_gen->addToScannedAttributes(*attr);
  //                                }
  //                            }

  retrieveScannedAndProjectedAttributesFromScannedTable(
      code_gen, context, this->table_, this->version_);

  // create for loop for code_gen
  if (!code_gen->createForLoop(this->table_, this->version_)) {
    COGADB_FATAL_ERROR("Creating of for loop failed!", "");
  }
  if (parent_) {
    parent_->consume(code_gen, context);
  }
}

void Logical_Scan::consume_impl(CodeGeneratorPtr code_gen,
                                QueryContextPtr context) {
  // do nothing, as a scan is a leaf operator
}

}  // end namespace logical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
