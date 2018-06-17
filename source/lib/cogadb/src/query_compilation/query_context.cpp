
#include <core/attribute_reference.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/pipeline.hpp>
#include <query_compilation/query_context.hpp>
#include <set>

namespace CoGaDB {

const QueryContextPtr createQueryContext(PipelineType pipe_type) {
  return QueryContextPtr(new QueryContext(pipe_type));
}

QueryContext::QueryContext(PipelineType _pipe_type)
    : projection_list(),
      accessed_columns(),
      referenced_attributes_from_other_pipelines(),
      computed_attributes(),
      unresolved_projection_columns(),
      rename_map_(),
      pipe_type(_pipe_type),
      compilation_time_in_s(0),
      host_only_compilation_time_in_s(0),
      kernel_only_compilation_time_in_s(0),
      execution_time_in_s(0),
      is_original_context(false),
      parent_context() {
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");
  if (debug_code_generator) {
    std::cout << "[DEBUG]: Create New Pipeline (Type: " << (int)pipe_type
              << ", id: " << (void*)this << ")" << std::endl;
  }
}

QueryContext::~QueryContext() {
  bool debug_code_generator =
      VariableManager::instance().getVariableValueBoolean(
          "debug_code_generator");
  if (debug_code_generator) {
    std::cout << "[DEBUG]: Close Pipeline (Type: " << (int)pipe_type
              << ", id: " << (void*)this << ")" << std::endl;
  }
}

void QueryContext::addAccessedColumn(const std::string& column_name) {
  accessed_columns.push_back(column_name);
}

void QueryContext::addColumnToProjectionList(const std::string& column_name) {
  AttributeReferencePtr attr = getAttributeFromColumnIdentifier(column_name);
  if (!attr) {
    this->addUnresolvedProjectionAttribute(column_name);
  } else {
    this->addColumnToProjectionList(attr);
  }
}

void QueryContext::addColumnToProjectionList(AttributeReferencePtr attr) {
  assert(attr != NULL);
  projection_list.push_back(attr);
}

void QueryContext::clearProjectionList() { projection_list.clear(); }

void QueryContext::addAttributeRename(const std::string& column_name,
                                      const std::string& new_column_name) {
  rename_map_.insert(std::make_pair(column_name, new_column_name));
}

void QueryContext::addReferencedAttributeFromOtherPipeline(
    const AttributeReferencePtr attr) {
  referenced_attributes_from_other_pipelines.push_back(attr);
}

void QueryContext::addComputedAttribute(const std::string& column_name,
                                        const AttributeReferencePtr attr) {
  assert(attr != NULL);
  if (computed_attributes.find(column_name) == computed_attributes.end()) {
    computed_attributes.insert(std::make_pair(column_name, attr));
  } else {
    COGADB_FATAL_ERROR("Computed attribute exists already!", "");
  }
}

void QueryContext::addUnresolvedProjectionAttribute(
    const std::string& column_name) {
  unresolved_projection_columns.insert(column_name);
}

const std::set<std::string> QueryContext::getUnresolvedProjectionAttributes() {
  return unresolved_projection_columns;
}

const AttributeReferencePtr QueryContext::getComputedAttribute(
    const std::string& column_name) {
  if (computed_attributes.find(column_name) != computed_attributes.end()) {
    return computed_attributes[column_name];
  }
  //        ComputedAttributes::const_iterator cit;
  //        for(cit=computed_attributes.begin();cit!=computed_attributes.end();++cit){
  //            std::cout << "Computed Attribute Rename: " <<
  //            cit->second->getResultAttributeName() << std::endl;
  //            if(cit->second->getResultAttributeName()==column_name){
  //                return cit->second;
  //            }
  //        }
  return AttributeReferencePtr();
}

bool isEquivalent(AttributeReferencePtr lhs, AttributeReferencePtr rhs) {
  if (!lhs || !rhs) return false;
  if (lhs->getUnversionedAttributeName() ==
          rhs->getUnversionedAttributeName() &&
      lhs->getVersion() == rhs->getVersion()) {
    return true;
  }
  return false;
}

const AttributeReferencePtr QueryContext::getAttributeFromOtherPipelineByName(
    const std::string& name) {
  AttributeReferencePtr attr = getAttributeFromColumnIdentifier(name);
  if (!attr) {
    COGADB_FATAL_ERROR("Could not find column '" << name << "'!", "");
  }
  for (size_t i = 0; i < referenced_attributes_from_other_pipelines.size();
       ++i) {
    if (isEquivalent(attr, referenced_attributes_from_other_pipelines[i])) {
      return referenced_attributes_from_other_pipelines[i];
    }
  }
  return AttributeReferencePtr();
}

PipelineType QueryContext::getPipelineType() const { return pipe_type; }

const std::vector<AttributeReferencePtr> QueryContext::getProjectionList()
    const {
  return projection_list;
}

const std::vector<std::string> QueryContext::getAccessedColumns() const {
  return accessed_columns;
}

const std::vector<AttributeReferencePtr>
QueryContext::getReferencedAttributeFromOtherPipelines() const {
  return this->referenced_attributes_from_other_pipelines;
}

const std::map<std::string, std::string> QueryContext::getRenameMap() const {
  return this->rename_map_;
}

void QueryContext::fetchInformationFromParentContext(
    QueryContextPtr parent_context) {
  this->parent_context = parent_context;
  /* fetch referenced columns from parent query context */
  std::vector<std::string> accessed_columns =
      parent_context->getAccessedColumns();
  for (size_t i = 0; i < accessed_columns.size(); ++i) {
    this->addAccessedColumn(accessed_columns[i]);
    this->addColumnToProjectionList(accessed_columns[i]);
  }
  /* fetch projection list from parent query context */
  std::vector<AttributeReferencePtr> projected_columns =
      parent_context->getProjectionList();
  for (size_t i = 0; i < projected_columns.size(); ++i) {
    this->addColumnToProjectionList(projected_columns[i]);
  }
}

void QueryContext::updateStatistics(PipelinePtr pipeline) {
  if (!pipeline) return;
  //          static int counter=0;
  //          counter++;
  //          std::cout << "Update Stats: " << counter << std::endl;
  //          std::cout << "Current Compile Time: " <<
  //          this->compilation_time_in_s
  //          << std::endl;
  //          std::cout << "Current Execution Time: " <<
  //          this->execution_time_in_s
  //          << std::endl;
  if (VariableManager::instance().getVariableValueBoolean(
          "code_gen.opt.enable_profiling")) {
    std::cout << "Compile Time (Host+Kernel): " << pipeline->getCompileTimeSec()
              << "s" << std::endl;
    std::cout << "Compile Time (Host): " << pipeline->getHostCompileTimeSec()
              << "s" << std::endl;
    std::cout << "Compile Time (Kernel): "
              << pipeline->getKernelCompileTimeSec() << "s" << std::endl;
    if (pipeline->getExecutionTimeSec() != 0) {
      std::cout << "Execution Time: " << pipeline->getExecutionTimeSec() << "s"
                << std::endl;
    }
  }
  this->compilation_time_in_s += pipeline->getCompileTimeSec();
  this->host_only_compilation_time_in_s += pipeline->getHostCompileTimeSec();
  this->kernel_only_compilation_time_in_s +=
      pipeline->getKernelCompileTimeSec();
  this->execution_time_in_s += pipeline->getExecutionTimeSec();
}

void QueryContext::updateStatistics(QueryContextPtr context) {
  this->compilation_time_in_s += context->getCompileTimeSec();
  this->host_only_compilation_time_in_s += context->getHostCompileTimeSec();
  this->kernel_only_compilation_time_in_s += context->getKernelCompileTimeSec();
  this->execution_time_in_s += context->getExecutionTimeSec();
}

void QueryContext::addExecutionTime(double time_in_seconds) {
  this->execution_time_in_s += time_in_seconds;
  if (VariableManager::instance().getVariableValueBoolean(
          "code_gen.opt.enable_profiling")) {
    if (time_in_seconds != 0) {
      std::cout << "Execution Time: " << time_in_seconds << "s" << std::endl;
    }
  }
}

double QueryContext::getCompileTimeSec() const {
  return this->compilation_time_in_s;
}

double QueryContext::getHostCompileTimeSec() const {
  return this->host_only_compilation_time_in_s;
}
double QueryContext::getKernelCompileTimeSec() const {
  return this->kernel_only_compilation_time_in_s;
}

double QueryContext::getExecutionTimeSec() const {
  return this->execution_time_in_s;
}

void QueryContext::markAsOriginalContext() { is_original_context = true; }
bool QueryContext::isOriginalContext() const { return is_original_context; }

const QueryContextPtr QueryContext::getParentContext() const {
  return parent_context;
}

void QueryContext::print(std::ostream& out) {
  out << " === Query Context === " << std::endl;

  out << "Accessed Columns: " << std::endl;
  for (size_t i = 0; i < accessed_columns.size(); ++i) {
    std::cout << "\t" << accessed_columns[i] << std::endl;
  }
  out << "Projection List: " << std::endl;
  for (size_t i = 0; i < projection_list.size(); ++i) {
    std::cout << "\t"
              << createFullyQualifiedColumnIdentifier(projection_list[i])
              << std::endl;
  }
  out << "Referenced Columns from other Pipelines: " << std::endl;
  for (size_t i = 0; i < referenced_attributes_from_other_pipelines.size();
       ++i) {
    std::cout << "\t" << createFullyQualifiedColumnIdentifier(
                             referenced_attributes_from_other_pipelines[i])
              << " AS "
              << referenced_attributes_from_other_pipelines[i]
                     ->getResultAttributeName()
              << std::endl;
  }

  out << "Original Context: " << is_original_context << std::endl;
  out << " === Query Context End === " << std::endl;
}

}  // end namespace CoGaDB
