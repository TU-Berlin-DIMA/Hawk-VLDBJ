/*
 * File:   query_context.hpp
 * Author: sebastian
 *
 * Created on 1. September 2015, 15:01
 */

#ifndef QUERY_CONTEXT_HPP
#define QUERY_CONTEXT_HPP

#include <boost/shared_ptr.hpp>
#include <ios>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace CoGaDB {

  class QueryContext;
  typedef boost::shared_ptr<QueryContext> QueryContextPtr;

  class AttributeReference;
  typedef boost::shared_ptr<AttributeReference> AttributeReferencePtr;

  class Pipeline;
  typedef boost::shared_ptr<Pipeline> PipelinePtr;

  enum PipelineType {
    NORMAL_PIPELINE,
    BUILD_HASH_TABLE_PIPELINE,
    AGGREGATE_PIPELINE
  };

  const QueryContextPtr createQueryContext(PipelineType pipe = NORMAL_PIPELINE);

  class QueryContext {
   public:
    ~QueryContext();
    void addAccessedColumn(const std::string& column_name);
    void addColumnToProjectionList(const std::string& column_name);
    void addColumnToProjectionList(AttributeReferencePtr attr);
    /* resets the projection list */
    void clearProjectionList();
    void addAttributeRename(const std::string& column_name,
                            const std::string& new_column_name);
    void addReferencedAttributeFromOtherPipeline(
        const AttributeReferencePtr attr);
    void addComputedAttribute(const std::string& column_name,
                              const AttributeReferencePtr attr);
    void addUnresolvedProjectionAttribute(const std::string& column_name);

    const std::set<std::string> getUnresolvedProjectionAttributes();

    const AttributeReferencePtr getComputedAttribute(
        const std::string& column_name);

    const AttributeReferencePtr getAttributeFromOtherPipelineByName(
        const std::string& name);

    PipelineType getPipelineType() const;

    const std::vector<AttributeReferencePtr> getProjectionList() const;
    const std::vector<std::string> getAccessedColumns() const;
    const std::vector<AttributeReferencePtr>
    getReferencedAttributeFromOtherPipelines() const;
    const std::map<std::string, std::string> getRenameMap() const;

    void fetchInformationFromParentContext(QueryContextPtr parent_context);

    void updateStatistics(PipelinePtr);
    void updateStatistics(QueryContextPtr);
    void addExecutionTime(double time_in_seconds);
    double getCompileTimeSec() const;
    double getHostCompileTimeSec() const;
    double getKernelCompileTimeSec() const;
    double getExecutionTimeSec() const;

    /* we apply renaming in the last pipeline, and keep track of this via a
     * flag in the query context */
    void markAsOriginalContext();
    bool isOriginalContext() const;

    //    void setParentContext(QueryContextPtr parent_context);
    const QueryContextPtr getParentContext() const;

    void print(std::ostream& out);

   private:
    friend const QueryContextPtr createQueryContext(PipelineType);
    QueryContext(PipelineType);
    std::vector<AttributeReferencePtr> projection_list;
    std::vector<std::string> accessed_columns;
    std::vector<AttributeReferencePtr>
        referenced_attributes_from_other_pipelines;
    typedef std::map<std::string, AttributeReferencePtr> ComputedAttributes;
    ComputedAttributes computed_attributes;
    std::set<std::string> unresolved_projection_columns;
    std::map<std::string, std::string> rename_map_;
    PipelineType pipe_type;
    double compilation_time_in_s;
    double host_only_compilation_time_in_s;
    double kernel_only_compilation_time_in_s;
    double execution_time_in_s;
    bool is_original_context;
    QueryContextPtr parent_context;
  };

}  // end namespace CoGaDB

#endif /* QUERY_CONTEXT_HPP */
