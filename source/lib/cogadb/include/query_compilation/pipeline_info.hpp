/*
 * File:   pipeline_info.hpp
 * Author: sebastian
 *
 * Created on 29. Dezember 2015, 22:45
 */

#ifndef PIPELINE_INFO_HPP
#define PIPELINE_INFO_HPP

#include <boost/shared_ptr.hpp>
#include <core/global_definitions.hpp>

namespace CoGaDB {

  struct GroupByAggregateParam;
  typedef boost::shared_ptr<GroupByAggregateParam> GroupByAggregateParamPtr;

  class BaseTable;
  typedef boost::shared_ptr<BaseTable> TablePtr;

  class PipelineInfo;
  typedef boost::shared_ptr<PipelineInfo> PipelineInfoPtr;

  class State;
  typedef boost::shared_ptr<State> StatePtr;

  class PipelineInfo {
   public:
    PipelineInfo();
    const GroupByAggregateParamPtr getGroupByAggregateParam();
    PipelineEndType getPipelineType() const;
    const TablePtr getSourceTable();
    const StatePtr getGlobalState();

    void setGroupByAggregateParam(GroupByAggregateParamPtr);
    void setPipelineType(PipelineEndType);
    void setSourceTable(TablePtr);
    void setGlobalState(StatePtr);

   private:
    GroupByAggregateParamPtr groupby_param;
    PipelineEndType pipe_type;
    TablePtr source_table;
    StatePtr global_state;
  };

}  // end namespace CoGaDB

#endif /* PIPELINE_INFO_HPP */
