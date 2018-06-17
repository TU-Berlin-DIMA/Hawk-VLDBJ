

#include <query_compilation/pipeline_info.hpp>

namespace CoGaDB {

PipelineInfo::PipelineInfo()
    : groupby_param(),
      pipe_type(MATERIALIZE_FROM_ARRAY_TO_ARRAY),
      source_table() {}

const GroupByAggregateParamPtr PipelineInfo::getGroupByAggregateParam() {
  return groupby_param;
}
PipelineEndType PipelineInfo::getPipelineType() const { return pipe_type; }
const TablePtr PipelineInfo::getSourceTable() { return source_table; }

const StatePtr PipelineInfo::getGlobalState() { return global_state; }

void PipelineInfo::setGroupByAggregateParam(GroupByAggregateParamPtr groupby) {
  groupby_param = groupby;
}

void PipelineInfo::setPipelineType(PipelineEndType type) { pipe_type = type; }
void PipelineInfo::setSourceTable(TablePtr table) { source_table = table; }

void PipelineInfo::setGlobalState(StatePtr state) { global_state = state; }

}  // end namespace CoGaDB
