

#include <query_processing/query_processor.hpp>
#include <util/hardware_detector.hpp>

#include <core/data_dictionary.hpp>
#include <lookup_table/join_index.hpp>
#include <lookup_table/lookup_table.hpp>

#include <boost/utility.hpp>

#include <algorithm>  // for copy
#include <iterator>   // for ostream_iterator
//#include <../4.6.4/bits/stl_vector.h> // for ostream_iterator

namespace CoGaDB {

TablePtr reconstruct_tuples_reverse_join_index(
    TablePtr fact_table, PositionListPtr fact_table_tids,
    InvisibleJoinSelectionList dimensions,
    const ProcessorSpecification& proc_spec, std::ostream& out) {
  if (!fact_table) return TablePtr();

  InvisibleJoinSelectionList::iterator it;

  std::vector<LookupColumnPtr> lookup_columns;

  TableSchema new_schema = fact_table->getSchema();
  ColumnVectorPtr new_lookup_arrays(new ColumnVector());

  ColumnVectorPtr new_appended_dense_values_arrays(new ColumnVector());

  LookupColumnPtr lc(new LookupColumn(fact_table, fact_table_tids));
  lookup_columns.push_back(lc);

  for (it = dimensions.begin(); it != dimensions.end(); ++it) {
    TablePtr table = getTablebyName(it->table_name);

    PositionListPtr reverse_join_index =
        JoinIndexes::instance().getReverseJoinIndex(
            getTablebyName(it->table_name), it->join_pred.getColumn1Name(),
            getTablebyName(fact_table->getName()),
            it->join_pred.getColumn2Name());

    PositionListPtr placed_fact_table_tids =
        copy_if_required(fact_table_tids, proc_spec);
    if (!placed_fact_table_tids) return TablePtr();
    PositionListPtr placed_reverse_join_index =
        copy_if_required(reverse_join_index, proc_spec);
    if (!placed_reverse_join_index) return TablePtr();

    GatherParam gather_param(proc_spec);

    ColumnPtr result =
        placed_reverse_join_index->gather(placed_fact_table_tids, gather_param);
    PositionListPtr result_tids =
        boost::dynamic_pointer_cast<PositionList>(result);

    LookupColumnPtr lc(new LookupColumn(table, result_tids));
    lookup_columns.push_back(lc);
  }

  for (size_t i = 0; i < lookup_columns.size(); ++i) {
    TableSchema table_schema = lookup_columns[i]->getTable()->getSchema();
    ColumnVectorPtr lookup_arrays = lookup_columns[i]->getLookupArrays();
    new_schema.insert(new_schema.end(), table_schema.begin(),
                      table_schema.end());
    new_lookup_arrays->insert(new_lookup_arrays->end(), lookup_arrays->begin(),
                              lookup_arrays->end());
  }

  return LookupTablePtr(new LookupTable(std::string(""), new_schema,
                                        lookup_columns, *new_lookup_arrays,
                                        *new_appended_dense_values_arrays));
}

namespace query_processing {

namespace physical_operator {

bool IndexedTupleReconstructionOperator::execute() {
  ProcessorSpecification proc_spec(
      this->sched_dec_.getDeviceSpecification().getProcessingDeviceID());

  this->result_ = reconstruct_tuples_reverse_join_index(
      this->getInputData(), this->param_.fact_table_tids,
      this->param_.dimensions, proc_spec, *this->out);

  if (this->result_) {
    setResultSize((this->result_)->getNumberofRows());
    return true;
  } else
    return false;
}

TypedOperatorPtr create_Indexed_Tuple_Reconstruction_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  logical_operator::Logical_IndexedTupleReconstruction& log_selection_ref =
      static_cast<logical_operator::Logical_IndexedTupleReconstruction&>(
          logical_node);
  // std::cout << "create IndexedTupleReconstructionOperator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  assert(right_child == NULL);  // unary operator
  return TypedOperatorPtr(new IndexedTupleReconstructionOperator(
      sched_dec, left_child,
      log_selection_ref.getIndexedTupleReconstructionParam(),
      log_selection_ref.getDeviceConstraint(),
      log_selection_ref.getMaterializationStatus()));
}

Physical_Operator_Map_Ptr
map_init_function_indexed_tuple_reconstruction_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for INVISIBLE JOIN operator!"
              << std::endl;
// hype::Scheduler::instance().addAlgorithm("SELECTION","CPU_ChainJoin_Algorithm",hype::CPU,"Least
// Squares 1D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("SELECTION","GPU_ChainJoin_Algorithm",hype::GPU,"Least
// Squares 1D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER
  hype::AlgorithmSpecification alg_spec_cpu(
      "CPU_INDEXED_TUPLE_RECONSTRUCTION", "INDEXED_TUPLE_RECONSTRUCTION",
      hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification alg_spec_gpu(
      "GPU_INDEXED_TUPLE_RECONSTRUCTION", "INDEXED_TUPLE_RECONSTRUCTION",
      hype::KNN_Regression, hype::Periodic);
#else
  hype::AlgorithmSpecification selection_alg_spec_cpu(
      "CPU_INDEXED_TUPLE_RECONSTRUCTION", "INDEXED_TUPLE_RECONSTRUCTION",
      hype::KNN_Regression, hype::Periodic);
#endif

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
      hype::Scheduler::instance().addAlgorithm(alg_spec_cpu, dev_specs[i]);
      // hype::Scheduler::instance().addAlgorithm(selection_alg_spec_cpu_parallel,dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
      hype::Scheduler::instance().addAlgorithm(alg_spec_gpu, dev_specs[i]);
    }
  }

  map["CPU_INDEXED_TUPLE_RECONSTRUCTION"] =
      create_Indexed_Tuple_Reconstruction_Operator;
  map["GPU_INDEXED_TUPLE_RECONSTRUCTION"] =
      create_Indexed_Tuple_Reconstruction_Operator;

  // map["CPU_ParallelChainJoin_Algorithm"]=create_CPU_ParallelChainJoin_Operator;
  // map["GPU_ChainJoin_Algorithm"]=create_GPU_ChainJoin_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
