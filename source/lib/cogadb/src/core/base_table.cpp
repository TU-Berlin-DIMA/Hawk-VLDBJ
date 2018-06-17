
#include <fstream>
#include <iostream>
#include <utility>

#include <core/base_table.hpp>
#include <core/table.hpp>
//#include <core/gpu_column_cache.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <core/block_iterator.hpp>
#include <core/user_defined_function.hpp>
#include <core/vector.hpp>

#include <lookup_table/lookup_table.hpp>

#include <util/filesystem.hpp>
#include <util/iostream.hpp>
#include <util/time_measurement.hpp>

#include <boost/any.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <boost/iterator/zip_iterator.hpp>
#include <boost/range/algorithm/count.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include <query_processing/operator_extensions.hpp>
#include <query_processing/query_processor.hpp>
#include <query_processing/sort_operator.hpp>

#include "backends/processor_backend.hpp"
#include "core/block_iterator.hpp"
#include "core/variable_manager.hpp"
#include "lookup_table/join_index.hpp"

#include <boost/tr1/memory.hpp>
#include <new>
#include <util/column_grouping_keys.hpp>
#include <util/result_output_format.hpp>
#include <util/utility_functions.hpp>

using namespace std;

namespace CoGaDB {

BaseTable::BaseTable(const std::string& name, const TableSchema& schema)
    : name_(name), schema_(schema), hash_tables_() {}

BaseTable::~BaseTable() {}

const std::string& BaseTable::getName() const throw() { return name_; }

void BaseTable::setName(const std::string& new_name) throw() {
  name_ = new_name;
}

const TableSchema BaseTable::getSchema() const throw() { return schema_; }

const TablePtr BaseTable::getNext(
    TablePtr table, BlockIteratorPtr it,
    /* \todo ProcessorSpecification can later be used to signal
     * on which processor the block is preferred. Then, a block that
     * is already stored in the correct memory is returned. */
    const ProcessorSpecification&) {
  if (!table) return TablePtr();
  assert(it->hasNext());

  std::vector<ColumnPtr> cols = table->getColumns();
  loadColumnsInMainMemory(cols);
  std::vector<ColumnPtr> vectors;
  for (size_t i = 0; i < cols.size(); ++i) {
    ColumnPtr vector = cols[i]->getVector(cols[i], it);
    vectors.push_back(vector);
  }

  it->advance();
  TablePtr result_table(new Table(table->getName(), vectors));
  return result_table;
}
const HashTablePtr BaseTable::getHashTablebyName(
    const std::string& column_name) const throw() {
  auto find = hash_tables_.find(column_name);
  return find != hash_tables_.end() ? find->second : nullptr;
}

bool BaseTable::addHashTable(const std::string& column_name,
                             HashTablePtr hash_table) {
  HashTables::const_iterator cit = hash_tables_.find(column_name);
  if (cit == hash_tables_.end()) {
    hash_tables_.insert(std::make_pair(column_name, hash_table));
    return true;
  }
  return false;
}

CompressionSpecifications BaseTable::getCompressionSpecifications() {
  CompressionSpecifications compressionSpecifications;
  const std::vector<ColumnPtr>& columns_ = this->getColumns();
  std::vector<ColumnPtr>::const_iterator it = columns_.begin();
  for (; it != columns_.end(); ++it) {
    compressionSpecifications.insert(
        CompressionSpecification((*it)->getName(), (*it)->getColumnType()));
  }
  return compressionSpecifications;
}

const std::vector<ColumnProperties> BaseTable::getPropertiesOfColumns() const {
  std::vector<ColumnProperties> col_props;
  const std::vector<ColumnPtr>& columns = this->getColumns();
  for (size_t i = 0; i < columns.size(); ++i) {
    ColumnProperties props;
    props.name = columns[i]->getName();
    props.attribute_type = columns[i]->getType();
    props.column_type = columns[i]->getColumnType();
    props.is_in_main_memory = columns[i]->isLoadedInMainMemory();
    props.size_in_main_memory = columns[i]->getSizeinBytes();
    props.number_of_rows = columns[i]->size();
    props.number_of_accesses = 0;
    props.last_access_timestamp = 0;
    props.statistics_up_to_date =
        columns[i]->getColumnStatistics().statistics_up_to_date_;
    props.is_sorted_ascending =
        columns[i]->getColumnStatistics().is_sorted_ascending_;
    props.is_sorted_descending =
        columns[i]->getColumnStatistics().is_sorted_descending_;
    props.is_dense_value_array_starting_with_zero =
        columns[i]
            ->getColumnStatistics()
            .is_dense_value_array_starting_with_zero_;
    col_props.push_back(props);
  }

  return col_props;
}

std::string BaseTable::toString() {
  std::stringstream ss;
  const std::vector<ColumnPtr>& columns = this->getColumns();
  loadColumnsInMainMemory(columns);
  ResultFormatPtr formatter =
      getResultFormatter(VariableManager::instance().getVariableValueString(
          "result_output_format"));
  assert(formatter != NULL);
  return formatter->getResult(this->name_, columns, true);
}

std::string BaseTable::toString(const std::string& result_output_format,
                                bool include_header) {
  std::stringstream ss;
  const std::vector<ColumnPtr>& columns = this->getColumns();

  ResultFormatPtr formatter = getResultFormatter(result_output_format);
  assert(formatter != NULL);
  return formatter->getResult(this->name_, columns, include_header);
}

//	const TableSchema mergeTableSchemas(const TableSchema& schema1, const
// std::string& join_attributname_table1,
//													const
// TableSchema& schema2, const std::string& join_attributname_table2){
const TableSchema mergeTableSchemas(const TableSchema& schema1,
                                    const std::string&,
                                    const TableSchema& schema2,
                                    const std::string&) {
  TableSchema result_schema, schema_(schema1), schema(schema2);
  TableSchema::iterator it, it2;
  map<string, pair<AttributeType, string> > schema_map;
  map<string, pair<AttributeType, string> >::iterator map_it;

  for (it = schema_.begin(); it != schema_.end(); it++) {
    schema_map[it->second] = *it;
    if (!quiet && verbose) cout << "it->second: " << it->second << endl;
    result_schema.push_back(*it);
  }

  // TableSchema schema=table->getSchema();
  for (it2 = schema.begin(); it2 != schema.end(); it2++) {
    map_it = schema_map.find(it2->second);
    if (map_it != schema_map.end()) {
      schema_map[it2->second] = *it2;
      if (!quiet && verbose) cout << "FOUND Duplicate!" << it2->second << endl;
      result_schema.push_back(*it2);
    } else {
      schema_map[it2->second] = *it2;
      if (!quiet && verbose) cout << "it2->second: " << it2->second << endl;
      result_schema.push_back(*it2);
    }
  }
  if (!quiet && verbose)
    cout << "[DEBUG]: JOIN(): schema of new table:" << endl << "| ";
  if (!quiet && verbose) {
    for (it = result_schema.begin(); it != result_schema.end(); ++it)
      cout << (it->second) << " |";  // << endl;
    cout << endl;
  }
  return result_schema;
}

const TablePtr BaseTable::createResultTable(
    TablePtr table, PositionListPtr tids, MaterializationStatus mat_stat,
    const std::string& operation_name,
    const ProcessorSpecification& proc_spec) {
  if (!table || !tids) return TablePtr();

  TablePtr result_table;
  if (mat_stat == MATERIALIZE) {
    result_table = TablePtr(
        new Table(operation_name + string("( ") + table->getName() + " )",
                  table->getSchema()));  // tmp_schema));
    // fetch result tuples
    for (unsigned int i = 0; i < tids->size(); i++) {
      TID j = (*tids)[i];
      result_table->insert(table->fetchTuple(j));
    }

  } else if (mat_stat == LOOKUP) {
    if (!quiet)
      cout << "create Lookup Table for Table " << table->getName() << " ..."
           << endl;

    result_table = createLookupTableforUnaryOperation(table->getName(), table,
                                                      tids, proc_spec);

  } else {
    cout << "Error! Unkown Materialization Option!" << endl;
    return TablePtr();  ////error, return NULL pointer
  }

  return result_table;
}

const TablePtr BaseTable::createResultTable(
    TablePtr table1, const std::string& join_column_table1, TablePtr table2,
    const std::string& join_column_table2, PositionListPairPtr join_tids,
    TableSchema result_schema, MaterializationStatus mat_stat,
    const std::string& operation_name,
    const ProcessorSpecification& proc_spec) {
  if (!table1 || !table2 || !join_tids) return TablePtr();
  TablePtr result_table;
  if (mat_stat == MATERIALIZE) {
    result_table = TablePtr(
        new Table(string("join( ") + table1->getName() + string(".") +
                      join_column_table1 + string(",") + table2->getName() +
                      string(".") + join_column_table2 + " )",
                  result_schema));

    const std::vector<ColumnPtr>& columns_ = table1->getColumns();
    const std::vector<ColumnPtr>& tab_columns_ = table2->getColumns();
    // ensure tables are loaded in-memory
    loadColumnsInMainMemory(columns_);
    loadColumnsInMainMemory(tab_columns_);

    TID index_table1, index_table2;
    for (unsigned int i = 0; i < join_tids->first->size(); i++) {
      index_table1 = (*(join_tids->first))[i];
      index_table2 = (*(join_tids->second))[i];

      Tuple t;

      for (unsigned int i = 0; i < columns_.size(); i++)
        t.push_back(columns_[i]->get(index_table1));

      for (unsigned int i = 0; i < tab_columns_.size(); i++)
        t.push_back(tab_columns_[i]->get(index_table2));

      result_table->insert(t);
    }

  } else if (mat_stat == LOOKUP) {
    if (!quiet) cout << "create Lookup Table for resulting Table..." << endl;
    string result_lookup_table_name =
        string("Lookup_Table_") + operation_name + std::string("( ") +
        table1->getName() + string(".") + join_column_table1 + string(",") +
        table2->getName() + string(".") + join_column_table2 + " )";
    // consider join as "two" semi joins
    LookupTablePtr lookup_table_for_table1 = createLookupTableforUnaryOperation(
        string("AGGREGATED( ") + table1->getName() + " )", table1,
        join_tids->first, proc_spec);
    LookupTablePtr lookup_table_for_table2 = createLookupTableforUnaryOperation(
        string("AGGREGATED( ") + table2->getName() + " )", table2,
        join_tids->second, proc_spec);
    assert(lookup_table_for_table1 != NULL);
    assert(lookup_table_for_table2 != NULL);

    result_table = LookupTable::concatenate(result_lookup_table_name,
                                            *lookup_table_for_table1,
                                            *lookup_table_for_table2);

  } else {
    COGADB_FATAL_ERROR("Unkown Materialization Option!", "");
    // cout << "Error! Unkown Materialization Option!" << endl;
    return TablePtr();  ////error, return NULL pointer
  }
  return result_table;
}

/***************** status report *****************/
size_t BaseTable::getNumberofRows() const throw() {
  const std::vector<ColumnPtr>& columns_ = this->getColumns();
  if (columns_.size() > 0) {
    if (!columns_[0]->isLoadedInMainMemory()) loadColumnFromDisk(columns_[0]);
    return columns_[0]->size();
  } else {
    return 0;
  }
}

void BaseTable::setNumberofRows(size_t numberofRows) {
  // do nothing
}

size_t BaseTable::getSizeinBytes() const throw() {
  loadColumnsInMainMemory(getColumns());
  size_t size_in_bytes = 0;

  for (const auto& col : getColumns()) {
    size_in_bytes += (col)->getSizeinBytes();
  }

  return size_in_bytes;
}

bool BaseTable::approximatelyEquals(TablePtr reference, TablePtr candidate) {
  std::cout << "ApproximatelyEquals getting called" << std::endl;

  if (candidate == nullptr || reference == nullptr) {
    return false;
  }

  if (candidate == reference) {
    return true;
  }

  if (candidate->getNumberofRows() != reference->getNumberofRows()) {
    // if (!quiet && debug)
    {
      std::cout << "Both tables have a different amount of rows: "
                << candidate->getNumberofRows() << " vs "
                << reference->getNumberofRows() << std::endl;
    }

    return false;
  }

  if (!isSameTableSchema(reference->getSchema(), candidate->getSchema())) {
    std::cout << "Table Schema not equal!" << std::endl;
    return false;
  }

  std::list<std::string> column_names;
  for (const auto& col : reference->getColumns()) {
    column_names.push_back(col->getName());
  }

  // sort both Tables
  std::cout << "[UNSORTED] Candidate: " << std::endl;
  candidate->print();

  std::cout << "[UNSORTED] Reference: " << std::endl;
  reference->print();

  std::cout << "Start sorting" << std::endl;

  TablePtr candidate_sorted =
      sort(candidate, column_names, ASCENDING, MATERIALIZE);

  std::cout << "Candidate table successfully sorted" << std::endl;

  TablePtr reference_sorted =
      sort(reference, column_names, ASCENDING, MATERIALIZE);

  for (const auto& col : candidate_sorted->getColumns()) {
    if (!col->isApproximatelyEqual(
            reference_sorted->getColumnbyName(col->getName()))) {
      std::cout << "The column " << col->getName()
                << " does not match the column in the comparing table"
                << std::endl;

      std::cout << "Reference Table: " << std::endl;
      reference_sorted->print();

      std::cout << "Candidate Table: " << std::endl;
      candidate_sorted->print();

      return false;
    }
  }

  return true;
}

bool BaseTable::equals(TablePtr reference, TablePtr candidate) {
  if (candidate == nullptr || reference == nullptr) {
    return false;
  }

  if (candidate == reference) {
    return true;
  }

  if (candidate->getNumberofRows() != reference->getNumberofRows()) {
    if (!quiet && debug) {
      std::cout << "Both tables have a different amount of rows: "
                << candidate->getNumberofRows() << " vs "
                << reference->getNumberofRows() << std::endl;
    }

    return false;
  }

  bool equal =
      isSameTableSchema(reference->getSchema(), candidate->getSchema());

  if (!equal) {
    if (!quiet && debug) {
      std::cout << "The schemas differ." << std::endl;

      TableSchema refSchema = reference->getSchema();
      TableSchema candidateSchema = candidate->getSchema();

      for (std::list<Attribut>::const_iterator
               it = refSchema.begin(),
               it_that = candidateSchema.begin();
           it != refSchema.end(); ++it, ++it_that) {
        std::cout << "[This] [Attribute] is " << util::getName((*it).first)
                  << " [Name] is "
                  << "\"" << (*it).second << "\"" << std::endl;
        std::cout << "[Candidate] [Attribute] is "
                  << util::getName((*it_that).first) << " [Name] is "
                  << "\"" << (*it_that).second << "\"" << std::endl;
        if ((*it) == (*it_that)) {
          std::cout << "Is Equal!" << std::endl;
        } else {
          std::cout << "Is Unequal!" << std::endl;
        }
      }
    }

    return equal;
  }

  std::list<std::string> column_names;
  for (const auto& col : reference->getColumns()) {
    column_names.push_back(col->getName());
  }

  // sort both Tables
  TablePtr sortedCandidate = sort(candidate, column_names, ASCENDING, LOOKUP);
  TablePtr sortedReference = sort(reference, column_names, ASCENDING, LOOKUP);

  // get candidates columns
  std::vector<ColumnPtr> candidateColumns = sortedCandidate->getColumns();
  std::vector<ColumnPtr> referenceColumns = sortedReference->getColumns();

  for (size_t i = 0; i < candidateColumns.size(); ++i) {
    if (!candidateColumns[i]->is_equal(referenceColumns[i])) {
      if (!quiet && debug) {
        std::cout << "The column " << candidateColumns[i]->getName()
                  << " does not match the column in the comparing table"
                  << std::endl;
      }
      return false;
    }
  }

  return true;
}

/***************** relational operations *****************/
const TablePtr BaseTable::selection(TablePtr table,
                                    const std::string& column_name,
                                    const SelectionParam& param) {
  if (!table) {
    cout << "Fatal Error: in BaseTable::selection(): input table pointer is "
            "NULL!"
         << endl;
    return TablePtr();
  }

  ColumnPtr column = table->getColumnbyName(column_name);
  if (column.get() == NULL) {
    cout << "Error! Could not look up Column: " << column_name << endl;
    exit(-1);
  }

  // perform selection
  PositionListPtr tids;
  column = copy_if_required(column,
                            hype::util::getMemoryID(param.proc_spec.proc_id));
  if (!column) return TablePtr();
  tids = column->selection(param);
  if (!tids) return TablePtr();
  TablePtr result_table =
      createResultTable(table, tids, LOOKUP, "selection", param.proc_spec);
  return result_table;
}

const TablePtr BaseTable::selection(
    TablePtr table, const Disjunction& disjunction,
    const ProcessorSpecification& proc_spec,
    const hype::DeviceConstraint& dev_constr_param) {
  if (!table) {
    cout << "Fatal Error: in BaseTable::selection(): input table pointer is "
            "NULL!"
         << endl;
    return TablePtr();
  }

  if (disjunction.size() == 1) {
    if (disjunction.front().getPredicateType() == ValueConstantPredicate ||
        disjunction.front().getPredicateType() ==
            ValueRegularExpressionPredicate) {
      SelectionParam param(proc_spec, disjunction.front().getPredicateType(),
                           disjunction.front().getConstant(),
                           disjunction.front().getValueComparator());
      return selection(table, disjunction.front().getColumn1Name(), param);
    } else if (disjunction.front().getPredicateType() == ValueValuePredicate) {
      // ValueValuePredicate, this means we need to compare two columns with
      // each other
      ColumnPtr comparison_column =
          table->getColumnbyName(disjunction.front().getColumn2Name());
      assert(comparison_column != NULL);
      SelectionParam param(proc_spec, disjunction.front().getPredicateType(),
                           comparison_column,
                           disjunction.front().getValueComparator());
      return selection(table, disjunction.front().getColumn1Name(), param);
    } else {
      COGADB_FATAL_ERROR("Invalid PredicateType!", "");
    }
  }

  hype::DeviceConstraint dev_constr = dev_constr_param;

  if (dev_constr == hype::ANY_DEVICE) {
    if (hype::util::isCPU(proc_spec.proc_id)) {
      dev_constr = hype::CPU_ONLY;
    } else if (hype::util::isCoprocessor(proc_spec.proc_id)) {
      dev_constr = hype::GPU_ONLY;
    } else {
      COGADB_FATAL_ERROR("Unknown Processor Type!", "");
    }
  }

  query_processing::column_processing::cpu::LogicalQueryPlanPtr log_plan =
      query_processing::createColumnPlanforDisjunction(
          table, disjunction,
          dev_constr);  // CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint());
  if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
    log_plan->print();
  }
  query_processing::column_processing::cpu::PhysicalQueryPlanPtr phy_plan;
  if (RuntimeConfiguration::instance().isQueryChoppingEnabled()) {
    phy_plan = log_plan->runChoppedPlan();
  } else {
    phy_plan = log_plan->convertToPhysicalQueryPlan();
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      phy_plan->print();
    }
    phy_plan->run();
  }
  if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
    cout << "Physical TOPPO Plan: " << endl;
    phy_plan->print();
  }

  query_processing::PositionListOperator* positionlist_op =
      dynamic_cast<query_processing::PositionListOperator*>(
          phy_plan->getRoot().get());
  assert(positionlist_op != NULL);
  assert(positionlist_op->hasResultPositionList());

  PositionListPtr tids =
      copy_if_required(positionlist_op->getResultPositionList(), proc_spec);
  if (!tids) return TablePtr();
  //        assert(tids!=NULL);

  TablePtr result_table =
      createResultTable(table, tids, LOOKUP, "selection", proc_spec);

  return result_table;
}

const TablePtr BaseTable::selection(TablePtr table,
                                    const KNF_Selection_Expression& knf_expr,
                                    MaterializationStatus mat_stat,
                                    ParallelizationMode comp_mode) {
  hype::ProcessingDeviceID id = hype::PD0;
  ProcessorSpecification proc_spec(id);

  if (!table) {
    cout << "Fatal Error: in BaseTable::selection(): input table pointer is "
            "NULL!"
         << endl;
    return TablePtr();
  }
  if (knf_expr.disjunctions.empty()) {
    // return TablePtr();
    // workaround, so invisible join works even in case no filtering is
    // performed on a dimension table
    PositionListPtr result_tids = createPositionList();
    for (size_t i = 0; i < table->getNumberofRows(); ++i) {
      result_tids->push_back(i);
    }
    hype::ProcessingDeviceID id = hype::PD0;
    ProcessorSpecification proc_spec(id);
    return createResultTable(table, result_tids, mat_stat,
                             "selection(empty selection condition)", proc_spec);
  }

  // stores the result for each disjunction
  std::vector<PositionListPtr> disjunctions_result_tid_lists(
      knf_expr.disjunctions.size());
  for (unsigned int i = 0; i < knf_expr.disjunctions.size(); i++) {
    // stores the tid list for each predicate
    std::vector<PositionListPtr> predicate_result_tid_lists(
        knf_expr.disjunctions[i].size());
    for (unsigned j = 0; j < knf_expr.disjunctions[i].size(); j++) {
      if (knf_expr.disjunctions[i][j].getPredicateType() ==
          ValueValuePredicate) {
        ColumnPtr col = table->getColumnbyName(
            knf_expr.disjunctions[i][j].getColumn1Name());
        if (!col) {
          cout << "Error! in BaseTable::selection(): Could not find Column "
               << knf_expr.disjunctions[i][j].getColumn1Name() << " in Table "
               << table->getName() << endl;
          cout << "In File " << __FILE__ << ":" << __LINE__ << endl;
          return TablePtr();
        }
        ColumnPtr col2 = table->getColumnbyName(
            knf_expr.disjunctions[i][j].getColumn2Name());
        if (!col2) {
          cout << "Error! in BaseTable::selection(): Could not find Column "
               << knf_expr.disjunctions[i][j].getColumn2Name() << " in Table "
               << table->getName() << endl;
          cout << "In File " << __FILE__ << ":" << __LINE__ << endl;
          return TablePtr();
        }
        //                        hype::ProcessingDeviceID id=hype::PD0;
        //                        ProcessorSpecification proc_spec(id);
        SelectionParam param(proc_spec, ValueValuePredicate, col2,
                             knf_expr.disjunctions[i][j].getValueComparator());
        predicate_result_tid_lists[j] = col->selection(param);
      } else if (knf_expr.disjunctions[i][j].getPredicateType() ==
                 ValueConstantPredicate) {
        ColumnPtr col = table->getColumnbyName(
            knf_expr.disjunctions[i][j].getColumn1Name());
        if (!col) {
          cout << "Error! in BaseTable::selection(): Could not find Column "
               << knf_expr.disjunctions[i][j].getColumn1Name() << " in Table "
               << table->getName() << endl;
          cout << "In File " << __FILE__ << ":" << __LINE__ << endl;
          return TablePtr();
        }
        //                        hype::ProcessingDeviceID id=hype::PD0;
        //                        ProcessorSpecification proc_spec(id);
        SelectionParam param(proc_spec, ValueConstantPredicate,
                             knf_expr.disjunctions[i][j].getConstant(),
                             knf_expr.disjunctions[i][j].getValueComparator());
        predicate_result_tid_lists[j] = col->selection(param);
        //                        predicate_result_tid_lists[j]=col->selection(knf_expr.disjunctions[i][j].getConstant(),knf_expr.disjunctions[i][j].getValueComparator());
      } else {
        cerr
            << "FATAL ERROR! in BaseTable::selection(): Unknown Predicate Type!"
            << endl;
        cerr << "In File " << __FILE__ << ":" << __LINE__ << endl;
        exit(-1);
      }
    }
    // merge sorted tid lists (compute the disjunction)
    // PositionListPtr result_tids=createPositionList();
    // use first tid list as result tid list, then perform union with all other
    // lists to get the result for this disjunction
    PositionListPtr result_tids(
        new PositionList("", OID, predicate_result_tid_lists[0]->begin(),
                         predicate_result_tid_lists[0]->end()));
    for (unsigned int k = 1; k < predicate_result_tid_lists.size(); ++k) {
      PositionListPtr tmp_tids(createPositionList(
          result_tids->size() + predicate_result_tid_lists[k]->size()));
      PositionList::iterator it;
      it = std::set_union(result_tids->begin(), result_tids->end(),
                          predicate_result_tid_lists[k]->begin(),
                          predicate_result_tid_lists[k]->end(),
                          tmp_tids->begin());
      tmp_tids->resize(it - tmp_tids->begin());
      result_tids = tmp_tids;
    }
    disjunctions_result_tid_lists[i] = result_tids;
    if (!quiet && verbose && debug) {
      cout << "Disjunction " << i << "result:" << endl;
      for (unsigned int i = 0; i < result_tids->size(); i++) {
        cout << (*result_tids)[i] << endl;
      }
    }
  }
  // intersect tid list for each disjunction to compute final result
  // use first tid list as result tid list, then intersect with all other lists
  // to get the overall result
  PositionListPtr tids(
      new PositionList("", OID, disjunctions_result_tid_lists[0]->begin(),
                       disjunctions_result_tid_lists[0]->end()));
  for (unsigned int i = 1; i < knf_expr.disjunctions.size(); i++) {
    PositionListPtr tmp_tids(createPositionList(
        tids->size() + disjunctions_result_tid_lists[i]->size()));
    PositionList::iterator it;
    it = std::set_intersection(
        tids->begin(), tids->end(), disjunctions_result_tid_lists[i]->begin(),
        disjunctions_result_tid_lists[i]->end(), tmp_tids->begin());
    tmp_tids->resize(it - tmp_tids->begin());
    // it=std::set_intersection (first, first+5, second, second+5, v.begin());
    // 10 20 0  0  0  0  0  0  0  0
    // v.resize(it-v.begin());

    tids = tmp_tids;
  }
  if (!quiet && verbose && debug) {
    cout << "Konjunction result:" << endl;
    for (unsigned int i = 0; i < tids->size(); i++) {
      cout << (*tids)[i] << endl;
    }
  }
  // selection is completed, now the result table has to be cosntructred!
  TablePtr result_table =
      createResultTable(table, tids, mat_stat, "selection", proc_spec);
  return result_table;
}

const TablePtr BaseTable::projection(
    TablePtr table, const std::list<std::string>& columns_to_select,
    MaterializationStatus, const ComputeDevice) {
  if (!table) {
    cout << "Fatal Error: in BaseTable::projection(): input table pointer is "
            "NULL!"
         << endl;
    return TablePtr();
  }
  //	assert(mat_stat==MATERIALIZE);
  Timestamp begin, end;
  begin = getTimestamp();

  TablePtr result_table;

  // if(mat_stat==MATERIALIZE){
  if (table->isMaterialized()) {
    if (!quiet && verbose)
      cout << "Performing Projection on Materialized Table..." << endl;
    list<string>::const_iterator it;
    vector<ColumnPtr> new_schema;
    const std::vector<ColumnPtr>& columns = table->getColumns();
    // loadColumnsInMainMemory(columns, this->name_);
    for (it = columns_to_select.begin(); it != columns_to_select.end(); it++) {
      // std::cout << "Search column " << *it << std::endl;
      for (unsigned int i = 0; i < columns.size(); i++) {
        // cout << "Projection: Examine Column " << columns[i]->getName() <<
        // endl;
        if (*it == columns[i]->getName()) {
          if (!columns[i]->isLoadedInMainMemory()) {
            loadColumnFromDisk(columns[i]);
          }
          // ColumnPtr new_column = columns[i]->copy();
          // new_schema.push_back(new_column);
          new_schema.push_back(columns[i]);
        }
      }
    }
    //                new_schema = copy_if_required(new_schema,
    //                hype::PD_Memory_0);
    new_schema = copy(new_schema, hype::PD_Memory_0);
    assert(!new_schema.empty());

    result_table =
        TablePtr(new Table(string("projection( ") + table->getName() + " )",
                           new_schema));  // tmp_schema));
    // TablePtr result_table(new Table(string(""),new_schema)) ;//tmp_schema));

    //		}else if(mat_stat==LOOKUP){
  } else {
    if (!quiet && verbose)
      cout << "Performing Projection on Lookup Table..." << endl;

    if (!table->copyColumnsInMainMemory()) {
      COGADB_FATAL_ERROR("Could not copy columns back to CPU memory!", "");
    }

    LookupTablePtr input_lookup_table =
        shared_pointer_namespace::static_pointer_cast<LookupTable>(table);
    assert(input_lookup_table != NULL);
    const std::vector<LookupColumnPtr>& input_lookup_columns =
        input_lookup_table->getLookupColumns();
    // create copy of LookupColumns, so the can be passed to the new Lookup
    // Table
    vector<LookupColumnPtr> new_lookup_columns;
    if (!quiet && verbose) cout << "Copying Lookup Columns" << endl;
    for (unsigned int i = 0; i < input_lookup_columns.size(); i++) {
      LookupColumnPtr lcol = input_lookup_columns[i]->copy();
      new_lookup_columns.push_back(lcol);
    }
    // collect LookupArrays of new LookupColumns
    std::vector<ColumnPtr> new_lookup_arrays;
    for (unsigned int i = 0; i < new_lookup_columns.size(); i++) {
      ColumnVectorPtr v = new_lookup_columns[i]->getLookupArrays();
      new_lookup_arrays.insert(new_lookup_arrays.end(), v->begin(), v->end());
    }
    // collect dense Value Columns
    const ColumnVector& dense_value_arrays =
        input_lookup_table->getDenseValueColumns();
    new_lookup_arrays.insert(new_lookup_arrays.end(),
                             dense_value_arrays.begin(),
                             dense_value_arrays.end());

    list<string>::const_iterator it;
    vector<ColumnPtr> result_lookup_arrays;
    TableSchema new_schema;
    for (it = columns_to_select.begin(); it != columns_to_select.end(); it++) {
      for (unsigned int i = 0; i < new_lookup_arrays.size(); i++) {
        if (!quiet && verbose)
          cout << "Projection: Examining Column: "
               << new_lookup_arrays[i]->getName() << endl;
        if (*it == new_lookup_arrays[i]->getName()) {
          new_schema.push_back(Attribut(
              new_lookup_arrays[i]->getType(),
              new_lookup_arrays[i]->getName()));  // create entry in schema
          ColumnPtr new_column = new_lookup_arrays[i];  // lookup Array is
                                                        // already a copy!
                                                        // //columns[i]->copy();
          result_lookup_arrays.push_back(new_column);
        }
      }
    }

    result_table = TablePtr(
        new LookupTable(string("projection( ") + table->getName() + " )",
                        new_schema, new_lookup_columns, result_lookup_arrays));
  }

  end = getTimestamp();
  assert(end >= begin);
  if (print_time_measurement)
    cout << "Time for Projection: " << end - begin << "ns ("
         << double(end - begin) / 1000000 << "ms)" << endl;

  return result_table;
}

const PositionListPairPtr gather_join(ColumnPtr void_compressed_column,
                                      ColumnPtr tid_column,
                                      const ProcessorSpecification& proc_spec) {
  assert(void_compressed_column->getType() == OID);
  assert(tid_column->getType() == OID);

  if (tid_column->getColumnType() == VOID_COMPRESSED_NUMBER &&
      void_compressed_column->getColumnType() != VOID_COMPRESSED_NUMBER) {
    PositionListPairPtr result =
        gather_join(tid_column, void_compressed_column, proc_spec);
    if (!result) {
      return PositionListPairPtr();
    }
    std::swap(result->first, result->second);
    return result;
  }

  tid_column = decompress_if_required(tid_column);

  assert(void_compressed_column->getColumnType() == VOID_COMPRESSED_NUMBER);

  /* the tid_column already contains the result of the right table */
  ColumnPtr untyped_tid_column = tid_column->copy();
  PositionListPtr left_tids =
      boost::dynamic_pointer_cast<PositionList>(untyped_tid_column);
  assert(left_tids != NULL);
  if (!left_tids) PositionListPairPtr();
  /* the result for the left table is a dense value TID column */
  ProcessorBackend<TID>* backend =
      ProcessorBackend<TID>::get(proc_spec.proc_id);
  PositionListPtr right_tids =
      createPositionList(tid_column->size(), proc_spec);
  //            std::cout << util::getName(tid_column->getColumnType()) <<
  //            std::endl;
  //            std::cout << tid_column->size() << std::endl;
  assert(right_tids != NULL);
  if (!right_tids) return PositionListPairPtr();

  if (!backend->generateAscendingSequence(
          right_tids->data(), right_tids->size(), TID(0), proc_spec)) {
    COGADB_FATAL_ERROR("Cannot generate sequence!", "");
  }

  return PositionListPairPtr(new PositionListPair(left_tids, right_tids));
}

const TablePtr BaseTable::join(TablePtr table1,
                               const std::string& join_column_table1,
                               TablePtr table2,
                               const std::string& join_column_table2,
                               const JoinParam& param) {
  //											 JoinAlgorithm
  // join_alg,
  // MaterializationStatus mat_stat, const ComputeDevice comp_dev){

  if (!table1) {
    cout << "Fatal Error: in BaseTable::join(): input table pointer (table1) "
            "is NULL!"
         << endl;
    return TablePtr();
  }
  if (!table2) {
    cout << "Fatal Error: in BaseTable::join(): input table pointer (table2) "
            "is NULL!"
         << endl;
    return TablePtr();
  }

  // assert(comp_dev!=GPU);

  Timestamp begin, end;
  begin = getTimestamp();
  // shared_pointer_namespace::shared_ptr<Table> table =
  // shared_pointer_namespace::static_pointer_cast<Table>(table_);

  ColumnPtr col1 = table1->getColumnbyName(join_column_table1);
  if (col1 == NULL) {
    cout << "table " << table1->getName() << " has no Column named "
         << join_column_table1 << endl;
    cout << "join failed!" << endl;
    return TablePtr();
  }
  ColumnPtr col2 = table2->getColumnbyName(join_column_table2);
  if (col2 == NULL) {
    cout << "table " << table2->getName() << " has no Column named "
         << join_column_table2 << endl;
    cout << "join failed!" << endl;
    return TablePtr();
  }

  // PositionListPairPtr join_tids;
  PositionListPairPtr join_tids;  //( new PositionListPair());

  ColumnPtr placed_col1 = copy_if_required(col1, getMemoryID(param.proc_spec));
  if (!placed_col1) return TablePtr();
  ColumnPtr placed_col2 = copy_if_required(col2, getMemoryID(param.proc_spec));
  if (!placed_col2) return TablePtr();

  if (param.join_type == GATHER_JOIN) {
    join_tids = gather_join(placed_col1, placed_col2, param.proc_spec);
  } else {
    join_tids = placed_col1->join(placed_col2, param);
  }

  if (!join_tids) return TablePtr();

  TableSchema result_schema;
  result_schema = mergeTableSchemas(table1->getSchema(), join_column_table1,
                                    table2->getSchema(), join_column_table2);

  assert(join_tids->first->size() == join_tids->second->size());

  TablePtr result_table;

  if (!quiet) cout << "create Lookup Table for resulting Table..." << endl;

  string result_lookup_table_name =
      string("Lookup_Table_join( ") + table1->getName() + string(".") +
      join_column_table1 + string(",") + table2->getName() + string(".") +
      join_column_table2 + " )";

  // consider join as "two" semi joins
  LookupTablePtr lookup_table_for_table1 = createLookupTableforUnaryOperation(
      string("AGGREGATED( ") + table1->getName() + " )", table1,
      join_tids->first, param.proc_spec);
  LookupTablePtr lookup_table_for_table2 = createLookupTableforUnaryOperation(
      string("AGGREGATED( ") + table2->getName() + " )", table2,
      join_tids->second, param.proc_spec);

  if (!lookup_table_for_table1 || !lookup_table_for_table2) return TablePtr();

  result_table = LookupTable::concatenate(result_lookup_table_name,
                                          *lookup_table_for_table1,
                                          *lookup_table_for_table2);

  end = getTimestamp();
  assert(end >= begin);
  if (print_time_measurement)
    cout << "Time for Join: " << end - begin << "ns ("
         << double(end - begin) / 1000000 << "ms)" << endl;

  return result_table;
}

const TablePtr BaseTable::semi_join(TablePtr table1,
                                    const std::string& join_column_table1,
                                    TablePtr table2,
                                    const std::string& join_column_table2,
                                    const JoinParam& param) {
  if (!table1) {
    cout << "Fatal Error: in BaseTable::join(): input table pointer (table1) "
            "is NULL!"
         << endl;
    return TablePtr();
  }
  if (!table2) {
    cout << "Fatal Error: in BaseTable::join(): input table pointer (table2) "
            "is NULL!"
         << endl;
    return TablePtr();
  }

  Timestamp begin, end;
  begin = getTimestamp();
  // shared_pointer_namespace::shared_ptr<Table> table =
  // shared_pointer_namespace::static_pointer_cast<Table>(table_);

  ColumnPtr col1 = table1->getColumnbyName(join_column_table1);
  if (col1 == NULL) {
    cout << "table " << table1->getName() << " has no Column named "
         << join_column_table1 << endl;
    cout << "join failed!" << endl;
    return TablePtr();
  }
  ColumnPtr col2 = table2->getColumnbyName(join_column_table2);
  if (col2 == NULL) {
    cout << "table " << table2->getName() << " has no Column named "
         << join_column_table2 << endl;
    cout << "join failed!" << endl;
    return TablePtr();
  }

  // PositionListPairPtr join_tids;
  PositionListPtr join_tids;  //( new PositionListPair());

  ColumnPtr placed_col1 = copy_if_required(col1, getMemoryID(param.proc_spec));
  if (!placed_col1) return TablePtr();
  ColumnPtr placed_col2 = copy_if_required(col2, getMemoryID(param.proc_spec));
  if (!placed_col2) return TablePtr();

  join_tids = placed_col1->tid_semi_join(placed_col2, param);

//#define COGADB_TEST_GPU_SEMI_JOIN
#ifdef COGADB_TEST_GPU_SEMI_JOIN
  if (hype::util::isCoprocessor(param.proc_spec.proc_id)) {
    PositionListPtr cpu_join_tids;  //( new PositionListPair());
    ProcessorSpecification proc_spec(hype::PD0);
    ColumnPtr cpu_placed_col1 = copy_if_required(col1, getMemoryID(proc_spec));
    if (!cpu_placed_col1) return TablePtr();
    ColumnPtr cpu_placed_col2 = copy_if_required(col2, getMemoryID(proc_spec));
    if (!cpu_placed_col2) return TablePtr();

    JoinParam new_param(param);
    new_param.proc_spec = proc_spec;
    cpu_join_tids = cpu_placed_col1->tid_semi_join(cpu_placed_col2, new_param);
    PositionListPtr gpu_join_tids =
        copy_if_required(join_tids, getMemoryID(proc_spec));

    std::sort(cpu_join_tids->data(),
              cpu_join_tids->data() + cpu_join_tids->size());
    std::sort(gpu_join_tids->data(),
              gpu_join_tids->data() + gpu_join_tids->size());

    if (!cpu_join_tids->is_equal(gpu_join_tids)) {
      std::cerr << "Wrong result of GPU Semi Join: " << std::endl;
      std::cerr << "CPU Returned: " << cpu_join_tids->size() << "rows"
                << std::endl;
      std::cerr << "GPU Returned: " << gpu_join_tids->size() << "rows"
                << std::endl;
      std::cerr << "Result TIDs: " << std::endl;
      TID* cpu = cpu_join_tids->data();
      TID* gpu = gpu_join_tids->data();
      for (size_t i = 0;
           i < std::min(cpu_join_tids->size(), gpu_join_tids->size()); ++i) {
        if (cpu[i] != gpu[i])
          std::cerr << cpu[i] << ", " << gpu[i] << std::endl;
      }
      COGADB_FATAL_ERROR("Detected Wrong Result of GPU Semi Join!", "");
    }
  }
#endif

  if (!join_tids) return TablePtr();

  TableSchema result_schema;
  TablePtr result_side_table;
  if (param.join_type == LEFT_SEMI_JOIN ||
      param.join_type == LEFT_ANTI_SEMI_JOIN) {
    result_side_table = table1;
  } else if (param.join_type == RIGHT_SEMI_JOIN ||
             param.join_type == RIGHT_ANTI_SEMI_JOIN) {
    result_side_table = table2;
  } else {
    COGADB_FATAL_ERROR("Passed Invalid Join Type to SemiJoin!", "");
  }

  TablePtr result_table =
      createResultTable(result_side_table, join_tids, LOOKUP,
                        util::getName(param.join_type), param.proc_spec);

  end = getTimestamp();
  assert(end >= begin);
  if (print_time_measurement)
    cout << "Time for Join: " << end - begin << "ns ("
         << double(end - begin) / 1000000 << "ms)" << endl;

  return result_table;
}

const TablePtr BaseTable::pk_fk_join(
    TablePtr table1, const std::string& join_column_table1, TablePtr table2,
    const std::string& join_column_table2, JoinAlgorithm join_alg,
    MaterializationStatus mat_stat, const ComputeDevice comp_dev) {
  if (!table1) {
    cout << "Fatal Error: in BaseTable::pk_fk_join(): input table pointer "
            "(table1) is NULL!"
         << endl;
    return TablePtr();
  }
  if (!table2) {
    cout << "Fatal Error: in BaseTable::pk_fk_join(): input table pointer "
            "(table2) is NULL!"
         << endl;
    return TablePtr();
  }

  // shared_pointer_namespace::shared_ptr<Table> table =
  // shared_pointer_namespace::static_pointer_cast<Table>(table_);

  ColumnPtr col1 = table1->getColumnbyName(join_column_table1);
  if (col1 == NULL) {
    cout << "table " << table1->getName() << " has no Column named "
         << join_column_table1 << endl;
    cout << "join failed!" << endl;
    return TablePtr();
  }
  assert(col1->hasPrimaryKeyConstraint());
  ColumnPtr col2 = table2->getColumnbyName(join_column_table2);
  if (col2 == NULL) {
    cout << "table " << table2->getName() << " has no Column named "
         << join_column_table2 << endl;
    cout << "join failed!" << endl;
    return TablePtr();
  }
  assert(col2->hasForeignKeyConstraint());
  PositionListPairPtr join_tids(new PositionListPair());

  hype::ProcessingDeviceID id = hype::PD0;
  //                if(comp_dev==GPU) id=hype::PD1;
  ProcessorSpecification proc_spec(id);

  if (comp_dev == CPU) {
    if (join_alg ==
        SORT_MERGE_JOIN) {  // SORT_MERGE_JOIN,NESTED_LOOP_JOIN,HASH_JOIN
      if (!quiet) cout << "Perform Sort Merge Join..." << endl;
      join_tids = col1->sort_merge_join(col2);
    } else if (join_alg == NESTED_LOOP_JOIN) {
      if (!quiet) cout << "Perform Nested Loop Join..." << endl;
      join_tids = col1->nested_loop_join(col2);
    } else if (join_alg == HASH_JOIN) {
      if (!quiet) cout << "Perform Hash Join..." << endl;
      if (col1->size() < col2->size()) {
        join_tids = col1->hash_join(col2);
      } else {
        join_tids = col2->hash_join(col1);  // hash table is build for smaller
                                            // relation, which is the object the
                                            // method is called on
        std::swap(join_tids->first, join_tids->second);
      }
    } else if (join_alg == PARALLEL_HASH_JOIN) {
      if (col1->size() < col2->size()) {
        join_tids = col1->parallel_hash_join(col2, 6);
      } else {
        join_tids = col2->parallel_hash_join(
            col1, 6);  // hash table is build for smaller relation, which is the
                       // object the method is called on
        std::swap(join_tids->first, join_tids->second);
      }
    } else {
      COGADB_FATAL_ERROR("Unknown join algorithm!", "");
    }
  } else {
    COGADB_FATAL_ERROR("Called unimplemented method!", "");
  }
  assert(join_tids != NULL);

  TableSchema result_schema;
  result_schema = mergeTableSchemas(table1->getSchema(), join_column_table1,
                                    table2->getSchema(), join_column_table2);

  assert(join_tids->first->size() == join_tids->second->size());

  TablePtr result_table = BaseTable::createResultTable(
      table1, join_column_table1, table2, join_column_table2, join_tids,
      result_schema, mat_stat, "pk_fk_join", proc_spec);
  return result_table;
}

const TablePtr BaseTable::fetch_join(TablePtr filtered_pk_table,
                                     const std::string& join_column_table1,
                                     TablePtr fk_table,
                                     const std::string& join_column_table2,
                                     const FetchJoinParam& param) {
  if (!filtered_pk_table) {
    cout << "Fatal Error: in BaseTable::pk_fk_join(): input table pointer "
            "(table1) is NULL!"
         << endl;
    return TablePtr();
  }
  if (!fk_table) {
    cout << "Fatal Error: in BaseTable::pk_fk_join(): input table pointer "
            "(table2) is NULL!"
         << endl;
    return TablePtr();
  }

  //            hype::ProcessingDeviceID id=hype::PD0;
  ////            if(comp_dev==GPU) id=hype::PD1;
  //            ProcessorSpecification proc_spec(id);
  //            FetchJoinParam param(proc_spec);

  LookupTablePtr lookup_table =
      boost::dynamic_pointer_cast<LookupTable>(filtered_pk_table);
  // ColumnPtr filtered_pk_column =
  // lookup_table->getColumnbyName(join_column_table1);
  const std::vector<LookupColumnPtr> lookup_columns =
      lookup_table->getLookupColumns();
  // fetch join assumes a prefiltered table, so has to be a lookup table,
  // and no other operation may have been applied
  assert(lookup_columns.size() == 1);
  PositionListPtr pk_table_tids = lookup_columns.front()->getPositionList();
  TablePtr pk_table = lookup_columns.front()->getTable();
  assert(pk_table != NULL);

  JoinIndexPtr join_index = JoinIndexes::instance().getJoinIndex(
      pk_table, join_column_table1, fk_table, join_column_table2);

  PositionListPtr placed_pk_table_tids =
      copy_if_required(pk_table_tids, getMemoryID(param.proc_spec));
  JoinIndexPtr placed_join_index =
      copy_if_required(join_index, getMemoryID(param.proc_spec));

  //            //PositionListPtr matching_tids_fk_table =
  //            fetchMatchingTIDsFromJoinIndex(join_index, pk_table_tids);
  //            PositionListPairPtr join_tids =
  //            fetchJoinResultFromJoinIndex(join_index, pk_table_tids);
  //            assert(join_tids!=NULL);
  // PositionListPtr matching_tids_fk_table =
  // fetchMatchingTIDsFromJoinIndex(join_index, pk_table_tids);

  ProcessorBackend<TID>* backend =
      ProcessorBackend<TID>::get(param.proc_spec.proc_id);
  PositionListPairPtr join_tids =
      backend->fetch_join(placed_join_index, placed_pk_table_tids, param);

  if (!join_tids) return TablePtr();

  //            if(comp_dev==CPU){
  //                join_tids = fetchJoinResultFromJoinIndex(placed_join_index,
  //                placed_pk_table_tids);
  //                assert(join_tids!=NULL);
  //            }else if(comp_dev==GPU){
  //
  //                COGADB_FATAL_ERROR("Currently unimplemented","");
  //
  ////                COGADB_EXECUTE_GPU_OPERATOR("Fetch_Join");
  ////
  ////                gpu::GPU_PositionlistPtr gpu_pk_table_tids =
  /// gpu::copy_PositionList_host_to_device(pk_table_tids);
  ////                gpu::GPU_JoinIndexPtr gpu_join_index =
  /// GPU_Column_Cache::instance().getGPUJoinIndex(join_index);
  ////                gpu::GPU_PositionListPairPtr gpu_tid_pairs =
  /// gpu::GPU_Operators::fetchJoinResultFromJoinIndex(gpu_join_index,
  /// gpu_pk_table_tids);
  ////                //                        PositionListPtr
  /// reference_matching_fact_table_tids=copy_PositionList_device_to_host(gpu_matching_fact_table_tids);
  ////                // assert(reference_matching_fact_table_tids!=NULL);
  ////                //
  /// assert((*reference_matching_fact_table_tids)==(*matching_fact_table_tids));
  ////                //input_tids =
  /// gpu::copy_PositionList_device_to_host(pos_list_op->getResult_GPU_PositionList());
  ////
  ////                //check whether GPU operator was successfull
  ////                if(!gpu_tid_pairs){
  ////                    //workaround, where variable has_aborted_ is in an
  /// operator
  ////                    bool has_aborted_=false;
  ////                    //ok, GPU operator aborted, execute operator on CPU
  ////                    COGADB_ABORT_GPU_OPERATOR("Fetch_Join");
  ////                    join_tids = fetchJoinResultFromJoinIndex(join_index,
  /// pk_table_tids);
  ////                    assert(join_tids!=NULL);
  ////                }else{
  ////                    join_tids =
  /// gpu::copy_PositionListPair_device_to_host(gpu_tid_pairs);
  ////                    assert(join_tids!=NULL);
  ////
  ////#ifdef VALIDATE_GPU_RESULTS_ON_CPU
  ////                     PositionListPairPtr cpu_join_tids =
  /// fetchJoinResultFromJoinIndex(join_index, pk_table_tids);
  ////                     if(*cpu_join_tids->first != *join_tids->first){
  ////                         COGADB_FATAL_ERROR("Error in Result of GPU Fetch
  /// Join Detected!","");
  ////                     }
  ////                     if(*cpu_join_tids->second != *join_tids->second){
  ////                         COGADB_FATAL_ERROR("Error in Result of GPU Fetch
  /// Join Detected!","");
  ////                     }
  ////#endif
  ////
  ////                }
  //
  //            }else{
  //                COGADB_FATAL_ERROR("Unkown Processing Device Type!","");
  //            }

  TableSchema result_schema;
  result_schema = mergeTableSchemas(pk_table->getSchema(), join_column_table1,
                                    fk_table->getSchema(), join_column_table2);

  assert(join_tids->first->size() == join_tids->second->size());

  TablePtr result_table = BaseTable::createResultTable(
      pk_table, join_column_table1, fk_table, join_column_table2, join_tids,
      result_schema, LOOKUP, "pk_fk_join", param.proc_spec);
  return result_table;
}

const TablePtr BaseTable::crossjoin(TablePtr table1, TablePtr table2,
                                    MaterializationStatus mat_stat) {
  if (!table1) {
    cout << "Fatal Error: in BaseTable::crossjoin(): input table pointer "
            "(table1) is NULL!"
         << endl;
    return TablePtr();
  }
  if (!table2) {
    cout << "Fatal Error: in BaseTable::crossjoin(): input table pointer "
            "(table2) is NULL!"
         << endl;
    return TablePtr();
  }
  Timestamp begin, end;
  begin = getTimestamp();

  hype::ProcessingDeviceID id = hype::PD0;
  ProcessorSpecification proc_spec(id);

  PositionListPairPtr join_tids(new PositionListPair());
  assert(join_tids != NULL);
  join_tids->first = createPositionList();
  join_tids->second = createPositionList();
  // perform cross join by generating tids lists
  size_t join_column1_size = table1->getNumberofRows();
  size_t join_column2_size = table2->getNumberofRows();
  for (unsigned int i = 0; i < join_column1_size; i++) {
    for (unsigned int j = 0; j < join_column2_size; j++) {
      join_tids->first->push_back(i);
      join_tids->second->push_back(j);
    }
  }
  // check whether tid lsits are of same length
  assert(join_tids->first->size() == join_tids->second->size());
  // create schema of new table
  TableSchema result_schema;
  result_schema =
      mergeTableSchemas(table1->getSchema(), "", table2->getSchema(), "");

  TablePtr result_table;

  // materialize Table
  if (mat_stat == MATERIALIZE) {
    result_table =
        TablePtr(new Table(string("crossjoin( ") + table1->getName() +
                               string(",") + table2->getName() + " )",
                           result_schema));

    const std::vector<ColumnPtr>& columns_ = table1->getColumns();
    const std::vector<ColumnPtr>& tab_columns_ = table2->getColumns();
    // ensure tables are loaded in-memory
    loadColumnsInMainMemory(columns_);
    loadColumnsInMainMemory(tab_columns_);
    TID index_table1, index_table2;
    for (unsigned int i = 0; i < join_tids->first->size(); i++) {
      index_table1 = (*(join_tids->first))[i];
      index_table2 = (*(join_tids->second))[i];
      Tuple t;
      for (unsigned int i = 0; i < columns_.size(); i++)
        t.push_back(columns_[i]->get(index_table1));

      for (unsigned int i = 0; i < tab_columns_.size(); i++)
        t.push_back(tab_columns_[i]->get(index_table2));

      result_table->insert(t);
    }
  } else if (mat_stat == LOOKUP) {
    if (!quiet) cout << "create Lookup Table for resulting Table..." << endl;
    string result_lookup_table_name = string("Lookup_Table_crossjoin( ") +
                                      table1->getName() + string(",") +
                                      table2->getName() + " )";
    // consider join as "two" semi joins
    LookupTablePtr lookup_table_for_table1 = createLookupTableforUnaryOperation(
        string("AGGREGATED( ") + table1->getName() + " )", table1,
        join_tids->first, proc_spec);
    LookupTablePtr lookup_table_for_table2 = createLookupTableforUnaryOperation(
        string("AGGREGATED( ") + table2->getName() + " )", table2,
        join_tids->second, proc_spec);
    assert(lookup_table_for_table1 != NULL);
    assert(lookup_table_for_table2 != NULL);
    // concatenate LookupTables
    result_table = LookupTable::concatenate(result_lookup_table_name,
                                            *lookup_table_for_table1,
                                            *lookup_table_for_table2);
  } else {
    cout << "Error! Unkown Materialization Option!" << endl;
    return TablePtr();  ////error, return NULL pointer
  }
  end = getTimestamp();
  assert(end >= begin);
  if (print_time_measurement)
    cout << "Time for CrossJoin: " << end - begin << "ns ("
         << double(end - begin) / 1000000 << "ms)" << endl;
  return result_table;
}

const TablePtr BaseTable::sort(TablePtr table, const std::string& column_name,
                               SortOrder order, MaterializationStatus mat_stat,
                               ComputeDevice comp_dev) {
  if (!table) {
    cout << "Fatal Error: in BaseTable::sort(): input table pointer is NULL!"
         << endl;
    return TablePtr();
  }
  if (!table->isMaterialized()) {
    table = table->materialize();
  }

  // TODO DEBUG CODE entfernen
  if (!quiet && debug) {
    std::cout << "Single Sort getting called" << std::endl;
    std::cout << "Is table marterialized?: " << table->isMaterialized()
              << std::endl;
  }
  // end

  Timestamp begin, end;
  begin = getTimestamp();

  hype::ProcessingDeviceID proc_dev_id = hype::PD0;
  if (comp_dev == GPU) proc_dev_id = hype::PD1;
  SortParam sort_param(ProcessorSpecification(proc_dev_id), order,
                       true);  // sort stable

  ColumnPtr column = table->getColumnbyName(column_name);
  if (column.get() == NULL) {
    // TODO Debug Code entfernen --> if wieder aktivieren
    if (!quiet && debug) {
      table->printSchema();
      std::cout << "Columns of table: " << std::endl;
      const std::vector<ColumnPtr>& columns = table->getColumns();
      for (size_t i = 0; i < columns.size(); ++i) {
        std::cout << columns[i]->getName() << std::endl;
      }
      table->print();
    }

    // endDEbug
    assert(column.get() != NULL);
    COGADB_FATAL_ERROR("Error! Could not look up Column: " << column_name, "");
    //				cout << "Error! Could not look up Column: " <<
    // column_name
    //<<
    // endl;
    //				exit(-1);
  }

  PositionListPtr ids;

  if (comp_dev == CPU) {
    ids = column->sort(sort_param);
  } else if (comp_dev == GPU) {
    COGADB_FATAL_ERROR("Called unimplemented method!", "");
    //                                COGADB_EXECUTE_GPU_OPERATOR("GPU_Sort_Algorithm");
    //				//copy original column to GPU RAM
    //				gpu::GPU_Base_ColumnPtr device_column =
    // GPU_Column_Cache::instance().getGPUColumn(column);
    ////copy_column_host_to_device(*this);
    //				//compute result tid list on GPU
    //				gpu::GPU_PositionlistPtr device_tids =
    // gpu::GPU_Operators::sort(device_column, sort_param.order);
    //				if(!device_tids ){
    //					//cout << "SORT Operator on GPU failed!"
    //<<
    // endl;
    //					//cout << "For column: '" << column_name
    //<<
    //"'
    // with
    //"
    //<<
    // column->size() << " elements" << endl;
    //                                       //workaround, where variable
    //                                       has_aborted_ is in an operator
    //                                       bool has_aborted_=false;
    //                                        //ok, GPU operator aborted,
    //                                        execute operator on CPU
    //					COGADB_ABORT_GPU_OPERATOR("GPU_Sort_Algorithm");
    //					ids = column->sort(sort_param);
    //				}else{
    //                                        //copy result tid list back to CPU
    //                                        RAM
    //                                        ids =
    //                                        copy_PositionList_device_to_host(device_tids);
    //                                        if(!ids){
    //                                                cout << "COPY Operation
    //                                                from GPU to CPU RAM
    //                                                failed!" << endl;
    //                                                cout << "For column: '" <<
    //                                                column_name << "' with "
    //                                                << column->size() << "
    //                                                elements" << endl;
    //                                                return TablePtr ();
    //                                                //return NULL Pointer
    //                                        }
    //                                }
  }

  assert(ids != NULL);
  assert(column->size() == ids->size());  // number of input values is equal to
                                          // number of output values

  // const std::vector<ColumnPtr>& unsortedTableColumns = table->getColumns();

  TablePtr result_table =
      createResultTable(table, ids, mat_stat, "sort", sort_param.proc_spec);
  if (result_table) {
    result_table->setName(table->getName());
  }

  // result_table->materialize();

  // const std::vector<ColumnPtr>& sortedTableColumns =
  // result_table->getColumns();

  // TODO Debug Code entfernen
  /*if(!quiet && debug) {
          std::cout << "unsortedTable has " << unsortedTableColumns.size()
          << " Columns and sortedTable has " << sortedTableColumns.size() <<
  std::endl;
  }*/

  // assert(unsortedTableColumns.size() == sortedTableColumns.size());

  end = getTimestamp();
  assert(end >= begin);
  if (print_time_measurement)
    cout << "Time for Sorting: " << end - begin << "ns ("
         << double(end - begin) / 1000000 << "ms)" << endl;
  return result_table;
}

const TablePtr BaseTable::sort(TablePtr table,
                               const std::list<std::string>& column_names,
                               SortOrder order, MaterializationStatus mat_stat,
                               ComputeDevice comp_dev) {
  if (!table) {
    cout << "Fatal Error: in BaseTable::sort(): input table pointer is NULL!"
         << endl;
    return TablePtr();
  }
  // sort(A1,..,An)
  if (!quiet && verbose) {
    cout << "Sorting after: ";
    std::list<std::string>::const_iterator cit;
    for (cit = column_names.begin(); cit != column_names.end(); ++cit) {
      cout << *cit << ",";
    }
    cout << std::endl;
  }

  std::cout << "Schema: ";
  table->printSchema();
  std::cout << "Columns of table at start of sort: " << std::endl;
  const std::vector<ColumnPtr>& columns = table->getColumns();
  for (size_t i = 0; i < columns.size(); ++i) {
    std::cout << columns[i]->getName() << std::endl;
  }

  if (column_names.empty()) return table;

  table->copyColumnsInMainMemory();
  // check whether we have just one column to sort: sort(An)
  if (column_names.size() == 1) {
    return BaseTable::sort(table, column_names.front(), order, mat_stat,
                           comp_dev);
  }

  // ok, no special case found, use generic algorithm
  std::list<std::string> tmp_column_names(column_names);
  tmp_column_names.pop_front();
  // sort(Ai+1,..,An))

  // TODO delete debug Code
  // std::cout << "Before actual sorting: table looks like this"
  //<< std::endl;
  // table->print();

  table = sort(table, tmp_column_names, order, mat_stat, comp_dev);

  // std::cout << "After actual sorting: table looks like this"
  //<< std::endl;

  // sort(Ai,..,An))
  // table->print();
  return BaseTable::sort(table, column_names.front(), order, mat_stat,
                         comp_dev);
}

const TablePtr BaseTable::sort(TablePtr table,
                               const std::list<SortAttribute>& sort_attributes,
                               MaterializationStatus mat_stat,
                               ComputeDevice comp_dev) {
  if (!table) {
    cout << "Fatal Error: in BaseTable::sort(): input table pointer is NULL!"
         << endl;
    return TablePtr();
  }

  if (sort_attributes.empty()) return table;

  if (!quiet && verbose) {
    cout << "Sorting after...";
    std::list<SortAttribute>::const_iterator iter;
    std::string asc("ASCENDING");
    std::string desc("DESCENDING");

    for (iter = sort_attributes.begin(); iter != sort_attributes.end();
         iter++) {
      cout << "Attribute: " << (*iter).first << " in SortOrder ";
      if ((*iter).second == ASCENDING) {
        cout << asc << endl;
      } else {
        cout << desc << endl;
      }
    }
  }

  table->copyColumnsInMainMemory();
  // if there is only attribute to sort
  if (sort_attributes.size() == 1) {
    SortAttribute attr = sort_attributes.front();
    return BaseTable::sort(table, attr.first, attr.second, mat_stat, comp_dev);
  }

  std::list<SortAttribute>::const_reverse_iterator rev_iter =
      sort_attributes.rbegin();
  TablePtr tab = BaseTable::sort(table, (*rev_iter).first, (*rev_iter).second,
                                 mat_stat, comp_dev);
  rev_iter++;
  for (; rev_iter != sort_attributes.rend(); ++rev_iter) {
    SortAttribute attr = *rev_iter;
    tab = BaseTable::sort(tab, attr.first, attr.second, mat_stat, comp_dev);
  }

  return tab;
}

const std::pair<ColumnPtr, ColumnPtr> cpu_basic_groupby(
    ColumnPtr key_col, ColumnPtr value_col, AggregationMethod agg_meth) {
  std::pair<ColumnPtr, ColumnPtr> result_;
  //            if(value_col->getType()==INT){
  //                result_=
  //                reduce_by_keys(shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<int>
  //                > (key_col),
  //                                       shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<int>
  //                                       > (value_col),
  //                                       agg_meth);
  //            }else if(value_col->getType()==FLOAT){
  //                result_=
  //                reduce_by_keys(shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<int>
  //                > (key_col),
  //                                       shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<float>
  //                                       > (value_col),
  //                                       agg_meth);
  //            }else{
  ////                assert(key_col->size()>0);
  ////                assert(value_col->size()>0);
  ////
  /// shared_pointer_namespace::shared_ptr<DictionaryCompressedColumn<std::string>
  ///> compressed_col =
  /// shared_pointer_namespace::dynamic_pointer_cast<DictionaryCompressedColumn<std::string>
  ///> (host_column);
  ////                assert(host_col != NULL);
  ////                vector<uint32_t>& compressed_values =
  /// compressed_col->getCompressedValues();
  ////                uint32_t* host_ptr = &compressed_values[0];
  //                result_=
  //                reduce_by_keys(shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<int>
  //                > (key_col),
  //                                       shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<std::string>
  //                                       > (value_col),
  //                                       agg_meth);
  //
  //                //COGADB_FATAL_ERROR("Unsupported data type for
  //                aggregation!","");
  //            }

  COGADB_FATAL_ERROR("UNIMPLEMENTED METHOD", "");
  return result_;
}

const TablePtr BaseTable::groupby(TablePtr table,
                                  const std::string& grouping_column,
                                  const std::string& aggregation_column,
                                  const std::string& result_column_name,
                                  AggregationMethod agg_meth,
                                  ComputeDevice comp_dev) {
  if (!table) {
    cout << "Fatal Error: in BaseTable::groupby(): input table pointer is NULL!"
         << endl;
    return TablePtr();
  }

  ColumnPtr key_col = table->getColumnbyName(grouping_column);
  if (key_col == NULL) {
    cout << "Error! Could not look up Column: " << grouping_column << endl;
    cout << "In File " << __FILE__ << ":" << __LINE__ << endl;
    // table->print();
    cout << "Schema of Input Table: " << endl;
    table->printSchema();
    exit(-1);
  }
  ColumnPtr value_col = table->getColumnbyName(aggregation_column);
  if (value_col == NULL) {
    cout << "Error! Could not look up Column: " << aggregation_column << endl;
    cout << "In File " << __FILE__ << ":" << __LINE__ << endl;
    // table->print();
    cout << "Schema of Input Table: " << endl;
    table->printSchema();
    exit(-1);
  }

  ColumnPtr result_key_col;
  ColumnPtr result_value_col;

  if (comp_dev == CPU) {
    // tids=column->selection(value_for_comparison, comp, CPU);

    //				std::pair<ColumnPtr,ColumnPtr> result_;
    //                                if(value_col->getType()==INT){
    //                                    result_=
    //                                    reduce_by_keys(shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<int>
    //                                    > (key_col),
    //                                                           shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<int>
    //                                                           > (value_col),
    //                                                           agg_meth);
    //                                }else if(value_col->getType()==FLOAT){
    //                                    result_=
    //                                    reduce_by_keys(shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<int>
    //                                    > (key_col),
    //                                                           shared_pointer_namespace::static_pointer_cast<ColumnBaseTyped<float>
    //                                                           > (value_col),
    //                                                           agg_meth);
    //                                }
    //
    //				result_key_col = result_.first;
    //				result_value_col = result_.second;

    std::pair<ColumnPtr, ColumnPtr> result =
        cpu_basic_groupby(key_col, value_col, agg_meth);
    result_key_col = result.first;
    result_value_col = result.second;

  } else if (comp_dev == GPU) {
    COGADB_FATAL_ERROR("Called unimplemented Method!", "");
    // copy original column to GPU RAM
    //                                COGADB_EXECUTE_GPU_OPERATOR("GPU_Groupby_Algorithm");
    //
    //				gpu::GPU_Base_ColumnPtr dev_key_col   =
    // GPU_Column_Cache::instance().getGPUColumn(key_col);
    //				gpu::GPU_Base_ColumnPtr dev_value_col =
    // GPU_Column_Cache::instance().getGPUColumn(value_col);
    //
    ////				assert(dev_key_col!=NULL);
    ////				assert(dev_value_col!=NULL);
    //
    //				//std::pair<gpu::GPU_Base_ColumnPtr,gpu::GPU_Base_ColumnPtr>
    // result = gpu::GPU_Operators::groupby(dev_key_col,
    // dev_value_col,agg_meth);
    //
    //                                gpu::GPU_ColumnPairPtr result =
    //                                gpu::GPU_Operators::groupby(dev_key_col,
    //                                dev_value_col,agg_meth);
    //                                if(result){
    //                                    result_key_col =
    //                                    gpu::copy_column_device_to_host(result->first);
    //                                    result_value_col =
    //                                    gpu::copy_column_device_to_host(result->second);
    //
    //                                }else{
    //                                     //workaround, where variable
    //                                     has_aborted_ is in an operator
    //                                     bool has_aborted_=false;
    //                                    //GPU operator failed, execute
    //                                    COGADB_ABORT_GPU_OPERATOR("GPU_Groupby_Algorithm");
    //                                    std::pair<ColumnPtr,ColumnPtr>  result
    //                                    = cpu_basic_groupby(key_col,
    //                                    value_col, agg_meth);
    //                                    result_key_col = result.first;
    //                                    result_value_col = result.second;
    //
    //                                }

    //				assert(result.first  != NULL);
    //				assert(result.second != NULL);

    //				result_key_col =
    // gpu::copy_column_device_to_host(result.first);
    //				result_value_col =
    // gpu::copy_column_device_to_host(result.second);
  }

  //			TableSchema schema;
  //			schema.push_back( Attribut(VARCHAR,grouping_column) );
  //			schema.push_back( Attribut(INT,aggregation_column) );
  //			TablePtr result_table(new Table(string("groupby(
  //")+table->getName()+string(".")+grouping_column+" )",schema));
  //

  assert(result_key_col != NULL);
  assert(result_value_col != NULL);
  result_value_col->setName(result_column_name);
  std::vector<ColumnPtr> columns;
  columns.push_back(result_key_col);
  columns.push_back(result_value_col);
  TablePtr result_table(new Table(string("groupby( ") + table->getName() +
                                      string(".") + grouping_column + " )",
                                  columns));

  return result_table;
}

AggregationResult aggregate(ColumnGroupingKeysPtr grouping_keys,
                            ColumnPtr aggregation_column,
                            const std::string& result_column_name,
                            const AggregationParam& agg_param) {
  if (!quiet)
    std::cout << "Aggregate Column "
              << ""
              << " using aggregation function "
              << util::getName(agg_param.agg_func) << " with "
              << util::getName(agg_param.agg_alg) << " aggregation"
              << std::endl;

  // place columns
  ColumnPtr placed_aggregation_column =
      copy_if_required(aggregation_column, agg_param.proc_spec);
  if (!placed_aggregation_column) return AggregationResult();

  //                if(hype::util::isCoprocessor(agg_param.proc_spec.proc_id)
  //                   && agg_param.agg_func!=MIN
  //                   && agg_param.agg_func!=MAX
  //                   && agg_param.agg_func!=COUNT){
  //                    if(aggregation_column->getType()!=DOUBLE){
  //                        ColumnPtr placed_double_aggregation_column =
  //                        placed_aggregation_column->convertToDenseValueDoubleColumn(agg_param.proc_spec);
  //                        if(!placed_double_aggregation_column) return
  //                        AggregationResult();
  //                        placed_aggregation_column=placed_double_aggregation_column;
  //                        if(!quiet)
  //                            std::cout << "Converted Column " <<
  //                            placed_aggregation_column->getName() << " to
  //                            double column!" << std::endl;
  //                    }else{
  //                        if(!quiet)
  //                            std::cout << "Column " <<
  //                            placed_aggregation_column->getName() << " is
  //                            already a double column!" << std::endl;
  //                    }
  //                }else{
  //                    if(!quiet)
  //                        std::cout << "I will NOT convert Column " <<
  //                        placed_aggregation_column->getName() << " to double
  //                        column!" << std::endl;
  //                }
  // ColumnPtr col;
  AggregationResult result = placed_aggregation_column->aggregateByGroupingKeys(
      grouping_keys, agg_param);

  if (result.second) result.second->setName(result_column_name);

  return result;
}

/* specialized function in case we have no grouping columns*/
const TablePtr aggregate(TablePtr table, const GroupbyParam& param) {
  assert(param.grouping_columns.empty());

  vector<ColumnPtr> result_columns;
  AggregationFunctions::const_iterator cit;

  for (cit = param.aggregation_functions.begin();
       cit != param.aggregation_functions.end(); ++cit) {
    if (!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
      cout << "Group by grouping_keys for column: " << cit->first << endl;

    ColumnPtr column_to_aggregate = table->getColumnbyName(cit->first);
    ColumnPtr placed_column_to_aggregate =
        copy_if_required(column_to_aggregate, cit->second.proc_spec);
    if (!placed_column_to_aggregate) return TablePtr();

    AggregationResult agg_result =
        placed_column_to_aggregate->aggregate(cit->second);

    ColumnPtr result = agg_result.second;
    if (!result) return TablePtr();
    result->setName(cit->second.new_column_name);

    result_columns.push_back(result);
  }

  TablePtr result(
      new Table(string("groupby(") + table->getName() + ")", result_columns));
  return result;
}

void aggregate_thread(ColumnGroupingKeysPtr grouping_keys,
                      ColumnPtr aggregation_column,
                      const std::string& result_column_name,
                      const AggregationParam& agg_param,
                      AggregationResult* result) {
  assert(result != NULL);
}

int compareTupleValues(vector<ColumnPtr>& grouping_columns, int j, int k) {
  for (unsigned int i = 0; i < grouping_columns.size(); ++i) {
    if (!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
      cout << "Compare Attribute "
           << grouping_columns[i]->getName();  // << endl;
    int ret = grouping_columns[i]->compareValuesAtIndexes(j, k);
    if (!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
      cout << " Result: " << ret << endl;
    if (ret) return ret;
  }
  if (!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
    cout << "Tuples are Equal and belong to the same group!" << endl;
  return 0;
}

vector<ColumnPtr> finalizeGroupingColumns(
    vector<ColumnPtr> grouping_column_ptrs, PositionListPtr position_list,
    const GatherParam& param) {
  vector<ColumnPtr> distinct_groups;  //(grouping_column_ptrs.size());

  //            vector<ColumnPtr> placed_grouping_column_ptrs;

  for (unsigned int i = 0; i < grouping_column_ptrs.size(); ++i) {
    ColumnPtr placed_col =
        copy_if_required(grouping_column_ptrs[i], param.proc_spec);
    if (!placed_col) {
      COGADB_ERROR("Could not finalize Groupingcolumns! Processor ID: "
                       << (int)param.proc_spec.proc_id,
                   "");
      return vector<ColumnPtr>();
    }
    // LookupArray(grouping_column_ptrs[i]->getName(),
    // grouping_column_ptrs[i]->getType(), grouping_column_ptrs[i],
    // position_list);
    ColumnPtr col = placed_col->gather(position_list, param);
    distinct_groups.push_back(col);
  }
  return distinct_groups;
}

std::string getTableName(const ColumnPtr col) {
  if (!col) return std::string();
  std::list<std::pair<ColumnPtr, TablePtr> > columns =
      DataDictionary::instance().getColumnsforColumnName(col->getName());
  // we assume unique column names
  assert(columns.size() <= 1);
  if (columns.size() == 1) {
    return columns.front().second->getName();
  } else {
    return std::string();
  }
}

bool is_derived_from_primary_key_column(const ColumnPtr col) {
  if (!col) return false;
  std::list<std::pair<ColumnPtr, TablePtr> > columns =
      DataDictionary::instance().getColumnsforColumnName(col->getName());
  // we assume unique column names
  assert(columns.size() <= 1);
  if (columns.size() == 1) {
    return columns.front().second->hasPrimaryKeyConstraint(col->getName());
  } else {
    return false;
  }
}

const vector<ColumnPtr> createMinimalEquivalentGroupingSet(
    const vector<ColumnPtr>& grouping_column_ptrs) {
  typedef std::map<std::string, ColumnPtr> Map;

  Map key_columns;

  vector<ColumnPtr> minimal_grouping_set;
  /* First find out all columns that have the primary key property,
   and store them in the key_columns map. */
  for (size_t i = 0; i < grouping_column_ptrs.size(); ++i) {
    assert(grouping_column_ptrs[i] != NULL);
    Map::const_iterator cit =
        key_columns.find(getTableName(grouping_column_ptrs[i]));
    if (cit == key_columns.end()) {
      if (is_derived_from_primary_key_column(grouping_column_ptrs[i])) {
        minimal_grouping_set.push_back(grouping_column_ptrs[i]);
        key_columns.insert(std::make_pair(getTableName(grouping_column_ptrs[i]),
                                          grouping_column_ptrs[i]));
      }
    }
  }
  /* Second pass, here we only add columns to the grouping set if we
   have no key column of that table in our grouping set. */
  for (size_t i = 0; i < grouping_column_ptrs.size(); ++i) {
    assert(grouping_column_ptrs[i] != NULL);

    Map::const_iterator cit =
        key_columns.find(getTableName(grouping_column_ptrs[i]));
    if (cit == key_columns.end()) {
      minimal_grouping_set.push_back(grouping_column_ptrs[i]);
      if (!quiet && verbose && debug) {
        std::cout << "Adding column '" << grouping_column_ptrs[i]->getName()
                  << "' from table '" << getTableName(grouping_column_ptrs[i])
                  << "' to minimal grouping set!" << std::endl;
      }
    } else {
      bool is_in_minimal_grouping_set = false;
      for (size_t j = 0; j < minimal_grouping_set.size(); ++j) {
        if (minimal_grouping_set[j] == grouping_column_ptrs[i]) {
          is_in_minimal_grouping_set = true;
        }
      }
      if (!is_in_minimal_grouping_set) {
        std::cout << "Omitting grouping column '"
                  << grouping_column_ptrs[i]->getName() << "' from table '"
                  << getTableName(grouping_column_ptrs[i])
                  << "' since key column '" << cit->second->getName()
                  << "' comes before this grouping column!" << std::endl;
      }
    }
    if (is_derived_from_primary_key_column(grouping_column_ptrs[i])) {
      key_columns.insert(std::make_pair(getTableName(grouping_column_ptrs[i]),
                                        grouping_column_ptrs[i]));
    }
  }

  return minimal_grouping_set;
}

const TablePtr BaseTable::groupby(TablePtr table, const GroupbyParam& param) {
  if (!table) {
    cout << "Fatal Error: in BaseTable::groupby(): input table pointer is NULL!"
         << endl;
    return TablePtr();
  }

  //            AggregationAlgorithm agg_alg = HASH_BASED_AGGREGATION;
  //            //SORT_BASED_AGGREGATION;
  //            AggregationAlgorithm agg_alg = SORT_BASED_AGGREGATION;
  //            bool requires_stable_sort=false;
  bool cannot_group_by_varchar_column = false;
  bool omit_sorting_for_sort_based_aggregation = false;
  AggregationAlgorithm agg_alg =
      param.aggregation_functions.front().second.agg_alg;

  {
    AggregationFunctions::const_iterator cit;
    for (cit = param.aggregation_functions.begin();
         cit != param.aggregation_functions.end(); ++cit) {
      if (cit->second.agg_func == AGG_CONCAT_BASES) {
        //                    requires_stable_sort=true;
        cannot_group_by_varchar_column = true;
      }
    }
  }

  if (table->getNumberofRows() == 0) {
    // create empty table of the expected schema
    TableSchema schema;
    {
      std::list<std::string>::const_iterator cit;
      vector<ColumnPtr> grouping_column_ptrs;
      for (cit = param.grouping_columns.begin();
           cit != param.grouping_columns.end(); ++cit) {
        ColumnPtr col = table->getColumnbyName(*cit);
        assert(col != NULL);
        schema.push_back(Attribut(col->getType(), col->getName()));
        // grouping_column_ptrs.push_back(col);
      }
    }
    //                std::list<std::pair<string,AggregationMethod>
    //                >::const_iterator cit;
    AggregationFunctions::const_iterator cit;
    for (cit = param.aggregation_functions.begin();
         cit != param.aggregation_functions.end(); ++cit) {
      ColumnPtr col = table->getColumnbyName(cit->first);
      assert(col != NULL);
      schema.push_back(Attribut(col->getType(), col->getName()));
    }
    TablePtr empty_table(
        new Table(string("groupby(") + table->getName() + ")", schema));
    return empty_table;
  }

  if (param.grouping_columns.empty()) {
    return aggregate(table, param);
  } else if (param.grouping_columns.size() == 1 && table->isMaterialized()) {
    ColumnPtr col = table->getColumnbyName(param.grouping_columns.front());
    assert(col != NULL);
    if (col->getColumnStatistics().statistics_up_to_date_) {
      if (col->getColumnStatistics().is_sorted_ascending_) {
        omit_sorting_for_sort_based_aggregation = true;
        std::cout << "Omit Sorting of grouping column: '" << col->getName()
                  << "'" << std::endl;
      }
    }
  }

  // create grouping key column
  vector<ColumnPtr> grouping_column_ptrs;
  GroupingColumns::const_iterator cit;
  for (cit = param.grouping_columns.begin();
       cit != param.grouping_columns.end(); ++cit) {
    if (!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug) cout << *cit << ",";
    ColumnPtr col = table->getColumnbyName(*cit);
    assert(col != NULL);
    if (col->getType() == VARCHAR && cannot_group_by_varchar_column) {
      COGADB_FATAL_ERROR(
          "Cannot group by VARCHAR column ("
              << col->getName()
              << ") when using certain genome aggregations functions!",
          "");
    }
    grouping_column_ptrs.push_back(col);
  }

  vector<ColumnPtr> minimal_grouping_set =
      createMinimalEquivalentGroupingSet(grouping_column_ptrs);

  // place grouping_column_ptrs on correct processor memory

  // try bitpacked grouping first, which is much faster than the generic
  // algorithm
  ColumnGroupingKeysPtr grouping_keys =
      CDK::aggregation::computeColumnGroupingKeys(minimal_grouping_set,
                                                  param.proc_spec);

  PositionListPtr tids_for_sorted_table;
  if (grouping_keys) {
    if (agg_alg == SORT_BASED_AGGREGATION &&
        !omit_sorting_for_sort_based_aggregation) {
      tids_for_sorted_table = grouping_keys->sort(param.proc_spec);
      if (!tids_for_sorted_table) return TablePtr();
      TablePtr tmp = CoGaDB::createLookupTableforUnaryOperation(
          "", table, tids_for_sorted_table, param.proc_spec);
      if (!tmp) return TablePtr();
      table = tmp;
      {  // rebuild grouping columns
        grouping_column_ptrs.clear();
        GroupingColumns::const_iterator cit;
        for (cit = param.grouping_columns.begin();
             cit != param.grouping_columns.end(); ++cit) {
          if (!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
            cout << *cit << ",";
          ColumnPtr col = table->getColumnbyName(*cit);
          assert(col != NULL);
          grouping_column_ptrs.push_back(col);
        }
        //                            minimal_grouping_set =
        //                            createMinimalEquivalentGroupingSet(grouping_column_ptrs);
        //                            grouping_keys =
        //                            CDK::aggregation::computeColumnGroupingKeys(minimal_grouping_set,
        //                            param.proc_spec);
        //                            grouping_keys->print(std::cout);
      }
    }
  } else {
    COGADB_ERROR("Cannot apply optimized groupby on processor with ID "
                     << (int)param.proc_spec.proc_id,
                 "");
    return TablePtr();
  }

  vector<ColumnPtr> result_columns;
  {
    AggregationFunctions::const_iterator cit;
    // compute the aggregation functions in the query
    // we can parallelize this loop to get more performance for groupby's with
    // multiple aggregations
    //                #pragma omp parallel for
    for (cit = param.aggregation_functions.begin();
         cit != param.aggregation_functions.end(); ++cit) {
      if (!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
        cout << "Group by grouping_keys for column: " << cit->first << endl;

      ColumnPtr column_to_aggregate = table->getColumnbyName(cit->first);
      ColumnPtr placed_column_to_aggregate =
          copy_if_required(column_to_aggregate, cit->second.proc_spec);
      if (!placed_column_to_aggregate) return TablePtr();

      // write TID array for grouping columns only once
      AggregationParam new_param(cit->second);
      if (cit == param.aggregation_functions.begin() &&
          grouping_column_ptrs.front()->getName() != "_GROUPING_COLUMN") {
        new_param.write_group_tid_array = true;
      }

      AggregationResult agg_result =
          aggregate(grouping_keys, placed_column_to_aggregate, cit->first,
                    new_param);  // cit->second);

      ColumnPtr result = agg_result.second;
      if (!result) return TablePtr();
      result->setName(cit->second.new_column_name);

      if (cit == param.aggregation_functions.begin() &&
          grouping_column_ptrs.front()->getName() != "_GROUPING_COLUMN") {
        // result_columns.push_back(tmp->getColumnbyName("grouping_keys"));
        ColumnPtr grouping_column =
            agg_result.first;  // tmp->getColumnbyName("grouping_keys");
        if (!grouping_column) return TablePtr();
        //                       assert(grouping_column!=NULL);
        grouping_column = copy_if_required(grouping_column, hype::PD_Memory_0);
        if (!grouping_column) return TablePtr();
        //                       if(!CoGaDB::quiet && CoGaDB::verbose &&
        //                       CoGaDB::debug) grouping_column->print();
        // assert(grouping_column->getType()==INT);
        // assert(grouping_column->isMaterialized());
        if (!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
          cout << "Type: " << typeid(*grouping_column.get()).name();
        // exit(0);
        // shared_pointer_namespace::shared_ptr<Column<int> > int_column =
        // shared_pointer_namespace::dynamic_pointer_cast<Column<int> >(
        // grouping_column);
        //                        Column<GroupingKeys::value_type>* int_column =
        //                        dynamic_cast<Column<GroupingKeys::value_type>*
        //                        >(grouping_column.get());
        //                        assert(int_column!=NULL);
        //                        PositionListPtr tids =createPositionList();
        //                        for(size_t i=0;i<int_column->size();i++){
        //                            tids->push_back((TID)((*int_column)[i]));
        //                        }
        //                        //tids->insert(tids->begin(),int_column->getContent().begin(),int_column->getContent().end());
        //                        if(!CoGaDB::quiet && CoGaDB::verbose &&
        //                        CoGaDB::debug)
        //                        {
        //                            cout << "Group TIDS" << endl;
        //                            for(size_t i=0;i<tids->size();i++){
        //                                cout << "tid: " <<  (*tids)[i] <<
        //                                endl;  //<< " original: " <<
        //                                TID((*int_column)[i])  << endl;
        //                            }
        //                        }
        PositionListPtr tids =
            boost::dynamic_pointer_cast<PositionList>(grouping_column);
        assert(tids != NULL);

        ProcessorSpecification ps(hype::PD0);
        GatherParam gather_param(ps);

        vector<ColumnPtr> final_grouping_columns =
            finalizeGroupingColumns(grouping_column_ptrs, tids, gather_param);
        if (final_grouping_columns.empty()) return TablePtr();
        result_columns.insert(result_columns.begin(),
                              final_grouping_columns.begin(),
                              final_grouping_columns.end());
      }
      result_columns.push_back(result);
    }
  }

  TablePtr result(
      new Table(string("groupby(") + table->getName() + ")", result_columns));
  return result;
}

/***************** Aggregation Functions *****************/

TablePtr BaseTable::ColumnConstantOperation(
    TablePtr tab, const std::string& col_name, const boost::any& value,
    const std::string& result_col_name, const AlgebraOperationParam& param) {
  if (!tab) {
    cout << "Error! In BaseTable::ColumnConstantOperation(): Invalid TablePtr!"
         << endl;
    cout << "In File " << __FILE__ << " Line: " << __LINE__ << endl;
    return TablePtr();
  }
  // assert(comp_dev==CPU);
  ColumnPtr col = tab->getColumnbyName(col_name);
  if (!col) {
    cout << "Error! In BaseTable::ColumnConstantOperation(): Column '"
         << col_name << "' not found in table '" << tab->getName() << "'"
         << endl;
    cout << "In File " << __FILE__ << " Line: " << __LINE__ << endl;
    return TablePtr();
  }
  assert(col->getType() != VARCHAR);

  ColumnPtr placed_col1 = copy(col, getMemoryID(param.proc_spec));
  if (!placed_col1) return TablePtr();

  ColumnPtr result_col = placed_col1->column_algebra_operation(value, param);
  if (!result_col) return TablePtr();

  result_col->setName(result_col_name);
  TablePtr ret_tab = tab;

  if (tab->isMaterialized()) {
    // create a LookupTable with same content as original table
    PositionListPtr tids = createPositionList();

    for (size_t i = 0; i < tab->getNumberofRows(); i++) {
      tids->push_back(i);
    }
    LookupTablePtr result_lookup_table = createLookupTableforUnaryOperation(
        string("lookup(") + tab->getName() + ")", tab, tids, param.proc_spec);
    ret_tab = result_lookup_table;
  }
  // append column to Lookup Table
  ret_tab->addColumn(result_col);
  return ret_tab;
}

TablePtr BaseTable::ColumnAlgebraOperation(TablePtr tab,
                                           const std::string& col1_name,
                                           const std::string& col2_name,
                                           const std::string& result_col_name,
                                           const AlgebraOperationParam& param) {
  if (!tab) {
    cout << "Error! In BaseTable::ColumnAlgebraOperation(): Invalid TablePtr!"
         << endl;
    cout << "In File " << __FILE__ << " Line: " << __LINE__ << endl;
    return TablePtr();
  }

  ColumnPtr col1 = tab->getColumnbyName(col1_name);
  // if(!col1) return false;
  if (!col1) {
    cout << "Error! In BaseTable::ColumnConstantOperation(): Column '"
         << col1_name << "' not found in table '" << tab->getName() << "'"
         << endl;
    cout << "In File " << __FILE__ << " Line: " << __LINE__ << endl;
    return TablePtr();
  }
  ColumnPtr col2 = tab->getColumnbyName(col2_name);
  // if(!col2) return false;
  if (!col2) {
    cout << "Error! In BaseTable::ColumnConstantOperation(): Column '"
         << col2_name << "' not found in table '" << tab->getName() << "'"
         << endl;
    cout << "In File " << __FILE__ << " Line: " << __LINE__ << endl;
    return TablePtr();
  }

  ColumnPtr placed_col1 = copy_if_required(col1, param.proc_spec);
  if (!placed_col1) return TablePtr();
  ColumnPtr placed_col2 = copy_if_required(col2, param.proc_spec);
  if (!placed_col2) return TablePtr();

  ColumnPtr result_col =
      placed_col1->column_algebra_operation(placed_col2, param);
  if (!result_col) return TablePtr();

  result_col->setName(result_col_name);
  TablePtr ret_tab = tab;

  if (tab->isMaterialized()) {
    // create a LookupTable with same content as original table
    PositionListPtr tids = createPositionList();

    for (size_t i = 0; i < tab->getNumberofRows(); i++) {
      tids->push_back(i);
    }
    LookupTablePtr result_lookup_table = createLookupTableforUnaryOperation(
        string("lookup(") + tab->getName() + ")", tab, tids, param.proc_spec);
    ret_tab = result_lookup_table;
  }
  // append column to Lookup Table
  ret_tab->addColumn(result_col);
  return ret_tab;
}

/* Utility Functions */

bool BaseTable::hasColumn(const std::string& column_name) {
  TableSchema::const_iterator it, result_it, end;
  result_it = schema_.end();

  for (it = schema_.begin(), end = schema_.end(); it != end; ++it) {
    std::string current_column = it->second;

    if (boost::count(current_column, '.') == 0) {
      current_column = this->getName() + "." + it->second;
    }

    if (compareAttributeReferenceNames(current_column, column_name)) {
      result_it = it;
      break;
    }
  }

  if (result_it != end) {
    return true;
  } else {
    return false;
  }
}

bool BaseTable::hasColumn(const AttributeReference& attr) {
  std::string column_name = CoGaDB::toString(attr);
  TableSchema::const_iterator it, result_it, end;
  result_it = schema_.end();

  for (it = schema_.begin(), end = schema_.end(); it != end; ++it) {
    std::string current_column = it->second;

    if (boost::count(current_column, '.') == 0) {
      current_column = this->getName() + "." + it->second;
    }

    if (!isFullyQualifiedColumnIdentifier(current_column)) {
      std::string attribute_type_identifier;
      std::string attribute_name;
      uint32_t version;
      if (!parseColumnIndentifierName(current_column, attribute_type_identifier,
                                      attribute_name, version)) {
        return false;
      }
      version = attr.getVersion();
      std::string fully_qualified_name;
      fully_qualified_name = attribute_type_identifier;
      fully_qualified_name += ".";
      fully_qualified_name += attribute_name;
      fully_qualified_name += ".";
      fully_qualified_name += boost::lexical_cast<std::string>(version);
      current_column = fully_qualified_name;
    }
    if (current_column == column_name) {
      result_it = it;
      break;
    }
  }

  if (result_it != end) {
    return true;
  } else {
    return false;
  }
}

bool BaseTable::setPrimaryKeyConstraint(const std::string& column_name) {
  ColumnPtr col = this->getColumnbyName(column_name);
  if (!col) return false;
  return col->setPrimaryKeyConstraint();
}
bool BaseTable::hasPrimaryKeyConstraint(const std::string& column_name) const
    throw() {
  ColumnPtr col = this->getColumnbyName(column_name);
  if (!col) return false;
  return col->hasPrimaryKeyConstraint();
}
bool BaseTable::hasForeignKeyConstraint(const std::string& column_name) const
    throw() {
  ColumnPtr col = this->getColumnbyName(column_name);
  if (!col) return false;
  return col->hasForeignKeyConstraint();
}
bool BaseTable::setForeignKeyConstraint(
    const std::string& foreign_key_column_name,
    const std::string& primary_key_column_name,
    const std::string& primary_key_table_name) {  // const ForeignKeyConstraint&
  // prim_foreign_key_reference){
  ColumnPtr col = this->getColumnbyName(foreign_key_column_name);
  if (!col) return false;
  ForeignKeyConstraint fk_constr(primary_key_column_name,
                                 primary_key_table_name,
                                 foreign_key_column_name, this->getName());
  return col->setForeignKeyConstraint(fk_constr);
}
const ForeignKeyConstraint* BaseTable::getForeignKeyConstraint(
    const std::string& column_name) {
  ColumnPtr col = this->getColumnbyName(column_name);
  if (!col) return NULL;
  if (this->hasForeignKeyConstraint(column_name)) {
    return &col->getForeignKeyConstraint();
  } else {
    return NULL;
  }
}
std::vector<const ForeignKeyConstraint*> BaseTable::getForeignKeyConstraints() {
  std::vector<const ForeignKeyConstraint*> result;
  TableSchema::iterator it;
  for (it = schema_.begin(); it != schema_.end(); ++it) {
    const ForeignKeyConstraint* fk_constr =
        this->getForeignKeyConstraint(it->second);
    if (fk_constr) {
      result.push_back(fk_constr);
    }
  }
  return result;
}

bool BaseTable::renameColumns(const RenameList& rename_list) {
  RenameList::const_iterator cit;
  unsigned int number_of_renamed_columns = 0;
  for (cit = rename_list.begin(); cit != rename_list.end(); ++cit) {
    std::vector<ColumnPtr> cols = this->getColumns();
    ColumnPtr col;
    for (size_t i = 0; i < cols.size(); ++i) {
      if (cols[i]->getName() == cit->first) {
        col = cols[i];
      }
    }
    //                col = getColumnbyName(cit->first);
    if (!col) {
      COGADB_ERROR(std::string("Could not find Column ") + cit->first +
                       std::string(" in table ") + this->getName(),
                   "");
      return false;
    }
    // rename column
    col->setName(cit->second);
    // rename column in schema
    TableSchema::iterator it;
    for (it = schema_.begin(); it != schema_.end(); ++it) {
      if (it->second == cit->first) {
        it->second = cit->second;
        number_of_renamed_columns++;
      }
    }
  }
  if (number_of_renamed_columns == rename_list.size()) {
    return true;
  } else {
    return false;
  }
}

TablePtr BaseTable::AddConstantValueColumnOperation(
    TablePtr tab, const std::string& col_name, AttributeType type,
    const boost::any& value, const ProcessorSpecification& proc_spec) {
  if (!tab) {
    COGADB_ERROR(
        "Error! In BaseTable::AddConstantValueColumnOperation(): Invalid "
        "TablePtr!",
        "");
    return TablePtr();
  }
  //
  //            hype::ProcessingDeviceID id=hype::PD0;
  ////                if(comp_dev==GPU) id=hype::PD1;
  //            ProcessorSpecification proc_spec(id);

  ColumnPtr result_column;
  TablePtr ret_tab = tab;
  /* \todo This is ugly and should be replaced with a generic function that
   * sets all elements of a column to a specific value
   */
  try {
    if (type == INT) {
      boost::shared_ptr<Column<int> > col(
          new Column<int>(col_name, type, tab->getNumberofRows(),
                          boost::any_cast<int>(value), getMemoryID(proc_spec)));
      result_column = col;
    } else if (type == FLOAT) {
      boost::shared_ptr<Column<float> > col(new Column<float>(
          col_name, type, tab->getNumberofRows(), boost::any_cast<float>(value),
          getMemoryID(proc_spec)));
      result_column = col;
    } else if (type == DOUBLE) {
      boost::shared_ptr<Column<double> > col(new Column<double>(
          col_name, type, tab->getNumberofRows(),
          boost::any_cast<double>(value), getMemoryID(proc_spec)));
      result_column = col;
    } else if (type == VARCHAR) {
      boost::shared_ptr<Column<string> > col(new Column<string>(
          col_name, type, tab->getNumberofRows(),
          boost::any_cast<string>(value), getMemoryID(proc_spec)));
      result_column = col;
    } else {
      COGADB_FATAL_ERROR("Invalid Column Type: " << type, "");
    }
  } catch (boost::bad_any_cast& e) {
    COGADB_FATAL_ERROR(e.what(), "");
    return TablePtr();
  } catch (std::bad_alloc& e) {
    COGADB_ERROR(e.what(), "");
    return TablePtr();
  }
  if (tab->isMaterialized()) {
    // create a LookupTable with same content as original table
    PositionListPtr tids =
        createAscendingPositionList(tab->getNumberofRows(), proc_spec);
    if (!tids) return TablePtr();
    LookupTablePtr result_lookup_table = createLookupTableforUnaryOperation(
        string("lookup(") + tab->getName() + ")", tab, tids, proc_spec);
    ret_tab = result_lookup_table;
  }
  ret_tab->addColumn(result_column);
  return ret_tab;
}

const TablePtr BaseTable::user_defined_function(
    TablePtr table, const std::string& function_name,
    const std::vector<boost::any>& function_parameters,
    const ProcessorSpecification& proc_spec) {
  UserDefinedFunctionPtr udf =
      UserDefinedFunctions::instance().get(function_name);
  assert(udf != NULL);
  return (*udf)(table, function_name, function_parameters, proc_spec);
}

const LookupTablePtr createLookupTableforUnaryOperation(
    const std::string& lookup_table_name, const TablePtr table,
    PositionListPtr ids, const ProcessorSpecification& proc_spec) {
  if (!table || !ids) return LookupTablePtr();

  LookupTablePtr result_table;
  if (table->isMaterialized()) {
    // table is materialized, just create view (Lookup Table) on table
    // (aggregation not neccessary)
    if (!quiet) cout << "passed table is Materialized..." << endl;

    std::vector<LookupColumnPtr> lookup_columns;
    LookupColumnPtr lookup_col(new LookupColumn(table, ids));
    assert(lookup_col != NULL);

    lookup_columns.push_back(lookup_col);
    ColumnVectorPtr lookup_arrays_lookup_table = lookup_col->getLookupArrays();

    result_table = LookupTablePtr(
        new LookupTable(lookup_table_name, table->getSchema(), lookup_columns,
                        *lookup_arrays_lookup_table));

  } else {
    if (!quiet) cout << "passed table is LookupTable..." << endl;
    // is lookuptable, aggregate results
    LookupTablePtr input_lookup_table =
        shared_pointer_namespace::static_pointer_cast<LookupTable>(table);
    assert(input_lookup_table != NULL);

    // place ids
    PositionListPtr placed_ids = copy_if_required(ids, proc_spec);
    if (!placed_ids) {
      return LookupTablePtr();
    }
    LookupTablePtr result_lookup_table =
        LookupTable::aggregate(lookup_table_name, *input_lookup_table,
                               LookupColumn(table, placed_ids), proc_spec);
    result_table = result_lookup_table;
    // no concatenation necessary because this is unary operation
  }
  return result_table;
}

bool isSameTableSchema(const TableSchema& schema,
                       const TableSchema& candidate) {
  if (schema == candidate) {
    return true;
  } else {
    if (!quiet && debug) {
      std::cout << "The schemas differ." << std::endl;

      TableSchema refSchema = schema;
      TableSchema candidateSchema = candidate;

      for (std::list<Attribut>::const_iterator
               it = refSchema.begin(),
               it_that = candidateSchema.begin();
           it != refSchema.end(); ++it, ++it_that) {
        std::cout << "[This] [Attribute] is " << util::getName((*it).first)
                  << " [Name] is "
                  << "\"" << (*it).second << "\"" << std::endl;
        std::cout << "[Candidate] [Attribute] is "
                  << util::getName((*it_that).first) << " [Name] is "
                  << "\"" << (*it_that).second << "\"" << std::endl;
        if ((*it) == (*it_that)) {
          std::cout << "Is Equal!" << std::endl;
        } else {
          std::cout << "Is Unequal!" << std::endl;
        }
      }
    }

    return false;
  }
}

void renameFullyQualifiedNamesToUnqualifiedNames(TablePtr table) {
  if (!table) return;

  TableSchema schema = table->getSchema();
  TableSchema::const_iterator cit;
  RenameList rename_list;
  for (cit = schema.begin(); cit != schema.end(); ++cit) {
    rename_list.push_back(RenameEntry(
        cit->second, getAttributeNameFromColumnIdentifier(cit->second)));
  }
  table->renameColumns(rename_list);
}

void expandUnqualifiedColumnNamesToQualifiedColumnNames(TablePtr table) {
  if (!table) return;
  RenameList rename_list;
  TableSchema schema = table->getSchema();
  TableSchema::const_iterator cit;
  for (cit = schema.begin(); cit != schema.end(); ++cit) {
    std::string name = cit->second;
    if (isPlainAttributeName(
            cit->second)) {  // boost::count(cit->second,'.')==0){
      name = table->getName();
      name += ".";
      name += cit->second;
    }
    rename_list.push_back(RenameEntry(cit->second, name));
  }
  table->renameColumns(rename_list);
}

}  // end namespace CogaDB
