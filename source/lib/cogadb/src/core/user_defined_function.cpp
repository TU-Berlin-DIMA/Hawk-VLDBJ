

#include <backends/processor_backend.hpp>
#include <boost/make_shared.hpp>
#include <core/column.hpp>
#include <core/user_defined_function.hpp>
#include <limits>

#include "core/variable_manager.hpp"
#include "moderngpu/include/util/static.h"

namespace CoGaDB {

const TablePtr udf_hand_compiled_tpch_q1(
    TablePtr table, const std::string& function_name,
    const std::vector<boost::any>& function_parameters,
    const ProcessorSpecification& proc_spec);

UserDefinedFunctions::UserDefinedFunctions() : map_() {
  // add build in user defined functions
  map_.insert(std::make_pair("LIMIT", &limit));
  map_.insert(std::make_pair("EXTRACT_YEAR", &extract_year));
  map_.insert(std::make_pair("UDF.EXPERIMENTAL.HAND_COMPILED_TPCH_Q1",
                             &udf_hand_compiled_tpch_q1));
}

UserDefinedFunctions& UserDefinedFunctions::instance() {
  static UserDefinedFunctions udfs;
  return udfs;
}

UserDefinedFunctionPtr UserDefinedFunctions::get(
    const std::string& function_name) {
  UDFMap::const_iterator it = map_.find(function_name);
  if (it != map_.end()) {
    return it->second;
  } else {
    COGADB_FATAL_ERROR("Unknown UDF: '" << function_name << "'!", "");
    return NULL;
  }
}

bool UserDefinedFunctions::add(const std::string& function_name,
                               UserDefinedFunctionPtr function) {
  UDFMap::const_iterator it = map_.find(function_name);
  if (it == map_.end()) {
    map_.insert(std::make_pair(function_name, function));
    return true;
  } else {
    COGADB_FATAL_ERROR("User defined function '" << function_name << "'", "");
    return false;
  }
}

/* Build in UDFs */
const TablePtr limit(TablePtr table, const std::string& function_name,
                     const std::vector<boost::any>& function_parameters,
                     const ProcessorSpecification& proc_spec) {
  if (!table) return TablePtr();

  assert(function_parameters.size() == 1);
  assert(!function_parameters.front().empty());
  assert(function_parameters.front().type() == typeid(size_t));
  assert(function_name == "LIMIT");

  size_t num_result_rows = boost::any_cast<size_t>(function_parameters.front());
  num_result_rows = std::min(num_result_rows, table->getNumberofRows());

  ProcessorBackend<TID>* backend =
      ProcessorBackend<TID>::get(proc_spec.proc_id);

  PositionListPtr tids = createPositionList(num_result_rows, proc_spec);
  if (!tids) return TablePtr();
  bool ret = backend->generateAscendingSequence(tids->data(), num_result_rows,
                                                TID(0), proc_spec);
  assert(ret == true);
  TablePtr result =
      BaseTable::createResultTable(table, tids, LOOKUP, "LIMIT", proc_spec);

  return result;
}

const TablePtr top_k_sort(TablePtr table, const std::string& function_name,
                          const std::vector<boost::any>& function_parameters,
                          const ProcessorSpecification& proc_spec) {
  return TablePtr();
}

const TablePtr extract_year(TablePtr table, const std::string& function_name,
                            const std::vector<boost::any>& function_parameters,
                            const ProcessorSpecification& proc_spec) {
  if (!table) {
    COGADB_ERROR("Invalid TablePtr!", "");
    return TablePtr();
  }

  assert(function_name == "EXTRACT_YEAR");
  assert(function_parameters.size() == 2);
  assert(!function_parameters.front().empty());
  assert(function_parameters.front().type() == typeid(std::string));
  assert(!function_parameters.back().empty());
  assert(function_parameters.back().type() == typeid(std::string));

  std::string column_name =
      boost::any_cast<std::string>(function_parameters.front());
  std::string new_column_name =
      boost::any_cast<std::string>(function_parameters.back());

  ColumnPtr col = table->getColumnbyName(column_name);
  if (!col) {
    COGADB_ERROR("Column '" << column_name << "' not found in table '"
                            << table->getName() << "'",
                 "");
    return TablePtr();
  }
  assert(col->getType() == DATE);

  ColumnPtr placed_col = copy(col, getMemoryID(proc_spec));
  if (!placed_col) return TablePtr();

  //            ColumnPtr placed_col2 = copy_if_required(col2, param.proc_spec);
  //            if(!placed_col2) return TablePtr();

  typedef Column<uint32_t> DateDenseValueColumn;
  typedef boost::shared_ptr<DateDenseValueColumn> DateDenseValueColumnPtr;

  DateDenseValueColumnPtr divisor = boost::make_shared<DateDenseValueColumn>(
      "10000", UINT32, getMemoryID(proc_spec));
  //            DateDenseValueColumnPtr divisor =
  //            boost::make_shared<DateDenseValueColumn>("10000",DATE,
  //            getMemoryID(proc_spec));
  if (!divisor) return TablePtr();

  try {
    divisor->resize(placed_col->size());
  } catch (std::bad_alloc& e) {
    COGADB_ERROR(
        "Out of memory for processor memory: " << (int)divisor->getMemoryID(),
        "");
    return TablePtr();
  }

  ProcessorBackend<uint32_t>* backend =
      ProcessorBackend<uint32_t>::get(proc_spec.proc_id);

  bool ret = backend->generateConstantSequence(divisor->data(), divisor->size(),
                                               uint32_t(10000), proc_spec);

  assert(ret == true);

  AlgebraOperationParam param(proc_spec, DIV);

  boost::shared_ptr<ColumnBaseTyped<uint32_t> > typed_source_column =
      boost::dynamic_pointer_cast<ColumnBaseTyped<uint32_t> >(placed_col);
  assert(typed_source_column != NULL);
  DateDenseValueColumnPtr dense_placed_column =
      typed_source_column->copyIntoDenseValueColumn(param.proc_spec);
  //            if(PLAIN_MATERIALIZED==placed_col->getColumnType()){
  //                DateDenseValueColumnPtr dense_placed_column =
  //                boost::dynamic_pointer_cast<DateDenseValueColumn>(placed_col);
  //                assert(dense_placed_column!=NULL);
  //            }else{
  //                boost::shared_ptr<ColumnBaseTyped<T> > typed_source_column =
  //                boost::dynamic_pointer_cast<ColumnBaseTyped<T>
  //                >(placed_col);
  //                assert(typed_source_column!=NULL);
  //                dense_placed_column =
  //                typed_source_column->copyIntoDenseValueColumn(param.proc_spec);
  //            }
  if (!dense_placed_column) return TablePtr();

  if (!backend->column_algebra_operation(dense_placed_column->data(),
                                         divisor->data(),
                                         dense_placed_column->size(), param)) {
    COGADB_ERROR("Column algebra operation failed!", "");
  }

  dense_placed_column->setType(UINT32);
  ColumnPtr result_col =
      dense_placed_column;  // placed_col->column_algebra_operation(ColumnPtr(divisor),
                            // param);
  if (!result_col) return TablePtr();

  result_col->setName(new_column_name);
  TablePtr ret_tab = table;

  if (table->isMaterialized()) {
    // create a LookupTable with same content as original table
    ProcessorBackend<TID>* backend =
        ProcessorBackend<TID>::get(proc_spec.proc_id);

    PositionListPtr tids =
        createPositionList(table->getNumberofRows(), proc_spec);
    if (!tids) return TablePtr();
    bool ret = backend->generateAscendingSequence(
        tids->data(), table->getNumberofRows(), TID(0), proc_spec);
    assert(ret == true);

    //                PositionListPtr tids=createPositionList();
    //
    //                for(size_t i=0;i<tab->getNumberofRows();i++){
    //                    tids->push_back(i);
    //                }

    LookupTablePtr result_lookup_table = createLookupTableforUnaryOperation(
        std::string("lookup(") + table->getName() + ")", table, tids,
        param.proc_spec);
    ret_tab = result_lookup_table;
  }
  // append column to Lookup Table
  ret_tab->addColumn(result_col);
  return ret_tab;

  return TablePtr();
}

const TablePtr hand_compiled_tpch_q1(const size_t& num_elements,
                                     const float* __restrict__ l_extended_price,
                                     const int* __restrict__ l_quantity,
                                     const float* __restrict__ l_discount,
                                     const float* __restrict__ l_tax,
                                     const char* __restrict__ l_returnflag,
                                     const char* __restrict__ l_linestatus,
                                     const uint32_t* __restrict__ l_shipdate) {
  const size_t num_hash_table_entries = std::numeric_limits<uint16_t>::max();

  double* ht_sum_quantity =
      (double*)calloc(num_hash_table_entries, sizeof(double));
  double* ht_sum_base_price =
      (double*)calloc(num_hash_table_entries, sizeof(double));
  double* ht_sum_disc_price =
      (double*)calloc(num_hash_table_entries, sizeof(double));
  double* ht_sum_disc = (double*)calloc(num_hash_table_entries, sizeof(double));
  double* ht_sum_charge =
      (double*)calloc(num_hash_table_entries, sizeof(double));
  double* ht_count_order =
      (double*)calloc(num_hash_table_entries, sizeof(double));
  bool* ht_valid_entries = (bool*)calloc(num_hash_table_entries, sizeof(bool));
  char* ht_returnflag = (char*)calloc(num_hash_table_entries, sizeof(char));
  char* ht_linestatus = (char*)calloc(num_hash_table_entries, sizeof(char));

  for (size_t i = 0; i < num_elements; ++i) {
    if (l_shipdate[i] <= 19980902) {
      TID index = l_returnflag[i] * sizeof(char) + l_linestatus[i];

      double disc_price = l_extended_price[i] * (1 - l_discount[i]);
      double charge = disc_price * (1 - l_tax[i]);

      ht_sum_quantity[index] += l_quantity[i];
      ht_sum_base_price[index] += l_extended_price[i];
      ht_sum_disc_price[index] += disc_price;
      ht_sum_disc[index] += l_discount[i];
      ht_sum_charge[index] += charge;
      ht_count_order[index] += 1;
      ht_valid_entries[index] = true;
      ht_returnflag[index] = l_returnflag[i];
      ht_linestatus[index] = l_linestatus[i];
    }
  }
  //         l_returnflag,
  //        l_linestatus
  boost::shared_ptr<Column<char> > result_col_returnflag =
      boost::make_shared<Column<char> >("L_RETURNFLAG", CHAR);
  boost::shared_ptr<Column<char> > result_col_linestatus =
      boost::make_shared<Column<char> >("L_LINESTATUS", CHAR);

  boost::shared_ptr<Column<double> > result_col_sum_quantity =
      boost::make_shared<Column<double> >("SUM_QUANTITY", DOUBLE);
  boost::shared_ptr<Column<double> > result_col_sum_base_price =
      boost::make_shared<Column<double> >("SUM_BASE_PRICE", DOUBLE);
  boost::shared_ptr<Column<double> > result_col_sum_disc_price =
      boost::make_shared<Column<double> >("SUM_DISC_PRICE", DOUBLE);
  boost::shared_ptr<Column<double> > result_col_sum_charge =
      boost::make_shared<Column<double> >("SUM_CHARGE", DOUBLE);
  boost::shared_ptr<Column<int32_t> > result_col_count_order =
      boost::make_shared<Column<int32_t> >("COUNT_ORDER", INT);

  boost::shared_ptr<Column<double> > result_col_avg_qty =
      boost::make_shared<Column<double> >("AVG_QTY", DOUBLE);
  boost::shared_ptr<Column<double> > result_col_avg_price =
      boost::make_shared<Column<double> >("AVG_PRICE", DOUBLE);
  boost::shared_ptr<Column<double> > result_col_avg_disc =
      boost::make_shared<Column<double> >("AVG_DISC", DOUBLE);

  for (size_t i = 0; i < num_hash_table_entries; ++i) {
    if (ht_valid_entries[i]) {
      result_col_returnflag->insert(ht_returnflag[i]);
      result_col_linestatus->insert(ht_linestatus[i]);
      result_col_sum_quantity->insert(ht_sum_quantity[i]);
      result_col_sum_base_price->insert(ht_sum_base_price[i]);
      result_col_sum_disc_price->insert(ht_sum_disc_price[i]);
      result_col_sum_charge->insert(ht_sum_charge[i]);
      result_col_count_order->insert(ht_count_order[i]);
      result_col_avg_qty->insert(ht_sum_quantity[i] / ht_count_order[i]);
      result_col_avg_price->insert(ht_sum_base_price[i] / ht_count_order[i]);
      result_col_avg_disc->insert(ht_sum_disc[i] / ht_count_order[i]);
    }
  }

  free(ht_sum_quantity);
  free(ht_sum_base_price);
  free(ht_sum_disc);
  free(ht_sum_disc_price);
  free(ht_sum_charge);
  free(ht_count_order);
  free(ht_valid_entries);
  free(ht_returnflag);
  free(ht_linestatus);

  std::vector<ColumnPtr> result_columns;
  result_columns.push_back(result_col_returnflag);
  result_columns.push_back(result_col_linestatus);
  result_columns.push_back(result_col_sum_quantity);
  result_columns.push_back(result_col_sum_base_price);
  result_columns.push_back(result_col_sum_disc_price);
  result_columns.push_back(result_col_sum_charge);
  result_columns.push_back(result_col_avg_qty);
  result_columns.push_back(result_col_avg_price);
  result_columns.push_back(result_col_avg_disc);
  result_columns.push_back(result_col_count_order);

  TablePtr result = boost::make_shared<Table>("", result_columns);

  return result;
}

const TablePtr udf_hand_compiled_tpch_q1(
    TablePtr table, const std::string& function_name,
    const std::vector<boost::any>& function_parameters,
    const ProcessorSpecification& proc_spec) {
  assert(function_name == "UDF.EXPERIMENTAL.HAND_COMPILED_TPCH_Q1");
  assert(function_parameters.size() == 0);

  assert(table != NULL);
  assert(table->getName() == "LINEITEM");
  assert(table->isMaterialized());

  ColumnPtr col_lineitem_extended_price =
      table->getColumnbyName("L_EXTENDEDPRICE");
  ColumnPtr col_lineitem_quantity = table->getColumnbyName("L_QUANTITY");
  ColumnPtr col_lineitem_discount = table->getColumnbyName("L_DISCOUNT");
  ColumnPtr col_lineitem_shipdate = table->getColumnbyName("L_SHIPDATE");
  ColumnPtr col_lineitem_tax = table->getColumnbyName("L_TAX");
  ColumnPtr col_lineitem_returnflag = table->getColumnbyName("L_RETURNFLAG");
  ColumnPtr col_lineitem_linestatus = table->getColumnbyName("L_LINESTATUS");

  boost::shared_ptr<Column<float> > typed_col_lineitem_extended_price =
      boost::dynamic_pointer_cast<Column<float> >(col_lineitem_extended_price);
  boost::shared_ptr<Column<int> > typed_col_lineitem_quantity =
      boost::dynamic_pointer_cast<Column<int> >(col_lineitem_quantity);
  boost::shared_ptr<Column<float> > typed_col_lineitem_discount =
      boost::dynamic_pointer_cast<Column<float> >(col_lineitem_discount);
  boost::shared_ptr<Column<float> > typed_col_lineitem_tax =
      boost::dynamic_pointer_cast<Column<float> >(col_lineitem_tax);
  //            boost::shared_ptr<Column<float> > typed_ =
  //            boost::dynamic_pointer_cast<Column<float> >();
  //            boost::shared_ptr<Column<float> > typed_ =
  //            boost::dynamic_pointer_cast<Column<float> >();
  //            boost::shared_ptr<Column<float> > typed_ =
  //            boost::dynamic_pointer_cast<Column<float> >();

  boost::shared_ptr<Column<char> > typed_col_lineitem_returnflag =
      boost::dynamic_pointer_cast<Column<char> >(col_lineitem_returnflag);
  boost::shared_ptr<Column<char> > typed_col_lineitem_linestatus =
      boost::dynamic_pointer_cast<Column<char> >(col_lineitem_linestatus);

  boost::shared_ptr<Column<uint32_t> > typed_col_lineitem_shipdate =
      boost::dynamic_pointer_cast<Column<uint32_t> >(col_lineitem_shipdate);

  return hand_compiled_tpch_q1(
      table->getNumberofRows(), typed_col_lineitem_extended_price->data(),
      typed_col_lineitem_quantity->data(), typed_col_lineitem_discount->data(),
      typed_col_lineitem_tax->data(), typed_col_lineitem_returnflag->data(),
      typed_col_lineitem_linestatus->data(),
      typed_col_lineitem_shipdate->data());

  //            ColumnPtr col_ = table->getColumnbyName("");
  //            ColumnPtr col_ = table->getColumnbyName("");
  //            ColumnPtr col_ = table->getColumnbyName("");

  //            schema.push_back(Attribut(INT, "L_ORDERKEY")); //"L_ORDERKEY");
  //            schema.push_back(Attribut(INT, "L_PARTKEY")); //l_partkey);
  //            schema.push_back(Attribut(INT, "L_SUPPKEY")); //l_suppkey);
  //            schema.push_back(Attribut(INT, "L_LINENUMBER"));
  //            //l_linenumber);
  //            schema.push_back(Attribut(INT, "L_QUANTITY")); //l_quantity );
  //            schema.push_back(Attribut(FLOAT, "L_EXTENDEDPRICE"));
  //            //l_extendedprice);
  //            schema.push_back(Attribut(FLOAT, "L_DISCOUNT")); //l_discount);
  //            schema.push_back(Attribut(FLOAT, "L_TAX")); //l_tax);
  //            schema.push_back(Attribut(CHAR, "L_RETURNFLAG"));
  //            //l_returnflag);
  //            schema.push_back(Attribut(CHAR, "L_LINESTATUS")); //l_linestatus
  //            );
  //            schema.push_back(Attribut(DATE, "L_SHIPDATE")); //l_shipdate);
  //            schema.push_back(Attribut(DATE, "L_COMMITDATE"));
  //            //l_commitdate);
  //            schema.push_back(Attribut(DATE, "L_RECEIPTDATE"));
  //            //l_receiptdate);
  //            schema.push_back(Attribut(VARCHAR, "L_SHIPINSTRUCT"));
  //            //l_shipinstruct);
  //            schema.push_back(Attribut(VARCHAR, "L_SHIPMODE"));
  //            //l_shipmode);
  //            schema.push_back(Attribut(VARCHAR, "L_COMMENT")); //l_comment);
}

}  // end namespace CoGaDB
