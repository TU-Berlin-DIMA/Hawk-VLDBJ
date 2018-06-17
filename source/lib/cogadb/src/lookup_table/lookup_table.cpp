
#include <boost/lexical_cast.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <core/lookup_array.hpp>
#include <core/table.hpp>
#include <iomanip>
#include <lookup_table/lookup_table.hpp>

#include <core/copy_function_factory.hpp>
#include <util/utility_functions.hpp>
#include "util/getname.hpp"
#include "util/hardware_detector.hpp"

using namespace std;

namespace CoGaDB {

LookupTable::LookupTable(const std::string& name, const TableSchema& schema,
                         const std::vector<LookupColumnPtr>& lookup_columns,
                         const std::vector<ColumnPtr> lookup_arrays,
                         const std::vector<ColumnPtr> dense_value_arrays)
    : BaseTable(name, schema),
      lookup_columns_(lookup_columns),
      lookup_arrays_to_real_columns_(lookup_arrays),
      appended_dense_value_columns_(dense_value_arrays),
      all_columns_() {
  all_columns_.insert(all_columns_.begin(),
                      lookup_arrays_to_real_columns_.begin(),
                      lookup_arrays_to_real_columns_.end());
  all_columns_.insert(all_columns_.end(), appended_dense_value_columns_.begin(),
                      appended_dense_value_columns_.end());
}

LookupTable::~LookupTable() {}
/***************** utility functions *****************/

const ColumnVectorPtr LookupTable::getLookupArrays() {
  ColumnVectorPtr lookup_array_vec_ptr(new ColumnVector());

  TableSchema::const_iterator it;

  for (it = this->schema_.begin(); it != this->schema_.end(); it++) {
    bool matching_column_found = false;
    for (unsigned int i = 0; i < lookup_columns_.size(); i++) {
      ColumnPtr col = lookup_columns_[i]->getLookupArrayforColumnbyName(*it);
      if (col) {
        lookup_array_vec_ptr->push_back(col);
        matching_column_found = true;
      }
    }
    if (!matching_column_found) {
      cout << "FATAL ERROR! could not find column " << it->second
           << " in any of the tables indexed by Lookup Table " << this->name_
           << endl;
      return ColumnVectorPtr();  // return NULL Pointer
    }
  }

  return lookup_array_vec_ptr;
}

const ColumnVector& LookupTable::getDenseValueColumns() {
  return appended_dense_value_columns_;
}

bool LookupTable::copyColumnsInMainMemory() {
  ColumnVectorPtr new_lookup_arrays(new ColumnVector());

  for (unsigned int i = 0; i < lookup_columns_.size(); i++) {
    PositionListPtr tids = lookup_columns_[i]->getPositionList();
    tids = copy_if_required(tids, hype::PD_Memory_0);
    if (!tids) {
      COGADB_FATAL_ERROR(
          "Failed to transfer column back "
              << lookup_columns_[i]->getPositionList()->getName() << " ("
              << util::getName(
                     lookup_columns_[i]->getPositionList()->getColumnType())
              << ") to main memory!",
          "");
    }
    lookup_columns_[i] =
        LookupColumnPtr(new LookupColumn(lookup_columns_[i]->getTable(), tids));
    ColumnVectorPtr vec = lookup_columns_[i]->getLookupArrays();
    new_lookup_arrays->insert(new_lookup_arrays->end(), vec->begin(),
                              vec->end());
  }
  lookup_arrays_to_real_columns_ = *new_lookup_arrays;
  all_columns_.clear();
  all_columns_.insert(all_columns_.begin(),
                      lookup_arrays_to_real_columns_.begin(),
                      lookup_arrays_to_real_columns_.end());
  all_columns_.insert(all_columns_.end(), appended_dense_value_columns_.begin(),
                      appended_dense_value_columns_.end());
  return true;
}

void LookupTable::print() { std::cout << toString() << std::endl; }

/*! tries to store table in database*/
bool LookupTable::store(const std::string&) {
  return false;  // not allowed for Lookup columns
}

/*! tries to load table form database*/
bool LookupTable::load(TableLoaderMode loader_mode) {
  return false;  // not allowed for Lookup columns
}

bool LookupTable::loadDatafromFile(std::string, bool quiet) {
  return false;  // not allowed for Lookup columns
}

const TablePtr LookupTable::materialize() const {
  std::vector<ColumnPtr> materialized_columns;
  for (unsigned int i = 0; i < all_columns_.size(); i++) {
    ColumnPtr ptr;
    ptr = all_columns_[i]->materialize();
    if (!ptr) {
      COGADB_FATAL_ERROR("Materialization of Table Failed! For Column: "
                             << all_columns_[i]->getName(),
                         "");
    }
    materialized_columns.push_back(ptr);
  }
  TablePtr result_table =
      TablePtr(new Table(string("Materialize( ") + this->getName() + " )",
                         materialized_columns));  // tmp_schema));
  // result_table->printSchema();
  return result_table;
}

bool LookupTable::addColumn(ColumnPtr col) {
  if (!col) return false;
  // this->lookup_arrays_to_real_columns_.push_back(col);
  if (!quiet && verbose)
    cout << "ADD Column " << col->getName() << " to LookupTable "
         << this->getName() << endl;
  appended_dense_value_columns_.push_back(col);
  all_columns_.push_back(col);
  // this->columns_.push_back(col);
  this->schema_.push_back(Attribut(col->getType(), col->getName()));
  return true;
}

/*! \brief aggregates LookupTable wrt Lookup Column, which indexes the
 * LookupTable. IT takes care of copying the LookupColumns and their
 * corresponding PostitionList. Furthermore, it calls functions which createa
 * new LookupArraylist, which represents the columns of the Table (a view, to be
 * more precise) represented by the Lookup Table*/
const LookupTablePtr LookupTable::aggregate(
    const std::string& result_lookup_table_name,
    const LookupTable& lookup_table, const LookupColumn& lookup_col,
    const ProcessorSpecification& proc_spec) {
  std::vector<LookupColumnPtr> new_aggregated_lookup_columns;
  // apply aggregation to each Lookup column
  for (unsigned int i = 0; i < lookup_table.lookup_columns_.size(); i++) {
    //			LookupColumnPtr tmp =
    // shared_pointer_namespace::static_pointer_cast<LookupColumn>
    //(lookup_table_columns[i]);
    LookupColumnPtr col = lookup_table.lookup_columns_[i]->aggregate(
        lookup_col, proc_spec);  // returns new lookup column
    if (!col) return LookupTablePtr();
    new_aggregated_lookup_columns.push_back(col);
  }
  // get all new Lookup Arrays of all new Lookup Columns, so that they can be
  // passed to new Lookup Table
  ColumnVectorPtr new_lookup_arrays(new ColumnVector());
  for (unsigned int i = 0; i < new_aggregated_lookup_columns.size(); i++) {
    ColumnVectorPtr tmp_lookup_arrays =
        new_aggregated_lookup_columns[i]->getLookupArrays();
    assert(tmp_lookup_arrays != NULL);
    // only add Lookup Arrays that are part of the schema
    TableSchema schema = lookup_table.getSchema();
    TableSchema::const_iterator it;
    for (unsigned int i = 0; i < tmp_lookup_arrays->size(); i++) {
      for (it = schema.begin(); it != schema.end(); it++) {
        if ((*tmp_lookup_arrays)[i]->getName() == it->second) {
          new_lookup_arrays->push_back((*tmp_lookup_arrays)[i]);
        }
      }
    }
  }

  return LookupTablePtr(
      new LookupTable(result_lookup_table_name, lookup_table.getSchema(),
                      new_aggregated_lookup_columns, *new_lookup_arrays,
                      lookup_table.appended_dense_value_columns_));
}

const LookupTablePtr LookupTable::concatenate(
    const std::string&, const LookupTable& lookup_table1,
    const LookupTable& lookup_table2) {
  assert(&lookup_table1 != &lookup_table2);  // it makes no sense to concatenate
                                             // the same Lookup Table to itself
  assert(lookup_table1.getNumberofRows() == lookup_table2.getNumberofRows());

  // concatenate Lookupcolumn vector
  std::vector<LookupColumnPtr> new_lookup_columns;  //(new LookupColumn());
  new_lookup_columns.insert(new_lookup_columns.end(),
                            lookup_table1.lookup_columns_.begin(),
                            lookup_table1.lookup_columns_.end());
  new_lookup_columns.insert(new_lookup_columns.end(),
                            lookup_table2.lookup_columns_.begin(),
                            lookup_table2.lookup_columns_.end());
  // concatenate schemas
  TableSchema new_schema;
  new_schema.insert(new_schema.end(), lookup_table1.schema_.begin(),
                    lookup_table1.schema_.end());
  new_schema.insert(new_schema.end(), lookup_table2.schema_.begin(),
                    lookup_table2.schema_.end());
  // concatenate lookup arrays
  ColumnVectorPtr new_lookup_arrays(new ColumnVector());
  new_lookup_arrays->insert(
      new_lookup_arrays->end(),
      lookup_table1.lookup_arrays_to_real_columns_.begin(),
      lookup_table1.lookup_arrays_to_real_columns_.end());
  new_lookup_arrays->insert(
      new_lookup_arrays->end(),
      lookup_table2.lookup_arrays_to_real_columns_.begin(),
      lookup_table2.lookup_arrays_to_real_columns_.end());

  ColumnVectorPtr new_appended_dense_values_arrays(new ColumnVector());
  new_appended_dense_values_arrays->insert(
      new_appended_dense_values_arrays->end(),
      lookup_table1.appended_dense_value_columns_.begin(),
      lookup_table1.appended_dense_value_columns_.end());
  new_appended_dense_values_arrays->insert(
      new_appended_dense_values_arrays->end(),
      lookup_table2.appended_dense_value_columns_.begin(),
      lookup_table2.appended_dense_value_columns_.end());

  return LookupTablePtr(
      new LookupTable(string("concat( ") + lookup_table1.getName() +
                          string(",") + lookup_table2.getName() + " )",
                      new_schema, new_lookup_columns, *new_lookup_arrays,
                      *new_appended_dense_values_arrays));
}

/***************** status report *****************/
bool LookupTable::isMaterialized() const throw() { return false; }

/***************** read and write operations at table level *****************/
const Tuple LookupTable::fetchTuple(const TID& id) const {
  Tuple t;
  for (unsigned int i = 0; i < lookup_arrays_to_real_columns_.size(); i++) {
    if (!lookup_arrays_to_real_columns_[i]->isLoadedInMainMemory()) {
      // load column in memory
      //                this->getColumnbyName(columns_[j]->getName());
      loadColumnFromDisk(lookup_arrays_to_real_columns_[i]);
    }
    t.push_back(lookup_arrays_to_real_columns_[i]->get(id));
  }
  for (unsigned int i = 0; i < appended_dense_value_columns_.size(); i++) {
    t.push_back(appended_dense_value_columns_[i]->get(id));
  }
  return t;  // not allowed for Lookup columns
}

bool LookupTable::insert(const Tuple&) {
  return false;  // not allowed for Lookup columns
}

bool LookupTable::update(const std::string&, const boost::any&) {
  return false;  // not allowed for Lookup columns
}

bool LookupTable::remove(const std::string&, const boost::any&) {
  return false;  // not allowed for Lookup columns
}

bool LookupTable::append(TablePtr table) {
  if (this->schema_ != table->getSchema()) {
    COGADB_FATAL_ERROR("Cannot append to table if schema differs!", "");
    return false;
  }

  if (!table->isMaterialized()) {
    LookupTablePtr input_lookup_table =
        shared_pointer_namespace::static_pointer_cast<LookupTable>(table);
    assert(input_lookup_table != NULL);
    assert(this->lookup_columns_.size() ==
           input_lookup_table->lookup_columns_.size());
    assert(this->appended_dense_value_columns_.size() ==
           input_lookup_table->appended_dense_value_columns_.size());

    for (size_t i = 0; i < input_lookup_table->lookup_columns_.size(); ++i) {
      assert(this->lookup_columns_[i]->getTable() ==
             input_lookup_table->lookup_columns_[i]->getTable());
      bool ret = this->lookup_columns_[i]->append(
          input_lookup_table->lookup_columns_[i]);
      assert(ret == true);
    }

    for (size_t i = 0;
         i < input_lookup_table->appended_dense_value_columns_.size(); ++i) {
      bool ret = this->appended_dense_value_columns_[i]->append(
          input_lookup_table->appended_dense_value_columns_[i]);
      assert(ret == true);
    }

    return true;
  } else {
    COGADB_FATAL_ERROR(
        "Appending Materialized Tables to Intermediate Tables not implemented!",
        "");
    return false;
  }
}

bool LookupTable::replaceColumn(const std::string& column_name,
                                const ColumnPtr new_column) {
  COGADB_FATAL_ERROR("Called unimplemented function!", "");
  return false;
}

const ColumnPtr LookupTable::getColumnbyName(
    const std::string& column_name) const throw() {
  for (unsigned int i = 0; i < lookup_arrays_to_real_columns_.size(); i++) {
    if (compareAttributeReferenceNames(
            lookup_arrays_to_real_columns_[i]->getName(), column_name)) {
      if (!quiet && verbose)
        std::cout << "Found Column: " << column_name << ":  "
                  << lookup_arrays_to_real_columns_[i].get() << std::endl;
      if (!lookup_arrays_to_real_columns_[i]->isLoadedInMainMemory()) {
        loadColumnFromDisk(lookup_arrays_to_real_columns_[i]);
      }
      /* update the access statistics in the stored table */
      for (size_t j = 0; j < lookup_columns_.size(); ++j) {
        /* statistics are updated by getColumnbyName() function*/
        ColumnPtr col =
            lookup_columns_[j]->getTable()->getColumnbyName(column_name);
        if (col) {
          /* column found, statistics were updated, leave for loop*/
          break;
        }
      }

      return lookup_arrays_to_real_columns_[i];
    }
  }
  for (unsigned int i = 0; i < appended_dense_value_columns_.size(); i++) {
    if (appended_dense_value_columns_[i]->getName() == column_name) {
      if (!quiet && verbose)
        std::cout << "Found Column: " << column_name << ":  "
                  << appended_dense_value_columns_[i].get() << std::endl;
      return appended_dense_value_columns_[i];
    }
  }
  this->printSchema();
  COGADB_ERROR("Error: could not find column "
                   << column_name << " in Intermediate Table '" << name_
                   << "'!",
               "");
  return ColumnPtr();  // not found, return NULL Pointer
}

const ColumnPtr LookupTable::getColumnbyId(const unsigned int id) const
    throw() {
  COGADB_FATAL_ERROR("Not implemented!", "");
}

unsigned int LookupTable::getColumnIdbyColumnName(
    const std::string& column_name) const throw() {
  COGADB_FATAL_ERROR("Not implemented!", "");
}

const std::vector<ColumnPtr>& LookupTable::getColumns() const {
  return this->all_columns_;
}

const std::vector<LookupColumnPtr>& LookupTable::getLookupColumns() const {
  return lookup_columns_;
}

}  // end namespace CogaDB
