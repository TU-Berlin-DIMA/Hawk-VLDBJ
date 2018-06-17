
#include <boost/algorithm/string.hpp>
#include <boost/any.hpp>
#include <boost/bind.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/tokenizer.hpp>
#include <core/runtime_configuration.hpp>
#include <core/table.hpp>
#include <fstream>
#include <iostream>

#include <util/filesystem.hpp>
#include <util/getname.hpp>
#include <util/iostream.hpp>
#include <util/time_measurement.hpp>
#include <util/types.hpp>

#include <persistence/storage_manager.hpp>

#include <core/foreign_key_constraint.hpp>
// serialization
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/algorithm/count.hpp>
#include <boost/serialization/binary_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

namespace CoGaDB {

using namespace std;

const std::vector<ColumnPtr>& Table::getColumns() const { return columns_; }

Table::Table(const std::string& name, const TableSchema& schema)
    : BaseTable(name, expandToQualifiedColumnNamesIfRequired(schema, name)),
      columns_(),
      name_of_primary_key_column_(),
      foreign_key_constraints_(),
      number_of_rows_(0),
      table_mutex_(),
      column_mutexes_(),
      access_statistics_map_(),
      access_statistics_map_mutex_(),
      path_to_database_() {
  TableSchema::const_iterator it;
  for (it = schema_.begin(); it != schema_.end(); it++) {
    ColumnPtr ptr = createColumn(it->first, it->second);
    if (ptr.get() != NULL) {
      // ptr->setStatusLoadedInMainMemory(false);
      columns_.push_back(ptr);
      column_mutexes_.push_back(boost::make_shared<boost::mutex>());
    }
  }
}

Table::Table(const std::string& name, const TableSchema& schema,
             const CompressionSpecifications& compression_specifications)
    : BaseTable(name, expandToQualifiedColumnNamesIfRequired(schema, name)),
      columns_(),
      name_of_primary_key_column_(),
      foreign_key_constraints_(),
      number_of_rows_(0),
      table_mutex_(),
      column_mutexes_(),
      access_statistics_map_(),
      access_statistics_map_mutex_(),
      path_to_database_() {
  TableSchema::const_iterator it;
  CompressionSpecifications::const_iterator compress_spec;
  for (it = schema_.begin(); it != schema_.end(); it++) {
    ColumnPtr ptr;
    compress_spec = compression_specifications.find(it->second);
    if (compress_spec == compression_specifications.end()) {
      ptr = createColumn(it->first, it->second);
    } else {
      ptr = createColumn(it->first, it->second, compress_spec->second);
    }
    if (ptr.get() != NULL) {
      // tr->setStatusLoadedInMainMemory(false);
      columns_.push_back(ptr);
      column_mutexes_.push_back(boost::make_shared<boost::mutex>());
    }
  }
}

Table::Table(const std::string& name, const std::string& path_to_database,
             bool& error_occured)
    : BaseTable(name, TableSchema()),
      columns_(),
      name_of_primary_key_column_(),
      foreign_key_constraints_(),
      number_of_rows_(0),
      table_mutex_(),
      column_mutexes_(),
      access_statistics_map_(),
      access_statistics_map_mutex_(),
      path_to_database_(path_to_database) {
  // identify exact path we need to load a table from
  using namespace boost::filesystem;
  string dir_path(path_to_database);
  if (!exists(dir_path)) {
    error_occured = true;
    return;
  }
  dir_path += "/";
  dir_path += name_;
  if (!exists(dir_path)) {
    cout << "No file at '" << dir_path << "' Aborting..." << endl;
    error_occured = true;
    return;
  }
  // load meta data from disk
  CompressionSpecifications compression_specifications;
  std::ifstream infile(dir_path.c_str(),
                       std::ios_base::binary | std::ios_base::in);
  boost::archive::binary_iarchive ia(infile);
  ia >> schema_;
  ia >> compression_specifications;
  ia >> name_of_primary_key_column_;
  ia >> foreign_key_constraints_;
  ia >> number_of_rows_;

  infile.close();

  string table_dir_path(path_to_database);
  table_dir_path += "/tables/";
  table_dir_path += this->name_;

  // create columns according to loaded meta data
  TableSchema::const_iterator it;
  CompressionSpecifications::const_iterator compress_spec;
  for (it = schema_.begin(); it != schema_.end(); it++) {
    ColumnPtr ptr;
    compress_spec = compression_specifications.find(it->second);
    if (compress_spec == compression_specifications.end()) {
      ptr = createColumn(it->first, it->second);
    } else {
      ptr = createColumn(it->first, it->second, compress_spec->second);
    }
    if (ptr.get() != NULL) {
      if (!ptr->load(table_dir_path, LOAD_META_DATA_ONLY)) {
        COGADB_FATAL_ERROR(
            "Failed to load column meta data of column " << it->second, "");
      }
      ptr->setStatusLoadedInMainMemory(false);
      columns_.push_back(ptr);
      column_mutexes_.push_back(boost::make_shared<boost::mutex>());
    }
  }

  RenameList rename_list;
  TableSchema schema = this->getSchema();
  TableSchema::const_iterator cit;
  for (cit = schema.begin(); cit != schema.end(); ++cit) {
    std::string name = cit->second;
    if (boost::count(cit->second, '.') == 0) {
      name = this->getName();
      name += ".";
      name += cit->second;
    }
    rename_list.push_back(RenameEntry(cit->second, name));
  }
  this->renameColumns(rename_list);
}

Table::Table(const std::string& name, const std::vector<ColumnPtr>& columns)
    : BaseTable(name, TableSchema()),
      columns_(columns),
      name_of_primary_key_column_(),
      foreign_key_constraints_(),
      number_of_rows_(0),
      table_mutex_(),
      column_mutexes_(),
      access_statistics_map_(),
      access_statistics_map_mutex_(),
      path_to_database_() {
  vector<ColumnPtr>::const_iterator it;
  for (it = columns.begin(); it != columns.end(); it++) {
    schema_.push_back(Attribut((*it)->getType(), (*it)->getName()));
    column_mutexes_.push_back(boost::make_shared<boost::mutex>());
  }
  if (!columns.empty()) {
    if (columns.front()) number_of_rows_ = columns.front()->size();
  }
}

Table::~Table() {}

const TableSchema Table::expandToQualifiedColumnNamesIfRequired(
    const TableSchema& schema, const std::string& table_name) {
  TableSchema schema_;
  TableSchema::const_iterator cit;
  for (cit = schema.begin(); cit != schema.end(); ++cit) {
    std::string qualified_attribute_name = cit->second;
    if (boost::count(cit->second, '.') == 0) {
      qualified_attribute_name = table_name;
      qualified_attribute_name += ".";
      qualified_attribute_name += cit->second;
    }
    schema_.push_back(Attribut(cit->first, qualified_attribute_name));
  }
  return schema_;
}

void Table::print() { std::cout << toString() << std::endl; }

bool Table::store(const std::string& path_to_table_dir) {
  path_to_database_ = path_to_table_dir;

  using namespace boost::filesystem;

  if (!exists(path_to_table_dir)) {
    boost::filesystem::create_directories(path_to_table_dir);
  }

  string table_schema_filename = path_to_table_dir;
  table_schema_filename += "/";
  table_schema_filename += name_;

  string dir_path(path_to_table_dir);
  dir_path += "/tables/";
  dir_path += name_;
  if (!exists(dir_path)) {
    create_directory(dir_path);
  }

  CompressionSpecifications compress_spec =
      this->getCompressionSpecifications();
  // cout << "Storing Table schema of Table '" << this->name_ << "' in File '"
  // << table_schema_filename << "' ..." << endl;
  std::ofstream outfile(table_schema_filename.c_str(),
                        std::ios_base::binary | std::ios_base::out);
  boost::archive::binary_oarchive oa(outfile);

  oa << schema_;
  oa << compress_spec;
  oa << name_of_primary_key_column_;
  oa << foreign_key_constraints_;
  oa << number_of_rows_;

  outfile.flush();
  outfile.close();

  // store all files (columns)
  for (unsigned int i = 0; i < columns_.size(); i++) {
    columns_[i]->store(dir_path);
  }

  return true;
}

bool Table::load(TableLoaderMode loader_mode) {
  using namespace boost::filesystem;
  string dir_path(RuntimeConfiguration::instance().getPathToDatabase());
  if (!exists(dir_path)) return false;
  dir_path += "/tables/";
  dir_path += name_;
  if (!exists(dir_path)) {
    cout << "No directory '" << dir_path << "' Aborting..." << endl;
    return false;
  }

  vector<string> v = getFilesinDirectory(dir_path);

  if (loader_mode == LOAD_ALL_COLUMNS) {
    boost::thread_group threads;

    for (unsigned int i = 0; i < columns_.size(); i++) {
      threads.add_thread(new boost::thread(boost::bind(
          &ColumnBase::load, columns_[i], dir_path, LOAD_ALL_DATA)));
      // TODO must be set in sync with according thread
      columns_[i]->setStatusLoadedInMainMemory(true);
    }
    threads.join_all();

  } else if (loader_mode == LOAD_ELEMENTARY_COLUMNS) {
    // this loads the foreign key constrained columns due to its internal use of
    // getColumnbyName()
    // std::vector<const ForeignKeyConstraint*> constraints =
    // this->getForeignKeyConstraints();
    // this->hasPrimaryKeyConstraint("");
    COGADB_FATAL_ERROR(
        "Loading elementary columns only is currently not supported!", "");
  } else if (loader_mode == LOAD_NO_COLUMNS) {
    // do nothing
  } else {
    COGADB_FATAL_ERROR("Invalid TableLoaderMode!", "");
  }

  return true;
}

bool Table::loadDatafromFile(std::string filepath, bool quiet) {
  std::ifstream fin(filepath.c_str());  //"Makefile");
  std::string buffer;

  unsigned int samplecounter = 0;
  if (!fin.is_open()) {
    cout << "Error: could not open file " << filepath << endl;
    return false;
  }
  while (fin.good()) {
    getline(fin, buffer, '\n');
    // std::cout <<buffer<< std::endl;
    /* ignore lines that start with '#' */
    if (!buffer.empty()) {
      if (buffer[0] == '#') continue;
    }
    // TODO: entfernen
    std::string str = ";;Hello|world||-foo--bar;yow;baz|";
    str = buffer;  // s;

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep("|");
    tokenizer tokens(str, sep);

    unsigned int tokencounter = 0;

    // create tuple from string
    Tuple t;
    for (tokenizer::iterator tok_iter = tokens.begin();
         tok_iter != tokens.end(); ++tok_iter) {
      tokenizer::iterator tmp_tok_it = tok_iter;
      tmp_tok_it++;
      if (tmp_tok_it == tokens.end() && (*tok_iter == "" || *tok_iter == "\r"))
        break;  // if reached last token and token is null value, then process
                // next line

      if (tokencounter >= columns_.size()) {
        COGADB_FATAL_ERROR("Fatal Error! While parsing file '"
                               << filepath << "' by processing line number: "
                               << samplecounter << std::endl
                               << "Line: '" << buffer << "'" << std::endl
                               << "Number Of Tokens: " << tokencounter << endl
                               << "More Tokens found than columns in Table '"
                               << name_ << "'" << endl,
                           "");
        return false;
      }

      // trim whitespaces from input value
      std::string value(*tok_iter);
      boost::algorithm::trim(value);

      AttributeType type = columns_[tokencounter]->getType();

      try {
        if (type == INT) {
          t.push_back(boost::lexical_cast<int>(value));
        } else if (type == CHAR) {
          t.push_back(boost::lexical_cast<char>(value));
        } else if (type == FLOAT) {
          t.push_back(boost::lexical_cast<float>(value));
        } else if (type == DOUBLE) {
          t.push_back(boost::lexical_cast<double>(value));
        } else if (type == DATE) {
          uint32_t date_internal = 0;
          if (!convertStringToInternalDateType(value, date_internal)) {
            COGADB_FATAL_ERROR("Invalid Date!", "");
          }
          t.push_back(date_internal);
        } else if (type == VARCHAR) {
          t.push_back(value);  // is already a string
        } else if (type == BOOLEAN) {
          t.push_back(boost::lexical_cast<bool>(value));
        } else if (type == UINT32) {
          t.push_back(boost::lexical_cast<uint32_t>(value));
        } else if (type == OID) {
          t.push_back(boost::lexical_cast<uint64_t>(value));
        } else if (type == CHAR) {
          t.push_back(boost::lexical_cast<char>(value));
        }
      } catch (boost::bad_lexical_cast& e) {
        COGADB_FATAL_ERROR("Error Processing row "
                               << samplecounter << " column "
                               << columns_[tokencounter]->getName()
                               << ": Type cast failed: '" << *tok_iter << "'",
                           "");
      }

      tokencounter++;
    }

    // cout << "Processed Line: '" << str << "'" << endl;
    if (str != "")  // if processed line is empty, the tuple is empty as well,
                    // so to ignore empty lines, we just do not insert tuples
                    // coming from empty lines
      if (this->insert(t) == false) {
        cout << "Failed to load data from file: '" << filepath << "'" << endl;
        return false;
      }
    if (!quiet && samplecounter % 100000 == 0) {
      cout << "\rProcessed " << samplecounter << " rows...";  // << endl;
      cout.flush();
    }
    samplecounter++;
  }
  fin.close();
  if (!quiet)
    cout << "\rProcessed " << this->getNumberofRows() << " rows..." << endl;

  return true;
}

const TablePtr Table::materialize() const {
  /* this function performs a table copy */
  std::vector<ColumnPtr> materialized_columns;
  for (unsigned int i = 0; i < this->columns_.size(); i++) {
    ColumnPtr ptr;
    ptr = this->columns_[i]->materialize();
    if (!ptr) {
      COGADB_FATAL_ERROR("Materialization of Table Failed! For Column: "
                             << this->columns_[i]->getName(),
                         "");
    }
    materialized_columns.push_back(ptr);
  }
  TablePtr result_table = TablePtr(
      new Table(this->getName(), materialized_columns));  // tmp_schema));
  return result_table;
}

bool Table::addColumn(ColumnPtr col) {
  if (!col) return false;
  this->columns_.push_back(col);
  this->schema_.push_back(Attribut(col->getType(), col->getName()));
  return true;
}

/***************** status report *****************/
size_t Table::getNumberofRows() const throw() { return number_of_rows_; }

void Table::setNumberofRows(size_t _number_of_rows) {
  number_of_rows_ = _number_of_rows;
}

bool Table::isMaterialized() const throw() { return true; }

/***************** read and write operations at table level *****************/
const Tuple Table::fetchTuple(const TID& id) const {
  // TODO: add check the bounding and whether id is valid
  Tuple t;
  for (unsigned int j = 0; j < columns_.size(); j++) {
    {
      boost::lock_guard<boost::mutex> lock(*column_mutexes_[j]);
      if (!columns_[j]->isLoadedInMainMemory()) {
        // load column in memory
        // this->getColumnbyName(columns_[j]->getName());
        // boost::lock_guard<boost::mutex> lock(table_mutex_);
        loadColumnFromDisk(columns_[j]);
      }
    }
    t.push_back(columns_[j]->get(id));
  }
  return t;
}

bool Table::insert(const Tuple& t) {
  if (t.size() != columns_.size()) {
    COGADB_FATAL_ERROR("Fatal Error! Tuple has invalid number of values!"
                           << endl
                           << "Mumber of Columns in Table '" << name_
                           << "': " << columns_.size() << endl
                           << "Number of values in Tuple: " << t.size() << endl
                           << t << endl,
                       "");

    return false;
  }
  bool ret_val = true;
  Tuple::const_iterator it;
  unsigned int i = 0;
  for (it = t.begin(); it != t.end(); it++, i++) {
    { /* Protect for concurrent column loads from different threads. */
      boost::lock_guard<boost::mutex> lock(*column_mutexes_[i]);
      if (!columns_[i]->isLoadedInMainMemory()) {
        // load column in memory
        loadColumnFromDisk(columns_[i]);
      }
    }
    ret_val = columns_[i]->insert(*it);
    if (ret_val == false) {
      COGADB_FATAL_ERROR("Fatal Error! could not insert value in Column "
                             << columns_[i]->getName() << endl,
                         "");
      return false;
    }
  }
  number_of_rows_++;
  return true;
}

bool Table::update(const std::string& attribute_name,
                   const boost::any&) {  // value){
  // TODO: add implementation
  ColumnPtr col = this->getColumnbyName(attribute_name);
  if (col) {
    // return col->update();
  }
  return false;
}

bool Table::remove(const std::string& attribute_name,
                   const boost::any&) {  // value){
  ColumnPtr col = this->getColumnbyName(attribute_name);
  if (col) {
    // return col->update();
  }
  // number_of_rows_--;
  return false;
}

bool Table::append(TablePtr table) {
  if (!table) return false;
  TableSchema schema = table->getSchema();
  assert(this->schema_ == schema);

  this->number_of_rows_ += table->getNumberofRows();
  bool ret = true;
  TableSchema::const_iterator cit;
  for (cit = this->schema_.begin(); cit != this->schema_.end(); ++cit) {
    ColumnPtr this_col = this->getColumnbyName(cit->second);
    ColumnPtr col_to_append = table->getColumnbyName(cit->second);
    ret = ret && this_col->append(col_to_append);
    if (!ret) {
      COGADB_FATAL_ERROR("Append Failed!", "");
    }
  }
  return ret;
}

bool Table::replaceColumn(const std::string& column_name,
                          const ColumnPtr new_column) {
  COGADB_FATAL_ERROR("Called unimplemented function!", "");
  return false;

  //        if(!this->hasColumn(column_name)) return false;
  //        for(size_t i=0;i<columns_.size();++i){
  //            if(columns_[i]->getName()==column_name){
  ////                columns_[i]->eraseFromDisk();
  ////                new_column->store(..);
  //                columns_[i]=new_column;
  //            }
  //        }
}

const ColumnPtr Table::getColumnbyName(const std::string& column_name) const
    throw() {
  return getColumnbyId(getColumnIdbyColumnName(column_name));
}

const ColumnPtr Table::getColumnbyId(const unsigned int id) const throw() {
  if (id >= schema_.size()) {
    return ColumnPtr();
  }

  ColumnPtr column;
  { /* Protect for concurrent column loads from different threads. */
    boost::lock_guard<boost::mutex> lock(*column_mutexes_[id]);

    column = columns_[id];
    if (!column->isLoadedInMainMemory()) {
      using namespace boost::filesystem;
      string dir_path(path_to_database_);
      if (!exists(dir_path)) {
        COGADB_ERROR("Cannot find database in '" << dir_path
                                                 << "': No such directory!",
                     "");
        return ColumnPtr();
      }

      dir_path += "/tables/";
      dir_path += name_;

      if (!exists(dir_path)) {
        COGADB_ERROR("No directory '" << dir_path << "' Aborting...", "");
        return ColumnPtr();
      }

      column->load(dir_path, LOAD_ALL_DATA);
      column->setStatusLoadedInMainMemory(true);
    }
  }

  {
    boost::lock_guard<boost::mutex> lock(access_statistics_map_mutex_);
    access_statistics_map_[column->getName()].number_of_accesses++;
    access_statistics_map_[column->getName()].last_access_timestamp =
        getTimestamp();
  }

  return column;
}

unsigned int Table::getColumnIdbyColumnName(
    const std::string& column_name) const throw() {
  TableSchema::const_iterator it, result_it;
  result_it = schema_.end();
  unsigned int index = 0;
  /* We need to expand the qualified column names and compare them to
   * the column_name to find the correct column. However, this can be very
   * expensive if we always have to compute the fully qualified name.
   * In many cases, the column name is equal to "column_name", just the
   * version information needs to be added. As this comparison is much
   * cheaper, we try to find the correct column in two phases.
   *
   * In the first phase, we try to find the correct column using the
   * inexpensive method that succeeds most of the time. In the second phase,
   * we perform the full check, which will always identify the correct column
   * to read from, but is more expensive.
   */
  for (it = schema_.begin(); it != schema_.end(); ++it, ++index) {
    /* perform first phase */
    if (it->second == column_name || it->second + ".1" == column_name) {
      result_it = it;
      break;
    }
  }

  if (result_it == schema_.end()) {
    /* perform second phase */
    for (it = schema_.begin(), index = 0; it != schema_.end(); ++it, ++index) {
      AttributeReferencePtr attr =
          getAttributeFromColumnIdentifier(column_name);
      /* reference to an actual attribute? */
      if (attr) {
        uint32_t version;
        /* has current table attribute version information? */
        if (getVersionFromColumnIdentifier(it->second, version)) {
          if (it->second == createFullyQualifiedColumnIdentifier(attr)) {
            result_it = it;
            break;
          }
        } else {
          /* ok, no version information, we assign the version of the column
           * identifier we are looking for */
          std::string column_fully_qualified = it->second;
          column_fully_qualified += ".";
          column_fully_qualified +=
              boost::lexical_cast<std::string>(attr->getVersion());
          if (column_fully_qualified ==
              createFullyQualifiedColumnIdentifier(attr)) {
            result_it = it;
            break;
          }
        }
      } else {
        /* attribute is computed by query */
        if (it->second == column_name) {
          result_it = it;
          break;
        }
      }
    }
  }

  if (result_it == schema_.end()) {
    return schema_.size();
  } else {
    return index;
  }
}

const std::vector<ColumnProperties> Table::getPropertiesOfColumns() const {
  // get all properties provided by BaseTable
  std::vector<ColumnProperties> result = BaseTable::getPropertiesOfColumns();
  // add additional information
  AccessStatisticsMap::const_iterator cit;
  for (size_t i = 0; i < result.size(); ++i) {
    boost::lock_guard<boost::mutex> lock(access_statistics_map_mutex_);
    cit = access_statistics_map_.find(result[i].name);
    if (cit != access_statistics_map_.end()) {
      result[i].number_of_accesses = cit->second.number_of_accesses;
      result[i].last_access_timestamp = cit->second.last_access_timestamp;
    }
  }
  return result;
}

bool Table::copyColumnsInMainMemory() {
  for (size_t i = 0; i < columns_.size(); ++i) {
    // don't do anything for columns on disk
    if (columns_[i]->isLoadedInMainMemory()) {
      ColumnPtr tmp = copy_if_required(columns_[i], hype::PD_Memory_0);
      if (tmp) {
        columns_[i] = tmp;
      } else {
        COGADB_FATAL_ERROR("Failed to transfer column back "
                               << columns_[i]->getName() << " ("
                               << util::getName(columns_[i]->getColumnType())
                               << ") to main memory!",
                           "");
      }
    }
  }
  return true;
}

ColumnAccessStatisticsVector Table::getColumnAccessStatistics() const {
  ColumnAccessStatisticsVector statistics;
  AccessStatisticsMap::const_iterator cit;
  boost::lock_guard<boost::mutex> lock(access_statistics_map_mutex_);
  for (cit = access_statistics_map_.begin();
       cit != access_statistics_map_.end(); ++cit) {
    statistics.push_back(
        ColumnAccessStatistics(this->getName(), cit->first, cit->second));
  }

  return statistics;
}

}  // end namespace CogaDB
