

#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <core/table.hpp>
#include <lookup_table/join_index.hpp>
#include <sstream>

#include "core/runtime_configuration.hpp"
//#include "core/gpu_column_cache.hpp"
#include <core/data_dictionary.hpp>
#include <core/processor_data_cache.hpp>

#include <hardware_optimizations/primitives.hpp>

namespace CoGaDB {
boost::mutex global_join_index_mutex;

JoinIndexes& JoinIndexes::instance() {
  static JoinIndexes join_indx;
  return join_indx;
}

std::string JoinIndexes::joinExpressionToString(TablePtr pk_tab,
                                                std::string pk_tab_col_name,
                                                TablePtr fk_tab,
                                                std::string fk_tab_col_name) {
  std::stringstream ss;
  ss << "JOIN_INDEX(" << pk_tab->getName() << "." << pk_tab_col_name << ","
     << fk_tab->getName() << "." << fk_tab_col_name << ")";
  return ss.str();
}

JoinIndexPtr JoinIndexes::getJoinIndex(TablePtr pk_tab,
                                       std::string pk_tab_col_name,
                                       TablePtr fk_tab,
                                       std::string fk_tab_col_name) {
  boost::lock_guard<boost::mutex> lock(global_join_index_mutex);
  //            assert(pk_tab->hasPrimaryKeyConstraint(pk_tab_col_name)==true);
  //            assert(fk_tab->hasForeignKeyConstraint(fk_tab_col_name)==true);
  std::string index_name =
      joinExpressionToString(pk_tab, pk_tab_col_name, fk_tab, fk_tab_col_name);
  Indexes::iterator it = indeces_.find(index_name);
  if (it != indeces_.end()) {
    return it->second;
  } else {
    // try to load index from disk
    JoinIndexPtr join_index = loadJoinIndex(index_name);
    if (!join_index) {
      // ok, no join index found, compute it and store it on disk
      join_index =
          createJoinIndex(pk_tab, pk_tab_col_name, fk_tab, fk_tab_col_name);
      storeJoinIndex(join_index, index_name);
    }
    this->indeces_.insert(std::make_pair(index_name, join_index));
    return join_index;
  }
}

const PositionListPtr JoinIndexes::getReverseJoinIndex(
    TablePtr pk_tab, std::string pk_tab_col_name, TablePtr fk_tab,
    std::string fk_tab_col_name) {
  boost::lock_guard<boost::mutex> lock(global_join_index_mutex);
  std::string index_name =
      joinExpressionToString(pk_tab, pk_tab_col_name, fk_tab, fk_tab_col_name);
  ReverseJoinIndexes::iterator it = reverse_indeces_.find(index_name);
  if (it != reverse_indeces_.end()) {
    return it->second;
  } else {
    // try to load index from disk
    PositionListPtr reverse_join_index = loadReverseJoinIndex(index_name);
    if (!reverse_join_index) {
      // ok, no join index found, compute it and store it on disk
      reverse_join_index = createReverseJoinIndex(pk_tab, pk_tab_col_name,
                                                  fk_tab, fk_tab_col_name);
      storeReverseJoinIndex(reverse_join_index, index_name);
    }
    this->reverse_indeces_.insert(
        std::make_pair(index_name, reverse_join_index));
    return reverse_join_index;
  }
}

bool JoinIndexes::isReverseJoinIndex(const PositionListPtr tids) const {
  boost::lock_guard<boost::mutex> lock(global_join_index_mutex);
  if (!tids) return false;
  ReverseJoinIndexes::const_iterator it =
      reverse_indeces_.find(tids->getName());
  if (it != reverse_indeces_.end()) {
    return true;
  } else {
    return false;
  }
}

TablePtr JoinIndexes::getSystemTable() {
  boost::lock_guard<boost::mutex> lock(global_join_index_mutex);
  TableSchema result_schema;
  result_schema.push_back(Attribut(VARCHAR, "JOIN INDEX NAME"));
  result_schema.push_back(Attribut(VARCHAR, "PRIMARY KEY TABLE NAME"));
  result_schema.push_back(Attribut(VARCHAR, "FOREIGN KEY TABLE NAME"));
  result_schema.push_back(Attribut(VARCHAR, "INDEX TYPE"));

  TablePtr result_tab(new Table("SYS_JOIN_INDEXES", result_schema));
  Indexes::iterator it;
  ReverseJoinIndexes::iterator rev_it;
  for (it = indeces_.begin(); it != indeces_.end(); ++it) {
    Tuple t;
    t.push_back(it->first);
    t.push_back(it->second->first->getTable()->getName());
    t.push_back(it->second->second->getTable()->getName());
    t.push_back(std::string("JOIN_INDEX"));
    result_tab->insert(t);
    rev_it = reverse_indeces_.find(it->first);
    if (rev_it != reverse_indeces_.end()) {
      Tuple t;
      t.push_back(it->first);
      t.push_back(it->second->first->getTable()->getName());
      t.push_back(it->second->second->getTable()->getName());
      t.push_back(std::string("REVERSE_JOIN_INDEX"));
      result_tab->insert(t);
    }
  }

  return result_tab;
}

const JoinIndexPtr JoinIndexes::createJoinIndex(TablePtr pk_tab,
                                                std::string pk_tab_col_name,
                                                TablePtr fk_tab,
                                                std::string fk_tab_col_name) {
  hype::ProcessingDeviceID id = hype::PD0;
  ProcessorSpecification proc_spec(id);
  JoinParam param(proc_spec, HASH_JOIN);

  TablePtr tab =
      BaseTable::join(pk_tab, pk_tab_col_name, fk_tab, fk_tab_col_name, param);
  //            std::list<std::string> column_names;
  //            column_names.push_back(pk_tab_col_name);
  //            column_names.push_back(fk_tab_col_name);
  //            tab=BaseTable::sort(tab,column_names, ASCENDING, LOOKUP, CPU);

  LookupTablePtr lookup_table = boost::dynamic_pointer_cast<LookupTable>(tab);
  std::vector<LookupColumnPtr> lookup_columns =
      lookup_table->getLookupColumns();
  assert(lookup_columns.size() == 2);
  JoinIndexPtr join_index(
      new JoinIndex(std::make_pair(lookup_columns[0], lookup_columns[1])));

  //
  //            std::map<TID,TID> sorted_pairs;
  //            PositionListPtr pk_column_tids =
  //            join_index->first->getPositionList();
  //            PositionListPtr fk_column_tids =
  //            join_index->second->getPositionList();
  //            for(unsigned int i=0;i<lookup_table->getNumberofRows();++i){
  //                sorted_pairs.push_back(TID_Pair(pk_column_tids[i],
  //                fk_column_tids[i]));
  //            }

  // TODO: FIXME: sort after column 0, then after column 1

  typedef std::pair<TID, TID> TID_Pair;
  typedef std::vector<TID_Pair> TID_Pair_Vector;
  TID_Pair_Vector join_pairs;
  PositionListPtr pk_column_tids = join_index->first->getPositionList();
  PositionListPtr fk_column_tids = join_index->second->getPositionList();
  for (size_t i = 0; i < lookup_table->getNumberofRows(); ++i) {
    join_pairs.push_back(TID_Pair((*pk_column_tids)[i], (*fk_column_tids)[i]));
  }
  std::sort(join_pairs.begin(), join_pairs.end());

  PositionListPtr pk_column_tids_sorted(
      createPositionList(lookup_table->getNumberofRows()));
  PositionListPtr fk_column_tids_sorted(
      createPositionList(lookup_table->getNumberofRows()));
  for (unsigned int i = 0; i < join_pairs.size(); ++i) {
    (*pk_column_tids_sorted)[i] = join_pairs[i].first;
    (*fk_column_tids_sorted)[i] = join_pairs[i].second;
  }

  LookupColumnPtr pk_lc_sorted(
      new LookupColumn(join_index->first->getTable(), pk_column_tids_sorted));
  LookupColumnPtr fk_lc_sorted(
      new LookupColumn(join_index->second->getTable(), fk_column_tids_sorted));
  join_index->first = pk_lc_sorted;
  join_index->second = fk_lc_sorted;
  return join_index;
}

const PositionListPtr JoinIndexes::createReverseJoinIndex(
    TablePtr pk_tab, std::string pk_tab_col_name, TablePtr fk_tab,
    std::string fk_tab_col_name) {
  hype::ProcessingDeviceID id = hype::PD0;
  ProcessorSpecification proc_spec(id);
  JoinParam param(proc_spec, HASH_JOIN);

  TablePtr tab =
      BaseTable::join(pk_tab, pk_tab_col_name, fk_tab, fk_tab_col_name, param);

  LookupTablePtr lookup_table = boost::dynamic_pointer_cast<LookupTable>(tab);
  std::vector<LookupColumnPtr> lookup_columns =
      lookup_table->getLookupColumns();
  assert(lookup_columns.size() == 2);
  JoinIndexPtr join_index(
      new JoinIndex(std::make_pair(lookup_columns[0], lookup_columns[1])));

  // TODO: FIXME: sort after column 0, then after column 1

  typedef std::pair<TID, TID> TID_Pair;
  typedef std::vector<TID_Pair> TID_Pair_Vector;
  TID_Pair_Vector join_pairs;
  PositionListPtr pk_column_tids = join_index->first->getPositionList();
  PositionListPtr fk_column_tids = join_index->second->getPositionList();
  for (size_t i = 0; i < lookup_table->getNumberofRows(); ++i) {
    join_pairs.push_back(TID_Pair((*fk_column_tids)[i], (*pk_column_tids)[i]));
  }
  std::sort(join_pairs.begin(), join_pairs.end());

  PositionListPtr reverse_join_index(
      createPositionList(lookup_table->getNumberofRows()));
  for (unsigned int i = 0; i < join_pairs.size(); ++i) {
    (*reverse_join_index)[i] = join_pairs[i].second;
    //                (*fk_column_tids_sorted)[i]=join_pairs[i].second;
  }

  return reverse_join_index;
}

std::string toString(JoinIndexPtr join_index) {
  // std::string res;
  if (!join_index) return "";
  std::stringstream ss;
  ss << "JOIN_INDEX(" << join_index->first->getTable()->getName() << ","
     << join_index->second->getTable()->getName() << ")";
  //<< std::endl;
  return ss.str();
}

size_t getSizeInBytes(const JoinIndexPtr join_index) {
  if (!join_index) return 0;
  size_t size_of_joinindex = (join_index->first->getPositionList()->size() +
                              join_index->second->getPositionList()->size()) *
                             sizeof(PositionList::value_type);
  return size_of_joinindex;
}

bool storeJoinIndex(JoinIndexPtr join_index,
                    const std::string& join_index_name) {
  using namespace boost::filesystem;
  std::string path = RuntimeConfiguration::instance().getPathToDatabase();
  if (!exists(path)) {
    create_directory(path);
  }
  path += "/join_indexes/";
  if (!exists(path)) {
    create_directory(path);
  }

  path += join_index_name;  // toString(join_index);
  std::string path_meta_data = path + "/INDEXED_TABLE_NAMES";

  if (!exists(path)) {
    create_directory(path);
  }

  join_index->first->getPositionList()->setName("HEAD");
  join_index->second->getPositionList()->setName("TAIL");

  join_index->first->getPositionList()->store(path);
  join_index->second->getPositionList()->store(path);

  std::ofstream outfile(path_meta_data.c_str(),
                        std::ios_base::binary | std::ios_base::out);
  boost::archive::binary_oarchive oa(outfile);

  oa << join_index->first->getTable()->getName();
  oa << join_index->second->getTable()->getName();

  // join_index->first

  outfile.flush();
  outfile.close();

  return true;
}

JoinIndexPtr loadJoinIndex(const std::string& join_index_name) {
  using namespace boost::filesystem;
  std::string path = RuntimeConfiguration::instance().getPathToDatabase();
  if (!exists(path)) {
    return JoinIndexPtr();
  }
  path += "/join_indexes/";
  if (!exists(path)) {
    return JoinIndexPtr();
  }
  path += join_index_name;
  if (!exists(path)) {
    return JoinIndexPtr();
  }

  std::string path_meta_data = path + "/INDEXED_TABLE_NAMES";

  if (!exists(path_meta_data)) return JoinIndexPtr();

  std::ifstream infile(path_meta_data.c_str(),
                       std::ios_base::binary | std::ios_base::in);
  boost::archive::binary_iarchive ia(infile);
  std::string head_name;
  std::string tail_name;

  ia >> head_name;
  ia >> tail_name;
  infile.close();

  PositionListPtr head = createPositionList();
  head->setName("HEAD");
  head->load(path, LOAD_ALL_DATA);

  PositionListPtr tail = createPositionList();
  tail->setName("TAIL");
  tail->load(path, LOAD_ALL_DATA);

  LookupColumnPtr lc1(new LookupColumn(getTablebyName(head_name), head));
  LookupColumnPtr lc2(new LookupColumn(getTablebyName(tail_name), tail));

  JoinIndexPtr join_index(new JoinIndex(std::make_pair(lc1, lc2)));
  return join_index;
}

bool storeReverseJoinIndex(PositionListPtr reverse_join_index,
                           const std::string& join_index_name) {
  using namespace boost::filesystem;
  std::string path = RuntimeConfiguration::instance().getPathToDatabase();
  if (!exists(path)) {
    create_directory(path);
  }
  path += "/join_indexes/";
  if (!exists(path)) {
    create_directory(path);
  }

  path += join_index_name;

  if (!exists(path)) {
    create_directory(path);
  }

  reverse_join_index->setName("REVERSE_JOIN_INDEX");
  reverse_join_index->store(path);

  return true;
}

PositionListPtr loadReverseJoinIndex(const std::string& join_index_name) {
  using namespace boost::filesystem;
  std::string path = RuntimeConfiguration::instance().getPathToDatabase();
  if (!exists(path)) {
    create_directory(path);
  }
  path += "/join_indexes/";
  if (!exists(path)) {
    create_directory(path);
  }

  path += join_index_name;

  if (!exists(path)) {
    create_directory(path);
  }

  std::string full_path = path + "/REVERSE_JOIN_INDEX";
  if (!exists(full_path)) return PositionListPtr();

  PositionListPtr reverse_join_index = createPositionList();
  reverse_join_index->setName("REVERSE_JOIN_INDEX");
  reverse_join_index->load(path, LOAD_ALL_DATA);

  // give proper name, sow we can use it later to determine whether
  // a PositionList is a ReverseJoinIndex or not
  reverse_join_index->setName(join_index_name);

  return reverse_join_index;
}

bool JoinIndexes::loadJoinIndexesFromDisk() {
  using namespace boost::filesystem;
  std::string path = RuntimeConfiguration::instance().getPathToDatabase();
  if (!exists(path)) {
    return false;
  }
  path += "/join_indexes/";
  if (!exists(path)) {
    std::cout << "No join indexes found!" << std::endl;
    return true;
  }

  boost::filesystem::directory_iterator end_iter;
  if (boost::filesystem::exists(path) &&
      boost::filesystem::is_directory(path)) {
    for (boost::filesystem::directory_iterator dir_iter(path);
         dir_iter != end_iter; ++dir_iter) {
      //            if (boost::filesystem::is_regular_file(dir_iter->status()) )
      if (boost::filesystem::is_directory(dir_iter->status())) {
        boost::filesystem::path index_path = *dir_iter;
        JoinIndexPtr join_index =
            loadJoinIndex(index_path.filename().generic_string());
        if (!join_index) return false;
        this->indeces_.insert(
            std::make_pair(index_path.filename().generic_string(), join_index));
        PositionListPtr reverse_join_index =
            loadReverseJoinIndex(index_path.filename().generic_string());
        if (!reverse_join_index) return false;
        this->reverse_indeces_.insert(std::make_pair(
            index_path.filename().generic_string(), reverse_join_index));
      }
    }
    return true;
  } else {
    return false;
  }
  return true;
}

// fetch TIDs of FK Column from JoinIndex
// matching tids comes from a filter operation on a primary key table, and we
// now seek the corresponding foreign key tids
// Note: assumes the Join Index is sorted by the first PositionList (the Primary
// Key Table TID list)
PositionListPtr fetchMatchingTIDsFromJoinIndex(JoinIndexPtr join_index,
                                               PositionListPtr matching_tids) {
  assert(join_index != NULL);
  assert(matching_tids != NULL);

  PositionListPtr fk_column_matching_tids = createPositionList();
  PositionListPtr pk_column_tids = join_index->first->getPositionList();
  PositionListPtr fk_column_tids = join_index->second->getPositionList();
  // intersect(pk_column_matching_tids, matching_tids) -> matching fks
  TID left_pos = 0;
  TID right_pos = 0;
  size_t number_of_join_pairs = join_index->first->getPositionList()->size();
  // assume that matching tids contains no duplicates and is sorted
  while (left_pos < matching_tids->size() && right_pos < number_of_join_pairs) {
    // std::cout << "Left: " << (*matching_tids)[left_pos] << "   Right: " <<
    // (*pk_column_tids)[right_pos] << std::endl;
    TID left_value = (*matching_tids)[left_pos];
    TID right_value = (*pk_column_tids)[right_pos];
    if ((*matching_tids)[left_pos] == (*pk_column_tids)[right_pos]) {
      fk_column_matching_tids->push_back((*fk_column_tids)[right_pos]);
      right_pos++;
    } else if ((*matching_tids)[left_pos] < (*pk_column_tids)[right_pos]) {
      left_pos++;
    } else if ((*matching_tids)[left_pos] > (*pk_column_tids)[right_pos]) {
      right_pos++;
    }
  }
  return fk_column_matching_tids;
}

bool JoinIndexes::placeJoinIndexesOnGPU(
    const hype::ProcessingDeviceMemoryID& mem_id) {
  Indexes::iterator it;
  size_t max_num_of_rows = 0;
  std::map<size_t, JoinIndexPtr> join_indexes;
  for (it = indeces_.begin(); it != indeces_.end(); ++it) {
    size_t num_rows = (*it).second->first->getTable()->getNumberofRows();
    join_indexes.insert(std::make_pair(num_rows, (*it).second));
  }
  //            std::map<size_t,JoinIndexPtr>::reverse_iterator rit;
  //            for(rit=join_indexes.rbegin();rit!=join_indexes.rend();++rit){
  //                if(GPU_Column_Cache::instance().getAvailableGPUBufferSize()>getSizeInBytes((*rit).second)){
  //                    gpu::GPU_JoinIndexPtr gpu_join_index =
  //                    GPU_Column_Cache::instance().getGPUJoinIndex((*rit).second);
  //                }else{
  //                    break;
  //                }
  //                //GPU_Column_Cache::instance().getGPUJoinIndex();
  //            }

  // this order is the resutl of a manual analysis of the SSBM Queries:
  // each query accesses several dimension tables, and based on the number of
  // accesses per dimension table in all queries,
  // we determiend the order, where the Join Indexes are load in the GPU buffer
  //              Q1.1	Q1.2	Q1.3	Q2.1	Q2.2	Q2.3
  //              Q3.1	Q3.2	Q3.3	Q3.4	Q4.1	Q4.2
  //              Q4.3	ALL without FILTER	ALL
  //    DATES	x	x	x	x (no filter)	x (no
  //    filter)	x (no filter)	x	x	x	x
  //    x (no filter)	x	x	9	13
  //    SUPPLIER	-	-	-	x	x	x	x
  //    x	x	x	x	x	x	10	10
  //    CUSTOMER	-	-	-	-	-	-	x
  //    x	x	x	x	x	-	6	6
  //    PART	-	-	-	x	x	x	-
  //    -	-	-	x	x	x	6	6

  for (it = indeces_.begin(); it != indeces_.end(); ++it) {
    if ((*it).second->first->getTable()->getName() == "DATES") {
      if (DataCacheManager::instance()
              .getDataCache(mem_id)
              .getAvailableBufferSize() > getSizeInBytes((*it).second)) {
        JoinIndexPtr join_index =
            DataCacheManager::instance().getDataCache(mem_id).getJoinIndex(
                (*it).second);
      } else {
        break;
      }
    }
  }
  for (it = indeces_.begin(); it != indeces_.end(); ++it) {
    if ((*it).second->first->getTable()->getName() == "SUPPLIER") {
      if (DataCacheManager::instance()
              .getDataCache(mem_id)
              .getAvailableBufferSize() > getSizeInBytes((*it).second)) {
        JoinIndexPtr join_index =
            DataCacheManager::instance().getDataCache(mem_id).getJoinIndex(
                (*it).second);
      } else {
        break;
      }
    }
  }
  for (it = indeces_.begin(); it != indeces_.end(); ++it) {
    if ((*it).second->first->getTable()->getName() == "PART") {
      if (DataCacheManager::instance()
              .getDataCache(mem_id)
              .getAvailableBufferSize() > getSizeInBytes((*it).second)) {
        JoinIndexPtr join_index =
            DataCacheManager::instance().getDataCache(mem_id).getJoinIndex(
                (*it).second);
      } else {
        break;
      }
    }
  }
  for (it = indeces_.begin(); it != indeces_.end(); ++it) {
    if ((*it).second->first->getTable()->getName() == "CUSTOMER") {
      if (DataCacheManager::instance()
              .getDataCache(mem_id)
              .getAvailableBufferSize() > getSizeInBytes((*it).second)) {
        JoinIndexPtr join_index =
            DataCacheManager::instance().getDataCache(mem_id).getJoinIndex(
                (*it).second);
      } else {
        break;
      }
    }
  }

  // pin join indexes, so they are not evicted by the buffer manager
  // GPU_Column_Cache::instance().pinJoinIndexesOnGPU(true);
  // GPU_Column_Cache::instance().pinColumnsOnGPU(true);

  return true;
}

TablePtr getSystemTableJoinIndexes() {
  return JoinIndexes::instance().getSystemTable();
}

bool placeColumnOnCoprocessor(ClientPtr client,
                              const hype::ProcessingDeviceMemoryID& mem_id,
                              const std::string& table_name,
                              const std::string& column_name) {
  std::ostream& out = client->getOutputStream();
  TablePtr table = getTablebyName(table_name);
  if (!table) return false;
  ColumnPtr col = table->getColumnbyName(column_name);
  if (!col) return false;

  if (DataCacheManager::instance()
          .getDataCache(mem_id)
          .getAvailableBufferSize() > col->getSizeinBytes()) {
    ColumnPtr device_column =
        DataCacheManager::instance().getDataCache(mem_id).getColumn(col);
    if (!device_column) {
      return false;
    } else {
      out << "Placed column " << col->getName()
          << " on coprocessor with mem_id " << (int)mem_id << std::endl;
      return true;
    }
  }

  return false;
}

bool placeSelectedColumnsOnGPU(ClientPtr client) {
  std::ostream& out = client->getOutputStream();
  // GPU Cache after several executions of the star schema benchmark workload
  // we add columns to the buffer starting form the most frequently used column,
  // and
  // stop after we added all columns or the buffer is exhausted
  // Cached Columns:
  // D_YEAR of Size 0.00975037MB (2556 elements), currently in use: 0
  //	Referenced: 86	Last Used Timestamp: 4102238747223108
  // S_REGION of Size 0.00770569MB (2000 elements), currently in use: 0
  //	Referenced: 46	Last Used Timestamp: 4102237989553950
  // P_MFGR of Size 0.763016MB (200000 elements), currently in use: 0
  //	Referenced: 29	Last Used Timestamp: 4102237989695602
  // C_CITY of Size 0.118256MB (30000 elements), currently in use: 0
  //	Referenced: 28	Last Used Timestamp: 4102237859113055
  // S_CITY of Size 0.0114441MB (2000 elements), currently in use: 0
  //	Referenced: 28	Last Used Timestamp: 4102237859055776
  // C_REGION of Size 0.114517MB (30000 elements), currently in use: 0
  //	Referenced: 23	Last Used Timestamp: 4102237989678026
  // P_CATEGORY of Size 0.763321MB (200000 elements), currently in use: 0
  //	Referenced: 16	Last Used Timestamp: 4102238746963163
  // S_NATION of Size 0.00801086MB (2000 elements), currently in use: 0
  //	Referenced: 15	Last Used Timestamp: 4102238747013965
  // P_BRAND of Size 0.778198MB (200000 elements), currently in use: 0
  //	Referenced: 7	Last Used Timestamp: 4102237592621586
  // D_YEARMONTH of Size 0.0110321MB (2556 elements), currently in use: 0
  //	Referenced: 7	Last Used Timestamp: 4102237858982994
  // C_NATION of Size 0.114822MB (30000 elements), currently in use: 0
  //	Referenced: 7	Last Used Timestamp: 4102237714772697
  // D_YEARMONTHNUM of Size 0.00975037MB (2556 elements), currently in use: 0
  //	Referenced: 2	Last Used Timestamp: 4102231161959441

  // Cached Join Indexes:
  // JOIN_INDEX(SUPPLIER,LINEORDER): 45MB	Referenced: 75	Last Used
  // Timestamp: 4102238750838514
  // JOIN_INDEX(DATES,LINEORDER): 45MB	Referenced: 59	Last Used Timestamp:
  // 4102238763715114
  // JOIN_INDEX(CUSTOMER,LINEORDER): 45MB	Referenced: 45	Last Used
  // Timestamp: 4102237991520227
  // JOIN_INDEX(PART,LINEORDER): 45MB	Referenced: 43	Last Used Timestamp:
  // 4102238751422509

  const hype::ProcessingDeviceMemoryID mem_id = hype::PD_Memory_1;

  // optimized for plans produced by star_join_optimizer
  //        const char* column_names[] = {"D_YEAR",
  //                                      "S_REGION",
  //                                      "P_MFGR",
  //                                      "C_CITY",
  //                                      "S_CITY",
  //                                      "C_REGION",
  //                                      "P_CATEGORY",
  //                                      "S_NATION",
  //                                      "P_BRAND",
  //                                      "D_YEARMONTH",
  //                                      "C_NATION",
  //                                      "D_YEARMONTHNUM",
  //                                      "LO_DISCOUNT",
  //                                      "LO_QUANTITY"};

  // optimized for plans produced by default_optimizer
  const char* column_names[] = {
      "S_SUPPKEY",   "S_CITY",         "S_NATION",   "S_REGION",
      "LO_PARTKEY",  "P_BRAND",        "P_CATEGORY", "P_MFGR",
      "P_PARTKEY",   "LO_CUSTKEY",     "LO_SUPPKEY", "LO_ORDERDATE",
      "LO_QUANTITY", "LO_DISCOUNT",    "LO_REVENUE", "D_WEEKNUMINYEAR",
      "D_YEARMONTH", "D_YEARMONTHNUM", "D_YEAR",     "D_DATEKEY",
      "C_CUSTKEY",   "C_NATION",       "C_CITY",     "C_REGION"};

  size_t number_of_columns = sizeof(column_names) / sizeof(char*);
  // GPU_Column_Cache::instance().pinColumnsOnGPU(true);

  for (unsigned int i = 0; i < number_of_columns; ++i) {
    std::list<std::pair<ColumnPtr, TablePtr> > columns =
        DataDictionary::instance().getColumnsforColumnName(column_names[i]);
    assert(columns.size() == 1);
    ColumnPtr col = columns.front().first;
    assert(col != NULL);
    if (DataCacheManager::instance()
            .getDataCache(mem_id)
            .getAvailableBufferSize() > col->getSizeinBytes()) {
      ColumnPtr device_column =
          DataCacheManager::instance().getDataCache(mem_id).getColumn(col);
      if (!device_column)
        return false;
      else
        std::cout << "Placed column " << col->getName() << " on GPU"
                  << std::endl;
    }
  }

  return true;
}

// Note: assumes the Join Index is sorted by the first PositionList (the Primary
// Key Table TID list)
PositionListPtr fetchMatchingTIDsFromJoinIndexInParallel(
    JoinIndexPtr join_index, PositionListPtr matching_tids) {
  assert(join_index != NULL);
  assert(matching_tids != NULL);

  PositionListPtr fk_column_matching_tids = createPositionList();
  PositionListPtr pk_column_tids = join_index->first->getPositionList();
  PositionListPtr fk_column_tids = join_index->second->getPositionList();

  // intersect(pk_column_matching_tids, matching_tids) -> matching fks
  TID left_pos = 0;
  TID right_pos = 0;
  size_t number_of_join_pairs = join_index->first->getPositionList()->size();
  // assume that matching tids contains no duplicates and is sorted
  while (left_pos < matching_tids->size() && right_pos < number_of_join_pairs) {
    // std::cout << "Left: " << (*matching_tids)[left_pos] << "   Right: " <<
    // (*pk_column_tids)[right_pos] << std::endl;
    TID left_value = (*matching_tids)[left_pos];
    TID right_value = (*pk_column_tids)[right_pos];
    if ((*matching_tids)[left_pos] == (*pk_column_tids)[right_pos]) {
      fk_column_matching_tids->push_back((*fk_column_tids)[right_pos]);
      right_pos++;
    } else if ((*matching_tids)[left_pos] < (*pk_column_tids)[right_pos]) {
      left_pos++;
    } else if ((*matching_tids)[left_pos] > (*pk_column_tids)[right_pos]) {
      right_pos++;
    }
  }
  return fk_column_matching_tids;
}

PositionListPairPtr fetchJoinResultFromJoinIndex(
    JoinIndexPtr join_index, PositionListPtr matching_tids_pk_column) {
  // join result tids
  PositionListPtr pk_column_matching_tids = createPositionList();
  PositionListPtr fk_column_matching_tids = createPositionList();
  PositionListPairPtr join_tids(new PositionListPair(
      std::make_pair(pk_column_matching_tids, fk_column_matching_tids)));

  PositionListPtr pk_column_tids = join_index->first->getPositionList();
  PositionListPtr fk_column_tids = join_index->second->getPositionList();
  // intersect(pk_column_matching_tids, matching_tids) -> matching fks
  TID left_pos = 0;
  TID right_pos = 0;
  size_t number_of_join_pairs = join_index->first->getPositionList()->size();
  // assume that matching tids contains no duplicates and is sorted
  while (left_pos < matching_tids_pk_column->size() &&
         right_pos < number_of_join_pairs) {
    if ((*matching_tids_pk_column)[left_pos] == (*pk_column_tids)[right_pos]) {
      // found matching join pair
      join_tids->first->push_back((*pk_column_tids)[right_pos]);
      join_tids->second->push_back((*fk_column_tids)[right_pos]);
      right_pos++;
    } else if ((*matching_tids_pk_column)[left_pos] <
               (*pk_column_tids)[right_pos]) {
      left_pos++;
    } else if ((*matching_tids_pk_column)[left_pos] >
               (*pk_column_tids)[right_pos]) {
      right_pos++;
    }
  }
  return join_tids;
}

//
//
////    void setBitmapMatchingTIDsFromJoinIndex(JoinIndexPtr join_index,
/// PositionListPtr matching_tids, char* fk_column_matching_rows_bitmap, size_t
/// bitmap_num_bits){
//    BitmapPtr createBitmapOfMatchingTIDsFromJoinIndex(JoinIndexPtr join_index,
//    PositionListPtr matching_tids){
//        assert(join_index!=NULL);
//        assert(matching_tids!=NULL);
//        //create bitmap, init every bit with 0
//        BitmapPtr result(new
//        Bitmap(join_index->second->getPositionList()->size(), false, true));
//        std::cout << "Bitmap Fetch Join: #Rows in Bitmap: " <<
//        join_index->second->getPositionList()->size() << std::endl;
//        char* fk_column_matching_rows_bitmap = result->data();
//        char* byte_wise_bitmap = (char*)
//        calloc(join_index->second->getPositionList()->size(),sizeof(char));
//        PositionListPtr pk_column_tids = join_index->first->getPositionList();
//        PositionListPtr fk_column_tids =
//        join_index->second->getPositionList();
//        TID* ji_fks = fk_column_tids->data();
//        //intersect(pk_column_matching_tids, matching_tids) -> matching fks
//        unsigned int left_pos=0;
//        unsigned int right_pos=0;
//        unsigned int
//        number_of_join_pairs=join_index->first->getPositionList()->size();
//        //assume that matching tids contains no duplicates and is sorted
//        while(left_pos < matching_tids->size() && right_pos <
//        number_of_join_pairs){
////            TID left_value = (*matching_tids)[left_pos];
////            TID right_value = (*pk_column_tids)[right_pos];
//            if((*matching_tids)[left_pos] == (*pk_column_tids)[right_pos]){
//                //fk_column_matching_tids->push_back((*fk_column_tids)[right_pos]);
//                unsigned int current_bit= right_pos & 7; //i%8;;
//                char bitmask = 1 << current_bit;
//                TID bit_number = ji_fks[right_pos];
//                std::cout << "Set Bit: " << right_pos << "" << std::endl;
//                //fk_column_matching_rows_bitmap[bit_number/8]|=bitmask;
//                byte_wise_bitmap[right_pos]=1;
//                right_pos++;
//            }else if((*matching_tids)[left_pos] <
//            (*pk_column_tids)[right_pos]){
//                left_pos++;
//            }else if((*matching_tids)[left_pos] >
//            (*pk_column_tids)[right_pos]){
//                right_pos++;
//            }
//        }
//        CoGaDB::CDK::selection::PositionListToBitmap_pack_flag_array_thread(byte_wise_bitmap,
//        join_index->second->getPositionList()->size(),
//        fk_column_matching_rows_bitmap);
//        return result;
//    }

BitmapPtr createBitmapOfMatchingTIDsFromJoinIndex(
    JoinIndexPtr join_index, PositionListPtr matching_tids) {
  assert(join_index != NULL);
  assert(matching_tids != NULL);
  size_t count = 0;
  // PositionListPtr fk_column_matching_tids=createPositionList();
  BitmapPtr result(
      new Bitmap(join_index->second->getPositionList()->size(), false, true));
  char* bitmap = result->data();
  PositionListPtr pk_column_tids = join_index->first->getPositionList();
  PositionListPtr fk_column_tids = join_index->second->getPositionList();
  // intersect(pk_column_matching_tids, matching_tids) -> matching fks
  TID left_pos = 0;
  TID right_pos = 0;
  size_t number_of_join_pairs = join_index->first->getPositionList()->size();
  // assume that matching tids contains no duplicates and is sorted
  while (left_pos < matching_tids->size() && right_pos < number_of_join_pairs) {
    // std::cout << "Left: " << (*matching_tids)[left_pos] << "   Right: " <<
    // (*pk_column_tids)[right_pos] << std::endl;
    TID left_value = (*matching_tids)[left_pos];
    TID right_value = (*pk_column_tids)[right_pos];
    if ((*matching_tids)[left_pos] == (*pk_column_tids)[right_pos]) {
      // fk_column_matching_tids->push_back((*fk_column_tids)[right_pos]);
      bitmap[(*fk_column_tids)[right_pos] / 8] |=
          (1 << ((*fk_column_tids)[right_pos] & 7));
      // std::cout << "Set Bit: " << (*fk_column_tids)[right_pos] << "" <<
      // std::endl;
      count++;
      right_pos++;
    } else if ((*matching_tids)[left_pos] < (*pk_column_tids)[right_pos]) {
      left_pos++;
    } else if ((*matching_tids)[left_pos] > (*pk_column_tids)[right_pos]) {
      right_pos++;
    }
  }
  // std::cout << "Hits: " << count << std::endl;
  return result;
}

//    TID binary_index_search(TID* in, TID elem, long startIdx, long endIdx){
//            long from = startIdx;
//            long to = endIdx;
//            long mid = (long) (from + to) / 2;
//
//            while (from <= to && !(in[mid - 1] <= elem && in[mid] > elem))
//            {
//                    if (in[mid] > elem)
//                            to = mid - 1;
//                    else
//                            from = mid + 1;
//                    mid = (long) (from + to) / 2;
//            }
//            return (mid == endIdx && in[endIdx] <= elem) ? mid + 1 : mid;
//    }

void setFetchJoinFlagArray_thread(TID* matching_tids,
                                  size_t number_of_matching_tids,
                                  TID* pk_column_tids, TID* fk_column_tids,
                                  size_t right_pos_begin, size_t right_pos_end,
                                  char* flag_array) {
  // intersect(pk_column_matching_tids, matching_tids) -> matching fks
  TID left_pos = 0;
  // search for correct begin position to start the merging
  // unsigned int left_pos= binary_index_search(matching_tids,
  // pk_column_tids[right_pos_begin],0, number_of_matching_tids);
  TID right_pos = right_pos_begin;

  //        while(left_pos>0 && matching_tids[left_pos] ==
  //        pk_column_tids[right_pos_begin]){
  //            left_pos--;
  //        }
  // unsigned int
  // number_of_join_pairs=join_index->first->getPositionList()->size();
  // assume that matching tids contains no duplicates and is sorted
  while (left_pos < number_of_matching_tids && right_pos < right_pos_end) {
    // std::cout << "Left: " << (*matching_tids)[left_pos] << "   Right: " <<
    // (*pk_column_tids)[right_pos] << std::endl;
    TID left_value = matching_tids[left_pos];
    TID right_value = pk_column_tids[right_pos];
    if (matching_tids[left_pos] == pk_column_tids[right_pos]) {
      // bitmap[fk_column_tids[right_pos]/8]|=(1 <<
      // (fk_column_tids[right_pos]&7));
      flag_array[fk_column_tids[right_pos]] = 1;
      right_pos++;
    } else if (matching_tids[left_pos] < pk_column_tids[right_pos]) {
      left_pos++;
    } else if (matching_tids[left_pos] > pk_column_tids[right_pos]) {
      right_pos++;
    }
  }
}

BitmapPtr createBitmapOfMatchingTIDsFromJoinIndexParallel(
    JoinIndexPtr join_index, PositionListPtr matching_tids) {
  assert(join_index != NULL);
  assert(matching_tids != NULL);
  size_t count = 0;
  // PositionListPtr fk_column_matching_tids=createPositionList();
  BitmapPtr result(
      new Bitmap(join_index->second->getPositionList()->size(), false, true));
  char* bitmap = result->data();
  PositionListPtr pk_column_tids = join_index->first->getPositionList();
  PositionListPtr fk_column_tids = join_index->second->getPositionList();
  // intersect(pk_column_matching_tids, matching_tids) -> matching fks
  //        unsigned int left_pos=0;
  //        unsigned int right_pos=0;
  size_t number_of_join_pairs = join_index->first->getPositionList()->size();

  char* flag_array = (char*)calloc(number_of_join_pairs, sizeof(char));

  // boost::thread* thr = new boost::thread(boost::bind(&Foo::some_function,
  // &f));

  // TODO: parallelize
  //	setFetchJoinFlagArray_thread(matching_tids->data(),
  // matching_tids->size(), pk_column_tids->data(), fk_column_tids->data(),
  //	 			     0, number_of_join_pairs, flag_array);

  size_t number_of_threads = 8;
  size_t chunk_size =
      (number_of_join_pairs + number_of_threads - 1) / number_of_threads;
  //        //std::cout << "rows in fact table: " << number_of_join_pairs << ""
  //        << std::endl;
  //        for(unsigned int
  //        thread_id=0;thread_id<number_of_threads;++thread_id){
  //            TID begin=chunk_size*thread_id; //std::min(chunk_size*thread_id,
  //            number_of_join_pairs);
  //            TID end = std::min(chunk_size*(thread_id+1),
  //            number_of_join_pairs);
  //            //std::cout << "BFJ: begin: " << begin << " end: " << end <<
  //            std::endl;
  //            setFetchJoinFlagArray_thread(matching_tids->data(),
  //            matching_tids->size(), pk_column_tids->data(),
  //            fk_column_tids->data(),
  //                                         begin, end, flag_array);
  //        }
  boost::thread_group threads;
  for (unsigned int thread_id = 0; thread_id < number_of_threads; ++thread_id) {
    TID begin = std::min(chunk_size * thread_id, number_of_join_pairs);
    TID end = std::min(chunk_size * (thread_id + 1), number_of_join_pairs);
    threads.add_thread(
        new boost::thread(setFetchJoinFlagArray_thread, matching_tids->data(),
                          matching_tids->size(), pk_column_tids->data(),
                          fk_column_tids->data(), begin, end, flag_array));
  }
  threads.join_all();

  //        boost::thread* thr = new
  //        boost::thread(setFetchJoinFlagArray_thread,matching_tids->data(),
  //        matching_tids->size(), pk_column_tids->data(),
  //        fk_column_tids->data(),
  //	 			     0, number_of_join_pairs, flag_array);
  //
  //        thr->join();
  //        delete thr;

  CDK::selection::PositionListToBitmap_pack_flag_array_thread(
      flag_array, number_of_join_pairs, bitmap);
  free(flag_array);
  return result;
}

}  // end namespace CoGaDB
