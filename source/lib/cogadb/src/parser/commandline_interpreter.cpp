#ifdef HAVE_CONFIG_H

#include "compression/bitpacked_dictionary_compressed_column.hpp"
#include "compression/rle_compressed_column.hpp"
#include "config.h"
#include "util/profiling.hpp"

#endif

#include <boost/unordered_map.hpp>

#include <fstream>
#include <iostream>

#include <boost/tokenizer.hpp>
#include <core/table.hpp>
#include <lookup_table/join_index.hpp>
#include <lookup_table/lookup_table.hpp>
#include <parser/commandline_interpreter.hpp>

#include <unittests/unittests.hpp>
#include <util/star_schema_benchmark.hpp>
#include <util/tpch_benchmark.hpp>

#ifdef BAM_FOUND

#include <util/genomics_extension/base_centric_simple_key_importer.hpp>
#include <util/genomics_extension/exporter.hpp>
#include <util/genomics_extension/genomics_definitions.hpp>
#include <util/genomics_extension/sequence_centric_simple_key_importer.hpp>
#include <util/genomics_extension/sequence_centric_simple_key_importer_stashing_unmapped_reads.hpp>
#include <util/genomics_extension/storage_experiments_importer.hpp>

#endif

#include <core/runtime_configuration.hpp>

#include <boost/lexical_cast.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/bind.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>

#include <algorithm>
#include <queue>
#include <string>
#include <thread>
#include <utility>

#include <parser/generated/Parser.h>
#include <query_processing/query_processor.hpp>

#include <hype.h>
#include <time.h>
#include <compression/dictionary_compressed_column.hpp>
#include <core/data_dictionary.hpp>
#include <sql/server/sql_driver.hpp>

#include <core/processor_data_cache.hpp>
#include <core/variable_manager.hpp>

#ifdef LIBREADLINE_FOUND

#include <readline/history.h>
#include <readline/readline.h>
#include <stdio.h>

#endif

#include <core/memory_allocator.hpp>
#include <parser/json_parser.hpp>
#include <util/functions.hpp>
#include <util/hardware_detector.hpp>
#include <util/query_processing.hpp>
#include <util/statistics.hpp>

#include <backends/cpu/hashtable.hpp>

#ifdef ENABLE_GPU_ACCELERATION
#include <query_compilation/gpu_handbuilt/queries.h>
#endif

#include <query_compilation/pipeline_selectivity_estimates.hpp>

#include <tbb/parallel_sort.h>

#include <util/compilation_tpch_experiments.hpp>

using namespace std;

namespace CoGaDB {
using namespace query_processing;

boost::mutex global_command_queue_mutex;
std::queue<std::string> commandQueue;

std::string getNextCommandFromGlobalCommandQueue() {
  std::string command;
  {
    boost::lock_guard<boost::mutex> lock(global_command_queue_mutex);
    if (commandQueue.empty()) return "TERMINATE";
    command = commandQueue.front();
    commandQueue.pop();
  }
  return command;
}

bool printStatusOfCaches(ClientPtr client);

bool printStatusOfGPUCache(ClientPtr client) {
  return printStatusOfCaches(client);
}

bool printStatusOfCaches(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  DataCacheManager::instance().print(out);
  return true;
}

bool printHypeStatus(ClientPtr client) {
  hype_printStatus();
  return true;
}

bool loadJoinIndexes(ClientPtr client) {
  return JoinIndexes::instance().loadJoinIndexesFromDisk();
}

bool placeSelectedJoinIndexesOnGPU(ClientPtr client) {
  return JoinIndexes::instance().placeJoinIndexesOnGPU(hype::PD_Memory_1);
}

bool placeSelectedColumn(const std::string &param, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<std::string> strs;
  boost::split(strs, param, boost::is_any_of(" "));

  if (strs.size() < 3) {
    out << "Error: Missing Parameter! " << std::endl;
    return false;
  } else if (strs.size() > 3) {
    out << "Error: To Many Parameters! " << std::endl;
    return false;
  }

  unsigned int i = 0;
  try {
    i = boost::lexical_cast<unsigned int>(strs[0]);
  } catch (const boost::bad_lexical_cast &) {
    out << "Error! Invalid Memory ID: '" << i << "'" << endl;
    out << "Valid values are unsigned integers!" << endl;
    return false;
  }

  hype::ProcessingDeviceMemoryID mem_id =
      static_cast<hype::ProcessingDeviceMemoryID>(i);
  std::string table_name = strs[1];
  std::string column_name = strs[2];

  if (!placeColumnOnCoprocessor(client, mem_id, table_name, column_name)) {
    out << "Failed to load column " << table_name << "." << column_name
        << " in memory with ID " << mem_id << std::endl;
  }
  return true;
}

bool placeMostFrequentlyUsedColumns(const std::string &param,
                                    ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<std::string> strs;
  boost::split(strs, param, boost::is_any_of(" "));

  if (strs.size() < 1) {
    out << "Error: Missing Parameter! " << std::endl;
    return false;
  } else if (strs.size() > 1) {
    out << "Error: To Many Parameters! " << std::endl;
    return false;
  }

  unsigned int i = 0;
  try {
    i = boost::lexical_cast<unsigned int>(strs[0]);
  } catch (const boost::bad_lexical_cast &) {
    out << "Error! Invalid Memory ID: '" << i << "'" << endl;
    out << "Valid values are unsigned integers!" << endl;
    return false;
  }

  assert(i == 1);
  hype::ProcessingDeviceMemoryID mem_id =
      static_cast<hype::ProcessingDeviceMemoryID>(i);
  return DataCacheManager::instance()
      .getDataCache(mem_id)
      .placeMostFrequentlyUsedColumns(out);
}

bool execShellCommand(const std::string &file_name, ClientPtr client) {
  int err = system(file_name.c_str());
  if (err == -1) {
    return false;
  } else {
    return true;
  }
}

bool printHistory(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  HIST_ENTRY **history = history_list();
  if (!history) return false;
  //        int length=history_length();
  out << "History: " << endl;
  for (int i = 0; i < history_length; ++i) {
    out << "\t" << history[i]->line << endl;
  }

  return true;
}

bool printAvailableGPU(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  return printGPUs(out);
}

bool redirectOutputToFile(const std::string &file_name, ClientPtr client) {
  static std::ofstream filestr;
  static std::ofstream err;
  filestr.open(file_name.c_str(), fstream::app);
  std::string err_file_name = file_name + ".error_log";
  err.open(err_file_name.c_str(), fstream::app);
  std::cout.rdbuf(filestr.rdbuf());
  std::cerr.rdbuf(err.rdbuf());

  return true;
}

bool dumpEstimationErrors(const std::string &file_name, ClientPtr client) {
  std::ostream &out = client->getOutputStream();

  std::list<std::pair<std::string, double> > avg_est_errors =
      hype::Scheduler::instance().getAverageEstimationErrors();
  std::list<std::pair<std::string, double> >::const_iterator cit;

  std::string last_alg_name = "";

  std::stringstream ss_header;
  std::stringstream ss_values;
  for (cit = avg_est_errors.begin(); cit != avg_est_errors.end(); ++cit) {
    // cut of the _DeviceIDnumber suffix
    auto pos = cit->first.find_last_of("_");
    std::string alg_name =
        cit->first.substr(0,
                          pos);  // return the original algorithm name, which is
                                 // independent of the device type //cit->first;

    // omit duplicate algorithms (algorithms on the same processor share their
    // algorithm statistics)
    if (alg_name != last_alg_name) {
      if (cit != avg_est_errors.begin()) {
        ss_header << "\t";
        ss_values << "\t";
      }
      ss_header << alg_name;
      ss_values << cit->second;
      last_alg_name = alg_name;
    }
  }

  std::string header = ss_header.str();
  std::string values = ss_values.str();
  std::fstream file(file_name.c_str(), std::ios_base::out | std::ios_base::app);

  file.seekg(0, std::ios::end);     // put the "cursor" at the end of the file
  auto file_length = file.tellg();  // find the position of the cursor

  if (file_length == 0) {  // if file empty, write header
    file << header << std::endl;
  }

  file << values << std::endl;
  file.close();

  out << "Average Estimation Errors of Algorithms:" << std::endl;
  out << header << std::endl;
  out << values << std::endl;
  return true;
}

bool testRLEColumn() {
  {
    RLECompressedColumn<int> col("lol", INT);

    for (int i = 0; i < 10; ++i) {
      for (int j = 0; j < 10; ++j) {
        col.insert(i);
        std::cout << i << std::endl;
      }
    }

    ProcessorSpecification proc_spec(hype::PD0);
    boost::shared_ptr<Column<int> > result =
        col.copyIntoDenseValueColumn(proc_spec);

    result->print();
  }

  {
    ColumnPtr col(new RLECompressedColumn<int>("lol", INT));
    ColumnPtr col2(new Column<int>("lol2", INT));

    for (int i = 0; i < 10000; ++i) {
      for (int j = 0; j < 10; ++j) {
        col->insert(i);
      }
    }
    for (int u = 0; u < 1000000; ++u) {
      col2->insert(rand() % 1000);
    }
    uint64_t begin = getTimestamp();
    ProcessorSpecification proc_spec(hype::PD0);
    JoinParam param(proc_spec, HASH_JOIN);
    PositionListPairPtr res1 = col->join(col2, param);
    PositionListPairPtr res2 = col2->join(col, param);
    assert(res1 != NULL);
    assert(res2 != NULL);
    uint64_t end = getTimestamp();
    std::cout << "Time: " << double(end - begin) / (1000 * 1000 * 1000) << "s"
              << std::endl;
  }
  return true;
}

bool testHashTablePerformance(ClientPtr client) {
  size_t num_build_elements = static_cast<std::size_t>(1000) * 1000 * 20;
  size_t num_probe_elements = static_cast<std::size_t>(1000) * 1000 * 10;

  std::vector<TID> build_keys(num_build_elements);
  std::vector<TID> probe_keys(num_probe_elements);

  std::cout << "Generating data: " << std::endl;

  for (size_t i = 0; i < num_build_elements; ++i) {
    build_keys[i] = rand() % 1000000;
  }

  for (size_t i = 0; i < num_probe_elements; ++i) {
    probe_keys[i] = rand() % 1000000;
  }

  std::cout << "Start Benchmark: " << std::endl;

#define USE_SIMPLE_HASH_TABLE

//        google::dense_hash_map<TID,TID> google_ht;

#ifdef USE_SIMPLE_HASH_TABLE
  typedef TypedHashTable<TID, TID> HashTable;
  HashTable ht(num_build_elements);
#else
  //        using namespace std::tr1;
  //        typedef std::tr1::unordered_map<TID,TID,std::tr1::hash<TID>,
  //        std::tr1::equal_to<TID> > HashTable;
  typedef boost::unordered_multimap<TID, TID> HashTable;
  HashTable ht;
  typedef HashTable::iterator HashTableIterator;
#endif

  Timestamp begin_build = getTimestamp();
  {
    //        COGADB_PCM_START_PROFILING("hash_build",std::cout);

    for (size_t i = 0; i < num_build_elements; ++i) {
#ifdef USE_SIMPLE_HASH_TABLE
      HashTable::tuple_t t = {build_keys[i], i};
      ht.put(t);
#else
      ht.insert(std::make_pair(build_keys[i], i));
#endif
    }
    //        COGADB_PCM_STOP_PROFILING("hash_build", std::cout,
    //        num_build_elements,
    //                sizeof(TID), false, false, true);
  }
  Timestamp end_build = getTimestamp();

  Timestamp begin_probe = getTimestamp();
  {
    //        COGADB_PCM_START_PROFILING("hash_probe",std::cout);
    size_t counter = 0;
    //#pragma omp parallel for

    for (size_t i = 0; i < num_probe_elements; ++i) {
#ifdef USE_SIMPLE_HASH_TABLE
      HashTable::hash_bucket_t *bucket = ht.getBucket(probe_keys[i]);
      if (bucket) do {
          for (size_t bucket_slot = 0; bucket_slot < bucket->count;
               bucket_slot++) {
            if (bucket->tuples[bucket_slot].key == probe_keys[i]) {
              counter++;
            }
          }
          bucket = bucket->next;
        } while (bucket);
#else
      HashTableIterator it = ht.find(probe_keys[i]);
      if (it != ht.end()) {
        counter++;
      }
#endif
    }

    //        COGADB_PCM_STOP_PROFILING("hash_probe", std::cout,
    //        num_probe_elements,
    //                sizeof(TID), false, false, true);
  }
  Timestamp end_probe = getTimestamp();

  Timestamp begin_sort = getTimestamp();
  tbb::parallel_sort(build_keys.begin(), build_keys.end());
  Timestamp end_sort = getTimestamp();

  Timestamp begin_binary_search_probe = getTimestamp();
  size_t counter = 0;
  for (size_t i = 0; i < num_probe_elements; ++i) {
    auto bounds =
        std::equal_range(build_keys.begin(), build_keys.end(), probe_keys[i]);
    for (auto cit = bounds.first; cit != bounds.second; ++cit) {
      counter++;
    }
  }
  Timestamp end_binary_search_probe = getTimestamp();

  double time_hash_build_in_sec =
      double(end_build - begin_build) / (1000 * 1000 * 1000);
  double time_hash_probe_in_sec =
      double(end_probe - begin_probe) / (1000 * 1000 * 1000);
  double time_sort_in_sec =
      double(end_sort - begin_sort) / (1000 * 1000 * 1000);
  double time_binary_search_probe_in_sec =
      double(end_binary_search_probe - begin_binary_search_probe) /
      (1000 * 1000 * 1000);

  std::cout << "Build Time: " << time_hash_build_in_sec << "s\t"
            << (double(sizeof(TID) * num_build_elements) /
                (1024 * 1024 * 1024)) /
                   time_hash_build_in_sec
            << " GB/s\t"
            << "AVG Access Time: "
            << (time_hash_build_in_sec / num_build_elements) *
                   (1000 * 1000 * 1000)
            << "ns" << std::endl;
  std::cout << "Probe Time: " << time_hash_probe_in_sec << "s\t"
            << (double(sizeof(TID) * num_probe_elements) /
                (1024 * 1024 * 1024)) /
                   time_hash_probe_in_sec
            << " GB/s\t"
            << "AVG Access Time: "
            << (time_hash_probe_in_sec / num_probe_elements) *
                   (1000 * 1000 * 1000)
            << "ns" << std::endl;

  std::cout << "Sort Time: " << time_sort_in_sec << "s\t"
            << (double(sizeof(TID) * num_build_elements) /
                (1024 * 1024 * 1024)) /
                   time_sort_in_sec
            << " GB/s\t"
            << "AVG Access Time: "
            << (time_sort_in_sec / num_build_elements) * (1000 * 1000 * 1000)
            << "ns" << std::endl;

  std::cout << "Binary Search Probe Time: " << time_binary_search_probe_in_sec
            << "s\t"
            << (double(sizeof(TID) * num_probe_elements) /
                (1024 * 1024 * 1024)) /
                   time_binary_search_probe_in_sec
            << " GB/s\t"
            << "AVG Access Time: "
            << (time_binary_search_probe_in_sec / num_probe_elements) *
                   (1000 * 1000 * 1000)
            << "ns" << std::endl;

#ifdef USE_SIMPLE_HASH_TABLE
  ht.printStatistics();
#endif

  return true;
}

bool createVeryLargeTable(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  Column<int> *col = new Column<int>("LARGE_COLUMN", INT);

  // insert ten billion rows
  //        size_t number_of_rows=10000000000llu;
  size_t number_of_rows = 4300000000llu;  // 4300000000llu;
  for (size_t i = 0; i < number_of_rows; ++i) {
    col->insert(rand());
    if (i % 10000000 == 0) {
      out << "\r" << i << "rows";
      out.flush();
    }
  }
  // reference value whch we will query to check correctness
  col->insert(int(0));
  out << std::endl;
  std::vector<ColumnPtr> columns;
  columns.push_back(ColumnPtr(col));
  TablePtr table(new Table("LARGE_TABLE", columns));

  return addToGlobalTableList(table);
}

//    static char** cogadb_completion( const char * text, int start,  int end)
//    {
//        char **matches;
//
//        matches = (char **)NULL;
//
//        if (start == 0)
//            matches = rl_completion_matches ((char*)text,
//            &autocompletion_candidate_generator);
//        else
//            rl_bind_key('\t',rl_abort);
//
//        return (matches);
//
//    }

//    char* autocompletion_candidate_generator(const char* text, int state)
//    {
//        static int list_index, len;
//        char *name;
//
//        if (!state) {
//            list_index = 0;
//            len = strlen (text);
//        }
//
//        while (name = cmd[list_index]) {
//            list_index++;
//
//            if (strncmp (name, text, len) == 0)
//                return (dupstr(name));
//        }
//
//        /* If no names matched, then return NULL. */
//        return ((char *)NULL);
//
//    }
//
//    char * createNewCString (char* s) {
//      char *r;
//      //use C function, because Readline will call free() later
//      r = (char*) malloc ((strlen (s) + 1));
//      assert(r!=NULL);
//      strcpy (r, s);
//      return (r);
//    }

// queries for revision experiments
bool QC_TPCH_Q1(ClientPtr client) {
  // json available
  return (qcrev_tpch1(MULTI_STAGE_CODE_GENERATOR) != NULL);
}

bool QC_TPCH_Q5(ClientPtr client) {
  // json available
  return (qcrev_tpch5_join(MULTI_STAGE_CODE_GENERATOR) != NULL);
}

bool QC_TPCH_Q9(ClientPtr client) {
  return (qcrev_tpch9(MULTI_STAGE_CODE_GENERATOR) != NULL);
}

bool QC_TPCH_Q13(ClientPtr client) {
  return (qcrev_tpch13(MULTI_STAGE_CODE_GENERATOR) != NULL);
}

bool QC_TPCH_Q17(ClientPtr client) {
  return (qcrev_tpch17(MULTI_STAGE_CODE_GENERATOR) != NULL);
}

bool QC_TPCH_Q18(ClientPtr client) {
  return (qcrev_tpch18(MULTI_STAGE_CODE_GENERATOR) != NULL);
}

bool QC_TPCH_Q19(ClientPtr client) {
  // json available
  return (qcrev_tpch19(MULTI_STAGE_CODE_GENERATOR) != NULL);
}

bool QC_TPCH_Q21(ClientPtr client) {
  return (qcrev_tpch21(MULTI_STAGE_CODE_GENERATOR) != NULL);
}

bool simple_SSB_Query_Selection() {
  // hype::DeviceConstraint default_device_constraint(hype::ANY_DEVICE);
  // //hype::CPU_ONLY); //if you have only a CPU
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(
      new logical_operator::Logical_Scan("LINEORDER"));

  // Attribut(INT,"YEARMONTHNUM")); //  numeric (YYYYMM) -- e.g. 199803

  boost::shared_ptr<logical_operator::Logical_Selection> selection_lineorder(
      new logical_operator::Logical_Selection(
          "LO_DISCOUNT", boost::any(float(0)), GREATER, LOOKUP,
          default_device_constraint));

  selection_lineorder->setLeft(scan_lineorder);

  LogicalQueryPlan log_plan(selection_lineorder, std::cout);
  log_plan.print();

  // cout << "Executing Query: " << endl;
  CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
      log_plan.convertToPhysicalQueryPlan();
  plan->print();

  plan->run();

  return true;
}

bool simple_SSB_Query_Join() {
  // hype::DeviceConstraint default_device_constraint(hype::ANY_DEVICE);
  // //hype::CPU_ONLY); //if you have only a CPU
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(
      new logical_operator::Logical_Scan("LINEORDER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_date(
      new logical_operator::Logical_Scan("DATE"));

  // Attribut(INT,"YEARMONTHNUM")); //  numeric (YYYYMM) -- e.g. 199803

  boost::shared_ptr<logical_operator::Logical_Selection> selection_date(
      new logical_operator::Logical_Selection("D_YEARMONTHNUM",
                                              boost::any(199803), EQUAL, LOOKUP,
                                              default_device_constraint));

  boost::shared_ptr<logical_operator::Logical_Join> join(
      new logical_operator::Logical_Join(
          "D_DATEKEY", "LO_ORDERDATE", INNER_JOIN,
          hype::DeviceConstraint(hype::CPU_ONLY)));  // GPU Join not supported

  selection_date->setLeft(scan_date);
  join->setLeft(selection_date);
  join->setRight(scan_lineorder);

  LogicalQueryPlan log_plan(join, std::cout);
  log_plan.print();

  // cout << "Executing Query: " << endl;
  CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
      log_plan.convertToPhysicalQueryPlan();
  plan->print();
  // Timestamp begin=getTimestamp();
  plan->run();
  // Timestamp end=getTimestamp();
  // assert(end>=begin);
  // cout << "Needed " << double(end-begin)/(1000*1000) << "ms ("<<
  // double(end-begin)/(1000*1000*1000) <<"s) to process query '" << "1" <<  "'
  // ..." << endl;
  // cout << "Result has " << plan->getResult()->getNumberofRows() << " rows" <<
  // endl;

  return true;
}

bool simple_SSB_Query_Aggregation() {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
  boost::shared_ptr<logical_operator::Logical_Scan> scan_lineorder(
      new logical_operator::Logical_Scan("LINEORDER"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_date(
      new logical_operator::Logical_Scan("DATE"));

  boost::shared_ptr<logical_operator::Logical_Selection> selection_date(
      new logical_operator::Logical_Selection("D_YEAR", boost::any(1993), EQUAL,
                                              LOOKUP,
                                              default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Join> join(
      new logical_operator::Logical_Join(
          "D_DATEKEY", "LO_ORDERDATE", INNER_JOIN,
          default_device_constraint));  // GPU Join not supported

  selection_date->setLeft(scan_date);
  join->setLeft(selection_date);
  join->setRight(scan_lineorder);

  std::list<std::string> column_list;
  column_list.push_back("D_YEAR");
  column_list.push_back("LO_EXTENDEDPRICE");
  column_list.push_back("LO_DISCOUNT");
  boost::shared_ptr<logical_operator::Logical_Projection> projection(
      new logical_operator::Logical_Projection(column_list));

  projection->setLeft(join);

  boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
      column_algebra_operation_lineorder(
          new logical_operator::Logical_ColumnAlgebraOperator(
              "LO_EXTENDEDPRICE", "LO_DISCOUNT", "REVENUE", MUL,
              default_device_constraint));

  column_algebra_operation_lineorder->setLeft(projection);

  std::list<std::string> sorting_column_names;
  sorting_column_names.push_back("D_YEAR");
  // column_names.push_back("Score");

  SortAttributeList sort_attributes;
  sort_attributes.push_back(SortAttribute("D_YEAR", ASCENDING));

  // boost::shared_ptr<logical_operator::Logical_Sort> sort(new
  // logical_operator::Logical_Sort(sorting_column_names, ASCENDING, LOOKUP,
  // default_device_constraint)); //operation on String column only supported on
  // CPU

  boost::shared_ptr<logical_operator::Logical_Sort> sort(
      new logical_operator::Logical_Sort(
          sort_attributes, LOOKUP, default_device_constraint));  // operation on
                                                                 // String
                                                                 // column only
                                                                 // supported on
                                                                 // CPU

  sort->setLeft(column_algebra_operation_lineorder);

  std::list<std::pair<string, Aggregate> > aggregation_functions;
  aggregation_functions.push_back(
      make_pair("REVENUE", std::make_pair(SUM, "REVENUE")));

  boost::shared_ptr<logical_operator::Logical_Groupby> groupby(
      new logical_operator::Logical_Groupby(sorting_column_names,
                                            aggregation_functions, MATERIALIZE,
                                            default_device_constraint));

  groupby->setLeft(sort);

  LogicalQueryPlan log_plan(groupby,
                            std::cout);  // column_algebra_operation_lineorder);
  log_plan.print();

  CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
      log_plan.convertToPhysicalQueryPlan();
  plan->print();
  plan->run();

  if (!plan->getResult()) return false;

  plan->getResult()->print();
  // boost::shared_ptr<logical_operator::Logical_Scan>  scan_sale(new
  // logical_operator::Logical_Scan("Sale"));
  // boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
  // column_algebra_operation_sale(new
  // logical_operator::Logical_ColumnAlgebraOperator("Price","Sales","Add_Price_Sales_LogOp",ADD,hype::DeviceConstraint(hype::CPU_ONLY)));
  return true;
}

bool simple_SSB_Queries() {
  typedef bool (*SimpleCommandHandlerPtr)();
  typedef std::map<std::string, SimpleCommandHandlerPtr> SimpleCommandMap;
  SimpleCommandMap map;
  //        map.insert(std::make_pair("Q1", &simple_SSB_Query_Join));
  //        map.insert(std::make_pair("Q2", &simple_SSB_Query_Selection));
  //        map.insert(std::make_pair("Q3", &simple_SSB_Query_Aggregation));
  //        map.insert(std::make_pair("ssb11", &SSB_Q11));
  //        map.insert(std::make_pair("ssb12", &SSB_Q12));
  //        map.insert(std::make_pair("ssb13", &SSB_Q13));
  //        map.insert(std::make_pair("ssb21", &SSB_Q21));
  //        map.insert(std::make_pair("ssb22", &SSB_Q22));
  //        map.insert(std::make_pair("ssb23", &SSB_Q23));
  //        map.insert(std::make_pair("ssb31", &SSB_Q31));
  //        map.insert(std::make_pair("ssb32", &SSB_Q32));
  //        map.insert(std::make_pair("ssb33", &SSB_Q33));
  //        map.insert(std::make_pair("ssb34", &SSB_Q34));
  //        map.insert(std::make_pair("ssb41", &SSB_Q41));
  //        map.insert(std::make_pair("ssb42", &SSB_Q42));
  //        map.insert(std::make_pair("ssb43", &SSB_Q43));
  string input;
  cout << "Star Schema Query Test" << endl;
  cout << "Type 'back' to return to cmd" << endl;
  for (SimpleCommandMap::iterator it = map.begin(); it != map.end(); ++it) {
    cout << it->first << endl;
  }
  // workaroudn to use the CommandLineInterpreters History in this sub shell
  CommandLineInterpreter cmd;
  std::string prompt = "CoGaDB>";

  while (true) {
    // cout << "CoGaDB>"; // << endl;
    //            std::getline(std::cin, input);
    //            if (cin.eof()) {
    //                cout << endl;
    //                return true;
    //            }
    if (cmd.getline(prompt, input)) {
      continue;
    }

    if (input == "back") return true;

    SimpleCommandMap::iterator it = map.find(input);
    if (it != map.end()) {
      cout << "Executing Query: " << it->first << endl;
      Timestamp begin = getTimestamp();
      if (!it->second())
        cout << "Error: while processing Query '" << input << "'" << endl;
      Timestamp end = getTimestamp();
      cout << "Needed " << double(end - begin) / (1000 * 1000) << "ms ("
           << double(end - begin) / (1000 * 1000 * 1000)
           << "s) to process query '" << it->first << "' ..." << endl;

    } else {
      cout << "Query '" << input << "' not found" << endl;
    }
    // if(!cmd.execute(input)) cout << "Error! Command '" << input << "'
    // failed!" << endl;
  }

  return true;
}

bool complex_tpch_queries() {
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
  cout << "Running Tests based on the TPC-H benchmark dataset" << endl;

  // scan operations
  // boost::shared_ptr<logical_operator::Logical_Scan> scan_nation(new
  // logical_operator::Logical_Scan("NATION"));
  // boost::shared_ptr<logical_operator::Logical_Scan> scan_region(new
  // logical_operator::Logical_Scan("REGION"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_part(
      new logical_operator::Logical_Scan("PART"));
  // boost::shared_ptr<logical_operator::Logical_Scan> scan_supplier(new
  // logical_operator::Logical_Scan("SUPPLIER"));
  // boost::shared_ptr<logical_operator::Logical_Scan> scan_partsupp(new
  // logical_operator::Logical_Scan("PARTSUPP"));
  // boost::shared_ptr<logical_operator::Logical_Scan> scan_customer(new
  // logical_operator::Logical_Scan("CUSTOMER"));
  // boost::shared_ptr<logical_operator::Logical_Scan> scan_orders(new
  // logical_operator::Logical_Scan("ORDERS"));
  // boost::shared_ptr<logical_operator::Logical_Scan> scan_lineitem(new
  // logical_operator::Logical_Scan("LINEITEM"));

  /*
  Ich suche alle Lieferanten X, die ein Produkt von HERSTELLER Y verkaufen,
  welches einen Wert größer als Z hat. Dabei werden nur Produkte betrachtet,
  deren Lagerstand < A ist.
   */

  // T1 Suche alle Nationen einer Region -> EUROPE heraus
  // boost::shared_ptr<logical_operator::Logical_Selection> regionselection(new
  // logical_operator::Logical_Selection("R_NAME",
  // boost::any(std::string("EUROPE")), EQUAL, LOOKUP,
  // hype::DeviceConstraint(hype::CPU_ONLY)));
  // boost::shared_ptr<logical_operator::Logical_Join> regionnationjoin(new
  // logical_operator::Logical_Join("R_REGIONKEY", "N_REGIONKEY", LOOKUP,
  // default_device_constraint)); //GPU Join not supported
  // T2 Suche alle Produkte eines Herstellers heraus die einen gewissen Preis
  // haben
  boost::shared_ptr<logical_operator::Logical_Selection> partselection1(
      new logical_operator::Logical_Selection(
          "P_SIZE", boost::any(35), LESSER, LOOKUP, default_device_constraint));
  boost::shared_ptr<logical_operator::Logical_Selection> partselection2(
      new logical_operator::Logical_Selection(
          "P_RETAILPRICE", boost::any(float(1100.99)), GREATER, LOOKUP,
          default_device_constraint));
  // T3 Suche Zuordnungen heraus, die eine bestimmte Anzahl haben und joine mit
  // T2
  // boost::shared_ptr<logical_operator::Logical_Selection>
  // partsuppselection(new logical_operator::Logical_Selection("PS_AVAILQTY",
  // boost::any(1000), LESSER, LOOKUP, default_device_constraint));
  // boost::shared_ptr<logical_operator::Logical_Join> partpartsuppjoin(new
  // logical_operator::Logical_Join("P_PARTKEY", "PS_PARTKEY", LOOKUP,
  // default_device_constraint)); //GPU Join not supported
  // T4 Suche alle Supplier einer Region heruas (mit Hilfe von T1)
  // boost::shared_ptr<logical_operator::Logical_Join> t1supplierjoin(new
  // logical_operator::Logical_Join("N_NATIONKEY", "S_NATIONKEY", LOOKUP,
  // default_device_constraint)); //GPU Join not supported
  // T5 führe T3 und T4 zusammen
  // boost::shared_ptr<logical_operator::Logitidscal_Join> t3t4join(new
  // logical_operator::Logical_Join("S_SUPPKEY", "PS_SUPPKEY", LOOKUP,
  // default_device_constraint)); //GPU Join not supported
  // T6 sortiere die Ergebnisse nach verfügbarer Menge
  // boost::shared_ptr<logical_operator::Logical_Sort> availqtysort(new
  // logical_operator::Logical_Sort("PS_AVAILQTY", ASCENDING, LOOKUP,
  // default_device_constraint)); //operation on String column only supported on
  // CPU
  // T7 Pjection
  // std::list<std::string> column_list;
  // column_list.push_back("S_NAME");
  // column_list.push_back("P_NAME");
  // column_list.push_back("S_ACCTBAL");
  // column_list.push_back("P_RETAILPRICE");
  // boost::shared_ptr<logical_operator::Logical_Projection> projection(new
  // logical_operator::Logical_Projection(column_list)); //GPU Projection not
  // supported

  // Führe T1 aus
  // regionselection->setLeft(scan_region);

  // regionnationjoin->setLeft(regionselection);
  // regionnationjoin->setRight(scan_nation);
  // Führe T2 aus
  partselection1->setLeft(scan_part);
  partselection2->setLeft(partselection1);
  // Führe T3 aus
  // partsuppselection->setLeft(scan_partsupp);

  // partpartsuppjoin->setLeft(partselection2);
  // partpartsuppjoin->setRight(partsuppselection);
  // Führe T4 aus
  // t1supplierjoin->setLeft(regionnationjoin);
  // t1supplierjoin->setRight(scan_supplier);
  // Führe T5 aus
  // t3t4join->setLeft(t1supplierjoin);
  // t3t4join->setRight(partpartsuppjoin);
  // Führe T6 aus
  // availqtysort->setLeft(partpartsuppjoin);
  // Führe T7 aus
  // projection->setLeft(partselection2);

  // Erstelle logischen Plan
  LogicalQueryPlan log_plan(partselection2, std::cout);

  log_plan.print();
  // Führe Physischen Plan 30 mal aus
  for (unsigned int i = 0; i < 30; i++) {
    cout << "Executing Query: " << endl;
    CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
        log_plan.convertToPhysicalQueryPlan();

    plan->print();

    plan->run();

    // typename PhysicalQueryPlan::Type
    TablePtr ptr = plan->getResult();
    if (ptr) {
      // ptr->print();
    } else {
      cout << "Error while processing query: pointer to result table is NULL, "
              "no results could be retrieved"
           << endl;

      return false;
    }
  }

  return true;
}

bool Genome_Test_Query(ClientPtr client) {
  // select rb_c_id, rb_position, rb_base_value, genotype(sb_base_value)
  //    as genotype_, genotype_statistics(sb_base_value) as genotype_stats
  // from reference_base join sample_base on rb_id = sb_rb_id where
  //    sb_insert_offset = 0 group by rb_c_id, rb_position, rb_base_value
  // having rb_base_value <> genotype order by rb_c_id, rb_position;

  std::ostream &out = client->getOutputStream();
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan_sample_base(
      new logical_operator::Logical_Scan("SAMPLE_BASE"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_reference_base(
      new logical_operator::Logical_Scan("REFERENCE_BASE"));

  // SELECTION
  Disjunction d1;
  d1.push_back(Predicate(std::string("SB_INSERT_OFFSET"), boost::any(0),
                         ValueConstantPredicate, EQUAL));
  KNF_Selection_Expression selection1;
  selection1.disjunctions.push_back(d1);
  boost::shared_ptr<logical_operator::Logical_ComplexSelection> selection_sb(
      new logical_operator::Logical_ComplexSelection(
          selection1, LOOKUP, default_device_constraint));
  selection_sb->setLeft(scan_sample_base);

  // JOIN
  boost::shared_ptr<logical_operator::Logical_Join> join(
      new logical_operator::Logical_Join(
          "SB_RB_ID", "RB_ID", INNER_JOIN,
          hype::DeviceConstraint(hype::CPU_ONLY)));  // GPU Join not supported
  join->setLeft(selection_sb);
  join->setRight(scan_reference_base);

  // GROUPING and AGGREGATING
  std::list<std::string> grouping_column_names;
  grouping_column_names.push_back("RB_C_ID");
  grouping_column_names.push_back("RB_POSITION");
  grouping_column_names.push_back("RB_BASE_VALUE");

  std::list<ColumnAggregation> aggregation_functions;
  aggregation_functions.push_back(
      ColumnAggregation("SB_BASE_VALUE", Aggregate(AGG_GENOTYPE, "GENOTYPE")));
  aggregation_functions.push_back(ColumnAggregation(
      "SB_BASE_VALUE", Aggregate(AGG_GENOTYPE_STATISTICS, "GENOTYPE_STATS")));

  boost::shared_ptr<logical_operator::Logical_Groupby> groupby(
      new logical_operator::Logical_Groupby(grouping_column_names,
                                            aggregation_functions, LOOKUP,
                                            default_device_constraint));

  groupby->setLeft(join);

  // RENAMING
  // RenameList rename_list;
  // rename_list.push_back(RenameEntry("SB_BASE_VALUE", "GENOTYPE"));
  // boost::shared_ptr<logical_operator::Logical_Rename> rename(new
  // logical_operator::Logical_Rename(rename_list));
  // rename->setLeft(groupby);

  // HAVING
  Disjunction d2;
  d2.push_back(Predicate(std::string("RB_BASE_VALUE"), std::string("GENOTYPE"),
                         ValueValuePredicate, LESSER));
  d2.push_back(Predicate(std::string("RB_BASE_VALUE"), std::string("GENOTYPE"),
                         ValueValuePredicate, GREATER));
  KNF_Selection_Expression selection2;
  selection2.disjunctions.push_back(d2);
  boost::shared_ptr<logical_operator::Logical_ComplexSelection> having(
      new logical_operator::Logical_ComplexSelection(
          selection2, LOOKUP, default_device_constraint));
  having->setLeft(groupby);

  // SORTING
  //        std::list<std::string> sorting_column_names;
  //        sorting_column_names.push_back("RB_C_ID");
  //        sorting_column_names.push_back("RB_POSITION");

  SortAttributeList sortAttributes;
  sortAttributes.push_back(SortAttribute("RB_C_ID", ASCENDING));
  sortAttributes.push_back(SortAttribute("RB_POSITION", ASCENDING));

  boost::shared_ptr<logical_operator::Logical_Sort> sort(
      new logical_operator::Logical_Sort(sortAttributes, LOOKUP,
                                         default_device_constraint));

  sort->setLeft(having);

  // projection
  std::list<std::string> column_list;
  column_list.push_back("RB_C_ID");
  column_list.push_back("RB_POSITION");
  column_list.push_back("RB_BASE_VALUE");
  column_list.push_back("GENOTYPE");
  column_list.push_back("GENOTYPE_STATS");
  boost::shared_ptr<logical_operator::Logical_Projection> projection(
      new logical_operator::Logical_Projection(column_list));

  projection->setLeft(sort);

  LogicalQueryPlan log_plan(projection, out);
  log_plan.print();

  CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
      log_plan.convertToPhysicalQueryPlan();
  plan->print();
  plan->run();

  if (!plan->getResult()) return false;

  plan->getResult()->print();
  return true;
}

bool like_test_query(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  hype::DeviceConstraint default_device_constraint =
      CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();

  boost::shared_ptr<logical_operator::Logical_Scan> scan(
      new logical_operator::Logical_Scan("SUPPLIER"));
  // SELECTION
  Disjunction d1;
  d1.push_back(Predicate(std::string("S_PHONE"),
                         boost::any(std::string("16-.*")),
                         ValueRegularExpressionPredicate, EQUAL));
  KNF_Selection_Expression selection1;
  selection1.disjunctions.push_back(d1);
  boost::shared_ptr<logical_operator::Logical_ComplexSelection> selection(
      new logical_operator::Logical_ComplexSelection(
          selection1, LOOKUP, default_device_constraint));
  selection->setLeft(scan);

  LogicalQueryPlan log_plan(selection, out);
  log_plan.print();

  CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
      log_plan.convertToPhysicalQueryPlan();
  plan->print();
  plan->run();

  if (!plan->getResult()) return false;

  plan->getResult()->print();
  return true;
}

// Returns human readable size and according unit string

std::pair<double, string> getSizeAsHumanReadableString(
    unsigned long size_in_bytes) {
  int unit = 0;
  double size = size_in_bytes;
  const string units[5] = {"bytes", "KB", "MB", "GB", "TB"};
  while (size > 1024 && unit < 4) {
    size /= 1024;
    unit++;
  }
  return std::make_pair(size, units[unit]);
}

bool printDatabaseSize(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<TablePtr> &tables = getGlobalTableList();
  out << "Main memory size of database '";
  out << RuntimeConfiguration::instance().getPathToDatabase();
  out << "':" << endl;

  out << fixed << showpoint << setprecision(2);

  unsigned long overall_size = 0;
  for (auto i = 0ul; i < tables.size(); i++) {
    size_t size_in_bytes = tables[i]->getSizeinBytes();
    overall_size += size_in_bytes;
    std::pair<double, string> size_unit =
        getSizeAsHumanReadableString(size_in_bytes);
    out << "\t" << tables[i]->getName() << ": ";
    out << tables[i]->getNumberofRows() << " rows - ";
    out << size_unit.first << " " << size_unit.second;
    out << " (" << size_in_bytes << " bytes)" << endl;
    // column statistics
    std::vector<ColumnProperties> col_props =
        tables[i]->getPropertiesOfColumns();
    for (size_t j = 0; j < col_props.size(); j++) {
      size_t size_in_bytes =
          col_props[j].size_in_main_memory;  // columns[i]->getSizeinBytes();
      std::pair<double, string> size_unit =
          getSizeAsHumanReadableString(size_in_bytes);
      out << "\t\t" << col_props[j].name << " - ";
      out << util::getName(col_props[j].attribute_type) << " - ";
      out << util::getName(col_props[j].column_type) << " - ";
      out << size_unit.first << " " << size_unit.second;
      out << " (" << size_in_bytes << " bytes)" << endl;
    }
  }
  std::pair<double, string> size_unit =
      getSizeAsHumanReadableString(overall_size);
  out << "Overall: " << size_unit.first;
  out << " " << size_unit.second;
  out << " (" << overall_size << " bytes)" << endl;
  return true;
}

bool printDatabaseSchema(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<TablePtr> &tables = getGlobalTableList();
  out << "Tables in database '"
      << RuntimeConfiguration::instance().getPathToDatabase() << "':" << endl;

  for (unsigned int i = 0; i < tables.size(); i++) {
    out << tables[i]->getName() << " (";

    TableSchema schema = tables[i]->getSchema();
    TableSchema::iterator it;
    for (it = schema.begin(); it != schema.end(); ++it) {
      if (it != schema.begin()) out << ",";
      out << it->second;
    }

    out << ") [" << tables[i]->getNumberofRows() << " rows]" << endl;
  }
  return true;
}

bool include_cogascript_file(const std::string &path_to_source_file,
                             ClientPtr client) {
  CommandLineInterpreter cmd(client);
  return cmd.executeFromFile(path_to_source_file, client);
}

bool compile_and_execute_cpp_query(const std::string &path_to_source_file,
                                   ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  TablePtr result = compileAndExecuteQueryFile(path_to_source_file);
  if (!result) {
    out << "Error: Could compile or executed hard coded query from file: '"
        << path_to_source_file << "'" << std::endl;
    return false;
  } else {
    out << result->toString() << std::endl;
    return true;
  }
}

bool load_and_execute_query_from_JSON(const std::string &path_to_file,
                                      ClientPtr client) {
  std::pair<bool, TablePtr> result =
      load_and_execute_query_from_json(path_to_file, client);
  return result.first;
}

bool drop_table(const std::string &table_name, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  if (!getTablebyName(table_name)) {
    out << "Error: Table '" << table_name << "' not found!" << std::endl;
    return false;
  }

  return dropTable(table_name);
}

bool rename_table(const std::string &param, ClientPtr client) {
  std::ostream &out = client->getOutputStream();

  std::vector<std::string> strs;
  boost::split(strs, param, boost::is_any_of(" "));

  if (strs.size() < 2) {
    out << "Error: Missing Parameter! " << std::endl;
    return false;
  } else if (strs.size() > 2) {
    out << "Error: To Many Parameters! " << std::endl;
    return false;
  }

  std::string table_name = strs[0];
  std::string new_table_name = strs[1];

  if (!getTablebyName(table_name)) {
    out << "Error: Table '" << table_name << "' not found!" << std::endl;
    return false;
  }
  if (getTablebyName(new_table_name)) {
    out << "Error: A Table with name '" << new_table_name << "' already exists!"
        << std::endl;
    return false;
  }

  return renameTable(table_name, new_table_name);
}

bool printDatabaseStatus(ClientPtr client) {
  std::ostream &out = client->getOutputStream();

  TablePtr status = getSystemTableDatabaseSchema();
  if (status) {
    out << status->toString() << std::endl;
  }
  return true;
}

bool addPrimaryKeyConstraintToTable(const std::string &param,
                                    ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<std::string> strs;
  boost::split(strs, param, boost::is_any_of(" "));

  if (strs.size() < 2) {
    out << "Error: Missing Parameter! " << std::endl;
    return false;
  } else if (strs.size() > 2) {
    out << "Error: To Many Parameters! " << std::endl;
    return false;
  }

  TablePtr table = getTablebyName(strs[0]);
  if (!table) {
    out << "Error: Table '" << strs[0] << "' not found!" << std::endl;
    return false;
  }

  if (!table->hasPrimaryKeyConstraint(strs[1])) {
    if (table->setPrimaryKeyConstraint(strs[1])) {
      out << "Set Primary Key Constraint for Column " << strs[0] << "."
          << strs[1] << std::endl;
    } else {
      out << "Error: Could not set Primary Key Constraint for Column "
          << strs[0] << "." << strs[1] << std::endl;
      return false;
    }

  } else {
    out << "Primary Key Constraint already set!" << std::endl;
    return false;
  }

  return true;
}

bool addForeignKeyConstraintToTable(const std::string &param,
                                    ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<std::string> strs;
  boost::split(strs, param, boost::is_any_of(" "));

  if (strs.size() < 4) {
    out << "Error: Missing Parameter! " << std::endl;
    return false;
  } else if (strs.size() > 4) {
    out << "Error: To Many Parameters! " << std::endl;
    return false;
  }

  TablePtr foreign_key_table = getTablebyName(strs[0]);
  if (!foreign_key_table) {
    out << "Error: Table '" << strs[0] << "' not found!" << std::endl;
    return false;
  }

  TablePtr primary_key_table = getTablebyName(strs[2]);
  if (!primary_key_table) {
    out << "Error: Table '" << strs[2] << "' not found!" << std::endl;
    return false;
  }

  if (!foreign_key_table->hasForeignKeyConstraint(strs[1])) {
    if (foreign_key_table->setForeignKeyConstraint(strs[1], strs[3], strs[2])) {
      out << "Set Foreign Key Constraint for Column " << strs[0] << "."
          << strs[1] << " referencing " << strs[2] << "." << strs[3]
          << std::endl;
    } else {
      out << "Error: Could not set Foreign Key Constraint for Column "
          << strs[0] << "." << strs[1] << " referencing " << strs[2] << "."
          << strs[3] << std::endl;
      return false;
    }

  } else {
    out << "Foreign Key Constraint already set!" << std::endl;
    return false;
  }

  return true;
}

bool getColumnTypeFromString(const std::string &column_type_spec,
                             ColumnType &ret) {
  typedef std::map<std::string, ColumnType> ColumnTypeMap;

  static ColumnTypeMap map;
  static bool is_initialized = false;
  if (is_initialized) {
    map.insert(
        std::make_pair(util::getName(PLAIN_MATERIALIZED), PLAIN_MATERIALIZED));
    map.insert(std::make_pair(util::getName(LOOKUP_ARRAY), LOOKUP_ARRAY));
    map.insert(std::make_pair(util::getName(DICTIONARY_COMPRESSED),
                              DICTIONARY_COMPRESSED));
    map.insert(std::make_pair(util::getName(RUN_LENGTH_COMPRESSED),
                              RUN_LENGTH_COMPRESSED));
    map.insert(
        std::make_pair(util::getName(DELTA_COMPRESSED), DELTA_COMPRESSED));
    map.insert(std::make_pair(util::getName(BIT_VECTOR_COMPRESSED),
                              BIT_VECTOR_COMPRESSED));
    map.insert(std::make_pair(util::getName(BITPACKED_DICTIONARY_COMPRESSED),
                              BITPACKED_DICTIONARY_COMPRESSED));
    map.insert(
        std::make_pair(util::getName(RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER),
                       RUN_LENGTH_DELTA_ONE_COMPRESSED_NUMBER));
    map.insert(std::make_pair(util::getName(VOID_COMPRESSED_NUMBER),
                              VOID_COMPRESSED_NUMBER));
    map.insert(std::make_pair(util::getName(REFERENCE_BASED_COMPRESSED),
                              REFERENCE_BASED_COMPRESSED));
    is_initialized = true;
  }
  ColumnTypeMap::const_iterator cit;
  cit = map.find(column_type_spec);
  if (cit != map.end()) {
    ret = cit->second;
    return true;
  }
  return false;
}

bool changeColumnCompression(const std::string &param, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<std::string> strs;
  boost::split(strs, param, boost::is_any_of(" "));

  if (strs.size() < 2) {
    out << "Error: Missing Parameter! " << std::endl;
    return false;
  } else if (strs.size() > 2) {
    out << "Error: To Many Parameters! " << std::endl;
    return false;
  }

  ColumnType col_type;
  if (!getColumnTypeFromString(strs[2], col_type) || col_type == LOOKUP_ARRAY) {
    out << "Error: '" << strs[2]
        << "' is not a valid name for a compression technique!" << std::endl;
    return false;
  }

  TablePtr table = getTablebyName(strs[0]);
  if (!table) {
    out << "Error: Table '" << strs[0] << "' not found!" << std::endl;
    return false;
  }
  ColumnPtr col = table->getColumnbyName(strs[1]);
  if (!col) {
    out << "Error: Column '" << strs[1] << "' not found!" << std::endl;
    return false;
  }
  if (col->getColumnType() == col_type) {
    out << "Error: Column '" << strs[1] << "' is already compressed as '"
        << util::getName(col_type) << "'!" << std::endl;
    return false;
  }
  ColumnPtr new_compressed_column = col->changeCompression(col_type);

  return table->replaceColumn(col->getName(), new_compressed_column);
}

bool importCSVIntoTable(const std::string &param, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<std::string> strs;
  boost::split(strs, param, boost::is_any_of(" "));

  if (strs.size() < 2) {
    out << "Error: Missing Parameter! " << std::endl;
    return false;
  } else if (strs.size() > 2) {
    out << "Error: To Many Parameters! " << std::endl;
    return false;
  }

  TablePtr table = getTablebyName(strs[0]);
  if (!table) {
    out << "Error: Table '" << strs[0] << "' not found!" << std::endl;
    return false;
  }

  if (table->loadDatafromFile(strs[1])) {
    table->store(RuntimeConfiguration::instance().getPathToDatabase());
    return true;
  } else {
    return false;
  }
}

bool exportTableIntoCSV(const std::string &param, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<std::string> strs;
  boost::split(strs, param, boost::is_any_of(" "));

  if (strs.size() < 2) {
    out << "Error: Missing Parameter! " << std::endl;
    return false;
  } else if (strs.size() > 2) {
    out << "Error: To Many Parameters! " << std::endl;
    return false;
  }

  TablePtr table = getTablebyName(strs[0]);
  if (!table) {
    out << "Error: Table '" << strs[0] << "' not found!" << std::endl;
    return false;
  }

  ofstream file(strs[1].c_str(), std::ios::trunc);
  if (!file.good()) {
    out << "Could not open file: '" << strs[1] << "'" << std::endl;
    return false;
  }
  file << table->toString("csv");

  file.close();

  return true;
}

const std::vector<ColumnProperties> getInMemoryColumnProperties(
    ClientPtr client) {
  std::vector<TablePtr> &tables = getGlobalTableList();
  std::vector<ColumnProperties> result;
  for (size_t i = 0; i < tables.size(); ++i) {
    const std::vector<ColumnProperties> props =
        tables[i]->getPropertiesOfColumns();
    for (size_t j = 0; j < props.size(); ++j) {
      if (props[j].is_in_main_memory) {
        result.push_back(props[j]);
      }
    }
  }
  return result;
}

bool printInMemoryColumns(ClientPtr client) {
  std::ostream &out = client->getOutputStream();

  out << "Tables in database '"
      << RuntimeConfiguration::instance().getPathToDatabase() << "':" << endl;

  const std::vector<ColumnProperties> props =
      getInMemoryColumnProperties(client);
  for (size_t j = 0; j < props.size(); ++j) {
    out << props[j].name << " (Type: " << util::getName(props[j].attribute_type)
        << ", Referenced: " << props[j].number_of_accesses
        << ", Compressed: " << util::getName(props[j].column_type)
        << ", In Memory: "
        << double(props[j].size_in_main_memory) / (1024 * 1024) << " MB)"
        << std::endl;
  }
  return true;
}

bool printFootprintOfInMemoryColumns(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  out << "Tables in database '"
      << RuntimeConfiguration::instance().getPathToDatabase() << "':" << endl;
  const std::vector<ColumnProperties> props =
      getInMemoryColumnProperties(client);
  size_t size_of_working_set_in_byte = 0;
  for (size_t j = 0; j < props.size(); ++j) {
    out << props[j].name << std::endl;
    size_of_working_set_in_byte += props[j].size_in_main_memory;
  }
  out << "SIZE_OF_IN_MEMORY_COLUMNS_BYTE: " << size_of_working_set_in_byte
      << std::endl;
  out << "SIZE_OF_IN_MEMORY_COLUMNS_GB: "
      << double(size_of_working_set_in_byte) / (1024 * 1024 * 1024)
      << std::endl;

  return true;
}

bool printStatisticsOfInMemoryColumns(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::vector<TablePtr> &tables = getGlobalTableList();
  std::vector<ColumnProperties> result;
  for (size_t i = 0; i < tables.size(); ++i) {
    const std::vector<ColumnProperties> props =
        tables[i]->getPropertiesOfColumns();
    for (size_t j = 0; j < props.size(); ++j) {
      if (props[j].is_in_main_memory) {
        ColumnPtr col = tables[i]->getColumnbyName(props[j].name);
        assert(col != NULL);
        out << "Column: " << props[j].name << std::endl;
        out << col->getColumnStatistics().toString() << std::endl;
      }
    }
  }
  return true;
}

bool toggleQueryChopping(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  RuntimeConfiguration::instance().setQueryChoppingEnabled(
      !RuntimeConfiguration::instance().isQueryChoppingEnabled());
  if (RuntimeConfiguration::instance().isQueryChoppingEnabled()) {
    out << "Query Chopping is now activated." << endl;
  } else {
    out << "Query Chopping is now deactivated." << endl;
  }
  return true;
}

bool about(ClientPtr client) {
  std::ostream &out = client->getOutputStream();

  out << "*********************************************************************"
         "***************************************"
      << endl;
  out << "Copyright (c) 2014, Sebastian Breß, Otto-von-Guericke University of "
         "Magdeburg, Germany. All rights reserved."
      << endl;
  out << "" << endl;
  out << "This program and accompanying materials are made available under the "
         "terms of the "
      << endl;
  out << "GNU GENERAL PUBLIC LICENSE - Version 3, "
         "http://www.gnu.org/licenses/gpl-3.0.txt"
      << endl;
  out << "*********************************************************************"
         "***************************************"
      << endl;

  out << endl << "Credits:" << endl << endl;
  out << "Project members:" << endl;
  out << "\tSebastian Breß (University of Magdeburg)" << endl;
  out << "\tRobin Haberkorn (University of Magdeburg)" << endl;
  out << "\tSteven Ladewig (University of Magdeburg)" << endl;
  out << "\tTobias Lauer (Jedox AG)" << endl;
  out << "\tManh Lan Nguyen (University of Magdeburg)" << endl;
  out << "\tGunter Saake (University of Magdeburg)" << endl;
  out << "\tNorbert Siegmund (University of Passau)" << endl;
  out << endl;

  out << "Contributors:" << endl;
  out << "\tDarius Brückers (contributed Compression Technique: Run Length "
         "Encoding)"
      << endl;
  out << "\tSebastian Krieter (contributed Compression Technique: Delta Coding)"
      << endl;
  out << "\tSteffen Schulze (contributed Compression Technique: Bit Vector "
         "Encoding)"
      << endl;
  out << endl;

  out << "Former project members:" << endl;
  out << "\tRené Hoyer (University of Magdeburg)" << endl;
  out << "\tPatrick Sulkowski (University of Magdeburg)" << endl;
  out << endl;

  return true;
}

bool version(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  out << "CoGaDB " << COGADB_VERSION << endl;
  return true;
}

bool printSystemTableIntegrityConstraints(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  TablePtr tab = DataDictionary::instance().getTableWithIntegrityConstraints();
  if (!tab) return false;
  out << tab->toString() << std::endl;
  return true;
}

bool printSystemTableJoinIndexes(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  TablePtr tab = JoinIndexes::instance().getSystemTable();
  if (!tab) return false;
  out << tab->toString() << std::endl;
  return true;
}

bool printCollectedStatistics(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  std::string result = StatisticsManager::instance().toString();
  out << result << std::endl;
  return true;
}

bool resetCollectedStatistics(ClientPtr client) {
  std::ignore = client;
  StatisticsManager::instance().reset();
  return true;
}

bool help(ClientPtr client) {
  std::ostream &out = client->getOutputStream();

  out << "supported commands:" << endl
      << "###################" << endl
      << "loaddatabase\t\t\tloads complete database in main memory" << endl
      << "unittests\t\t\tperforms a self check of CoGaDB" << endl
      << "printschema\t\t\tprints the schema of the active database" << endl
      << "databasestatus\t\t\tshows the detailed database schema of the active "
         "database"
      << endl
      << "printdatabasesize\t\t\tprints size of the active database" << endl
      << "printstatistics\tprints collected statistics" << endl
      << "resetstatistics\tdeletes collected statistics and start collecting "
         "statistics from scratch"
      << endl
      << "printmemoryusage\tPrints the amount of total main memory currently "
         "used by CoGaDB"
      << endl
      << "print_in_memory_columns\tPrint names of column that are loaded in "
         "main memory"
      << endl
      << "print_memory_footprint_of_in_memory_columns\tPrints the amount of "
         "main memory used by persistent columns (excludes memory for "
         "intermediate results and join indexes)"
      << endl
      << "print_column_statistics_of_in_memory_columns\tPrints column "
         "statistics of columns in main memory"
      << endl
      << "list_gpus\tlists all available GPUs and provides some basic "
         "information about them"
      << endl
      << "showgpucache\t\t\tprints status information of the GPU column cache"
      << endl
      //<< "simple_ssb_queries\tsimple demonstrator for queries on SSB Benchmark
      // data set" << endl
      << "exec <SQL>\t\t\tExecute SQL statements" << endl
      << "explain <SQL>\t\t\tDisplay query plan generated from SQL expression"
      << endl
      << "explain_unoptimized <SQL>\tAs above, but does not apply logical "
         "optimizer before showing the plan"
      << endl
      << "exec_cmdline <bash commands>\tExecutes an arbitrary shell command "
         "using Bash"
      << endl
      << "compile_and_execute_query <path_to_file>\tCompiles file and executes "
         "the query function"
      << endl
      << "redirectoutput <FILENAME>\tRedirects output to file FILENAME" << endl
      << "listen <PORT>\t\t\tListens on port <PORT> for incoming TCP "
         "connections"
      << endl
      << "execute_query_from_json <FILENAME>\tloads a query plan from a JSON "
         "file and executes it"
      << endl
      << "rename_table <TABLE_NAME> <NEW_TABLE_NAME>\trenames a table in the "
         "database"
      << endl
      << "drop_table <TABLE_NAME>\tdeletes a table from the database" << endl
      << "hypestatus \t\t\tPrints all operations and corresponding algorithms "
         "registered in HyPE for CoGaDB'S operators"
      << endl
      << "integrityconstraints \t\tPrints integrity constraints configured for "
         "current database"
      << endl
      << "add_primary_key_constraint\t<TABLE NAME> <COLUMN NAME>" << endl
      << "add_foreign_key_constraint\t<FK_TABLE NAME> <FK_COLUMN NAME> "
         "<PK_TABLE NAME> <PK_COLUMN NAME>"
      << endl
      << "import_csv_file \t\t<TABLE_NAME> <CSV_FILE>" << endl
      << "export_csv_file \t\t<TABLE_NAME> <CSV_FILE>" << endl
      << "joinindexes \t\t\tPrints join indexes for current database" << endl
      << "loadjoinindexes\t\t\tLoads join indexes found on disk into main "
         "memory"
      << endl
      << "placejoinindexes\t\tPlaces selected join indexes from main memory "
         "into GPU memory"
      << endl
      << "placecolumns\t\t\tPlaces selected columns from main memory into GPU "
         "memory"
      << endl
      << "placecolumn <MEM_ID> <TABLE> <COLUMN>\tPlaces selected column from "
         "main memory into selected memory"
      << endl
      << "placecolumnfrequencybased <MEM_ID>\tPlaces most frequently accessed "
         "columns in the cache, until buffer size is reached"
      << endl
      << "toggleQC\t\t\tToggle the state of Query Chopping activation. Per "
         "default QC is off."
      << endl
      << "ssbXY \t\t\t\tExecute SSB-Query X.Y (X has to be a number between 1 "
         "and 4; Y has to be a number between 1 and 3 except when X is 3, in "
         "this case 4 is valid for Y as well)"
      << endl
      << "setdevice <DEVICE> \t\tSets the default device, which is used for "
         "execution. Possible values are 'cpu', 'gpu' or 'any' to use either "
         "the CPU or the GPU or both."
      << endl
      << endl
      << "set_global_load_adaption_policy <POLICY>\tSets the default "
         "recomputation heuristic (policy) for HyPE'S load adaption mode. "
         "Valid values are 'periodic_recomputation' and 'no_recomputation'."
      << endl
      << "setparallelizationmode <PARALLELIZATION MODE> \tSets the default "
         "parallelization mode for Subplans generated during Two Phase "
         "Physical Optimization (TOPPO) in the second phase (currently only "
         "for complex selections). Valid values are 'serial' and 'parallel'"
      << endl
      << "create_tpch_database <path to *.tbl files>\timport tables of TPC-H "
         "benchmark in CoGaDB"
      << endl
      << "create_ssb_database <path to *.tbl files>\timport tables of star "
         "schema benchmark in CoGaDB"
      << endl
      << endl
      << "starttimer\tstarts internal timer" << endl
      << "stoptimer\tstops internal timer an prints worklaod execution time"
      << endl
      << "about \t\tshows credits" << endl
      << "version\t\tshows version of CoGaDB" << endl
      << "set <variablename>=<variablevalue>" << endl
      << "print <variable>\t\tprint value of variable, for a full list a "
         "supported variables refer read ahead"
      << endl
      << "quit" << endl
      << "exit" << endl
#ifdef BAM_FOUND
      << endl
      << "genomics extension:" << endl
      << "\t"
      << "create_genome_database_schema" << endl
      << "\t"
      << "import_reference_genome <path_to_fasta_file> "
         "[<reference_genome_name>]\timports genome data form fasta file into "
         "existing genome schema"
      << endl
      << "\t"
      << "import_sample_genome <path_to_sam/bam_file> "
         "[<sample_genome_name>]\timports genome data fro sam/bam file into "
         "existing genome schema"
      << endl
#endif
      << endl
      << "supported variables:" << endl
      << "####################" << endl
      << "path_to_database\tabsolute or relative path to directory where the "
         "database is stored"
      << endl
      << "optimizer\t\tname of optimizer used to optimize query plans "
         "(default_optimizer, star_join_optimizer, chain_join_optimizer, "
         "no_join_order_optimizer, gather_join_optimizer)"
      << endl
      << "hybrid_query_optimizer\tname of physical optimizer heuristic "
         "(backtracking, greedy_heuristic, interactive)"
      << endl
      << "print_query_plan\tprint the generated query plans for a SQL query "
         "(true,false)"
      << endl
      << "enable_profiling\tprint the query execution plan after execution "
         "with timings for each operator (true,false)"
      << endl
      << "enable_pull_based_query_chopping\tmakes query chopping pull based"
      << endl
      << "gpu_buffer_size\t\tGPU Buffer size in byte (unsigned integer)" << endl
      << "gpu_buffer_management_strategy\tBuffer management strategy for GPU "
         "Buffer (least_recently_used,least_frequently_used), ignored after "
         "call to 'placejoinindexes'"
      << endl
      << "pin_columns_in_gpu_buffer\tPins columns in the GPU buffer, meaning "
         "that they cannot be evicted after they are cached"
      << endl
      << "pin_join_indexes_in_gpu_buffer\tPins join indexes in the GPU buffer, "
         "meaning that they cannot be evicted after they are cached"
      << endl
      << "enable_dataplacement_aware_query_optimization\tmake the query "
         "optimization driven by the current data placement, requires prior "
         "call to 'placejoinindexes'"
      << endl
      << "result_output_format\tChanges the format the result is displayed "
         "(table, csv)"
      << endl
      << "print_query_result\tPrint result table of queries (true, false)"
      << endl
      << "keep_last_generated_query_code\tKeep generated C code of last "
         "compiled pipeline (true, false)"
      << endl
      << endl
      << "use_radix_hash_join\tConfigures whether to use the default hash join "
         "or the radix hash join"
      << endl
      << "Query Compilation:" << endl
      << "query_execution_policy\tsteers whether queries are executed by "
         "compilation or interpretation (compiled, interpreted)"
      << endl
      << "show_generated_code\tshows generated code in stdout (true, false)"
      << endl
      << "default_code_generator\tselects code generator (cpp, c, cuda, "
         "multi_staged)"
      << endl
      << "debug_code_generator\tshows debug output for code generators (true, "
         "false)"
      << endl
      << "enable_parallel_pipelines\texecutes compiled pipelines in parallel "
         "if possible (true, false)"
      << endl
      << "generate_llvm_ir\tadditionally generates LLVM IR code for generated "
         "pipeline (true, false)"
      << endl
      << "cleanup_generated_files\tcleanup files generated by code generator "
         "(true,false)"
      << endl
      << "profiling.keep_shared_libraries_loaded\tKeep dynamically loaded "
         "libraries (no dlclose), useful for profiling generated code (e.g., "
         "with callgrind)"
      << endl
#ifdef BAM_FOUND
      << endl
      << "genomics extension:" << endl
      << "\t"
      << "genome_schema_type\tspecifies genome data schema for importers "
         "(weak_entities, default:simple_keys)"
      << endl
      << "\t"
      << "genome_schema_compression\tspecifies whether the chosen genome data "
         "schema is compressed (true, default:false)"
      << endl
      << "\t"
      << "genotype_frequency\tspecifies threshold of frequency based variant "
         "calling approach ([0.0 - 1.0], default:0.8)"
      << endl
#endif
      ;
  return true;
}

bool setEstimate(const std::string &input, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  unsigned int tokencounter = 0;
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep("=");
  tokenizer tok(input, sep);
  std::string variable_name;
  std::string parameter;
  for (tokenizer::iterator beg = tok.begin(); beg != tok.end(); ++beg) {
    if (tokencounter == 0) variable_name = *beg;
    if (tokencounter == 1) parameter = *beg;
    tokencounter++;
  }
  if (tokencounter < 2) {
    out << "Error! For Command set-estimate: Invalid Expression: \"" << input
        << "\" set-estimate <table-name>=<pipeline-selectivity>" << endl;
    return false;
  }
  if (tokencounter > 2) {
    out << "Error! For Command set-estimate: Too Many Parameters: \"" << input
        << "\" set-estimate <table-name>=<pipeline-selectivity>" << endl;
    return false;
  }
  if (variable_name == "all") {
    PipelineSelectivityTable::instance().dropSelectivities();
  }
  double selectivity = 0.0;
  try {
    selectivity = boost::lexical_cast<double>(parameter);
  } catch (const boost::bad_lexical_cast &e) {
    out << "Error! Second value has to be double precision float. \"" << input
        << "\" set-estimate <table-name>=<pipeline-selectivity>" << endl;
    return false;
  }
  PipelineSelectivityTable::instance().updateSelectivity(variable_name,
                                                         selectivity);
  return true;
}

bool setVariable(const std::string &input, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  unsigned int tokencounter = 0;
  typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
  boost::char_separator<char> sep("=");
  tokenizer tok(input, sep);
  std::string variable_name;
  std::string parameter;
  for (tokenizer::iterator beg = tok.begin(); beg != tok.end(); ++beg) {
    // cout << *beg << "\n";
    if (tokencounter == 0) variable_name = *beg;
    if (tokencounter == 1) parameter = *beg;
    tokencounter++;
  }

  if (tokencounter < 2) {
    out << "Error! For Command SET: Invalid Expression: \"" << input
        << "\" <variable>=<value>" << endl;
    return false;
  }
  if (tokencounter > 2) {
    out << "Error! For Command SET: To many parameters: <variable>=<value>"
        << endl;
    return false;
  }

  int current_cuda_device_id =
      VariableManager::instance().getVariableValueInteger(
          "current_cuda_device_id");
  hype::ProcessingDeviceMemoryID current_cuda_device_mem_id =
      getMemoryIDForDeviceID(current_cuda_device_id);

  if (variable_name == "path_to_database") {
    RuntimeConfiguration::instance().setPathToDatabase(parameter);
  } else if (variable_name == "optimizer") {
    if (parameter != "default_optimizer" &&
        parameter != "star_join_optimizer" &&
        parameter != "chain_join_optimizer" &&
        parameter != "no_join_order_optimizer" &&
        parameter != "gather_join_optimizer") {
      out << "Error! For Command SET: Invalid parameter: " << parameter << endl;
      out << "Valid values are: default_optimizer, star_join_optimizer, "
          << "chain_join_optimizer, no_join_order_optimizer, "
          << "gather_join_optimizer" << endl;
      return false;
    }
    RuntimeConfiguration::instance().setOptimizer(parameter);
  } else if (variable_name == "hybrid_query_optimizer") {
    if (parameter != "backtracking" && parameter != "greedy_heuristic" &&
        parameter != "interactive" && parameter != "greedy_chainer_heuristic" &&
        parameter != "critical_path_heuristic" &&
        parameter != "best_effort_gpu_heuristic") {
      out << "Error! For Command SET: Invalid parameter: " << parameter << endl;
      out << "Valid values are: backtracking, greedy_heuristic, interactive, "
             "greedy_chainer_heuristic, critical_path_heuristic"
          << endl;
      return false;
    }
    hype::QueryOptimizationHeuristic opt = hype::GREEDY_HEURISTIC;
    if (parameter == "backtracking") {
      opt = hype::BACKTRACKING;
    } else if (parameter == "greedy_heuristic") {
      opt = hype::GREEDY_HEURISTIC;
    } else if (parameter == "interactive") {
      opt = hype::INTERACTIVE_USER_OPTIMIZATION;
    } else if (parameter == "greedy_chainer_heuristic") {
      opt = hype::GREEDY_CHAINER_HEURISTIC;
    } else if (parameter == "critical_path_heuristic") {
      opt = hype::CRITICAL_PATH_HEURISTIC;
    } else if (parameter == "best_effort_gpu_heuristic") {
      opt = hype::BEST_EFFORT_GPU_HEURISTIC;
    }
    RuntimeConfiguration::instance().setQueryOptimizationHeuristic(opt);

  } else if (variable_name == "gpu_buffer_management_strategy") {
    GPUBufferManagementStrategy strategy;
    if (parameter == "least_recently_used") {
      strategy = LEAST_RECENTLY_USED;
      DataCacheManager::instance()
          .getDataCache(current_cuda_device_mem_id)
          .setCacheEnabledStatus(true);
    } else if (parameter == "least_frequently_used") {
      strategy = LEAST_FREQUENTLY_USED;
      DataCacheManager::instance()
          .getDataCache(current_cuda_device_mem_id)
          .setCacheEnabledStatus(true);
    } else if (parameter == "disbled_gpu_buffer") {
      strategy = DISABLED_GPU_BUFFER;
      DataCacheManager::instance()
          .getDataCache(current_cuda_device_mem_id)
          .setCacheEnabledStatus(false);
    } else {
      out << "Error! For Command SET: Invalid parameter: " << parameter << endl;
      out << "Valid values are: least_recently_used, least_frequently_used, "
             "disbled_gpu_buffer"
          << endl;
      return false;
    }
    RuntimeConfiguration::instance().setGPUBufferManagementStrategy(strategy);

  } else if (variable_name == "print_query_plan") {
    if (parameter == "true" || parameter == "TRUE" || parameter == "ON") {
      RuntimeConfiguration::instance().setPrintQueryPlan(true);
    } else if (parameter == "false" || parameter == "FALSE" ||
               parameter == "OFF") {
      RuntimeConfiguration::instance().setPrintQueryPlan(false);
    } else {
      out << "Error! For Command SET: Invalid parameter: " << parameter << endl;
      out << "Valid values are: true, false" << endl;
      return false;
    }
  } else if (variable_name == "enable_profiling") {
    if (parameter == "true" || parameter == "TRUE" || parameter == "ON") {
      RuntimeConfiguration::instance().setProfileQueries(true);
    } else if (parameter == "false" || parameter == "FALSE" ||
               parameter == "OFF") {
      RuntimeConfiguration::instance().setProfileQueries(false);
    } else {
      out << "Error! For Command SET: Invalid parameter: " << parameter << endl;
      out << "Valid values are: true, false" << endl;
      return false;
    }
  } else if (variable_name == "enable_dataplacement_aware_query_optimization") {
    if (parameter == "true" || parameter == "TRUE" || parameter == "ON") {
      hype::core::Runtime_Configuration::instance()
          .setDataPlacementDrivenOptimization(true);
    } else if (parameter == "false" || parameter == "FALSE" ||
               parameter == "OFF") {
      hype::core::Runtime_Configuration::instance()
          .setDataPlacementDrivenOptimization(false);
    } else {
      out << "Error! For Command SET: Invalid parameter: " << parameter << endl;
      out << "Valid values are: true, false" << endl;
      return false;
    }
  } else if (variable_name == "gpu_buffer_size") {
    size_t new_buffer_size = 0;
    try {
      new_buffer_size = boost::lexical_cast<size_t>(parameter);
    } catch (const boost::bad_lexical_cast &) {
      out << "Error! For Command SET: Invalid parameter: '" << parameter << "'"
          << endl;
      out << "Valid values are unsigned integers!" << endl;
      return false;
    };

    // if (GPU_Column_Cache::instance().setGPUBufferSizeInByte(new_buffer_size))
    // {
    if (DataCacheManager::instance()
            .getDataCache(current_cuda_device_mem_id)
            .setBufferSizeInByte(new_buffer_size)) {
      out << "SET Variable: '" << variable_name << "' to '" << new_buffer_size
          << "' bytes (" << double(new_buffer_size) / (1024 * 1024) << " MB)"
          << endl;
    } else {
      out << "Error! For Command SET: specified buffer size (" << parameter
          << " bytes) larger than physical available memory!" << endl;
      return false;
    }

  } else if (variable_name == "pin_columns_in_gpu_buffer") {
    if (parameter == "true" || parameter == "TRUE" || parameter == "ON") {
      //                GPU_Column_Cache::instance().pinColumnsOnGPU(true);
      DataCacheManager::instance()
          .getDataCache(current_cuda_device_mem_id)
          .pinColumns(true);
    } else if (parameter == "false" || parameter == "FALSE" ||
               parameter == "OFF") {
      //                GPU_Column_Cache::instance().pinColumnsOnGPU(false);
      DataCacheManager::instance()
          .getDataCache(current_cuda_device_mem_id)
          .pinColumns(false);
    } else {
      out << "Error! For Command SET: Invalid parameter: " << parameter << endl;
      out << "Valid values are: true, false" << endl;
      return false;
    }

  } else if (variable_name == "pin_join_indexes_in_gpu_buffer") {
    if (parameter == "true" || parameter == "TRUE" || parameter == "ON") {
      DataCacheManager::instance()
          .getDataCache(current_cuda_device_mem_id)
          .pinJoinIndexes(true);
    } else if (parameter == "false" || parameter == "FALSE" ||
               parameter == "OFF") {
      DataCacheManager::instance()
          .getDataCache(current_cuda_device_mem_id)
          .pinJoinIndexes(false);
    } else {
      out << "Error! For Command SET: Invalid parameter: " << parameter << endl;
      out << "Valid values are: true, false" << endl;
      return false;
    }

  } else if (variable_name == "enable_pull_based_query_chopping") {
    if (parameter == "true" || parameter == "TRUE" || parameter == "ON") {
      hype::core::Runtime_Configuration::instance()
          .setPullBasedQueryChoppingEnabled(true);
    } else if (parameter == "false" || parameter == "FALSE" ||
               parameter == "OFF") {
      hype::core::Runtime_Configuration::instance()
          .setPullBasedQueryChoppingEnabled(false);
    } else {
      out << "Error! For Command SET: Invalid parameter: " << parameter << endl;
      out << "Valid values are: true, false" << endl;
      return false;
    }

  } else {
    // if (VariableManager::instance().setVariableValue(variable_name,
    // boost::to_lower_copy(parameter))) return true;
    if (VariableManager::instance().setVariableValue(variable_name, parameter))
      return true;

    out << "Error! For Command SET: Unknown variable: '" << variable_name << "'"
        << endl;
    return false;
  }
  if (variable_name != "gpu_buffer_size")
    out << "SET Variable: '" << variable_name << "' to '" << parameter << "'"
        << endl;
  return true;
}

bool setDefaultDevice(const std::string &input, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  if (input == "cpu") {
    // default_device_constraint = hype::DeviceConstraint(hype::CPU_ONLY);
    CoGaDB::RuntimeConfiguration::instance().setGlobalDeviceConstraint(
        hype::DeviceConstraint(hype::CPU_ONLY));
    out << "Set device to CPU." << endl;
  } else if (input == "gpu") {
    // default_device_constraint = hype::DeviceConstraint(hype::GPU_ONLY);
    CoGaDB::RuntimeConfiguration::instance().setGlobalDeviceConstraint(
        hype::DeviceConstraint(hype::GPU_ONLY));
    out << "Set device to GPU." << endl;
  } else if (input == "any") {
    // default_device_constraint = hype::DeviceConstraint(hype::ANY_DEVICE);
    CoGaDB::RuntimeConfiguration::instance().setGlobalDeviceConstraint(
        hype::DeviceConstraint(hype::ANY_DEVICE));
    out << "Set device to any device." << endl;
  } else {
    out << "No device with the given devicecode could be found. Try 'cpu', "
           "'gpu' or 'any' to switch the defaultdevice."
        << endl;

    return false;
  }
  return true;
}

bool setGlobalLoadAdaptionPolicy(const std::string &input, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  if (input == "periodic_recomputation") {
    hype::Scheduler::instance().setGlobalLoadAdaptionPolicy(hype::Periodic);
  } else if (input == "no_recomputation") {
    hype::Scheduler::instance().setGlobalLoadAdaptionPolicy(
        hype::No_Recomputation);
  } else {
    out << "Unkown recomputation heuristic: '" << input
        << "'. Try 'periodic_recomputation' or 'no_recomputation'." << endl;

    return false;
  }
  out << "Set HyPE's global load adaption policy to " << input << "." << endl;
  return true;
}

bool setDefaultParallelizationModeTwoPhasePhysicalOptimization(
    const std::string &input, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  if (input == "serial") {
    // default_device_constraint = hype::DeviceConstraint(hype::CPU_ONLY);
    CoGaDB::RuntimeConfiguration::instance()
        .setParallelizationModeForTwoPhasePhysicalOptimizationQueryPlans(
            CoGaDB::SERIAL);
    out << "Set TOPPO parallelization mode to serial." << endl;
  } else if (input == "parallel") {
    // default_device_constraint = hype::DeviceConstraint(hype::GPU_ONLY);
    CoGaDB::RuntimeConfiguration::instance()
        .setParallelizationModeForTwoPhasePhysicalOptimizationQueryPlans(
            CoGaDB::PARALLEL);
    out << "Set TOPPO parallelization mode to parallel." << endl;
  } else {
    out << "Unknown TOPPO Parallelization mode! Try 'serial' or 'parallel' to "
           "switch the TOPPO (Two Phase Physical Optimization) parallelization "
           "mode."
        << endl;

    return false;
  }
  return true;
}

bool printVariableValue(const std::string &input, ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  int current_cuda_device_id =
      VariableManager::instance().getVariableValueInteger(
          "current_cuda_device_id");
  hype::ProcessingDeviceMemoryID current_cuda_device_mem_id =
      getMemoryIDForDeviceID(current_cuda_device_id);
  if (input == "path_to_database") {
    out << "'" << RuntimeConfiguration::instance().getPathToDatabase() << "'"
        << endl;
  } else if (input == "hybrid_query_optimizer") {
    out << "'"
        << hype::util::getName(
               RuntimeConfiguration::instance().getQueryOptimizationHeuristic())
        << "'" << endl;
  } else if (input == "gpu_buffer_management_strategy") {
    out << "'" << util::getName(RuntimeConfiguration::instance()
                                    .getGPUBufferManagementStrategy())
        << "'" << endl;
  } else if (input == "gpu_buffer_size") {
    // out << GPU_Column_Cache::instance().getGPUBufferSize() << " bytes (" <<
    // double(GPU_Column_Cache::instance().getGPUBufferSize()) / (1024 * 1024)
    // << " MB)" << endl;
    out << DataCacheManager::instance()
               .getDataCache(hype::PD_Memory_1)
               .getBufferSize()
        << " bytes ("
        << double(DataCacheManager::instance()
                      .getDataCache(hype::PD_Memory_1)
                      .getBufferSize()) /
               (1024 * 1024)
        << " MB)" << endl;
  } else if (input == "optimizer") {
    out << "'" << RuntimeConfiguration::instance().getOptimizer() << "'"
        << endl;
  } else if (input == "print_query_plan") {
    if (RuntimeConfiguration::instance().getPrintQueryPlan()) {
      out << "'true'" << endl;
    } else {
      out << "'false'" << endl;
    }
  } else if (input == "enable_profiling") {
    if (RuntimeConfiguration::instance().getProfileQueries()) {
      out << "'true'" << endl;
    } else {
      out << "'false'" << endl;
    }
    //            cout << "'" <<
    //            RuntimeConfiguration::instance().getPrintQueryPlan() << "'" <<
    //            endl;
  } else if (input == "enable_dataplacement_aware_query_optimization") {
    if (hype::core::Runtime_Configuration::instance()
            .getDataPlacementDrivenOptimization()) {
      out << "'true'" << endl;
    } else {
      out << "'false'" << endl;
    }
  } else if (input == "pin_columns_in_gpu_buffer") {
    if (DataCacheManager::instance()
            .getDataCache(current_cuda_device_mem_id)
            .haveColumnsPinned()) {
      out << "'true'" << endl;
    } else {
      out << "'false'" << endl;
    }
  } else if (input == "pin_join_indexes_in_gpu_buffer") {
    if (DataCacheManager::instance()
            .getDataCache(current_cuda_device_mem_id)
            .haveJoinIndexesPinned()) {
      out << "'true'" << endl;
    } else {
      out << "'false'" << endl;
    }
  } else if (input == "enable_pull_based_query_chopping") {
    if (hype::core::Runtime_Configuration::instance()
            .isPullBasedQueryChoppingEnabled()) {
      out << "'true'" << endl;
    } else {
      out << "'false'" << endl;
    }
  } else {
    std::string result =
        VariableManager::instance().getVariableValueString(input);
    if (result != "INVALID") {
      out << result << endl;
      return true;
    }
    out << "Error in print: Invalid variable name: '" << input << "'" << endl;
  }

  return true;
}

static uint64_t workload_execution_begin_timestamp = 0;
static uint64_t workload_execution_end_timestamp = 0;

static clock_t start_cpu_time_measurement = 0;
static clock_t end_cpu_time_measurement = 0;

bool startWorkloadExecutionTimer(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  workload_execution_begin_timestamp = getTimestamp();
  start_cpu_time_measurement = clock();
  out << "Started timer to measure workload execution time..." << std::endl;
  return true;
}

bool stopWorkloadExecutionTimer(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  workload_execution_end_timestamp = getTimestamp();
  end_cpu_time_measurement = clock();
  if (workload_execution_begin_timestamp == 0) {
    COGADB_ERROR("stoptimer called without prior call to starttimer!", "");
    out << "stoptimer called without prior call to starttimer!" << std::endl;
  } else {
    out << "WORKLOAD EXECUTION TIME: "
        << double(workload_execution_end_timestamp -
                  workload_execution_begin_timestamp) /
               (1000 * 1000)
        << "ms\t"
        << double(workload_execution_end_timestamp -
                  workload_execution_begin_timestamp) /
               (1000 * 1000 * 1000)
        << "s" << std::endl;
    double total_cpu_time_used =
        static_cast<double>(end_cpu_time_measurement -
                            start_cpu_time_measurement) /
        CLOCKS_PER_SEC;
    out << "TOTAL CPU TIME: " << total_cpu_time_used * 1000 << "ms\t"
        << total_cpu_time_used << "s" << std::endl;
  }

  workload_execution_begin_timestamp = 0;
  workload_execution_end_timestamp = 0;
  return true;
}

bool printMainMemoryUsage(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  double memory_footprint_in_gb =
      double(getUsedMainMemoryInBytes()) / (1024 * 1024 * 1024);
  double memory_footprint_gpu_in_gb =
      double(HardwareDetector::instance().getFreeMemorySizeInByte(
          hype::PD_Memory_1)) /
      (1024 * 1024 * 1024);
  out << "Current memory usage: " << memory_footprint_in_gb << "GiB"
      << std::endl;
  out << "Current GPU memory usage: " << memory_footprint_gpu_in_gb << "GiB"
      << std::endl;
  return true;
}

bool CheckValueOfVariableTableLoaderMode(const std::string &val) {
  if (val == "disk" || val == "main_memory")
    return true;
  else
    return false;
}

bool SetValueOfVariableTableLoaderMode(VariableState &var_stat,
                                       const std::string &new_val) {
  std::ignore = var_stat;

  if (!CheckValueOfVariableTableLoaderMode(new_val)) return false;
  if (new_val == "disk") {
    RuntimeConfiguration::instance().setTableLoaderMode(LOAD_NO_COLUMNS);
  } else if (new_val == "main_memory") {
    RuntimeConfiguration::instance().setTableLoaderMode(LOAD_ALL_COLUMNS);
  }
  return true;
}

std::string GetValueOfVariableTableLoaderMode(const VariableState &var_stat) {
  std::ignore = var_stat;

  return util::getName(RuntimeConfiguration::instance().getTableLoaderMode());
}

bool CheckValueOfVariableResultOutputFormat(const std::string &val) {
  if (val == "table" || val == "csv")
    return true;
  else
    return false;
}

// Returns human readable time and according unit string
std::pair<double, std::string> _getTimeAsHumanReadableString(
    Timestamp time_in_nanoseconds) {
  int unit = 0;
  double time = time_in_nanoseconds;
  const string units[6] = {"ns", "us", "ms", "s", "min", "h"};
  while (time >= 1000 && unit < 3) {
    time /= 1000;
    unit++;
  }
  while (time >= 60 && unit >= 3 && unit < 5) {
    time /= 60;
    unit++;
  }
  return std::make_pair(time, units[unit]);
}

#ifdef BAM_FOUND

bool gather_join_no_filtering(ClientPtr client, PositionListPtr &sb_tids_gather,
                              PositionListPtr &rb_tids_gather,
                              PositionListPtr &c_tids_gather,
                              PositionListPtr &r_tids_gather) {
  std::ostream &out = client->getOutputStream();
  out << "Gather join performance test:" << endl;
  out << "JOIN: sample_base join reference_base on sb_rb_id = rb_id join "
         "contig on rb_c_id = c_id join read on sb_read_id = r_id"
      << endl;
  ProcessorSpecification processorSpecification(hype::PD0);

  TablePtr sb_tbl = getTablebyName(SB_TBL_NAME);
  ColumnPtr sb_id_col = sb_tbl->getColumnbyName(SB_ID_COL_NAME);
  ColumnPtr sb_rb_id_col = sb_tbl->getColumnbyName(SB_RB_ID_COL_NAME);
  ColumnPtr sb_read_id_col = sb_tbl->getColumnbyName(SB_READ_ID_COL_NAME);

  TablePtr rb_tbl = getTablebyName(RB_TBL_NAME);
  ColumnPtr rb_id_col = rb_tbl->getColumnbyName(RB_ID_COL_NAME);
  ColumnPtr rb_c_id_col = rb_tbl->getColumnbyName(RB_CONTIG_ID_COL_NAME);

  TablePtr c_tbl = getTablebyName(C_TBL_NAME);
  ColumnPtr c_id_col = c_tbl->getColumnbyName(C_ID_COL_NAME);

  TablePtr r_tbl = getTablebyName(R_TBL_NAME);
  ColumnPtr r_id_col = r_tbl->getColumnbyName(R_ID_COL_NAME);

  Timestamp start_gather_join = getTimestamp();

  boost::shared_ptr<ColumnBaseTyped<TID> > sb_id_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(sb_id_col);
  assert(sb_id_col_typed != NULL);

  boost::shared_ptr<ColumnBaseTyped<TID> > sb_read_id_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(sb_read_id_col);
  assert(sb_read_id_col_typed != NULL);

  boost::shared_ptr<ColumnBaseTyped<TID> > sb_rb_id_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(sb_rb_id_col);
  assert(sb_rb_id_col_typed != NULL);

  boost::shared_ptr<ColumnBaseTyped<TID> > rb_c_id_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(rb_c_id_col);
  assert(rb_c_id_col_typed != NULL);

  // get sb tids
  if (sb_id_col_typed->getColumnType() == PLAIN_MATERIALIZED) {
    sb_tids_gather = boost::dynamic_pointer_cast<PositionList>(sb_id_col_typed);
  } else {
    sb_tids_gather =
        sb_id_col_typed->copyIntoDenseValueColumn(processorSpecification);
  }
  assert(sb_tids_gather != NULL);

  // get rb tids
  if (sb_rb_id_col_typed->getColumnType() == PLAIN_MATERIALIZED) {
    rb_tids_gather =
        boost::dynamic_pointer_cast<PositionList>(sb_rb_id_col_typed);
  } else {
    rb_tids_gather =
        sb_rb_id_col_typed->copyIntoDenseValueColumn(processorSpecification);
  }
  assert(rb_tids_gather != NULL);

  // get r tids
  if (sb_read_id_col_typed->getColumnType() == PLAIN_MATERIALIZED) {
    r_tids_gather =
        boost::dynamic_pointer_cast<PositionList>(sb_read_id_col_typed);
  } else {
    r_tids_gather =
        sb_read_id_col_typed->copyIntoDenseValueColumn(processorSpecification);
  }
  assert(r_tids_gather != NULL);

  // get c tids
  ColumnPtr sb_rb_c_col =
      rb_c_id_col_typed->gather(rb_tids_gather, processorSpecification);
  boost::shared_ptr<ColumnBaseTyped<TID> > sb_rb_c_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(sb_rb_c_col);
  assert(sb_rb_c_col_typed != NULL);
  if (sb_rb_c_col_typed->getColumnType() == PLAIN_MATERIALIZED) {
    c_tids_gather =
        boost::dynamic_pointer_cast<PositionList>(sb_rb_c_col_typed);
  } else {
    c_tids_gather =
        sb_rb_c_col_typed->copyIntoDenseValueColumn(processorSpecification);
  }
  Timestamp end_gather_join = getTimestamp();

  std::pair<double, string> time_gather_join =
      _getTimeAsHumanReadableString(end_gather_join - start_gather_join);
  out << "Time for gather join: " << time_gather_join.first
      << time_gather_join.second << " computing " << sb_tids_gather->size()
      << " rows." << endl;
  return true;
}

bool hash_join_no_filtering(ClientPtr client, PositionListPtr &sb_tids_hash,
                            PositionListPtr &rb_tids_hash,
                            PositionListPtr &c_tids_hash,
                            PositionListPtr &r_tids_hash) {
  std::ostream &out = client->getOutputStream();
  out << "Hash join performance test:" << endl;
  out << "JOIN: sample_base join reference_base on sb_rb_id = rb_id join "
         "contig on rb_c_id = c_id join read on sb_read_id = r_id"
      << endl;
  ProcessorSpecification processorSpecification(hype::PD0);

  TablePtr sb_tbl = getTablebyName(SB_TBL_NAME);
  ColumnPtr sb_id_col = sb_tbl->getColumnbyName(SB_ID_COL_NAME);
  ColumnPtr sb_rb_id_col = sb_tbl->getColumnbyName(SB_RB_ID_COL_NAME);
  ColumnPtr sb_read_id_col = sb_tbl->getColumnbyName(SB_READ_ID_COL_NAME);

  TablePtr rb_tbl = getTablebyName(RB_TBL_NAME);
  ColumnPtr rb_id_col = rb_tbl->getColumnbyName(RB_ID_COL_NAME);
  ColumnPtr rb_c_id_col = rb_tbl->getColumnbyName(RB_CONTIG_ID_COL_NAME);

  TablePtr c_tbl = getTablebyName(C_TBL_NAME);
  ColumnPtr c_id_col = c_tbl->getColumnbyName(C_ID_COL_NAME);

  TablePtr r_tbl = getTablebyName(R_TBL_NAME);
  ColumnPtr r_id_col = r_tbl->getColumnbyName(R_ID_COL_NAME);

  Timestamp start_hash_join = getTimestamp();
  // compute matching TIDs
  PositionListPairPtr hash_join_rb_and_c_pairs =
      c_id_col->join(rb_c_id_col, JoinParam(processorSpecification, HASH_JOIN));

  ColumnPtr joined_rb_ids = rb_id_col->gather(hash_join_rb_and_c_pairs->second,
                                              processorSpecification);
  PositionListPairPtr hash_join_sb_and_rb_pairs = joined_rb_ids->join(
      sb_rb_id_col, JoinParam(processorSpecification, HASH_JOIN));
  sb_tids_hash = hash_join_sb_and_rb_pairs->second;
  ColumnPtr tmp_ptr = hash_join_rb_and_c_pairs->second->gather(
      hash_join_sb_and_rb_pairs->first, processorSpecification);
  rb_tids_hash = boost::dynamic_pointer_cast<PositionList>(tmp_ptr);
  assert(rb_tids_hash != NULL);
  tmp_ptr = hash_join_rb_and_c_pairs->first->gather(
      hash_join_sb_and_rb_pairs->first, processorSpecification);
  c_tids_hash = boost::dynamic_pointer_cast<PositionList>(tmp_ptr);
  assert(c_tids_hash != NULL);

  ColumnPtr joined_sb_read_ids = sb_read_id_col->gather(
      hash_join_sb_and_rb_pairs->second, processorSpecification);
  PositionListPairPtr hash_join_sb_and_read_pairs = r_id_col->join(
      joined_sb_read_ids, JoinParam(processorSpecification, HASH_JOIN));
  r_tids_hash = hash_join_sb_and_read_pairs->first;

  // compute result
  Timestamp end_hash_join = getTimestamp();

  std::pair<double, string> time_hash_join =
      _getTimeAsHumanReadableString(end_hash_join - start_hash_join);
  out << fixed << showpoint << setprecision(2);
  out << "Time for hash join: " << time_hash_join.first << time_hash_join.second
      << " computing " << sb_tids_hash->size() << " rows." << endl;

  return true;
}

bool gather_join_no_filtering_cli(ClientPtr client) {
  PositionListPtr sb_tids_gather;
  PositionListPtr rb_tids_gather;
  PositionListPtr c_tids_gather;
  PositionListPtr r_tids_gather;

  return gather_join_no_filtering(client, sb_tids_gather, rb_tids_gather,
                                  c_tids_gather, r_tids_gather);
}

bool hash_join_no_filtering_cli(ClientPtr client) {
  PositionListPtr sb_tids_hash;
  PositionListPtr rb_tids_hash;
  PositionListPtr c_tids_hash;
  PositionListPtr r_tids_hash;

  hash_join_no_filtering(client, sb_tids_hash, rb_tids_hash, c_tids_hash,
                         r_tids_hash);

  return true;
}

bool gather_join_vs_hash_join_no_filtering(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  out << "Gather join vs hash join performance test and sanity checks:" << endl;

  ProcessorSpecification processorSpecification(hype::PD0);
  PositionListPtr sb_tids_hash;
  PositionListPtr rb_tids_hash;
  PositionListPtr c_tids_hash;
  PositionListPtr r_tids_hash;

  PositionListPtr c_tids_gather;
  PositionListPtr sb_tids_gather;
  PositionListPtr rb_tids_gather;
  PositionListPtr r_tids_gather;

  gather_join_no_filtering(client, sb_tids_gather, rb_tids_gather,
                           c_tids_gather, r_tids_gather);
  hash_join_no_filtering(client, sb_tids_hash, rb_tids_hash, c_tids_hash,
                         r_tids_hash);

  // sanity checks
  if ((sb_tids_hash->size() != rb_tids_hash->size()) ||
      (rb_tids_hash->size() != c_tids_hash->size())) {
    out << "ERROR: Hash TID lists have different size!" << endl;
  };
  if ((sb_tids_gather->size() != rb_tids_gather->size()) ||
      (rb_tids_gather->size() != c_tids_gather->size())) {
    out << "ERROR: Gather TID lists have different size!" << endl;
  }
  if ((sb_tids_hash->size() != sb_tids_hash->size())) {
    out << "ERROR: Hash TID lists have different size than Gather TID lists!"
        << endl;
  }
  PositionListPtr sorted_sb_tids_hash =
      sb_tids_hash->sort(SortParam(processorSpecification, ASCENDING, true));
  PositionListPtr sorted_sb_tids_gather =
      sb_tids_gather->sort(SortParam(processorSpecification, ASCENDING, true));
  bool error = false;
  for (TID i = 0; i < sorted_sb_tids_hash->size(); i++) {
    TID sb_tid_hash = (*sb_tids_hash)[(*sorted_sb_tids_hash)[i]];
    TID sb_tid_gather = (*sb_tids_gather)[(*sorted_sb_tids_gather)[i]];
    if (sb_tid_hash != sb_tid_gather) {
      out << "ERROR: Sorted SB TID lists differ at index " << i
          << ". Hash join value: " << sb_tid_hash
          << " | Gather join value: " << sb_tid_gather << endl;
      error = true;
      break;
    }
    TID rb_tid_hash = (*rb_tids_hash)[(*sorted_sb_tids_hash)[i]];
    TID rb_tid_gather = (*rb_tids_gather)[(*sorted_sb_tids_gather)[i]];
    if (rb_tid_hash != rb_tid_gather) {
      out << "ERROR: Sorted RB TID lists differ at index " << i
          << ". Hash join value: " << rb_tid_hash
          << " | Gather join value: " << rb_tid_gather << endl;
      error = true;
      break;
    }
    TID c_tid_hash = (*c_tids_hash)[(*sorted_sb_tids_hash)[i]];
    TID c_tid_gather = (*c_tids_gather)[(*sorted_sb_tids_gather)[i]];
    if (c_tid_hash != c_tid_gather) {
      out << "ERROR: Sorted C TID lists differ at index " << i
          << ". Hash join value: " << c_tid_hash
          << " | Gather join value: " << c_tid_gather << endl;
      error = true;
      break;
    }
    TID r_tid_hash = (*r_tids_hash)[(*sorted_sb_tids_hash)[i]];
    TID r_tid_gather = (*r_tids_gather)[(*sorted_sb_tids_gather)[i]];
    if (r_tid_hash != r_tid_gather) {
      out << "ERROR: Sorted READ TID lists differ at index " << i
          << ". Hash join value: " << r_tid_hash
          << " | Gather join value: " << r_tid_gather << endl;
      error = true;
      break;
    }
  }
  if (!error) {
    out << "SUCCESS: Checks passed!" << endl;
  }
  return !error;
}

bool hash_join_dim_filtering(ClientPtr client, PositionListPtr &sb_tids_hash,
                             PositionListPtr &rb_tids_hash,
                             PositionListPtr &c_tids_hash,
                             PositionListPtr &r_tids_hash) {
  std::ostream &out = client->getOutputStream();
  out << "Hash join performance test:" << endl;
  out << "Query: sample_base join reference_base on sb_rb_id = rb_id join "
         "contig on rb_c_id = c_id join read on sb_read_id = r_id where c_name "
         "= 'contig_7'"
      << endl;
  ProcessorSpecification processorSpecification(hype::PD0);

  TablePtr sb_tbl = getTablebyName(SB_TBL_NAME);
  ColumnPtr sb_id_col = sb_tbl->getColumnbyName(SB_ID_COL_NAME);
  ColumnPtr sb_rb_id_col = sb_tbl->getColumnbyName(SB_RB_ID_COL_NAME);
  ColumnPtr sb_read_id_col = sb_tbl->getColumnbyName(SB_READ_ID_COL_NAME);

  TablePtr rb_tbl = getTablebyName(RB_TBL_NAME);
  ColumnPtr rb_id_col = rb_tbl->getColumnbyName(RB_ID_COL_NAME);
  ColumnPtr rb_c_id_col = rb_tbl->getColumnbyName(RB_CONTIG_ID_COL_NAME);

  TablePtr c_tbl = getTablebyName(C_TBL_NAME);
  ColumnPtr c_id_col = c_tbl->getColumnbyName(C_ID_COL_NAME);
  ColumnPtr c_name_col = c_tbl->getColumnbyName(C_NAME_COL_NAME);

  TablePtr r_tbl = getTablebyName(R_TBL_NAME);
  ColumnPtr r_id_col = r_tbl->getColumnbyName(R_ID_COL_NAME);

  Timestamp start_hash_selection = getTimestamp();
  PositionListPtr selected_c_tids = c_name_col->selection(
      SelectionParam(processorSpecification, ValueConstantPredicate,
                     string("contig_7"), EQUAL));
  Timestamp end_hash_selection = getTimestamp();

  Timestamp start_hash_join = getTimestamp();
  // compute matching TIDs
  ColumnPtr selected_c_ids =
      c_id_col->gather(selected_c_tids, processorSpecification);
  PositionListPairPtr hash_join_rb_and_c_pairs = selected_c_ids->join(
      rb_c_id_col, JoinParam(processorSpecification, HASH_JOIN));

  ColumnPtr joined_rb_ids = rb_id_col->gather(hash_join_rb_and_c_pairs->second,
                                              processorSpecification);
  PositionListPairPtr hash_join_sb_and_rb_pairs = joined_rb_ids->join(
      sb_rb_id_col, JoinParam(processorSpecification, HASH_JOIN));
  sb_tids_hash = hash_join_sb_and_rb_pairs->second;
  ColumnPtr tmp_ptr = hash_join_rb_and_c_pairs->second->gather(
      hash_join_sb_and_rb_pairs->first, processorSpecification);
  rb_tids_hash = boost::dynamic_pointer_cast<PositionList>(tmp_ptr);
  assert(rb_tids_hash != NULL);
  tmp_ptr = hash_join_rb_and_c_pairs->first->gather(
      hash_join_sb_and_rb_pairs->first, processorSpecification);
  PositionListPtr tmp_pos_list =
      boost::dynamic_pointer_cast<PositionList>(tmp_ptr);
  assert(tmp_pos_list != NULL);
  tmp_ptr = selected_c_ids->gather(tmp_pos_list, processorSpecification);
  c_tids_hash = boost::dynamic_pointer_cast<PositionList>(tmp_ptr);
  assert(c_tids_hash != NULL);

  ColumnPtr joined_sb_read_ids = sb_read_id_col->gather(
      hash_join_sb_and_rb_pairs->second, processorSpecification);
  PositionListPairPtr hash_join_sb_and_read_pairs = r_id_col->join(
      joined_sb_read_ids, JoinParam(processorSpecification, HASH_JOIN));
  r_tids_hash = hash_join_sb_and_read_pairs->first;

  Timestamp end_hash_join = getTimestamp();

  std::pair<double, string> time_hash_join =
      _getTimeAsHumanReadableString(end_hash_join - start_hash_join);
  std::pair<double, string> time_hash_selection =
      _getTimeAsHumanReadableString(end_hash_selection - start_hash_selection);
  out << "Time for hash join: " << time_hash_join.first << time_hash_join.second
      << endl;
  out << "Time for hash selection: " << time_hash_selection.first
      << time_hash_selection.second << endl;
  out << " computing " << sb_tids_hash->size() << " rows." << endl;

  return true;
}

bool gather_join_dim_filtering(ClientPtr client,
                               PositionListPtr &sb_tids_gather,
                               PositionListPtr &rb_tids_gather,
                               PositionListPtr &c_tids_gather,
                               PositionListPtr &r_tids_gather) {
  std::ostream &out = client->getOutputStream();
  out << "Gather join performance test:" << endl;
  out << "Query: sample_base join reference_base on sb_rb_id = rb_id join "
         "contig on rb_c_id = c_id join read on sb_read_id = r_id where c_name "
         "= 'contig_7'"
      << endl;
  ProcessorSpecification processorSpecification(hype::PD0);

  TablePtr sb_tbl = getTablebyName(SB_TBL_NAME);
  ColumnPtr sb_id_col = sb_tbl->getColumnbyName(SB_ID_COL_NAME);
  ColumnPtr sb_rb_id_col = sb_tbl->getColumnbyName(SB_RB_ID_COL_NAME);
  ColumnPtr sb_read_id_col = sb_tbl->getColumnbyName(SB_READ_ID_COL_NAME);

  TablePtr rb_tbl = getTablebyName(RB_TBL_NAME);
  ColumnPtr rb_id_col = rb_tbl->getColumnbyName(RB_ID_COL_NAME);
  ColumnPtr rb_c_id_col = rb_tbl->getColumnbyName(RB_CONTIG_ID_COL_NAME);

  TablePtr c_tbl = getTablebyName(C_TBL_NAME);
  ColumnPtr c_id_col = c_tbl->getColumnbyName(C_ID_COL_NAME);
  ColumnPtr c_name_col = c_tbl->getColumnbyName(C_NAME_COL_NAME);

  TablePtr r_tbl = getTablebyName(R_TBL_NAME);
  ColumnPtr r_id_col = r_tbl->getColumnbyName(R_ID_COL_NAME);

  Timestamp start_gather_join = getTimestamp();

  boost::shared_ptr<ColumnBaseTyped<TID> > sb_id_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(sb_id_col);
  assert(sb_id_col_typed != NULL);

  boost::shared_ptr<ColumnBaseTyped<TID> > sb_read_id_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(sb_read_id_col);
  assert(sb_read_id_col_typed != NULL);

  boost::shared_ptr<ColumnBaseTyped<TID> > sb_rb_id_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(sb_rb_id_col);
  assert(sb_rb_id_col_typed != NULL);

  boost::shared_ptr<ColumnBaseTyped<TID> > rb_c_id_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(rb_c_id_col);
  assert(rb_c_id_col_typed != NULL);

  // get sb tids
  if (sb_id_col_typed->getColumnType() == PLAIN_MATERIALIZED) {
    sb_tids_gather = boost::dynamic_pointer_cast<PositionList>(sb_id_col_typed);
  } else {
    sb_tids_gather =
        sb_id_col_typed->copyIntoDenseValueColumn(processorSpecification);
  }
  assert(sb_tids_gather != NULL);

  // get rb tids
  if (sb_rb_id_col_typed->getColumnType() == PLAIN_MATERIALIZED) {
    rb_tids_gather =
        boost::dynamic_pointer_cast<PositionList>(sb_rb_id_col_typed);
  } else {
    rb_tids_gather =
        sb_rb_id_col_typed->copyIntoDenseValueColumn(processorSpecification);
  }
  assert(rb_tids_gather != NULL);

  // get r tids
  if (sb_read_id_col_typed->getColumnType() == PLAIN_MATERIALIZED) {
    r_tids_gather =
        boost::dynamic_pointer_cast<PositionList>(sb_read_id_col_typed);
  } else {
    r_tids_gather =
        sb_read_id_col_typed->copyIntoDenseValueColumn(processorSpecification);
  }
  assert(r_tids_gather != NULL);

  // get c tids
  ColumnPtr sb_rb_c_col =
      rb_c_id_col_typed->gather(rb_tids_gather, processorSpecification);
  boost::shared_ptr<ColumnBaseTyped<TID> > sb_rb_c_col_typed =
      boost::dynamic_pointer_cast<ColumnBaseTyped<TID> >(sb_rb_c_col);
  assert(sb_rb_c_col_typed != NULL);
  if (sb_rb_c_col_typed->getColumnType() == PLAIN_MATERIALIZED) {
    c_tids_gather =
        boost::dynamic_pointer_cast<PositionList>(sb_rb_c_col_typed);
  } else {
    c_tids_gather =
        sb_rb_c_col_typed->copyIntoDenseValueColumn(processorSpecification);
  }
  Timestamp end_gather_join = getTimestamp();

  Timestamp start_gather_selection = getTimestamp();
  ColumnPtr sb_rb_c_name_col =
      c_name_col->gather(c_tids_gather, processorSpecification);
  PositionListPtr selected_tids = sb_rb_c_name_col->selection(
      SelectionParam(processorSpecification, ValueConstantPredicate,
                     string("contig_7"), EQUAL));

  sb_tids_gather = selected_tids;
  rb_tids_gather = boost::dynamic_pointer_cast<PositionList>(
      rb_tids_gather->gather(selected_tids, processorSpecification));
  assert(rb_tids_gather != NULL);
  c_tids_gather = boost::dynamic_pointer_cast<PositionList>(
      c_tids_gather->gather(selected_tids, processorSpecification));
  assert(c_tids_gather != NULL);
  r_tids_gather = boost::dynamic_pointer_cast<PositionList>(
      r_tids_gather->gather(selected_tids, processorSpecification));
  assert(r_tids_gather != NULL);

  Timestamp end_gather_selection = getTimestamp();

  std::pair<double, string> time_gather_join =
      _getTimeAsHumanReadableString(end_gather_join - start_gather_join);
  std::pair<double, string> time_gather_selection =
      _getTimeAsHumanReadableString(end_gather_selection -
                                    start_gather_selection);
  out << "Time for gather join: " << time_gather_join.first
      << time_gather_join.second << endl;
  out << "Time for gather selection: " << time_gather_selection.first
      << time_gather_selection.second << endl;
  out << " computing " << selected_tids->size() << " rows." << endl;
  return true;
}

bool gather_join_dim_filtering_cli(ClientPtr client) {
  PositionListPtr sb_tids_gather;
  PositionListPtr rb_tids_gather;
  PositionListPtr c_tids_gather;
  PositionListPtr r_tids_gather;

  return gather_join_dim_filtering(client, sb_tids_gather, rb_tids_gather,
                                   c_tids_gather, r_tids_gather);
}

bool hash_join_dim_filtering_cli(ClientPtr client) {
  PositionListPtr sb_tids_hash;
  PositionListPtr rb_tids_hash;
  PositionListPtr c_tids_hash;
  PositionListPtr r_tids_hash;

  return hash_join_dim_filtering(client, sb_tids_hash, rb_tids_hash,
                                 c_tids_hash, r_tids_hash);
}

bool gather_join_vs_hash_join_dim_filtering(ClientPtr client) {
  std::ostream &out = client->getOutputStream();
  out << "Gather join vs hash join performance test and sanity checks:" << endl;

  ProcessorSpecification processorSpecification(hype::PD0);
  PositionListPtr sb_tids_hash;
  PositionListPtr rb_tids_hash;
  PositionListPtr c_tids_hash;
  PositionListPtr r_tids_hash;

  PositionListPtr c_tids_gather;
  PositionListPtr sb_tids_gather;
  PositionListPtr rb_tids_gather;
  PositionListPtr r_tids_gather;

  gather_join_dim_filtering(client, sb_tids_gather, rb_tids_gather,
                            c_tids_gather, r_tids_gather);
  hash_join_dim_filtering(client, sb_tids_hash, rb_tids_hash, c_tids_hash,
                          r_tids_hash);

  // sanity checks
  if ((sb_tids_hash->size() != rb_tids_hash->size()) ||
      (rb_tids_hash->size() != c_tids_hash->size())) {
    out << "ERROR: Hash TID lists have different size!" << endl;
  };
  if ((sb_tids_gather->size() != rb_tids_gather->size()) ||
      (rb_tids_gather->size() != c_tids_gather->size())) {
    out << "ERROR: Gather TID lists have different size!" << endl;
  }
  if ((sb_tids_hash->size() != sb_tids_hash->size())) {
    out << "ERROR: Hash TID lists have different size than Gather TID lists!"
        << endl;
  }
  PositionListPtr sorted_sb_tids_hash =
      sb_tids_hash->sort(SortParam(processorSpecification, ASCENDING, true));
  PositionListPtr sorted_sb_tids_gather =
      sb_tids_gather->sort(SortParam(processorSpecification, ASCENDING, true));
  bool error = false;
  for (TID i = 0; i < sorted_sb_tids_hash->size(); i++) {
    TID sb_tid_hash = (*sb_tids_hash)[(*sorted_sb_tids_hash)[i]];
    TID sb_tid_gather = (*sb_tids_gather)[(*sorted_sb_tids_gather)[i]];
    if (sb_tid_hash != sb_tid_gather) {
      out << "ERROR: Sorted SB TID lists differ at index " << i
          << ". Hash join value: " << sb_tid_hash
          << " | Gather join value: " << sb_tid_gather << endl;
      error = true;
      break;
    }
    TID rb_tid_hash = (*rb_tids_hash)[(*sorted_sb_tids_hash)[i]];
    TID rb_tid_gather = (*rb_tids_gather)[(*sorted_sb_tids_gather)[i]];
    if (rb_tid_hash != rb_tid_gather) {
      out << "ERROR: Sorted RB TID lists differ at index " << i
          << ". Hash join value: " << rb_tid_hash
          << " | Gather join value: " << rb_tid_gather << endl;
      error = true;
      break;
    }
    TID c_tid_hash = (*c_tids_hash)[(*sorted_sb_tids_hash)[i]];
    TID c_tid_gather = (*c_tids_gather)[(*sorted_sb_tids_gather)[i]];
    if (c_tid_hash != c_tid_gather) {
      out << "ERROR: Sorted C TID lists differ at index " << i
          << ". Hash join value: " << c_tid_hash
          << " | Gather join value: " << c_tid_gather << endl;
      error = true;
      break;
    }
    TID r_tid_hash = (*r_tids_hash)[(*sorted_sb_tids_hash)[i]];
    TID r_tid_gather = (*r_tids_gather)[(*sorted_sb_tids_gather)[i]];
    if (r_tid_hash != r_tid_gather) {
      out << "ERROR: Sorted READ TID lists differ at index " << i
          << ". Hash join value: " << r_tid_hash
          << " | Gather join value: " << r_tid_gather << endl;
      error = true;
      break;
    }
  }
  if (!error) {
    out << "SUCCESS: Checks passed!" << endl;
  }
  return !error;
}

#endif

#ifdef BAM_FOUND

bool loadgenomedatabase(ClientPtr client) {
  loadTables(client);
  std::ostream &out = client->getOutputStream();
  std::vector<TablePtr> &tables = getGlobalTableList();
  out << "Initializing reference based compressed columns" << endl;
  for (auto i = 0ul; i < tables.size(); i++) {
    if (tables[i]->getName().compare("SAMPLE_BASE") == 0) {
      ColumnPtr column = tables[i]->getColumnbyName("SB_BASE_VALUE");
      assert(column != NULL);
      if (column->getColumnType() == REFERENCE_BASED_COMPRESSED) {
        shared_pointer_namespace::shared_ptr<
            ReferenceBasedCompressedColumn<std::string> >
            base_value_col = shared_pointer_namespace::dynamic_pointer_cast<
                ReferenceBasedCompressedColumn<std::string> >(column);
        assert(base_value_col != NULL);
        base_value_col->initReferenceColumnPointers();
      }
    }
  }
  return true;
}

// genomics extension
static boost::shared_ptr<GenomeDataImporter> importer;

boost::shared_ptr<GenomeDataImporter> _GenomeDataImporterInstance(
    ClientPtr ptr) {
  if (importer) return importer;
  string genome_schema_type =
      VariableManager::instance().getVariableValueString(
          GENOME_SCHEMA_TYPE_PARAMETER);
  bool genome_importer_verbose =
      VariableManager::instance().getVariableValueBoolean(
          GENOME_IMPORTER_VERBOSE_PARAMETER);
  bool genome_importer_compress =
      VariableManager::instance().getVariableValueBoolean(
          GENOME_SCHEMA_COMPRESSION_PARAMETER);
  if (BASE_CENTRIC_SCHEMA_TYPE_PARAMETER_VALUE == genome_schema_type) {
    importer = boost::shared_ptr<GenomeDataImporter>(
        new BaseCentric_SimpleKey_Importer(ptr->getOutputStream(),
                                           genome_importer_compress,
                                           genome_importer_verbose));
  } else if (STORAGE_EXPERIMENTS_SCHEMA_TYPE_PARAMETER_VALUE ==
             genome_schema_type) {
    importer =
        boost::shared_ptr<GenomeDataImporter>(new Storage_Experiments_Importer(
            ptr->getOutputStream(), genome_importer_verbose));
  } else if (SEQUENCE_CENTRIC_SCHEMA_TYPE_PARAMETER_VALUE ==
             genome_schema_type) {
    importer = boost::shared_ptr<GenomeDataImporter>(
        new SequenceCentric_SimpleKey_Importer(ptr->getOutputStream(),
                                               genome_importer_compress,
                                               genome_importer_verbose));
  } else if (SEQUENCE_CENTRIC_SCHEMA_WITH_STASH_TYPE_PARAMETER_VALUE ==
             genome_schema_type) {
    importer = boost::shared_ptr<GenomeDataImporter>(
        new SequenceCentric_SimpleKey_WithStash_Importer(
            ptr->getOutputStream(), genome_importer_compress,
            genome_importer_verbose));
  }
  return importer;
}

bool importReferenceGenome(const std::string &val, ClientPtr ptr) {
  ostream &out = ptr->getOutputStream();
  // Parse command line arguments
  vector<string> args;
  boost::split(args, val, boost::is_any_of(" "));
  string path_to_fasta_file = args[0];
  if (path_to_fasta_file.empty()) {
    out << "ERROR: No FASTA file specified for import!" << endl;
    return false;
  }
  string reference_genome_name;
  if (!args[1].empty()) {
    // reference name provided
    reference_genome_name = args[1];
  } else {
    // if not extract from file name
    vector<string> path_parts;
    boost::split(path_parts, args[0], boost::is_any_of("/"));
    reference_genome_name = path_parts[path_parts.size() - 1];
  }
  return _GenomeDataImporterInstance(ptr)->importReferenceGenomeData(
      path_to_fasta_file, reference_genome_name);
}

bool importSampleGenome(const std::string &val, ClientPtr ptr) {
  ostream &out = ptr->getOutputStream();

  out << "Importing sample genome data ... " << endl;

  // Parse commandline arguments
  vector<string> args;
  boost::split(args, val, boost::is_any_of(" "));

  if (args.size() < 2) {
    out << "ERROR: Not enough parameters!" << endl;
    return false;
  }

  string path_to_sam_bam_file = args[0];
  if (path_to_sam_bam_file.empty()) {
    out << "ERROR: No SAM/ BAM file specified for import!" << endl;
    return false;
  }

  string reference_genome_name = args[1];
  string sample_genome_name;
  if (args.size() < 3) {
    // reference name not provided -> extract from file name
    vector<string> path_parts;
    boost::split(path_parts, path_to_sam_bam_file, boost::is_any_of("/"));
    sample_genome_name = path_parts[path_parts.size() - 1];
  } else {
    // reference name provided
    sample_genome_name = args[2];
  }

  return _GenomeDataImporterInstance(ptr)->importSampleGenomeData(
      path_to_sam_bam_file, reference_genome_name, sample_genome_name);
}
#endif

bool CheckValueOfVariableQueryExecutionPolicy(const std::string &val) {
  if (val == "compiled" || val == "interpreted")
    return true;
  else
    return false;
}

bool CheckValueOfVariableDefaultCodeGenerator(const std::string &val) {
  if (val == "cpp" || val == "cuda" || val == "c" || val == "multi_staged")
    return true;
  else
    return false;
}

bool CheckValueOfVariablePipelineExecutionStrategy(const std::string &val) {
  if (val == "c" || val == "opencl")
    return true;
  else
    return false;
}

bool CheckValueOfVariableOpenCLDeviceType(const std::string &val) {
  if (val == "cpu" || val == "igpu" || val == "dgpu" || val == "phi") {
    return true;
  } else if (val == "gpu") {
    std::cerr << "HINT: device type 'gpu' was refined to 'dgpu' (dedicated "
                 "gpus) and 'igpu' (integrated gpus)"
              << std::endl;
    return false;
  } else {
    return false;
  }
}

bool CheckValueOfVariableMemoryAccess(const std::string &val) {
  if (val == "sequential" || val == "coalesced")
    return true;
  else
    return false;
}

bool CheckValueOfVariableExecutionStrategy(const std::string &val) {
  if (val == "serial_single_pass" || val == "parallel_three_pass" ||
      "parallel_global_atomic_single_pass" || val == "single_pass_scan")
    return true;
  else
    return false;
}

bool CheckValueOfAggregationExecutionStrategy(const std::string &val) {
  if (val == "single_pass_reduce" || val == "global_reduce_kernel" ||
      val == "multipass")
    return true;
  else
    return false;
}

bool CheckValueOfVariableVariantExplorationMode(const std::string &val) {
  if (val == "no_exploration" || val == "full_exploration" ||
      "feature_wise_exploration" || val == "genetic")
    return true;
  else
    return false;
}

bool CheckValueOfVariableDefaultHashTable(const std::string &val) {
  if (val == "ocl_hash_table") {
    std::cerr
        << "Error: Join Hash Table 'ocl_hash_table' is no longer supported! "
        << "Try 'ocl_linear_probing' or 'ocl_cuckoo' instead!" << std::endl;
    return false;
  }
  if (VariableManager::instance().getVariableValueString(
          "code_gen.exec_strategy") == "opencl") {
    if (val == "ocl_cuckoo" || val == "ocl_cuckoo2hashes" ||
        val == "ocl_seeded_linear_probing" || val == "ocl_linear_probing") {
      return true;
    } else if (val == "cuckoo" || val == "bucketchained" ||
               val == "linear_probing") {
      std::cerr << "Hash table '" << val << "'"
                << " can only be used with the C code generator! "
                << "Try 'set code_gen.exec_strategy=c' and try again!"
                << std::endl;
      return false;
    } else {
      return false;
    }
  } else if (VariableManager::instance().getVariableValueString(
                 "code_gen.exec_strategy") == "c" ||
             VariableManager::instance().getVariableValueString(
                 "default_code_generator") == "c") {
    if (val == "ocl_cuckoo" || val == "ocl_cuckoo2hashes" ||
        val == "ocl_seeded_linear_probing" || val == "ocl_linear_probing") {
      std::cerr << "Hash table '" << val << "'"
                << " can only be used with the OpenCL code generator! "
                << "Try 'set code_gen.exec_strategy=opencl' and try again!"
                << std::endl;
      return false;
    } else if (val == "cuckoo" || val == "bucketchained" ||
               val == "linear_probing") {
      return true;
    } else {
      return false;
    }
  } else {
    std::cerr << "Cannot determine validity of variable value: '" << val << "'"
              << std::endl;
    return false;
  }
}

bool CheckValueOfVariableOCLGroupedAggregationHashTable(
    const std::string &val) {
  if (val == "linear_probing" || val == "quadratic_probing" ||
      val == "cuckoo_hashing") {
    return true;
  } else {
    return false;
  }
}
bool CheckValueOfVariableOCLGroupedAggregationHashFunc(const std::string &val) {
  if (val == "multiply_shift" || val == "murmur3") {
    return true;
  } else {
    return false;
  }
}

bool CheckValueOfVariableOCLGroupedAggregationStrategy(const std::string &val) {
  if (val == "sequential" || val == "semaphore" || val == "atomic" ||
      val == "atomic_workgroup" || val == "reduce_atomics" ||
      val == "multipass" || val == "fused_scan_global_group") {
    return true;
  } else {
    return false;
  }
}

bool CheckValueOfVariableCommandQueueStrategy(const std::string &val) {
  if (val == "multiple" || val == "subdevices" || val == "outoforder") {
    return true;
  } else {
    return false;
  }
}

bool CheckValueOfVariableBenchExplorationQueryType(const std::string &val) {
  if (val == "projection" || val == "aggregation" ||
      val == "grouped_aggregation" || val == "grouped_aggregation_with_join" ||
      val == "aggregation_with_join") {
    return true;
  } else {
    return false;
  }
}

#ifdef PERSEUS_FOUND
bool CheckValueOfVariablePerseusUpdateStrategy(const std::string &value) {
  return value == "full-pool-baseline" || value == "vw-greedy-baseline" ||
         value == "greedy" || value == "genetic" || value == "iterative" ||
         value == "markov";
}
#endif

CommandLineInterpreter::CommandLineInterpreter(ClientPtr client)
    : simple_command_map_(),
      query_command_map_(),
      command_map_(),
      prompt_("CoGaDB>"),
      global_logfile_mutex_(),
      client_(client) {
  query_command_map_.insert(std::make_pair("loaddatabase", &loadTables));
  query_command_map_.insert(std::make_pair("unloaddatabase", &unloadTables));

  query_command_map_.insert(
      std::make_pair("unittests", &CoGaDB::unit_tests::executeUnitTests));
  query_command_map_.insert(
      std::make_pair("printschema", &printDatabaseSchema));
  query_command_map_.insert(
      std::make_pair("databasestatus", &printDatabaseStatus));
  query_command_map_.insert(
      std::make_pair("printdatabasesize", &printDatabaseSize));
  query_command_map_.insert(std::make_pair("help", &help));
  query_command_map_.insert(std::make_pair("about", &about));
  query_command_map_.insert(std::make_pair("version", &version));
  // simple_command_map_.insert(std::make_pair("simple_ssb_queries",
  // &simple_SSB_Queries));
  query_command_map_.insert(
      std::make_pair("showgpucache", &printStatusOfGPUCache));
  query_command_map_.insert(std::make_pair("showcaches", &printStatusOfCaches));
  query_command_map_.insert(std::make_pair("toggleQC", &toggleQueryChopping));
  query_command_map_.insert(std::make_pair("hypestatus", &printHypeStatus));
  query_command_map_.insert(std::make_pair(
      "integrityconstraints", &printSystemTableIntegrityConstraints));
  query_command_map_.insert(
      std::make_pair("joinindexes", &printSystemTableJoinIndexes));
  query_command_map_.insert(
      std::make_pair("loadjoinindexes", &loadJoinIndexes));
#ifdef ENABLE_GPU_ACCELERATION
  query_command_map_.insert(
      std::make_pair("placejoinindexes", &placeSelectedJoinIndexesOnGPU));
  query_command_map_.insert(
      std::make_pair("placecolumns", &placeSelectedColumnsOnGPU));
  query_command_map_.insert(std::make_pair("list_gpus", &printAvailableGPU));
#endif
  //        query_command_map_.insert(std::make_pair("placecolumnsfrequencybased",
  //        &placeMostFrequentlyUsedColumns));

  query_command_map_.insert(
      std::make_pair("printstatistics", &printCollectedStatistics));
  query_command_map_.insert(
      std::make_pair("printmemoryusage", &printMainMemoryUsage));
  query_command_map_.insert(
      std::make_pair("print_in_memory_columns", &printInMemoryColumns));
  query_command_map_.insert(
      std::make_pair("print_memory_footprint_of_in_memory_columns",
                     &printFootprintOfInMemoryColumns));
  query_command_map_.insert(
      std::make_pair("print_column_statistics_of_in_memory_columns",
                     &printStatisticsOfInMemoryColumns));
  query_command_map_.insert(
      std::make_pair("resetstatistics", &resetCollectedStatistics));
  query_command_map_.insert(
      std::make_pair("starttimer", &startWorkloadExecutionTimer));
  query_command_map_.insert(
      std::make_pair("stoptimer", &stopWorkloadExecutionTimer));
  query_command_map_.insert(std::make_pair(
      "create_denormalized_ssb_database",
      &Unittest_Create_Denormalized_Star_Schema_Benchmark_Database));
  query_command_map_.insert(std::make_pair("history", &printHistory));
  query_command_map_.insert(
      std::make_pair("createlargetable", &createVeryLargeTable));
  query_command_map_.insert(
      std::make_pair("testhashtable", &testHashTablePerformance));

  query_command_map_.insert(std::make_pair("ssb11", &SSB_Q11));
  query_command_map_.insert(std::make_pair("ssb12", &SSB_Q12));
  query_command_map_.insert(std::make_pair("ssb13", &SSB_Q13));
  query_command_map_.insert(std::make_pair("ssb21", &SSB_Q21));
  query_command_map_.insert(std::make_pair("ssb22", &SSB_Q22));
  query_command_map_.insert(std::make_pair("ssb23", &SSB_Q23));
  query_command_map_.insert(std::make_pair("ssb31", &SSB_Q31));
  query_command_map_.insert(std::make_pair("ssb32", &SSB_Q32));
  query_command_map_.insert(std::make_pair("ssb33", &SSB_Q33));
  query_command_map_.insert(std::make_pair("ssb34", &SSB_Q34));
  query_command_map_.insert(std::make_pair("ssb41", &SSB_Q41));
  query_command_map_.insert(std::make_pair("ssb42", &SSB_Q42));
  query_command_map_.insert(std::make_pair("ssb43", &SSB_Q43));

  query_command_map_.insert(std::make_pair("ssb_select", &SSB_Selection_Query));
  query_command_map_.insert(
      std::make_pair("ssb_semi_join", &SSB_SemiJoin_Query));

  // queries for revision experiments
  query_command_map_.insert(std::make_pair("qc_tpch1", &QC_TPCH_Q1));
  query_command_map_.insert(std::make_pair("qc_tpch5_join", &QC_TPCH_Q5));
  query_command_map_.insert(std::make_pair("qc_tpch9", &QC_TPCH_Q9));
  query_command_map_.insert(std::make_pair("qc_tpch13", &QC_TPCH_Q13));
  query_command_map_.insert(std::make_pair("qc_tpch17", &QC_TPCH_Q17));
  query_command_map_.insert(std::make_pair("qc_tpch18", &QC_TPCH_Q18));
  query_command_map_.insert(std::make_pair("qc_tpch19", &QC_TPCH_Q19));
  query_command_map_.insert(std::make_pair("qc_tpch21", &QC_TPCH_Q21));

  query_command_map_.insert(std::make_pair("tpch1", &TPCH_Q1));
  query_command_map_.insert(std::make_pair("tpch2", &TPCH_Q2));
  query_command_map_.insert(std::make_pair("tpch3", &TPCH_Q3));
  query_command_map_.insert(std::make_pair("tpch4", &TPCH_Q4));
  query_command_map_.insert(std::make_pair("tpch5", &TPCH_Q5));
  query_command_map_.insert(std::make_pair("tpch6", &TPCH_Q6));
  query_command_map_.insert(std::make_pair("tpch7", &TPCH_Q7));
  query_command_map_.insert(std::make_pair("tpch9", &TPCH_Q9));
  query_command_map_.insert(std::make_pair("tpch01", &TPCH_Q1));
  query_command_map_.insert(std::make_pair("tpch02", &TPCH_Q2));
  query_command_map_.insert(std::make_pair("tpch03", &TPCH_Q3));
  query_command_map_.insert(std::make_pair("tpch04", &TPCH_Q4));
  query_command_map_.insert(std::make_pair("tpch05", &TPCH_Q5));
  query_command_map_.insert(std::make_pair("tpch06", &TPCH_Q6));
  query_command_map_.insert(std::make_pair("tpch07", &TPCH_Q7));
  query_command_map_.insert(std::make_pair("tpch09", &TPCH_Q9));
  query_command_map_.insert(std::make_pair("tpch10", &TPCH_Q10));
  query_command_map_.insert(std::make_pair("tpch15", &TPCH_Q15));
  //        query_command_map_.insert(std::make_pair("tpch17", &TPCH_Q17));
  query_command_map_.insert(std::make_pair("tpch18", &TPCH_Q18));
  query_command_map_.insert(std::make_pair("tpch20", &TPCH_Q20));
  //        query_command_map_.insert(std::make_pair("tpch21", &TPCH_Q21));
  query_command_map_.insert(std::make_pair(
      "tpch1_hand_compiled", &TPCH_Q1_hand_compiled_cpu_single_threaded));

  command_map_.insert(std::make_pair("exec", &SQL::commandlineExec));
  // command_map_.insert(std::make_pair("exec_cmdline", &execShellCommand));
  command_map_.insert(
      std::make_pair("graphical_explain", &SQL::commandlineExplain));
  command_map_.insert(std::make_pair(
      "explain", &SQL::commandlineExplainStatementsWithOptimization));
  command_map_.insert(
      std::make_pair("explain_unoptimized",
                     &SQL::commandlineExplainStatementsWithoutOptimization));
  command_map_.insert(
      std::make_pair("analyze_table", &computeStatisticsOnTable));
  command_map_.insert(std::make_pair("add_primary_key_constraint",
                                     &addPrimaryKeyConstraintToTable));
  command_map_.insert(std::make_pair("add_foreign_key_constraint",
                                     &addForeignKeyConstraintToTable));
  command_map_.insert(std::make_pair("import_csv_file", &importCSVIntoTable));
  command_map_.insert(std::make_pair("export_csv_file", &exportTableIntoCSV));

  command_map_.insert(std::make_pair("compile_and_execute_query",
                                     &compile_and_execute_cpp_query));
  command_map_.insert(std::make_pair("execute_query_from_json",
                                     &load_and_execute_query_from_JSON));
  command_map_.insert(
      std::make_pair("execute_cogascript", &include_cogascript_file));
  command_map_.insert(std::make_pair("rename_table", &rename_table));
  command_map_.insert(std::make_pair("drop_table", &drop_table));

  query_command_map_.insert(std::make_pair("dssb11", &Denormalized_SSB_Q11));
  query_command_map_.insert(std::make_pair("dssb12", &Denormalized_SSB_Q12));
  query_command_map_.insert(std::make_pair("dssb13", &Denormalized_SSB_Q13));
  query_command_map_.insert(std::make_pair("dssb21", &Denormalized_SSB_Q21));
  query_command_map_.insert(std::make_pair("dssb22", &Denormalized_SSB_Q22));
  query_command_map_.insert(std::make_pair("dssb23", &Denormalized_SSB_Q23));
  query_command_map_.insert(std::make_pair("dssb31", &Denormalized_SSB_Q31));
  query_command_map_.insert(std::make_pair("dssb32", &Denormalized_SSB_Q32));
  query_command_map_.insert(std::make_pair("dssb33", &Denormalized_SSB_Q33));
  query_command_map_.insert(std::make_pair("dssb34", &Denormalized_SSB_Q34));
  query_command_map_.insert(std::make_pair("dssb41", &Denormalized_SSB_Q41));
  query_command_map_.insert(std::make_pair("dssb42", &Denormalized_SSB_Q42));
  query_command_map_.insert(std::make_pair("dssb43", &Denormalized_SSB_Q43));

  query_command_map_.insert(std::make_pair("ssb_like_test", &like_test_query));

#ifdef ENABLE_GPU_ACCELERATION
  // hand compiled kernels for tpch queries
  query_command_map_.insert(
      std::make_pair("tpch6_hand_compiled", &tpch6_hand_compiled));  // cpu
  query_command_map_.insert(std::make_pair("tpch6_hand_compiled_kernel",
                                           &tpch6_hand_compiled_kernel));
  query_command_map_.insert(
      std::make_pair("tpch6_hand_compiled_holistic_kernel",
                     &tpch6_hand_compiled_holistic_kernel));
  // query_command_map_.insert(std::make_pair("tpch4_holistic_kernel",
  // &tpch4_holistic_kernel));
  // query_command_map_.insert(std::make_pair("ssb4_holistic_kernel",
  // &ssb4_hand_compiled_holistic_kernel));
  query_command_map_.insert(
      std::make_pair("tpch3_holistic_kernel", &tpch3_holistic_kernel));
// query_command_map_.insert(std::make_pair("tpch5_holistic_kernel",
// &tpch5_holistic_kernel));
#endif

  // test commands, not an official part of the commandline interface!
  simple_command_map_.insert(std::make_pair(
      "join_performance_test", &CoGaDB::unit_tests::cdk_join_performance_test));
  simple_command_map_.insert(std::make_pair(
      "bitmap_selection_test", &CoGaDB::unit_tests::cdk_selection_bitmap_test));
  simple_command_map_.insert(
      std::make_pair("selection_performance_test",
                     &CoGaDB::unit_tests::cdk_selection_performance_test));
  simple_command_map_.insert(
      std::make_pair("unrolling_performance_test",
                     &CoGaDB::unit_tests::cdk_unrolling_performance_test));
  simple_command_map_.insert(std::make_pair(
      "selection_performance_test_float",
      &CoGaDB::unit_tests::cdk_selection_performance_test_float));
  simple_command_map_.insert(std::make_pair(
      "unrolling_performance_test_float",
      &CoGaDB::unit_tests::cdk_unrolling_performance_test_float));
  simple_command_map_.insert(std::make_pair(
      "bitmap_test",
      &CoGaDB::unit_tests::cdk_selection_bitmap_performance_test));

  simple_command_map_.insert(std::make_pair(
      "main_memory_join_test", &CoGaDB::unit_tests::main_memory_join_tests));

  simple_command_map_.insert(std::make_pair(
      "invisible_join_test", &CoGaDB::unit_tests::cdk_invisiblejoin_test));
  simple_command_map_.insert(std::make_pair(
      "histogram_test", &CoGaDB::unit_tests::equi_width_histogram_test));
  simple_command_map_.insert(
      std::make_pair("histogram_range_test",
                     &CoGaDB::unit_tests::equi_width_histogram_range_test));
  simple_command_map_.insert(std::make_pair("test_rle_column", &testRLEColumn));

#ifdef ENABLE_GPU_ACCELERATION
  // add commands to test kernels from gpu_work
  simple_command_map_.insert(std::make_pair(
      "gpu_work_test", &CoGaDB::unit_tests::gpu_work_kernels_test));
  simple_command_map_.insert(
      std::make_pair("testAllocationAndTransfer",
                     &CoGaDB::unit_tests::testAllocationAndTransfer));
  simple_command_map_.insert(std::make_pair(
      "testBinningKernel", &CoGaDB::unit_tests::testBinningKernel));
  simple_command_map_.insert(
      std::make_pair("experiment1", &CoGaDB::unit_tests::experiment1));
  simple_command_map_.insert(
      std::make_pair("experiment2", &CoGaDB::unit_tests::experiment2));
  simple_command_map_.insert(
      std::make_pair("experiment3", &CoGaDB::unit_tests::experiment3));
  simple_command_map_.insert(
      std::make_pair("experiment4", &CoGaDB::unit_tests::experiment4));
  simple_command_map_.insert(
      std::make_pair("experiment5", &CoGaDB::unit_tests::experiment5));
  simple_command_map_.insert(
      std::make_pair("experiment6", &CoGaDB::unit_tests::experiment6));
#endif

  command_map_.insert(std::make_pair("set", &setVariable));
  command_map_.insert(std::make_pair("set-estimate", &setEstimate));
  command_map_.insert(
      std::make_pair("create_tpch_database", &Unittest_Create_TPCH_Database));
  command_map_.insert(std::make_pair(
      "create_ssb_database", &Unittest_Create_Star_Schema_Benchmark_Database));
  // command_map_.insert(std::make_pair("create_denormalized_ssb_database",
  // &Unittest_Create_Denormalized_Star_Schema_Benchmark_Database));
  command_map_.insert(std::make_pair("print", &printVariableValue));
  command_map_.insert(std::make_pair("setdevice", &setDefaultDevice));
  command_map_.insert(std::make_pair(
      "setparallelizationmode",
      &setDefaultParallelizationModeTwoPhasePhysicalOptimization));
  command_map_.insert(std::make_pair("set_global_load_adaption_policy",
                                     &setGlobalLoadAdaptionPolicy));
  command_map_.insert(std::make_pair("redirectoutput", &redirectOutputToFile));
  command_map_.insert(
      std::make_pair("dumpestimationerrors", &dumpEstimationErrors));
  command_map_.insert(
      std::make_pair("placecolumn", &CoGaDB::placeSelectedColumn));
  command_map_.insert(std::make_pair("placecolumnfrequencybased",
                                     &placeMostFrequentlyUsedColumns));

#ifdef BAM_FOUND
  // Commands for Genome_visualization
  command_map_.insert(
      std::make_pair("export_sample_genome", &exportSampleGenome));
  // TODO Remove debug function
  command_map_.insert(std::make_pair("sam_debug", &debugFunction));
  command_map_.insert(std::make_pair("sam_verificator", &verificateSamFiles));
#endif

  command_map_.insert(std::make_pair("listen", &acceptNetworkConnections));

  static bool initialized_variables = false;

  if (!initialized_variables) {
    initialized_variables = true;
    VariableManager::instance().addVariable("my_string",
                                            VariableState("cheese", VARCHAR));
    VariableManager::instance().addVariable(
        "my_bool", VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "my_float", VariableState("0.3", FLOAT, checkStringIsFloat));
    VariableManager::instance().addVariable(
        "my_int", VariableState("1", INT, checkStringIsInteger));

    VariableManager::instance().addVariable(
        "table_loader_mode",
        VariableState("disk", VARCHAR, CheckValueOfVariableTableLoaderMode,
                      GetValueOfVariableTableLoaderMode,
                      SetValueOfVariableTableLoaderMode));
    VariableManager::instance().addVariable(
        "result_output_format",
        VariableState("table", VARCHAR,
                      CheckValueOfVariableResultOutputFormat));
    VariableManager::instance().addVariable(
        "use_indexed_tuple_reconstruction",
        VariableState("true", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "use_radix_hash_join",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "query_execution_policy",
        VariableState("interpreted", VARCHAR,
                      CheckValueOfVariableQueryExecutionPolicy));
    VariableManager::instance().addVariable("path_to_ssb_sf1_database",
                                            VariableState("", VARCHAR));
    VariableManager::instance().addVariable("path_to_tpch_sf1_database",
                                            VariableState("", VARCHAR));
    VariableManager::instance().addVariable(
        "default_code_generator",
        VariableState("c", VARCHAR, CheckValueOfVariableDefaultCodeGenerator));
    VariableManager::instance().addVariable(
        "default_hash_table",
        VariableState("bucketchained", VARCHAR,
                      CheckValueOfVariableDefaultHashTable));
    VariableManager::instance().addVariable(
        "default_num_hash_tables",
        VariableState("2", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "default_cuckoo_seed",
        VariableState("123456", INT, checkStringIsInteger));

    VariableManager::instance().addVariable(
        "show_generated_code",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "show_generated_kernel_code",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "keep_last_generated_query_code",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "generate_llvm_ir",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "cleanup_generated_files",
        VariableState("true", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "profiling.keep_shared_libraries_loaded",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "debug_code_generator",
        VariableState("false", BOOLEAN, checkStringIsBoolean));

    VariableManager::instance().addVariable(
        "code_gen.exec_strategy",
        VariableState("c", VARCHAR,
                      CheckValueOfVariablePipelineExecutionStrategy));

    VariableManager::instance().addVariable(
        "code_gen.cl_device_type",
        VariableState("cpu", VARCHAR, CheckValueOfVariableOpenCLDeviceType));

    VariableManager::instance().addVariable(
        "code_gen.cl_command_queue_strategy",
        VariableState("multiple", VARCHAR,
                      CheckValueOfVariableCommandQueueStrategy));

    VariableManager::instance().addVariable(
        "code_gen.memory_access",
        VariableState("sequential", VARCHAR, CheckValueOfVariableMemoryAccess));

    VariableManager::instance().addVariable(
        "code_gen.c_compiler_clang_jit",
        VariableState("false", BOOLEAN, checkStringIsBoolean));

    VariableManager::instance().addVariable(
        "code_gen.pipe_exec_strategy",
        VariableState("serial_single_pass", VARCHAR,
                      CheckValueOfVariableExecutionStrategy));
    VariableManager::instance().addVariable(
        "code_gen.aggregation_exec_strategy",
        VariableState("global_reduce_kernel", VARCHAR,
                      CheckValueOfAggregationExecutionStrategy));
    VariableManager::instance().addVariable(
        "code_gen.variant_exploration_mode",
        VariableState("no_exploration", VARCHAR,
                      CheckValueOfVariableVariantExplorationMode));

    VariableManager::instance().addVariable(
        "code_gen.feature_wise_exploration.max_iteration_count",
        VariableState("4", INT, checkStringIsInteger));

    std::stringstream num_threads;
    num_threads << std::thread::hardware_concurrency();
    VariableManager::instance().addVariable(
        "code_gen.num_threads",
        VariableState(num_threads.str(), INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.projection.global_size_multiplier",
        VariableState("1", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.block_size",
        VariableState("10000000", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.enable_caching",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "code_gen.opt.enable_predication",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation_strategy",
        VariableState("semaphore", VARCHAR,
                      CheckValueOfVariableOCLGroupedAggregationStrategy));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation_hashtable",
        VariableState("linear_probing", VARCHAR,
                      CheckValueOfVariableOCLGroupedAggregationHashTable));

    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.hash_function",
        VariableState("multiply_shift", VARCHAR,
                      CheckValueOfVariableOCLGroupedAggregationHashFunc));

    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.hack.enable_manual_ht_size",
        VariableState("false", BOOLEAN, checkStringIsBoolean));

    VariableManager::instance().addVariable(
        "code_gen.opt.hack.ignore_bitpacking_max_bits",
        VariableState("false", BOOLEAN, checkStringIsBoolean));

    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.hack.ht_size",
        VariableState("800000", INT, checkStringIsInteger));

    VariableManager::instance().addVariable(
        "code_gen.cl_properties.use_32bit_atomics", /* if true use 32 bit
                                                       aotmics, if false,
                                                       use 64 bit atomics */
        VariableState("true", BOOLEAN, checkStringIsBoolean));

    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.global_size_multiplier",
        VariableState("1", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.hashtable_size_multiplier",
        VariableState("0.1", FLOAT, checkStringIsFloat));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.atomic.workgroup.local_size",
        VariableState("128", INT, checkStringIsInteger));

    // opencl gpu execution strategies
    VariableManager::instance().addVariable(
        "code_gen.ocl.workgroup_function_implementation",
        VariableState("var1", VARCHAR));
    VariableManager::instance().addVariable(
        "code_gen.ocl.use_builtin_workgroup_functions",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.gpu.reduce_atomics.global_size",
        VariableState("16384", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.gpu.reduce_atomics.local_size",
        VariableState("128", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.gpu.reduce_atomics.enable_buffer",
        VariableState("true", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_grouped_aggregation.gpu.reduce_atomics.values_per_"
        "thread",
        VariableState("1", INT, checkStringIsInteger));

    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_aggregation.gpu.single_pass_reduce.global_size",
        VariableState("16384", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_aggregation.gpu.single_pass_reduce.local_size",
        VariableState("128", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_aggregation.gpu.single_pass_reduce.values_per_thread",
        VariableState("1", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_aggregation.gpu.single_pass_reduce.use_atomics",
        VariableState("false", BOOLEAN, checkStringIsBoolean));

    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_projection.gpu.single_pass_scan.global_size",
        VariableState("16384", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_projection.gpu.single_pass_scan.local_size",
        VariableState("128", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_projection.gpu.single_pass_scan.values_per_thread",
        VariableState("1", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.opt.ocl_projection.gpu.single_pass_scan.use_ocl_workgroup_"
        "scan",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "code_gen.use_tested_hash_constants",
        VariableState("false", BOOLEAN, checkStringIsBoolean));

    VariableManager::instance().addVariable(
        "code_gen.opt.enable_profiling",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "code_gen.opt.profiling_timer_prefix", VariableState("", VARCHAR));
    VariableManager::instance().addVariable(
        "code_gen.insert_artificial_pipeline_breakers",
        VariableState("false", BOOLEAN, checkStringIsBoolean));

    VariableManager::instance().addVariable(
        "code_gen.genetic_exploration.population_size",
        VariableState("10", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "code_gen.genetic_exploration.elitism",
        VariableState("true", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "code_gen.genetic_exploration.max_iteration_count",
        VariableState("300", INT, checkStringIsInteger));

    VariableManager::instance().addVariable(
        "enable_parallel_pipelines",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "show_versions_in_output",
        VariableState("false", BOOLEAN, checkStringIsBoolean));

    VariableManager::instance().addVariable(
        "current_cuda_device_id",
        VariableState("0", INT, checkStringIsInteger));
    VariableManager::instance().addVariable(
        "print_query_result",
        VariableState("true", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        "enable_automatic_data_placement",
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    // make this only true to measure the effects of compile-time operator
    // placement
    // under error conditions. NEVER EVER ENABLE IN ANY OTHER CASE!
    VariableManager::instance().addVariable(
        "unsafe_feature.enable_immediate_selection_abort",
        VariableState("false", BOOLEAN, checkStringIsBoolean));

    // Variables for Sam-Exporter
    VariableManager::instance().addVariable("sam_save_path",
                                            VariableState("/tmp", VARCHAR));
    VariableManager::instance().addVariable("sam_exporter_input_api",
                                            VariableState("simple", VARCHAR));
    VariableManager::instance().addVariable("igv_snapshot_path",
                                            VariableState("/tmp", VARCHAR));
    VariableManager::instance().addVariable(
        "igv_snapshot_name", VariableState("snapshot.jpg", VARCHAR));
    VariableManager::instance().addVariable(
        "igv_port", VariableState("60151", INT, checkStringIsInteger));

    VariableManager::instance().addVariable(
        "bench.exploration.querytype",
        VariableState("projection", VARCHAR,
                      CheckValueOfVariableBenchExplorationQueryType));

    VariableManager::instance().addVariable(
        "bench.exploration.max_pipe_exec_time_in_s",
        VariableState("60", INT, checkStringIsInteger));

#ifdef BAM_FOUND
    // options: 'simple key' or 'weak entities' keys
    VariableManager::instance().addVariable(
        GENOME_SCHEMA_TYPE_PARAMETER,
        VariableState(BASE_CENTRIC_SCHEMA_TYPE_PARAMETER_VALUE, VARCHAR));
    VariableManager::instance().addVariable(
        GENOME_IMPORTER_VERBOSE_PARAMETER,
        VariableState("true", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        GENOME_SCHEMA_COMPRESSION_PARAMETER,
        VariableState("false", BOOLEAN, checkStringIsBoolean));
    VariableManager::instance().addVariable(
        GENOTYPE_FREQUENCY_MIN_PARAMETER,
        VariableState("0.2", FLOAT, checkStringIsFloat));
    VariableManager::instance().addVariable(
        GENOTYPE_FREQUENCY_MAX_PARAMETER,
        VariableState("0.8", FLOAT, checkStringIsFloat));
#endif
  }

#ifdef BAM_FOUND
  // genomics extension
  command_map_.insert(
      std::make_pair("import_reference_genome", &importReferenceGenome));
  command_map_.insert(
      std::make_pair("import_sample_genome", &importSampleGenome));

  query_command_map_.insert(
      std::make_pair("genome_test_query", &Genome_Test_Query));

  query_command_map_.insert(
      std::make_pair("loadgenomedatabase", &loadgenomedatabase));

  query_command_map_.insert(std::make_pair("gather_join_no_filtering",
                                           &gather_join_no_filtering_cli));
  query_command_map_.insert(std::make_pair("gather_join_dim_filtering",
                                           &gather_join_dim_filtering_cli));
  query_command_map_.insert(
      std::make_pair("hash_join_no_filtering", &hash_join_no_filtering_cli));
  query_command_map_.insert(
      std::make_pair("hash_join_dim_filtering", &hash_join_dim_filtering_cli));
  query_command_map_.insert(
      std::make_pair("gather_join_vs_hash_join_no_filtering_checks",
                     &gather_join_vs_hash_join_no_filtering));
  query_command_map_.insert(
      std::make_pair("gather_join_vs_hash_join_dim_filtering_checks",
                     &gather_join_vs_hash_join_dim_filtering));

#endif

#ifdef PERSEUS_FOUND
  VariableManager::instance().addVariable(
      "perseus.update_strategy",
      VariableState("full-pool-baseline", VARCHAR,
                    CheckValueOfVariablePerseusUpdateStrategy));
  VariableManager::instance().addVariable(
      "perseus.elitism", VariableState("2", INT, checkStringIsInteger));
  VariableManager::instance().addVariable(
      "perseus.minimal_pool_size",
      VariableState("8", INT, checkStringIsInteger));
  VariableManager::instance().addVariable(
      "perseus.explore_period",
      VariableState("1024", INT, checkStringIsInteger));
  VariableManager::instance().addVariable(
      "perseus.explore_length", VariableState("2", INT, checkStringIsInteger));
  VariableManager::instance().addVariable(
      "perseus.exploit_period", VariableState("8", INT, checkStringIsInteger));
  VariableManager::instance().addVariable(
      "perseus.skip_length", VariableState("2", INT, checkStringIsInteger));
#endif
}

bool CommandLineInterpreter::parallelExecution(const string &input,
                                               ifstream *in_stream,
                                               ofstream *log_file_for_timings) {
  string input_trimmed = boost::algorithm::trim_copy(input.substr(13));
  unsigned int threads = boost::lexical_cast<unsigned int>(input_trimmed);
  cout << "Used Threads: " << threads << endl;

  string putput;
  while (!in_stream->eof()) {
    std::getline(*in_stream, putput);
    if (putput == "quit" || putput == "exit" || putput == "serial_execution") {
      break;
    }
    {
      boost::lock_guard<boost::mutex> lock(global_command_queue_mutex);
      commandQueue.push(putput);
    }
    cout << "Execute Command " << putput << endl;
  }
  std::vector<ClientPtr> clients;
  for (unsigned int i = 0; i < threads; ++i) {
    std::stringstream ss;
    ss << "parallel_results_user_" << i << ".txt";
    ClientPtr client(new EmulatedClient(ss.str()));
    clients.push_back(client);
  }
  boost::thread_group threadgroup;
  for (unsigned int i = 0; i < threads; ++i) {
    threadgroup.add_thread(new boost::thread(
        boost::bind(&CommandLineInterpreter::threadedExecution, this, i,
                    log_file_for_timings, clients[i])));
  }
  threadgroup.join_all();
  // process serially until we find parallel_execution command
  while (!in_stream->eof()) {
    std::string input;
    std::getline(*in_stream, input);
    // cout << "Execute Command '" << input << "'" << endl;
    if (input.substr(0, 12) ==
        "parallelExec") {  // || input.substr(0, 18) == "parallel_execution") {
      this->parallelExecution(input, in_stream, log_file_for_timings);
      return 0;
    }
    if (input == "quit" || input == "exit") return 0;
    //                if(!cmd.execute(input)){ cout << "Error! Command '" <<
    //                input << "' failed!" << endl; return -1;}
    CoGaDB::Timestamp begin = CoGaDB::getTimestamp();
    bool ret = this->execute(input, this->getClient());
    CoGaDB::Timestamp end = CoGaDB::getTimestamp();
    if (!ret) {
      cout << "Error! Command '" << input << "' failed!" << endl;
      return -1;
    } else {
      cout << input << "\t(" << double(end - begin) / (1000 * 1000) << "ms)"
           << endl;
    }
  }

  return 1;
}

void CommandLineInterpreter::threadedExecution(unsigned int thread_id,
                                               ofstream *log_file_for_timings,
                                               ClientPtr client) {
  assert(log_file_for_timings != NULL);
  string tid =
      boost::lexical_cast<string>(thread_id);  // boost::this_thread::get_id());

  while (true) {  // (!commandQueue.empty()) {

    string command;

    //            {boost::lock_guard<boost::mutex>
    //            lock(global_command_queue_mutex);
    //            if(commandQueue.empty()) break;
    //            command = commandQueue.front();
    //            commandQueue.pop();
    //            }
    command = client->getNextCommand();
    if (command == "TERMINATE") break;
    uint64_t begin = getTimestamp();
    //            cout << "Execute Command " << command << endl;
    ostream &out = client->getOutputStream();
    out << "Execute Command " << command << endl;
    execute(command, client);
    uint64_t end = getTimestamp();

    {
      boost::lock_guard<boost::mutex> lock(global_logfile_mutex_);
      (*log_file_for_timings) << command << "\t"
                              << double(end - begin) / (1000 * 1000) << "ms"
                              << std::endl;
    }

    // threadedExecution();
  }
  // std::cout.flush();
  // filestr.close();
}

bool CommandLineInterpreter::execute(const string &input, ClientPtr client) {
  if (!client) {
    COGADB_FATAL_ERROR("Invalid client! (pointer to client is NULL)", "");
    return false;
  }
  std::ostream &out = client->getOutputStream();
  string input_trimmed = boost::algorithm::trim_copy(input);

  if (input_trimmed.empty()) return true;
  if (input_trimmed[0] == '#') return true;

  size_t command_len = input_trimmed.find_first_of(" \t");
  size_t param_begin = input_trimmed.find_first_not_of(" \t", command_len + 1);

  string command = input_trimmed.substr(0, command_len);
  string parameter = input_trimmed.substr(param_begin);

  QueryCommandMap::iterator query_it = query_command_map_.find(input_trimmed);

  if (query_it != query_command_map_.end()) return query_it->second(client);

  SimpleCommandMap::iterator simple_it =
      simple_command_map_.find(input_trimmed);

  if (simple_it != simple_command_map_.end()) return simple_it->second();

  ParameterizedCommandMap::iterator command_it = command_map_.find(command);
  if (command_it != command_map_.end())
    return command_it->second(parameter, client);

  std::string sql_command = input_trimmed.substr(0, 6);
  boost::to_lower(sql_command);
  if (sql_command == "select" || sql_command == "create" ||
      sql_command == "update" || sql_command == "insert") {
    return SQL::commandlineExec(input_trimmed, client);
  }

  out << "Error! Command '" << command << "' not found!" << endl;

  return false;
}

bool CommandLineInterpreter::executeFromFile(const std::string &filepath,
                                             ClientPtr client) {
  ifstream in_stream;
  std::string input;

  in_stream.open(filepath.c_str());
  if (in_stream.good()) {
    while (!in_stream.eof()) {
      std::getline(in_stream, input);

      cout << "Execute command '" << input << "'" << endl;

      if (!execute(input, client)) {
        cout << "Error! Not all commands in file '" << filepath
             << "' could be executed!" << endl;
        return false;
      }
    }
    return true;
  } else {
    cout << "Error! Could not open file '" << filepath << "' for "
         << "executing commands." << endl;
    return false;
  }
}

#ifdef LIBREADLINE_FOUND

bool CommandLineInterpreter::getline(const string &prompt, string &result) {
  static char *line = NULL;

  if (line) free(line);

  line = readline(prompt.c_str());
  if (!line) return true; /* EOF */

  if (*line) add_history(line);
  result = line;

  return false;
}

#else  /* !LIBREADLINE_FOUND */

bool CommandLineInterpreter::getline(const string &prompt, string &result) {
  cout << prompt;
  std::getline(cin, result);

  return cin.eof();
}
#endif /* !LIBREADLINE_FOUND */

ClientPtr CommandLineInterpreter::getClient() { return this->client_; }

}  // end namespace CogaDB
