
#include <hardware_optimizations/main_memory_joins/hash_joins.hpp>
#include <lookup_table/join_index.hpp>
#include <unittests/unittests.hpp>
#include <util/reduce_by_keys.hpp>
#include <util/tpch_benchmark.hpp>

#include <query_processing/query_processor.hpp>

#include <boost/lexical_cast.hpp>

//#include <boost/math/distributions/beta.hpp>

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>

#include <sys/mman.h>
#include <fstream>
#include <iostream>

#include <utility>

#include <stdint.h>

#undef ENABLE_GPU_ACCELERATION

#if !defined(__clang__) && defined(__GNUG__)
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
//#define __SSE__
//#define __SSE2__
//#define __SSE3__
//#define __SSE4_1__
#define __MXX__

// SSE compiler intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif

// For SSE2:
#ifdef __SSE2__
extern "C" {
#include <emmintrin.h>
#include <mmintrin.h>
}
#endif

// For SSE3:
#ifdef __SSE3__
extern "C" {
#include <immintrin.h>  // (Meta-header, for GCC only)
#include <pmmintrin.h>
}
#endif

// For SSE4: (WITHOUT extern "C")
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

//#include <stdlib.h>     // posix_memalign()

using namespace std;
// using namespace CoGaDB;

namespace CoGaDB {

using namespace query_processing;

namespace unit_tests {

const bool unittest_debug = false;
typedef bool (*UnittestFunctionPtr)();

bool executeUnitTests(ClientPtr client) {
  std::ostream& out = client->getOutputStream();
  out << "Unittests are now seperate executables!" << std::endl;
  return false;

  //	std::vector<pair<std::string,UnittestFunctionPtr> > unittests;
  //        unittests.push_back(std::make_pair("GeneralJoinTest",generalJoinTest));
  //        unittests.push_back(std::make_pair("Filling Table
  //        Unittest",fillTableTest));
  //        unittests.push_back(std::make_pair("renameColumnTest",renameColumnTest));
  //        unittests.push_back(std::make_pair("PrimaryForeignKeyTest",PrimaryForeignKeyTest));
  //        unittests.push_back(std::make_pair("compressedColumnsTest",compressioned_columns_tests));
  //        unittests.push_back(std::make_pair("JoinTest",joinTest));
  //// unittests.push_back(std::make_pair("GeneralJoinTest",generalJoinTest));
  ////
  /// unittests.push_back(std::make_pair("JoinPerformanceTest",JoinPerformanceTest));
  ////
  /////unittests.push_back(std::make_pair("memory_benchmark",memory_benchmark));
  //
  ////	//unittests.push_back(std::make_pair("Update Tuple
  /// Unittest",updateTuplesTest));
  //	//unittests.push_back(std::make_pair("Basic Operation
  // Unittest",basicOperationTest));
  //#ifdef ENABLE_GPU_ACCELERATION
  //	unittests.push_back(std::make_pair("GPU Column Computation
  // Unittest",GPUColumnComputationTest));
  ////	unittests.push_back(std::make_pair("Basic GPU accelerated Query
  /// Test",basicGPUacceleratedQueryTest));
  //#endif
  //	unittests.push_back(std::make_pair("Table-based ColumnConstantOperator
  // Test",TableBasedColumnConstantOperatorTest));
  ////	unittests.push_back(std::make_pair("ComplexSortTest",ComplexSortTest));
  // 	//unittests.push_back(std::make_pair("JoinTest",joinTest));
  //        unittests.push_back(std::make_pair("CrossJoinTest",crossjoinTest));
  //        //unittests.push_back(std::make_pair("ComplexSelectionTest",complexSelectionTest));
  //        //unittests.push_back(std::make_pair("addConstantColumnTest",addConstantColumnTest));
  //        unittests.push_back(std::make_pair("PositionListTe
  //        st",PositionListTest));
  //        unittests.push_back(std::make_pair("CDK_PrimitivesTest",primitives_unittests));
  //        unittests.push_back(std::make_pair("selectionTest",selectionTest));
  //
  //        unittests.push_back(std::make_pair("GPU_accelerated_scansTest",GPU_accelerated_scans));
  //// unittests.push_back(std::make_pair("memory_benchmark",memory_benchmark));
  //        unittests.push_back(std::make_pair("ColumnorientedQueryPlanTest",ColumnorientedQueryPlanTest));
  //
  //
  //
  ////
  /// unittests.push_back(std::make_pair("JoinPerformanceTest",JoinPerformanceTest));
  //
  //	for(unsigned int i=0;i<unittests.size();i++){
  //		cout << "Execute '" << unittests[i].first << "'..."; // << endl;
  //		if(unittests[i].second())
  //			cout << "SUCCESS" << endl;
  //		else{
  //			cout << "FAILED" << endl;
  //			return false;
  //		}
  //	}

  //	cout << "Execute 'Filling Table Unittest'..." << endl;
  //	if(fillTableTest())
  //		cout << "SUCCESS" << endl;
  //	else{
  //		cout << "FAILED" << endl;
  //		return false;
  //	}
  //
  return true;
}

bool joinTestOld() {
  TableSchema schema;
  schema.push_back(Attribut(INT, "SID"));
  schema.push_back(Attribut(VARCHAR, "Studiengang"));

  TablePtr tab1(new Table("Studiengänge", schema));
  if (!tab1) {
    cout << "Failed to create Table 'Studiengänge'!" << endl;
    return false;
  }

  {
    Tuple t;
    t.push_back(1);
    t.push_back(string("INF"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(2);
    t.push_back(string("CV"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(3);
    t.push_back(string("CSE"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(4);
    t.push_back(string("WIF"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(5);
    t.push_back(string("INF Fernst."));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(6);
    t.push_back(string("CV Master"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(7);
    t.push_back(string("INGINF"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(8);
    t.push_back(string("Lehramt"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }

  TableSchema schema2;
  schema2.push_back(Attribut(VARCHAR, "Name"));
  schema2.push_back(Attribut(INT, "MatrikelNr."));
  schema2.push_back(Attribut(INT, "SID"));

  TablePtr tab2(new Table("Studenten", schema2));

  {
    Tuple t;
    t.push_back(string("Tom"));
    t.push_back(15487);
    t.push_back(3);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Jennifer"));
    t.push_back(12341);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Maria"));
    t.push_back(19522);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Johannes"));
    t.push_back(11241);
    t.push_back(2);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Julia"));
    t.push_back(19541);
    t.push_back(7);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Chris"));
    t.push_back(13211);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Matthias"));
    t.push_back(19422);
    t.push_back(2);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Maria"));
    t.push_back(11875);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Maximilian"));
    t.push_back(13487);
    t.push_back(4);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Bob"));
    t.push_back(14267);
    t.push_back(2);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Susanne"));
    t.push_back(16755);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Stefanie"));
    t.push_back(19774);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Johannes"));
    t.push_back(13254);
    t.push_back(2);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }
  {
    Tuple t;
    t.push_back(string("Karl"));
    t.push_back(13324);
    t.push_back(3);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }

  cout << "initial tables: " << endl;
  tab1->print();
  tab2->print();

  JoinAlgorithm join_alg =
      SORT_MERGE_JOIN;  // NESTED_LOOP_JOIN; // HASH_JOIN; //SORT_MERGE_JOIN;
                        // //HASH_JOIN; //NESTED_LOOP_JOIN NESTED_LOOP_JOIN

  cout << endl
       << "Join Table Studenten, Studiengänge where "
          "Studenten.SID=Studiengänge.SID ..."
       << endl;

  hype::ProcessingDeviceID id = hype::PD0;
  ProcessorSpecification proc_spec(id);
  JoinParam param(proc_spec, join_alg);

  TablePtr tab3 = BaseTable::join(tab1, "SID", tab2, "SID", param);

  if (!tab3) {
    cout << "Join Operation Failed! (in MATERIALIZE Mode)" << endl;
    return false;
  }

  tab3->print();
  return true;
}

bool fillTableTest() {
  TableSchema schema;
  schema.push_back(Attribut(INT, "SID"));
  schema.push_back(Attribut(VARCHAR, "Studiengang"));

  TablePtr tab1(new Table("Studiengänge", schema));
  if (!tab1) {
    cout << "Failed to create Table 'Studiengänge'!" << endl;
    return false;
  }

  vector<Tuple> comp_table;  // comparison Table

  {
    Tuple t;
    t.push_back(1);
    t.push_back(string("INF"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table.push_back(t);
  }
  {
    Tuple t;
    t.push_back(2);
    t.push_back(string("CV"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table.push_back(t);
  }
  {
    Tuple t;
    t.push_back(3);
    t.push_back(string("CSE"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table.push_back(t);
  }
  {
    Tuple t;
    t.push_back(4);
    t.push_back(string("WIF"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table.push_back(t);
  }
  {
    Tuple t;
    t.push_back(5);
    t.push_back(string("INF Fernst."));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table.push_back(t);
  }
  {
    Tuple t;
    t.push_back(6);
    t.push_back(string("CV Master"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table.push_back(t);
  }
  {
    Tuple t;
    t.push_back(7);
    t.push_back(string("INGINF"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table.push_back(t);
  }
  {
    Tuple t;
    t.push_back(8);
    t.push_back(string("Lehramt"));
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table.push_back(t);
  }

  if (tab1->getNumberofRows() != comp_table.size()) {
    cout << "Fatal Error! Wrong number of rows in Table 'Studiengänge'!"
         << endl;
    return false;
  }

  //	comp_table.clear();
  //	Tuple t; t.push_back(2); t.push_back(string("INF"));
  // comp_table.push_back(t);

  //	Tuple t=comp_table[0];
  //	Tuple t2=tab1->fetchTuple(0);

  for (unsigned int i = 0; i < comp_table.size(); i++) {
    if (boost::any_cast<int>(comp_table[i][0]) !=
        boost::any_cast<int>(tab1->fetchTuple(i)[0])) {
      cout << "Fatal Error! Wrong content stored in Table 'Studiengänge'!"
           << endl;
      return false;
    }
    if (boost::any_cast<std::string>(comp_table[i][1]) !=
        boost::any_cast<std::string>(tab1->fetchTuple(i)[1])) {
      cout << "Fatal Error! Wrong content stored in Table 'Studiengänge'!"
           << endl;
      return false;
    }
  }

  TableSchema schema2;
  schema2.push_back(Attribut(VARCHAR, "Name"));
  schema2.push_back(Attribut(INT, "MatrikelNr."));
  schema2.push_back(Attribut(INT, "SID"));

  TablePtr tab2(new Table("Studenten", schema2));

  if (!tab2) {
    cout << "Failed to create Table 'Studenten'!" << endl;
    return false;
  }

  vector<Tuple> comp_table2;  // comparison Table

  {
    Tuple t;
    t.push_back(string("Tom"));
    t.push_back(15487);
    t.push_back(3);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Jennifer"));
    t.push_back(12341);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Maria"));
    t.push_back(19522);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Johannes"));
    t.push_back(11241);
    t.push_back(2);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Julia"));
    t.push_back(19541);
    t.push_back(7);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Chris"));
    t.push_back(13211);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Matthias"));
    t.push_back(19422);
    t.push_back(2);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Maria"));
    t.push_back(11875);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Maximilian"));
    t.push_back(13487);
    t.push_back(4);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Bob"));
    t.push_back(14267);
    t.push_back(2);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Susanne"));
    t.push_back(16755);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Stefanie"));
    t.push_back(19774);
    t.push_back(1);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Johannes"));
    t.push_back(13254);
    t.push_back(2);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  {
    Tuple t;
    t.push_back(string("Karl"));
    t.push_back(13324);
    t.push_back(3);
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    comp_table2.push_back(t);
  }
  //{Tuple t; t.push_back(string(""));		t.push_back(); t.push_back();
  // if(!tab2->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}

  for (unsigned int i = 0; i < comp_table2.size(); i++) {
    if (boost::any_cast<std::string>(comp_table2[i][0]) !=
        boost::any_cast<std::string>(tab2->fetchTuple(i)[0])) {
      cout << "Fatal Error! Wrong content stored in Table 'Studiengänge'!"
           << endl;
      return false;
    }
    if (boost::any_cast<int>(comp_table2[i][1]) !=
        boost::any_cast<int>(tab2->fetchTuple(i)[1])) {
      cout << "Fatal Error! Wrong content stored in Table 'Studiengänge'!"
           << endl;
      return false;
    }
    if (boost::any_cast<int>(comp_table2[i][2]) !=
        boost::any_cast<int>(tab2->fetchTuple(i)[2])) {
      cout << "Fatal Error! Wrong content stored in Table 'Studiengänge'!"
           << endl;
      return false;
    }
  }
  if (unittest_debug) {
    cout << "Created Tables: " << endl;
    tab1->print();
    tab2->print();
  }
  return true;
}

/*
*	BEGIN OF JOIN TEST
*	BEGIN OF JOIN TEST
*	BEGIN OF JOIN TEST
*	BEGIN OF JOIN TEST
*	BEGIN OF JOIN TEST
*	BEGIN OF JOIN TEST
*/

bool warmupGPU() {
  TableSchema schema;
  schema.push_back(Attribut(INT, "SID"));
  TablePtr tab1(new Table("Studiengänge", schema));
  if (!tab1) {
    cout << "Failed to create Table 1!" << endl;
    return false;
  }

  TableSchema schema2;
  schema2.push_back(Attribut(INT, "SID"));
  TablePtr tab2(new Table("Studenten", schema2));
  if (!tab2) {
    cout << "Failed to create Table 2!" << endl;
    return false;
  }

  for (int i = 0; i < 100; i++) {
    Tuple t;
    t.push_back(i);
    if (!tab1->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
    if (!tab2->insert(t)) {
      cout << "Failed to insert Tuple" << endl;
      return false;
    }
  }

  //        TablePtr tab3 = BaseTable::join(tab1, "SID", tab2, "SID",
  //        SORT_MERGE_JOIN_2, LOOKUP, GPU);
  //
  //	if(!tab3){
  //		cout << "WARMUP FAILED" << endl;
  //		return false;
  //	}

  // cout << "gpu warmup phase ended" << endl;

  return true;
}

bool validateJoinResult(TablePtr tab, int size, int type) {
  for (int i = 0; i < size; i++) {
    if (type == 0) {
      if (boost::any_cast<int>(tab->fetchTuple(i)[0]) !=
          boost::any_cast<int>(tab->fetchTuple(i)[1]))
        return false;
    } else {
      if (boost::any_cast<float>(tab->fetchTuple(i)[0]) !=
          boost::any_cast<float>(tab->fetchTuple(i)[1]))
        return false;
    }
  }
  return true;
}

pair<pair<TablePtr, TablePtr>, pair<TablePtr, TablePtr> > createJoinTestcase(
    int NUM_primary, int NUM_foreign) {
  std::vector<int> primary_values(NUM_primary);
  std::vector<float> primary_values_float(NUM_primary);

  for (int i = 0; i < NUM_primary; i++) primary_values[i] = i;

  random_shuffle(primary_values.begin(), primary_values.end());
  for (int i = 0; i < NUM_primary; i++)
    primary_values_float[i] = (float)primary_values[i];

  TableSchema schema;
  schema.push_back(Attribut(INT, "SID"));
  TablePtr tab1(new Table("t1", schema));
  if (!tab1) {
    cout << "Failed to create Table 1!" << endl;
  }

  TableSchema schema2;
  schema2.push_back(Attribut(INT, "SID"));
  TablePtr tab2(new Table("t2", schema2));
  if (!tab2) {
    cout << "Failed to create Table 2!" << endl;
  }

  TableSchema schema3;
  schema3.push_back(Attribut(FLOAT, "SID"));
  TablePtr tab3(new Table("t3", schema3));
  if (!tab3) {
    cout << "Failed to create Table 3!" << endl;
  }

  TableSchema schema4;
  schema4.push_back(Attribut(FLOAT, "SID"));
  TablePtr tab4(new Table("t4", schema4));
  if (!tab4) {
    cout << "Failed to create Table 4!" << endl;
  }

  for (int i = 0; i < NUM_primary; i++) {
    Tuple t_1;
    t_1.push_back(primary_values[i]);
    if (!tab1->insert(t_1)) {
      cout << "Failed to insert Tuple into t1" << endl;
    }
    Tuple t_3;
    t_3.push_back(primary_values_float[i]);
    if (!tab3->insert(t_3)) {
      cout << "Failed to insert Tuple into t3" << endl;
    }
  }

  /* initiate tables */
  for (int i = 0; i < NUM_foreign; i++) {
    int randindex = rand() % primary_values.size();

    Tuple t_2;
    t_2.push_back(primary_values[randindex]);
    if (!tab2->insert(t_2)) {
      cout << "Failed to insert Tuple into t2" << endl;
    }
    Tuple t_4;
    t_4.push_back(primary_values_float[randindex]);
    if (!tab4->insert(t_4)) {
      cout << "Failed to insert Tuple into t4" << endl;
    }
  }

  pair<pair<TablePtr, TablePtr>, pair<TablePtr, TablePtr> > tables;

  tables.first.first = tab1;
  tables.first.second = tab2;
  tables.second.first = tab3;
  tables.second.second = tab4;

  return tables;
}

bool executeJoinTest(int testcaseNum, int primaryCount, int foreignCount) {
  ofstream file;
  file.open("eval.txt", std::fstream::app);

  cout << "start test " << testcaseNum << ": " << primaryCount << " | "
       << foreignCount << endl;
  // file << "start test " <<  testcaseNum << ": " << primaryCount << " | " <<
  // foreignCount << endl;

  pair<pair<TablePtr, TablePtr>, pair<TablePtr, TablePtr> > tables =
      createJoinTestcase(primaryCount, foreignCount);

  // warmup GPU before Testing
  warmupGPU();

  //	file << "x " << testcaseNum << "\t" << "\t" << primaryCount << "\t" <<
  // foreignCount << "\t";
  //
  //
  //
  //	for (int i = 1; i <= 1; i++)
  //	{
  //		TablePtr table1 = BaseTable::join(tables.first.first, "SID",
  // tables.first.second, "SID", SORT_MERGE_JOIN, LOOKUP, GPU); // int
  //		//TablePtr table2 = BaseTable::join(tables.second.first, "SID",
  // tables.second.second, "SID", SORT_MERGE_JOIN, LOOKUP, GPU); // float
  //		//TablePtr table3 = BaseTable::join(tables.first.first, "SID",
  // tables.first.second, "SID", HASH_JOIN, LOOKUP, CPU); // int
  //		//TablePtr table4 = BaseTable::join(tables.second.first, "SID",
  // tables.second.second, "SID", HASH_JOIN, LOOKUP, CPU); // float
  //		if(!table1 || !validateJoinResult(table1, foreignCount, 0)){cout
  //<<
  //"GPU Join failed" << endl;return false;}
  //		//if(!table2 || !validateJoinResult(table2, foreignCount,
  // 1)){cout
  //<< "GPU Join failed" << endl;return false;}
  //		//if(!table3 || !validateJoinResult(table3, foreignCount,
  // 0)){cout
  //<< "CPU Join failed" << endl;return false;}
  //		//if(!table4 || !validateJoinResult(table4, foreignCount,
  // 1)){cout
  //<< "CPU Join failed" << endl;return false;}
  //	}
  //
  //	cout << "end test " <<  testcaseNum << ": " << primaryCount << " | " <<
  // foreignCount << endl;
  //	//file << "end test " <<  testcaseNum << ": " << primaryCount << " | "
  //<< foreignCount << endl;
  //
  //	file << endl;
  //
  //	file.close();

  return true;
}

bool GPU_ExperimentjoinTest() {
  mlockall(MCL_CURRENT | MCL_FUTURE);

  ofstream file;
  file.open("eval.txt");
  file << "";

  int maxElem = 10000000;

  // executeJoinTest(1, maxElem, maxElem);
  file.open("eval.txt", std::fstream::app);
  file << "Testcase 1: 1 -> " << maxElem << " (*10) | both" << endl;
  file.close();
  for (int i = 1; i <= maxElem; i *= 10) {
    executeJoinTest(1, i, i);
  }

  file.open("eval.txt", std::fstream::app);
  file << "Testcase 2: 1 -> " << maxElem << " (*10) | fixed primary key size"
       << endl;
  file.close();
  for (int i = 1; i <= maxElem; i *= 10) {
    for (int j = 1000; j <= maxElem; j *= 100) executeJoinTest(2, j, i);
  }

  file.open("eval.txt", std::fstream::app);
  file << "Testcase 3: 1 -> " << maxElem << "  (*10) | fixed foreign key size "
       << endl;
  file.close();
  for (int i = 1; i <= maxElem; i *= 10) {
    for (int j = 1000; j <= maxElem; j *= 100) executeJoinTest(3, i, j);
  }

  munlockall();

  return true;
}

/*
*	END OF JOIN TEST
*	END OF JOIN TEST
*	END OF JOIN TEST
*	END OF JOIN TEST
*	END OF JOIN TEST
*	END OF JOIN TEST
*/

bool updateTuplesTest() { return false; }

bool basicOperationTest() {
  //	TableSchema schema;
  //	schema.push_back(Attribut(INT,"SID"));
  //	schema.push_back(Attribut(VARCHAR,"Studiengang"));
  //
  //	TablePtr tab1(new Table("Studiengänge",schema));
  //	if(!tab1){
  //		cout << "Failed to create Table 'Studiengänge'!" << endl;
  //		return false;
  //	}
  //
  //	{Tuple t; t.push_back(1); t.push_back(string("INF"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //	{Tuple t; t.push_back(2); t.push_back(string("CV"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //	{Tuple t; t.push_back(3); t.push_back(string("CSE"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //	{Tuple t; t.push_back(4); t.push_back(string("WIF"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //	{Tuple t; t.push_back(5); t.push_back(string("INF Fernst."));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //	{Tuple t; t.push_back(6); t.push_back(string("CV Master"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //	{Tuple t; t.push_back(7); t.push_back(string("INGINF"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //	{Tuple t; t.push_back(8); t.push_back(string("Lehramt"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //
  //	TableSchema schema2;
  //	schema2.push_back(Attribut(VARCHAR,"Name"));
  //	schema2.push_back(Attribut(INT,"MatrikelNr."));
  //	schema2.push_back(Attribut(INT,"SID"));
  //
  //	TablePtr tab2(new Table("Studenten",schema2));
  //
  //	{Tuple t; t.push_back(string("Tom"));
  // t.push_back(15487);
  // t.push_back(3); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Jennifer"));	t.push_back(12341);
  // t.push_back(1); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Maria"));		t.push_back(19522);
  // t.push_back(1); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Johannes"));	t.push_back(11241);
  // t.push_back(2); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Julia"));		t.push_back(19541);
  // t.push_back(7); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Chris"));		t.push_back(13211);
  // t.push_back(1); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Matthias"));	t.push_back(19422);
  // t.push_back(2); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Maria"));		t.push_back(11875);
  // t.push_back(1); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Maximilian"));	t.push_back(13487);
  // t.push_back(4); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Bob"));
  // t.push_back(14267);
  // t.push_back(2); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Susanne"));
  // t.push_back(16755);
  // t.push_back(1); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Stefanie"));	t.push_back(19774);
  // t.push_back(1); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Johannes"));	t.push_back(13254);
  // t.push_back(2); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //	{Tuple t; t.push_back(string("Karl"));
  // t.push_back(13324);
  // t.push_back(3); if(!tab2->insert(t)){ cout << "Failed to insert Tuple" <<
  // endl; return false;}}
  //
  //	if(unittest_debug){
  //		cout << "initial tables: " << endl;
  //		tab1->print();
  //		tab2->print();
  //	}
  //
  //	JoinAlgorithm join_alg = SORT_MERGE_JOIN; //HASH_JOIN;
  ////NESTED_LOOP_JOIN; // HASH_JOIN; //SORT_MERGE_JOIN; //HASH_JOIN;
  ////NESTED_LOOP_JOIN NESTED_LOOP_JOIN
  //
  //	{
  //	if(unittest_debug)
  //		cout << endl << "Join Table Studenten, Studiengänge where
  // Studenten.SID=Studiengänge.SID ..." << endl;
  //
  //
  //        hype::ProcessingDeviceID id=hype::PD0;
  //        ProcessorSpecification proc_spec(id);
  //        JoinParam param(proc_spec, HASH_JOIN);
  //
  //	TablePtr tab3 = BaseTable::join(tab1,"SID",tab2,"SID",param);
  //
  //	if(!tab3){
  //		cout << "Join Operation Failed! (in MATERIALIZE Mode)" << endl;
  //		return false;
  //	}
  //
  //	if(unittest_debug){
  //		tab3->print();
  //
  ////	TablePtr
  /// tab3=tab1->join(tab2,"SID","SID",SORT_MERGE_JOIN,MATERIALIZE,CPU);
  ////				tab3->print();
  //
  //		cout << endl << "Projection on Table with Columns Name,
  // Studiengang..." << endl;
  //	}
  //
  //		list<string> columns;
  //		columns.push_back("Name");
  //		columns.push_back("MatrikelNr.");
  //		columns.push_back("Studiengang");
  //		tab3=BaseTable::projection(tab3,columns,MATERIALIZE,CPU);
  //
  //		if(unittest_debug){
  //			tab3->print();
  //			cout << endl << "Selection on Table on column
  // Studiengang
  // equals
  //'INF'..." << endl;
  //		}
  //
  //		//ValueComparator{LESSER,GREATER,EQUAL};
  //		TablePtr
  // tab4=BaseTable::selection(tab3,string("Studiengang"),string("INF"),EQUAL,MATERIALIZE,SERIAL);
  //		if(!tab4){
  //			cout << "Selection Operation Failed! (in MATERIALIZE
  // Mode)"
  //<<
  // endl;
  //			return false;
  //		}
  //
  //		if(unittest_debug){
  //			tab4->print();
  //			cout << endl << "Sorting Table by Name (ascending)..."
  //<<
  // endl;
  //		}
  //
  //		TablePtr
  // tab5=BaseTable::sort(tab4,"Name",ASCENDING,MATERIALIZE,CPU);
  //		if(!tab5){
  //			cout << "Sort Operation Failed! (in MATERIALIZE Mode)"
  //<<
  // endl;
  //			return false;
  //		}
  //
  //		if(unittest_debug){
  //			assert(tab5!=NULL);
  //			tab5->print();
  //			cout <<
  //"=========================================================================================="
  //<< endl;
  //		}
  //	}
  //	if(unittest_debug)
  //		cout << "Query using Lookup Tables..." << endl;
  //
  //	{
  //	if(unittest_debug)
  //		cout << endl << "Join Table Studenten, Studiengänge where
  // Studenten.SID=Studiengänge.SID ..." << endl;
  //
  //        hype::ProcessingDeviceID id=hype::PD0;
  //        ProcessorSpecification proc_spec(id);
  //        JoinParam param(proc_spec, HASH_JOIN);
  //	TablePtr tab3 = BaseTable::join(tab1,"SID",tab2,"SID",param);
  //	if(!tab3){
  //		cout << "Join Operation Failed! (in LOOKUP Mode)" << endl;
  //		return false;
  //	}
  //
  //		if(unittest_debug){
  //			tab3->print();
  //			cout << endl << "Projection on Table with Columns Name,
  // Studiengang..." << endl;
  //		}
  //
  //		list<string> columns;
  //		columns.push_back("Name");
  //		columns.push_back("MatrikelNr.");
  //		columns.push_back("Studiengang");
  //
  //		tab3=BaseTable::projection(tab3,columns,MATERIALIZE,CPU);
  //		if(unittest_debug){
  //			tab3->print();
  //			cout << endl << "Selection on Table on column
  // Studiengang
  // equals
  //'INF'..." << endl;
  //		}
  //
  //		//ValueComparator{LESSER,GREATER,EQUAL};
  //		TablePtr
  // tab4=BaseTable::selection(tab3,string("Studiengang"),string("INF"),EQUAL,LOOKUP,SERIAL);
  //		if(unittest_debug){
  //			tab4->print();
  //			cout << endl << "Sorting Table by Name (ascending)..."
  //<<
  // endl;
  //		}
  //
  //		TablePtr tab5=BaseTable::sort(tab4,"Name",ASCENDING,LOOKUP,CPU);
  //		assert(tab5!=NULL);
  //		if(unittest_debug){
  //			tab5->print();
  //			cout <<
  //"=========================================================================================="
  //<< endl;
  //		}
  //	}

  return true;
}

bool selectionTest() {
  int selection_value;
  ValueComparator selection_comparison_value;  // 0 EQUAL, 1 LESSER, 2 LARGER

  cout << "Enter number of threads:" << endl;
  unsigned int number_of_threads = 4;

  // cin >> number_of_threads;
  number_of_threads = 8;

  boost::mt19937 rng;
  boost::uniform_int<> selection_values(0, 1000);
  boost::uniform_int<> filter_condition(0, 2);

  boost::shared_ptr<Column<int> > column(new Column<int>("Val", INT));
  // fill column
  for (unsigned int i = 0; i < 10 * 1000 * 1000; i++) {
    // for(unsigned int i=0;i<300;i++){
    column->insert(int(rand() % 1000));
  }

  for (unsigned int i = 0; i < 100; i++) {
    selection_value = selection_values(rng);
    selection_comparison_value =
        (ValueComparator)filter_condition(rng);  // rand()%3;

    Timestamp begin;
    Timestamp end;

    PositionListPtr serial_selection_tids;
    PositionListPtr parallel_selection_tids;
    PositionListPtr lock_free_parallel_selection_tids;

    //                            {
    //                            begin=getTimestamp();
    //                            serial_selection_tids =
    //                            column->selection(selection_value,selection_comparison_value);
    //                            end=getTimestamp();
    //                            cout << "Serial Selection: " <<
    //                            double(end-begin)/(1000*1000) << "ms" << endl;
    //                            }
    {
      begin = getTimestamp();
      parallel_selection_tids = column->parallel_selection(
          selection_value, selection_comparison_value, number_of_threads);
      end = getTimestamp();
      cout << "Parallel Selection: " << double(end - begin) / (1000 * 1000)
           << "ms" << endl;
    }
    //                            {
    //                            begin=getTimestamp();
    //                            lock_free_parallel_selection_tids =
    //                            column->lock_free_parallel_selection(selection_value,selection_comparison_value,number_of_threads);
    //                            end=getTimestamp();
    //                            cout << "Lock Free Parallel Selection: " <<
    //                            double(end-begin)/(1000*1000) << "ms" << endl;
    //                            }

    //                            if((*serial_selection_tids)!=(*parallel_selection_tids)){
    //                                cout << "TID lists are not equal!" <<
    //                                endl;
    //                                cout << "Serial Selection result size: "
    //                                << serial_selection_tids->size() << endl;
    //                                cout << "Parallel Selection result size: "
    //                                << parallel_selection_tids->size() <<
    //                                endl;
    //
    //                                unsigned int
    //                                size=std::min(serial_selection_tids->size(),parallel_selection_tids->size());
    ////                                for(unsigned int i=0;i<size;i++){
    ////                                    cout << "Serial id: " <<
    ///(*serial_selection_tids)[i] << " \tParallel id:"<<
    ///(*parallel_selection_tids)[i] << endl;
    ////                                }
    //                                for(unsigned int i=0;i<size;i++){
    //                                    if((*serial_selection_tids)[i]!=(*parallel_selection_tids)[i])
    //                                         cout << "Serial id: " <<
    //                                         (*serial_selection_tids)[i] << "
    //                                         \tParallel id:"<<
    //                                         (*parallel_selection_tids)[i] <<
    //                                         endl;
    //                                }
    //                                if(size<serial_selection_tids->size()){
    //                                    cout << "Detected additional values
    //                                    for serial selection " << endl;
    //                                    for(unsigned int
    //                                    i=size;i<serial_selection_tids->size();i++){
    //                                           cout << "id: " << i << " val: "
    //                                           << (*serial_selection_tids)[i]
    //                                           << endl;
    //                                    }
    //                                }
    //                                COGADB_FATAL_ERROR("Selection Unittests
    //                                failed! At least one algorithm works
    //                                incorrect!","");
    //
    //                            }
    //                            assert((*serial_selection_tids)==(*lock_free_parallel_selection_tids));
  }

  return true;
}

bool renameColumnTest() {
  TableSchema schema;
  for (unsigned int i = 0; i < 10; ++i) {
    std::stringstream ss;
    ss << "COL_" << i;
    schema.push_back(Attribut(INT, ss.str()));
  }

  TablePtr table(new Table("Table", schema));
  if (!quiet && verbose && debug) table->print();

  RenameList rename_list;
  for (unsigned int i = 0; i < 10; i += 2) {
    std::stringstream ss1;
    ss1 << "COL_" << i;
    std::stringstream ss2;
    ss2 << "RENAMED_COL_" << i;
    rename_list.push_back(RenameEntry(ss1.str(), ss2.str()));
  }
  if (!table->renameColumns(rename_list)) {
    COGADB_ERROR("Rename Column Failed!", "");
  }
  if (!quiet && verbose && debug) table->print();
  if (!quiet && verbose && debug) table->printSchema();

  // typedef std::pair<AttributeType,std::string> Attribut;
  // typedef std::list<Attribut> TableSchema;
  TableSchema new_schema = table->getSchema();
  TableSchema::iterator it;
  unsigned int i = 0;
  for (it = new_schema.begin(); it != new_schema.end(); ++it) {
    if (i % 2 == 0) {
      std::stringstream ss2;
      ss2 << "RENAMED_COL_" << i;
      if (it->second != ss2.str()) {
        COGADB_ERROR("Column not renamed correctly!", "");
        return false;
      }
    } else {
      std::stringstream ss1;
      ss1 << "COL_" << i;
      if (it->second != ss1.str()) {
        COGADB_ERROR("Column was incorrectly renamed!", "");
        return false;
      }
    }
    ++i;
  }

  RenameList rename_list2;
  // now rename all odd numbered columns
  for (unsigned int i = 1; i < 10; i += 2) {
    std::stringstream ss1;
    ss1 << "COL_" << i;
    std::stringstream ss2;
    ss2 << "RENAMED_COL_" << i;
    rename_list2.push_back(RenameEntry(ss1.str(), ss2.str()));
  }

  boost::shared_ptr<logical_operator::Logical_Scan> scan(
      new logical_operator::Logical_Scan(table));
  boost::shared_ptr<logical_operator::Logical_Rename> rename(
      new logical_operator::Logical_Rename(rename_list2));
  rename->setLeft(scan);
  LogicalQueryPlan log_plan(rename);

  ClientPtr client(new LocalClient());
  if (!client) return false;
  CoGaDB::query_processing::PhysicalQueryPlanPtr result_plan =
      optimize_and_execute("", log_plan, client);
  assert(result_plan != NULL);
  TablePtr result = result_plan->getResult();
  if (!result) {
    log_plan.print();
    COGADB_ERROR("Failed to execute Rename Query!", "");
    return false;
  }
  if (!quiet && verbose && debug) result->print();

  new_schema = result->getSchema();
  // TableSchema::iterator it;
  i = 0;
  for (it = new_schema.begin(); it != new_schema.end(); ++it) {
    std::stringstream ss2;
    ss2 << "RENAMED_COL_" << i;
    if (it->second != ss2.str()) {
      COGADB_ERROR("Column not renamed correctly!", "");
      return false;
    }
    ++i;
  }

  return true;
}

bool PrimaryForeignKeyTest() {
  {  // test case where everything is fine
    TableSchema schema;
    schema.push_back(Attribut(INT, "SID"));
    schema.push_back(Attribut(VARCHAR, "Studiengang"));

    TablePtr pk_tab(new Table("Studiengänge", schema));
    if (!pk_tab) {
      cout << "Failed to create Table 'Studiengänge'!" << endl;
      return false;
    }

    {
      Tuple t;
      t.push_back(1);
      t.push_back(string("INF"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(2);
      t.push_back(string("CV"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(3);
      t.push_back(string("CSE"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(4);
      t.push_back(string("WIF"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(5);
      t.push_back(string("INF Fernst."));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(6);
      t.push_back(string("CV Master"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(7);
      t.push_back(string("INGINF"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(8);
      t.push_back(string("Lehramt"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }

    if (!pk_tab->setPrimaryKeyConstraint("SID")) {
      COGADB_FATAL_ERROR(
          "Failed to set Primary Key Constraint on a column containing only "
          "unique keys!",
          "");
    }

    getGlobalTableList().push_back(pk_tab);

    TableSchema schema2;
    schema2.push_back(Attribut(VARCHAR, "Name"));
    schema2.push_back(Attribut(INT, "SID"));

    TablePtr fk_tab(new Table("Studenten", schema2));

    {
      Tuple t;
      t.push_back(string("Tom"));
      t.push_back(3);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Jennifer"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Maria"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Johannes"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Julia"));
      t.push_back(7);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Chris"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Matthias"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Maria"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Maximilian"));
      t.push_back(4);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Bob"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Susanne"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Stefanie"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Johannes"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Karl"));
      t.push_back(3);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }

    //        if(!fk_tab->setForeignKeyConstraint("SID",ForeignKeyConstraint("SID",
    //        "Studiengänge"))){
    if (!fk_tab->setForeignKeyConstraint("SID", "SID", "Studiengänge")) {
      COGADB_FATAL_ERROR(
          "Failed to set Foreign Key Constraint on a column containing only "
          "valid references to primary key column!",
          "");
    }
    // cleanup table list (delete pk_tab)
    getGlobalTableList().pop_back();
  }

  {  // test case where foreign key column has broken references
    //(i.e., FK column contains values not in PK column)
    TableSchema schema;
    schema.push_back(Attribut(INT, "SID"));
    schema.push_back(Attribut(VARCHAR, "Studiengang"));

    TablePtr pk_tab(new Table("Studiengänge", schema));
    if (!pk_tab) {
      cout << "Failed to create Table 'Studiengänge'!" << endl;
      return false;
    }

    {
      Tuple t;
      t.push_back(1);
      t.push_back(string("INF"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(2);
      t.push_back(string("CV"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(3);
      t.push_back(string("CSE"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(4);
      t.push_back(string("WIF"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(5);
      t.push_back(string("INF Fernst."));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(6);
      t.push_back(string("CV Master"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(7);
      t.push_back(string("INGINF"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(8);
      t.push_back(string("Lehramt"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }

    if (!pk_tab->setPrimaryKeyConstraint("SID")) {
      COGADB_FATAL_ERROR(
          "Failed to set Primary Key Constraint on a column containing only "
          "unique keys!",
          "");
    }

    getGlobalTableList().push_back(pk_tab);

    TableSchema schema2;
    schema2.push_back(Attribut(VARCHAR, "Name"));
    schema2.push_back(Attribut(INT, "SID"));

    TablePtr fk_tab(new Table("Studenten", schema2));

    {
      Tuple t;
      t.push_back(string("Tom"));
      t.push_back(3);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Jennifer"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Maria"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Johannes"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Julia"));
      t.push_back(10);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Chris"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Matthias"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Maria"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Maximilian"));
      t.push_back(4);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Bob"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Susanne"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Stefanie"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Johannes"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Karl"));
      t.push_back(3);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }

    if (fk_tab->setForeignKeyConstraint("SID", "SID", "Studiengänge")) {
      COGADB_FATAL_ERROR(
          "Incorrectly set Foreign Key Constraint on a column containing "
          "invalid references to primary key column!",
          "");
    }
    // cleanup table list (delete pk_tab)
    getGlobalTableList().pop_back();
  }

  {  // test case where PK column contains duplicates and (set PK constraint has
     // to fail!)
    // foreign key column has broken references and does not refererence a PK
    // column
    TableSchema schema;
    schema.push_back(Attribut(INT, "SID"));
    schema.push_back(Attribut(VARCHAR, "Studiengang"));

    TablePtr pk_tab(new Table("Studiengänge", schema));
    if (!pk_tab) {
      cout << "Failed to create Table 'Studiengänge'!" << endl;
      return false;
    }

    {
      Tuple t;
      t.push_back(1);
      t.push_back(string("INF"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(2);
      t.push_back(string("CV"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(3);
      t.push_back(string("CSE"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(4);
      t.push_back(string("WIF"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(5);
      t.push_back(string("INF Fernst."));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(6);
      t.push_back(string("CV Master"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(7);
      t.push_back(string("INGINF"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(8);
      t.push_back(string("Lehramt"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(6);
      t.push_back(string("CV Master"));
      if (!pk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }

    if (pk_tab->setPrimaryKeyConstraint("SID")) {
      COGADB_FATAL_ERROR(
          "Incorrectly set Primary Key Constraint on a column containing "
          "duplicate keys!",
          "");
    }

    getGlobalTableList().push_back(pk_tab);

    TableSchema schema2;
    schema2.push_back(Attribut(VARCHAR, "Name"));
    schema2.push_back(Attribut(INT, "SID"));

    TablePtr fk_tab(new Table("Studenten", schema2));

    {
      Tuple t;
      t.push_back(string("Tom"));
      t.push_back(3);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Jennifer"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Maria"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Johannes"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Julia"));
      t.push_back(10);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Chris"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Matthias"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Maria"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Maximilian"));
      t.push_back(4);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Bob"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Susanne"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Stefanie"));
      t.push_back(1);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Johannes"));
      t.push_back(2);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }
    {
      Tuple t;
      t.push_back(string("Karl"));
      t.push_back(3);
      if (!fk_tab->insert(t)) {
        cout << "Failed to insert Tuple" << endl;
        return false;
      }
    }

    if (fk_tab->setForeignKeyConstraint("SID", "SID", "Studiengänge")) {
      COGADB_FATAL_ERROR(
          "Incorrectly set Foreign Key Constraint on a column containing "
          "invalid references to a column that has no PK constraint!",
          "");
    }
    // cleanup table list (delete pk_tab)
    getGlobalTableList().pop_back();
  }
  return true;
}

#define ENABLE_GPU_ACCELERATION

bool joinTest() {
  TableSchema schema;
  schema.push_back(Attribut(INT, "PK_ID"));
  schema.push_back(Attribut(INT, "VAL_Table1"));

  for (unsigned int u = 0; u < 10; u++) {
    TablePtr primary_key_tab(new Table("PrimaryKeyTable", schema));

    for (unsigned int i = 0; i < 300; ++i) {
      {
        Tuple t;
        t.push_back(int(i));
        t.push_back(rand() % 1000);
        primary_key_tab->insert(t);
      }
      //{Tuple t; t.push_back(rand()%100); t.push_back(rand()%1000);
      // primary_key_tab->insert(t);}
    }

    TableSchema schema2;
    schema2.push_back(Attribut(INT, "FK_ID"));
    schema2.push_back(Attribut(INT, "VAL_Table2"));

    TablePtr foreign_key_tab(new Table("ForeignKeyTable", schema2));

    for (unsigned int i = 0; i < 10000; ++i) {
      // generate also foreign keys with no matching primary key (in case a
      // primary key table is prefiltered)
      {
        Tuple t;
        t.push_back(rand() % 1000);
        t.push_back(rand() % 1000);
        foreign_key_tab->insert(t);
      }
    }
    // for(unsigned int j=0;j<100;j++){
    Timestamp begin;
    Timestamp end;

    hype::ProcessingDeviceID id = hype::PD0;
    ProcessorSpecification proc_spec(id);
    JoinParam hash_join_param(proc_spec, HASH_JOIN);
    JoinParam parallel_hash_join_param(proc_spec, PARALLEL_HASH_JOIN);
    JoinParam nlj_param(proc_spec, NESTED_LOOP_JOIN);
    JoinParam sort_merge_param(proc_spec, SORT_MERGE_JOIN);

    begin = getTimestamp();
    TablePtr result_nested_loop_join = BaseTable::join(
        primary_key_tab, "PK_ID", foreign_key_tab, "FK_ID", nlj_param);
    end = getTimestamp();
    if (!quiet)
      cout << "NLJ: " << double(end - begin) / (1000 * 1000) << "ms" << endl;
    TablePtr result_sort_merge_join = BaseTable::join(
        primary_key_tab, "PK_ID", foreign_key_tab, "FK_ID", sort_merge_param);
    begin = getTimestamp();
    TablePtr result_hash_join = BaseTable::join(
        primary_key_tab, "PK_ID", foreign_key_tab, "FK_ID", hash_join_param);
    end = getTimestamp();
    if (!quiet)
      cout << "Serial HJ: " << double(end - begin) / (1000 * 1000) << "ms"
           << endl;
    begin = getTimestamp();
    TablePtr result_parallel_hash_join;
    // for(unsigned int i=0;i<100;++i)
    result_parallel_hash_join =
        BaseTable::join(primary_key_tab, "PK_ID", foreign_key_tab, "FK_ID",
                        parallel_hash_join_param);
    end = getTimestamp();
    if (!quiet)
      cout << "Parallel HJ: " << double(end - begin) / (1000 * 1000) << "ms"
           << endl;

#ifdef ENABLE_GPU_ACCELERATION
//	TablePtr result_gpu_merge_join =
// BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab,
//"FK_ID",SORT_MERGE_JOIN_2, LOOKUP,GPU);
#endif

    std::list<std::string> column_names;
    column_names.push_back("PK_ID");
    column_names.push_back("FK_ID");
    column_names.push_back("VAL_Table1");
    column_names.push_back("VAL_Table2");

    //        result can be the sam,e but differ in sort rder, so first sort
    //        both reatliosn and the compare tables
    //        Hint: assumes that complex sort works!
    result_nested_loop_join =
        BaseTable::sort(result_nested_loop_join, column_names);
    result_hash_join =
        BaseTable::sort(result_hash_join, column_names, ASCENDING, LOOKUP);
    result_parallel_hash_join = BaseTable::sort(
        result_parallel_hash_join, column_names, ASCENDING, LOOKUP);
#ifdef ENABLE_GPU_ACCELERATION
//	result_gpu_merge_join =
// BaseTable::sort(result_gpu_merge_join,column_names);
#endif
    /*
    if(result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(result_sort_merge_join->getColumnbyName("PK_ID"))){
        cerr << "Error in SortMergeJoin! PK_ID Column not correct!" << endl;
        return false;
    }
    if(result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(result_sort_merge_join->getColumnbyName("FK_ID"))){
        cerr << "Error in SortMergeJoin! FK_ID Column not correct!" << endl;
        return false;
    }*/
    if (!result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(
            result_hash_join->getColumnbyName("PK_ID"))) {
      cerr << "Error in HashJoin! PK_ID Column not correct!" << endl;
      cout << "PK_ID Column Sizes: NLJ: "
           << result_nested_loop_join->getColumnbyName("PK_ID")->size()
           << " HJ: " << result_hash_join->getColumnbyName("PK_ID")->size()
           << endl;
      //            cout << "Nested Loop Join:" << endl;
      //            result_nested_loop_join->getColumnbyName("PK_ID")->print();
      //            cout << "Hash Join:" << endl;
      //            result_hash_join->getColumnbyName("PK_ID")->print();
      //            cout << "Nested Loop Join:" << endl;
      //            result_nested_loop_join->print();
      //            cout << "Hash Join:" << endl;
      //            result_hash_join->print();
      return false;
    }
    if (!result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(
            result_hash_join->getColumnbyName("FK_ID"))) {
      cerr << "Error in HashJoin! FK_ID Column not correct!" << endl;
      return false;
    }

    if (!result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(
            result_parallel_hash_join->getColumnbyName("PK_ID"))) {
      cerr << "Error in parallel HashJoin! PK_ID Column not correct!" << endl;
      cout << "PK_ID Column Sizes: NLJ: "
           << result_nested_loop_join->getColumnbyName("PK_ID")->size()
           << " HJ: "
           << result_parallel_hash_join->getColumnbyName("PK_ID")->size()
           << endl;
      //            cout << "Nested Loop Join:" << endl;
      //            result_nested_loop_join->getColumnbyName("PK_ID")->print();
      //            cout << "Hash Join:" << endl;
      //            result_hash_join->getColumnbyName("PK_ID")->print();
      //            cout << "Nested Loop Join:" << endl;
      //            result_nested_loop_join->print();
      //            cout << "Hash Join:" << endl;
      //            result_hash_join->print();
      return false;
    }
    if (!result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(
            result_parallel_hash_join->getColumnbyName("FK_ID"))) {
      cerr << "Error in parallel HashJoin! FK_ID Column not correct!" << endl;
      return false;
    }

#ifdef ENABLE_GPU_ACCELERATION
//        if(!result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(result_gpu_merge_join->getColumnbyName("PK_ID"))){
//            cerr << "Error in GPU SortMergeJoin! PK_ID Column not correct!" <<
//            endl;
//            cout << "Nested Loop Join:" << endl;
//            result_nested_loop_join->getColumnbyName("PK_ID")->print();
//            cout << "GPU Merge Join:" << endl;
//            result_gpu_merge_join->getColumnbyName("PK_ID")->print();
//            cout << "Nested Loop Join:" << endl;
//            result_nested_loop_join->print();
//            cout << "GPU Merge Join:" << endl;
//            result_gpu_merge_join->print();
//            return false;
//        }
//        if(!result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(result_gpu_merge_join->getColumnbyName("FK_ID"))){
//            cerr << "Error in GPU SortMergeJoin! FK_ID Column not correct!" <<
//            endl;
//            return false;
//        }
#endif
  }

  return true;
}

bool generalJoinTest() {
  TableSchema schema;
  schema.push_back(Attribut(INT, "T1_ID"));
  schema.push_back(Attribut(INT, "VAL_Table1"));

  for (unsigned int u = 0; u < 10; u++) {
    TablePtr primary_key_tab(new Table("PrimaryKeyTable", schema));

    const size_t SCALE_FACTOR = 100;

    for (unsigned int i = 0; i < 300 * SCALE_FACTOR; ++i) {
      //{Tuple t; t.push_back(int(i)); t.push_back(rand()%1000);
      // primary_key_tab->insert(t);}
      {
        Tuple t;
        t.push_back(int(rand() % (300 * SCALE_FACTOR)));
        t.push_back(rand() % 1000);
        primary_key_tab->insert(t);
      }
    }

    TableSchema schema2;
    schema2.push_back(Attribut(INT, "T2_ID"));
    schema2.push_back(Attribut(INT, "VAL_Table2"));

    TablePtr foreign_key_tab(new Table("ForeignKeyTable", schema2));

    for (unsigned int i = 0; i < 10000 * SCALE_FACTOR; ++i) {
      // generate also foreign keys with no matching primary key (in case a
      // primary key table is prefiltered)
      {
        Tuple t;
        t.push_back(int(rand() % (300 * SCALE_FACTOR)));
        t.push_back(rand() % 1000);
        foreign_key_tab->insert(t);
      }
    }

    hype::ProcessingDeviceID id = hype::PD0;
    ProcessorSpecification proc_spec(id);
    JoinParam hash_join_param(proc_spec, HASH_JOIN);
    JoinParam nlj_param(proc_spec, NESTED_LOOP_JOIN);
    JoinParam sort_merge_param(proc_spec, SORT_MERGE_JOIN);

    ProcessorSpecification gpu_spec(getIDOfFirstGPU());
    JoinParam gpu_join_param(gpu_spec, NESTED_LOOP_JOIN);

    Timestamp begin;
    Timestamp end;
    begin = getTimestamp();
    TablePtr result_nested_loop_join = BaseTable::join(
        primary_key_tab, "T1_ID", foreign_key_tab, "T2_ID", nlj_param);
    end = getTimestamp();
    if (!quiet)
      cout << "NLJ: " << double(end - begin) / (1000 * 1000) << "ms" << endl;
    TablePtr result_sort_merge_join = BaseTable::join(
        primary_key_tab, "T1_ID", foreign_key_tab, "T2_ID", sort_merge_param);
    begin = getTimestamp();
    TablePtr result_hash_join = BaseTable::join(
        primary_key_tab, "T1_ID", foreign_key_tab, "T2_ID", hash_join_param);
    end = getTimestamp();
    if (!quiet)
      cout << "Serial HJ: " << double(end - begin) / (1000 * 1000) << "ms"
           << endl;
    begin = getTimestamp();
    // TablePtr result_parallel_hash_join;
    // for(unsigned int i=0;i<100;++i)
    // result_parallel_hash_join =
    // BaseTable::join(primary_key_tab,"T1_ID",foreign_key_tab,
    // "T2_ID",PARALLEL_HASH_JOIN, LOOKUP,CPU);
    end = getTimestamp();
    if (!quiet)
      cout << "Parallel HJ: " << double(end - begin) / (1000 * 1000) << "ms"
           << endl;

#ifdef ENABLE_GPU_ACCELERATION
    TablePtr result_gpu_merge_join = BaseTable::join(
        primary_key_tab, "T1_ID", foreign_key_tab, "T2_ID", gpu_join_param);
#endif

    std::list<std::string> column_names;
    column_names.push_back("T1_ID");
    column_names.push_back("T2_ID");
    column_names.push_back("VAL_Table1");
    column_names.push_back("VAL_Table2");

    //        result can be the sam,e but differ in sort rder, so first sort
    //        both reatliosn and the compare tables
    //        Hint: assumes that complex sort works!
    result_nested_loop_join =
        BaseTable::sort(result_nested_loop_join, column_names);
    result_hash_join =
        BaseTable::sort(result_hash_join, column_names, ASCENDING, LOOKUP);
// result_parallel_hash_join =
// BaseTable::sort(result_parallel_hash_join,column_names,ASCENDING,LOOKUP);
#ifdef ENABLE_GPU_ACCELERATION
    result_gpu_merge_join =
        BaseTable::sort(result_gpu_merge_join, column_names);
#endif
    /*
    if(result_nested_loop_join->getColumnbyName("T1_ID")->is_equal(result_sort_merge_join->getColumnbyName("T1_ID"))){
        cerr << "Error in SortMergeJoin! T1_ID Column not correct!" << endl;
        return false;
    }
    if(result_nested_loop_join->getColumnbyName("T2_ID")->is_equal(result_sort_merge_join->getColumnbyName("T2_ID"))){
        cerr << "Error in SortMergeJoin! T2_ID Column not correct!" << endl;
        return false;
    }*/
    if (!result_nested_loop_join->getColumnbyName("T1_ID")->is_equal(
            result_hash_join->getColumnbyName("T1_ID"))) {
      cerr << "Error in HashJoin! T1_ID Column not correct!" << endl;
      cout << "T1_ID Column Sizes: NLJ: "
           << result_nested_loop_join->getColumnbyName("T1_ID")->size()
           << " HJ: " << result_hash_join->getColumnbyName("T1_ID")->size()
           << endl;
      //            cout << "Nested Loop Join:" << endl;
      //            result_nested_loop_join->getColumnbyName("T1_ID")->print();
      //            cout << "Hash Join:" << endl;
      //            result_hash_join->getColumnbyName("T1_ID")->print();
      //            cout << "Nested Loop Join:" << endl;
      //            result_nested_loop_join->print();
      //            cout << "Hash Join:" << endl;
      //            result_hash_join->print();
      return false;
    }
    if (!result_nested_loop_join->getColumnbyName("T2_ID")->is_equal(
            result_hash_join->getColumnbyName("T2_ID"))) {
      cerr << "Error in HashJoin! T2_ID Column not correct!" << endl;
      return false;
    }

//        if(!result_nested_loop_join->getColumnbyName("T1_ID")->is_equal(result_parallel_hash_join->getColumnbyName("T1_ID"))){
//            cerr << "Error in parallel HashJoin! T1_ID Column not correct!" <<
//            endl;
//            cout << "T1_ID Column Sizes: NLJ: " <<
//            result_nested_loop_join->getColumnbyName("T1_ID")->size() << " HJ:
//            " << result_parallel_hash_join->getColumnbyName("T1_ID")->size()
//            << endl;
////            cout << "Nested Loop Join:" << endl;
////            result_nested_loop_join->getColumnbyName("T1_ID")->print();
////            cout << "Hash Join:" << endl;
////            result_hash_join->getColumnbyName("T1_ID")->print();
////            cout << "Nested Loop Join:" << endl;
////            result_nested_loop_join->print();
////            cout << "Hash Join:" << endl;
////            result_hash_join->print();
//            return false;
//        }
//        if(!result_nested_loop_join->getColumnbyName("T2_ID")->is_equal(result_parallel_hash_join->getColumnbyName("T2_ID"))){
//            cerr << "Error in parallel HashJoin! T2_ID Column not correct!" <<
//            endl;
//            return false;
//        }

#ifdef ENABLE_GPU_ACCELERATION
    if (!result_nested_loop_join->getColumnbyName("T1_ID")->is_equal(
            result_gpu_merge_join->getColumnbyName("T1_ID"))) {
      cerr << "Error in GPU SortMergeJoin! T1_ID Column not correct!" << endl;
      cout << "Nested Loop Join:" << endl;
      result_nested_loop_join->getColumnbyName("T1_ID")->print();
      cout << "GPU Merge Join:" << endl;
      result_gpu_merge_join->getColumnbyName("PK_ID")->print();
      cout << "Nested Loop Join:" << endl;
      result_nested_loop_join->print();
      cout << "GPU Merge Join:" << endl;
      result_gpu_merge_join->print();
      return false;
    }
    if (!result_nested_loop_join->getColumnbyName("T2_ID")->is_equal(
            result_gpu_merge_join->getColumnbyName("T2_ID"))) {
      cerr << "Error in GPU SortMergeJoin! T2_ID Column not correct!" << endl;
      return false;
    }
#endif
  }

  return true;
}

bool JoinIndexTest() {
  //     	TableSchema schema;
  //	schema.push_back(Attribut(INT,"PK_ID"));
  //	schema.push_back(Attribut(INT,"VAL_Table1"));
  //
  ////        for(unsigned int u=0;u<10;u++)
  ////        {
  //	TablePtr primary_key_tab(new Table("PrimaryKeyTable",schema));
  //
  //
  //
  //        for(unsigned int i=0;i<300;++i){
  //                {Tuple t; t.push_back(int(i)); t.push_back(rand()%1000);
  //                primary_key_tab->insert(t);}
  //                //{Tuple t; t.push_back(rand()%100);
  //                t.push_back(rand()%1000); primary_key_tab->insert(t);}
  //
  //        }
  //
  //
  //        TableSchema schema2;
  //	schema2.push_back(Attribut(INT,"FK_ID"));
  //	schema2.push_back(Attribut(INT,"VAL_Table2"));
  //
  //	TablePtr foreign_key_tab(new Table("ForeignKeyTable",schema2));
  //
  //        for(unsigned int i=0;i<10000;++i){
  //                //generate also foreign keys with no matching primary key
  //                (in case a primary key table is prefiltered)
  //                {Tuple t; t.push_back(rand()%1000);
  //                t.push_back(rand()%1000); foreign_key_tab->insert(t);}
  //        }
  //     	//for(unsigned int j=0;j<100;j++){
  //        Timestamp begin;
  //        Timestamp end;
  //
  //        if(!quiet)
  //        cout << "FetchJoin: " << double(end-begin)/(1000*1000) << "ms" <<
  //        endl;
  //
  //	TablePtr result_fetch_join; // =
  // BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab,
  //"FK_ID",SORT_MERGE_JOIN, LOOKUP,CPU);
  //        begin=getTimestamp();
  // 	TablePtr result_hash_join =
  // BaseTable::join(primary_key_tab,"PK_ID",foreign_key_tab, "FK_ID",HASH_JOIN,
  // LOOKUP,CPU);
  //        end=getTimestamp();
  //        if(!quiet)
  //        cout << "Serial HJ: " << double(end-begin)/(1000*1000) << "ms" <<
  //        endl;
  //
  //        std::list<std::string> column_names;
  //        column_names.push_back("PK_ID");
  //        column_names.push_back("FK_ID");
  //        column_names.push_back("VAL_Table1");
  //        column_names.push_back("VAL_Table2");
  //
  //
  ////        result can be the same but differ in sort order, so first sort
  /// both relations and the compare tables
  ////        Hint: assumes that complex sort works!
  //        result_hash_join =
  //        BaseTable::sort(result_hash_join,column_names,ASCENDING,LOOKUP);
  //
  //
  //        /*
  //        if(result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(result_sort_merge_join->getColumnbyName("PK_ID"))){
  //            cerr << "Error in SortMergeJoin! PK_ID Column not correct!" <<
  //            endl;
  //            return false;
  //        }
  //        if(result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(result_sort_merge_join->getColumnbyName("FK_ID"))){
  //            cerr << "Error in SortMergeJoin! FK_ID Column not correct!" <<
  //            endl;
  //            return false;
  //        }*/
  //        if(!result_fetch_join->getColumnbyName("PK_ID")->is_equal(result_hash_join->getColumnbyName("PK_ID"))){
  //            cerr << "Error in HashJoin! PK_ID Column not correct!" << endl;
  //            cout << "PK_ID Column Sizes: NLJ: " <<
  //            result_fetch_join->getColumnbyName("PK_ID")->size() << " HJ: "
  //            << result_hash_join->getColumnbyName("PK_ID")->size() << endl;
  ////            cout << "Nested Loop Join:" << endl;
  ////            result_nested_loop_join->getColumnbyName("PK_ID")->print();
  ////            cout << "Hash Join:" << endl;
  ////            result_hash_join->getColumnbyName("PK_ID")->print();
  ////            cout << "Nested Loop Join:" << endl;
  ////            result_nested_loop_join->print();
  ////            cout << "Hash Join:" << endl;
  ////            result_hash_join->print();
  //            return false;
  //        }
  //        if(!result_fetch_join->getColumnbyName("FK_ID")->is_equal(result_hash_join->getColumnbyName("FK_ID"))){
  //            cerr << "Error in HashJoin! FK_ID Column not correct!" << endl;
  //            return false;
  //        }
  //
  return true;
}

bool JoinPerformanceTest() {
  boost::shared_ptr<Column<int> > pk_col(new Column<int>("PK", INT));
  boost::shared_ptr<Column<int> > fk_col(new Column<int>("FK", INT));
  for (unsigned int i = 0; i < 10000; ++i) {
    pk_col->insert(i);
  }

  TableSchema schema2;
  schema2.push_back(Attribut(INT, "FK_ID"));
  schema2.push_back(Attribut(INT, "VAL_Table2"));

  TablePtr foreign_key_tab(new Table("ForeignKeyTable", schema2));

  for (unsigned int i = 0; i < 1000 * 1000; ++i) {
    fk_col->insert(rand() % 10000);
  }

  unsigned int number_of_threads = 4;
  cout << "Enter number of threads:" << endl;
  cin >> number_of_threads;

  CoGaDB::PositionListPairPtr result_nested_loop_join;
  CoGaDB::PositionListPairPtr result_hash_join;
  CoGaDB::PositionListPairPtr result_parallel_hash_join;

  Timestamp begin;
  Timestamp end;
  for (unsigned int i = 0; i < 100; ++i) {
    //        {
    //        begin=getTimestamp();
    //        result_nested_loop_join = pk_col->nested_loop_join(fk_col);
    //        end=getTimestamp();
    //        cout << "Serial Nested Loop Join: " <<
    //        double(end-begin)/(1000*1000) << "ms" << endl;
    //        }
    {
      begin = getTimestamp();
      result_hash_join = pk_col->hash_join(fk_col);
      end = getTimestamp();
      cout << "Serial Hash Join: " << double(end - begin) / (1000 * 1000)
           << "ms" << endl;
    }
    {
      begin = getTimestamp();
      result_parallel_hash_join =
          pk_col->parallel_hash_join(fk_col, number_of_threads);
      end = getTimestamp();
      cout << "Parallel Hash Join: " << double(end - begin) / (1000 * 1000)
           << "ms" << endl;
    }
  }

  if ((*result_hash_join->first) != (*result_parallel_hash_join->first)) {
    cout << "TID lists (PK) are not equal!" << endl;
    cout << "Serial Hash Join result size: " << result_hash_join->first->size()
         << endl;
    cout << "Parallel Hash Join result size: "
         << result_parallel_hash_join->first->size() << endl;
    COGADB_FATAL_ERROR(
        "Selection Unittests failed! At least one algorithm works incorrect!",
        "");
  }

  if ((*result_hash_join->second) != (*result_parallel_hash_join->second)) {
    cout << "TID lists (FK) are not equal!" << endl;
    cout << "Serial Hash Join result size: " << result_hash_join->second->size()
         << endl;
    cout << "Parallel Hash Join result size: "
         << result_parallel_hash_join->second->size() << endl;
    COGADB_FATAL_ERROR(
        "Selection Unittests failed! At least one algorithm works incorrect!",
        "");
  }

  // assert((*result_nested_loop_join)==(*result_hash_join));

  // exit(0);
  return true;
}

bool crossjoinTest() {
  TableSchema schema;
  schema.push_back(Attribut(INT, "PK_ID"));
  schema.push_back(Attribut(INT, "VAL_Table1"));

  TablePtr primary_key_tab(new Table("PrimaryKeyTable", schema));

  for (unsigned int i = 0; i < 10; ++i) {
    {
      Tuple t;
      t.push_back(0);
      t.push_back(rand() % 1000);
      primary_key_tab->insert(t);
    }
  }

  TableSchema schema2;
  schema2.push_back(Attribut(INT, "FK_ID"));
  schema2.push_back(Attribut(INT, "VAL_Tabl2"));

  TablePtr foreign_key_tab(new Table("ForeignKeyTable", schema2));

  for (unsigned int i = 0; i < 100; ++i) {
    {
      Tuple t;
      t.push_back(0);
      t.push_back(rand() % 1000);
      foreign_key_tab->insert(t);
    }
  }

  hype::ProcessingDeviceID id = hype::PD0;
  ProcessorSpecification proc_spec(id);
  JoinParam param(proc_spec, NESTED_LOOP_JOIN);

  TablePtr result_nested_loop_join = BaseTable::join(
      primary_key_tab, "PK_ID", foreign_key_tab, "FK_ID", param);
  TablePtr result_crossjoin =
      BaseTable::crossjoin(primary_key_tab, foreign_key_tab, LOOKUP);

  if (!quiet) {
    primary_key_tab->print();
    foreign_key_tab->print();
    cout << "Result for Nested Loop Join:" << endl;
    result_nested_loop_join->print();
    cout << "Result for Cross Join:" << endl;
    result_crossjoin->print();
  }

  if (!result_nested_loop_join->getColumnbyName("PK_ID")->is_equal(
          result_crossjoin->getColumnbyName("PK_ID"))) {
    cerr << "Error in CrossJoin! PK_ID Column not correct!" << endl;
    return false;
  }
  if (!result_nested_loop_join->getColumnbyName("FK_ID")->is_equal(
          result_crossjoin->getColumnbyName("FK_ID"))) {
    cerr << "Error in CrossJoin! FK_ID Column not correct!" << endl;
    return false;
  }

  CoGaDB::getGlobalTableList().push_back(primary_key_tab);
  CoGaDB::getGlobalTableList().push_back(foreign_key_tab);

  boost::shared_ptr<logical_operator::Logical_Scan> scan_pk_table(
      new logical_operator::Logical_Scan("PrimaryKeyTable"));
  boost::shared_ptr<logical_operator::Logical_Scan> scan_fk_table(
      new logical_operator::Logical_Scan("ForeignKeyTable"));
  boost::shared_ptr<logical_operator::Logical_CrossJoin> cross_join(
      new logical_operator::Logical_CrossJoin());

  cross_join->setLeft(scan_pk_table);
  cross_join->setRight(scan_fk_table);

  LogicalQueryPlan log_plan(cross_join);
  // log_plan.print();

  PhysicalQueryPlanPtr plan = log_plan.convertToPhysicalQueryPlan();
  // plan->print();
  plan->run();
  // TablePtr ptr = plan->getResult();
  // ptr->print();

  return true;
}

bool complexSelectionTest() {
  TableSchema schema;
  schema.push_back(Attribut(INT, "YEAR"));
  schema.push_back(Attribut(VARCHAR, "Name"));
  schema.push_back(Attribut(INT, "Age"));
  schema.push_back(Attribut(FLOAT, "Score"));

  TablePtr tab1(new Table("Person", schema));

  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Peter"));
    t.push_back(24);
    t.push_back(float(23.1));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2011);
    t.push_back(string("James"));
    t.push_back(20);
    t.push_back(float(40.6));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Karl"));
    t.push_back(30);
    t.push_back(float(99.4));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2012);
    t.push_back(string("Max"));
    t.push_back(17);
    t.push_back(float(10.8));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2011);
    t.push_back(string("Bob"));
    t.push_back(21);
    t.push_back(float(84.2));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Alice"));
    t.push_back(22);
    t.push_back(float(30.6));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2012);
    t.push_back(string("Susanne"));
    t.push_back(25);
    t.push_back(float(29.3));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Joe"));
    t.push_back(40);
    t.push_back(float(72.0));
    tab1->insert(t);
  }
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Alice"));
    t.push_back(33);
    t.push_back(float(95.4));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2012);
    t.push_back(string("Anne"));
    t.push_back(16);
    t.push_back(float(52.3));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2011);
    t.push_back(string("James"));
    t.push_back(9);
    t.push_back(float(5.0));
    tab1->insert(t);
  }

  CoGaDB::getGlobalTableList().push_back(tab1);

  KNF_Selection_Expression
      knf_expr;  //(YEAR<2013 OR Name="Peter") AND (Age>18))
  {
    Disjunction d;
    d.push_back(Predicate("YEAR", boost::any(2013), ValueConstantPredicate,
                          LESSER));  // YEAR<2013
    d.push_back(Predicate("Name", boost::any(std::string("Peter")),
                          ValueConstantPredicate, EQUAL));
    knf_expr.disjunctions.push_back(d);
  }
  {
    Disjunction d;
    d.push_back(
        Predicate("Age", boost::any(18), ValueConstantPredicate, GREATER));
    knf_expr.disjunctions.push_back(d);
  }

  TablePtr result = BaseTable::selection(tab1, knf_expr, LOOKUP);
  tab1->print();
  cout << "Results for Query (YEAR<2013 OR Name=\"Peter\") AND (Age>18))"
       << endl;
  result->print();

  TablePtr result2 =
      CoGaDB::query_processing::two_phase_physical_optimization_selection(
          tab1, knf_expr, hype::DeviceConstraint(hype::CPU_ONLY), LOOKUP);
  result2->print();
  // exit(0);

  boost::shared_ptr<logical_operator::Logical_Scan> scan(
      new logical_operator::Logical_Scan("Person"));
  boost::shared_ptr<logical_operator::Logical_ComplexSelection>
      complex_selection(
          new logical_operator::Logical_ComplexSelection(knf_expr));
  complex_selection->setLeft(scan);
  LogicalQueryPlan log_plan(complex_selection);
  log_plan.print();

  PhysicalQueryPlanPtr plan = log_plan.convertToPhysicalQueryPlan();
  // plan->print();
  plan->run();
  TablePtr ptr = plan->getResult();
  ptr->print();

  // query_processing::column_processing::cpu::LogicalQueryPlanPtr
  // createColumnBasedQueryPlan(TablePtr table, const KNF_Selection_Expression&
  // knf_expr);
  query_processing::column_processing::cpu::LogicalQueryPlanPtr column_log_plan;
  column_log_plan = createColumnBasedQueryPlan(
      tab1, knf_expr, hype::DeviceConstraint(hype::CPU_ONLY));
  column_log_plan->print();
  query_processing::column_processing::cpu::PhysicalQueryPlanPtr
      column_phy_plan = column_log_plan->convertToPhysicalQueryPlan();
  column_phy_plan->run();
  // column_phy_plan->getResult()-
  // exit(0);

  {
    TableSchema schema;
    schema.push_back(Attribut(INT, "NUMBER1"));
    schema.push_back(Attribut(INT, "NUMBER2"));

    TablePtr tab1(new Table("ComparisonTestTable", schema));
    {
      Tuple t;
      t.push_back(1);
      t.push_back(24);
      tab1->insert(t);
    }
    {
      Tuple t;
      t.push_back(33);
      t.push_back(33);
      tab1->insert(t);
    }
    {
      Tuple t;
      t.push_back(45);
      t.push_back(24);
      tab1->insert(t);
    }
    {
      Tuple t;
      t.push_back(5);
      t.push_back(3);
      tab1->insert(t);
    }
    {
      Tuple t;
      t.push_back(33);
      t.push_back(32);
      tab1->insert(t);
    }
    {
      Tuple t;
      t.push_back(45);
      t.push_back(45);
      tab1->insert(t);
    }
    {
      Tuple t;
      t.push_back(2);
      t.push_back(2);
      tab1->insert(t);
    }
    {
      Tuple t;
      t.push_back(34);
      t.push_back(33);
      tab1->insert(t);
    }
    {
      Tuple t;
      t.push_back(4);
      t.push_back(24);
      tab1->insert(t);
    }

    getGlobalTableList().push_back(tab1);
    tab1->print();
  }

  {
    KNF_Selection_Expression
        knf_expr;  //(YEAR<2013 OR Name="Peter") AND (Age>18))
    {
      Disjunction d;
      d.push_back(Predicate(string("NUMBER1"), string("NUMBER2"),
                            ValueValuePredicate, EQUAL));  // YEAR<2013
      knf_expr.disjunctions.push_back(d);
    }

    boost::shared_ptr<logical_operator::Logical_Scan> scan(
        new logical_operator::Logical_Scan("ComparisonTestTable"));
    boost::shared_ptr<logical_operator::Logical_ComplexSelection>
        complex_selection(
            new logical_operator::Logical_ComplexSelection(knf_expr));
    complex_selection->setLeft(scan);
    LogicalQueryPlan log_plan(complex_selection);
    log_plan.print();

    PhysicalQueryPlanPtr plan = log_plan.convertToPhysicalQueryPlan();
    // plan->print();
    plan->run();
    TablePtr ptr = plan->getResult();
    cout << "Result of Query: NUMBER1==NUMBER2:" << endl;
    ptr->print();
  }

  {
    KNF_Selection_Expression
        knf_expr;  //(YEAR<2013 OR Name="Peter") AND (Age>18))
    {
      Disjunction d;
      d.push_back(Predicate(string("NUMBER1"), string("NUMBER2"),
                            ValueValuePredicate, GREATER));  // YEAR<2013
      knf_expr.disjunctions.push_back(d);
    }

    boost::shared_ptr<logical_operator::Logical_Scan> scan(
        new logical_operator::Logical_Scan("ComparisonTestTable"));
    boost::shared_ptr<logical_operator::Logical_ComplexSelection>
        complex_selection(
            new logical_operator::Logical_ComplexSelection(knf_expr));
    complex_selection->setLeft(scan);
    LogicalQueryPlan log_plan(complex_selection);
    log_plan.print();

    PhysicalQueryPlanPtr plan = log_plan.convertToPhysicalQueryPlan();
    // plan->print();
    plan->run();
    TablePtr ptr = plan->getResult();
    cout << "Result of Query: NUMBER1>NUMBER2:" << endl;
    ptr->print();
  }
  {
    KNF_Selection_Expression
        knf_expr;  //(YEAR<2013 OR Name="Peter") AND (Age>18))
    {
      Disjunction d;
      d.push_back(Predicate(string("NUMBER1"), string("NUMBER2"),
                            ValueValuePredicate, LESSER));  // YEAR<2013
      knf_expr.disjunctions.push_back(d);
    }

    boost::shared_ptr<logical_operator::Logical_Scan> scan(
        new logical_operator::Logical_Scan("ComparisonTestTable"));
    boost::shared_ptr<logical_operator::Logical_ComplexSelection>
        complex_selection(
            new logical_operator::Logical_ComplexSelection(knf_expr));
    complex_selection->setLeft(scan);
    LogicalQueryPlan log_plan(complex_selection);
    log_plan.print();

    PhysicalQueryPlanPtr plan = log_plan.convertToPhysicalQueryPlan();
    // plan->print();
    plan->run();
    TablePtr ptr = plan->getResult();
    cout << "Result of Query: NUMBER1<NUMBER2:" << endl;
    ptr->print();
  }

  {
    KNF_Selection_Expression
        knf_expr;  //(YEAR<2013 OR Name="Peter") AND (Age>18))
    {
      Disjunction d;
      d.push_back(Predicate("YEAR", boost::any(2013), ValueConstantPredicate,
                            LESSER));  // YEAR<2013
      d.push_back(Predicate("Name", boost::any(std::string("Peter")),
                            ValueConstantPredicate, EQUAL));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(
          Predicate("Age", boost::any(18), ValueConstantPredicate, GREATER));
      d.push_back(Predicate("Score", boost::any(float(80)),
                            ValueConstantPredicate, GREATER));
      knf_expr.disjunctions.push_back(d);
    }
    {
      Disjunction d;
      d.push_back(
          Predicate("Age", boost::any(18), ValueConstantPredicate, GREATER));
      d.push_back(Predicate("Score", boost::any(float(80)),
                            ValueConstantPredicate, GREATER));
      knf_expr.disjunctions.push_back(d);
    }
    query_processing::column_processing::cpu::LogicalQueryPlanPtr
        column_log_plan;
    column_log_plan = createColumnBasedQueryPlan(tab1, knf_expr);
    column_log_plan->print();
    // exit(0);
  }

  return true;
}

bool ColumnComputationTest() {
  // std::transform ( first, first+5, second, results, std::plus<int>() );
  boost::shared_ptr<Column<int> > sales_column(new Column<int>("sales", INT));
  boost::shared_ptr<Column<int> > pricing_column(new Column<int>("price", INT));

  vector<int> sales;
  vector<int> pricing;

  for (unsigned int i = 0; i < 10; i++) {
    sales.push_back(rand() % 100);
    pricing.push_back((rand() % 10) +
                      1);  // ensure that no zeros are in pricing vector
  }

  sales_column->insert(sales.begin(), sales.end());
  pricing_column->insert(pricing.begin(), pricing.end());

  // verify columns are equal to vectors
  for (unsigned int i = 0; i < sales.size(); i++) {
    if (sales[i] != (*sales_column)[i]) return false;
    if (pricing[i] != (*pricing_column)[i]) return false;
  }

  return true;
}

bool GPUColumnComputationTest() {
  //	TableSchema schema;
  //	schema.push_back(Attribut(INT,"Shop_ID"));
  //	schema.push_back(Attribut(INT,"Price"));
  //	schema.push_back(Attribut(INT,"Sales"));
  //
  //	TablePtr tab1(new Table("Sale",schema));
  //
  //	for(unsigned int i=0;i<10;i++){
  //		Tuple t;
  //		t.push_back((int)i%10);
  //		t.push_back((int)rand()%100);
  //		t.push_back((int)(rand()%1000)+1);
  //		tab1->insert(t);
  //	}
  //
  //	//tab1->print();
  //
  //	ColumnPtr key_col = tab1->getColumnbyName(string("Shop_ID"));
  //	ColumnPtr price_col = tab1->getColumnbyName(string("Price"));
  //	ColumnPtr sales_col = tab1->getColumnbyName(string("Sales"));
  //    //create GPU columns
  //	gpu::GPU_Base_ColumnPtr dev_key_col   =
  // gpu::copy_column_host_to_device(key_col);
  //	gpu::GPU_Base_ColumnPtr dev_price_col =
  // gpu::copy_column_host_to_device(price_col);
  //	gpu::GPU_Base_ColumnPtr dev_sales_col =
  // gpu::copy_column_host_to_device(sales_col);
  //
  //	//copy columns to check correctness
  //	ColumnPtr host_price_col = price_col->copy();
  //	ColumnPtr host_sales_col = sales_col->copy();
  //
  ///**** CHECK Correctness for Computation using Columns*****/
  //	{
  //	host_price_col->add(host_sales_col);
  //	dev_price_col->add(dev_sales_col);
  //
  //	ColumnPtr host_res = gpu::copy_column_device_to_host(dev_price_col);
  //	//host_price_col->print();
  //	//dev_price_col->print();
  //	//host_res->print();
  //	if(!host_res->is_equal(host_price_col)) return false;
  //	}
  //	{
  //	host_price_col->minus(host_sales_col);
  //	dev_price_col->minus(dev_sales_col);
  //	ColumnPtr host_res = gpu::copy_column_device_to_host(dev_price_col);
  //	if(!host_res->is_equal(host_price_col)) return false;
  //	}
  //	{
  //	host_price_col->multiply(host_sales_col);
  //	dev_price_col->multiply(dev_sales_col);
  //	ColumnPtr host_res = gpu::copy_column_device_to_host(dev_price_col);
  //	if(!host_res->is_equal(host_price_col)) return false;
  //	}
  //	{
  //	host_price_col->division(host_sales_col);
  //	dev_price_col->division(dev_sales_col);
  //	ColumnPtr host_res = gpu::copy_column_device_to_host(dev_price_col);
  //	if(!host_res->is_equal(host_price_col)) return false;
  //	}
  //
  ///**** CHECK Correctness for Computation using Constants*****/
  //
  //	int random_number=(rand()%100)+1;
  //	{
  //	host_price_col->add(random_number);
  //	dev_price_col->add(random_number);
  //
  //	ColumnPtr host_res = gpu::copy_column_device_to_host(dev_price_col);
  //	//host_price_col->print();
  //	//dev_price_col->print();
  //	//host_res->print();
  //	if(!host_res->is_equal(host_price_col)) return false;
  //	}
  //	{
  //	host_price_col->minus(random_number);
  //	dev_price_col->minus(random_number);
  //	ColumnPtr host_res = gpu::copy_column_device_to_host(dev_price_col);
  //	if(!host_res->is_equal(host_price_col)) return false;
  //	}
  //	{
  //	host_price_col->multiply(random_number);
  //	dev_price_col->multiply(random_number);
  //	ColumnPtr host_res = gpu::copy_column_device_to_host(dev_price_col);
  //	if(!host_res->is_equal(host_price_col)) return false;
  //	}
  //	{
  //	host_price_col->division(random_number);
  //	dev_price_col->division(random_number);
  //	ColumnPtr host_res = gpu::copy_column_device_to_host(dev_price_col);
  //	if(!host_res->is_equal(host_price_col)) return false;
  //	}
  return true;
}

bool addConstantColumnTest() {
  TableSchema schema;
  schema.push_back(Attribut(INT, "Shop_ID"));
  schema.push_back(Attribut(INT, "Price"));
  schema.push_back(Attribut(INT, "Sales"));

  TablePtr tab1(new Table("Sale", schema));

  for (unsigned int i = 0; i < 10; i++) {
    Tuple t;
    t.push_back((int)i % 10);
    t.push_back((int)rand() % 100);
    t.push_back((int)(rand() % 1000) + 1);
    tab1->insert(t);
  }

  getGlobalTableList().push_back(tab1);
  ProcessorSpecification proc_spec(hype::PD0);
  TablePtr ptr = BaseTable::AddConstantValueColumnOperation(
      tab1, "my_constant_value_column", INT, int(5), proc_spec);
  ptr->print();
  ptr = BaseTable::AddConstantValueColumnOperation(
      ptr, "my_constant_value_column2", FLOAT, float(34.5), proc_spec);
  ptr->print();
  ptr = BaseTable::AddConstantValueColumnOperation(
      ptr, "my_constant_value_column3", VARCHAR, string("string"), proc_spec);
  ptr->print();
  getGlobalTableList().push_back(ptr);

  list<string> columns;
  columns.push_back("my_constant_value_column3");
  columns.push_back("Sales");
  columns.push_back("Price");
  columns.push_back("my_constant_value_column");
  columns.push_back("Shop_ID");
  columns.push_back("my_constant_value_column2");

  ptr = BaseTable::projection(ptr, columns, MATERIALIZE, CPU);
  ptr->print();

  int value = 22;
  boost::shared_ptr<logical_operator::Logical_Scan> scan_sale(
      new logical_operator::Logical_Scan("Sale"));
  boost::shared_ptr<logical_operator::Logical_AddConstantValueColumn>
      add_constant_value_column(
          new logical_operator::Logical_AddConstantValueColumn(
              "new_constant_column", INT, value));

  add_constant_value_column->setLeft(scan_sale);
  LogicalQueryPlan log_plan(add_constant_value_column);
  // log_plan.print();

  CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
      log_plan.convertToPhysicalQueryPlan();
  plan->run();
  if (!plan->getResult()) return false;
  plan->getResult()->print();

  //        if(!plan->getResult()->getColumnbyName("Add_Price_Value")) return
  //        false;
  //        if(!plan->getResult()->getColumnbyName("Add_Price_Value_LogOp"))
  //        return false;
  //        if(!plan->getResult()->getColumnbyName("Add_Price_Value")->is_equal(plan->getResult()->getColumnbyName("Add_Price_Value_LogOp"))){
  //            cerr << "TableBasedColumnAlgebraOperatorTest: ADD Test for
  //            Logical_ColumnConstantOperator Failed!" << endl;
  //            return false;
  //        }

  return true;
}

bool basicQueryTest() {
  //	//create test database
  // 	{
  //	TableSchema schema;
  //	schema.push_back(Attribut(INT,"Shop_ID"));
  //	schema.push_back(Attribut(INT,"Sales"));
  //
  //
  //	TablePtr tab1(new Table("Sale",schema));
  //	if(!tab1){
  //		cout << "Failed to create Table 'Sale'!" << endl;
  //		return false;
  //	}
  //
  //	{Tuple t; t.push_back(1); t.push_back(5); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //	{Tuple t; t.push_back(1); t.push_back(3); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //	{Tuple t; t.push_back(1); t.push_back(6); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //	{Tuple t; t.push_back(1); t.push_back(7); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //	{Tuple t; t.push_back(2); t.push_back(2); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //	{Tuple t; t.push_back(2); t.push_back(6); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //	{Tuple t; t.push_back(2); t.push_back(12); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //	{Tuple t; t.push_back(3); t.push_back(4); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //	{Tuple t; t.push_back(3); t.push_back(7); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //	{Tuple t; t.push_back(3); t.push_back(3); if(!tab1->insert(t)){ cout <<
  //"Failed to insert Tuple" << endl; return false;}}
  //
  //	tab1->print();
  //	getGlobalTableList().push_back(tab1);
  //
  //	}
  //	{
  //	TableSchema schema;
  //	schema.push_back(Attribut(INT,"ID"));
  //	schema.push_back(Attribut(VARCHAR,"Name"));
  //
  //	TablePtr tab1(new Table("Shop",schema));
  //	if(!tab1){
  //		cout << "Failed to create Table 'Shop'!" << endl;
  //		return false;
  //	}
  //	{Tuple t; t.push_back(1); t.push_back(std::string("Tante Emma Laden"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //	{Tuple t; t.push_back(2); t.push_back(std::string("Aldi"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //	{Tuple t; t.push_back(3); t.push_back(std::string("Kaufland"));
  // if(!tab1->insert(t)){ cout << "Failed to insert Tuple" << endl; return
  // false;}}
  //
  //	tab1->print();
  //
  //	getGlobalTableList().push_back(tab1);
  //
  //	}
  //	//create operators
  //	boost::shared_ptr<logical_operator::Logical_Scan>  scan_sales(new
  // logical_operator::Logical_Scan("Sale"));
  //
  //	boost::shared_ptr<logical_operator::Logical_Scan>  scan_shops(new
  // logical_operator::Logical_Scan("Shop"));
  //
  //	boost::shared_ptr<logical_operator::Logical_Selection>  selection(new
  // logical_operator::Logical_Selection("Sales",boost::any(6),LESSER));
  //
  //	boost::shared_ptr<logical_operator::Logical_Join>  join(new
  // logical_operator::Logical_Join("Shop_ID","ID"));
  //
  //	std::list<std::string> column_list;
  //	column_list.push_back("Name");
  //	column_list.push_back("Sales");
  //	boost::shared_ptr<logical_operator::Logical_Projection>  projection(new
  // logical_operator::Logical_Projection(column_list));
  //
  //
  //        std::list<std::string> sorting_columns;
  //        sorting_columns.push_back("Name");
  //	boost::shared_ptr<logical_operator::Logical_Sort>  sort(new
  // logical_operator::Logical_Sort(sorting_columns));
  //
  //        //sort->setLeft(column_algebra_operation_lineorder);
  //
  //        std::list<std::pair<string,AggregationMethod> >
  //        aggregation_functions;
  //        aggregation_functions.push_back(make_pair("Sales",SUM));
  //
  //	boost::shared_ptr<logical_operator::Logical_Groupby>  groupby(new
  // logical_operator::Logical_Groupby(sorting_columns,aggregation_functions));
  //
  //	//connect operators to logical query plan
  //	selection->setLeft(scan_sales);
  //
  //	join->setLeft(selection);
  //	join->setRight(scan_shops);
  //
  //	sort->setLeft(join);
  //
  //	projection->setLeft(sort);
  //
  //	groupby->setLeft(projection);
  //
  //	LogicalQueryPlan log_plan(groupby);
  //
  //	//pritn logical query plan
  //	log_plan.print();
  //
  //	//create physical plan from logical plan and execute it (several times
  // to first train decision model?!)
  //	for(unsigned int i=0;i<30;i++){
  //		PhysicalQueryPlanPtr plan =
  // log_plan.convertToPhysicalQueryPlan();
  //
  //		plan->print();
  //
  //		plan->run();
  //
  //		//typename PhysicalQueryPlan::Type
  //		TablePtr ptr = plan->getResult();
  //		ptr->print();
  //	}

  return true;
}

bool basicGPUacceleratedQueryTest() { return true; }

bool TableBasedColumnConstantOperatorTest() {
  // 	TableSchema schema;
  //	schema.push_back(Attribut(INT,"Shop_ID"));
  //	schema.push_back(Attribut(INT,"Price"));
  //	schema.push_back(Attribut(INT,"Sales"));
  //
  //	TablePtr tab1(new Table("Sale",schema));
  //
  //	for(unsigned int i=0;i<10;i++){
  //		Tuple t;
  //		t.push_back((int)i%10);
  //		t.push_back((int)rand()%100);
  //		t.push_back((int)(rand()%1000)+1);
  //		tab1->insert(t);
  //	}
  //        //storeTable(tab1);
  //        if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tab1->print();
  //        //getGlobalTableList().push_back(tab1);
  //        /*####################################################################################*/
  //        /*############# Operations between a Column and a scalar value
  //        #######################*/
  //        /*####################################################################################*/
  //
  //
  //        /******** Test ADDITION Column with constant value****************/
  //        {
  //            ColumnPtr col1 = tab1->getColumnbyName("Price");
  //            //ColumnPtr col2 = tab1->getColumnbyName("Sales");
  //            int value = 22;
  //            tab1=BaseTable::ColumnConstantOperation(tab1,"Price",value,"Add_Price_Value",ADD);
  //            if(!tab1) return false;
  //            //Validation
  //
  //            ColumnPtr tmp = col1->materialize();
  //            tmp->add(value);
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tab1->print();
  //            ColumnPtr result = tab1->getColumnbyName("Add_Price_Value");
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            result->print();
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tmp->print();
  //            if(!result->is_equal(tmp)){
  //               cerr << "TableBasedColumnConstantOperatorTest: ADD Test
  //               Failed!" << endl;
  //               return false;
  //            }
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //               tab1->print();
  //
  //        }
  //
  //        /******** Test Substraction Column with constant
  //        value****************/
  //        {
  //            ColumnPtr col1 = tab1->getColumnbyName("Price");
  //            //ColumnPtr col2 = tab1->getColumnbyName("Sales");
  //            int value = 22;
  //            tab1=BaseTable::ColumnConstantOperation(tab1,"Price",value,"Sub_Price_Value",SUB);
  //            if(!tab1) return false;
  //            //Validation
  //            ColumnPtr tmp = col1->materialize();
  //            tmp->minus(value);
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tab1->print();
  //            ColumnPtr result = tab1->getColumnbyName("Sub_Price_Value");
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            result->print();
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tmp->print();
  //            if(!result->is_equal(tmp)){
  //               cerr << "TableBasedColumnConstantOperatorTest: SUB Test
  //               Failed!" << endl;
  //               return false;
  //            }
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //               tab1->print();
  //        }
  //        /******** Test Multiplication Column with constant
  //        value****************/
  //        {
  //            ColumnPtr col1 = tab1->getColumnbyName("Price");
  //            //ColumnPtr col2 = tab1->getColumnbyName("Sales");
  //            int value = 22;
  //            tab1=BaseTable::ColumnConstantOperation(tab1,"Price",value,"Mul_Price_Value",MUL);
  //            if(!tab1) return false;
  //            //Validation
  //            ColumnPtr tmp = col1->materialize();
  //            tmp->multiply(value);
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tab1->print();
  //            ColumnPtr result = tab1->getColumnbyName("Mul_Price_Value");
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            result->print();
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tmp->print();
  //            if(!result->is_equal(tmp)){
  //               cerr << "TableBasedColumnConstantOperatorTest: MUL Test
  //               Failed!" << endl;
  //               return false;
  //            }
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //               tab1->print();
  //        }
  //        /******** Test Division Column with constant value****************/
  //        {
  //            ColumnPtr col1 = tab1->getColumnbyName("Price");
  //            //ColumnPtr col2 = tab1->getColumnbyName("Sales");
  //            int value = 2;
  //            tab1=BaseTable::ColumnConstantOperation(tab1,"Price",value,"Div_Price_Value",DIV);
  //            if(!tab1) return false;
  //            //Validation
  //            ColumnPtr tmp = col1->materialize();
  //            tmp->division(value);
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tab1->print();
  //            ColumnPtr result = tab1->getColumnbyName("Div_Price_Value");
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            result->print();
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tmp->print();
  //            if(!result->is_equal(tmp)){
  //               cerr << "TableBasedColumnConstantOperatorTest: DIV Test
  //               Failed!" << endl;
  //               return false;
  //            }
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //               tab1->print();
  //        }
  //        /*####################################################################################*/
  //        /*######################## Operations between two Columns
  //        ############################*/
  //        /*####################################################################################*/
  //
  //        /******** Test ADDITION Column with Column ****************/
  //        {
  //
  //            int value = 22;
  //            tab1=BaseTable::ColumnAlgebraOperation(tab1,"Price","Sales","Add_Price_Sales",ADD);
  //            if(!tab1) return false;
  //            //Validation
  //            ColumnPtr col1 = tab1->getColumnbyName("Price");
  //            ColumnPtr col2 = tab1->getColumnbyName("Sales");
  //            ColumnPtr tmp = col1->materialize();
  //            tmp->add(col2);
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tab1->print();
  //            ColumnPtr result = tab1->getColumnbyName("Add_Price_Sales");
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            result->print();
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tmp->print();
  //            if(!result->is_equal(tmp)){
  //               cerr << "TableBasedColumnAlgebraOperatorTest: ADD Test
  //               Failed!" << endl;
  //               return false;
  //            }
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //               tab1->print();
  //        }
  //
  //        /******** Test SUBSTRACTION Column with Column ****************/
  //        {
  //            int value = 22;
  //            tab1=BaseTable::ColumnAlgebraOperation(tab1,"Price","Sales","Sub_Price_Sales",SUB);
  //            if(!tab1) return false;
  //            //Validation
  //            ColumnPtr col1 = tab1->getColumnbyName("Price");
  //            ColumnPtr col2 = tab1->getColumnbyName("Sales");
  //            ColumnPtr tmp = col1->materialize();
  //            tmp->minus(col2);
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tab1->print();
  //            ColumnPtr result = tab1->getColumnbyName("Sub_Price_Sales");
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            result->print();
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tmp->print();
  //            if(!result->is_equal(tmp)){
  //               cerr << "TableBasedColumnAlgebraOperatorTest: SUB Test
  //               Failed!" << endl;
  //               return false;
  //            }
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //               tab1->print();
  //        }
  //
  //        /******** Test MULTIPLICATION Column with Column ****************/
  //        {
  //            int value = 22;
  //            tab1=BaseTable::ColumnAlgebraOperation(tab1,"Price","Sales","Mul_Price_Sales",MUL);
  //            if(!tab1) return false;
  //
  //            //Validation
  //            ColumnPtr col1 = tab1->getColumnbyName("Price");
  //            ColumnPtr col2 = tab1->getColumnbyName("Sales");
  //            ColumnPtr tmp = col1->materialize();
  //            tmp->multiply(col2);
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tab1->print();
  //            ColumnPtr result = tab1->getColumnbyName("Mul_Price_Sales");
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            result->print();
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tmp->print();
  //            if(!result->is_equal(tmp)){
  //               cerr << "TableBasedColumnAlgebraOperatorTest: MUL Test
  //               Failed!" << endl;
  //               return false;
  //            }
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //               tab1->print();
  //        }
  //
  //        /******** Test DIVISION Column with Column ****************/
  //        {
  //            int value = 22;
  //            tab1=BaseTable::ColumnAlgebraOperation(tab1,"Sales","Price","Div_Sales_Price",DIV);
  //            if(!tab1) return false;
  //            //Validation
  //            ColumnPtr col1 = tab1->getColumnbyName("Price");
  //            ColumnPtr col2 = tab1->getColumnbyName("Sales");
  //            ColumnPtr tmp = col2->materialize();
  //            tmp->division(col1);
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //                tab1->print();
  //            ColumnPtr result = tab1->getColumnbyName("Div_Sales_Price");
  //
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            result->print();
  //            if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //            tmp->print();
  //            if(!result->is_equal(tmp)){
  //               cerr << "TableBasedColumnAlgebraOperatorTest: DIV Test
  //               Failed!" << endl;
  //               return false;
  //            }
  //
  //        }
  //        if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug){
  //           tab1->print();
  //           cout << "add Table " << tab1->getName() << " to Tablelist " <<
  //           endl;
  //        }
  //        getGlobalTableList().push_back(tab1);
  //        /*####################################################################################*/
  //        /*######################## Test of Queryplan Operators
  //        ############################*/
  //        /*####################################################################################*/
  //        {
  //        int value = 22;
  //        boost::shared_ptr<logical_operator::Logical_Scan>  scan_sale(new
  //        logical_operator::Logical_Scan("lookup(Sale)"));
  //        boost::shared_ptr<logical_operator::Logical_ColumnConstantOperator>
  //        column_constant_operation_sale(new
  //        logical_operator::Logical_ColumnConstantOperator("Price",value,"Add_Price_Value_LogOp",ADD,hype::DeviceConstraint(hype::CPU_ONLY)));
  //
  //        column_constant_operation_sale->setLeft(scan_sale);
  //        LogicalQueryPlan log_plan(column_constant_operation_sale);
  //        //log_plan.print();
  //
  //        CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
  //        log_plan.convertToPhysicalQueryPlan();
  //        plan->run();
  //        if(!CoGaDB::quiet && CoGaDB::verbose && CoGaDB::debug)
  //           plan->getResult()->print();
  //        if(!plan->getResult()) return false;
  //        if(!plan->getResult()->getColumnbyName("Add_Price_Value")) return
  //        false;
  //        if(!plan->getResult()->getColumnbyName("Add_Price_Value_LogOp"))
  //        return false;
  //        if(!plan->getResult()->getColumnbyName("Add_Price_Value")->is_equal(plan->getResult()->getColumnbyName("Add_Price_Value_LogOp"))){
  //            cerr << "TableBasedColumnAlgebraOperatorTest: ADD Test for
  //            Logical_ColumnConstantOperator Failed!" << endl;
  //            return false;
  //        }
  //        }
  //
  //
  //        {
  //        boost::shared_ptr<logical_operator::Logical_Scan>  scan_sale(new
  //        logical_operator::Logical_Scan("lookup(Sale)"));
  //        boost::shared_ptr<logical_operator::Logical_ColumnAlgebraOperator>
  //        column_algebra_operation_sale(new
  //        logical_operator::Logical_ColumnAlgebraOperator("Price","Sales","Add_Price_Sales_LogOp",ADD,hype::DeviceConstraint(hype::CPU_ONLY)));
  //
  //        column_algebra_operation_sale->setLeft(scan_sale);
  //        LogicalQueryPlan log_plan(column_algebra_operation_sale);
  //        //log_plan.print();
  //
  //        CoGaDB::query_processing::PhysicalQueryPlanPtr plan =
  //        log_plan.convertToPhysicalQueryPlan();
  //        assert(plan!=NULL);
  //        plan->run();
  //        assert(plan->getResult()!=NULL);
  //        //plan->getResult()->print();
  //        assert(plan->getResult()->getColumnbyName("Add_Price_Sales")!=NULL);
  //        if(!plan->getResult()->getColumnbyName("Add_Price_Sales")->is_equal(plan->getResult()->getColumnbyName("Add_Price_Sales_LogOp"))){
  //            cerr << "TableBasedColumnAlgebraOperatorTest: ADD Test for
  //            Logical_ColumnConstantOperator Failed!" << endl;
  //            return false;
  //        }
  //        }

  return true;
}

bool ComplexSortTest() {
  TableSchema schema;
  schema.push_back(Attribut(INT, "YEAR"));
  schema.push_back(Attribut(VARCHAR, "Name"));
  schema.push_back(Attribut(INT, "Age"));
  schema.push_back(Attribut(FLOAT, "Score"));

  TablePtr tab1(new Table("Person", schema));

  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Peter"));
    t.push_back(24);
    t.push_back(float(23.1));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2011);
    t.push_back(string("James"));
    t.push_back(20);
    t.push_back(float(40.6));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Karl"));
    t.push_back(30);
    t.push_back(float(99.4));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2012);
    t.push_back(string("Max"));
    t.push_back(17);
    t.push_back(float(10.8));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2011);
    t.push_back(string("Bob"));
    t.push_back(21);
    t.push_back(float(84.2));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Alice"));
    t.push_back(22);
    t.push_back(float(30.6));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2012);
    t.push_back(string("Susanne"));
    t.push_back(25);
    t.push_back(float(29.3));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Joe"));
    t.push_back(40);
    t.push_back(float(72.0));
    tab1->insert(t);
  }
  //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  {
    Tuple t;
    t.push_back(2013);
    t.push_back(string("Alice"));
    t.push_back(33);
    t.push_back(float(95.4));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2012);
    t.push_back(string("Anne"));
    t.push_back(16);
    t.push_back(float(52.3));
    tab1->insert(t);
  }
  {
    Tuple t;
    t.push_back(2011);
    t.push_back(string("James"));
    t.push_back(9);
    t.push_back(float(5.0));
    tab1->insert(t);
  }
  std::list<std::string> column_names;
  column_names.push_back("YEAR");
  column_names.push_back("Name");
  // column_names.push_back("Score");

  TablePtr result = BaseTable::sort(tab1, column_names);

  tab1->print();
  result->print();
  std::list<std::pair<string, AggregationMethod> > aggregation_functions;
  aggregation_functions.push_back(make_pair("Score", SUM));
  aggregation_functions.push_back(make_pair("Age", SUM));
  result->print();

  return result != nullptr;
}

bool ColumnorientedQueryPlanTest() { return true; }

#ifdef ENABLE_SIMD_ACCELERATION
std::string toString(__m128i val) {
  std::string result;
  int __attribute__((aligned(16))) printBuf[4];
  _mm_store_ps((float*)printBuf, (__m128)val);
  result += printBuf[0];
  result += ",";
  result += printBuf[1];
  result += ",";
  result += printBuf[2];
  result += ",";
  result += printBuf[3];
  cout << printBuf[0] << "," << printBuf[1] << "," << printBuf[2] << ","
       << printBuf[3] << endl;
  return result;
}
#endif

#define COGADB_SIMD_SCAN_INT_OPTIMIZED(array, array_size, result_array,      \
                                       comparison_value,                     \
                                       SIMD_COMPARISON_FUNCTION,             \
                                       COMPARISON_OPERATOR, result_size)     \
  __m128i* sse_array = reinterpret_cast<__m128i*>(array);                    \
  assert(sse_array != NULL);                                                 \
  int alignment_offset = ((intptr_t)sse_array) % sizeof(__m128i);            \
  const int sse_array_length =                                               \
      (array_size - alignment_offset) * sizeof(int) / sizeof(__m128i);       \
  cout << "SSE Array Length: " << sse_array_length << endl;                  \
  char* tmp_array = (char*)sse_array;                                        \
  tmp_array += alignment_offset;                                             \
  sse_array = reinterpret_cast<__m128i*>(tmp_array);                         \
  cout << "array adress: " << (void*)array                                   \
       << "sse array: " << (void*)sse_array << endl;                         \
  cout << "First SSE Array Element: " << ((int*)sse_array)[0] << endl;       \
  unsigned int pos = 0;                                                      \
  __m128i comp_val = _mm_set1_epi32(comparison_value);                       \
  __m128i read_value = _mm_set1_epi32(0);                                    \
  cout << "alignment_offset " << alignment_offset << endl;                   \
  if (alignment_offset != 0) {                                               \
    cout << "process first unaligned data chunk: index 0 to "                \
         << alignment_offset << endl;                                        \
    for (unsigned int i = 0; i < alignment_offset / sizeof(int); i++) {      \
      if (simd_debug_mode) {                                                 \
        cout << "index " << i << endl;                                       \
        cout << "value " << array[i]                                         \
             << " match:" << (array[i] COMPARISON_OPERATOR comparison_value) \
             << endl;                                                        \
      }                                                                      \
      if (array[i] COMPARISON_OPERATOR comparison_value) {                   \
        result_array[pos++] = i;                                             \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  for (unsigned int i = 0; i < sse_array_length; i++) {                      \
    assert(((intptr_t)sse_array) % sizeof(__m128i) == 0);                    \
    read_value = _mm_load_si128(&sse_array[i]);                              \
    if (simd_debug_mode) {                                                   \
      cout << "index: " << i << endl;                                        \
      toString(read_value);                                                  \
    }                                                                        \
    __m128 comp_result =                                                     \
        (__m128)SIMD_COMPARISON_FUNCTION(read_value, comp_val);              \
    int mask = _mm_movemask_ps(comp_result);                                 \
    if (simd_debug_mode)                                                     \
      cout << "Mask: " << std::hex << mask << std::dec << endl;              \
    if (mask) {                                                              \
      if (simd_debug_mode) cout << "at least one match!" << endl;            \
      for (unsigned j = 0; j < sizeof(__m128i) / sizeof(int); ++j) {         \
        if (simd_debug_mode)                                                 \
          cout << "sub index: " << j << " value: " << ((mask >> j) & 1)      \
               << endl;                                                      \
        if ((mask >> j) & 1) {                                               \
          result_array[pos++] = i * (sizeof(__m128i) / sizeof(int)) + j +    \
                                (alignment_offset / sizeof(int));            \
        }                                                                    \
      }                                                                      \
    }                                                                        \
  }                                                                          \
  cout << "Remaining offsets: "                                              \
       << (sse_array_length * sizeof(__m128i) / sizeof(int)) +               \
              alignment_offset                                               \
       << " to " << array_size << endl;                                      \
  for (unsigned int i = (sse_array_length * sizeof(__m128i) / sizeof(int)) + \
                        (alignment_offset / sizeof(int));                    \
       i < array_size; i++) {                                                \
    if (array[i] COMPARISON_OPERATOR comparison_value) {                     \
      result_array[pos++] = i;                                               \
    }                                                                        \
  }                                                                          \
  result_size = pos;

bool test() {
#ifdef ENABLE_SIMD_ACCELERATION
  const bool simd_debug_mode = true;
  unsigned int array_size = 1001;
  int* array = new int[array_size];
  int comparison_value = 50;
  int result_count = 0;
  for (unsigned int i = 0; i < array_size; i++) {
    array[i] = rand() % 200;  // rand()%6;//rand()%20;
    if (array[i] < comparison_value) {
      result_count++;
    }
    cout << array[i] << endl;
  }
  int* result_array = new int[array_size];
  unsigned int result_size = 0;
  // COGADB_SIMD_SCAN_INT(array,array_size,result_array,comparison_value,_mm_cmpeq_epi32,==,result_size);
  COGADB_SIMD_SCAN_INT_OPTIMIZED(array, array_size, result_array,
                                 comparison_value, _mm_cmplt_epi32, <,
                                 result_size);

  cout << "Result Size: " << result_size << " (Should be " << result_count
       << ")" << endl;

  for (unsigned int i = 0; i < result_size; i++) {
    cout << "index: " << result_array[i] << " value: " << array[result_array[i]]
         << endl;
  }

#endif

  return true;
}

bool SIMD_selection() {
#ifdef ENABLE_SIMD_ACCELERATION
  //        test();
  //        exit(0);

  const bool simd_debug_mode = true;  // false;

  unsigned int array_size = 1001;
  int* array = new int[array_size];
  int comparison_value = 50;
  int result_count = 0;
  for (unsigned int i = 0; i < array_size; i++) {
    array[i] = rand() % 200;  // rand()%6;//rand()%20;
    if (array[i] == comparison_value) result_count++;
    cout << array[i] << endl;
  }

  __m128i* sse_array =
      reinterpret_cast<__m128i*>(array);  // Intermediate pointer
  assert(sse_array != NULL);
  //__m128 x = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);  // Initialize x to
  //<1,2,3,4>.
  //__m128 xDelta = _mm_set1_ps(4.0f);  // Set the xDelta to <4,4,4,4>.

  int* result_array = new int[array_size];

  if (!quiet && verbose && debug) cout << "array_size: " << array_size << endl;

  // ensure the SIMD loop processes 16 Byte aligned memory
  int alignment_offset = ((intptr_t)sse_array) % sizeof(__m128i);
  const int sse_array_length =
      (array_size - alignment_offset) * sizeof(int) / sizeof(__m128i);
  // const int sse_array_length =
  // ((((array_size*sizeof(int))-(sizeof(__m128i)-alignment_offset))/sizeof(__m128i))*sizeof(__m128i))/sizeof(__m128i);
  // //((array_size*sizeof(int))-alignment_offset)/sizeof(__m128i);
  // const int sse_array_length =
  // ((((array_size*sizeof(int))/sizeof(__m128i))*sizeof(__m128i))-(sizeof(__m128i)-alignment_offset))/sizeof(__m128i);
  char* tmp_array = (char*)sse_array;
  tmp_array += alignment_offset;
  sse_array = reinterpret_cast<__m128i*>(tmp_array);
  // for(unsigned int
  // i=0;i<sse_array_length-(sizeof(__m128i)-1);i+=sizeof(__m128i)){
  if (simd_debug_mode) cout << "SSE Array Length: " << sse_array_length << endl;

  // index in result buffer
  unsigned int pos = 0;
  // load comparison value in SSE register
  __m128i comp_val = _mm_set1_epi32(comparison_value);
  // read value will contain the data read from main memory
  __m128i read_value =
      _mm_set1_epi32(0);  //_mm_set_epi64(uint64_t(0), uint64_t(0));

  cout << "alignment_offset " << alignment_offset << endl;
  if (alignment_offset != 0) {
    cout << "process first unaligned data chunk: index 0 to "
         << alignment_offset << endl;
    for (unsigned int i = 0; i < alignment_offset; i++) {
      if (simd_debug_mode) {
        cout << "index " << i << endl;
        cout << "value " << array[i] << endl;
      }
      if (array[i] == comparison_value) {
        result_array[pos++] = i;
      }
    }
  }

  for (unsigned int i = 0; i < sse_array_length; i++) {
    // read_value=_mm_loadu_si128(&sse_array[i]);
    // assure that memory is 16 Byte aligned
    assert(((intptr_t)sse_array) % sizeof(__m128i) == 0);
    // read from 16 byte aligned memory
    read_value = _mm_load_si128(&sse_array[i]);
    if (simd_debug_mode) {
      cout << "index: " << i << endl;
      cout << toString(read_value) << endl;
      cout << "array: " << array[i * sizeof(__m128i) + alignment_offset] << ","
           << array[i * sizeof(__m128i) + 1 + alignment_offset] << ","
           << array[i * sizeof(__m128i) + 2 + alignment_offset] << ","
           << array[i * sizeof(__m128i) + 3 + alignment_offset] << endl;
    }
    __m128 comp_result = (__m128)_mm_cmpeq_epi32(
        read_value, comp_val);  //_mm_cmpeq_ps for floats!
    /*comp_result is number between 0 and 2^sizeof(__m128i)*/
    // get bitmask of length 4, where each bit states the most significant bit
    // for the 4 integers
    int mask = _mm_movemask_ps(comp_result);

    if (simd_debug_mode)
      cout << "Mask: " << std::hex << mask << std::dec << endl;

    if (mask) {
      if (simd_debug_mode) cout << "at least one match!" << endl;

      for (unsigned j = 0; j < sizeof(__m128i) / sizeof(int); ++j) {
        if (simd_debug_mode)
          cout << "sub index: " << j << " value: " << ((mask >> j) & 1) << endl;

        if ((mask >> j) & 1) { /*jth bit*/
          result_array[pos++] =
              i * (sizeof(__m128i) / sizeof(int)) + j + alignment_offset;
        }
      }
    }
  }
  if (simd_debug_mode)
    // cout << "Remaining offsets: " <<
    // ((((array_size*sizeof(int))-alignment_offset)/sizeof(__m128i))*sizeof(__m128i))/sizeof(int)
    // << " to " << array_size << endl;
    cout << "Remaining offsets: "
         << (sse_array_length * sizeof(__m128i) / sizeof(int)) +
                alignment_offset
         << " to " << array_size << endl;
  // processed the remainining part of the array
  // for(unsigned int
  // i=((((array_size*sizeof(int))-alignment_offset)/sizeof(__m128i))*sizeof(__m128i))/sizeof(int);i<array_size;i++){
  for (unsigned int i = (sse_array_length * sizeof(__m128i) / sizeof(int)) +
                        alignment_offset;
       i < array_size; i++) {
    if (simd_debug_mode) {
      cout << "index " << i << endl;
      cout << "value " << array[i] << endl;
    }
    if (array[i] == comparison_value) {
      result_array[pos++] = i;
    }
  }

  cout << "Result Size: " << pos << " (Should be " << result_count << ")"
       << endl;

  for (unsigned int i = 0; i < pos; i++) {
    cout << result_array[i] << endl;
  }

  delete array;
  delete result_array;

  exit(0);
#endif
  // for(unsigned int
  // i=sse_array_length-(sizeof(__m128i)-1);;i+=sizeof(__m128i)){
  return true;
}

bool memory_benchmark() {
  // get memory information in linux
  // sudo dmidecode --type 17
  // SIMD_selection();

  std::cout << "starting memory benchmark:" << endl;

  //	std::vector<uint64_t> v()
  uint64_t one_gb = 1024 * 1024 * 1024;
  uint64_t array_size = one_gb / sizeof(uint64_t);
  uint64_t* array = new uint64_t[array_size];
  // uint64_t* array= // (uint64_t*) malloc (sizeof(uint64_t)*array_size);
  std::cout << "fill temp buffer:" << endl;
  for (unsigned int i = 0; i < array_size; ++i) {
    array[i] = rand() % 1000;
  }

  std::cout << "run read memory benchmark:" << endl;
  {
    Timestamp begin = getTimestamp();
    volatile uint64_t read_value = 0;
    for (unsigned int i = 0; i < array_size; ++i) {
      read_value = array[i];
    }
    Timestamp end = getTimestamp();
    assert(end > begin);

    cout << "Time: " << double(end - begin) / (1000 * 1000) << "ms" << endl;
    double time_in_seconds = double(end - begin) / (1000 * 1000 * 1000);
    cout << "Bandwidth: " << 1 / time_in_seconds << "GB/s" << endl;
  }

  std::cout << "run write memory benchmark:" << endl;
  {
    Timestamp begin = getTimestamp();
    // volatile uint64_t read_value=0;
    for (unsigned int i = 0; i < array_size; ++i) {
      array[i] = 26;
    }
    Timestamp end = getTimestamp();
    assert(end > begin);

    cout << "Time: " << double(end - begin) / (1000 * 1000) << "ms" << endl;
    double time_in_seconds = double(end - begin) / (1000 * 1000 * 1000);
    cout << "Bandwidth: " << 1 / time_in_seconds << "GB/s" << endl;
  }

#ifdef __SSE2__
  std::cout << "run read memory benchmark (SSE):" << endl;
  {
    Timestamp begin = getTimestamp();
    volatile __m128i read_value;
    read_value = _mm_set1_epi32(0);  //_mm_set_epi64(uint64_t(0), uint64_t(0));

    __m128i* sse_array =
        reinterpret_cast<__m128i*>(array);  // Intermediate pointer
    assert(sse_array != NULL);
    //__m128 x = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);  // Initialize x to
    //<1,2,3,4>.
    //__m128 xDelta = _mm_set1_ps(4.0f);  // Set the xDelta to <4,4,4,4>.

    const auto sse_array_length =
        one_gb / sizeof(__m128i);  // array_size/sizeof(uint64_t);
    for (unsigned int i = 0; i < sse_array_length; ++i) {
      // if(i%10000) cout << i << endl;
      // read_value=sse_array[i];
      // v1 = _mm_loadu_si128((__m128i *)&sse_array[k]);
      read_value = _mm_loadu_si128(sse_array);
    }
    Timestamp end = getTimestamp();
    assert(end > begin);

    cout << "Time: " << double(end - begin) / (1000 * 1000) << "ms" << endl;
    double time_in_seconds = double(end - begin) / (1000 * 1000 * 1000);
    cout << "Bandwidth: " << 1 / time_in_seconds << "GB/s" << endl;
  }

  std::cout << "run write memory benchmark (SSE):" << endl;
  {
    Timestamp begin = getTimestamp();
    volatile __m128i write_value;
    write_value =
        _mm_set1_epi32(26);  //_mm_set_epi64(uint64_t(26), uint64_t(26));

    __m128i* sse_array =
        reinterpret_cast<__m128i*>(array);  // Intermediate pointer
    assert(sse_array != NULL);
    //__m128 x = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);  // Initialize x to
    //<1,2,3,4>.
    //__m128 xDelta = _mm_set1_ps(4.0f);  // Set the xDelta to <4,4,4,4>.

    const auto sse_array_length =
        one_gb / sizeof(__m128i);  // array_size/sizeof(uint64_t);
    for (unsigned int i = 0; i < sse_array_length; ++i) {
      //_mm_store_si128(&sse_array[i],write_value); //assumes 16 byte aligned
      // memory
      _mm_storeu_si128(&sse_array[i],
                       write_value);  // does not assume 16 byte aligned memory
      // sse_array[i]=write_value;
    }
    Timestamp end = getTimestamp();
    assert(end > begin);

    cout << "Time: " << double(end - begin) / (1000 * 1000) << "ms" << endl;
    double time_in_seconds = double(end - begin) / (1000 * 1000 * 1000);
    cout << "Bandwidth: " << 1 / time_in_seconds << "GB/s" << endl;
  }
#endif

  delete[] array;

  return true;
}
}

}  // end namespace CogaDB
