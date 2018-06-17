#pragma once
#include <core/table.hpp>
#include <iostream>
#include <persistence/storage_manager.hpp>
#include <util/time_measurement.hpp>
// Unittests

//#include <gpu/gpu_algorithms.hpp>

#include <lookup_table/lookup_table.hpp>
#include <util/tpch_benchmark.hpp>

namespace CoGaDB {

  namespace unit_tests {

    bool generalJoinTest();

    bool joinTest();

    bool executeUnitTests(ClientPtr client);

    bool fillTableTest();

    bool updateTuplesTest();

    bool basicOperationTest();

    bool selectionTest();

    bool joinTest();

    bool JoinPerformanceTest();

    bool crossjoinTest();

    bool complexSelectionTest();

    bool ColumnComputationTest();

    bool TableBasedColumnConstantOperatorTest();

    bool ComplexSortTest();

    bool basicQueryTest();

    bool addConstantColumnTest();

    bool GPUColumnComputationTest();

    bool basicGPUacceleratedQueryTest();

    bool ColumnorientedQueryPlanTest();

    bool memory_benchmark();

    bool compressioned_columns_tests();

    bool GPU_accelerated_scans();

    bool PositionListTest();

    bool primitives_unittests();

    bool renameColumnTest();

    bool PrimaryForeignKeyTest();

    // CDK unitests
    bool cdk_gather_test();
    bool cdk_selection_tid_test();
    bool cdk_selection_bitmap_test();
    bool cdk_join_test();
    bool cdk_join_performance_test();
    bool cdk_invisiblejoin_test();

#ifdef ENABLE_GPU_ACCELERATION
    // GPU work unittests
    bool testBinningKernel();
    bool testAllocationAndTransfer();
    bool gpu_work_kernels_test();
    bool experiment1();
    bool experiment2();
    bool experiment3();
    bool experiment4();
    bool experiment5();
    bool experiment6();
#endif

    bool cdk_selection_performance_test();
    bool cdk_unrolling_performance_test();
    bool cdk_selection_performance_test_float();
    bool cdk_unrolling_performance_test_float();
    bool cdk_selection_bitmap_performance_test();

    bool main_memory_join_tests();
    bool hash_join_benchmark();

    bool equi_width_histogram_test();
    bool equi_width_histogram_range_test();
  }

  // inline bool LoadDatabase(){
  //
  //	return loadTables();
  //
  //}
}

// inline bool Unittest_Fill_Tables_with_Data(){

//	TableSchema schema;
//	schema.push_back(Attribut(INT,"SID"));
//	schema.push_back(Attribut(VARCHAR,"Studiengang"));
//
//	TablePtr tab1(new Table("Studiengänge",schema));

//
//	{Tuple t; t.push_back(1); t.push_back(string("INF"));
// if(!tab1->insert(t)) cout << "Failed to insert Tuple" << endl;}
//	{Tuple t; t.push_back(2); t.push_back(string("CV"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(3); t.push_back(string("CSE"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(4); t.push_back(string("WIF"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(5); t.push_back(string("INF Fernst."));
// tab1->insert(t);}
//	{Tuple t; t.push_back(6); t.push_back(string("CV Master"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(7); t.push_back(string("INGINF"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(8); t.push_back(string("Lehramt"));
// tab1->insert(t);}
//
////1 INF

////2 CV
////4 CSE
////3 WIF
////5 INF Fernst.
////6 CV Master
////7 INGINF
////8 Lehramt

//	TableSchema schema2;
//	schema2.push_back(Attribut(VARCHAR,"Name"));
//	schema2.push_back(Attribut(INT,"MatrikelNr."));
//	schema2.push_back(Attribut(INT,"SID"));
//
//	TablePtr tab2(new Table("Studenten",schema2));

//	{Tuple t; t.push_back(string("Tom"));
// t.push_back(15487); t.push_back(3); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Anja"));
// t.push_back(12341); t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Maria"));		t.push_back(19522);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Jan"));
// t.push_back(11241); t.push_back(2); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Julia"));		t.push_back(19541);
// t.push_back(7); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Chris"));		t.push_back(13211);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Norbert"));
// t.push_back(19422);
// t.push_back(2); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Maria"));		t.push_back(11875);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Marko"));		t.push_back(13487);
// t.push_back(4); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Ingolf"));
// t.push_back(14267);
// t.push_back(2); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Susanne"));
// t.push_back(16755);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Stefanie"));	t.push_back(19774);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Jan"));
// t.push_back(13254); t.push_back(2); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Klaus"));		t.push_back(13324);
// t.push_back(3); tab2->insert(t);}
//	//{Tuple t; t.push_back(string(""));		t.push_back();
// t.push_back(); tab2->insert(t);}

//	cout << "initial tables: " << endl;
//	tab1->print();
//	tab2->print();

//	return true;
//}

// inline bool Unittest_GPU_Queries(){

//	TableSchema schema;
//	schema.push_back(Attribut(INT,"PID"));
//	schema.push_back(Attribut(VARCHAR,"Name"));
//	schema.push_back(Attribut(INT,"Age"));
//	schema.push_back(Attribut(FLOAT,"Score"));
//
//	TablePtr tab1(new Table("Person",schema));

//	{Tuple t; t.push_back(1); t.push_back(string("Peter")); t.push_back(24);
// t.push_back(float(23.1));				tab1->insert(t);}
//	{Tuple t; t.push_back(2); t.push_back(string("James"));
// t.push_back(20);	t.push_back(float(40.6));
// tab1->insert(t);}
//	{Tuple t; t.push_back(3); t.push_back(string("Karl")); t.push_back(30);
// t.push_back(float(99.4));			tab1->insert(t);}
//	{Tuple t; t.push_back(4); t.push_back(string("Max"));
// t.push_back(17);	t.push_back(float(10.8)); tab1->insert(t);}
//	{Tuple t; t.push_back(5); t.push_back(string("Bob")); t.push_back(21);
// t.push_back(float(84.2));	tab1->insert(t);}
//	{Tuple t; t.push_back(6); t.push_back(string("Alice")); t.push_back(22);
// t.push_back(float(30.6));	tab1->insert(t);}
//	{Tuple t; t.push_back(7); t.push_back(string("Susanne"));
// t.push_back(25);	t.push_back(float(29.3));	tab1->insert(t);}
//	{Tuple t; t.push_back(8); t.push_back(string("Joe"));
// t.push_back(40);	t.push_back(float(72.0)); tab1->insert(t);}
////%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

//	{Tuple t; t.push_back(9); t.push_back(string("Maik")); t.push_back(33);
// t.push_back(float(95.4));	tab1->insert(t);}
//	{Tuple t; t.push_back(10); t.push_back(string("Anne")); t.push_back(16);
// t.push_back(float(52.3));	tab1->insert(t);}
//	{Tuple t; t.push_back(11); t.push_back(string("Jenny")); t.push_back(9);
// t.push_back(float(5.0));	tab1->insert(t);}
//
//	tab1->print();

////	TablePtr
/// tab2=BaseTable::selection(tab1,string("Age"),int(24),LESSER,MATERIALIZE,GPU);
////	tab2->print();

//	{ //Query 1
//	ColumnPtr col = tab1->getColumnbyName(string("Age"));

//	gpu::GPU_Base_ColumnPtr dev_col = gpu::copy_column_host_to_device(col);
//	dev_col->print();
//

//	gpu::GPU_PositionlistPtr gpu_result_tids =
// gpu::GPU_Operators::selection(dev_col, int(24), LESSER);
//	assert(gpu_result_tids!=NULL);
//
////	gpu::GPU_Base_ColumnPtr dev_col_2 =
/// gpu::materializeColumn(dev_col,gpu_result_tids); //create materialized GPU
/// column from existing GPU Column and result tid list
////	gpu_result_tids->print();
//	cout << endl << "Materialize..." << endl;
//	gpu::GPU_Base_ColumnPtr dev_col_2 =
// dev_col->materialize(gpu_result_tids); //create materialized GPU column from
// existing GPU Column and result tid list
//	dev_col_2->print();
//	cout << endl << "Perform Sort..." << endl;
//	gpu::GPU_PositionlistPtr gpu_result_tids2 = gpu::GPU_Operators::sort(
// dev_col_2, ASCENDING);  //perform next operation
//	gpu_result_tids2->print();
//	cout << endl << "Aggregate Lookup Tables..." << endl;
//	gpu::GPU_PositionlistPtr gpu_aggregated_result_positionlist =
// gpu_result_tids->aggregate(gpu_result_tids2);
//	gpu_aggregated_result_positionlist->print();
//	PositionListPtr result_tids =
// copy_PositionList_device_to_host(gpu_aggregated_result_positionlist);
//	assert(result_tids!=NULL);

//	TablePtr tab2 =
// createLookupTableforUnaryOperation(std::string("Lookup_selection_and_sort(")+tab1->getName()+")",tab1,
// result_tids );
//	assert(tab2!=NULL);
//	tab2->print();
//	}
//
//
//
//		{ //Query 2
//	ColumnPtr col = tab1->getColumnbyName(string("Age"));

//	gpu::GPU_Base_ColumnPtr dev_col = gpu::copy_column_host_to_device(col);
//	dev_col->print();
//

//	gpu::GPU_PositionlistPtr gpu_result_tids =
// gpu::GPU_Operators::selection(dev_col, int(24), LESSER);
//	assert(gpu_result_tids!=NULL);
//
//	gpu::GPU_Base_ColumnPtr dev_col_2 =
// gpu::materializeColumn(dev_col,gpu_result_tids); //create materialized GPU
// column from existing GPU Column and result tid list
//	gpu_result_tids->print();
//	cout << endl << "Materialize..." << endl;
//
//	ColumnPtr col2 = tab1->getColumnbyName(string("Score"));
//	gpu::GPU_Base_ColumnPtr dev_col3 =
// gpu::copy_column_host_to_device(col2);
//
//	gpu::GPU_Base_ColumnPtr dev_col_4 =
// dev_col3->materialize(gpu_result_tids); //create materialized GPU column from
// existing GPU Column and result tid list
//
//	dev_col_4->print();
//	cout << endl << "Perform Sort..." << endl;
//	gpu::GPU_PositionlistPtr gpu_result_tids2 = gpu::GPU_Operators::sort(
// dev_col_4, ASCENDING);  //perform next operation
//	gpu_result_tids2->print();
//	cout << endl << "Aggregate Lookup Tables..." << endl;
//	gpu::GPU_PositionlistPtr gpu_aggregated_result_positionlist =
// gpu_result_tids->aggregate(gpu_result_tids2);
//	gpu_aggregated_result_positionlist->print();
//	PositionListPtr result_tids =
// copy_PositionList_device_to_host(gpu_aggregated_result_positionlist);
//	assert(result_tids!=NULL);

//	TablePtr tab2 =
// createLookupTableforUnaryOperation(std::string("Lookup_selection_and_sort(")+tab1->getName()+")",tab1,
// result_tids );
//	assert(tab2!=NULL);
//	tab2->print();
//	}
//
//
//
//
//
////		const ColumnPtr copy_column_device_to_host(GPU_Base_ColumnPtr
/// device_column);
////		const GPU_Base_ColumnPtr copy_column_host_to_device(ColumnPtr
/// host_column);

////		const PositionListPtr
/// copy_PositionList_device_to_host(GPU_PositionlistPtr device_pos_list);
////		const GPU_PositionlistPtr
/// copy_PositionList_host_to_device(PositionListPtr pos_list);

////			static const GPU_PositionlistPtr sort(GPU_Base_ColumnPtr
/// column);
////			static const GPU_PositionlistPtr
/// selection(GPU_Base_ColumnPtr
/// column, const boost::any& value_for_comparison, const ValueComparator comp);
////			//join algorithms
//////			static const
/// std::pair<GPU_Positionlist*,GPU_Positionlist*>
/// gpu_join(GPU_Base_Column* join_column1, GPU_Base_Column* join_column2,
/// JoinAlgorithm join_alg);
////			static const
/// std::pair<GPU_Base_ColumnPtr,GPU_Base_ColumnPtr>
/// groupby(GPU_Base_ColumnPtr grouping_column,
////
/// GPU_Base_ColumnPtr aggregation_column,
////
/// AggregationMethod agg_meth=SUM,
////
/// ComputeDevice comp_dev=CPU);

////			static const GPU_Base_ColumnPtr
/// materializeIntermediateResult(GPU_Base_ColumnPtr column, GPU_PositionlistPtr
/// pos_list);

////			static const
/// std::pair<GPU_PositionlistPtr,GPU_PositionlistPtr>
/// hash_join(GPU_Base_ColumnPtr join_column1, GPU_Base_ColumnPtr join_column2);
////			static const
/// std::pair<GPU_PositionlistPtr,GPU_PositionlistPtr>
/// sort_merge_join(GPU_Base_ColumnPtr join_column1, GPU_Base_ColumnPtr
/// join_column2);
////			static const
/// std::pair<GPU_PositionlistPtr,GPU_PositionlistPtr>
/// nested_loop_join(GPU_Base_ColumnPtr join_column1, GPU_Base_ColumnPtr
/// join_column2);

////	gpu::GPU_Positionlist* gpu_result_tids =
/// gpu::GPU_Operators::selection(gpu::GPU_Base_Column* column, int(24),
/// LESSER);

////	TablePtr
/// tab3=BaseTable::selection(tab1,string("Age"),int(24),LESSER,MATERIALIZE,GPU);
////	tab3->print();

//	return true;
//}

// vector<int> generate_dataset(unsigned int size_in_number_of_elements,
// unsigned int maximal_value_size){
//	vector<int> data;2
//	for(unsigned int i=0;i<size_in_number_of_elements;i++)
//		data.push_back(rand()%maximal_value_size);
//	return data;
//}

// TablePtr selectionCPU(TablePtr tab, int selection_value, ValueComparator
// comp){

//
//	return
// BaseTable::selection(tab,string("Number"),selection_value,comp,LOOKUP,SERIAL);

//}

// TablePtr selectionCPU_parallel(TablePtr tab, int selection_value,
// ValueComparator comp){

//
//	return
// BaseTable::selection(tab,string("Number"),selection_value,comp,LOOKUP,PARALLEL);

//}

// TablePtr selectionGPU(TablePtr tab, int selection_value, ValueComparator
// comp){

//	ColumnPtr col = tab->getColumnbyName(string("Number"));
//	assert(col!=NULL);
//	gpu::GPU_Base_ColumnPtr dev_col = gpu::copy_column_host_to_device(col);
//	assert(dev_col!=NULL);
//
//	gpu::GPU_PositionlistPtr gpu_result_tids =
// gpu::GPU_Operators::selection(dev_col, selection_value, comp);
//	assert(gpu_result_tids!=NULL);
//
//	PositionListPtr result_tids =
// copy_PositionList_device_to_host(gpu_result_tids);
//	assert(result_tids!=NULL);

//	return
// createLookupTableforUnaryOperation(std::string("Lookup_selection(")+tab->getName()+")",tab,
// result_tids );
//
//}

// bool Unittest_GPU_Selection_Benchmark(unsigned int number_of_runs,
// std::string logfilename){
//	srand(0);
//	//const unsigned int MAX_DATASET_SIZE=(10*1000*1000)/sizeof(int);
////1000000;
//	const unsigned int MAX_DATASET_SIZE=(50*1024*1024)/sizeof(int);
////(50*1000*1000)/sizeof(int); //1000000;
//	const unsigned int MAX_VALUE_SIZE=1000*1000;
//	vector<int> dataset;
//	int selection_value;
//	int selection_comparison_value; //0 EQUAL, 1 LESSER, 2 LARGER
//	//fstream file("selection.data",ios::out);
//	fstream file(logfilename.c_str(),ios::out);
//	unsigned int warmupcounter=0;
//	const unsigned int length_of_warmup_phase=2;
//	for(unsigned int i=0;i<number_of_runs+length_of_warmup_phase;i++){
//		selection_value = rand()%MAX_VALUE_SIZE;
//		selection_comparison_value = rand()%3;
//		dataset=generate_dataset(rand()%MAX_DATASET_SIZE,MAX_VALUE_SIZE);
//		TableSchema schema;
//		schema.push_back(Attribut(INT,"Number"));
//
//		cout << "Run: " << i << "/" << number_of_runs << "   Elements to
// Process: " << dataset.size() <<  endl;

//		TablePtr tab1(new Table("Test_Table",schema));
//		for(unsigned int j=0;j<dataset.size();j++){
//			Tuple t;
//			t.push_back(dataset[j]);
//			tab1->insert(t);
//		}
//		ValueComparator comp;
//		if(selection_comparison_value==0){
//			comp=EQUAL;
//		}else if(selection_comparison_value==1){
//			comp=LESSER;
//		}else if(selection_comparison_value==2){
//			comp=GREATER;
//		}else{
//			assert(true==false);
//		}
//
//
//		uint64_t begin_cpu = getTimestamp();
//		TablePtr tab_cpu = selectionCPU(tab1, selection_value, comp);
//		uint64_t end_cpu = getTimestamp();
//		assert(tab_cpu!=NULL);
//		assert(end_cpu>begin_cpu);
//

//		uint64_t begin_cpu_parallel = getTimestamp();
//		TablePtr tab_cpu_parallel = selectionCPU_parallel(tab1,
// selection_value, comp);
//		uint64_t end_cpu_parallel = getTimestamp();
//		assert(tab_cpu_parallel!=NULL);
//		assert(end_cpu_parallel>begin_cpu_parallel);

//		uint64_t begin_gpu = getTimestamp();
//		TablePtr tab_gpu = selectionGPU(tab1, selection_value, comp);
//		uint64_t end_gpu = getTimestamp();
//		assert(tab_gpu!=NULL);
//		assert(end_gpu>begin_gpu);

//		//tab_cpu->print();
//		//tab_cpu_parallel->print();
//		//tab_gpu->print();
//		//std::string s;
//		//cin >> s;

//
//		ColumnPtr col_cpu = tab_cpu -> getColumnbyName("Number");
//		ColumnPtr col_cpu_parallel = tab_cpu_parallel ->
// getColumnbyName("Number");
//		ColumnPtr col_gpu = tab_gpu -> getColumnbyName("Number");
//		assert(col_cpu->size()==col_gpu->size());
//		assert(col_cpu_parallel->size()==col_gpu->size());
//
//		double selectivity = double(col_cpu->size())/dataset.size();

//		if(warmupcounter++>=length_of_warmup_phase)
//		file << dataset.size() << "	" << selectivity << "	" <<
// comp
//		     << "	" << end_cpu - begin_cpu
//		     << "	" << end_gpu-begin_gpu
//		     << "	" << end_cpu_parallel - begin_cpu_parallel
//		     << endl;
//		cout << "number of elements: " << dataset.size() << "
// selectivity: " <<  selectivity  << "	comparison_value: "  << comp
//		     << "	CPU: " << end_cpu - begin_cpu
//		     << "	GPU: " << end_gpu-begin_gpu
//		     << "	CPU parallel: " << end_cpu_parallel -
// begin_cpu_parallel
//		     << endl;

//	}
//	return false;
//}

// inline bool Unittest_Basic_Query_Test(){

//	TableSchema schema;
//	schema.push_back(Attribut(INT,"SID"));
//	schema.push_back(Attribut(VARCHAR,"Studiengang"));
//
//	TablePtr tab1(new Table("Studiengänge",schema));

//	{Tuple t; t.push_back(1); t.push_back(string("INF"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(2); t.push_back(string("CV"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(3); t.push_back(string("CSE"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(4); t.push_back(string("WIF"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(5); t.push_back(string("INF Fernst."));
// tab1->insert(t);}
//	{Tuple t; t.push_back(6); t.push_back(string("CV Master"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(7); t.push_back(string("INGINF"));
// tab1->insert(t);}
//	{Tuple t; t.push_back(8); t.push_back(string("Lehramt"));
// tab1->insert(t);}

//	TableSchema schema2;
//	schema2.push_back(Attribut(VARCHAR,"Name"));
//	schema2.push_back(Attribut(INT,"MatrikelNr."));
//	schema2.push_back(Attribut(INT,"SID"));
//
//	TablePtr tab2(new Table("Studenten",schema2));

//	{Tuple t; t.push_back(string("Tom"));
// t.push_back(15487); t.push_back(3); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Anja"));
// t.push_back(12341); t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Maria"));		t.push_back(19522);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Jan"));
// t.push_back(11241); t.push_back(2); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Julia"));		t.push_back(19541);
// t.push_back(7); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Chris"));		t.push_back(13211);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Norbert"));
// t.push_back(19422);
// t.push_back(2); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Maria"));		t.push_back(11875);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Marko"));		t.push_back(13487);
// t.push_back(4); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Ingolf"));
// t.push_back(14267);
// t.push_back(2); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Susanne"));
// t.push_back(16755);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Stefanie"));	t.push_back(19774);
// t.push_back(1); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Jan"));
// t.push_back(13254); t.push_back(2); tab2->insert(t);}
//	{Tuple t; t.push_back(string("Klaus"));		t.push_back(13324);
// t.push_back(3); tab2->insert(t);}

//	cout << "initial tables: " << endl;
//	tab1->print();
//	tab2->print();
///*
//	cout << endl << "Join Table Studenten, Studiengänge where
// Studenten.SID=Studiengänge.SID ..." << endl;

//	//tab1->join(tab2,"SID","SID",CPU);

//  //SORT_MERGE_JOIN,NESTED_LOOP_JOIN,HASH_JOIN
//	//TablePtr tab3 =
// BaseTable::join(tab1,"SID",tab2,"SID",SORT_MERGE_JOIN,MATERIALIZE,CPU);
//	//TablePtr tab3 =
// BaseTable::join(tab1,"SID",tab2,"SID",NESTED_LOOP_JOIN,MATERIALIZE,CPU);

//	//TablePtr tab3 =
// BaseTable::join(tab1,"SID",tab2,"SID",SORT_MERGE_JOIN,MATERIALIZE,CPU);
//	//TablePtr tab3 =
// BaseTable::join(tab1,"SID",tab2,"SID",SORT_MERGE_JOIN,LOOKUP,CPU);

//	TablePtr tab3 =
// BaseTable::join(tab1,"SID",tab2,"SID",NESTED_LOOP_JOIN,MATERIALIZE,CPU);
//				tab3->print();
//				cout <<
//"=========================================================================================="
//<< endl;
//				tab3 =
// BaseTable::join(tab1,"SID",tab2,"SID",NESTED_LOOP_JOIN,LOOKUP,CPU);

////	TablePtr
/// tab3=tab1->join(tab2,"SID","SID",SORT_MERGE_JOIN,MATERIALIZE,CPU);
//				tab3->print();
//
//		cout << endl << "Projection on Table with Columns Name,
// Studiengang..." << endl;

//		list<string> columns;
//		columns.push_back("Name");
//		columns.push_back("MatrikelNr.");
//		columns.push_back("Studiengang");
//		tab3=BaseTable::projection(tab3,columns,MATERIALIZE,CPU);
//		tab3->print();

//		cout << endl << "Selection on Table on column Studiengang equals
//'INF'..." << endl;

//		//ValueComparator{LESSER,GREATER,EQUAL};
//		TablePtr
// tab4=BaseTable::selection(tab3,string("Studiengang"),string("INF"),EQUAL,MATERIALIZE,CPU);
//				tab4->print();
//				cout <<
//"=========================================================================================="
//<< endl;
//				tab4=BaseTable::selection(tab3,string("Studiengang"),string("INF"),EQUAL,LOOKUP,CPU);
//				tab4->print();

//		cout << endl << "Sorting Table by Name (ascending)..." << endl;

//		TablePtr
// tab5=BaseTable::sort(tab4,"Name",ASCENDING,MATERIALIZE,CPU);
//					tab5->print();
//					assert(tab5!=NULL);
//					cout <<
//"=========================================================================================="
//<< endl;
//					tab5=BaseTable::sort(tab4,"Name",ASCENDING,LOOKUP,CPU);
//					assert(tab5!=NULL);
//					tab5->print();
//*/

////		tab3=tab3->sort("MatrikelNr.",ASCENDING,GPU);
////		tab3->print();

//	JoinAlgorithm join_alg = HASH_JOIN; //NESTED_LOOP_JOIN; // HASH_JOIN;
////SORT_MERGE_JOIN; //HASH_JOIN; //NESTED_LOOP_JOIN NESTED_LOOP_JOIN

//	{
//	cout << endl << "Join Table Studenten, Studiengänge where
// Studenten.SID=Studiengänge.SID ..." << endl;

//	TablePtr tab3 =
// BaseTable::join(tab1,"SID",tab2,"SID",join_alg,MATERIALIZE,CPU);
//				tab3->print();

////	TablePtr
/// tab3=tab1->join(tab2,"SID","SID",SORT_MERGE_JOIN,MATERIALIZE,CPU);
////				tab3->print();
//
//		cout << endl << "Projection on Table with Columns Name,
// Studiengang..." << endl;

//		list<string> columns;
//		columns.push_back("Name");
//		columns.push_back("MatrikelNr.");
//		columns.push_back("Studiengang");
//		tab3=BaseTable::projection(tab3,columns,MATERIALIZE,CPU);
//		tab3->print();

//		cout << endl << "Selection on Table on column Studiengang equals
//'INF'..." << endl;

//		//ValueComparator{LESSER,GREATER,EQUAL};
//		TablePtr
// tab4=BaseTable::selection(tab3,string("Studiengang"),string("INF"),EQUAL,MATERIALIZE,SERIAL);
//				tab4->print();

//		cout << endl << "Sorting Table by Name (ascending)..." << endl;

//		TablePtr
// tab5=BaseTable::sort(tab4,"Name",ASCENDING,MATERIALIZE,CPU);
//					tab5->print();
//					assert(tab5!=NULL);
//					cout <<
//"=========================================================================================="
//<< endl;
//	}

//	cout << "Query using Lookup Tables..." << endl;

//	{
//	cout << endl << "Join Table Studenten, Studiengänge where
// Studenten.SID=Studiengänge.SID ..." << endl;

//	TablePtr tab3 =
// BaseTable::join(tab1,"SID",tab2,"SID",join_alg,LOOKUP,CPU);
//				tab3->print();

////	TablePtr
/// tab3=tab1->join(tab2,"SID","SID",SORT_MERGE_JOIN,MATERIALIZE,CPU);
////				tab3->print();
//
//		cout << endl << "Projection on Table with Columns Name,
// Studiengang..." << endl;

//		list<string> columns;
//		columns.push_back("Name");
//		columns.push_back("MatrikelNr.");
//		columns.push_back("Studiengang");
//		tab3=BaseTable::projection(tab3,columns,LOOKUP,CPU);
//		tab3->print();

//		cout << endl << "Selection on Table on column Studiengang equals
//'INF'..." << endl;

//		//ValueComparator{LESSER,GREATER,EQUAL};
//		TablePtr
// tab4=BaseTable::selection(tab3,string("Studiengang"),string("INF"),EQUAL,LOOKUP,SERIAL);
//					tab4->print();

//		cout << endl << "Sorting Table by Name (ascending)..." << endl;

//		TablePtr tab5=BaseTable::sort(tab4,"Name",ASCENDING,LOOKUP,CPU);
//					tab5->print();
//					assert(tab5!=NULL);
//					cout <<
//"=========================================================================================="
//<< endl;
//	}

//	return true;
//}

// void executeTestQuery(TablePtr tab1, TablePtr tab2, JoinAlgorithm join_alg,
// MaterializationStatus mat_stat, const ComputeDevice comp_dev){

////	TablePtr tab3=tab1->join(tab2,"ID","EID",CPU);
////	tab3->print();

////	cout << "perform selection..." << endl;
////	tab3=tab3->selection("Alter",60,GREATER,GPU);
////	tab3->print();

//	//TablePtr tab4=tab3->sort("Alter",ASCENDING,CPU);
//	//tab4->print();

////	cout << "Perform Join: " << endl;
//	TablePtr tab3 =
// BaseTable::join(tab1,"ID",tab2,"EID",join_alg,mat_stat,comp_dev);
//				if(!quiet && verbose)tab3->print();

//		//cout << "Selection on Table on column Alter greater than
// 60..."
//<< endl;

//		//ValueComparator{LESSER,GREATER,EQUAL};
//		TablePtr
// tab4=BaseTable::selection(tab3,string("Alter"),60,GREATER,mat_stat,SERIAL);
//					if(!quiet && verbose)tab4->print();

//		//cout << "Sorting Table by Alter (ascending)..." << endl;

//		TablePtr
// tab5=BaseTable::sort(tab4,"Alter",ASCENDING,mat_stat,CPU);
//					if(!quiet && verbose)tab5->print();
//					assert(tab5!=NULL);
//					//cout <<
//"=========================================================================================="
//<< endl;
//

//}

// void Unittest_Test_Benchmark(size_t size_of_main_table=20, size_t
// size_of_foreign_key_table=100, unsigned int number_of_runs_per_query=20){

////	size_t size_of_main_table=20//10000;//5;
////	size_t size_of_foreign_key_table=100; //100000; //20;

//	srand(0);

//	TableSchema schema;
//	schema.push_back(Attribut(INT,"ID"));
//	schema.push_back(Attribut(INT,"Alter"));
//
//	TablePtr tab1(new Table("Age",schema));

//	cout << "Filling Table Age with " << size_of_main_table << " rows" <<
// endl;
//	for(unsigned int i=0;i< size_of_main_table;i++)
//	{Tuple t; t.push_back(i); t.push_back(rand()%100); tab1->insert(t);}

//	TableSchema schema2;
//	schema2.push_back(Attribut(INT,"EID"));
//	schema2.push_back(Attribut(FLOAT,"Einkommen"));

//	TablePtr tab2(new Table("Salary",schema2));

//	cout << "Filling Table Salary with " << size_of_foreign_key_table << "
// rows" << endl;

//	for(unsigned int i=0;i<size_of_foreign_key_table;i++)
//	{Tuple t; t.push_back(int(rand()%size_of_main_table));
// t.push_back(float(rand()%10000)/100); tab2->insert(t);}
//
//	if(!quiet && verbose){
////	tab1->print();
////	tab2->print();
//	}

//	for(unsigned int i=0;i<number_of_runs_per_query;i++){
//   cout << "Query: Materialize Table		Join: Sort Merge Join " << endl;
//	Timestamp begin,end;
//	begin=getTimestamp();
//	executeTestQuery(tab1,  tab2, SORT_MERGE_JOIN, MATERIALIZE,CPU);
//	end=getTimestamp();
//	assert(end>=begin);
//	cout << "Time for Query: " << end-begin << "ns (" <<
// double(end-begin)/1000000 << "ms)" << endl;
//	cout <<
//"=========================================================================================="
//<< endl;
//	}
//	for(unsigned int i=0;i<number_of_runs_per_query;i++){
//	cout << "Query: Lookup Table		Join: Sort Merge Join " << endl;
//	Timestamp begin,end;
//	begin=getTimestamp();
//	executeTestQuery(tab1,  tab2, SORT_MERGE_JOIN, LOOKUP,CPU);
//	end=getTimestamp();
//	assert(end>=begin);
//	cout << "Time for Query: " << end-begin << "ns (" <<
// double(end-begin)/1000000 << "ms)" << endl;
//	cout <<
//"=========================================================================================="
//<< endl;
//	}
////	for(unsigned int i=0;i<number_of_runs_per_query;i++){
////	cout << "Query: Materialize Table		Join: Nested Loop Join "
///<<
/// endl;
////	Timestamp begin,end;
////	begin=getTimestamp();
////	executeTestQuery(tab1,  tab2, NESTED_LOOP_JOIN, MATERIALIZE,CPU);
////	end=getTimestamp();
////	assert(end>=begin);
////	cout << "Time for Query: " << end-begin << "ns (" <<
/// double(end-begin)/1000000 << "ms)" << endl;
////	cout <<
///"=========================================================================================="
///<< endl;
////	}
////	for(unsigned int i=0;i<number_of_runs_per_query;i++){
////	cout << "Query: Lookup Table		Join: Nested Loop Join " <<
/// endl;
////	Timestamp begin,end;
////	begin=getTimestamp();
////	executeTestQuery(tab1,  tab2, NESTED_LOOP_JOIN, LOOKUP,CPU);
////	end=getTimestamp();
////	assert(end>=begin);
////	cout << "Time for Query: " << end-begin << "ns (" <<
/// double(end-begin)/1000000 << "ms)" << endl;
////	cout <<
///"=========================================================================================="
///<< endl;
////	}

//	/*old*/
////	TablePtr tab3=tab1->join(tab2,"ID","EID",CPU);
////	tab3->print();

////	cout << "perform selection..." << endl;
////	tab3=tab3->selection("Alter",60,GREATER,GPU);
////	tab3->print();

////	TablePtr tab4=tab3->sort("Alter",ASCENDING,GPU);
////	tab4->print();
//	/*old*/

//	//TODO: add aggregation and group aggregate
//	//TODO: hash join

//

////	cout << "Materialize Table Nested Loop Join " << endl;
////	TablePtr tab3 =
/// BaseTable::join(tab1,"ID",tab2,"EID",NESTED_LOOP_JOIN,MATERIALIZE,CPU);
////	if(!quiet && verbose) tab3->print();

////	cout <<
///"=========================================================================================="
///<< endl;
////	cout << "Lookup Table Nested Loop Join " << endl;
////	tab3 =
/// BaseTable::join(tab1,"ID",tab2,"EID",NESTED_LOOP_JOIN,LOOKUP,CPU);
////	if(!quiet && verbose) tab3->print();
////	cout <<
///"=========================================================================================="
///<< endl;
////	cout << "Materialize Table Sort Merge Join " << endl;
////	tab3 =
/// BaseTable::join(tab1,"ID",tab2,"EID",SORT_MERGE_JOIN,MATERIALIZE,CPU);
////	if(!quiet && verbose) tab3->print();
////	cout <<
///"=========================================================================================="
///<< endl;
////	cout << "Lookup Table Sort Merge Join " << endl;
////	tab3 = BaseTable::join(tab1,"ID",tab2,"EID",SORT_MERGE_JOIN,LOOKUP,CPU);
////	if(!quiet && verbose) tab3->print();
////	cout <<
///"=========================================================================================="
///<< endl;

//}

// void ColumnJoinBenchmark(size_t size_of_main_table=20, size_t
// size_of_foreign_key_table=100, unsigned int number_of_runs_per_query=20){

////	size_t size_of_main_table=20//10000;//5;
////	size_t size_of_foreign_key_table=100; //100000; //20;

//	srand(0);

//	TableSchema schema;
//	schema.push_back(Attribut(INT,"ID"));
//	//schema.push_back(Attribut(INT,"Alter"));
//
//	TablePtr tab1(new Table("Age",schema));

//	cout << "Filling Table Age with " << size_of_main_table << " rows" <<
// endl;
//	for(unsigned int i=0;i< size_of_main_table;i++)
//	{Tuple t; t.push_back(i); tab1->insert(t);}

//	TableSchema schema2;
//	schema2.push_back(Attribut(INT,"EID"));
//	//schema2.push_back(Attribut(FLOAT,"Einkommen"));

//	TablePtr tab2(new Table("Salary",schema2));

//	cout << "Filling Table Salary with " << size_of_foreign_key_table << "
// rows" << endl;

//	for(unsigned int i=0;i<size_of_foreign_key_table;i++)
//	{Tuple t; t.push_back(int(rand()%size_of_main_table)); tab2->insert(t);}

//	TableSchema schema3;
//	schema3.push_back(Attribut(INT,"AID"));
//	//schema2.push_back(Attribut(FLOAT,"Einkommen"));

//	TablePtr tab3(new Table("Autos",schema3));

//	cout << "Filling Table Autos with " << size_of_foreign_key_table << "
// rows" << endl;

//	for(unsigned int i=0;i<size_of_foreign_key_table;i++)
//	{Tuple t; t.push_back(int(rand()%size_of_main_table)); tab3->insert(t);}
//
//	if(!quiet && verbose){
////	tab1->print();
////	tab2->print();
//	tab1->print();
//	tab2->print();
//	tab3->print();
//	}

//
//	cout << "Press Enter to start benchmark: " << endl;
//	string s;
//	cin >> s;
//	for(unsigned int i=0;i<number_of_runs_per_query;i++){
//		{
//  		 cout << "Query: Materialize Table		Join: Sort Merge
//  Join
//  "
//  <<
//  endl;
//		Timestamp begin,end;
//		begin=getTimestamp();
//		TablePtr tab4 = BaseTable::join(tab1,"ID",tab2,"EID",
// SORT_MERGE_JOIN, MATERIALIZE,CPU);
//		assert(tab4!=NULL);
//		//tab4->print();
//		TablePtr tab5 = BaseTable::join(tab4,"ID",tab3,"AID",
// SORT_MERGE_JOIN, MATERIALIZE,CPU);
//		//tab5->print();
//		end=getTimestamp();
//		assert(end>=begin);
//		if(!quiet && verbose)tab3->print();
//		cout << "Time for Query: " << end-begin << "ns (" <<
// double(end-begin)/1000000 << "ms)" << endl;
//		cout <<
//"=========================================================================================="
//<< endl;
//		}

//		{
//  		 cout << "Query: Lookup Table		Join: Sort Merge Join "
//  <<
//  endl;
//		Timestamp begin,end;
//		begin=getTimestamp();
//		TablePtr tab4 = BaseTable::join(tab1,"ID",tab2,"EID",
// SORT_MERGE_JOIN, LOOKUP,CPU);
//		assert(tab4!=NULL);
//		TablePtr tab5 = BaseTable::join(tab4,"ID",tab3,"AID",
// SORT_MERGE_JOIN, LOOKUP,CPU);

//		if(!quiet && verbose) cout << "Result: " << endl;
//		if(!quiet && verbose) tab5->print();
//		TablePtr tab6 = tab5->materialize();
//		if(!quiet && verbose) cout << "Materialized Result: " << endl;
//		if(!quiet && verbose) tab6->print();
//		end=getTimestamp();
//		assert(end>=begin);
//		if(!quiet && verbose)tab3->print();
//		cout << "Time for Query: " << end-begin << "ns (" <<
// double(end-begin)/1000000 << "ms)" << endl;
//		cout <<
//"=========================================================================================="
//<< endl;
//		}
//		/*{
//  		 cout << "Query: Materialize Table		Join: Nested
//  Loop
//  Join
//  "
//  << endl;
//		Timestamp begin,end;
//		begin=getTimestamp();
//		TablePtr tab4 = BaseTable::join(tab1,"ID",tab2,"EID",
// NESTED_LOOP_JOIN, MATERIALIZE,CPU);
//		assert(tab4!=NULL);
//		//tab4->print();
//		TablePtr tab5 = BaseTable::join(tab4,"ID",tab3,"AID",
// NESTED_LOOP_JOIN, MATERIALIZE,CPU);
//		//tab5->print();
//		end=getTimestamp();
//		assert(end>=begin);
//		if(!quiet && verbose) tab3->print();
//		cout << "Time for Query: " << end-begin << "ns (" <<
// double(end-begin)/1000000 << "ms)" << endl;
//		cout <<
//"=========================================================================================="
//<< endl;
//		}

//		{
//  		 cout << "Query: Lookup Table		Join: Nested Loop Join "
//  <<
//  endl;
//		Timestamp begin,end;
//		begin=getTimestamp();
//		TablePtr tab4 = BaseTable::join(tab1,"ID",tab2,"EID",
// NESTED_LOOP_JOIN, LOOKUP,CPU);
//		assert(tab4!=NULL);
//		TablePtr tab5 = BaseTable::join(tab4,"ID",tab3,"AID",
// NESTED_LOOP_JOIN, LOOKUP,CPU);
//		cout << "Result: " << endl;
//		//tab5->print();
//		TablePtr tab6 = tab5->materialize();
//		cout << "Materialized Result: " << endl;
//		//tab6->print();
//		end=getTimestamp();
//		assert(end>=begin);
//		if(!quiet && verbose)tab3->print();
//		cout << "Time for Query: " << end-begin << "ns (" <<
// double(end-begin)/1000000 << "ms)" << endl;
//		cout <<
//"=========================================================================================="
//<< endl;
//		}*/
//		{
//  		 cout << "Query: Materialize Table		Join: Hash Join
//  "
//  <<
//  endl;
//		Timestamp begin,end;
//		begin=getTimestamp();
//		TablePtr tab4 = BaseTable::join(tab1,"ID",tab2,"EID", HASH_JOIN,
// MATERIALIZE,CPU);
//		assert(tab4!=NULL);
//		//tab4->print();
//		TablePtr tab5 = BaseTable::join(tab4,"ID",tab3,"AID", HASH_JOIN,
// MATERIALIZE,CPU);
//		//tab5->print();
//		end=getTimestamp();
//		assert(end>=begin);
//		if(!quiet && verbose)tab3->print();
//		cout << "Time for Query: " << end-begin << "ns (" <<
// double(end-begin)/1000000 << "ms)" << endl;
//		cout <<
//"=========================================================================================="
//<< endl;
//		}

//		{
//  		 cout << "Query: Lookup Table		Join: Hash Join " <<
//  endl;
//		Timestamp begin,end;
//		begin=getTimestamp();
//		TablePtr tab4 = BaseTable::join(tab1,"ID",tab2,"EID", HASH_JOIN,
// LOOKUP,CPU);
//		assert(tab4!=NULL);
//		TablePtr tab5 = BaseTable::join(tab4,"ID",tab3,"AID", HASH_JOIN,
// LOOKUP,CPU);
//		cout << "Result: " << endl;
//		//tab5->print();
//		TablePtr tab6 = tab5->materialize();
//		cout << "Materialized Result: " << endl;
//		//tab6->print();
//		end=getTimestamp();
//		assert(end>=begin);
//		if(!quiet && verbose)tab3->print();
//		cout << "Time for Query: " << end-begin << "ns (" <<
// double(end-begin)/1000000 << "ms)" << endl;
//		cout <<
//"=========================================================================================="
//<< endl;
//		cout <<
//"******************************************************************************************"
//<< endl;
//		}

//	}

//}

// inline bool LoadDatabase(){

//	return loadTables();

//}
