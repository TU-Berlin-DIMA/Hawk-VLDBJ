

if [ $# -lt 10 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 <SCALE_FACTOR> <QUERY_NAME> <NUMBER_OF_PARALLEL_USERS> <NUMBER_OF_QUERIES> <NUMBER_OF_WARMUP_QUERIES> <SCRIPT_NAME> <RUN_TYPE> <RUN_TYPE_WARMUP_PHASE> <ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION> <PRELOAD_DATA_IN_GPU_BUFFER>"
	echo "<RUN_TYPE> corresponds to CoGaDB's device policy: 'any' uses all procesing devices, 'cpu' only CPU devices and 'gpu' only gpu devices" 
	exit -1
fi

if [ $# -gt 10 ]; then
	echo 'To many parameters!'
	echo "Usage: $0 <SCALE_FACTOR> <QUERY_NAME> <NUMBER_OF_PARALLEL_USERS> <NUMBER_OF_QUERIES> <NUMBER_OF_WARMUP_QUERIES> <SCRIPT_NAME> <RUN_TYPE> <RUN_TYPE_WARMUP_PHASE> <ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION> <PRELOAD_DATA_IN_GPU_BUFFER>"
	echo "<RUN_TYPE> corresponds to CoGaDB's device policy: 'any' uses all procesing devices, 'cpu' only CPU devices and 'gpu' only gpu devices" 
	exit -1
fi



SCALE_FACTOR=$1
QUERY_NAME=$2
NUMBER_OF_PARALLEL_USERS=$3
NUMBER_OF_QUERIES=$4
NUMBER_OF_WARMUP_QUERIES=$5
SCRIPT_NAME=$6
RUN_TYPE=$7
RUN_TYPE_WARMUP_PHASE=$8
ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION=$9
PRELOAD_DATA_IN_GPU_BUFFER=${10}

#if [[ "$QUERY_NAME" == "ssball" ]]; then
##if we execute a complete workload, we do not need to repeat the workload X times until we executed X queries
##therefore, we devide by the number of queries in the workload (13), and round up (X=(X+12)/13)
#NUMBER_OF_QUERIES=$(((NUMBER_OF_QUERIES+12)/13))
##execute the workload 2 times to train the cost models
#NUMBER_OF_WARMUP_QUERIES=2
#fi

if [[ "$RUN_TYPE" != "cpu" && "$RUN_TYPE" != "any" ]]; then
	echo "Error! Invalid RUN_TYPE, only 'cpu' or 'any' allowed!"
	exit -1
fi

if [[ "$RUN_TYPE_WARMUP_PHASE" != "cpu" && "$RUN_TYPE_WARMUP_PHASE" != "any" ]]; then
	echo "Error! Invalid RUN_TYPE_WARMUP_PHASE, only 'cpu' or 'any' allowed!"
	exit -1
fi



echo "Generating coga script file..."
echo "SCALE_FACTOR=$SCALE_FACTOR"
echo "QUERY_NAME=$QUERY_NAME"
echo "NUMBER_OF_PARALLEL_USERS=$NUMBER_OF_PARALLEL_USERS"
echo "NUMBER_OF_QUERIES=$NUMBER_OF_QUERIES"
echo "NUMBER_OF_WARMUP_QUERIES=$NUMBER_OF_WARMUP_QUERIES"
echo "SCRIPT_NAME=$SCRIPT_NAME"
echo "RUN_TYPE=$RUN_TYPE"
echo "ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION=$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION"
echo "PRELOAD_DATA_IN_GPU_BUFFER=$PRELOAD_DATA_IN_GPU_BUFFER"

if [[ "$QUERY_NAME" == "ssb"* ]]; then
   echo set path_to_database=`cat PATH_TO_COGADB_DATABASES`ssb_sf$SCALE_FACTOR > $SCRIPT_NAME
elif [[ "$QUERY_NAME" == "tpch"* ]]; then
   echo set path_to_database=`cat PATH_TO_COGADB_DATABASES`tpch_sf$SCALE_FACTOR > $SCRIPT_NAME
else
    echo "FATAL ERROR: Executing neither Star Schema Benchmark nor TPC-H Benchmark. Aborting..."
    exit -1
fi

echo loaddatabase >> $SCRIPT_NAME
echo loadjoinindexes >> $SCRIPT_NAME
if [ "$PRELOAD_DATA_IN_GPU_BUFFER" == "true" ]; then
#    echo placejoinindexes >> $SCRIPT_NAME
    if [[ "$QUERY_NAME" == "ssb"* ]]; then
        echo placecolumns >> $SCRIPT_NAME
    fi
fi
if [ "$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION" == "true" ]; then
    echo "set pin_columns_in_gpu_buffer=true" >> $SCRIPT_NAME
    echo "set pin_join_indexes_in_gpu_buffer=true" >> $SCRIPT_NAME
fi

if [[ "$QUERY_NAME" != "ssb_all" ]]; then
    echo "set print_query_result=false" >> $SCRIPT_NAME
    let NUMBER_OF_WARMUP_QUERIES=NUMBER_OF_WARMUP_QUERIES*10
    echo "NUMBER_OF_WARMUP_QUERIES: $NUMBER_OF_WARMUP_QUERIES"
fi

#the join order optimizer produces bad plans due to incorrect cardinality estimates
#we workaround this by using a special optimizer pipeline that performs no join ordering
#and we hardcode the join order in the query plan (which is now no longer changed by the optimizer)
echo "set optimizer=no_join_order_optimizer" >> $SCRIPT_NAME

#echo toggleQC >> $SCRIPT_NAME
#echo "set_global_load_adaption_policy no_recomputation" >> $SCRIPT_NAME
#insert warm up query, which we only process on CPU to avoid penalty times of GPU operators in the warm up query
#the execution time of the warmup query is not included in the workload execution time, because we measure a ''warm'' system
#echo "setdevice $RUN_TYPE_WARMUP_PHASE" >> $SCRIPT_NAME
for (( c=0; c<$NUMBER_OF_WARMUP_QUERIES; c++ ))
do
        #alternate execution on cpu and gpu in training phase
        let evencheck=c%2
        if [ $evencheck -eq 0 ] 
        then
        echo "setdevice gpu" >> $SCRIPT_NAME
        else
        echo "setdevice cpu" >> $SCRIPT_NAME
        fi

	if [[ "$QUERY_NAME" == "ssb_all" ]]; then
	    #use all 13 queries in workload
		echo "ssb11
ssb12
ssb13
ssb21
ssb22
ssb23
ssb31
ssb32
ssb33
ssb34
ssb41
ssb42
ssb43" >> $SCRIPT_NAME	
        elif [[ "$QUERY_NAME" == "tpch_all_supported" ]]; then
	    #use all supported queries in workload
 	     echo "#tpch01
tpch02
tpch03
tpch04
tpch05
tpch06
tpch07
#tpch09
#tpch10
#tpch15
#tpch18
#tpch20" >> $SCRIPT_NAME	
        elif [[ "$QUERY_NAME" == "tpch_all_supported_by_cogadb_and_ocelot" ]]; then
	    #use all supported queries in workload
 	     echo "tpch03
tpch04
tpch05
tpch06
tpch07
tpch10
tpch15" >> $SCRIPT_NAME	
	else
		#just use one query in workload
    	echo $QUERY_NAME >> $SCRIPT_NAME	
	fi
done
#delete statistics that was created for warmup queries
echo resetstatistics >> $SCRIPT_NAME
echo "setdevice $RUN_TYPE" >> $SCRIPT_NAME
echo "starttimer" >> $SCRIPT_NAME
echo "parallelExec $NUMBER_OF_PARALLEL_USERS" >> $SCRIPT_NAME
for (( c=0; c<$NUMBER_OF_QUERIES; c++ ))
do
	if [[ "$QUERY_NAME" == "ssb_all" ]]; then
	    #use all 13 queries in workload
		echo "ssb11
ssb12
ssb13
ssb21
ssb22
ssb23
ssb31
ssb32
ssb33
ssb34
ssb41
ssb42
ssb43" >> $SCRIPT_NAME
        elif [[ "$QUERY_NAME" == "tpch_all_supported" ]]; then
	    #use all supported queries in workload
 	     echo "#tpch01
tpch02
tpch03
tpch04
tpch05
tpch06
tpch07
#tpch09
#tpch10
#tpch15
#tpch18
#tpch20" >> $SCRIPT_NAME
        elif [[ "$QUERY_NAME" == "tpch_all_supported_by_cogadb_and_ocelot" ]]; then
	    #use all supported queries in workload
 	     echo "tpch03
tpch04
tpch05
tpch06
tpch07
tpch10
tpch15" >> $SCRIPT_NAME		
	else
	#just use the same query in workload
    	echo $QUERY_NAME >> $SCRIPT_NAME	
	fi
done
echo "serial_execution"  >> $SCRIPT_NAME
echo "stoptimer" >> $SCRIPT_NAME
echo "printstatistics" >> $SCRIPT_NAME
echo "print_in_memory_columns" >> $SCRIPT_NAME
echo "print_memory_footprint_of_in_memory_columns" >> $SCRIPT_NAME
echo "dumpestimationerrors average_estimation_errors.csv" >> $SCRIPT_NAME
echo "hypestatus" >> $SCRIPT_NAME
echo quit >> $SCRIPT_NAME

