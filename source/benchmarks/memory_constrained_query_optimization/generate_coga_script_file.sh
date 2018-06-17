
if [ $# -lt 8 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 <SCALE_FACTOR> <QUERY_NAME> <NUMBER_OF_PARALLEL_USERS> <NUMBER_OF_QUERIES> <NUMBER_OF_WARMUP_QUERIES> <SCRIPT_NAME> <RUN_TYPE> <RUN_TYPE_WARMUP_PHASE>"
	echo "<RUN_TYPE> corresponds to CoGaDB's device policy: 'any' uses all procesing devices, 'cpu' only CPU devices and 'gpu' only gpu devices" 
	exit -1
fi

if [ $# -gt 8 ]; then
	echo 'To many parameters!'
	echo "Usage: $0 <SCALE_FACTOR> <QUERY_NAME> <NUMBER_OF_PARALLEL_USERS> <NUMBER_OF_QUERIES> <NUMBER_OF_WARMUP_QUERIES> <SCRIPT_NAME> <RUN_TYPE> <RUN_TYPE_WARMUP_PHASE>"
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

if [[ "$QUERY_NAME" == "ssball" ]]; then
#if we execute a complete workload, we do not need to repeat the workload X times until we executed X queries
#therefore, we devide by the number of queries in the workload (13), and round up (X=(X+12)/13)
NUMBER_OF_QUERIES=$(((NUMBER_OF_QUERIES+12)/13))
#execute the workload 2 times to train the cost models
NUMBER_OF_WARMUP_QUERIES=2
fi



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

echo set path_to_database=`cat PATH_TO_COGADB_DATABASES`ssb_sf$SCALE_FACTOR > $SCRIPT_NAME
echo loaddatabase >> $SCRIPT_NAME
echo loadjoinindexes >> $SCRIPT_NAME
#echo "set_global_load_adaption_policy no_recomputation" >> $SCRIPT_NAME
#insert warm up query, which we only process on CPU to avoid penalty times of GPU operators in the warm up query
#the execution time of the warmup query is not included in the workload execution time, because we measure a ''warm'' system
echo "setdevice $RUN_TYPE_WARMUP_PHASE" >> $SCRIPT_NAME
for (( c=0; c<$NUMBER_OF_WARMUP_QUERIES; c++ ))
do
	if [[ "$QUERY_NAME" == "ssball" ]]; then
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
	else
		#just use one query in workload
    	echo $QUERY_NAME >> $SCRIPT_NAME	
	fi
done
echo "resetstatistics"  >> $SCRIPT_NAME
echo "setdevice $RUN_TYPE" >> $SCRIPT_NAME
echo "starttimer" >> $SCRIPT_NAME
echo "parallelExec $NUMBER_OF_PARALLEL_USERS" >> $SCRIPT_NAME
for (( c=0; c<$NUMBER_OF_QUERIES; c++ ))
do
	if [[ "$QUERY_NAME" == "ssball" ]]; then
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
	else
		#just use one query in workload
    	echo $QUERY_NAME >> $SCRIPT_NAME	
	fi
done
echo "serial_execution"  >> $SCRIPT_NAME
echo "stoptimer" >> $SCRIPT_NAME
echo "printstatistics"  >> $SCRIPT_NAME
echo "dumpestimationerrors average_estimation_errors.csv" >> $SCRIPT_NAME
echo "hypestatus" >> $SCRIPT_NAME
echo quit >> $SCRIPT_NAME

