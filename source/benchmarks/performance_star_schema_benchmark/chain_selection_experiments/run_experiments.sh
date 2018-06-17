

SCRIPT_FILE=startup.coga
PATH_TO_DATABASE=`cat ../PATH_TO_COGADB_DATABASES`

GPU_BUFFER_SIZE=1126694912
DATA_PLACEMENT_DRIVEN_OPT=false
PIN_COLUMNS=false
TOTAL_GPU_DEVICE_MEMORY_CAPACITY=`cat ../GPU_DEVICE_MEMORY_CAPACITY`
ABSOLUTE_PATH_TO_COGADB_EXECUTABLE=`cat ../PATH_TO_COGADB_EXECUTABLE`
SCALE_FACTOR=`cat ../SCALE_FACTOR`

function generate_script()
{

echo "set path_to_database=$PATH_TO_DATABASE/ssb_sf$SCALE_FACTOR" > "$SCRIPT_FILE"
echo "loaddatabase" >> "$SCRIPT_FILE"
echo "set enable_profiling=true" >> "$SCRIPT_FILE"
echo "set hybrid_query_optimizer=best_effort_gpu_heuristic" >> "$SCRIPT_FILE"
echo "setdevice gpu" >> "$SCRIPT_FILE"
echo "set gpu_buffer_size=$1" >> "$SCRIPT_FILE"
echo "set gpu_buffer_management_strategy=least_recently_used" >> "$SCRIPT_FILE"
echo "set enable_dataplacement_aware_query_optimization=$2" >> "$SCRIPT_FILE"
echo "set pin_columns_in_gpu_buffer=$2" >> "$SCRIPT_FILE"
echo "set print_query_result=false" >> "$SCRIPT_FILE"
echo "ssb_select" >> "$SCRIPT_FILE"
echo "placecolumnfrequencybased 1" >> "$SCRIPT_FILE"



if($3)
then
echo "toggleQC" >> "$SCRIPT_FILE"
fi

echo "set enable_pull_based_query_chopping=$4" >> "$SCRIPT_FILE"



##echo "set unsafe_feature.enable_immediate_selection_abort=$4" >> "$SCRIPT_FILE"
#echo "placecolumns" >> "$SCRIPT_FILE"
#echo "placecolumn 1 LINEORDER LO_REVENUE" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_QUANTITY" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_DISCOUNT" >> "$SCRIPT_FILE"
#echo "placecolumn 1 LINEORDER LO_SHIPPRIORITY" >> "$SCRIPT_FILE"
#echo "placecolumn 1 LINEORDER LO_EXTENDEDPRICE" >> "$SCRIPT_FILE"
#echo "placecolumn 1 LINEORDER LO_ORDTOTALPRICE" >> "$SCRIPT_FILE"
#echo "placecolumn 1 LINEORDER LO_REVENUE" >> "$SCRIPT_FILE"
#echo "placecolumn 1 LINEORDER LO_SUPPLYCOST" >> "$SCRIPT_FILE"
#echo "placecolumn 1 LINEORDER LO_TAX" >> "$SCRIPT_FILE"
#echo "set pin_join_indexes_in_gpu_buffer=true" > "$SCRIPT_FILE"
}

float_scale=0

function float_eval()
{
    local stat=0
    local result=0.0
    if [[ $# -gt 0 ]]; then
        result=$(echo "scale=$float_scale; $*" | bc -q 2>/dev/null | sed -e 's/\./,/g')
        stat=$?
        #result=`printf -v int %.0f "$result"`
        result=`printf %.0f "$result"`
        result=${result%,*}
        if [[ $stat -eq 0  &&  -z "$result" ]]; then stat=1; fi
    fi
    echo $result
    return $stat
}

rm -f measurements.csv
rm -f *-measurements.csv
rm -f parallel_results_user_*.txt
rm -rf logfiles
mkdir logfiles

#for i in 0 $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.01) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.05) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.1) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.15) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.2) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.25) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.3) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.35) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.4) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.45) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.5) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.55) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.6) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.65) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.7) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.75) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.8) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.85) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.9) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.95); do
#for number_of_users in 1 2 4 6 8 10 15 20; do
for number_of_users in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
   #for pinned_buffer in true false; do
   for data_driven in true false; do
      for query_chopping in true false; do
         for pull_based_query_chopping in true false; do
          for run_numer in 1 2 3 4 5; do


generate_script $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.5) $data_driven $query_chopping $pull_based_query_chopping


rm -f chain_selection_workload_generated.coga 
cat chain_selection_workload.coga | sed -e 's/TEMPLATE_VARIABLE_PARALLEL_USERS/'$number_of_users'/g' > chain_selection_workload_generated.coga 


$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE chain_selection_workload_generated.coga > cogadb_ssb_measurements.log

for i in parallel_results_user_*.txt; do
   mv $i logfiles/$i-$number_of_users-$data_driven-$query_chopping-$pull_based_query_chopping.txt
done
#cp parallel_results_user_0.txt logfiles/parallel_results_user_0_$number_of_users-$data_driven-$pinned_buffer.txt
cp cogadb_ssb_measurements.log logfiles/cogadb_ssb_measurements_$number_of_users-$data_driven-$query_chopping-$pull_based_query_chopping.log

EXEC_TIME=$(cat cogadb_ssb_measurements.log | grep "WORKLOAD EXECUTION TIME:" | awk '{print $5}' | sed -e 's/s//g')
COLUMN_CACHE_ACCESS=$(cat cogadb_ssb_measurements.log | grep "COLUMN_CACHE_ACCESS:" | awk '{print $2}')
COLUMN_CACHE_HIT=$(cat cogadb_ssb_measurements.log | grep "COLUMN_CACHE_HIT:" | awk '{print $2}')
COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC=$(cat cogadb_ssb_measurements.log | grep "COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC" | awk '{print $2}')
COLUMN_COPY_TIME_DEVICE_TO_HOST_IN_SEC=$(cat cogadb_ssb_measurements.log | grep "COLUMN_COPY_TIME_DEVICE_TO_HOST_IN_SEC" | awk '{print $2}')
NUMBER_OF_ABORTED_GPU_OPERATORS=$(cat cogadb_ssb_measurements.log | grep "NUMBER_OF_ABORTED_GPU_OPERATORS" | awk '{print $2}')
NUMBER_OF_EXECUTED_GPU_OPERATORS=$(cat "cogadb_ssb_measurements.log" | grep "NUMBER_OF_EXECUTED_GPU_OPERATORS" | awk '{print $2}')
TOTAL_COLUMN_EVICTIONS=$(cat "cogadb_ssb_measurements.log" | grep "TOTAL_COLUMN_EVICTIONS" | awk '{print $2}')
NUMBER_OF_COPY_OPERATIONS_DEVICE_TO_HOST_TOTAL=$(cat "cogadb_ssb_measurements.log" | grep "NUMBER_OF_COPY_OPERATIONS_DEVICE_TO_HOST_TOTAL" | awk '{print $2}')
NUMBER_OF_COPY_OPERATIONS_HOST_TO_DEVICE_TOTAL=$(cat "cogadb_ssb_measurements.log" | grep "NUMBER_OF_COPY_OPERATIONS_HOST_TO_DEVICE_TOTAL" | awk '{print $2}')
TOTAL_LOST_TIME_IN_NS_DUE_TO_ABORTED_GPU_OPERATORS=$(cat "cogadb_ssb_measurements.log" | grep "TOTAL_LOST_TIME_IN_NS_DUE_TO_ABORTED_GPU_OPERATORS" | awk '{print $2}')
 


if [ -z $COLUMN_CACHE_HIT ]; then
    COLUMN_CACHE_HIT=0
fi

if [ -z $COLUMN_CACHE_ACCESS ]; then
    COLUMN_CACHE_ACCESS=0
fi

if [ -z $COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC ]; then
    COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC=0
fi

if [ -z $COLUMN_COPY_TIME_DEVICE_TO_HOST_IN_SEC ]; then
    COLUMN_COPY_TIME_DEVICE_TO_HOST_IN_SEC=0
fi

if [ -z $NUMBER_OF_ABORTED_GPU_OPERATORS ]; then
    NUMBER_OF_ABORTED_GPU_OPERATORS=0
fi

if [ -z $NUMBER_OF_EXECUTED_GPU_OPERATORS ]; then
    NUMBER_OF_EXECUTED_GPU_OPERATORS=0
fi

if [ -z $TOTAL_COLUMN_EVICTIONS ]; then
    TOTAL_COLUMN_EVICTIONS=0
fi

if [ -z $NUMBER_OF_COPY_OPERATIONS_DEVICE_TO_HOST_TOTAL ]; then
    NUMBER_OF_COPY_OPERATIONS_DEVICE_TO_HOST_TOTAL=0
fi

if [ -z $NUMBER_OF_COPY_OPERATIONS_HOST_TO_DEVICE_TOTAL ]; then
    NUMBER_OF_COPY_OPERATIONS_HOST_TO_DEVICE_TOTAL=0
fi

if [ -z $TOTAL_LOST_TIME_IN_NS_DUE_TO_ABORTED_GPU_OPERATORS ]; then
    TOTAL_LOST_TIME_IN_NS_DUE_TO_ABORTED_GPU_OPERATORS=0
fi





echo "$number_of_users $data_driven $query_chopping $pull_based_query_chopping $EXEC_TIME $COLUMN_CACHE_HIT $COLUMN_CACHE_ACCESS $COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC $COLUMN_COPY_TIME_DEVICE_TO_HOST_IN_SEC $NUMBER_OF_ABORTED_GPU_OPERATORS $NUMBER_OF_EXECUTED_GPU_OPERATORS $NUMBER_OF_COPY_OPERATIONS_HOST_TO_DEVICE_TOTAL $NUMBER_OF_COPY_OPERATIONS_DEVICE_TO_HOST_TOTAL $TOTAL_COLUMN_EVICTIONS $TOTAL_LOST_TIME_IN_NS_DUE_TO_ABORTED_GPU_OPERATORS" >> $data_driven-$query_chopping-$pull_based_query_chopping-measurements.csv
             done
          done
       done
   done
done



