

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
echo "set gpu_buffer_size=$1" >> "$SCRIPT_FILE"
echo "set gpu_buffer_management_strategy=least_recently_used" >> "$SCRIPT_FILE"
echo "set enable_dataplacement_aware_query_optimization=$2" >> "$SCRIPT_FILE"
echo "set pin_columns_in_gpu_buffer=$3" >> "$SCRIPT_FILE"
#echo "placecolumns" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_REVENUE" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_QUANTITY" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_DISCOUNT" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_SHIPPRIORITY" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_EXTENDEDPRICE" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_ORDTOTALPRICE" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_REVENUE" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_SUPPLYCOST" >> "$SCRIPT_FILE"
echo "placecolumn 1 LINEORDER LO_TAX" >> "$SCRIPT_FILE"
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
rm -f *-*-measurements.csv
rm -rf logfiles
mkdir logfiles

for i in 0 $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.01) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.05) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.1) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.15) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.2) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.25) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.3) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.35) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.4) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.45) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.5) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.55) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.6) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.65) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.7) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.75) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.8) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.85) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.9) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.95); do
   for pinned_buffer in true false; do
       for data_driven in true false; do
generate_script $i $data_driven $pinned_buffer

$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE simple_selection_workload_8_selections.coga > cogadb_ssb_measurements.log

cp parallel_results_user_0.txt logfiles/parallel_results_user_0_$i-$data_driven-$pinned_buffer.txt
cp cogadb_ssb_measurements.log logfiles/cogadb_ssb_measurements_$i-$data_driven-$pinned_buffer.log

EXEC_TIME=$(cat cogadb_ssb_measurements.log | grep "WORKLOAD EXECUTION TIME:" | awk '{print $5}' | sed -e 's/s//g')
COLUMN_CACHE_ACCESS=$(cat cogadb_ssb_measurements.log | grep "COLUMN_CACHE_ACCESS:" | awk '{print $2}')
COLUMN_CACHE_HIT=$(cat cogadb_ssb_measurements.log | grep "COLUMN_CACHE_HIT:" | awk '{print $2}')
COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC=$(cat cogadb_ssb_measurements.log | grep "COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC" | awk '{print $2}')


if [ -z $COLUMN_CACHE_HIT ]; then
    COLUMN_CACHE_HIT=0
fi

if [ -z $COLUMN_CACHE_ACCESS ]; then
    COLUMN_CACHE_ACCESS=0
fi

if [ -z $COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC ]; then
    COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SEC=0
fi






echo "$i $data_driven $pinned_buffer $EXEC_TIME $COLUMN_CACHE_HIT $COLUMN_CACHE_ACCESS $COLUMN_COPY_TIME_HOST_TO_DEVICE_IN_SECS" >> $data_driven-$pinned_buffer-measurements.csv
       done
   done
done



