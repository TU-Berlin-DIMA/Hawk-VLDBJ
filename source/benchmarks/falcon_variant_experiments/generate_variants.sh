
#set -e

function compute_statistics_of_measurement_file {
   mkdir -p results
   echo "Computing stats for file: '$1'" 
   measurement_file="$1"
   experiment_name="$2"
   measured_metric_name="$3"
   echo "Experiment Name: '$experiment_name'"
   echo "Metric: '$measured_metric_name'"
   f3=$(echo $experiment_name | sed -e 's/\.csv/_summary_'"$measured_metric_name"'.csv/g')
   echo "$f3"
   echo "min"	"max"	"average"	"variance"	"standard_deviation"	"standard_error_of_the_mean"	"lower_bound_of_error_bar"	"upper_bound_of_error_bar"	"lower_bound_of_standard_error_of_the_mean_error_bar"	"upper_bound_of_standard_error_of_the_mean_error_bar" > results/$f3
   NUM_ROWS=$(cat execution_times.data | wc -l)
   echo $NUM_ROWS
   NUM_ROWS=$((NUM_ROWS-1))

   echo "'$NUM_ROWS'"
   ABANDONDED_MEASUREMENT=$(head -n 1 "$measurement_file")
   echo "Abandon first measurement: $ABANDONDED_MEASUREMENT"
#   export POSIXLY_CORRECT=1
   export LC_ALL=en_US.utf-8
   tail -n $NUM_ROWS "$measurement_file" | gawk  -f "compute_average.awk" >> results/$f3
   tail -n $NUM_ROWS "$measurement_file"
   echo "Done Computing stats for file: '$1'"
}  

#no caching of CUDA OpenCL kernels
export CUDA_CACHE_DISABLE=1

if [ ! -f PATH_TO_COGADB_EXECUTABLE ]
then
    echo "File 'PATH_TO_COGADB_EXECUTABLE' not found!"
    echo "Please enter the path to the cogadb executable:"
    read LINE
    echo "$LINE" > PATH_TO_COGADB_EXECUTABLE
fi

if [ ! -f PATH_TO_DATABASE ]
then
    echo "File 'PATH_TO_DATABASE' not found!"
    echo "Please enter the path to the cogadb database:"
    read LINE
    echo "$LINE" > PATH_TO_DATABASE
fi

PATH_TO_COGADB_EXECUTABLE=$(cat PATH_TO_COGADB_EXECUTABLE)
PATH_TO_DATABASE=$(cat PATH_TO_DATABASE)
QUERY_NAME=$(cat config/QUERY_NAME)

#first, generate all basic variants
rm -f variants/*.coga
rm -f *.variant
cd variants
if [[ $QUERY_NAME == *grouped_aggregation* ]]; then
    echo "Generate Variants for Grouped Aggregation Query"
    for hash_table in grouped_aggregation_hash_tables/*.coga; do
        for mem_access in memory_access_pattern/*.coga; do
            for multiplier in grouped_aggregation_gpu_global_size_multiplier/*.coga; do
                for local_size in grouped_aggregation_gpu_atomic_local_size_size/*.coga; do
                    for strategy in grouped_aggregation_strategies/*.coga; do
                        mem2=$(basename "$mem_access" | sed -e 's/\.coga//g') 
                        ht2=$(basename "$hash_table" | sed -e 's/\.coga//g')
                        multp2=$(basename "$multiplier" | sed -e 's/\.coga//g')
                        localsize2=$(basename "$local_size" | sed -e 's/\.coga//g')
                        strat2=$(basename "$strategy" | sed -e 's/\.coga//g')
                        
                        if [[ $strat2 == *"sequential"* ]]; then
                            variant_name=$(echo "$mem2"_"$ht2"_"$strat2".coga)
                            echo "$variant_name"
                            cat $mem_access > $variant_name
                            cat $hash_table >> $variant_name
                            cat $strategy >> $variant_name
                        elif [[ $strat2 == *"semaphore"* ]]; then
                            variant_name=$(echo "$mem2"_"$ht2"_mult"$multp2"_"$strat2".coga)
                            echo "$variant_name"
                            cat $mem_access > $variant_name
                            cat $hash_table >> $variant_name
                            cat $multiplier >> $variant_name
                            cat $strategy >> $variant_name
                        else
                            variant_name=$(echo "$mem2"_"$ht2"_mult"$multp2"_local_size"$localsize2"_"$strat2".coga)
                            echo "$variant_name"
                            cat $mem_access > $variant_name
                            cat $hash_table >> $variant_name
                            cat $multiplier >> $variant_name
                            cat $local_size >> $variant_name
                            cat $strategy >> $variant_name
                        fi
                    done
                done
            done
        done
    done
elif [[ $QUERY_NAME == *aggregation* ]]; then
    echo "Generate Variants for Aggregation Query"
    for mem_access in memory_access_pattern/*.coga; do
        for predication in predication/*.coga; do
            mem2=$(basename "$mem_access" | sed -e 's/\.coga//g') 
            pred2=$(basename "$predication" | sed -e 's/\.coga//g') 
            variant_name=$(echo "$mem2"_"$pred2".coga)
            echo "$variant_name"
            cat $mem_access > $variant_name
            cat $predication >> $variant_name
        done
    done
else
    echo "Generate Variants for Projection Query"
    for strat in execution_strategies/*.coga; do
        for mem_access in memory_access_pattern/*.coga; do
            for predication in predication/*.coga; do
                strat2=$(basename "$strat" | sed -e 's/\.coga//g') 
                mem2=$(basename "$mem_access" | sed -e 's/\.coga//g') 
                pred2=$(basename "$predication" | sed -e 's/\.coga//g') 
                variant_name=$(echo "$strat2"_"$mem2"_"$pred2".coga)
                echo "$variant_name"
                cat $strat > $variant_name
                cat $mem_access >> $variant_name
                cat $predication >> $variant_name
            done
        done
    done
fi

echo "Generate Variant for C-code execution strategy"
cat c_code/c_code.coga > c_code.coga

cd ..

#then, create scripts that run basic variants on all processors for all workloads
for d in devices/*.coga; do
    for v in variants/*.coga; do
        d2=$(basename $d)
        v2=$(basename $v)
        
        if [[ $d2 == *"cpu"* && $v2 == *"atomic"* ]]; then
            echo "Blacklisting variant \"$v2\" for device \"$d2\""
            continue
        fi
        
        variant_name=$(echo "$d2"-"$v2".variant | sed -e 's/\.coga//g')
        echo "$variant_name"
        cat $d > $variant_name
        cat $v >> $variant_name
        cat queries.coga >> $variant_name
    done
done

# prepare the startup.coga
echo "set path_to_database=$PATH_TO_DATABASE" > startup.coga
cat startup.coga.in >> startup.coga

#execute each variant for a given query workload
rm -rf results
rm -rf errors
rm -rf plots
mkdir -p results
mkdir -p errors
mkdir -p plots
for v in *.variant; do
    v2=$(echo $v | sed -e 's/variant/raw_data/g') 
    echo "Execute Experiment, writing in file '$v2'"
    rm -f finished
    NUM_TRIES=0;
    while [ ! -e finished ]; do
        ulimit -c unlimited 
        "$PATH_TO_COGADB_EXECUTABLE" $v > tmp.log
        RET=$?
        cat tmp.log | grep -E "Execution|Compile" > results/$v2
        #if we used STRG+C, terminate experiment
        if [ $RET -eq 130 ]; then break; fi
        #if some kind of error occured, repeat
        if [ $RET -ne 0 ]; then
        DATE=$(date)
        mv core core_file_"$v"_created_at_"$DATE"
        echo "Error executing variant: '$v'!"
        cp $v errors/
        mv core_file_"$v"_created_at_"$DATE" errors/
        let "NUM_TRIES++"
        if [ $NUM_TRIES -lt 3 ]; then
            echo "Repeat Execution...(TRY: $NUM_TRIES)"
            sleep 1
        else
            echo "Too many failures: fix variant '$v'!"
            exit -1 
        fi
        else
        #everything fine, signal success
        touch finished
        fi 
    done
done 

for f in results/*.raw_data; do
    echo $f
    f2=$(echo $f | sed -e 's/raw_data/csv/g')
    cat $f | grep "Pipeline Execution Time" | awk '{print $4}' | sed -E 's/s//g' > execution_times.data
    cat $f | grep "Compile Time (Host)" | awk '{print $4}' | sed -E 's/s//g' > compile_times_host.data

    if [[ $f == *"c_code"* ]]; then
        rm compile_times_kernel.data
        for a in `seq $(cat compile_times_host.data | wc -l)`; do echo "0" >> compile_times_kernel.data; done
    else
        cat $f | grep "Compile Time (Kernel)" | awk '{print $4}' | sed -E 's/s//g' > compile_times_kernel.data
    fi

    echo "execution_time	compile_time_host	compile_time_kernel" > $f2
    paste execution_times.data compile_times_host.data compile_times_kernel.data >> $f2

    EXPERIMENT_NAME=$(basename "$f2")
    compute_statistics_of_measurement_file execution_times.data "$EXPERIMENT_NAME" "execution_time"
    compute_statistics_of_measurement_file compile_times_host.data "$EXPERIMENT_NAME" "compile_time_host"
    compute_statistics_of_measurement_file compile_times_kernel.data "$EXPERIMENT_NAME" "compile_time_kernel"
done 

#create plots for measured results
bash plot_variant_measurements.sh 

#export results as zip archieve
DATE=$(date | sed -e 's/ /_/g' | sed -e 's/\:/_/g')
#OUTPUT_DIR="falcon_paper_experimental_results-$DATE-$HOSTNAME"
OUTPUT_DIR="experimental_results"
mkdir "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"/variants
cp -r results "$OUTPUT_DIR"/
cp -r plots "$OUTPUT_DIR"/
cp -r errors "$OUTPUT_DIR"/
cp -r startup.coga "$OUTPUT_DIR"/variants/
cp -r *.variant "$OUTPUT_DIR"/variants/
#zip -r "$OUTPUT_DIR".zip "$OUTPUT_DIR"
#rm -r "$OUTPUT_DIR"


exit 0

