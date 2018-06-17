
declare -i pid
#trap "echo Error! Experiments did not successfully complete!; killall gpu_memory_allocator; exit 0" SIGINT SIGTERM ERR SIGKILL
#trap "echo Error! Experiments did not successfully complete!; kill -INT -$pid; killall gpu_memory_allocator; exit 0" SIGINT SIGTERM SIGKILL
trap "echo Error! Experiments did not successfully complete!; killall timeout; killall cogadbd; killall gpu_memory_allocator; sleep 1; exit 0" SIGINT SIGTERM SIGKILL

set -e
export LC_NUMERIC="de_DE.UTF-8"
set +e

rm -f core
ulimit -c unlimited

if [ ! -f SCALE_FACTOR ]
then
    echo "File 'SCALE_FACTOR' not found!"
    echo "Please enter a scale factor (a positive integer):"
    read LINE
    if [[ $LINE = *[[:digit:]]* ]]; then
     echo "$LINE is valid scale factor"
     echo "$LINE" > SCALE_FACTOR
    else
     echo "$LINE is not numeric! Aborting..."
     exit -1;
    fi
fi

if [ ! -f NUMBER_OF_PARALLEL_USERS ]
then
    echo "File 'NUMBER_OF_PARALLEL_USERS' not found!"
    echo "Please enter a scale factor (a positive integer):"
    read LINE
    if [[ $LINE = *[[:digit:]]* ]]; then
     if [ $LINE -gt 0 ]; then
        echo "$LINE is valid number of parallel users"
     else
        echo "Illegal number of parallel users! Aborting..."
        exit -1;
     fi
     echo "$LINE" > NUMBER_OF_PARALLEL_USERS
    else
     echo "$LINE is not numeric! Aborting..."
     exit -1;
    fi
fi

if [ ! -f QUERY_NAME ]
then
    echo "File 'QUERY_NAME' not found!"
    echo "Please enter a query name (possible values: ssb_all, ssb_select):"
    read LINE
    if [[ $LINE == ssb_all || $LINE == ssb_select || $LINE == tpch_all_supported || $LINE == tpch_all_supported_by_cogadb_and_ocelot ]]; then
     echo "$LINE is valid query name!"
     echo "$LINE" > QUERY_NAME
    else
     echo "$LINE is not a valid query name! Aborting..."
     exit -1;
    fi
fi


if [ ! -f GPU_DEVICE_MEMORY_CAPACITY ]
then
    echo "File 'GPU_DEVICE_MEMORY_CAPACITY' not found!"
    echo "Please enter a gpu device memory capacity in bytes:"
    read LINE
    if [[ $LINE = *[[:digit:]]* ]]; then
     echo "$LINE is valid gpu device memory capacity"
     echo "$LINE" > GPU_DEVICE_MEMORY_CAPACITY
    else
     echo "$LINE is not numeric! Aborting..."
     exit -1;
    fi
fi

if [ ! -f PATH_TO_COGADB_EXECUTABLE ]
then
    echo "File 'PATH_TO_COGADB_EXECUTABLE' not found!"
    echo "Please enter the path to the cogadb executable:"
    read LINE
    echo "$LINE" > PATH_TO_COGADB_EXECUTABLE
fi



if [ ! -f PATH_TO_COGADB_DATABASES ]
then
    echo "File 'PATH_TO_COGADB_DATABASES' not found!"
    echo "Please enter the path to the cogadb databases:"
    read LINE
    echo "$LINE" > PATH_TO_COGADB_DATABASES
fi


CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_USERS=false
CONDUCT_EXPERIMENT_VARYING_GPU_BUFFER_SIZE=false
CONDUCT_EXPERIMENT_DATA_PLACEMENT_DRIVEN_QUERY_OPTIMIZATION=false
CONDUCT_EXPERIMENT_VARYING_GPU_BUFFER_MANAGER_STRATEGIES=false
CONDUCT_EXPERIMENT_VARYING_GPU_MEMORY_OCCUPATION=false
CONDUCT_EXPERIMENT_VARYING_QUERY_CHOPPING_CONFIGS=false


if [ $# -lt 1 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 [all|varying_number_of_users|varying_gpu_buffer_size|data_placement_driven_query_optimization|varying_gpu_buffer_manager_strategies|varying_gpu_memory_occupation|varying_qc_configurations|create_diagrams_only]"
	exit -1
fi

if [ "$1" = "varying_number_of_users" ]; then
	CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_USERS=true
elif [ "$1" = "varying_gpu_buffer_size" ]; then
	CONDUCT_EXPERIMENT_VARYING_GPU_BUFFER_SIZE=true
elif [ "$1" = "data_placement_driven_query_optimization" ]; then
	CONDUCT_EXPERIMENT_DATA_PLACEMENT_DRIVEN_QUERY_OPTIMIZATION=true
elif [ "$1" = "varying_gpu_buffer_manager_strategies" ]; then
	CONDUCT_EXPERIMENT_VARYING_GPU_BUFFER_MANAGER_STRATEGIES=true
elif [ "$1" = "varying_gpu_memory_occupation" ]; then
	CONDUCT_EXPERIMENT_VARYING_GPU_MEMORY_OCCUPATION=true
elif [ "$1" = "varying_qc_configurations" ]; then	
	CONDUCT_EXPERIMENT_VARYING_QUERY_CHOPPING_CONFIGS=true
elif [ "$1" = "create_diagrams_only" ]; then
	echo "Generating diagrams only..."
elif [ "$1" = "all" ]; then
	CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_USERS=true
	CONDUCT_EXPERIMENT_VARYING_GPU_BUFFER_SIZE=true
	CONDUCT_EXPERIMENT_DATA_PLACEMENT_DRIVEN_QUERY_OPTIMIZATION=true
	CONDUCT_EXPERIMENT_VARYING_GPU_BUFFER_MANAGER_STRATEGIES=true
	CONDUCT_EXPERIMENT_VARYING_GPU_MEMORY_OCCUPATION=true
	CONDUCT_EXPERIMENT_VARYING_QUERY_CHOPPING_CONFIGS=true
else
	echo "Invalid parameter: $1"
	echo "Usage: $0 [all|varying_number_of_users|varying_gpu_buffer_size|data_placement_driven_query_optimization|varying_gpu_buffer_manager_strategies|varying_gpu_memory_occupation|varying_qc_configurations]"
	exit -1
fi


#Configuration Section
#TOTAL_GPU_DEVICE_MEMORY_CAPACITY=1609564160
TOTAL_GPU_DEVICE_MEMORY_CAPACITY=`cat GPU_DEVICE_MEMORY_CAPACITY`
NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT=1
#ABSOLUTE_PATH_TO_COGADB_EXECUTABLE=/home/sebastian/gpudbms/cogadb/bin/cogadbd
ABSOLUTE_PATH_TO_COGADB_EXECUTABLE=`cat PATH_TO_COGADB_EXECUTABLE`
ABSOLUTE_PATH_TO_GPU_MEMORY_ALLOCATOR=$(dirname $ABSOLUTE_PATH_TO_COGADB_EXECUTABLE)/gpu_memory_allocator
#End Configuration Section

#number of logical cores!
nproc=$(grep -i "processor" /proc/cpuinfo | sort -u | wc -l)
#number of physical cores!
phycore=$(cat /proc/cpuinfo | egrep "core id|physical id" | tr -d "\n" | sed s/physical/\\nphysical/g | grep -v ^$ | sort | uniq | wc -l)

echo "Number of Physical Cores $phycore" 
echo "Number of Logical Cores $nproc"

mkdir -p generated_experiments
#default configuration for experiments
#SCALE_FACTOR=10
SCALE_FACTOR=$(cat SCALE_FACTOR)
#SCALE_FACTOR=15
#SCALE_FACTOR=10
#QUERY_NAME=ssball
QUERY_NAME=`cat QUERY_NAME`
NUMBER_OF_PARALLEL_USERS=$(cat NUMBER_OF_PARALLEL_USERS)
#NUMBER_OF_PARALLEL_USERS=10
NUMBER_OF_QUERIES=100
NUMBER_OF_WARMUP_QUERIES=2
SCRIPT_NAME=benchmark.coga
RUN_TYPE=any
RUN_TYPE_WARMUP_PHASE=any

##in case we have a single-user workload, we configure query chopping to use only physical cores
#if [ $NUMBER_OF_PARALLEL_USERS -eq 1 ]; then
##use always the number of physical cores, do not use hyper threading!
#NUMBER_OF_CPUS="$phycore"
#else
##in case we have a multi-user workload, we configure query chopping to use logical cores, including hyper threading
#NUMBER_OF_CPUS="$nproc"
#fi
#NUMBER_OF_GPUS=3

NUMBER_OF_CPUS=1
NUMBER_OF_GPUS=1

default_optimization_criterion=0
reuse_performance_models=0
track_memory_usage=0
ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION=false
#GPU_BUFFER_SIZE=$(( (TOTAL_GPU_DEVICE_MEMORY_CAPACITY*7)/10 ))
GPU_BUFFER_SIZE=$(( (TOTAL_GPU_DEVICE_MEMORY_CAPACITY)/2 ))


GPU_BUFFERMANAGEMENT_STRATEGY=least_recently_used
SUBFOLDER_NAME=test
#PRELOAD_DATA_IN_GPU_BUFFER=false
PRELOAD_DATA_IN_GPU_BUFFER=true

#HYPE_READY_QUEUE_LENGTH=50

#####################################################################
#External code start
#some util code for floating point math, taken from http://www.linuxjournal.com/content/floating-point-math-bash
# Default scale used by float functions.

float_scale=0


#####################################################################
# Evaluate a floating point number expression.

function float_eval()
{
    local stat=0
    local result=0.0
    if [[ $# -gt 0 ]]; then
        #result=$(echo "scale=$float_scale; $*" | bc -q 2>/dev/null | awk '{print int($1+0.5)}')
        result=$(echo "scale=$float_scale; $*" | bc -q 2>/dev/null | sed -e 's/\./,/g')
        stat=$?
        #result=`printf -v int %.0f "$result"`
        result=`printf %.0f "$result"`
        result=${result%,*}

        #result=$(echo "($result+0.5)/1" | bc -q)
        #result=$(echo "$float" | awk '{print int($1+0.5)}')
        # $(echo "$result" | awk '{print int($1+0.5)}')

        if [[ $stat -eq 0  &&  -z "$result" ]]; then stat=1; fi
    fi
    echo $result
    return $stat
}
#External code end
#####################################################################

pid=-1

function execute_cogadb_experiment {
     local COMMAND=$1
     while [ ! -e finished ]; do
     		ulimit -c unlimited 
     		#wake the NVIDIA Driver from his slumber, if necessary
     		nvidia-smi
     		#ok, we give CoGaDB two hours to perform its job otherwise, it is very likely it hang up, and we restart
			timeout 7200 $ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished &
			#$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished &
			pid=$!
           		wait $pid
			RET=$?
			#if we used STRG+C, terminate experiment
			if [ $RET -eq 130 ]; then break; fi
			#ok, some kind of error occured, repeat
			if [ $RET -ne 0 ]; then
			    tail cogadb_ssb_measurement.log
			    DATE=$(date)
			    mv core core_file_created_at_"$DATE"
			    echo "Error with executing benchmark command: '$COMMAND'!"
			    echo "Repeat Execution..."
			    sleep 1
			fi 
	 done
} 

#bash generate_experiment_config_files.sh $SCALE_FACTOR $QUERY_NAME $NUMBER_OF_PARALLEL_USERS $NUMBER_OF_QUERIES $NUMBER_OF_WARMUP_QUERIES $SCRIPT_NAME $RUN_TYPE $RUN_TYPE_WARMUP_PHASE $NUMBER_OF_CPUS $NUMBER_OF_GPUS $default_optimization_criterion $reuse_performance_models $track_memory_usage $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION $GPU_BUFFER_SIZE $GPU_BUFFERMANAGEMENT_STRATEGY $SUBFOLDER_NAME


##########################################################################################################################
#Experiments for varying number of users
##########################################################################################################################
if $CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_USERS; then

SUBFOLDER_NAME=varying_number_of_parallel_users
#for number_of_users in 1 3 6 10 15 20
for number_of_users in 1 3 10
do
	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	do
		bash generate_experiment_config_files.sh $SCALE_FACTOR $QUERY_NAME $number_of_users $NUMBER_OF_QUERIES $NUMBER_OF_WARMUP_QUERIES $SCRIPT_NAME $RUN_TYPE $RUN_TYPE_WARMUP_PHASE $NUMBER_OF_CPUS $NUMBER_OF_GPUS $default_optimization_criterion $reuse_performance_models $track_memory_usage $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION $GPU_BUFFER_SIZE $GPU_BUFFERMANAGEMENT_STRATEGY $SUBFOLDER_NAME $PRELOAD_DATA_IN_GPU_BUFFER
		
		for experiment in `find ./"generated_experiments/$SUBFOLDER_NAME/" -maxdepth 2 -mindepth 2 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*/*; do
		        current_path=$(pwd)
			cd "$experiment"
			pwd
			#only perform experiment in case no finished file exists, repeat the call until the command is executed successfully
			execute_cogadb_experiment "$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished"
#			while [ ! -e finished ]; do
#			$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished
#			if [ $? -ne 0 ]; then
#			    tail cogadb_ssb_measurement.log
#			    echo "Error with executing benchmark command! Repeat Execution..."
#			fi
#			done
			#create csv table of measurements
			bash ../../../../generate_csv_for_cogadb_parallel_benchmarkresult.sh
			#extract measured variables
			bash ../../../../collect_measured_variables_from_logfile.sh
			cd $current_path
			heuristic=$(basename $experiment)
			echo $heuristic"\t"$SCALE_FACTOR"\t"$QUERY_NAME"\t"$number_of_users"\t"$NUMBER_OF_QUERIES"\t"$NUMBER_OF_WARMUP_QUERIES"\t"$SCRIPT_NAME"\t"$RUN_TYPE"\t"$RUN_TYPE_WARMUP_PHASE"\t"$NUMBER_OF_CPUS"\t"$NUMBER_OF_GPUS"\t"$default_optimization_criterion"\t"$reuse_performance_models"\t"$track_memory_usage"\t"$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION"\t"$GPU_BUFFER_SIZE"\t"$GPU_BUFFERMANAGEMENT_STRATEGY"\t"$SUBFOLDER_NAME > $experiment/experiment_parameters.txt
			#echo "sf $SCALE_FACTOR, #users $number_of_users, #cpus $NUMBER_OF_CPUS, #gpus $NUMBER_OF_GPUS, gpu_buffer_size: $GPU_BUFFER_SIZE, $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION,\n $GPU_BUFFERMANAGEMENT_STRATEGY, mem_bookeeping $track_memory_usage, DPA_OPT $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION, $HOSTNAME" > $experiment/../performance_diagram_title.txt
		done
		
	done
done

fi
##########################################################################################################################
#END Experiments for varying number of users
##########################################################################################################################

##########################################################################################################################
#Experiments for varying GPU buffer size
##########################################################################################################################
if $CONDUCT_EXPERIMENT_VARYING_GPU_BUFFER_SIZE; then


echo $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.1) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.3) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.5) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.7) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.9) 
#exit 0

SUBFOLDER_NAME=varying_gpu_buffer_size
#for number_of_users in 1 3 6 10 15 20
for gpu_buffer_size in $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.1) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.3) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.5) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.7) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.9)    
do
	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	do
		bash generate_experiment_config_files.sh $SCALE_FACTOR $QUERY_NAME $NUMBER_OF_PARALLEL_USERS $NUMBER_OF_QUERIES $NUMBER_OF_WARMUP_QUERIES $SCRIPT_NAME $RUN_TYPE $RUN_TYPE_WARMUP_PHASE $NUMBER_OF_CPUS $NUMBER_OF_GPUS $default_optimization_criterion $reuse_performance_models $track_memory_usage $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION $gpu_buffer_size $GPU_BUFFERMANAGEMENT_STRATEGY $SUBFOLDER_NAME $PRELOAD_DATA_IN_GPU_BUFFER
		#EXPERIMENT_NAME="experiment_$SCALE_FACTOR-$QUERY_NAME-$NUMBER_OF_PARALLEL_USERS-$NUMBER_OF_QUERIES-$NUMBER_OF_WARMUP_QUERIES-$SCRIPT_NAME-$RUN_TYPE-$RUN_TYPE_WARMUP_PHASE-$NUMBER_OF_CPUS-$NUMBER_OF_GPUS-$default_optimization_criterion-$reuse_performance_models-$track_memory_usage-$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION-$gpu_buffer_size-$GPU_BUFFERMANAGEMENT_STRATEGY-$SUBFOLDER_NAME"
		EXPERIMENT_NAME="experiment_$SCALE_FACTOR-$QUERY_NAME-$NUMBER_OF_PARALLEL_USERS-$NUMBER_OF_QUERIES-$NUMBER_OF_WARMUP_QUERIES-$SCRIPT_NAME-$RUN_TYPE-$RUN_TYPE_WARMUP_PHASE-$NUMBER_OF_CPUS-$NUMBER_OF_GPUS-$default_optimization_criterion-$reuse_performance_models-$track_memory_usage-$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION-$gpu_buffer_size-$GPU_BUFFERMANAGEMENT_STRATEGY-$PRELOAD_DATA_IN_GPU_BUFFER-$SUBFOLDER_NAME$GPU_MEMORY_OCCUPATION_STRING"
		for experiment in `find ./"generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME" -maxdepth 1 -mindepth 1 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*/*; do
		        echo "Process Experiment: '$experiment'"
		        current_path=$(pwd)
			cd "$experiment"
			pwd
			#only perform experiment in case no finished file exists, repeat the call until the command is executed successfully
			execute_cogadb_experiment "$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished"
#			while [ ! -e finished ]; do
#			$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished
#			if [ $? -ne 0 ]; then
#			    tail cogadb_ssb_measurement.log
#			    echo "Error with executing benchmark command! Repeat Execution..."
#			fi
#			done
			#create csv table of measurements
			bash ../../../../generate_csv_for_cogadb_parallel_benchmarkresult.sh
			
			#extract measured variables
			bash ../../../../collect_measured_variables_from_logfile.sh
#		NUMBER_OF_EXECUTED_GPU_OPERATORS=`cat "cogadb_ssb_measurement.log" | grep "NUMBER_OF_EXECUTED_GPU_OPERATORS" | awk '{print $2}'` 
#		if [ -z $NUMBER_OF_EXECUTED_GPU_OPERATORS ];then
#		    NUMBER_OF_EXECUTED_GPU_OPERATORS=0
#		fi
#		NUMBER_OF_ABORTED_GPU_OPERATORS=`cat "cogadb_ssb_measurement.log" | grep "NUMBER_OF_ABORTED_GPU_OPERATORS" | awk '{print $2}'`
#		if [ -z $NUMBER_OF_ABORTED_GPU_OPERATORS ];then
#		    NUMBER_OF_ABORTED_GPU_OPERATORS=0
#		fi
#		
#		WORKLOAD_EXECUTION_TIME=`cat "cogadb_ssb_measurement.log" | grep "WORKLOAD EXECUTION TIME" | awk '{print $4}' | sed -e 's/ms//g' | sed -e 's/,.*//g'` 
#		TOTAL_CPU_TIME=`cat "cogadb_ssb_measurement.log" | grep "TOTAL CPU TIME" | awk '{print $4}' | sed -e 's/ms//g' | sed -e 's/,.*//g'` 
#                LOAD_BALANCING_STRATEGY=`cat "cogadb_ssb_measurement.log" | grep "LOAD_BALANCING_STRATEGY" | awk '{print $2}'`
#                USE_MEMORY_COST_MODELS=`cat "cogadb_ssb_measurement.log" | grep "USE_MEMORY_COST_MODELS" | awk '{print $2}'`
#                QUERY_OPTIMIZER_MODE=`cat "cogadb_ssb_measurement.log" | grep "QUERY_OPTIMIZER_MODE" | awk '{print $2}'`
#		WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS=`cat "cogadb_ssb_measurement.log" | grep "TOTAL_LOST_TIME_IN_NS_DUE_TO_ABORTED_GPU_OPERATORS" | awk '{print $2}' | sed -e 's/ms//g' | sed -e 's/,.*//g'`

#		if [ -z $WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS ];then
#		    WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS=0
#		fi

#		QUERY_RUNTIMES=cogascript_timings.log
#		
#		#compute minimal, maximal, and average query execution time in one run as well as the variance in execution time
#		#cat "$i" | grep "Execution Time" | awk '{print $3}' | sed -e 's/\./,/g' -e 's/,.*//g' > tmp
#		cat "$QUERY_RUNTIMES" | awk '{print $2}' | sed -e 's/\./,/g' -e 's/,.*//g' -e 's/ms//g' > tmp
#		QUERY_EXECUTION_STATISTICS=`cat tmp | awk -f ../compute_average.awk`
#		rm tmp
#		
#		#echo -e "$SCALE_FACTOR\t$QUERY_NAME\t$NUMBER_OF_USERS\t$NUMBER_OF_QUERIES\t$NUMBER_OF_CPUS\t$NUMBER_OF_GPUS\t$LOAD_BALANCING_STRATEGY\t$QUERY_OPTIMIZER_MODE\t$USE_MEMORY_COST_MODELS\t$NUMBER_OF_EXECUTED_GPU_OPERATORS\t$NUMBER_OF_ABORTED_GPU_OPERATORS\t$WORKLOAD_EXECUTION_TIME\t$TOTAL_CPU_TIME\t$WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS\t$QUERY_EXECUTION_STATISTICS" >> measured_variables.csv
#		echo -e "$NUMBER_OF_EXECUTED_GPU_OPERATORS\t$NUMBER_OF_ABORTED_GPU_OPERATORS\t$WORKLOAD_EXECUTION_TIME\t$TOTAL_CPU_TIME\t$WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS\t$QUERY_EXECUTION_STATISTICS" >> dependend_variables.csv	

			echo $SCALE_FACTOR"\t"$QUERY_NAME"\t"$NUMBER_OF_PARALLEL_USERS"\t"$NUMBER_OF_QUERIES"\t"$NUMBER_OF_WARMUP_QUERIES"\t"$SCRIPT_NAME"\t"$RUN_TYPE"\t"$RUN_TYPE_WARMUP_PHASE"\t"$NUMBER_OF_CPUS"\t"$NUMBER_OF_GPUS"\t"$default_optimization_criterion"\t"$reuse_performance_models"\t"$track_memory_usage"\t"$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION"\t"$gpu_buffer_size"\t"$GPU_BUFFERMANAGEMENT_STRATEGY"\t"$SUBFOLDER_NAME > experiment_parameters.txt
			#echo "sf $SCALE_FACTOR, #users $NUMBER_OF_PARALLEL_USERS, #cpus $NUMBER_OF_CPUS, #gpus $NUMBER_OF_GPUS, gpu_buffer_size: $gpu_buffer_size, $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION,\n $GPU_BUFFERMANAGEMENT_STRATEGY, mem_bookeeping $track_memory_usage, DPA_OPT $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION, $HOSTNAME" > ../performance_diagram_title.txt
			echo $gpu_buffer_size > ../gpu_buffer_size.txt
			cd $current_path 
		done
		
	done
done


#####################################
#Create buffer size specific diagrams

for heuristic in greedy_heuristic greedy_chainer_heuristic critical_path_heuristic query_chopping; do

rm -f ./"generated_experiments/$SUBFOLDER_NAME/"performance_diagram.gnuplot
touch ./"generated_experiments/$SUBFOLDER_NAME/"performance_diagram.gnuplot
echo -n "			
set title \"$SUBFOLDER_NAME\"
set auto x
set auto y

#set key top right Left reverse samplen 1
set key outside bottom center Left reverse samplen 1
set key box
#set key vertical maxrows 3
#set key width 2.1

set xlabel 'Queries of Star Schema Benchmark'
set ylabel 'Execution Time (s)'
#set yrange [0:300000]
#set xrange [0:14]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

set xtics   (\"Q1.1\" 0.00000, \"Q1.2\" 1.00000, \"Q1.3\" 2.00000, \"Q2.1\" 3.00000, \"Q2.2\" 4.00000,  \"Q2.3\" 5.00000, \"Q3.1\" 6.00000, \"Q3.2\" 7.00000, \"Q3.3\" 8.00000, \"Q3.4\" 9.00000,  \"Q4.1\" 10.00000,  \"Q4.2\" 11.00000,  \"Q4.3\" 12.00000)

		
plot " >> ./"generated_experiments/$SUBFOLDER_NAME/"performance_diagram.gnuplot

cd ./"generated_experiments/$SUBFOLDER_NAME"
pwd

DIRECTORIES=`find . -maxdepth 1 -mindepth 1 -type d`
NUM_ITERATIONS=$(echo $DIRECTORIES | wc -w)
counter=1


for measurement in `find . -maxdepth 1 -mindepth 1 -type d`; do
size_of_buffer=$(cat $measurement/gpu_buffer_size.txt)
#echo -n "'$measurement/experiment_result.csv' using (\$2/1000) title \"greedy_heuristic (gpu buffer: $size_of_buffer)\", \
#'$measurement/experiment_result.csv' using (\$3/1000) title \"greedy_chainer_heuristic (gpu buffer: $size_of_buffer)\", \
#'$measurement/experiment_result.csv' using (\$4/1000) title \"critical_path_heuristic (gpu buffer: $size_of_buffer)\", \
#'$measurement/experiment_result.csv' using (\$5/1000) title \"query_chopping (gpu buffer: $size_of_buffer)\"" >> performance_diagram.gnuplot
echo -n "'$measurement/experiment_result.csv' using (\$2/1000) title \"$heuristic (gpu buffer: $size_of_buffer)\"" >> performance_diagram.gnuplot
if [ $counter -lt $NUM_ITERATIONS ]; then
echo ", \\" >> performance_diagram.gnuplot
fi
let counter++
done


echo "
set output \""$heuristic"_performance_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" >> ./performance_diagram.gnuplot
#gnuplot performance_diagram.gnuplot
mv ./performance_diagram.gnuplot ./"$heuristic"_performance_diagram.gnuplot
gnuplot ./"$heuristic"_performance_diagram.gnuplot
cd ../..
done
#exit 0
#done

fi
##########################################################################################################################
#END Experiments for varying GPU buffer size
##########################################################################################################################


##########################################################################################################################
#Experiments for data placement driven query optimization
##########################################################################################################################
if $CONDUCT_EXPERIMENT_DATA_PLACEMENT_DRIVEN_QUERY_OPTIMIZATION; then


SUBFOLDER_NAME=varying_data_placement_strategy
#for number_of_users in 1 3 6 10 15 20
for enable_dataplacement_aware_query_optimization in "true" "false"
do
    #disable memory cost models for now
    for vary_track_memory_usage in 0; do
         #we need to prelaod the data, otherwise the GPU will not be used
         for vary_preload_data_in_gpu_buffer in "true"; do

		bash generate_experiment_config_files.sh $SCALE_FACTOR $QUERY_NAME $NUMBER_OF_PARALLEL_USERS $NUMBER_OF_QUERIES $NUMBER_OF_WARMUP_QUERIES $SCRIPT_NAME $RUN_TYPE $RUN_TYPE_WARMUP_PHASE $NUMBER_OF_CPUS $NUMBER_OF_GPUS $default_optimization_criterion $reuse_performance_models $vary_track_memory_usage $enable_dataplacement_aware_query_optimization $GPU_BUFFER_SIZE $GPU_BUFFERMANAGEMENT_STRATEGY $SUBFOLDER_NAME $vary_preload_data_in_gpu_buffer
		
#		for experiment in `find ./"generated_experiments/$SUBFOLDER_NAME/" -maxdepth 1 -mindepth 1 -type d`; do
#		     echo "$experiment"
#					echo $heuristic"\t"$SCALE_FACTOR"\t"$QUERY_NAME"\t"$NUMBER_OF_PARALLEL_USERS"\t"$NUMBER_OF_QUERIES"\t"$NUMBER_OF_WARMUP_QUERIES"\t"$SCRIPT_NAME"\t"$RUN_TYPE"\t"$RUN_TYPE_WARMUP_PHASE"\t"$NUMBER_OF_CPUS"\t"$NUMBER_OF_GPUS"\t"$default_optimization_criterion"\t"$reuse_performance_models"\t"$vary_track_memory_usage"\t"$enable_dataplacement_aware_query_optimization"\t"$GPU_BUFFER_SIZE"\t"$GPU_BUFFERMANAGEMENT_STRATEGY"\t"$SUBFOLDER_NAME > $experiment/experiment_parameters.txt
#			echo "sf $SCALE_FACTOR, #users $NUMBER_OF_PARALLEL_USERS, #cpus $NUMBER_OF_CPUS, #gpus $NUMBER_OF_GPUS, gpu_buffer_size: $GPU_BUFFER_SIZE, \n $GPU_BUFFERMANAGEMENT_STRATEGY, mem_bookeeping ${vary_track_memory_usage}, DPA_OPT ${enable_dataplacement_aware_query_optimization}, $HOSTNAME" > $experiment/performance_diagram_title.txt
#		done
		
	#for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	#do
		for experiment in `find ./"generated_experiments/$SUBFOLDER_NAME/" -maxdepth 2 -mindepth 2 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*/*; do
		        current_path=$(pwd)
			cd "$experiment"
			pwd
			#only perform experiment in case no finished file exists, repeat the call until the command is executed successfully
			execute_cogadb_experiment "$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished"
#			while [ ! -e finished ]; do
#			$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished
#			if [ $? -ne 0 ]; then
#			    tail cogadb_ssb_measurement.log
#			    echo "Error with executing benchmark command! Repeat Execution..."
#			fi
#			done
			#create csv table of measurements
			bash ../../../../generate_csv_for_cogadb_parallel_benchmarkresult.sh
			#extract measured variables
			bash ../../../../collect_measured_variables_from_logfile.sh
			cd $current_path 
			heuristic=$(basename $experiment)
			echo $heuristic"\t"$SCALE_FACTOR"\t"$QUERY_NAME"\t"$NUMBER_OF_PARALLEL_USERS"\t"$NUMBER_OF_QUERIES"\t"$NUMBER_OF_WARMUP_QUERIES"\t"$SCRIPT_NAME"\t"$RUN_TYPE"\t"$RUN_TYPE_WARMUP_PHASE"\t"$NUMBER_OF_CPUS"\t"$NUMBER_OF_GPUS"\t"$default_optimization_criterion"\t"$reuse_performance_models"\t"$vary_track_memory_usage"\t"$enable_dataplacement_aware_query_optimization"\t"$GPU_BUFFER_SIZE"\t"$GPU_BUFFERMANAGEMENT_STRATEGY"\t"$SUBFOLDER_NAME > $experiment/experiment_parameters.txt
##			echo "sf $SCALE_FACTOR, #users $NUMBER_OF_PARALLEL_USERS, #cpus $NUMBER_OF_CPUS, #gpus $NUMBER_OF_GPUS, gpu_buffer_size: $GPU_BUFFER_SIZE, \n $GPU_BUFFERMANAGEMENT_STRATEGY, mem_bookeeping ${vary_track_memory_usage}, DPA_OPT ${enable_dataplacement_aware_query_optimization}, $HOSTNAME" > $experiment/../performance_diagram_title.txt
			#read line
		done
	#done
	done
    done
done


fi
##########################################################################################################################
#END Experiments for data placement driven query optimization
##########################################################################################################################

##########################################################################################################################
#Experiments for varying gpu buffer manager strategies
##########################################################################################################################
if $CONDUCT_EXPERIMENT_VARYING_GPU_BUFFER_MANAGER_STRATEGIES; then


SUBFOLDER_NAME=varying_gpu_buffer_manager_strategies
for gpu_buffer_manager_strategy in least_recently_used least_frequently_used
do
	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	do
		bash generate_experiment_config_files.sh $SCALE_FACTOR $QUERY_NAME $NUMBER_OF_PARALLEL_USERS $NUMBER_OF_QUERIES $NUMBER_OF_WARMUP_QUERIES $SCRIPT_NAME $RUN_TYPE $RUN_TYPE_WARMUP_PHASE $NUMBER_OF_CPUS $NUMBER_OF_GPUS $default_optimization_criterion $reuse_performance_models $track_memory_usage $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION $GPU_BUFFER_SIZE $gpu_buffer_manager_strategy $SUBFOLDER_NAME $PRELOAD_DATA_IN_GPU_BUFFER
		
		for experiment in `find ./"generated_experiments/$SUBFOLDER_NAME/" -maxdepth 2 -mindepth 2 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*/*; do
		    current_path=$(pwd)
			cd "$experiment"
			pwd
			#only perform experiment in case no finished file exists, repeat the call until the command is executed successfully
			execute_cogadb_experiment "$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished"
#			while [ ! -e finished ]; do
#			$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished
#			if [ $? -ne 0 ]; then
#			    tail cogadb_ssb_measurement.log
#			    echo "Error with executing benchmark command! Repeat Execution..."
#			fi
#			done
			#create csv table of measurements
			bash ../../../../generate_csv_for_cogadb_parallel_benchmarkresult.sh
			#extract measured variables
			bash ../../../../collect_measured_variables_from_logfile.sh
			cd $current_path 
			heuristic=$(basename $experiment)
			echo $heuristic"\t"$SCALE_FACTOR"\t"$QUERY_NAME"\t"$NUMBER_OF_PARALLEL_USERS"\t"$NUMBER_OF_QUERIES"\t"$NUMBER_OF_WARMUP_QUERIES"\t"$SCRIPT_NAME"\t"$RUN_TYPE"\t"$RUN_TYPE_WARMUP_PHASE"\t"$NUMBER_OF_CPUS"\t"$NUMBER_OF_GPUS"\t"$default_optimization_criterion"\t"$reuse_performance_models"\t"$track_memory_usage"\t"$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION"\t"$GPU_BUFFER_SIZE"\t"$gpu_buffer_manager_strategy"\t"$SUBFOLDER_NAME > $experiment/experiment_parameters.txt
			#echo "sf $SCALE_FACTOR, #users $NUMBER_OF_PARALLEL_USERS, #cpus $NUMBER_OF_CPUS, #gpus $NUMBER_OF_GPUS, gpu_buffer_size: $GPU_BUFFER_SIZE, $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION,\n $gpu_buffer_manager_strategy, mem_bookeeping $track_memory_usage, DPA_OPT $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION, $HOSTNAME" > $experiment/../performance_diagram_title.txt
		done
		
	done
done

fi
##########################################################################################################################
#END Experiments for varying gpu buffer manager strategies
##########################################################################################################################

##########################################################################################################################
#Experiments for varying gpu memory occupation
##########################################################################################################################
if $CONDUCT_EXPERIMENT_VARYING_GPU_MEMORY_OCCUPATION; then 
	
SUBFOLDER_NAME=varying_gpu_memory_occupation
for gpu_memory_occupation in 1 $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.1) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.3) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.5) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.7) $(float_eval $TOTAL_GPU_DEVICE_MEMORY_CAPACITY*0.9) 
do
#    for vary_track_memory_usage in 0 1; do
    for vary_track_memory_usage in 0; do    
	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	do
#                echo "gpu_memory_occupation: $gpu_memory_occupation    track_mem_usage: $vary_track_memory_usage"
#                echo bash -c "cd \"generated_experiments/$SUBFOLDER_NAME\"; mv \"$EXPERIMENT_NAME\" \"$EXPERIMENT_NAME-$gpu_memory_occupation\""
		bash generate_experiment_config_files.sh $SCALE_FACTOR $QUERY_NAME $NUMBER_OF_PARALLEL_USERS $NUMBER_OF_QUERIES $NUMBER_OF_WARMUP_QUERIES $SCRIPT_NAME $RUN_TYPE $RUN_TYPE_WARMUP_PHASE $NUMBER_OF_CPUS $NUMBER_OF_GPUS $default_optimization_criterion $reuse_performance_models $vary_track_memory_usage $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION $GPU_BUFFER_SIZE $GPU_BUFFERMANAGEMENT_STRATEGY $SUBFOLDER_NAME $PRELOAD_DATA_IN_GPU_BUFFER $gpu_memory_occupation

		EXPERIMENT_NAME="experiment_$SCALE_FACTOR-$QUERY_NAME-$NUMBER_OF_PARALLEL_USERS-$NUMBER_OF_QUERIES-$NUMBER_OF_WARMUP_QUERIES-$SCRIPT_NAME-$RUN_TYPE-$RUN_TYPE_WARMUP_PHASE-$NUMBER_OF_CPUS-$NUMBER_OF_GPUS-$default_optimization_criterion-$reuse_performance_models-$vary_track_memory_usage-$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION-$GPU_BUFFER_SIZE-$GPU_BUFFERMANAGEMENT_STRATEGY-$PRELOAD_DATA_IN_GPU_BUFFER-$SUBFOLDER_NAME-$gpu_memory_occupation"
		#EXPERIMENT_NAME="experiment_$SCALE_FACTOR-$QUERY_NAME-$NUMBER_OF_PARALLEL_USERS-$NUMBER_OF_QUERIES-$NUMBER_OF_WARMUP_QUERIES-$SCRIPT_NAME-$RUN_TYPE-$RUN_TYPE_WARMUP_PHASE-$NUMBER_OF_CPUS-$NUMBER_OF_GPUS-$default_optimization_criterion-$reuse_performance_models-$track_memory_usage-$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION-$GPU_BUFFER_SIZE-$GPU_BUFFERMANAGEMENT_STRATEGY-$PRELOAD_DATA_IN_GPU_BUFFER-$SUBFOLDER_NAME$GPU_MEMORY_OCCUPATION_STRING"

		#bash -c "cd \"generated_experiments/$SUBFOLDER_NAME\"; mv -v \"$EXPERIMENT_NAME\" \"$EXPERIMENT_NAME-$gpu_memory_occupation\""
#		cp -r "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME" "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME-$gpu_memory_occupation"
#		rm -rf "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME"
		
		#mv "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME" "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME-$gpu_memory_occupation"
		echo "sf $SCALE_FACTOR, #users $NUMBER_OF_PARALLEL_USERS, #cpus $NUMBER_OF_CPUS, #gpus $NUMBER_OF_GPUS, gpu_buffer_size: $GPU_BUFFER_SIZE, \n $GPU_BUFFERMANAGEMENT_STRATEGY, mem_bookeeping $vary_track_memory_usage, DPA_OPT $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION, $HOSTNAME \n MEM_OCC $gpu_memory_occupation" > "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME-$gpu_memory_occupation"/performance_diagram_title.txt  
		#bash -c "cd \"generated_experiments/$SUBFOLDER_NAME\"; mv -v \"$EXPERIMENT_NAME\" \"$EXPERIMENT_NAME-$gpu_memory_occupation\"; cd ../.."
		#cp -r "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME" "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME-$gpu_memory_occupation"
		#rm -rf "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME"


		for experiment in `find ./"generated_experiments/$SUBFOLDER_NAME/" -maxdepth 2 -mindepth 2 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*/*; do
		        current_path=$(pwd)
			cd "$experiment"
			pwd
			#only perform experiment in case no finished file exists, repeat the call until the command is executed successfully
			$ABSOLUTE_PATH_TO_GPU_MEMORY_ALLOCATOR $gpu_memory_occupation &
			execute_cogadb_experiment "$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished"
			killall gpu_memory_allocator
#			while [ ! -e finished ]; do
#			$ABSOLUTE_PATH_TO_GPU_MEMORY_ALLOCATOR $gpu_memory_occupation &
#			$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished
#			if [ $? -ne 0 ]; then
#			    tail cogadb_ssb_measurement.log
#			    echo "Error with executing benchmark command! Repeat Execution..."
#			fi
#			killall gpu_memory_allocator
#			done
			
			#create csv table of measurements
			bash ../../../../generate_csv_for_cogadb_parallel_benchmarkresult.sh
			#extract measured variables
			bash ../../../../collect_measured_variables_from_logfile.sh
			cd $current_path 
			heuristic=$(basename $experiment)
			echo $heuristic"\t"$SCALE_FACTOR"\t"$QUERY_NAME"\t"$NUMBER_OF_PARALLEL_USERS"\t"$NUMBER_OF_QUERIES"\t"$NUMBER_OF_WARMUP_QUERIES"\t"$SCRIPT_NAME"\t"$RUN_TYPE"\t"$RUN_TYPE_WARMUP_PHASE"\t"$NUMBER_OF_CPUS"\t"$NUMBER_OF_GPUS"\t"$default_optimization_criterion"\t"$reuse_performance_models"\t"$vary_track_memory_usage"\t"$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION"\t"$GPU_BUFFER_SIZE"\t"$GPU_BUFFERMANAGEMENT_STRATEGY"\t"$SUBFOLDER_NAME > $experiment/experiment_parameters.txt
			
		done
		
	done
    done
done

fi
##########################################################################################################################
#END Experiments for varying gpu memory occupation
##########################################################################################################################

##########################################################################################################################
#Experiments for varying query chopping configurations
##########################################################################################################################
if $CONDUCT_EXPERIMENT_VARYING_QUERY_CHOPPING_CONFIGS; then

SUBFOLDER_NAME=varying_number_of_query_chopping_configurations

ORIGINAL_HYPE_READY_QUEUE_LENGTH="$HYPE_READY_QUEUE_LENGTH"

for number_of_cpus in 4 8 16
do
  for number_of_gpus in 0 1 2 4
  do
    for hype_ready_queue_length in 10 50 75; do
        #export HYPE_READY_QUEUE_LENGTH=$hype_ready_queue_length
	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	do
		bash generate_experiment_config_files.sh $SCALE_FACTOR $QUERY_NAME $NUMBER_OF_PARALLEL_USERS $NUMBER_OF_QUERIES $NUMBER_OF_WARMUP_QUERIES $SCRIPT_NAME $RUN_TYPE $RUN_TYPE_WARMUP_PHASE $number_of_cpus $number_of_gpus $default_optimization_criterion $reuse_performance_models $track_memory_usage $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION $GPU_BUFFER_SIZE $GPU_BUFFERMANAGEMENT_STRATEGY $SUBFOLDER_NAME $PRELOAD_DATA_IN_GPU_BUFFER 0 $hype_ready_queue_length
		rm -rf generated_experiments/$SUBFOLDER_NAME/*/greedy_heuristic
		rm -rf generated_experiments/$SUBFOLDER_NAME/*/greedy_chainer_heuristic
		rm -rf generated_experiments/$SUBFOLDER_NAME/*/greedy_heuristic_cpu_only
		rm -rf generated_experiments/$SUBFOLDER_NAME/*/critical_path_heuristic
		rm -rf generated_experiments/$SUBFOLDER_NAME/*/best_effort_gpu_heuristic
		for experiment in `find ./"generated_experiments/$SUBFOLDER_NAME/" -maxdepth 2 -mindepth 2 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*/*; do
		        current_path=$(pwd)
			cd "$experiment"
			pwd
			#only perform experiment in case no finished file exists, repeat the call until the command is executed successfully
			execute_cogadb_experiment "$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished"
#			while [ ! -e finished ]; do
#			$ABSOLUTE_PATH_TO_COGADB_EXECUTABLE benchmark.coga &> cogadb_ssb_measurement.log && touch finished
#			if [ $? -ne 0 ]; then
#			    tail cogadb_ssb_measurement.log
#			    echo "Error with executing benchmark command! Repeat Execution..."
#			fi
#			done
			#create csv table of measurements
			bash ../../../../generate_csv_for_cogadb_parallel_benchmarkresult.sh
			#extract measured variables
			bash ../../../../collect_measured_variables_from_logfile.sh
			#echo $gpu_buffer_size > ../gpu_buffer_size.txt



			cd $current_path 
			heuristic=$(basename $experiment)
			echo $heuristic"\t"$SCALE_FACTOR"\t"$QUERY_NAME"\t"$NUMBER_OF_PARALLEL_USERS"\t"$NUMBER_OF_QUERIES"\t"$NUMBER_OF_WARMUP_QUERIES"\t"$SCRIPT_NAME"\t"$RUN_TYPE"\t"$RUN_TYPE_WARMUP_PHASE"\t"$number_of_cpus"\t"$NUMBER_OF_GPUS"\t"$default_optimization_criterion"\t"$reuse_performance_models"\t"$track_memory_usage"\t"$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION"\t"$GPU_BUFFER_SIZE"\t"$GPU_BUFFERMANAGEMENT_STRATEGY"\t"$SUBFOLDER_NAME > $experiment/experiment_parameters.txt
#			echo "$number_of_cpus" > $experiment/number_of_cpus.txt
#			echo "$hype_ready_queue_length" > $experiment/hype_ready_queue_length.txt		
			#echo "sf $SCALE_FACTOR, #users $number_of_users, #cpus $NUMBER_OF_CPUS, #gpus $NUMBER_OF_GPUS, gpu_buffer_size: $GPU_BUFFER_SIZE, $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION,\n $GPU_BUFFERMANAGEMENT_STRATEGY, mem_bookeeping $track_memory_usage, DPA_OPT $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION, $HOSTNAME" > $experiment/../performance_diagram_title.txt
		done
		
	done
    done
  done
done
#####################################
#Create query chopping specific diagrams

for query_chopping; do

rm -f ./"generated_experiments/$SUBFOLDER_NAME/"performance_diagram.gnuplot
touch ./"generated_experiments/$SUBFOLDER_NAME/"performance_diagram.gnuplot
echo -n "			
set title \"$SUBFOLDER_NAME\"
set auto x
set auto y

#set key top right Left reverse samplen 1
set key outside bottom center Left reverse samplen 1
set key box
#set key vertical maxrows 3
#set key width 2.1

set xlabel 'Queries of Star Schema Benchmark'
set ylabel 'Execution Time (s)'
#set yrange [0:300000]
#set xrange [0:14]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

set xtics   (\"Q1.1\" 0.00000, \"Q1.2\" 1.00000, \"Q1.3\" 2.00000, \"Q2.1\" 3.00000, \"Q2.2\" 4.00000,  \"Q2.3\" 5.00000, \"Q3.1\" 6.00000, \"Q3.2\" 7.00000, \"Q3.3\" 8.00000, \"Q3.4\" 9.00000,  \"Q4.1\" 10.00000,  \"Q4.2\" 11.00000,  \"Q4.3\" 12.00000)

		
plot " >> ./"generated_experiments/$SUBFOLDER_NAME/"performance_diagram.gnuplot

cd ./"generated_experiments/$SUBFOLDER_NAME"
pwd

DIRECTORIES=`find . -maxdepth 1 -mindepth 1 -type d`
NUM_ITERATIONS=$(echo $DIRECTORIES | wc -w)
counter=1


for measurement in `find . -maxdepth 1 -mindepth 1 -type d`; do

                        DIR=$(pwd)
                        cd $measurement
			for i in \# {0..12}; do echo $i; done > linenumbers
			for i in `find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'`; do
			    echo $i > measurements_$i 
			    cat "$i"/averaged_ssb* >> measurements_$i
			done
			paste linenumbers measurements_* > experiment_result.csv
			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)
			cd $DIR


#size_of_buffer=$(cat $measurement/gpu_buffer_size.txt)
number_of_cpus=$(cat $measurement/query_chopping/number_of_cpus.txt)
number_of_gpus=$(cat $measurement/query_chopping/number_of_gpus.txt)
hype_ready_queue_length=$(cat $measurement/query_chopping/hype_ready_queue_length.txt)

echo -n "'$measurement/experiment_result.csv' using (\$2/1000) title \"$heuristic (#cpus: $number_of_cpus, #gpus: $number_of_gpus, RQL: $hype_ready_queue_length)\"" >> performance_diagram.gnuplot
if [ $counter -lt $NUM_ITERATIONS ]; then
    echo ", \\" >> performance_diagram.gnuplot
fi
let counter++
done


echo "
set output \""$heuristic"_performance_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" >> ./performance_diagram.gnuplot
#gnuplot performance_diagram.gnuplot
mv ./performance_diagram.gnuplot ./"$heuristic"_performance_diagram.gnuplot
gnuplot ./"$heuristic"_performance_diagram.gnuplot
cd ../..
done
#exit 0
#done


#END Create query chopping specific diagrams
#####################################


export HYPE_READY_QUEUE_LENGTH="$ORIGINAL_HYPE_READY_QUEUE_LENGTH"
fi
##########################################################################################################################
#END Experiments for varying query chopping configurations
##########################################################################################################################

if [[ "$QUERY_NAME" == "ssb"* ]]; then
##########################################################################################################################		
#build SSB performance diagrams for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*; do
		        
#			for i in \# {1..13}; do
#			    echo "$i" > "generated_experiments/$SUBFOLDER_NAME/"linenumbers
#			done
			current_path=$(pwd)
			cd $experiment
			pwd
			for i in \# {0..12}; do echo $i; done > linenumbers
			for i in `find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'`; do
			    echo $i > measurements_$i 
			    cat "$i"/averaged_ssb* >> measurements_$i
			    #headers
			    echo "AVERAGE" > "$i"/average_performance_measurements
			    echo "MINIMUM" > "$i"/min_performance_measurements
			    echo "MAXIMUM" > "$i"/max_performance_measurements
			    #measurement data
			    cat "$i"/averaged_ssb* >> "$i"/average_performance_measurements
			    cat "$i"/min_ssb* >> "$i"/min_performance_measurements
			    cat "$i"/max_ssb* >> "$i"/max_performance_measurements
			    paste linenumbers "$i"/average_performance_measurements "$i"/min_performance_measurements "$i"/max_performance_measurements > "$i"/performance_measurements.csv
			done
			paste linenumbers measurements_* > experiment_result.csv

			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

#set key top right Left reverse samplen 1
#set key box
#set key vertical maxrows 3
#set key width 2.1

set key top right Left reverse samplen 1
set key below
set key box
#set key inside above vertical maxrows 1
set key vertical maxrows 3
set key width 2.1


#set xlabel 'Queries of Star Schema Benchmark'
set ylabel 'Execution Time (s)'
#set yrange [0:300000]
#set xrange [0:14]
set style data histogram
#set style histogram cluster gap 1
#set style histogram errorbars gap 1 lw 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"
#set style fill pattern 0 border

set xtics   (\"Q1.1\" 0.00000, \"Q1.2\" 1.00000, \"Q1.3\" 2.00000, \"Q2.1\" 3.00000, \"Q2.2\" 4.00000,  \"Q2.3\" 5.00000, \"Q3.1\" 6.00000, \"Q3.2\" 7.00000, \"Q3.3\" 8.00000, \"Q3.4\" 9.00000,  \"Q4.1\" 10.00000,  \"Q4.2\" 11.00000,  \"Q4.3\" 12.00000)

#best_effort_gpu_heuristic       critical_path_heuristic greedy_chainer_heuristic        greedy_heuristic        greedy_heuristic_cpu_only       query_chopping	
#plot 'experiment_result.csv' using (\$6/1000) title \"greedy_heuristic_cpu_only\", \
#'experiment_result.csv' using (\$5/1000) title \"greedy_heuristic\", \
#'experiment_result.csv' using (\$4/1000) title \"greedy_chainer_heuristic\", \
#'experiment_result.csv' using (\$3/1000) title \"critical_path_heuristic\", \
#'experiment_result.csv' using (\$2/1000) title \"best_effort_gpu_heuristic\", \
#'experiment_result.csv' using (\$7/1000) title \"query_chopping\"

set style histogram cluster gap 1
plot 'greedy_heuristic_cpu_only/performance_measurements.csv' using (\$2/1000) title \"greedy_heuristic_cpu_only\", \
'greedy_heuristic/performance_measurements.csv' using (\$2/1000) title \"greedy_heuristic\", \
'greedy_chainer_heuristic/performance_measurements.csv' using (\$2/1000) title \"greedy_chainer_heuristic\", \
'critical_path_heuristic/performance_measurements.csv' using (\$2/1000) title \"critical_path_heuristic\", \
'best_effort_gpu_heuristic/performance_measurements.csv' using (\$2/1000) title \"best_effort_gpu_heuristic\", \
'query_chopping/performance_measurements.csv' using (\$2/1000) title \"query_chopping\"

#set style histogram errorbars gap 1 lw 1
#plot 'greedy_heuristic_cpu_only/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"greedy_heuristic_cpu_only\", \
#'greedy_heuristic/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"greedy_heuristic\", \
#'greedy_chainer_heuristic/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"greedy_chainer_heuristic\", \
#'critical_path_heuristic/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"critical_path_heuristic\", \
#'best_effort_gpu_heuristic/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"best_effort_gpu_heuristic\", \
#'query_chopping/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"query_chopping\"

set output \"performance_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > performance_diagram.gnuplot

echo "
set title \"$diagram_title\"
set auto x
set auto y

#set key top right Left reverse samplen 1
#set key box
#set key vertical maxrows 3
#set key width 2.1

set key top right Left reverse samplen 1
set key below
set key box
#set key inside above vertical maxrows 1
set key vertical maxrows 3
set key width 2.1


#set xlabel 'Queries of Star Schema Benchmark'
set ylabel 'Speedup compared to CPU Only' offset 0,-1,5
#set yrange [0:300000]
#set xrange [0:14]
set style data histogram
#set style histogram cluster gap 1
#set style histogram errorbars gap 1 lw 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"
#set style fill pattern 0 border

set xtics   (\"Q1.1\" 0.00000, \"Q1.2\" 1.00000, \"Q1.3\" 2.00000, \"Q2.1\" 3.00000, \"Q2.2\" 4.00000,  \"Q2.3\" 5.00000, \"Q3.1\" 6.00000, \"Q3.2\" 7.00000, \"Q3.3\" 8.00000, \"Q3.4\" 9.00000,  \"Q4.1\" 10.00000,  \"Q4.2\" 11.00000,  \"Q4.3\" 12.00000)

#best_effort_gpu_heuristic       critical_path_heuristic greedy_chainer_heuristic        greedy_heuristic        greedy_heuristic_cpu_only       query_chopping	
#plot 'experiment_result.csv' using (\$6/1000) title greedy_heuristic_cpu_only, #'experiment_result.csv' using (\$5/1000) title greedy_heuristic, 'experiment_result.csv' using (\$4/1000) title greedy_chainer_heuristic, #'experiment_result.csv' using (\$3/1000) title critical_path_heuristic, 'experiment_result.csv' using (\$2/1000) title best_effort_gpu_heuristic, #'experiment_result.csv' using (\$7/1000) title query_chopping

set style histogram cluster gap 1
plot 'experiment_result.csv' using (\$6/\$5) title \"Greedy Heuristic\", 'experiment_result.csv' using (\$6/\$4) title \"Greedy Chainer Heuristic\", 'experiment_result.csv' using (\$6/\$3) title \"Critical Path Heuristic\", 'experiment_result.csv' using (\$6/\$2) title \"Best Effort GPU Heuristic\", 'experiment_result.csv' using (\$6/\$7) title \"Query Chopping\"

set output \"speedup_queries_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > speedup_queries_diagram.gnuplot
         gnuplot performance_diagram.gnuplot
         gnuplot speedup_queries_diagram.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build performance diagrams for all experiments
##########################################################################################################################
elif [[ "$QUERY_NAME" == "tpch"* ]]; then
##########################################################################################################################		
#build TPC-H performance diagrams for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*; do
		        
#			for i in \# {1..13}; do
#			    echo "$i" > "generated_experiments/$SUBFOLDER_NAME/"linenumbers
#			done
			current_path=$(pwd)
			cd $experiment
			pwd
			for i in \# {0..21}; do echo $i; done > linenumbers
			for i in `find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'`; do
			    echo $i > measurements_$i 
			    cat $(ls "$i"/averaged_tpch* | sort -V) >> measurements_$i
			    cat $(ls "$i"/lower_bound_of_error_bar_tpch* | sort -V) >> lower_bound_of_error_bar_measurements_$i
			    cat $(ls "$i"/upper_bound_of_error_bar_tpch* | sort -V) >> upper_bound_of_error_bar_measurements_$i
			    #headers
			    echo "AVERAGE" > "$i"/average_performance_measurements
			    echo "MINIMUM" > "$i"/min_performance_measurements
			    echo "MAXIMUM" > "$i"/max_performance_measurements
			    echo "VARIANCE" > "$i"/variance_performance_measurements
			    echo "STANDARD_DEVIATION" > "$i"/standard_deviation_performance_measurements
			    echo "STANDARD_ERROR_MEAN" > "$i"/standard_error_of_the_mean_performance_measurements
			    echo "LOWER_ERROR_BAR" > "$i"/lower_bound_of_error_bar_performance_measurements
			    echo "UPPER_ERROR_BAR" > "$i"/upper_bound_of_error_bar_performance_measurements
			    echo "LOWER_STD_ERR_MEAN_BAR" > "$i"/lower_bound_of_standard_error_of_the_mean_error_bar_performance_measurements
			    echo "UPPER_STD_ERR_MEAN_BAR" > "$i"/upper_bound_of_standard_error_of_the_mean_error_bar_performance_measurements
			    
			    #measurement data
			    cat $(ls "$i"/averaged_tpch* | sort -V) >> "$i"/average_performance_measurements
			    cat $(ls "$i"/min_tpch* | sort -V) >> "$i"/min_performance_measurements
			    cat $(ls "$i"/max_tpch* | sort -V) >> "$i"/max_performance_measurements

			    cat $(ls "$i"/variance_tpch* | sort -V) >> "$i"/variance_performance_measurements			    
			    cat $(ls "$i"/standard_deviation_tpch* | sort -V) >> "$i"/standard_deviation_performance_measurements
			    cat $(ls "$i"/standard_error_of_the_mean_tpch* | sort -V) >> "$i"/standard_error_of_the_mean_performance_measurements
			    cat $(ls "$i"/lower_bound_of_error_bar_tpch* | sort -V) >> "$i"/lower_bound_of_error_bar_performance_measurements
			    cat $(ls "$i"/upper_bound_of_error_bar_tpch* | sort -V) >> "$i"/upper_bound_of_error_bar_performance_measurements
			    cat $(ls "$i"/lower_bound_of_standard_error_of_the_mean_error_bar_tpch* | sort -V) >> "$i"/lower_bound_of_standard_error_of_the_mean_error_bar_performance_measurements
			    cat $(ls "$i"/upper_bound_of_standard_error_of_the_mean_error_bar_tpch* | sort -V) >> "$i"/upper_bound_of_standard_error_of_the_mean_error_bar_performance_measurements
			    
			    #paste linenumbers "$i"/average_performance_measurements "$i"/min_performance_measurements "$i"/max_performance_measurements > "$i"/performance_measurements.csv
			    paste linenumbers "$i"/average_performance_measurements "$i"/min_performance_measurements "$i"/max_performance_measurements "$i"/variance_performance_measurements "$i"/standard_deviation_performance_measurements "$i"/standard_error_of_the_mean_performance_measurements "$i"/lower_bound_of_error_bar_performance_measurements "$i"/upper_bound_of_error_bar_performance_measurements "$i"/lower_bound_of_standard_error_of_the_mean_error_bar_performance_measurements "$i"/upper_bound_of_standard_error_of_the_mean_error_bar_performance_measurements > "$i"/performance_measurements.csv
			done
			paste linenumbers measurements_* lower_bound_of_error_bar_measurements_* upper_bound_of_error_bar_measurements_* > experiment_result.csv

			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

#set key top right Left reverse samplen 1
#set key box
#set key vertical maxrows 3
#set key width 2.1

set key top right Left reverse samplen 1
set key below
set key box
#set key inside above vertical maxrows 1
set key vertical maxrows 3
set key width 2.1


#set xlabel 'Queries of Star Schema Benchmark'
set ylabel 'Execution Time (s)'
#set yrange [0:300000]
#set xrange [0:14]
set style data histogram
#set style histogram cluster gap 1
#set style histogram errorbars gap 1 lw 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"
#set style fill pattern 0 border

set xtics border in scale 0,0 nomirror rotate by -45  autojustify
set xtics   (\"Q1\" 0.00000,\"Q2\" 1.00000,\"Q3\" 2.00000,\"Q4\" 3.00000,\"Q5\" 4.00000,\"Q6\" 5.00000,\"Q7\" 6.00000,\"Q8\" 7.00000,\"Q9\" 8.00000,\"Q10\" 9.00000,\"Q11\" 10.00000,\"Q12\" 11.00000,\"Q13\" 12.00000,\"Q14\" 13.00000,\"Q15\" 14.00000,\"Q16\" 15.00000,\"Q17\" 16.00000,\"Q18\" 17.00000,\"Q19\" 18.00000,\"Q20\" 19.00000,\"Q21\" 20.00000,\"Q22\" 21.00000)

#best_effort_gpu_heuristic       critical_path_heuristic greedy_chainer_heuristic        greedy_heuristic        greedy_heuristic_cpu_only       query_chopping	
#plot 'experiment_result.csv' using (\$6/1000) title \"greedy_heuristic_cpu_only\", \
#'experiment_result.csv' using (\$5/1000) title \"greedy_heuristic\", \
#'experiment_result.csv' using (\$4/1000) title \"greedy_chainer_heuristic\", \
#'experiment_result.csv' using (\$3/1000) title \"critical_path_heuristic\", \
#'experiment_result.csv' using (\$2/1000) title \"best_effort_gpu_heuristic\", \
#'experiment_result.csv' using (\$7/1000) title \"query_chopping\"

set style histogram cluster gap 1
plot 'greedy_heuristic_cpu_only/performance_measurements.csv' using (\$2/1000) title \"greedy_heuristic_cpu_only\", \
'greedy_heuristic/performance_measurements.csv' using (\$2/1000) title \"greedy_heuristic\", \
'greedy_chainer_heuristic/performance_measurements.csv' using (\$2/1000) title \"greedy_chainer_heuristic\", \
'critical_path_heuristic/performance_measurements.csv' using (\$2/1000) title \"critical_path_heuristic\", \
'best_effort_gpu_heuristic/performance_measurements.csv' using (\$2/1000) title \"best_effort_gpu_heuristic\", \
'query_chopping/performance_measurements.csv' using (\$2/1000) title \"query_chopping\"

#set style histogram errorbars gap 1 lw 1
#plot 'greedy_heuristic_cpu_only/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"greedy_heuristic_cpu_only\", \
#'greedy_heuristic/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"greedy_heuristic\", \
#'greedy_chainer_heuristic/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"greedy_chainer_heuristic\", \
#'critical_path_heuristic/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"critical_path_heuristic\", \
#'best_effort_gpu_heuristic/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"best_effort_gpu_heuristic\", \
#'query_chopping/performance_measurements.csv' using (\$2/1000):(\$3/1000):(\$4/1000) title \"query_chopping\"

set output \"performance_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > performance_diagram.gnuplot

echo "
set title \"$diagram_title\"
set auto x
set auto y

#set key top right Left reverse samplen 1
#set key box
#set key vertical maxrows 3
#set key width 2.1

set key top right Left reverse samplen 1
set key below
set key box
#set key inside above vertical maxrows 1
set key vertical maxrows 3
set key width 2.1

set ylabel 'Speedup compared to CPU Only' offset 0,-1,5
#set yrange [0:300000]
#set xrange [0:14]
set style data histogram
#set style histogram cluster gap 1
#set style histogram errorbars gap 1 lw 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"
#set style fill pattern 0 border

set xtics border in scale 0,0 nomirror rotate by -45  autojustify
set xtics   (\"Q1\" 0.00000,\"Q2\" 1.00000,\"Q3\" 2.00000,\"Q4\" 3.00000,\"Q5\" 4.00000,\"Q6\" 5.00000,\"Q7\" 6.00000,\"Q8\" 7.00000,\"Q9\" 8.00000,\"Q10\" 9.00000,\"Q11\" 10.00000,\"Q12\" 11.00000,\"Q13\" 12.00000,\"Q14\" 13.00000,\"Q15\" 14.00000,\"Q16\" 15.00000,\"Q17\" 16.00000,\"Q18\" 17.00000,\"Q19\" 18.00000,\"Q20\" 19.00000,\"Q21\" 20.00000,\"Q22\" 21.00000)

set style histogram cluster gap 1
plot 'experiment_result.csv' using (\$6/\$5) title \"Greedy Heuristic\", 'experiment_result.csv' using (\$6/\$4) title \"Greedy Chainer Heuristic\", 'experiment_result.csv' using (\$6/\$3) title \"Critical Path Heuristic\", 'experiment_result.csv' using (\$6/\$2) title \"Best Effort GPU Heuristic\", 'experiment_result.csv' using (\$6/\$7) title \"Query Chopping\"

set output \"speedup_queries_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > speedup_queries_diagram.gnuplot
         gnuplot performance_diagram.gnuplot
         gnuplot speedup_queries_diagram.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build performance diagrams for all experiments
##########################################################################################################################
else
    echo "Error: Could not determine for which workload I should create performance diagrams!"
    exit -1
fi


##########################################################################################################################		
#build performance diagrams for workload for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*; do
		        

			current_path=$(pwd)
			cd $experiment
			pwd
			for i in \# 0; do echo $i; done > linenumbers
			for i in `find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'`; do
			    echo $i > workload_time_$i 
			    cat "$i"/workload_execution_time_in_ms.txt >> workload_time_$i 
			done
			paste linenumbers workload_time_* > workload_execution_time.csv
			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key vertical maxrows 3
set key width 2.1

#set xlabel 'Queries of Star Schema Benchmark'
set ylabel 'Execution Time (s)'
set yrange [0:]
#set xrange [0:14]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

#set xtics   (\"Q1.1\" 0.00000, \"Q1.2\" 1.00000, \"Q1.3\" 2.00000, \"Q2.1\" 3.00000, \"Q2.2\" 4.00000,  \"Q2.3\" 5.00000, \"Q3.1\" 6.00000, \"Q3.2\" 7.00000, \"Q3.3\" 8.00000, \"Q3.4\" 9.00000,  \"Q4.1\" 10.00000,  \"Q4.2\" 11.00000,  \"Q4.3\" 12.00000)
set xtics(\"\" 0.00000)
	
plot 'workload_execution_time.csv' using (\$6/1000) title \"greedy_heuristic_cpu_only\", \
'workload_execution_time.csv' using (\$5/1000) title \"greedy_heuristic\", \
'workload_execution_time.csv' using (\$4/1000) title \"greedy_chainer_heuristic\", \
'workload_execution_time.csv' using (\$3/1000) title \"critical_path_heuristic\", \
'workload_execution_time.csv' using (\$2/1000) title \"best_effort_gpu_heuristic\", \
'workload_execution_time.csv' using (\$7/1000) title \"query_chopping\"
set output \"workload_performance_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > workload_performance_diagram.gnuplot
                        gnuplot workload_performance_diagram.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build performance diagrams for workload for all experiments
##########################################################################################################################

##########################################################################################################################		
#build gpu cache miss rate diagrams for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
		#"generated_experiments/$SUBFOLDER_NAME/"*; do
		        
#			for i in \# {1..13}; do
#			    echo "$i" > "generated_experiments/$SUBFOLDER_NAME/"linenumbers
#			done
			current_path=$(pwd)
			cd $experiment
			pwd
			for i in {0..1}; do echo $i; done > linenumbers
			for i in `find . -maxdepth 1 -mindepth 1 -type d -printf '%f\n'`; do
			    GPU_CACHE_COLUMN_MISSRATE=$(cat $i/gpu_cache_statistics.csv | awk '{print $1}')
			    GPU_CACHE_JOIN_INDEX_MISSRATE=$(cat $i/gpu_cache_statistics.csv | awk '{print $2}')
			    
			    echo $GPU_CACHE_COLUMN_MISSRATE > $i/gpu_cache_column_missrate
			    echo $GPU_CACHE_JOIN_INDEX_MISSRATE > $i/gpu_cache_join_index_missrate
			    cat $i/gpu_cache_column_missrate $i/gpu_cache_join_index_missrate > $i/gpu_cache_missrates
			    paste linenumbers $i/gpu_cache_missrates > $i/gpu_cache_missrates.csv
			    #echo $i > measurements_$i 
			    #cat "$i"/averaged_ssb* >> measurements_$i
			done
			#paste linenumbers gpu_cache_missrates > gpu_cache_missrates.csv
			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key vertical maxrows 3
set key width 2.1

set xlabel 'Access Structures'
#set ylabel 'Cache Miss Rate'
set ylabel 'Cache Hit Rate'
set yrange [0:1]
#set xrange [0:1]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

set xtics   (\"Columns\" 0.00000, \"Join Indexes\" 1.00000)

		
plot 'greedy_heuristic/gpu_cache_missrates.csv' using (1-\$2) title \"greedy_heuristic\", \
'greedy_chainer_heuristic/gpu_cache_missrates.csv' using (1-\$2) title \"greedy_chainer_heuristic\", \
'critical_path_heuristic/gpu_cache_missrates.csv' using (1-\$2) title \"critical_path_heuristic\", \
'best_effort_gpu_heuristic/gpu_cache_missrates.csv' using (1-\$2) title \"best_effort_gpu_heuristic\", \
'query_chopping/gpu_cache_missrates.csv' using (1-\$2) title \"query_chopping\"
set output \"gpu_cache_missrates_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > gpu_cache_missrates_diagram.gnuplot
                        gnuplot gpu_cache_missrates_diagram.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build gpu cache miss rate diagrams for all experiments
##########################################################################################################################

##########################################################################################################################		
#build aborted and executed gpu operators diagrams for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
			current_path="$PWD"
			cd $experiment
			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)	
		        
			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

#set key outside top center Left reverse samplen 1
set key outside bottom center Left reverse samplen 1
set key box
#set key vertical maxrows 3
#set key width 2.1

set xlabel 'Operator Types'
#set ylabel 'Cache Miss Rate'
set ylabel '#Operators'
#set yrange [0:1]
set xrange [-1:13]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

#set xtics   (\"Columns\" 0.00000, \"Join Indexes\" 1.00000)
#set xtics (\"ALL\" 0, \"Bitmap_Operator\" 0, \"COLUMN_CONSTANT_FILTER\" 1, \"CONVERT_BITMAP_TO_POSITIONLIST\" 2, \"CONVERT_POSITIONLIST_TO_BITMAP\" 3, \"ColumnAlgebraOperation\" 4, \"Column_Fetch_Join\" 5, \"Fetch_Join\" 6, \"GPU_Groupby_Algorithm\" 7, \"TOTAL\" 8, \"GPU_Sort_Algorithm\" 9, \"PositionList_Operator\" 10, \"SELECTION\" 11)
set xtics (\"ALL\" 0, \"BO\" 1, \"CF\" 2, \"B2P\" 3, \"P2B\" 4, \"CA\" 5, \"CFJ\" 6, \"FJ\" 7, \"GB\" 8, \"ALL\" 9, \"SO\" 10, \"PLO\" 11, \"SEL\" 12)
		
plot 'greedy_heuristic/number_of_aborted_and_executed_gpu_operators.csv' using (\$2) title \"greedy_heuristic aborted operators\", \
'greedy_heuristic/number_of_aborted_and_executed_gpu_operators.csv' using (\$3) title \"greedy_heuristic executed operators\", \
'greedy_chainer_heuristic/number_of_aborted_and_executed_gpu_operators.csv' using (\$2) title \"greedy_chainer_heuristic aborted operators\", \
'greedy_chainer_heuristic/number_of_aborted_and_executed_gpu_operators.csv' using (\$3) title \"greedy_chainer_heuristic executed operators\", \
'critical_path_heuristic/number_of_aborted_and_executed_gpu_operators.csv' using (\$2) title \"critical_path_heuristic aborted operators\", \
'critical_path_heuristic/number_of_aborted_and_executed_gpu_operators.csv' using (\$3) title \"critical_path_heuristic executed operators\", \
'best_effort_gpu_heuristic/number_of_aborted_and_executed_gpu_operators.csv' using (\$2) title \"best_effort_gpu_heuristic aborted operators\", \
'best_effort_gpu_heuristic/number_of_aborted_and_executed_gpu_operators.csv' using (\$3) title \"best_effort_gpu_heuristic executed operators\", \
'query_chopping/number_of_aborted_and_executed_gpu_operators.csv' using (\$2) title \"query_chopping aborted operators\", \
'query_chopping/number_of_aborted_and_executed_gpu_operators.csv' using (\$3) title \"query_chopping executed operators\"
set output \"number_of_aborted_and_executed_gpu_operators.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 6
replot" > number_of_aborted_and_executed_gpu_operators.gnuplot
                        gnuplot number_of_aborted_and_executed_gpu_operators.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build aborted and executed gpu operators diagrams for all experiments
##########################################################################################################################

##########################################################################################################################		
#build gpu memory traffic diagrams for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
			current_path="$PWD"
			cd $experiment
			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key vertical maxrows 3
set key width 2.1

set xlabel 'IO Source'
set ylabel 'Transferred Datasize in Byte'
#set yrange [0:300000]
set xrange [-1:9]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

#set xtics   (\"TOTAL_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 0.00000, \"TOTAL_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 1.00000, \"POSITIONLIST_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 2.00000, \"POSITIONLIST_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 3.00000, \"BITMAP_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 4.00000, \"BITMAP_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 5.00000, \"COLUMN_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 6.00000, \"COLUMN_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 7.00000, \"TRANSFERRED_BYTES_FOR_JOIN_INDEXES_HOST_TO_DEVICE\" 8.00000)
#set xtics   (\"TOTAL_TB_H2D\" 0.00000, \"TOTAL_TB_D2H\" 1.00000, \"PL_TB_H2D\" 2.00000, \"PL_TB_D2H\" 3.00000, \"BM_TB_H2D\" 4.00000, \"BM_TB_D2H\" 5.00000, \"COLUMN_TB_H2D\" 6.00000, \"COLUMN_TB_D2H\" 7.00000, \"JI_TB_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"T_D2H\" 1.00000, \"PL_H2D\" 2.00000, \"PL_D2H\" 3.00000, \"BM_H2D\" 4.00000, \"BM_D2H\" 5.00000, \"C_H2D\" 6.00000, \"C_D2H\" 7.00000, \"JI_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"T_D2H\" 1.00000, \"P_H2D\" 2.00000, \"P_D2H\" 3.00000, \"B_H2D\" 4.00000, \"B_D2H\" 5.00000, \"C_H2D\" 6.00000, \"C_D2H\" 7.00000, \"JI_H2D\" 8.00000)
set xtics   (\"T_H2D\" 0.00000, \"\nT_D2H\" 1.00000, \"P_H2D\" 2.00000, \"\nP_D2H\" 3.00000, \"B_H2D\" 4.00000, \"\nB_D2H\" 5.00000, \"C_H2D\" 6.00000, \"\nC_D2H\" 7.00000, \"JI_H2D\" 8.00000)
		
plot 'greedy_heuristic/transferred_bytes_over_pcie_bus.csv' using (\$2) title \"greedy_heuristic\", \
'greedy_chainer_heuristic/transferred_bytes_over_pcie_bus.csv' using (\$2) title \"greedy_chainer_heuristic\", \
'critical_path_heuristic/transferred_bytes_over_pcie_bus.csv' using (\$2) title \"critical_path_heuristic\", \
'best_effort_gpu_heuristic/transferred_bytes_over_pcie_bus.csv' using (\$2) title \"best_effort_gpu_heuristic\", \
'query_chopping/transferred_bytes_over_pcie_bus.csv' using (\$2) title \"query_chopping\"
set output \"transferred_bytes_over_pcie_bus_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > transferred_bytes_over_pcie_bus.gnuplot
            gnuplot transferred_bytes_over_pcie_bus.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build gpu memory traffic diagrams for all experiments
##########################################################################################################################

##########################################################################################################################		
#build gpu copy latency diagrams for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
			current_path="$PWD"
			cd $experiment
			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key vertical maxrows 3
set key width 2.1

set xlabel 'IO Source'
set ylabel 'Total Copy Time in s'
#set yrange [0:300000]
set xrange [-1:9]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

#set xtics   (\"TOTAL_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 0.00000, \"TOTAL_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 1.00000, \"POSITIONLIST_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 2.00000, \"POSITIONLIST_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 3.00000, \"BITMAP_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 4.00000, \"BITMAP_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 5.00000, \"COLUMN_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 6.00000, \"COLUMN_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 7.00000, \"TRANSFERRED_BYTES_FOR_JOIN_INDEXES_HOST_TO_DEVICE\" 8.00000)
#set xtics   (\"TOTAL_TB_H2D\" 0.00000, \"TOTAL_TB_D2H\" 1.00000, \"PL_TB_H2D\" 2.00000, \"PL_TB_D2H\" 3.00000, \"BM_TB_H2D\" 4.00000, \"BM_TB_D2H\" 5.00000, \"COLUMN_TB_H2D\" 6.00000, \"COLUMN_TB_D2H\" 7.00000, \"JI_TB_H2D\" 8.00000)
set xtics   (\"T_H2D\" 0.00000, \"\nT_D2H\" 1.00000, \"P_H2D\" 2.00000, \"\nP_D2H\" 3.00000, \"B_H2D\" 4.00000, \"\nB_D2H\" 5.00000, \"C_H2D\" 6.00000, \"\nC_D2H\" 7.00000, \"JI_H2D\" 8.00000)
		
plot 'greedy_heuristic/data_transfer_times_over_pcie_bus.csv' using (\$2) title \"greedy_heuristic\", \
'greedy_chainer_heuristic/data_transfer_times_over_pcie_bus.csv' using (\$2) title \"greedy_chainer_heuristic\", \
'critical_path_heuristic/data_transfer_times_over_pcie_bus.csv' using (\$2) title \"critical_path_heuristic\", \
'best_effort_gpu_heuristic/data_transfer_times_over_pcie_bus.csv' using (\$2) title \"best_effort_gpu_heuristic\", \
'query_chopping/data_transfer_times_over_pcie_bus.csv' using (\$2) title \"query_chopping\"
set output \"data_transfer_times_over_pcie_bus_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > data_transfer_times_over_pcie_bus.gnuplot
            gnuplot data_transfer_times_over_pcie_bus.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build gpu copy latency diagrams for all experiments
##########################################################################################################################

##########################################################################################################################		
#build gpu copy operation count diagrams for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
			current_path="$PWD"
			cd $experiment
			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key vertical maxrows 3
set key width 2.1

set xlabel 'IO Source'
set ylabel 'Number of Copy Operations'
#set yrange [0:300000]
set xrange [-1:9]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

#set xtics   (\"TOTAL_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 0.00000, \"TOTAL_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 1.00000, \"POSITIONLIST_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 2.00000, \"POSITIONLIST_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 3.00000, \"BITMAP_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 4.00000, \"BITMAP_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 5.00000, \"COLUMN_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 6.00000, \"COLUMN_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 7.00000, \"TRANSFERRED_BYTES_FOR_JOIN_INDEXES_HOST_TO_DEVICE\" 8.00000)
#set xtics   (\"TOTAL_TB_H2D\" 0.00000, \"TOTAL_TB_D2H\" 1.00000, \"PL_TB_H2D\" 2.00000, \"PL_TB_D2H\" 3.00000, \"BM_TB_H2D\" 4.00000, \"BM_TB_D2H\" 5.00000, \"COLUMN_TB_H2D\" 6.00000, \"COLUMN_TB_D2H\" 7.00000, \"JI_TB_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"T_D2H\" 1.00000, \"PL_H2D\" 2.00000, \"PL_D2H\" 3.00000, \"BM_H2D\" 4.00000, \"BM_D2H\" 5.00000, \"C_H2D\" 6.00000, \"C_D2H\" 7.00000, \"JI_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"T_D2H\" 1.00000, \"P_H2D\" 2.00000, \"P_D2H\" 3.00000, \"B_H2D\" 4.00000, \"B_D2H\" 5.00000, \"C_H2D\" 6.00000, \"C_D2H\" 7.00000, \"JI_H2D\" 8.00000)
set xtics   (\"T_H2D\" 0.00000, \"\nT_D2H\" 1.00000, \"P_H2D\" 2.00000, \"\nP_D2H\" 3.00000, \"B_H2D\" 4.00000, \"\nB_D2H\" 5.00000, \"C_H2D\" 6.00000, \"\nC_D2H\" 7.00000, \"JI_H2D\" 8.00000)
		
plot 'greedy_heuristic/number_of_data_transfer_operations_over_pcie_bus.csv' using (\$2) title \"greedy_heuristic\", \
'greedy_chainer_heuristic/number_of_data_transfer_operations_over_pcie_bus.csv' using (\$2) title \"greedy_chainer_heuristic\", \
'critical_path_heuristic/number_of_data_transfer_operations_over_pcie_bus.csv' using (\$2) title \"critical_path_heuristic\", \
'best_effort_gpu_heuristic/number_of_data_transfer_operations_over_pcie_bus.csv' using (\$2) title \"best_effort_gpu_heuristic\", \
'query_chopping/number_of_data_transfer_operations_over_pcie_bus.csv' using (\$2) title \"query_chopping\"
set output \"number_of_data_transfer_operations_over_pcie_bus_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > number_of_data_transfer_operations_over_pcie_bus.gnuplot
            gnuplot number_of_data_transfer_operations_over_pcie_bus.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build gpu copy operation count diagrams for all experiments
##########################################################################################################################

##########################################################################################################################		
#build gpu operator arbortion rate diagrams for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
			current_path="$PWD"
			cd $experiment
			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key vertical maxrows 3
set key width 2.1

#set xlabel 'IO Source'
set ylabel 'GPU Operator Abortion Rate'
set yrange [0:1]
#set xrange [-1:9]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

#set xtics   (\"TOTAL_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 0.00000, \"TOTAL_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 1.00000, \"POSITIONLIST_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 2.00000, \"POSITIONLIST_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 3.00000, \"BITMAP_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 4.00000, \"BITMAP_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 5.00000, \"COLUMN_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 6.00000, \"COLUMN_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 7.00000, \"TRANSFERRED_BYTES_FOR_JOIN_INDEXES_HOST_TO_DEVICE\" 8.00000)
#set xtics   (\"TOTAL_TB_H2D\" 0.00000, \"TOTAL_TB_D2H\" 1.00000, \"PL_TB_H2D\" 2.00000, \"PL_TB_D2H\" 3.00000, \"BM_TB_H2D\" 4.00000, \"BM_TB_D2H\" 5.00000, \"COLUMN_TB_H2D\" 6.00000, \"COLUMN_TB_D2H\" 7.00000, \"JI_TB_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"T_D2H\" 1.00000, \"PL_H2D\" 2.00000, \"PL_D2H\" 3.00000, \"BM_H2D\" 4.00000, \"BM_D2H\" 5.00000, \"C_H2D\" 6.00000, \"C_D2H\" 7.00000, \"JI_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"T_D2H\" 1.00000, \"P_H2D\" 2.00000, \"P_D2H\" 3.00000, \"B_H2D\" 4.00000, \"B_D2H\" 5.00000, \"C_H2D\" 6.00000, \"C_D2H\" 7.00000, \"JI_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"\nT_D2H\" 1.00000, \"P_H2D\" 2.00000, \"\nP_D2H\" 3.00000, \"B_H2D\" 4.00000, \"\nB_D2H\" 5.00000, \"C_H2D\" 6.00000, \"\nC_D2H\" 7.00000, \"JI_H2D\" 8.00000)
		
plot 'greedy_heuristic/abortion_rate.csv' using (\$2) title \"greedy_heuristic\", \
'greedy_chainer_heuristic/abortion_rate.csv' using (\$2) title \"greedy_chainer_heuristic\", \
'critical_path_heuristic/abortion_rate.csv' using (\$2) title \"critical_path_heuristic\", \
'greedy_heuristic_cpu_only/abortion_rate.csv' using (\$2) title \"greedy_heuristic_cpu_only\", \
'best_effort_gpu_heuristic/abortion_rate.csv' using (\$2) title \"best_effort_gpu_heuristic\", \
'query_chopping/abortion_rate.csv' using (\$2) title \"query_chopping\"
set output \"gpu_operator_abortion_rate.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > gpu_operator_abortion_rate.gnuplot
            gnuplot gpu_operator_abortion_rate.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build gpu operator arbortion rate diagrams for all experiments
##########################################################################################################################

##########################################################################################################################		
#build wasted time due to aborted GPU operators diagrams for all experiments
##########################################################################################################################	
for SUBFOLDER_NAME in varying_gpu_buffer_size varying_number_of_parallel_users varying_data_placement_strategy varying_gpu_buffer_manager_strategies varying_gpu_memory_occupation; do
		for experiment in `find ./generated_experiments/$SUBFOLDER_NAME/ -maxdepth 1 -mindepth 1 -type d`; do
			current_path="$PWD"
			cd $experiment
			exp_name=$(basename $experiment)
			diagram_title=$(cat performance_diagram_title.txt)			
echo "			
set title \"$diagram_title\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key vertical maxrows 3
set key width 2.1

#set xlabel 'IO Source'
set ylabel 'Wasted Time due to Aborted GPU Operations (s)'
#set yrange [0:300000]
#set xrange [-1:9]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"

#set xtics   (\"TOTAL_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 0.00000, \"TOTAL_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 1.00000, \"POSITIONLIST_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 2.00000, \"POSITIONLIST_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 3.00000, \"BITMAP_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 4.00000, \"BITMAP_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 5.00000, \"COLUMN_TRANSFERRED_BYTES_HOST_TO_DEVICE\" 6.00000, \"COLUMN_TRANSFERRED_BYTES_DEVICE_TO_HOST\" 7.00000, \"TRANSFERRED_BYTES_FOR_JOIN_INDEXES_HOST_TO_DEVICE\" 8.00000)
#set xtics   (\"TOTAL_TB_H2D\" 0.00000, \"TOTAL_TB_D2H\" 1.00000, \"PL_TB_H2D\" 2.00000, \"PL_TB_D2H\" 3.00000, \"BM_TB_H2D\" 4.00000, \"BM_TB_D2H\" 5.00000, \"COLUMN_TB_H2D\" 6.00000, \"COLUMN_TB_D2H\" 7.00000, \"JI_TB_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"T_D2H\" 1.00000, \"PL_H2D\" 2.00000, \"PL_D2H\" 3.00000, \"BM_H2D\" 4.00000, \"BM_D2H\" 5.00000, \"C_H2D\" 6.00000, \"C_D2H\" 7.00000, \"JI_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"T_D2H\" 1.00000, \"P_H2D\" 2.00000, \"P_D2H\" 3.00000, \"B_H2D\" 4.00000, \"B_D2H\" 5.00000, \"C_H2D\" 6.00000, \"C_D2H\" 7.00000, \"JI_H2D\" 8.00000)
#set xtics   (\"T_H2D\" 0.00000, \"\nT_D2H\" 1.00000, \"P_H2D\" 2.00000, \"\nP_D2H\" 3.00000, \"B_H2D\" 4.00000, \"\nB_D2H\" 5.00000, \"C_H2D\" 6.00000, \"\nC_D2H\" 7.00000, \"JI_H2D\" 8.00000)
		
plot 'greedy_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv' using (\$2/(1000*1000*1000)) title \"greedy_heuristic\", \
'greedy_chainer_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv' using (\$2/(1000*1000*1000)) title \"greedy_chainer_heuristic\", \
'critical_path_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv' using (\$2/(1000*1000*1000)) title \"critical_path_heuristic\", \
'best_effort_gpu_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv' using (\$2/(1000*1000*1000)) title \"best_effort_gpu_heuristic\", \
'query_chopping/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv' using (\$2/(1000*1000*1000)) title \"query_chopping\"
set output \"wasted_time_due_to_aborted_gpu_operations.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > wasted_time_due_to_aborted_gpu_operations.gnuplot
            gnuplot wasted_time_due_to_aborted_gpu_operations.gnuplot
			cd $current_path
		done
done

##########################################################################################################################		
#END build wasted time due to aborted GPU operators diagrams for all experiments
##########################################################################################################################



#if [[ "$QUERY_NAME" == "ssball" ]]; then
##if we execute a complete workload, we do not need to repeat the workload X times until we executed X queries
##therefore, we devide by the number of queries in the workload (13), and round up (X=(X+12)/13)
#NUMBER_OF_QUERIES=$(((NUMBER_OF_QUERIES+12)/13))
##execute the workload 2 times to train the cost models
#NUMBER_OF_WARMUP_QUERIES=2
#fi










