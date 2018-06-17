#!/bin/bash

set -x

if [ $# -lt 18 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 <SCALE_FACTOR> <QUERY_NAME> <NUMBER_OF_PARALLEL_USERS> <NUMBER_OF_QUERIES> <NUMBER_OF_WARMUP_QUERIES> <SCRIPT_NAME> <RUN_TYPE> <RUN_TYPE_WARMUP_PHASE> <NUMBER_OF_CPUS> <NUMBER_OF_GPUS> <default_optimization_criterion> <reuse_performance_models> <track_memory_usage> <ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION> <GPU_BUFFER_SIZE> <GPU_BUFFERMANAGEMENT_STRATEGY> <SUBFOLDER_NAME> <PRELOAD_DATA_IN_GPU_BUFFER>"
	echo "<RUN_TYPE> corresponds to CoGaDB's device policy: 'any' uses all procesing devices, 'cpu' only CPU devices and 'gpu' only gpu devices"
	echo "optional 19. argument:  gpu_memory_occupation" 
	exit -1
fi


GPU_MEMORY_OCCUPATION_STRING=""
GPU_MEMORY_OCCUPATION="0"
if [ $# -eq 19 ]; then
    GPU_MEMORY_OCCUPATION_STRING="-${19}"
    GPU_MEMORY_OCCUPATION=${19}
fi





if [ $# -gt 20 ]; then
	echo 'To many parameters!'
	echo "Usage: $0 <SCALE_FACTOR> <QUERY_NAME> <NUMBER_OF_PARALLEL_USERS> <NUMBER_OF_QUERIES> <NUMBER_OF_WARMUP_QUERIES> <SCRIPT_NAME> <RUN_TYPE> <RUN_TYPE_WARMUP_PHASE> <NUMBER_OF_CPUS> <NUMBER_OF_GPUS> <default_optimization_criterion> <reuse_performance_models> <track_memory_usage> <ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION> <GPU_BUFFER_SIZE> <GPU_BUFFERMANAGEMENT_STRATEGY> <SUBFOLDER_NAME> <PRELOAD_DATA_IN_GPU_BUFFER>"
	echo "<RUN_TYPE> corresponds to CoGaDB's device policy: 'any' uses all procesing devices, 'cpu' only CPU devices and 'gpu' only gpu devices" 
	echo "optional 19. argument:  gpu_memory_occupation" 
	echo "optional 20. argument:  hype_ready_queue_length" 
	exit -1
fi



#<GPU_BUFFER_SIZE> <GPU_BUFFERMANAGEMENT_STRATEGY> <SUBFOLDER_NAME>


SCALE_FACTOR=$1
QUERY_NAME=$2
NUMBER_OF_PARALLEL_USERS=$3
NUMBER_OF_QUERIES=$4
NUMBER_OF_WARMUP_QUERIES=$5
SCRIPT_NAME=$6
RUN_TYPE=$7
RUN_TYPE_WARMUP_PHASE=$8
NUMBER_OF_CPUS=$9
NUMBER_OF_GPUS=${10}
default_optimization_criterion=${11}
reuse_performance_models=${12}
track_memory_usage=${13}
ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION=${14}
GPU_BUFFER_SIZE=${15}
GPU_BUFFERMANAGEMENT_STRATEGY=${16}
SUBFOLDER_NAME=${17}
PRELOAD_DATA_IN_GPU_BUFFER=${18}

HYPE_READY_QUEUE_LENGTH_STRING=""
#in case we have a single-user workload, we configure query chopping to use a ready queue length of 10
if [ $NUMBER_OF_PARALLEL_USERS -eq 1 ]; then
HYPE_READY_QUEUE_LENGTH="10"
else
#in case we have a multi-user workload, we configure query chopping to use a ready queue length of 75
HYPE_READY_QUEUE_LENGTH="75"
fi

if [ $# -eq 20 ]; then
    HYPE_READY_QUEUE_LENGTH_STRING="-RQL${20}"
    HYPE_READY_QUEUE_LENGTH=${20}
fi

echo "Experiment Configuration:
SCALE_FACTOR=$SCALE_FACTOR
QUERY_NAME=$QUERY_NAME
NUMBER_OF_PARALLEL_USERS=$NUMBER_OF_PARALLEL_USERS
NUMBER_OF_QUERIES=$NUMBER_OF_QUERIES
NUMBER_OF_WARMUP_QUERIES=$NUMBER_OF_WARMUP_QUERIES
SCRIPT_NAME=$SCRIPT_NAME
RUN_TYPE=$RUN_TYPE
RUN_TYPE_WARMUP_PHASE=$RUN_TYPE_WARMUP_PHASE
NUMBER_OF_CPUS=$NUMBER_OF_CPUS
NUMBER_OF_GPUS=$NUMBER_OF_GPUS
default_optimization_criterion=$default_optimization_criterion
reuse_performance_models=$reuse_performance_models
track_memory_usage=$track_memory_usage
ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION=$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION
GPU_BUFFER_SIZE=$GPU_BUFFER_SIZE (bytes)
GPU_BUFFERMANAGEMENT_STRATEGY=$GPU_BUFFERMANAGEMENT_STRATEGY
SUBFOLDER_NAME=$SUBFOLDER_NAME
PRELOAD_DATA_IN_GPU_BUFFER=$PRELOAD_DATA_IN_GPU_BUFFER"


EXPERIMENT_NAME="experiment_$SCALE_FACTOR-$QUERY_NAME-$NUMBER_OF_PARALLEL_USERS-$NUMBER_OF_QUERIES-$NUMBER_OF_WARMUP_QUERIES-$SCRIPT_NAME-$RUN_TYPE-$RUN_TYPE_WARMUP_PHASE-$NUMBER_OF_CPUS-$NUMBER_OF_GPUS-$default_optimization_criterion-$reuse_performance_models-$track_memory_usage-$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION-$GPU_BUFFER_SIZE-$GPU_BUFFERMANAGEMENT_STRATEGY-$PRELOAD_DATA_IN_GPU_BUFFER-$SUBFOLDER_NAME$GPU_MEMORY_OCCUPATION_STRING$HYPE_READY_QUEUE_LENGTH_STRING"

mkdir -p "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME"

bash generate_coga_script_file.sh $SCALE_FACTOR $QUERY_NAME $NUMBER_OF_PARALLEL_USERS $NUMBER_OF_QUERIES $NUMBER_OF_WARMUP_QUERIES $SCRIPT_NAME $RUN_TYPE $RUN_TYPE_WARMUP_PHASE $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION $PRELOAD_DATA_IN_GPU_BUFFER
#bash generate_hardware_specification_file.sh $NUMBER_OF_CPUS $NUMBER_OF_GPUS
#all heuristics aside form query chopping work with one CPU and one GPU
bash generate_hardware_specification_file.sh 1 1 
#bash generate_hype_configuration_file.sh $default_optimization_criterion $reuse_performance_models $track_memory_usage
bash generate_hype_configuration_file.sh 0 $reuse_performance_models $track_memory_usage

for heuristic in greedy_heuristic greedy_chainer_heuristic critical_path_heuristic best_effort_gpu_heuristic; do
    mkdir -p "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic"
    rm -f "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    touch "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    echo "set enable_profiling=true" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    echo "set hybrid_query_optimizer=$heuristic" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    #set GPU Buffer size in byte (unsigned integer)
    echo "set gpu_buffer_size=$GPU_BUFFER_SIZE" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"	
    #set Buffer management strategy for GPU Buffer (least_recently_used,least_frequently_used), ignored after call to 'placejoinindexes'
    echo "set gpu_buffer_management_strategy=$GPU_BUFFERMANAGEMENT_STRATEGY" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"	
    #make the query optimization driven by the current data placement, requires prior call to 'placejoinindexes'
    echo "set enable_automatic_data_placement=$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    echo "set enable_dataplacement_aware_query_optimization=$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    cp hype.conf "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/"
    cp hardware_specification.conf "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/"    
    cp benchmark.coga "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/"  
done
    #workaround to make greedy chainer heuristic use WTAR
    bash generate_hype_configuration_file.sh 1 $reuse_performance_models $track_memory_usage
    mv hype.conf "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/greedy_chainer_heuristic/"
    
    
    heuristic="query_chopping"
    mkdir -p "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic"
    rm -f "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    touch "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    echo "set enable_profiling=true" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    echo "toggleQC" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    #set GPU Buffer size in byte (unsigned integer)
    echo "set gpu_buffer_size=$GPU_BUFFER_SIZE" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"	
    #set Buffer management strategy for GPU Buffer (least_recently_used,least_frequently_used), ignored after call to 'placejoinindexes'
    echo "set gpu_buffer_management_strategy=$GPU_BUFFERMANAGEMENT_STRATEGY" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"	
    #make the query optimization driven by the current data placement, requires prior call to 'placejoinindexes'
    echo "set enable_automatic_data_placement=$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    echo "set enable_dataplacement_aware_query_optimization=$ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    echo "set enable_pull_based_query_chopping=true" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    #always configure query chopping with WTAR heuristic
    bash generate_hype_configuration_file.sh 1 $reuse_performance_models $track_memory_usage 
    #for query chopping, the number of processors are very relevant, so generate the config file with benchmark parameters  
    #bash generate_hardware_specification_file.sh $NUMBER_OF_CPUS $NUMBER_OF_GPUS
    bash generate_hardware_specification_file.sh 1 1
    echo "ready_queue_length=$HYPE_READY_QUEUE_LENGTH" >> hype.conf
    cp hype.conf "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/"
    cp hardware_specification.conf "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/"    
    cp benchmark.coga "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/"  
    echo "$NUMBER_OF_CPUS" > "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/number_of_cpus.txt"
    echo "$NUMBER_OF_GPUS" > "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/number_of_gpus.txt"    
    echo "$HYPE_READY_QUEUE_LENGTH" > "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/hype_ready_queue_length.txt"
    
    #CPU ONLY RUN
    heuristic="greedy_heuristic_cpu_only"
    mkdir -p "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic"
    rm -f "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    touch "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    echo "set enable_profiling=true" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    #set GPU Buffer size in byte (unsigned integer)
    echo "set gpu_buffer_size=$GPU_BUFFER_SIZE" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"	
    #set Buffer management strategy for GPU Buffer (least_recently_used,least_frequently_used), ignored after call to 'placejoinindexes'
    echo "set gpu_buffer_management_strategy=$GPU_BUFFERMANAGEMENT_STRATEGY" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"	
    #make the query optimization driven by the current data placement, requires prior call to 'placejoinindexes'
    echo "set enable_automatic_data_placement=false" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    echo "set enable_dataplacement_aware_query_optimization=false" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    #CPU ONLY execution
    echo "setdevice cpu" >> "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/startup.coga"
    #always configure query chopping with WTAR heuristic
    bash generate_hype_configuration_file.sh 0 $reuse_performance_models $track_memory_usage
    #for query chopping, the number of processors are very relevant, so generate the config file with benchmark parameters  
    bash generate_hardware_specification_file.sh 1 1
    #generate specialized script that executes the measurement phase on CPU only
    bash generate_coga_script_file.sh $SCALE_FACTOR $QUERY_NAME $NUMBER_OF_PARALLEL_USERS $NUMBER_OF_QUERIES $NUMBER_OF_WARMUP_QUERIES $SCRIPT_NAME cpu $RUN_TYPE_WARMUP_PHASE $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION $PRELOAD_DATA_IN_GPU_BUFFER
    cp hype.conf "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/"
    cp hardware_specification.conf "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/"    
    cp benchmark.coga "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME/$heuristic/" 
    
    echo "sf $SCALE_FACTOR, #users $NUMBER_OF_PARALLEL_USERS, #cpus $NUMBER_OF_CPUS, #gpus $NUMBER_OF_GPUS, gpu_buffer_size: $GPU_BUFFER_SIZE, \n $GPU_BUFFERMANAGEMENT_STRATEGY, mem_bookeeping $track_memory_usage, DPA_OPT $ENABLE_DATAPLACEMENT_AWARE_QUERY_OPTIMIZATION, $HOSTNAME, \n PLD: $PRELOAD_DATA_IN_GPU_BUFFER GPU_Occ: $GPU_MEMORY_OCCUPATION" > "generated_experiments/$SUBFOLDER_NAME/$EXPERIMENT_NAME"/performance_diagram_title.txt    
    
exit 0
