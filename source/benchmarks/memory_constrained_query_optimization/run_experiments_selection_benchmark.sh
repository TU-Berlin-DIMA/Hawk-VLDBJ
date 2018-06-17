#!/bin/bash

#1MB=1073741824Byte
trap "Error! Experiments did not successfully complete!" SIGINT SIGTERM ERR SIGKILL

#NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT=10
#NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT=5
NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT=1

#default parameters for benchmark
DEFAULT_NUMBER_OF_DATASETS=100 #100 #100
DEFAULT_MAX_DATASET_SIZE=10 #10 #10 #MB #50 #MB #53687091200 #50MB
DEFAULT_NUMBER_OF_OPERATIONS=2000 #2000 #2000
DEFAULT_RANDOM_SEED=10

#default values for STEMOD!
export STEMOD_LENGTH_OF_TRAININGPHASE=10
export STEMOD_HISTORY_LENGTH=1000
export STEMOD_RECOMPUTATION_PERIOD=100
export STEMOD_ALGORITHM_MAXIMAL_IDLE_TIME=2
export STEMOD_RETRAINING_LENGTH=1
export STEMOD_MAXIMAL_SLOWDOWN_OF_NON_OPTIMAL_ALGORITHM_IN_PERCENT=50
export STEMOD_READY_QUEUE_LENGTH=100

BENCHMARK=../../bin/generic_selection_benchmark
#BENCHMARK="echo ../../bin/generic_selection_benchmark"

#if any program instance fails, the entire script failes and returns with an error code
set -e

##	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
##	do
##  
##  #TODO: vary number of datasets and max_dataset_size_in_MB!
##  
###      NUMBER_OF_CPU_CORES=$(nproc)
###      for number_of_gpus in 0 1 2 3 4 5 6 7 8 9 10; do
##      NUMBER_OF_CPU_CORES=1
##      for number_of_gpus in 1 2 3 4 5 6 7 8 9 10; do
##         #configure hardware 
##         bash generate_hardware_specification_file.sh $NUMBER_OF_CPU_CORES $number_of_gpus
###		  ../../bin/generic_selection_benchmark --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=1 --number_of_operations=200 --max_dataset_size_in_MB=64 --random_seed=$DEFAULT_RANDOM_SEED 
###		  ../../bin/generic_selection_benchmark --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=10 --number_of_operations=2000 --max_dataset_size_in_MB=256 --random_seed=$DEFAULT_RANDOM_SEED 
##      
##		  $BENCHMARK --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=10 --number_of_operations=2000 --max_dataset_size_in_MB=2 --random_seed=$DEFAULT_RANDOM_SEED 
##	done

##     done

#     for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
#     do
#      #NUMBER_OF_CPU_CORES=$(nproc)
#      NUMBER_OF_PARALLEL_QUERY_SESSIONS=20
##      for number_of_gpus in 0 1 2 3 4 5 6 7 8 9 10; do
#      NUMBER_OF_CPU_CORES=4
#      for number_of_gpus in 0 1 2 3 4 5 6 7 8 9 10; do
#         #configure hardware 
#         bash generate_hardware_specification_file.sh $NUMBER_OF_CPU_CORES $number_of_gpus
##		  ../../bin/generic_selection_benchmark --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=1 --number_of_operations=200 --max_dataset_size_in_MB=64 --random_seed=$DEFAULT_RANDOM_SEED 
##		  ../../bin/generic_selection_benchmark --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=10 --number_of_operations=2000 --max_dataset_size_in_MB=256 --random_seed=$DEFAULT_RANDOM_SEED 
#             for num_datasets in 1 3 5 10; do
#		  #$BENCHMARK --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=$num_datasets --number_of_operations=2000 --max_dataset_size_in_MB=256 --parallel_users=$NUMBER_OF_PARALLEL_QUERY_SESSIONS --random_seed=$DEFAULT_RANDOM_SEED
#		  $BENCHMARK --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=$num_datasets --number_of_operations=2000 --max_dataset_size_in_MB=1 --parallel_users=$NUMBER_OF_PARALLEL_QUERY_SESSIONS --random_seed=$DEFAULT_RANDOM_SEED 
#            done
#	done
#     done

     for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
     do
#      NUMBER_OF_CPU_CORES=$(nproc)
      NUMBER_OF_PARALLEL_QUERY_SESSIONS=20
#      for number_of_gpus in 0 1 2 3 4 5 6 7 8 9 10; do
      NUMBER_OF_CPU_CORES=4
      for number_of_gpus in 0 1 2 3 4 5 6 7 8 9 10; do
         #configure hardware 
         bash generate_hardware_specification_file.sh $NUMBER_OF_CPU_CORES $number_of_gpus
#		  ../../bin/generic_selection_benchmark --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=1 --number_of_operations=200 --max_dataset_size_in_MB=64 --random_seed=$DEFAULT_RANDOM_SEED 
#		  ../../bin/generic_selection_benchmark --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=10 --number_of_operations=2000 --max_dataset_size_in_MB=256 --random_seed=$DEFAULT_RANDOM_SEED 
             #for dataset_size in 1 10 128 256 512; do
		  #$BENCHMARK --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=10 --number_of_operations=2000 --max_dataset_size_in_MB=$dataset_size --parallel_users=$NUMBER_OF_PARALLEL_QUERY_SESSIONS --random_seed=$DEFAULT_RANDOM_SEED 
             for dataset_size in 1 10 100; do
		  $BENCHMARK --optimization_criterion="WaitingTimeAwareResponseTime" --number_of_datasets=10 --number_of_operations=2000 --max_dataset_size_in_MB=$dataset_size --parallel_users=$NUMBER_OF_PARALLEL_QUERY_SESSIONS --random_seed=$DEFAULT_RANDOM_SEED &> /dev/null
            done
	done
     done

exit 0



CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_DATASETS=false
CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_OPERATIONS=false
CONDUCT_EXPERIMENT_VARYING_MAXIMAL_DATASET_SIZE=false
CONDUCT_EXPERIMENT_VARYING_OPERATOR_QUEUE_LENGTH=false
CONDUCT_EXPERIMENT_VARYING_INITIAL_TRAINING_PHASE=false

if [ $# -lt 2 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 [test|full|varying_number_datasets|varying_number_operations|varying_number_dataset_size|varying_operator_queue_length|varying_training_length] [AGGREGATION|JOIN|SELECTION|SORT]"
	exit -1
fi

if [ "$1" = "varying_number_datasets" ]; then
	CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_DATASETS=true
elif [ "$1" = "varying_number_operations" ]; then
	CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_OPERATIONS=true
elif [ "$1" = "varying_number_dataset_size" ]; then
	CONDUCT_EXPERIMENT_VARYING_MAXIMAL_DATASET_SIZE=true
elif [ "$1" = "varying_operator_queue_length" ]; then
	CONDUCT_EXPERIMENT_VARYING_OPERATOR_QUEUE_LENGTH=true
elif [ "$1" = "varying_training_length" ]; then
	CONDUCT_EXPERIMENT_VARYING_INITIAL_TRAINING_PHASE=true
elif [ "$1" = "full" ]; then
	CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_DATASETS=true
	CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_OPERATIONS=true
	CONDUCT_EXPERIMENT_VARYING_MAXIMAL_DATASET_SIZE=true
	CONDUCT_EXPERIMENT_VARYING_OPERATOR_QUEUE_LENGTH=true
	#CONDUCT_EXPERIMENT_VARYING_INITIAL_TRAINING_PHASE=1
fi

OPERATION_NAME="$2"

#if [ "$OPERATION_NAME" != "SORT" ] && [ "$OPERATION_NAME" != "SELECTION" ] && [ "$OPERATION_NAME" != "AGGREGATION" ]; then
if [[ "$OPERATION_NAME" != "SORT" && "$OPERATION_NAME" != "SELECTION" && "$OPERATION_NAME" != "AGGREGATION" && "$OPERATION_NAME" != "JOIN" ]]; then
	echo "Second parameter has to be a valid Operation: [AGGREGATION|JOIN|SELECTION|SORT]"
	echo "Your Input: $OPERATION_NAME"
	echo "Aborting..."
	exit -1
fi

mkdir -p "eval/Results/$OPERATION_NAME"

#BENCHMARK_COMMAND=../bin/cogadb
#BENCHMARK_COMMAND=echo

if [ "$OPERATION_NAME" == "SORT" ]; then
	BENCHMARK_COMMAND=../../bin/sort_benchmark 
elif [ "$OPERATION_NAME" == "SELECTION" ]; then
	BENCHMARK_COMMAND=../../bin/selection_benchmark 
elif [ "$OPERATION_NAME" == "AGGREGATION" ]; then
	BENCHMARK_COMMAND=../../bin/aggregation_benchmark 
elif [ "$OPERATION_NAME" == "JOIN" ]; then
	BENCHMARK_COMMAND=../../bin/join_benchmark 
fi
#BENCHMARK_COMMAND="echo ../../bin/join_benchmark"
rm -f benchmark_results.log

echo -n "Start Time of Experiments: " 
date

if [ "$1" = "test" ]; then
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="CPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="GPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Simple Round Robin" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Response Time" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="WaitingTimeAwareResponseTime" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Throughput" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Throughput2" --random_seed=$DEFAULT_RANDOM_SEED 
fi







if $CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_DATASETS; then
for number_of_datasets in {50..200..50} {300..500..100} #values 50 to 200 in steps of 50 and values from 300 to 500 in steps of 100
do
	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	do
		 $BENCHMARK_COMMAND --number_of_datasets=$number_of_datasets --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="CPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$number_of_datasets --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="GPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$number_of_datasets --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Simple Round Robin" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$number_of_datasets --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Response Time" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$number_of_datasets --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="WaitingTimeAwareResponseTime" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$number_of_datasets --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Throughput" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$number_of_datasets --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Throughput2" --random_seed=$DEFAULT_RANDOM_SEED 
	done
done

mv benchmark_results.log "eval/Results/$OPERATION_NAME/$HOSTNAME-varying_number_of_datasets_benchmark_results.data"  

fi


#--stemod_optimization_criterion="Simple Round Robin"

if $CONDUCT_EXPERIMENT_VARYING_NUMBER_OF_OPERATIONS; then
for number_of_operations in {500..3000..500} {4000..8000..1000} #values 100 to 3000 in steps of 300
do
	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	do
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$number_of_operations --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="CPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$number_of_operations --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="GPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$number_of_operations --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Simple Round Robin" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$number_of_operations --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Response Time" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$number_of_operations --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="WaitingTimeAwareResponseTime" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$number_of_operations --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Throughput" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$number_of_operations --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Throughput2" --random_seed=$DEFAULT_RANDOM_SEED 

	done
done

mv benchmark_results.log "eval/Results/$OPERATION_NAME/$HOSTNAME-varying_number_of_operations_benchmark_results.data"

fi


##we devide with 4, because the benchmark expects the maximal data size in number of elements (an elements are form typ int, which is 4 byte on 32 bit platform)

if $CONDUCT_EXPERIMENT_VARYING_MAXIMAL_DATASET_SIZE; then
for max_dataset_size in {10..40..10} {50..150..50}
do
	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	do
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$max_dataset_size --scheduling_method="CPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$max_dataset_size --scheduling_method="GPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$max_dataset_size --scheduling_method="HYBRID" --optimization_criterion="Simple Round Robin" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$max_dataset_size --scheduling_method="HYBRID" --optimization_criterion="Response Time" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$max_dataset_size --scheduling_method="HYBRID" --optimization_criterion="WaitingTimeAwareResponseTime" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$max_dataset_size --scheduling_method="HYBRID" --optimization_criterion="Throughput" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$max_dataset_size --scheduling_method="HYBRID" --optimization_criterion="Throughput2" --random_seed=$DEFAULT_RANDOM_SEED 

	done
done
mv benchmark_results.log "eval/Results/$OPERATION_NAME/$HOSTNAME-varying_dataset_size_benchmark_results.data"

fi

if $CONDUCT_EXPERIMENT_VARYING_OPERATOR_QUEUE_LENGTH; then
for operator_queue_length in {10..40..10} {50..150..50}
do
	export STEMOD_READY_QUEUE_LENGTH=$operator_queue_length
	echo "Ready Queue Length: $STEMOD_READY_QUEUE_LENGTH"
	for (( c=0; c<$NUMBER_OF_RUNS_FOR_EACH_EXPERIMENT; c++ ))
	do
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="CPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="GPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Simple Round Robin" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Response Time" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="WaitingTimeAwareResponseTime" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Throughput" --random_seed=$DEFAULT_RANDOM_SEED 
		 $BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --optimization_criterion="Throughput2" --random_seed=$DEFAULT_RANDOM_SEED 

	done
done
mv benchmark_results.log "eval/Results/$OPERATION_NAME/$HOSTNAME-varying_operator_queue_length_benchmark_results.data"

fi

if $CONDUCT_EXPERIMENT_VARYING_INITIAL_TRAINING_PHASE; then
	echo "Error! NOT YET IMPLEMENTED VARYING TRANINGPHASE EXPERIMENTS SCRIPT!"
	exit -1;
fi

echo -n "End Time of Experiments: " 
date

#for number_of_datasets in {100..3000..100} #values 50 to 300 in steps of 10
#do
#	for (( c=0; c<=10; c++ ))
#	do
#		$BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="CPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
#		$BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="GPU_ONLY" --random_seed=$DEFAULT_RANDOM_SEED
#		$BENCHMARK_COMMAND --number_of_datasets=$DEFAULT_NUMBER_OF_DATASETS --number_of_operations=$DEFAULT_NUMBER_OF_OPERATIONS --max_dataset_size_in_MB=$DEFAULT_MAX_DATASET_SIZE --scheduling_method="HYBRID" --random_seed=$DEFAULT_RANDOM_SEED
#	done
#done

echo "Experiments have sucessfully finished!"

exit 0
