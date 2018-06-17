#!/bin/bash

set -e
set -x

if [ $# -lt 1 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 <QUERY_ID>"
	echo "Valid value are: 11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43" 
	exit -1
fi

if [ $# -gt 1 ]; then
	echo 'To many parameters!'
	echo "Usage: $0 <QUERY_ID>"
	echo "Valid value are: 11, 12, 13, 21, 22, 23, 31, 32, 33, 34, 41, 42, 43" 
	exit -1
fi

LOAD_BALANCING_STRATEGIES[0]="ResponseTime"
LOAD_BALANCING_STRATEGIES[1]="WaitingTimeAwareResponseTime"
LOAD_BALANCING_STRATEGIES[2]="Throughput"
LOAD_BALANCING_STRATEGIES[3]="Simple_Round_Robin"
LOAD_BALANCING_STRATEGIES[4]="ProbabilityBasedOutsourcing"
LOAD_BALANCING_STRATEGIES[5]="Throughput2"

#WaitingTimeAwareResponseTime: 1 (default)
#Throughput: 2
#Simple_Round_Robin: 3
#ProbabilityBasedOutsourcing: 4
#Throughput2: 5

QUERYID=$1

#cleanup
rm -f core

mkdir -p experimental_result_data_query_chopping
export TIMEFORMAT=%3R
PATH_TO_COGADB=`cat PATH_TO_COGADB_EXECUTABLE` #"../../../cogadb/bin/cogadbd"
#PATH_TO_COGADB="echo "
NUMBER_OF_RUNS=1
NUMBER_OF_QUERIES=100
#QUERYID=`cat QUERYID` 

SCALE_FACTORS_TO_BENCHMARK=`cat SCALE_FACTORS_TO_BENCHMARK`
for (( run=0; run<$NUMBER_OF_RUNS; run++ )); do

load_balancing_strategy=0 
use_memory_cost_models=0

for load_balancing_strategy in 0 1; do
for scale_factor in $SCALE_FACTORS_TO_BENCHMARK; do

      #number_of_parallel_users=10
      mkdir -p experimental_result_data
      for number_of_parallel_users in 1 3 10 15 20; do


	  #Query Chopping in CPU ONLY MODE
      optimizer=querychopping
      #NUMBER_OF_CPU_CORES=$(nproc)
      #NUMBER_OF_CPU_CORES=$((`nproc`/2))
      number_of_gpus=0
      for NUMBER_OF_CPU_CORES in 1 2 4 6 8 10 15 20; do
         #configure hardware 
         bash generate_hardware_specification_file.sh $NUMBER_OF_CPU_CORES $number_of_gpus
	 bash generate_hype_configuration_file.sh $load_balancing_strategy 0 $use_memory_cost_models
  	 #for run_type in "cpu" "any"; do
	run_type="cpu"	
	SCRIPT_NAME="sf_""$scale_factor""_query_""$QUERYID""_users_""$number_of_parallel_users""_num_queries_""$NUMBER_OF_QUERIES""_run_"$run"_runtype_"$run_type"_num_cpus_""$NUMBER_OF_CPU_CORES""_num_gpus_""$number_of_gpus""_lbs_"$load_balancing_strategy"_usememorymodel_"$use_memory_cost_models".coga"
		 ./generate_coga_script_file_query_chopping.sh $scale_factor ssb$QUERYID $number_of_parallel_users $NUMBER_OF_QUERIES 20 experimental_result_data_query_chopping/$SCRIPT_NAME $run_type any
		 LOG_FILE=experimental_result_data_query_chopping/`basename "$SCRIPT_NAME" ".coga"`.log
		 QUERY_RUNTIMES=experimental_result_data_query_chopping/`basename "$SCRIPT_NAME" ".coga"`_cogascript.timings
		 HARDWARE_CONFIGURATION=experimental_result_data_query_chopping/`basename "$SCRIPT_NAME" ".coga"`_hardware_specification.conf
		 HYPE_CONFIGURATION=experimental_result_data_query_chopping/`basename "$SCRIPT_NAME" ".coga"`_hype_configuration.conf
		 #ensure that a core file is created in case of a system crash
		 ulimit -c unlimited
	#     $PATH_TO_COGADB experimental_result_data_query_chopping/$SCRIPT_NAME > $LOG_FILE
		 { time $PATH_TO_COGADB experimental_result_data_query_chopping/$SCRIPT_NAME &> $LOG_FILE; } 2> tmp_execution_time.log
		 #EXECUTION_TIME=`cat tmp_execution_time.log`
		 EXECUTION_TIME=`tail -n 1 tmp_execution_time.log`
		 echo "WORKLOAD_EXECUTION_TIME: $EXECUTION_TIME" >> $LOG_FILE  
		 echo "NUMBER_OF_VIRTUAL_CPUS: $NUMBER_OF_CPU_CORES" >> $LOG_FILE
		 echo "NUMBER_OF_VIRTUAL_GPUS: $number_of_gpus" >> $LOG_FILE	 
		 echo "LOAD_BALANCING_STRATEGY: ${LOAD_BALANCING_STRATEGIES[$load_balancing_strategy]} : $load_balancing_strategy" >> $LOG_FILE
                 echo "USE_MEMORY_COST_MODELS: $use_memory_cost_models" >> $LOG_FILE
		 echo "QUERY_OPTIMIZER_MODE: $optimizer" >> $LOG_FILE 
		 mv cogascript_timings.log "$QUERY_RUNTIMES"
		 mv hardware_specification.conf "$HARDWARE_CONFIGURATION"
		 mv hype.conf "$HYPE_CONFIGURATION"
		 rm tmp_execution_time.log
      done
      
done
done
done
done    
      
bash build_csv_result_table_query_chopping_cpu_only.sh
      
NEW_DIRECTORY_NAME=experimental_result_data_query_chopping_cpu_only_ssb"$QUERYID"_machine_"$HOSTNAME"
mv experimental_result_data_query_chopping $NEW_DIRECTORY_NAME
zip -r $NEW_DIRECTORY_NAME.zip $NEW_DIRECTORY_NAME
exit 0
