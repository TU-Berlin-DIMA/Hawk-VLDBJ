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

QUERYID=$1

#cleanup
rm -f core

#configure hardware 
bash generate_hardware_specification_file.sh 1 1

mkdir -p experimental_result_data
export TIMEFORMAT=%3R
PATH_TO_COGADB=`cat PATH_TO_COGADB_EXECUTABLE` #"../../../cogadb/bin/cogadbd"
#PATH_TO_COGADB="echo "
NUMBER_OF_RUNS=10
NUMBER_OF_QUERIES=100
#QUERYID=`cat QUERYID` 
SCALE_FACTORS_TO_BENCHMARK=`cat SCALE_FACTORS_TO_BENCHMARK`
for (( run=0; run<$NUMBER_OF_RUNS; run++ )); do
#for scale_factor in 1 5 10 15; do
#for scale_factor in 1 5 10 20 30 50; do
for scale_factor in $SCALE_FACTORS_TO_BENCHMARK; do
#for scale_factor in 1; do
  for number_of_parallel_users in 1 2 3 4 6 8 10 12 15 20; do
  	 for run_type in "cpu" "any"; do
		 SCRIPT_NAME="sf_""$scale_factor""_query_""$QUERYID""_users_""$number_of_parallel_users""_num_queries_""$NUMBER_OF_QUERIES""_run_"$run"_runtype_"$run_type".coga" 
		 ./generate_coga_script_file.sh $scale_factor ssb$QUERYID $number_of_parallel_users 100 1 experimental_result_data/$SCRIPT_NAME $run_type $run_type
		 LOG_FILE=experimental_result_data/`basename "$SCRIPT_NAME" ".coga"`.log
		 QUERY_RUNTIMES=experimental_result_data/`basename "$SCRIPT_NAME" ".coga"`_cogascript.timings
		 #ensure that a core file is created in case of a system crash
		 ulimit -c unlimited
	#     $PATH_TO_COGADB experimental_result_data/$SCRIPT_NAME > $LOG_FILE
		 { time $PATH_TO_COGADB experimental_result_data/$SCRIPT_NAME > $LOG_FILE; } 2> tmp_execution_time.log
		 #EXECUTION_TIME=`cat tmp_execution_time.log`
		 EXECUTION_TIME=`tail -n 1 tmp_execution_time.log`
		 echo "WORKLOAD_EXECUTION_TIME: $EXECUTION_TIME" >> $LOG_FILE  
		 mv cogascript_timings.log "$QUERY_RUNTIMES"
		 rm tmp_execution_time.log
	#     $PATH_TO_COGADB experimental_result_data/$SCRIPT_NAME
	 done
  done
done
done

bash plot_scripts/create_plots.sh
dirlist=$(find plot_scripts -mindepth 1 -maxdepth 1 -type d)

for dir in $dirlist
do
    DIRECTORY=`basename $dir`
    mkdir -p experimental_result_data/diagrams/$DIRECTORY
    mv plot_scripts/$DIRECTORY/result/* experimental_result_data/diagrams/$DIRECTORY/
    cp plot_scripts/$DIRECTORY/*.* experimental_result_data/diagrams/$DIRECTORY/
done

NEW_DIRECTORY_NAME=experimental_result_data_aborted_gpu_operators_ssb"$QUERYID"_machine_"$HOSTNAME"
mv experimental_result_data $NEW_DIRECTORY_NAME
zip -r $NEW_DIRECTORY_NAME.zip $NEW_DIRECTORY_NAME

exit 0
