#!/bin/bash

#if any program instance fails, the entire script failes and returns with an error code
set -e

DATAFILE_CONTAINING_RESULTS=""

if [ $# -lt 2 ]; then
	echo "Missing parameter!"
	echo "Usage: $0 <path to datafile contraining experimental results> <output directory>"
	exit -1
fi


DATAFILE_CONTAINING_RESULTS="$1"
OUTPUT_DIRECTORY="$2"

echo "Reading File $DATAFILE_CONTAINING_RESULTS"

cat $DATAFILE_CONTAINING_RESULTS | awk '{if($4==0){print}}' | tee cpu_only.data
cat $DATAFILE_CONTAINING_RESULTS | awk '{if($4==1){print}}' | tee gpu_only.data

cat $DATAFILE_CONTAINING_RESULTS | awk 'BEGIN{FS="\t";}{if($4==2 && $7=="Response Time"){print}}' | tee response_time.data

cat $DATAFILE_CONTAINING_RESULTS | awk 'BEGIN{FS="\t";}{if($4==2 && $7=="Simple Round Robin"){print}}' | tee simple_round_robin.data
cat $DATAFILE_CONTAINING_RESULTS | awk 'BEGIN{FS="\t";}{if($4==2 && $7=="WaitingTimeAwareResponseTime"){print}}' | tee waiting_time_aware_response_time.data
cat $DATAFILE_CONTAINING_RESULTS | awk 'BEGIN{FS="\t";}{if($4==2 && $7=="Throughput"){print}}' | tee throughput.data
cat $DATAFILE_CONTAINING_RESULTS | awk 'BEGIN{FS="\t";}{if($4==2 && $7=="Throughput2"){print}}' | tee throughput2.data

#cat ../benchmark_results.log | awk 'BEGIN{FS="\t";}{if($4==2 && $6==""){print}}' | tee .data

#aggregate each 10 lines by computing average of it
#ATTENTION: the 10 corresponds to the number of rounds per experiment, if the number of roudn is modified, this script has to be modifed as well!
for i in cpu_only.data gpu_only.data response_time.data simple_round_robin.data waiting_time_aware_response_time.data throughput.data throughput2.data; do
	cat "$i" | awk '
BEGIN{FS="\t";
sum_execution_time=0;
sum_training_time=0;
sum_percentaged_time_spend_on_cpu=0;
sum_percentaged_time_spend_on_gpu=0;
sum_average_estimation_error_in_percent_of_cpu_algorithm=0;
sum_average_estimation_error_in_percent_of_gpu_algorithm=0;
number_of_rounds=5;
} 

{sum_execution_time+=$16; 
sum_training_time+=$17; 
sum_percentaged_time_spend_on_cpu+=$20;
sum_percentaged_time_spend_on_gpu+=$21;
sum_average_estimation_error_in_percent_of_cpu_algorithm+=$22;
sum_average_estimation_error_in_percent_of_gpu_algorithm+=$23;

if(NR%number_of_rounds==0){
exec_time_avg=sum_execution_time/number_of_rounds;
training_time_avg=sum_training_time/number_of_rounds;
percentaged_time_spend_on_cpu_avg=sum_percentaged_time_spend_on_cpu/number_of_rounds;
percentaged_time_spend_on_gpu_avg=sum_percentaged_time_spend_on_gpu/number_of_rounds;
average_estimation_error_in_percent_of_cpu_algorithm_avg=sum_average_estimation_error_in_percent_of_cpu_algorithm/number_of_rounds;
average_estimation_error_in_percent_of_gpu_algorithm_avg=sum_average_estimation_error_in_percent_of_gpu_algorithm/number_of_rounds;

print $0"\t"exec_time_avg"\t"training_time_avg"\t"percentaged_time_spend_on_cpu_avg"\t"percentaged_time_spend_on_gpu_avg"\t"average_estimation_error_in_percent_of_cpu_algorithm_avg"\t"average_estimation_error_in_percent_of_gpu_algorithm_avg;
sum_execution_time=0;
sum_training_time=0;
sum_percentaged_time_spend_on_cpu=0;
sum_percentaged_time_spend_on_gpu=0;
sum_average_estimation_error_in_percent_of_cpu_algorithm=0;
sum_average_estimation_error_in_percent_of_gpu_algorithm=0;
next
}}' > averaged_$i
done

mkdir -p $OUTPUT_DIRECTORY
mv *.data $OUTPUT_DIRECTORY/


exit 0
