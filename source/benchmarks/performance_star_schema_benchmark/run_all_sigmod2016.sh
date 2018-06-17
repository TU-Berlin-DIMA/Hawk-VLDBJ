#!/bin/bash

trap "echo Error! Experiments did not successfully complete!; exit 0" SIGINT SIGTERM SIGKILL

#adjust scale factor according to machine
#echo 10 > SCALE_FACTOR
#echo 1 > SCALE_FACTOR

########################################################################################################################
### STAR SCHEMA BENCHMARK
########################################################################################################################
echo 10 > SCALE_FACTOR
SCALE_FACTOR=$(cat SCALE_FACTOR)
echo ssb_all > QUERY_NAME
QUERYNAME=$(cat QUERY_NAME)

for num_parallel_users in 1 5 10 15 20; do
#for num_parallel_users in 10 20; do
   echo $num_parallel_users > NUMBER_OF_PARALLEL_USERS
   bash -x ./run_sigmod_experiments.sh data_placement_driven_query_optimization
   mv generated_experiments generated_experiments_$QUERYNAME-numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME
   zip -r generated_experiments_$QUERYNAME-numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_$QUERYNAME-numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME
   bash create_cleaned_evaluation_results.sh generated_experiments_$QUERYNAME-numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME
done 


echo 1 > NUMBER_OF_PARALLEL_USERS
for SCALE_FACTOR in 5 10 15 20 25 30; do
##for SCALE_FACTOR in 5; do
   echo $SCALE_FACTOR > SCALE_FACTOR
   bash -x ./run_sigmod_experiments.sh data_placement_driven_query_optimization
   mv generated_experiments generated_experiments_$QUERYNAME-single_user_sf$SCALE_FACTOR-$HOSTNAME
   zip -r generated_experiments_$QUERYNAME-single_user_sf$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_$QUERYNAME-single_user_sf$SCALE_FACTOR-$HOSTNAME
   bash create_cleaned_evaluation_results.sh generated_experiments_$QUERYNAME-single_user_sf$SCALE_FACTOR-$HOSTNAME
done

########################################################################################################################
### TPC-H BENCHMARK
########################################################################################################################
echo 10 > SCALE_FACTOR
SCALE_FACTOR=$(cat SCALE_FACTOR)
echo tpch_all_supported > QUERY_NAME
QUERYNAME=$(cat QUERY_NAME)

for num_parallel_users in 1 5 10 15 20; do
#for num_parallel_users in 10 20; do
   echo $num_parallel_users > NUMBER_OF_PARALLEL_USERS
   bash -x ./run_sigmod_experiments.sh data_placement_driven_query_optimization
   mv generated_experiments generated_experiments_$QUERYNAME-numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME
   zip -r generated_experiments_$QUERYNAME-numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_$QUERYNAME-numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME
   bash create_cleaned_evaluation_results.sh generated_experiments_$QUERYNAME-numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME
done 


echo 1 > NUMBER_OF_PARALLEL_USERS
for SCALE_FACTOR in 5 10 15 20; do
#for SCALE_FACTOR in 5; do
   echo $SCALE_FACTOR > SCALE_FACTOR
   bash -x ./run_sigmod_experiments.sh data_placement_driven_query_optimization
   mv generated_experiments generated_experiments_$QUERYNAME-single_user_sf$SCALE_FACTOR-$HOSTNAME
   zip -r generated_experiments_$QUERYNAME-single_user_sf$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_$QUERYNAME-single_user_sf$SCALE_FACTOR-$HOSTNAME
   bash create_cleaned_evaluation_results.sh generated_experiments_$QUERYNAME-single_user_sf$SCALE_FACTOR-$HOSTNAME
done

bash create_csv_files_for_workload_plots_sigmod2016.sh  

exit 0
