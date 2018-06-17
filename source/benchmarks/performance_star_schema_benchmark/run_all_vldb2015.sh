#!/bin/bash

trap "echo Error! Experiments did not successfully complete!; exit 0" SIGINT SIGTERM SIGKILL

#adjust scale factor according to machine
#echo 10 > SCALE_FACTOR
#echo 1 > SCALE_FACTOR


echo 10 > SCALE_FACTOR
SCALE_FACTOR=$(cat SCALE_FACTOR)
echo ssb_all > QUERY_NAME

#for num_parallel_users in 1 5 10 15 20; do
for num_parallel_users in 20; do
   echo $num_parallel_users > NUMBER_OF_PARALLEL_USERS
   bash -x ./run_sigmod_experiments.sh data_placement_driven_query_optimization
   mv generated_experiments generated_experiments_numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME
   zip -r generated_experiments_numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME
   bash create_cleaned_evaluation_results.sh generated_experiments_numuser_$num_parallel_users-sf_$SCALE_FACTOR-$HOSTNAME
done 
exit 0

echo 1 > NUMBER_OF_PARALLEL_USERS
#for SCALE_FACTOR in 10 20 30; do
for SCALE_FACTOR in 20 30; do
   echo $SCALE_FACTOR > SCALE_FACTOR
   bash -x ./run_sigmod_experiments.sh data_placement_driven_query_optimization
   mv generated_experiments generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME
   zip -r generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME
   bash create_cleaned_evaluation_results.sh generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME
done

exit 0
