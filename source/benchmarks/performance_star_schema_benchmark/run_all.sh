#!/bin/bash

trap "echo Error! Experiments did not successfully complete!; exit 0" SIGINT SIGTERM SIGKILL

#adjust scale factor according to machine
#echo 10 > SCALE_FACTOR
#echo 1 > SCALE_FACTOR

SCALE_FACTOR=$(cat SCALE_FACTOR)

echo 1 > NUMBER_OF_PARALLEL_USERS

echo ssb_all > QUERY_NAME

#experiments for single user worklaods
#bash ./run_sigmod_experiments.sh varying_number_of_users
bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
#bash ./run_sigmod_experiments.sh varying_qc_configurations

mv generated_experiments generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME
zip -r generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME
bash create_cleaned_evaluation_results.sh generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME

#experiments for multi user workloads
echo 10 > NUMBER_OF_PARALLEL_USERS
bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
#bash ./run_sigmod_experiments.sh varying_qc_configurations

mv generated_experiments generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME
zip -r generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME
bash create_cleaned_evaluation_results.sh generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME

#echo 20 > SCALE_FACTOR
#SCALE_FACTOR=$(cat SCALE_FACTOR)

#echo 1 > NUMBER_OF_PARALLEL_USERS

##experiments for single user worklaods
##bash ./run_sigmod_experiments.sh varying_number_of_users
#bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
##bash ./run_sigmod_experiments.sh varying_qc_configurations

#mv generated_experiments generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME
#zip -r generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME
#bash create_cleaned_evaluation_results.sh generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME

##experiments for multi user workloads
#echo 10 > NUMBER_OF_PARALLEL_USERS
#bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
##bash ./run_sigmod_experiments.sh varying_qc_configurations

#mv generated_experiments generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME
#zip -r generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME
#bash create_cleaned_evaluation_results.sh generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME

#echo 30 > SCALE_FACTOR
#SCALE_FACTOR=$(cat SCALE_FACTOR)

#echo 1 > NUMBER_OF_PARALLEL_USERS

##experiments for single user worklaods
##bash ./run_sigmod_experiments.sh varying_number_of_users
#bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
##bash ./run_sigmod_experiments.sh varying_qc_configurations

#mv generated_experiments generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME
#zip -r generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME
#bash create_cleaned_evaluation_results.sh generated_experiments_single_user_sf$SCALE_FACTOR-$HOSTNAME

##experiments for multi user workloads
#echo 10 > NUMBER_OF_PARALLEL_USERS
#bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
##bash ./run_sigmod_experiments.sh varying_qc_configurations

#mv generated_experiments generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME
#zip -r generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME".zip" generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME
#bash create_cleaned_evaluation_results.sh generated_experiments_multi_user_sf$SCALE_FACTOR-$HOSTNAME

exit 0
