#!/bin/bash

trap "echo Error! Experiments did not successfully complete!; exit 0" SIGINT SIGTERM SIGKILL

#adjust scale factor according to machine
echo 10 > SCALE_FACTOR
#echo 1 > SCALE_FACTOR

SCALE_FACTOR=$(cat SCALE_FACTOR)

CURRENT_DATE=$(date +%F)

OUTPUT_DIRECTORY=SIGMOD_EXPERIMENTS_SEPERATE_QUERIES_$CURRENT_DATE

mkdir -p $OUTPUT_DIRECTORY

for query_name in ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43; do

echo $query_name > QUERY_NAME
echo 1 > NUMBER_OF_PARALLEL_USERS

#experiments for single user worklaods
#bash ./run_sigmod_experiments.sh varying_number_of_users
bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
#bash ./run_sigmod_experiments.sh varying_qc_configurations

mv generated_experiments generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
zip -r generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME".zip" generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
bash create_cleaned_evaluation_results.sh generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
mv generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME* $OUTPUT_DIRECTORY/

#experiments for multi user workloads
echo 10 > NUMBER_OF_PARALLEL_USERS
bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
#bash ./run_sigmod_experiments.sh varying_qc_configurations

mv generated_experiments generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
zip -r generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME".zip" generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
bash create_cleaned_evaluation_results.sh generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
mv generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME* $OUTPUT_DIRECTORY/

done






#echo 20 > SCALE_FACTOR
#SCALE_FACTOR=$(cat SCALE_FACTOR)

#echo 1 > NUMBER_OF_PARALLEL_USERS

##experiments for single user worklaods
##bash ./run_sigmod_experiments.sh varying_number_of_users
#bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
##bash ./run_sigmod_experiments.sh varying_qc_configurations

#mv generated_experiments generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
#zip -r generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME".zip" generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
#bash create_cleaned_evaluation_results.sh generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME

##experiments for multi user workloads
#echo 10 > NUMBER_OF_PARALLEL_USERS
#bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
##bash ./run_sigmod_experiments.sh varying_qc_configurations

#mv generated_experiments generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
#zip -r generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME".zip" generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
#bash create_cleaned_evaluation_results.sh generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME

#echo 30 > SCALE_FACTOR
#SCALE_FACTOR=$(cat SCALE_FACTOR)

#echo 1 > NUMBER_OF_PARALLEL_USERS

##experiments for single user worklaods
##bash ./run_sigmod_experiments.sh varying_number_of_users
#bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
##bash ./run_sigmod_experiments.sh varying_qc_configurations

#mv generated_experiments generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
#zip -r generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME".zip" generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
#bash create_cleaned_evaluation_results.sh generated_experiments_single_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME

##experiments for multi user workloads
#echo 10 > NUMBER_OF_PARALLEL_USERS
#bash ./run_sigmod_experiments.sh data_placement_driven_query_optimization
##bash ./run_sigmod_experiments.sh varying_qc_configurations

#mv generated_experiments generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
#zip -r generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME".zip" generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME
#bash create_cleaned_evaluation_results.sh generated_experiments_multi_user_sf$SCALE_FACTOR-query-$query_name-$HOSTNAME

exit 0
