set -e
bash run_experiments_query_chopping.sh 11
bash run_experiments_query_chopping.sh 23
bash run_experiments_query_chopping.sh 31
bash run_experiments_query_chopping.sh 42

bash build_summary.sh
#bash run__query_chopping.sh ssb23
#bash run_benchmark_query_chopping.sh ssb31
#bash run_benchmark_query_chopping.sh ssb42
exit 0



