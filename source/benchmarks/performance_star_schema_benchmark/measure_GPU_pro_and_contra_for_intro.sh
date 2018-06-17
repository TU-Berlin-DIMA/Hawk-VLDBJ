#!/bin/bash

set -e
export LC_NUMERIC="de_DE.UTF-8"
set +e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

#QUERY=ssb42
QUERY=ssb33

echo "
set path_to_database=/home/sebastian/cogadb_databases/ssb_sf20/
loaddatabase
loadjoinindexes
showgpucache
set gpu_buffer_management_strategy=disbled_gpu_buffer
set hybrid_query_optimizer=best_effort_gpu_heuristic
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
quit
" | ./cogadbd | tee measurement_intro_gpu_only_no_caching_$QUERY".log"


echo "
set path_to_database=/home/sebastian/cogadb_databases/ssb_sf20/
loaddatabase
loadjoinindexes
showgpucache
set hybrid_query_optimizer=best_effort_gpu_heuristic
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
quit
" | ./cogadbd | tee measurement_intro_gpu_only_with_caching_$QUERY".log"


echo "
set path_to_database=/home/sebastian/cogadb_databases/ssb_sf20/
loaddatabase
loadjoinindexes
showgpucache
set hybrid_query_optimizer=greedy_heuristic
setdevice cpu
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
$QUERY
quit
" | ./cogadbd | tee measurement_intro_cpu_only_$QUERY".log"

egrep '^Execution Time' measurement_intro_gpu_only_no_caching_$QUERY".log" | awk 'BEGIN{FS=":";} {print $2}' | sed -e 's/ms//g' -e 's/ //g' | gawk --use-lc-numeric  '{ SUM += $1; COUNT+=1;} END { print SUM/COUNT}' > measurement_intro_gpu_only_no_caching_$QUERY".csv"
egrep '^Execution Time' measurement_intro_gpu_only_with_caching_$QUERY".log" | awk 'BEGIN{FS=":";} {print $2}' | sed -e 's/ms//g' -e 's/ //g' | gawk --use-lc-numeric  '{ SUM += $1; COUNT+=1;} END { print SUM/COUNT}' > measurement_intro_gpu_only_with_caching_$QUERY".csv"
egrep '^Execution Time' measurement_intro_cpu_only_$QUERY".log" | awk 'BEGIN{FS=":";} {print $2}' | sed -e 's/ms//g' -e 's/ //g' | gawk --use-lc-numeric  '{ SUM += $1; COUNT+=1;} END { print SUM/COUNT}' > measurement_intro_cpu_only_$QUERY".csv"


