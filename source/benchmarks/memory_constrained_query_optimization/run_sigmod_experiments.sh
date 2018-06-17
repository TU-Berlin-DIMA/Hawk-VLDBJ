#!/bin/bash
set -e
set -x

bash run_experiments_query_chopping.sh all
bash run_experiments_cpu_only_query_chopping.sh all
bash plot_diagrams.sh
exit 0
