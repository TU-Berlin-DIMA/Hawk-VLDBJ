#!/bin/bash

#RESULTDIR=result_sf5_1_user_no_mem_cost_models
RESULTDIR=result_sf5_1_user_with_mem_cost_models

echo "Start Experiments..."
for i in *_config; do 
cd $i
OPT=$(echo $i | sed -e 's/_config//g')
echo "Run Optimizer $OPT"
../../../bin/cogadbd ../benchmark.coga &> cogadb_ssb_measurement.log
bash ../generate_csv_for_cogadb_benchmarkresult.sh
cd ..
done


for i in *_config; do echo $i; mkdir -p $RESULTDIR/$i; cp $i/cogadb_ssb_measurement.* $i/average_estimation_errors.csv $i/*.coga $i/*.conf  $RESULTDIR/$i/; done

exit 0
