#!/bin/bash


#bash generate_benchmarkfile.sh  

#../../../cogadb/bin/cogadbd benchmark.coga > cogadb_ssb_measurement.log
#bash generate_csv_for_cogadb_benchmarkresult.sh
#mv cogadb_ssb_measurement.csv cogadb_ssb_measurement_mixed_query_workload.csv

for i in ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43; do

    ../../../cogadb/bin/cogadbd benchmark_query_"$i"_only.coga  > cogadb_ssb_measurement.log
    bash generate_csv_for_cogadb_benchmarkresult.sh
    mv cogadb_ssb_measurement.csv cogadb_ssb_measurement_single_query_workload_"$i"_only.csv

done 


exit 0
