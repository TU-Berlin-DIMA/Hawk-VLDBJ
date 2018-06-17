#!/bin/bash

set -e
export LC_NUMERIC="de_DE.UTF-8"
set +e

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

QUERY_NAME=$(cat "$SCRIPT_DIR/QUERY_NAME")

if [[ "$QUERY_NAME" == "ssb"* ]]; then
for i in ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43; do
cat cogascript_timings.log | grep $i | awk '{print $2}' | sed -e 's/(//g' -e 's/)//g' -e 's/ms//g' -e 's/\./,/g' > $i
#cat $i | awk -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $3}' > averaged_$i
#cat $i | awk -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $1}' > min_$i
#cat $i | awk -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $2}' > max_$i
cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $3}' > averaged_$i
cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $6}' > min_$i
cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $7}' > max_$i
done

echo -e "Q1.1\tQ1.2\tQ1.3\tQ2.1\tQ2.2\tQ2.3\tQ3.1\tQ3.2\tQ3.3\tQ3.4\tQ4.1\tQ4.2\tQ4.3" > cogadb_ssb_measurement.csv
paste ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43 >> cogadb_ssb_measurement.csv

echo -e "Q1.1\tQ1.2\tQ1.3\tQ2.1\tQ2.2\tQ2.3\tQ3.1\tQ3.2\tQ3.3\tQ3.4\tQ4.1\tQ4.2\tQ4.3" > cogadb_averaged_ssb_measurement.csv
paste averaged_ssb11 averaged_ssb12 averaged_ssb13 averaged_ssb21 averaged_ssb22 averaged_ssb23 averaged_ssb31 averaged_ssb32 averaged_ssb33 averaged_ssb34 averaged_ssb41 averaged_ssb42 averaged_ssb43 >> cogadb_averaged_ssb_measurement.csv

echo -e "Q1.1\tQ1.2\tQ1.3\tQ2.1\tQ2.2\tQ2.3\tQ3.1\tQ3.2\tQ3.3\tQ3.4\tQ4.1\tQ4.2\tQ4.3" > cogadb_min_ssb_measurement.csv
paste min_ssb11 min_ssb12 min_ssb13 min_ssb21 min_ssb22 min_ssb23 min_ssb31 min_ssb32 min_ssb33 min_ssb34 min_ssb41 min_ssb42 min_ssb43 >> cogadb_min_ssb_measurement.csv

echo -e "Q1.1\tQ1.2\tQ1.3\tQ2.1\tQ2.2\tQ2.3\tQ3.1\tQ3.2\tQ3.3\tQ3.4\tQ4.1\tQ4.2\tQ4.3" > cogadb_max_ssb_measurement.csv
paste max_ssb11 max_ssb12 max_ssb13 max_ssb21 max_ssb22 max_ssb23 max_ssb31 max_ssb32 max_ssb33 max_ssb34 max_ssb41 max_ssb42 max_ssb43 >> cogadb_max_ssb_measurement.csv

elif [[ "$QUERY_NAME" == "tpch"* ]]; then

for i in tpch01 tpch02 tpch03 tpch04 tpch05 tpch06 tpch07 tpch08 tpch09 tpch10 tpch11 tpch12 tpch13 tpch14 tpch15 tpch16 tpch17 tpch18 tpch19 tpch20 tpch21 tpch22; do
cat cogascript_timings.log | grep $i | awk '{print $2}' | sed -e 's/(//g' -e 's/)//g' -e 's/ms//g' -e 's/\./,/g' > $i
#compute statistics if file not empty
if [[ -s $i ]] ; then
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $1}' > min_$i
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $2}' > max_$i
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $3}' > averaged_$i
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $4}' > variance_$i
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $5}' > standard_deviation_$i
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $6}' > standard_error_of_the_mean_$i
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $7}' > lower_bound_of_error_bar_$i
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $8}' > upper_bound_of_error_bar_$i
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $9}' > lower_bound_of_standard_error_of_the_mean_error_bar_$i
   cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $10}' > upper_bound_of_standard_error_of_the_mean_error_bar_$i
   
   #cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $6}' > min_$i
   #cat $i | gawk --use-lc-numeric -f "$SCRIPT_DIR/compute_average.awk" | awk '{print $7}' > max_$i
else
   #touch min_$i max_$i averaged_$i variance_$i standard_deviation_$i lower_bound_of_error_bar_$i upper_bound_of_error_bar_$i lower_bound_of_standard_error_of_the_mean_error_bar_$i upper_bound_of_standard_error_of_the_mean_error_bar_$i
   echo 0 > min_$i
   echo 0 > max_$i
   echo 0 > averaged_$i
   echo 0 > variance_$i
   echo 0 > standard_deviation_$i
   echo 0 > standard_error_of_the_mean_$i
   echo 0 > lower_bound_of_error_bar_$i
   echo 0 > upper_bound_of_error_bar_$i
   echo 0 > lower_bound_of_standard_error_of_the_mean_error_bar_$i
   echo 0 > upper_bound_of_standard_error_of_the_mean_error_bar_$i
fi
done

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_tpch_measurement.csv
paste tpch01 tpch02 tpch03 tpch04 tpch05 tpch06 tpch07 tpch08 tpch09 tpch10 tpch11 tpch12 tpch13 tpch14 tpch15 tpch16 tpch17 tpch18 tpch19 tpch20 tpch21 tpch22 >> cogadb_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_averaged_tpch_measurement.csv
paste averaged_tpch01 averaged_tpch02 averaged_tpch03 averaged_tpch04 averaged_tpch05 averaged_tpch06 averaged_tpch07 averaged_tpch08 averaged_tpch09 averaged_tpch10 averaged_tpch11 averaged_tpch12 averaged_tpch13 averaged_tpch14 averaged_tpch15 averaged_tpch16 averaged_tpch17 averaged_tpch18 averaged_tpch19 averaged_tpch20 averaged_tpch21 averaged_tpch22 >> cogadb_averaged_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_min_tpch_measurement.csv
paste min_tpch01 min_tpch02 min_tpch03 min_tpch04 min_tpch05 min_tpch06 min_tpch07 min_tpch08 min_tpch09 min_tpch10 min_tpch11 min_tpch12 min_tpch13 min_tpch14 min_tpch15 min_tpch16 min_tpch17 min_tpch18 min_tpch19 min_tpch20 min_tpch21 min_tpch22 >> cogadb_min_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_max_tpch_measurement.csv
paste max_tpch01 max_tpch02 max_tpch03 max_tpch04 max_tpch05 max_tpch06 max_tpch07 max_tpch08 max_tpch09 max_tpch10 max_tpch11 max_tpch12 max_tpch13 max_tpch14 max_tpch15 max_tpch16 max_tpch17 max_tpch18 max_tpch19 max_tpch20 max_tpch21 max_tpch22 >> cogadb_max_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_variance_tpch_measurement.csv
paste variance_tpch01 variance_tpch02 variance_tpch03 variance_tpch04 variance_tpch05 variance_tpch06 variance_tpch07 variance_tpch08 variance_tpch09 variance_tpch10 variance_tpch11 variance_tpch12 variance_tpch13 variance_tpch14 variance_tpch15 variance_tpch16 variance_tpch17 variance_tpch18 variance_tpch19 variance_tpch20 variance_tpch21 variance_tpch22 >> cogadb_variance_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_standard_deviation_tpch_measurement.csv
paste standard_deviation_tpch01 standard_deviation_tpch02 standard_deviation_tpch03 standard_deviation_tpch04 standard_deviation_tpch05 standard_deviation_tpch06 standard_deviation_tpch07 standard_deviation_tpch08 standard_deviation_tpch09 standard_deviation_tpch10 standard_deviation_tpch11 standard_deviation_tpch12 standard_deviation_tpch13 standard_deviation_tpch14 standard_deviation_tpch15 standard_deviation_tpch16 standard_deviation_tpch17 standard_deviation_tpch18 standard_deviation_tpch19 standard_deviation_tpch20 standard_deviation_tpch21 standard_deviation_tpch22 >> cogadb_standard_deviation_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_standard_error_of_the_mean_tpch_measurement.csv
paste standard_error_of_the_mean_tpch01 standard_error_of_the_mean_tpch02 standard_error_of_the_mean_tpch03 standard_error_of_the_mean_tpch04 standard_error_of_the_mean_tpch05 standard_error_of_the_mean_tpch06 standard_error_of_the_mean_tpch07 standard_error_of_the_mean_tpch08 standard_error_of_the_mean_tpch09 standard_error_of_the_mean_tpch10 standard_error_of_the_mean_tpch11 standard_error_of_the_mean_tpch12 standard_error_of_the_mean_tpch13 standard_error_of_the_mean_tpch14 standard_error_of_the_mean_tpch15 standard_error_of_the_mean_tpch16 standard_error_of_the_mean_tpch17 standard_error_of_the_mean_tpch18 standard_error_of_the_mean_tpch19 standard_error_of_the_mean_tpch20 standard_error_of_the_mean_tpch21 standard_error_of_the_mean_tpch22 >> cogadb_standard_error_of_the_mean_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_lower_bound_of_error_tpch_measurement.csv
paste lower_bound_of_error_bar_tpch01 lower_bound_of_error_bar_tpch02 lower_bound_of_error_bar_tpch03 lower_bound_of_error_bar_tpch04 lower_bound_of_error_bar_tpch05 lower_bound_of_error_bar_tpch06 lower_bound_of_error_bar_tpch07 lower_bound_of_error_bar_tpch08 lower_bound_of_error_bar_tpch09 lower_bound_of_error_bar_tpch10 lower_bound_of_error_bar_tpch11 lower_bound_of_error_bar_tpch12 lower_bound_of_error_bar_tpch13 lower_bound_of_error_bar_tpch14 lower_bound_of_error_bar_tpch15 lower_bound_of_error_bar_tpch16 lower_bound_of_error_bar_tpch17 lower_bound_of_error_bar_tpch18 lower_bound_of_error_bar_tpch19 lower_bound_of_error_bar_tpch20 lower_bound_of_error_bar_tpch21 lower_bound_of_error_bar_tpch22 >> cogadb_lower_bound_of_error_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_upper_bound_of_error_tpch_measurement.csv
paste upper_bound_of_error_bar_tpch01 upper_bound_of_error_bar_tpch02 upper_bound_of_error_bar_tpch03 upper_bound_of_error_bar_tpch04 upper_bound_of_error_bar_tpch05 upper_bound_of_error_bar_tpch06 upper_bound_of_error_bar_tpch07 upper_bound_of_error_bar_tpch08 upper_bound_of_error_bar_tpch09 upper_bound_of_error_bar_tpch10 upper_bound_of_error_bar_tpch11 upper_bound_of_error_bar_tpch12 upper_bound_of_error_bar_tpch13 upper_bound_of_error_bar_tpch14 upper_bound_of_error_bar_tpch15 upper_bound_of_error_bar_tpch16 upper_bound_of_error_bar_tpch17 upper_bound_of_error_bar_tpch18 upper_bound_of_error_bar_tpch19 upper_bound_of_error_bar_tpch20 upper_bound_of_error_bar_tpch21 upper_bound_of_error_bar_tpch22 >> cogadb_upper_bound_of_error_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_lower_bound_of_standard_error_of_the_mean_error_bar_tpch_measurement.csv
paste lower_bound_of_standard_error_of_the_mean_error_bar_tpch01 lower_bound_of_standard_error_of_the_mean_error_bar_tpch02 lower_bound_of_standard_error_of_the_mean_error_bar_tpch03 lower_bound_of_standard_error_of_the_mean_error_bar_tpch04 lower_bound_of_standard_error_of_the_mean_error_bar_tpch05 lower_bound_of_standard_error_of_the_mean_error_bar_tpch06 lower_bound_of_standard_error_of_the_mean_error_bar_tpch07 lower_bound_of_standard_error_of_the_mean_error_bar_tpch08 lower_bound_of_standard_error_of_the_mean_error_bar_tpch09 lower_bound_of_standard_error_of_the_mean_error_bar_tpch10 lower_bound_of_standard_error_of_the_mean_error_bar_tpch11 lower_bound_of_standard_error_of_the_mean_error_bar_tpch12 lower_bound_of_standard_error_of_the_mean_error_bar_tpch13 lower_bound_of_standard_error_of_the_mean_error_bar_tpch14 lower_bound_of_standard_error_of_the_mean_error_bar_tpch15 lower_bound_of_standard_error_of_the_mean_error_bar_tpch16 lower_bound_of_standard_error_of_the_mean_error_bar_tpch17 lower_bound_of_standard_error_of_the_mean_error_bar_tpch18 lower_bound_of_standard_error_of_the_mean_error_bar_tpch19 lower_bound_of_standard_error_of_the_mean_error_bar_tpch20 lower_bound_of_standard_error_of_the_mean_error_bar_tpch21 lower_bound_of_standard_error_of_the_mean_error_bar_tpch22 >> cogadb_lower_bound_of_standard_error_of_the_mean_error_bar_tpch_measurement.csv

echo -e "Q1\tQ2\tQ3\tQ4\tQ5\tQ6\tQ7\tQ8\tQ9\tQ10\tQ11\tQ12\tQ13\tQ14\tQ15\tQ16\tQ17\tQ18\tQ19\tQ20\tQ21\tQ22" > cogadb_upper_bound_of_standard_error_of_the_mean_error_bar_tpch_measurement.csv
paste upper_bound_of_standard_error_of_the_mean_error_bar_tpch01 upper_bound_of_standard_error_of_the_mean_error_bar_tpch02 upper_bound_of_standard_error_of_the_mean_error_bar_tpch03 upper_bound_of_standard_error_of_the_mean_error_bar_tpch04 upper_bound_of_standard_error_of_the_mean_error_bar_tpch05 upper_bound_of_standard_error_of_the_mean_error_bar_tpch06 upper_bound_of_standard_error_of_the_mean_error_bar_tpch07 upper_bound_of_standard_error_of_the_mean_error_bar_tpch08 upper_bound_of_standard_error_of_the_mean_error_bar_tpch09 upper_bound_of_standard_error_of_the_mean_error_bar_tpch10 upper_bound_of_standard_error_of_the_mean_error_bar_tpch11 upper_bound_of_standard_error_of_the_mean_error_bar_tpch12 upper_bound_of_standard_error_of_the_mean_error_bar_tpch13 upper_bound_of_standard_error_of_the_mean_error_bar_tpch14 upper_bound_of_standard_error_of_the_mean_error_bar_tpch15 upper_bound_of_standard_error_of_the_mean_error_bar_tpch16 upper_bound_of_standard_error_of_the_mean_error_bar_tpch17 upper_bound_of_standard_error_of_the_mean_error_bar_tpch18 upper_bound_of_standard_error_of_the_mean_error_bar_tpch19 upper_bound_of_standard_error_of_the_mean_error_bar_tpch20 upper_bound_of_standard_error_of_the_mean_error_bar_tpch21 upper_bound_of_standard_error_of_the_mean_error_bar_tpch22 >> cogadb_upper_bound_of_standard_error_of_the_mean_error_bar_tpch_measurement.csv


else
   echo "Could not determine used workload!"
   exit -1
fi

exit 0
