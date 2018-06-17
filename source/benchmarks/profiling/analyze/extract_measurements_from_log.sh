

#SEARCH_TAG="OPENCL_CODE_GENERATION_CPU_OPTIMIZED_SSB"

echo "VARIANT_TAG;Q11;Q12;Q13;Q21;Q22;Q23;Q31;Q32;Q33;Q34;Q41;Q42;Q43" > all_configurations_in_seconds.csv

for SEARCH_TAG in "C_CODE_GENERATION_CPU_OPTIMIZED_SSB" "OPENCL_CODE_GENERATION_CPU_OPTIMIZED_SSB"; do
#SEARCH_TAG="OPENCL_CODE_GENERATION_CPU_OPTIMIZED_SSB"

EXEC_TIME_LINE=""

echo "SEARCH_TAG: $SEARCH_TAG"

for i in 11 12 13 21 22 23 31 32 33 34 41 42 43; do 


cat $1 | grep "$SEARCH_TAG$i" | grep "Pipeline Execution Time" | awk '{print $4}' | sed -e 's/s//g' > tmp

if [[ -s tmp ]]; then
  EXECUTION_TIME=$(gawk -f ../performance_star_schema_benchmark/compute_average.awk tmp | awk '{print $3}')
else
  EXECUTION_TIME=-1
fi

if [[ -z $EXEC_TIME_LINE ]]; then
   EXEC_TIME_LINE="$EXECUTION_TIME"
else
   EXEC_TIME_LINE="$EXEC_TIME_LINE;$EXECUTION_TIME"
fi
cat tmp
echo "SSB$i: $EXECUTION_TIME"
done

echo "Q11;Q12;Q13;Q21;Q22;Q23;Q31;Q32;Q33;Q34;Q41;Q42;Q43" > "$SEARCH_TAG"_in_seconds.csv
echo "$EXEC_TIME_LINE" >> "$SEARCH_TAG"_in_seconds.csv
echo "$SEARCH_TAG;$EXEC_TIME_LINE" >> all_configurations_in_seconds.csv


done


