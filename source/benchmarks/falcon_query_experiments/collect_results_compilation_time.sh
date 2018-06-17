#!/bin/bash

rm -f kernel_compile_times_collected_result.log
touch kernel_compile_times_collected_result.log

DELIMITER='&'
LINE_ENDING='\\'

HEADER="query"
for q in query_* tpch*; do
  cd $q/results;
  for d in cpu igpu dgpu phi; do
     HEADER="$HEADER$DELIMITER$d-cpu-optimized$DELIMITER$d-igpu-optimized$DELIMITER$d-dgpu-optimized$DELIMITER$d-phi-optimized"
  done
  cd ../..
  HEADER="$HEADER$LINE_ENDING"
  break
done

echo $HEADER > kernel_compile_times_collected_result.log

for q in query_* tpch*; do
  cd $q/results;
  LINE="$q"
  for d in cpu igpu dgpu phi; do
    echo -e "\nquery: $q, device: $d"
    for device_optimized in cpu_optimized igpu_optimized dgpu_optimized phi_optimized; do
       e="$d-$device_optimized""_csv_summary_compile_time_kernel.csv"
       if [ ! -f $e ]; then
           echo "File $e does not exist."
           execution_time="N/A"
       else
           cat $e | awk '{print $3"\t"$7"\t"$8}'
           execution_time=$(tail -n 1 $e | awk '{print $3}')
           #convert measurement value to fixed point with leading zero if required
           execution_time=$(echo "scale=3; $execution_time/1" | bc -l | sed 's/^\./0./')
       fi
       LINE="$LINE$DELIMITER$execution_time"
    done
  done

  cd ../..
  LINE="$LINE$LINE_ENDING"
  echo $LINE >> kernel_compile_times_collected_result.log
  cp kernel_compile_times_collected_result.log kernel_compile_times_collected_result.csv

done
