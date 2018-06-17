
SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
cd "$SCRIPTPATH"
pwd


cat ../experimental_result_data/measurements_cpu.csv | awk 'BEGIN{FS="|";}{if(NR!=1){print} }' > measurements_cpu.csv
cat ../experimental_result_data/measurements_any.csv | awk 'BEGIN{FS="|";}{if(NR!=1){print} }' > measurements_any.csv

rm -f measurements.db
sqlite3 measurements.db < import_measurement_data.sql

QUERY_ABORTED_GPU_OPERATORS="SELECT scale_factor as SF, number_of_users, MAX(number_of_executed_gpu_operators), MAX(number_of_aborted_gpu_operators), MAX(number_of_aborted_gpu_operators)/MAX(number_of_executed_gpu_operators) as ABORTION_RATE FROM measurements_any GROUP BY scale_factor, number_of_users;"

QUERY_WORKLOAD_EXECUTION_TIMES="SELECT measurements_any.scale_factor as SF, 
measurements_any.number_of_users, 
MAX(measurements_any.number_of_executed_gpu_operators), 
MAX(measurements_any.number_of_aborted_gpu_operators), 
MAX(measurements_any.number_of_aborted_gpu_operators)/MAX(measurements_any.number_of_executed_gpu_operators) as ABORTION_RATE,
AVG(measurements_any.workload_execution_time_ms) as HYBRID_WORKLOAD_EXECUTION_TIME,
AVG(measurements_cpu.workload_execution_time_ms) as CPU_ONLY_WORKLOAD_EXECUTION_TIME,
MAX(measurements_any.wasted_time_ms_due_to_aborted_gpu_operators) as MAX_WASTED_TIME
FROM measurements_any JOIN measurements_cpu ON (measurements_any.mid = measurements_cpu.mid) GROUP BY SF, measurements_any.number_of_users;"

QUERY_AVERAGE_QUERY_EXECUTION_TIMES="SELECT measurements_any.scale_factor as SF, 
measurements_any.number_of_users, 
MAX(measurements_any.number_of_executed_gpu_operators), 
MAX(measurements_any.number_of_aborted_gpu_operators), 
MAX(measurements_any.number_of_aborted_gpu_operators)/MAX(measurements_any.number_of_executed_gpu_operators) as ABORTION_RATE,
AVG(measurements_any.average_query_execution_time_ms) as AVERAGE_HYBRID_QUERY_EXECUTION_TIME,
AVG(measurements_cpu.average_query_execution_time_ms) as AVERAGE_CPU_ONLY_QUERY_EXECUTION_TIME,
MAX(measurements_any.wasted_time_ms_due_to_aborted_gpu_operators) as MAX_WASTED_TIME
FROM measurements_any JOIN measurements_cpu ON (measurements_any.mid = measurements_cpu.mid) GROUP BY SF, measurements_any.number_of_users;"

#QUERY_ABORTED_GPU_OPERATORS="SELECT scale_factor as SF, 
#number_of_users, MAX(number_of_executed_gpu_operators), 
#MAX(number_of_aborted_gpu_operators), 
#MAX(number_of_aborted_gpu_operators)/MAX(number_of_executed_gpu_operators) as ABORTION_RATE 
#FROM measurements_any 
#GROUP BY scale_factor, number_of_users;"

echo "$QUERY_ABORTED_GPU_OPERATORS" | sqlite3 -list measurements.db > abortion_rate_gpu_operators/data.csv
echo "$QUERY_WORKLOAD_EXECUTION_TIMES" | sqlite3 -list measurements.db > workload_execution_time/data.csv
echo "$QUERY_AVERAGE_QUERY_EXECUTION_TIMES" | sqlite3 -list measurements.db > query_execution_time/data.csv

for i in abortion_rate_gpu_operators workload_execution_time query_execution_time; do 
	cd $i;
	pwd
	bash create_plot.sh 
	cd ..
done

