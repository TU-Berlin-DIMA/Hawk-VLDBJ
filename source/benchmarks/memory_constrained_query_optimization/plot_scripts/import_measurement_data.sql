



create table measurements_cpu (mid INTEGER PRIMARY KEY, scale_factor integer, query_id integer, number_of_users integer, number_of_queries integer, number_of_executed_gpu_operators REAL, number_of_aborted_gpu_operators REAL, workload_execution_time_ms REAL, wasted_time_ms_due_to_aborted_gpu_operators REAL, minimal_query_execution_time_ms REAL, maximal_query_execution_time_ms REAL, average_query_execution_time_ms REAL, variance_in_query_execution_time_ms REAL);

create table measurements_any (mid INTEGER PRIMARY KEY, scale_factor integer, query_id integer, number_of_users integer, number_of_queries integer, number_of_executed_gpu_operators REAL, number_of_aborted_gpu_operators REAL, workload_execution_time_ms REAL, wasted_time_ms_due_to_aborted_gpu_operators REAL, minimal_query_execution_time_ms REAL, maximal_query_execution_time_ms REAL, average_query_execution_time_ms REAL, variance_in_query_execution_time_ms REAL);

.separator "\t"
--.import ../experimental_result_data/measurements_cpu.csv  measurements_cpu
--.import ../experimental_result_data/measurements_any.csv  measurements_any
.import measurements_cpu.csv  measurements_cpu
.import measurements_any.csv  measurements_any

