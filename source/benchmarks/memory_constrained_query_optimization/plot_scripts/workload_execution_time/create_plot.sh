
set -x
set -e

SCALE_FACTORS=`cat ../../SCALE_FACTORS_TO_BENCHMARK`

for i in $SCALE_FACTORS; do
	cat data.csv | awk 'BEGIN{FS="|";}{if($1=='$i'){ print }}' > SF$i.csv
done

gnuplot workload_execution_time_penalty_wrt_parallel_users.gnuplot
gnuplot workload_execution_time_wrt_parallel_users.gnuplot 
gnuplot workload_execution_time_wrt_abortion_rate.gnuplot
gnuplot workload_execution_time_penalty_wrt_parallel_users.gnuplot
gnuplot workload_execution_time_percentaged_penalty_wrt_parallel_users.gnuplot

mkdir -p result
mv *.csv result/
mv *.pdf result/

