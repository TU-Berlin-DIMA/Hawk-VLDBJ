
set -x
set -e

SCALE_FACTORS=`cat ../../SCALE_FACTORS_TO_BENCHMARK`

for i in $SCALE_FACTORS; do
	cat data.csv | awk 'BEGIN{FS="|";}{if($1=='$i'){ print }}' > SF$i.csv
done

gnuplot average_query_execution_time_wrt_parallel_users.gnuplot 
gnuplot average_query_execution_time_wrt_abortion_rate.gnuplot

mkdir -p result
mv *.csv result/
mv *.pdf result/

