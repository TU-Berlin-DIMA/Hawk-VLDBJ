
set -x
set -e

SCALE_FACTORS=`cat ../../SCALE_FACTORS_TO_BENCHMARK`

for i in $SCALE_FACTORS; do
	cat data.csv | awk 'BEGIN{FS="|";}{if($1=='$i'){ print }}' > SF$i.csv
done

gnuplot scheduled_gpu_operators.gnuplot 
gnuplot abortion_rate_gpu_operators.gnuplot

mkdir -p result
mv *.csv result/
mv *.pdf result/

