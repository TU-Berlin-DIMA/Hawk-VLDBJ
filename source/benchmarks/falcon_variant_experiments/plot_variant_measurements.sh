
mkdir -p config

for p in devices/*.coga; do
   processor=$(basename $p | sed -e 's/\.coga//g')
   for metric in compile_time_host compile_time_kernel execution_time; do
      echo "$processor-$metric" > config/PLOT_GENERATOR_OUTPUT_FILE_NAME
      echo "$processor-$metric" > config/PLOT_GENERATOR_OUTPUT_FILE_TITLE
      bash generate_latex_plot.sh results/$processor-*_summary_$metric.csv
#      for measurements in results/$processor-*_summary_$metric.csv; do
#         
#      done
   done 
done


