

RESULT_DIR=result_diagrams

mkdir -p "$RESULT_DIR"
SCALE_FACTORS_TO_BENCHMARK=`cat SCALE_FACTORS_TO_BENCHMARK`
for query in ssball; do
rm -f $query"_summary.csv"

for i in *traditional_optimizer*$query*$HOSTNAME; do 
   echo $i; 
   echo "CPU ONLY measurements"
   cat $i/measurements_cpu.csv  >> "$RESULT_DIR"/traditional_optimizer_measurements_cpu_only.csv; 
   echo "HYBRID CPU AND GPU measurements"
   cat $i/measurements_any.csv  >> "$RESULT_DIR"/traditional_optimizer_measurements_any.csv; 
done

for i in experimental_result_data_query_chopping_cpu_only_$query*$HOSTNAME; do 
   echo $i; 
   echo "CPU ONLY measurements" 
   cat $i/measurements_cpu.csv  >> "$RESULT_DIR"/query_chopping_optimizer_measurements_cpu_only.csv; 
   
done
for i in experimental_result_data_query_chopping_$query*$HOSTNAME; do
   echo $i 
   echo "HYBRID CPU AND GPU measurements"
   cat $i/measurements_any.csv  >> "$RESULT_DIR"/query_chopping_optimizer_measurements_any.csv; 

done

done

SCALE_FACTORS_TO_BENCHMARK=`cat SCALE_FACTORS_TO_BENCHMARK`


for i in $SCALE_FACTORS_TO_BENCHMARK; do

    mkdir -p "$RESULT_DIR"/"SF$i"
    cat "$RESULT_DIR"/traditional_optimizer_measurements_cpu_only.csv | awk '{if($2=='$i'){print}}' > "$RESULT_DIR"/"SF$i"/traditional_optimizer_measurements_cpu_only.csv
    cat "$RESULT_DIR"/traditional_optimizer_measurements_any.csv | awk '{if($2=='$i'){print}}' > "$RESULT_DIR"/"SF$i"/traditional_optimizer_measurements_any.csv
    cat "$RESULT_DIR"/query_chopping_optimizer_measurements_cpu_only.csv | awk '{if($2=='$i'){print}}' > "$RESULT_DIR"/"SF$i"/query_chopping_optimizer_measurements_cpu_only.csv
    cat "$RESULT_DIR"/query_chopping_optimizer_measurements_any.csv | awk '{if($2=='$i'){print}}' > "$RESULT_DIR"/"SF$i"/query_chopping_optimizer_measurements_any.csv
    
    NUM_PARALLEL_USERS=$(cat "$RESULT_DIR"/"SF$scale_factor"/query_chopping_optimizer_measurements_cpu_only.csv "$RESULT_DIR"/"SF$i"/query_chopping_optimizer_measurements_any.csv | awk '{print $4}' | sort -u)
    for num_users in $NUM_PARALLEL_USERS;do
        mkdir -p "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users
        
        cat "$RESULT_DIR"/"SF$i"/traditional_optimizer_measurements_cpu_only.csv | awk '{if($4=='$num_users'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/traditional_optimizer_measurements_cpu_only.csv
        cat "$RESULT_DIR"/"SF$i"/traditional_optimizer_measurements_any.csv | awk '{if($4=='$num_users'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/traditional_optimizer_measurements_any.csv
        cat "$RESULT_DIR"/"SF$i"/query_chopping_optimizer_measurements_cpu_only.csv | awk '{if($4=='$num_users'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/query_chopping_optimizer_measurements_cpu_only.csv
        cat "$RESULT_DIR"/"SF$i"/query_chopping_optimizer_measurements_any.csv | awk '{if($4=='$num_users'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/query_chopping_optimizer_measurements_any.csv
        
        LOAD_BALANCING_STRATEGIES=$(cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/query_chopping_optimizer_measurements_cpu_only.csv | awk '{print $8}' | sort -u)
        echo "LOAD_BALANCING_STRATEGIES: $LOAD_BALANCING_STRATEGIES"
        for load_balancer in $LOAD_BALANCING_STRATEGIES; do
            mkdir -p "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"
            cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/traditional_optimizer_measurements_cpu_only.csv | awk '{if($8=="'$load_balancer'"){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/traditional_optimizer_measurements_cpu_only.csv
            cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/traditional_optimizer_measurements_any.csv | awk '{if($8=="'$load_balancer'"){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/traditional_optimizer_measurements_any.csv
            cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/query_chopping_optimizer_measurements_cpu_only.csv | awk '{if($8=="'$load_balancer'"){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/query_chopping_optimizer_measurements_cpu_only.csv
            cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/query_chopping_optimizer_measurements_any.csv | awk '{if($8=="'$load_balancer'"){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/query_chopping_optimizer_measurements_any.csv
            
            for mem_cost_model in 0 1; do
               mkdir -p "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"
               cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/traditional_optimizer_measurements_cpu_only.csv | awk '{if($10=='$mem_cost_model'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/traditional_optimizer_measurements_cpu_only.csv
               cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/traditional_optimizer_measurements_any.csv | awk '{if($10=='$mem_cost_model'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/traditional_optimizer_measurements_any.csv
               cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/query_chopping_optimizer_measurements_cpu_only.csv | awk '{if($10=='$mem_cost_model'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/query_chopping_optimizer_measurements_cpu_only.csv
               cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/query_chopping_optimizer_measurements_any.csv | awk '{if($10=='$mem_cost_model'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/query_chopping_optimizer_measurements_any.csv
               
               mkdir -p "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/qc_varying_cpus
               CPU_CONFIGURATIONS=$(cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/query_chopping_optimizer_measurements_cpu_only.csv | awk '{print $6}' | sort -u)
               for num_cpus in $CPU_CONFIGURATIONS; do
                   cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/query_chopping_optimizer_measurements_cpu_only.csv | awk '{if($6=='$num_cpus'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/qc_varying_cpus/qc_"$num_cpus"_cpus.csv 
               done
               mkdir -p "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/qc_varying_gpus
               GPU_CONFIGURATIONS=$(cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/query_chopping_optimizer_measurements_any.csv | awk '{print $7}' | sort -u)
               for num_gpus in $GPU_CONFIGURATIONS; do
                   cat "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/query_chopping_optimizer_measurements_any.csv | awk '{if($7=='$num_gpus'){print}}' > "$RESULT_DIR"/"SF$i"/"$num_users"_parallel_users/lbs_"$load_balancer"/mem_cost_model_"$mem_cost_model"/qc_varying_gpus/qc_"$num_gpus"_gpus.csv 
               done
               
            done 
        done  
    done
done

#echo $SCALE_FACTORS_TO_BENCHMARK
GNUPLOT_SCRIPT_HEADER="
set terminal unknown
"

for scale_factor in $SCALE_FACTORS_TO_BENCHMARK; do

    NUM_PARALLEL_USERS=$(cat "$RESULT_DIR"/"SF$scale_factor"/query_chopping_optimizer_measurements_cpu_only.csv "$RESULT_DIR"/"SF$i"/query_chopping_optimizer_measurements_any.csv | awk '{print $4}' | sort -u)
    #NUM_PARALLEL_USERS=$(cat "$RESULT_DIR"/"SF$scale_factor"/query_chopping_optimizer_measurements_cpu_only.csv | awk '{print $4}' | sort -u)
    for num_users in $NUM_PARALLEL_USERS;do
         DIR="$RESULT_DIR"/"SF$scale_factor"/"$num_users"_parallel_users

#Workload Execution time
echo "$GNUPLOT_SCRIPT_HEADER
set xtics nomirror
set ytics nomirror
set xlabel 'Number of virtual CPUs'
set ylabel 'Workload Execution Time in ms' offset 0,-2
set title 'Workload Execution Time Depending on Number of virtual CPUs (SF $scale_factor, #Users: $num_users, Machine: $HOSTNAME)'
set key top right Left reverse samplen 1
set key below
set key box
set yrange [ 0: ] 
plot 'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_cpu_only.csv' using 6:13 title \"QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_cpu_only.csv' using 6:13 title \"Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_cpu_only.csv' using 6:13 title \"QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_cpu_only.csv' using 6:13 title \"Greedy WTAR\" w p lw 4
set output \"workload_execution_time_dep_on_num_virtual_cpus_sf_"$scale_factor"_num_users_"$num_users".pdf\"
set terminal pdfcairo font \"Helvetica,9\" size 5, 4
replot
" > "$DIR"/workload_execution_time_dep_on_num_virtual_cpus_sf_"$scale_factor"_num_users_"$num_users".gnuplot

echo "$GNUPLOT_SCRIPT_HEADER
set xtics nomirror
set ytics nomirror
set xlabel 'Number of virtual GPUs'
set ylabel 'Workload Execution Time in ms' offset 0,-2
set title 'Workload Execution Time Depending on Number of virtual GPUs (SF $scale_factor, #Users: $num_users, Machine: $HOSTNAME)'
set key below
set key box
set yrange [ 0: ] 
plot 'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:13 title \"QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:13 title \"Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:13 title \"QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:13 title \"Greedy WTAR\" w p lw 4
set output \"workload_execution_time_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".pdf\"
set terminal pdfcairo font \"Helvetica,9\" size 5, 4
replot
" > "$DIR"/workload_execution_time_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".gnuplot

#Total CPU (and Workload Execution Time)

#Workload Execution Time and Total CPU Time
echo "$GNUPLOT_SCRIPT_HEADER
set xtics nomirror
set ytics nomirror
set xlabel 'Number of virtual CPUs'
set ylabel 'Workload Execution Time and Total CPU Time in ms' offset 0,-2
set title 'Workload Execution Time and Total CPU Time Depending on Number of virtual CPUs (SF $scale_factor, #Users: $num_users, Machine: $HOSTNAME)'
set key top right Left reverse samplen 1
set key below
set key box
set yrange [ 0: ] 
plot 'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_cpu_only.csv' using 6:13 title \"Workload Execution Time and Total CPU Time QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_cpu_only.csv' using 6:13 title \"Workload Execution Time and Total CPU Time Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_cpu_only.csv' using 6:13 title \"Workload Execution Time and Total CPU Time QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_cpu_only.csv' using 6:13 title \"Workload Execution Time and Total CPU Time Greedy WTAR\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_cpu_only.csv' using 6:14 title \"Total CPU Time Workload QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_cpu_only.csv' using 6:14 title \"Total CPU Time Workload Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_cpu_only.csv' using 6:14 title \"Total CPU Time Workload QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_cpu_only.csv' using 6:14 title \"Total CPU Time Workload Greedy WTAR\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_1/query_chopping_optimizer_measurements_cpu_only.csv' using 6:13 title \"Workload Execution Time and Total CPU Time QC SRT with Mem Cost Model\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_1/traditional_optimizer_measurements_cpu_only.csv' using 6:13 title \"Workload Execution Time and Total CPU Time Greedy SRT with Mem Cost Model\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_1/query_chopping_optimizer_measurements_cpu_only.csv' using 6:13 title \"Workload Execution Time and Total CPU Time QC WTAR with Mem Cost Model\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_1/traditional_optimizer_measurements_cpu_only.csv' using 6:13 title \"Workload Execution Time and Total CPU Time Greedy WTAR with Mem Cost Model\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_1/query_chopping_optimizer_measurements_cpu_only.csv' using 6:14 title \"Total CPU Time Workload QC SRT with Mem Cost Model\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_1/traditional_optimizer_measurements_cpu_only.csv' using 6:14 title \"Total CPU Time Workload Greedy SRT with Mem Cost Model\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_1/query_chopping_optimizer_measurements_cpu_only.csv' using 6:14 title \"Total CPU Time Workload QC WTAR with Mem Cost Model\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_1/traditional_optimizer_measurements_cpu_only.csv' using 6:14 title \"Total CPU Time Workload Greedy WTAR with Mem Cost Model\" w p lw 4
set output \"total_cpu_time_and_workload_execution_time_dep_on_num_virtual_cpus_sf_"$scale_factor"_num_users_"$num_users".pdf\"
set terminal pdfcairo font \"Helvetica,9\" size 5, 4
replot
" > "$DIR"/total_cpu_time_and_workload_execution_time_dep_on_num_virtual_cpus_sf_"$scale_factor"_num_users_"$num_users".gnuplot

echo "$GNUPLOT_SCRIPT_HEADER
set xtics nomirror
set ytics nomirror
set xlabel 'Number of virtual GPUs'
set ylabel 'Workload Execution Time and Total CPU Time in ms' offset 0,-2
set title 'Workload Execution Time and Total CPU Time Depending on Number of virtual GPUs (SF $scale_factor, #Users: $num_users, Machine: $HOSTNAME)'
set key below
set key box
set yrange [ 0: ] 
plot 'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time and Total CPU Time QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time and Total CPU Time Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time and Total CPU Time QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time and Total CPU Time Greedy WTAR\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload Greedy WTAR\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_1/query_chopping_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time and Total CPU Time QC SRT with Mem Cost Model\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_1/traditional_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time and Total CPU Time Greedy SRT with Mem Cost Model\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_1/query_chopping_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time and Total CPU Time QC WTAR with Mem Cost Model\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_1/traditional_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time and Total CPU Time Greedy WTAR with Mem Cost Model\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_1/query_chopping_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload QC SRT with Mem Cost Model\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_1/traditional_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload Greedy SRT with Mem Cost Model\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_1/query_chopping_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload QC WTAR with Mem Cost Model\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_1/traditional_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload Greedy WTAR with Mem Cost Model\" w p lw 4
set output \"total_cpu_time_and_workload_execution_time_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".pdf\"
set terminal pdfcairo font \"Helvetica,9\" size 5, 4
replot
" > "$DIR"/total_cpu_time_and_workload_execution_time_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".gnuplot

#average query execution time
echo "$GNUPLOT_SCRIPT_HEADER
set xtics nomirror
set ytics nomirror
set xlabel 'Number of virtual CPUs'
set ylabel 'Average Query Execution Time in ms' offset 0,-2
set title 'Average Query Execution Time Depending on Number of virtual CPUs (SF $scale_factor, #Users: $num_users, Machine: $HOSTNAME)'
set key top right Left reverse samplen 1
set key below
set key box
set yrange [ 0: ] 
plot 'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_cpu_only.csv' using 6:18 title \"QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_cpu_only.csv' using 6:18 title \"Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_cpu_only.csv' using 6:18 title \"QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_cpu_only.csv' using 6:18 title \"Greedy WTAR\" w p lw 4
set output \"average_query_execution_time_dep_on_num_virtual_cpus_sf_"$scale_factor"_num_users_"$num_users".pdf\"
set terminal pdfcairo font \"Helvetica,9\" size 5, 4
replot
" > "$DIR"/average_query_execution_time_dep_on_num_virtual_cpus_sf_"$scale_factor"_num_users_"$num_users".gnuplot

echo "$GNUPLOT_SCRIPT_HEADER
set xtics nomirror
set ytics nomirror
set xlabel 'Number of virtual GPUs'
set ylabel 'Average Query Execution Time in ms' offset 0,-2
set title 'Average Query Execution Time Depending on Number of virtual GPUs (SF $scale_factor, #Users: $num_users, Machine: $HOSTNAME)'
set key below
set key box
set yrange [ 0: ] 
plot 'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:18 title \"QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:18 title \"Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:18 title \"QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:18 title \"Greedy WTAR\" w p lw 4
set output \"average_query_execution_time_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".pdf\"
set terminal pdfcairo font \"Helvetica,9\" size 5, 4
replot
" > "$DIR"/average_query_execution_time_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".gnuplot

#Penalty Time due to aborted GPU Operators
echo "$GNUPLOT_SCRIPT_HEADER
set xtics nomirror
set ytics nomirror
set xlabel 'Number of virtual GPUs'
set ylabel 'Time Penalty due to aborted GPU operators in ms' offset 0,-2
set title 'Time Penalty due to aborted GPU operators Depending on Number of virtual GPUs (SF $scale_factor, #Users: $num_users, Machine: $HOSTNAME)'
set key below
set key box
set yrange [ 0: ] 
plot 'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:(\$15/1000000) title \"Penalty Time QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:(\$15/1000000) title \"Penalty Time Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:(\$15/1000000) title \"Penalty Time QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:(\$15/1000000) title \"Penalty Time Greedy WTAR\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:13 title \"Workload Execution Time Greedy WTAR\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:14 title \"Total CPU Time Workload Greedy WTAR\" w p lw 4
set output \"penalty_time_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".pdf\"
set terminal pdfcairo font \"Helvetica,9\" size 5, 4
replot
" > "$DIR"/penalty_time_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".gnuplot

#GPU Operator Abortion Rate
echo "$GNUPLOT_SCRIPT_HEADER
set xtics nomirror
set ytics nomirror
set xlabel 'Number of virtual GPUs'
set ylabel 'GPU Operator Abortion Rate' offset 0,-2
set title 'GPU Operator Abortion Rate Depending on Number of virtual GPUs (SF $scale_factor, #Users: $num_users, Machine: $HOSTNAME)'
set key below
set key box
set yrange [ 0: ] 
plot 'lbs_ResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:(\$12/\$11) title \"QC SRT\" w p lw 4, \
'lbs_ResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:(\$12/\$11) title \"Greedy SRT\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/query_chopping_optimizer_measurements_any.csv' using 7:(\$12/\$11) title \"QC WTAR\" w p lw 4, \
'lbs_WaitingTimeAwareResponseTime/mem_cost_model_0/traditional_optimizer_measurements_any.csv' using 7:(\$12/\$11) title \"Greedy WTAR\" w p lw 4
set output \"gpu_operator_abortion_rate_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".pdf\"
set terminal pdfcairo font \"Helvetica,9\" size 5, 4
replot
" > "$DIR"/gpu_operator_abortion_rate_dep_on_num_virtual_gpus_sf_"$scale_factor"_num_users_"$num_users".gnuplot


        CURRENT_DIR=$(pwd)
        cd "$DIR"
        for script in *.gnuplot; do
            gnuplot $script
        done
        cd "$CURRENT_DIR"

    done
done

zip -r "$RESULT_DIR.zip" "$RESULT_DIR"

exit 0





for i in experimental_result_data_traditional_optimizer_ssb*; do

   cat "$i"/measurements_cpu.csv | grep -E "ID|ResponseTime" > cpu_only_opt_greedy_lbs_srt.csv
   cat "$i"/measurements_cpu.csv | grep -E "ID|WaitingTimeAwareResponseTime" > cpu_only_opt_greedy_lbs_wtar.csv
   
   cat "$i"/measurements_any.csv | grep -E "ID|ResponseTime" > cpu_gpu_opt_greedy_lbs_srt.csv
   cat "$i"/measurements_any.csv | grep -E "ID|WaitingTimeAwareResponseTime" > cpu_gpu_opt_greedy_lbs_wtar.csv

done

for i in experimental_result_data_query_chopping_*; do

   cat "$i"/measurements_any.csv | grep -E "ID|ResponseTime" > "$i"/cpu_gpu_opt_qc_lbs_srt.csv
   cat "$i"/measurements_any.csv | grep -E "ID|WaitingTimeAwareResponseTime" > "$i"/cpu_gpu_opt_qc_lbs_wtar.csv
  
   for u in "$i"/cpu_gpu_opt_qc_lbs_srt.csv "$i"/cpu_gpu_opt_qc_lbs_wtar.csv; do
       #cat "$u" | awk 
   done
done

