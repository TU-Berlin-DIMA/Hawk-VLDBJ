

for workload in ssb_all tpch_all_supported; do

 WORKLOAD_DIRECTORIES=`find . -maxdepth 1 -mindepth 1 -type d | grep "$workload" | grep "_cleaned"`
 echo $WORKLOAD_DIRECTORIES

 SINGLE_USER_WORKLOAD_DIRECTORIES=`find . -maxdepth 1 -mindepth 1 -type d | grep "$workload" | grep "_cleaned" | grep "single_user" | sort -V`
 MULTI_USER_WORKLOAD_DIRECTORIES=`find . -maxdepth 1 -mindepth 1 -type d | grep "$workload" | grep "_cleaned" | grep "numuser" | sort -V`



rm -f workload_time_collected_sf*.csv
rm -f workload_time_"$workload"_collected_varying_scale_factor.csv
rm -f workload_time_"$workload"_collected_varying_parallel_users.csv

rm -f aborted_gpu_operators_"$workload"_collected_varying_scale_factor.csv
rm -f executed_gpu_operators_"$workload"_collected_varying_scale_factor.csv
rm -f copy_times_cpu_to_gpu_"$workload"_collected_varying_scale_factor.csv
rm -f copy_times_gpu_to_cpu_"$workload"_collected_varying_scale_factor.csv
rm -f wasted_time_by_aborts_"$workload"_collected_varying_scale_factor.csv

rm -f aborted_gpu_operators_"$workload"_collected_varying_parallel_users.csv
rm -f executed_gpu_operators_"$workload"_collected_varying_parallel_users.csv
rm -f copy_times_cpu_to_gpu_"$workload"_collected_varying_parallel_users.csv
rm -f copy_times_gpu_to_cpu_"$workload"_collected_varying_parallel_users.csv
rm -f wasted_time_by_aborts_"$workload"_collected_varying_parallel_users.csv

for experiment in $SINGLE_USER_WORKLOAD_DIRECTORIES; do
   echo "'$experiment'"
   echo $experiment | awk 'BEGIN{FS="-";} {print $2}'  
   SCALE_FACTOR=$(echo $experiment | awk 'BEGIN{FS="-";} {print $2}' | awk 'BEGIN{FS="_";} {print $3}' | sed -e 's/sf//g')
   if [ ! -e workload_time_"$workload"_collected_varying_scale_factor.csv ]; then
       echo "scale_factor	dummy	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  workload_time_"$workload"_collected_varying_scale_factor.csv 
  fi
   CURRENT_LINE=""
##############################################################################################################################
#workload execution time
##############################################################################################################################
   cd $experiment/varying_data_placement_strategy 
   for i in `find . -name workload_execution_time.csv | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		echo "Data Driven"
		DATA_DRIVEN_COMPILE_TIME=$(tail -n 1 $i | awk '{print $2}')
		DATA_DRIVEN_QUERY_CHOPPING=$(tail -n 1 $i | awk '{print $7}')		
		CURRENT_LINE="$CURRENT_LINE	$DATA_DRIVEN_COMPILE_TIME	$DATA_DRIVEN_QUERY_CHOPPING"
		echo "$CURRENT_LINE" >> ../../workload_time_"$workload"_collected_varying_scale_factor.csv
                #echo "CSV: '$CURRENT_LINE'"
		CURRENT_LINE=""
	else
		echo "Operator Driven"
		CURRENT_LINE="$SCALE_FACTOR	"$(tail -n 1 $i)
	fi
   done
   cd ../..

##############################################################################################################################
#number of executed and aborted GPU operators
##############################################################################################################################
   if [ ! -e aborted_gpu_operators_"$workload"_collected_varying_scale_factor.csv ]; then
       echo "scale_factor	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  aborted_gpu_operators_"$workload"_collected_varying_scale_factor.csv
  fi

   if [ ! -e executed_gpu_operators_"$workload"_collected_varying_scale_factor.csv ]; then
       echo "scale_factor	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  executed_gpu_operators_"$workload"_collected_varying_scale_factor.csv
  fi

   cd $experiment/varying_data_placement_strategy 
   for i in `find . -maxdepth 1 -mindepth 1 -type d | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		DATA_DRIVEN=$(head -n 1 "$i/best_effort_gpu_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		DATA_DRIVEN_CHOPPING=$(head -n 1 "$i/query_chopping/number_of_aborted_and_executed_gpu_operators.csv")
	else

		BEST_EFFORT_GPU=$(head -n 1 "$i/best_effort_gpu_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		CRITICAL_PATH=$(head -n 1 "$i/critical_path_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		GREEDY_CHAINER=$(head -n 1 "$i/greedy_chainer_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		GREEDY=$(head -n 1 "$i/greedy_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		CPU_ONLY=$(head -n 1 "$i/greedy_heuristic_cpu_only/number_of_aborted_and_executed_gpu_operators.csv")
		CHOPPING=$(head -n 1 "$i/query_chopping/number_of_aborted_and_executed_gpu_operators.csv")
		echo "Operator Driven"
#		read line
	fi
   done
	###############################################################################################################################
	BEST_EFFORT_GPU_ABORTED_OPERATORS=$(echo "$BEST_EFFORT_GPU" | awk '{print $2}')
	CRITICAL_PATH_ABORTED_OPERATORS=$(echo "$CRITICAL_PATH" | awk '{print $2}')
	GREEDY_CHAINER_ABORTED_OPERATORS=$(echo "$GREEDY_CHAINER" | awk '{print $2}')
	GREEDY_ABORTED_OPERATORS=$(echo "$GREEDY" | awk '{print $2}')
	CPU_ONLY_ABORTED_OPERATORS=$(echo "$CPU_ONLY" | awk '{print $2}')
	CHOPPING_ABORTED_OPERATORS=$(echo "$CHOPPING" | awk '{print $2}')
	DATA_DRIVEN_ABORTED_OPERATORS=$(echo "$DATA_DRIVEN" | awk '{print $2}')
	DATA_DRIVEN_CHOPPING_ABORTED_OPERATORS=$(echo "$DATA_DRIVEN_CHOPPING" | awk '{print $2}')

	CURRENT_LINE="$SCALE_FACTOR	$BEST_EFFORT_GPU_ABORTED_OPERATORS	$CRITICAL_PATH_ABORTED_OPERATORS	$GREEDY_CHAINER_ABORTED_OPERATORS	$GREEDY_ABORTED_OPERATORS	$CPU_ONLY_ABORTED_OPERATORS	$CHOPPING_ABORTED_OPERATORS	$DATA_DRIVEN_ABORTED_OPERATORS	$DATA_DRIVEN_CHOPPING_ABORTED_OPERATORS"
	echo "$CURRENT_LINE" >> ../../aborted_gpu_operators_"$workload"_collected_varying_scale_factor.csv
	###############################################################################################################################
	BEST_EFFORT_GPU_EXECUTED_OPERATORS=$(echo "$BEST_EFFORT_GPU" | awk '{print $3}')
	CRITICAL_PATH_EXECUTED_OPERATORS=$(echo "$CRITICAL_PATH" | awk '{print $3}')
	GREEDY_CHAINER_EXECUTED_OPERATORS=$(echo "$GREEDY_CHAINER" | awk '{print $3}')
	GREEDY_EXECUTED_OPERATORS=$(echo "$GREEDY" | awk '{print $3}')
	CPU_ONLY_EXECUTED_OPERATORS=$(echo "$CPU_ONLY" | awk '{print $3}')
	CHOPPING_EXECUTED_OPERATORS=$(echo "$CHOPPING" | awk '{print $3}')
	DATA_DRIVEN_EXECUTED_OPERATORS=$(echo "$DATA_DRIVEN" | awk '{print $3}')
	DATA_DRIVEN_CHOPPING_EXECUTED_OPERATORS=$(echo "$DATA_DRIVEN_CHOPPING" | awk '{print $3}')

	CURRENT_LINE="$SCALE_FACTOR	$BEST_EFFORT_GPU_EXECUTED_OPERATORS	$CRITICAL_PATH_EXECUTED_OPERATORS	$GREEDY_CHAINER_EXECUTED_OPERATORS	$GREEDY_EXECUTED_OPERATORS	$CPU_ONLY_EXECUTED_OPERATORS	$CHOPPING_EXECUTED_OPERATORS	$DATA_DRIVEN_EXECUTED_OPERATORS	$DATA_DRIVEN_CHOPPING_EXECUTED_OPERATORS"
	echo "$CURRENT_LINE" >> ../../executed_gpu_operators_"$workload"_collected_varying_scale_factor.csv
   ###############################################################################################################################
   cd ../..


##############################################################################################################################
# COPY TIME CPU TO GPU
############################################################################################################################## 

   if [ ! -e copy_times_cpu_to_gpu_"$workload"_collected_varying_scale_factor.csv ]; then
       echo "scale_factor	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  copy_times_cpu_to_gpu_"$workload"_collected_varying_scale_factor.csv
  fi  
   cd $experiment/varying_data_placement_strategy 
   for i in `find . -maxdepth 1 -mindepth 1 -type d | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		DATA_DRIVEN_COPY_TIME_CPU_GPU=$(head -n 1 "$i/best_effort_gpu_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		DATA_DRIVEN_CHOPPING_COPY_TIME_CPU_GPU=$(head -n 1 "$i/query_chopping/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
	else
		BEST_EFFORT_GPU_COPY_TIME_CPU_GPU=$(head -n 1 "$i/best_effort_gpu_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		CRITICAL_PATH_COPY_TIME_CPU_GPU=$(head -n 1 "$i/critical_path_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		GREEDY_CHAINER_COPY_TIME_CPU_GPU=$(head -n 1 "$i/greedy_chainer_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		GREEDY_COPY_TIME_CPU_GPU=$(head -n 1 "$i/greedy_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		CPU_ONLY_COPY_TIME_CPU_GPU=$(head -n 1 "$i/greedy_heuristic_cpu_only/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		CHOPPING_COPY_TIME_CPU_GPU=$(head -n 1 "$i/query_chopping/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		
#		read line
	fi
   done

	CURRENT_LINE="$SCALE_FACTOR	$BEST_EFFORT_GPU_COPY_TIME_CPU_GPU	$CRITICAL_PATH_COPY_TIME_CPU_GPU	$GREEDY_CHAINER_COPY_TIME_CPU_GPU	$GREEDY_COPY_TIME_CPU_GPU	$CPU_ONLY_COPY_TIME_CPU_GPU	$CHOPPING_COPY_TIME_CPU_GPU	$DATA_DRIVEN_COPY_TIME_CPU_GPU	$DATA_DRIVEN_CHOPPING_COPY_TIME_CPU_GPU"
	echo "$CURRENT_LINE" >> ../../copy_times_cpu_to_gpu_"$workload"_collected_varying_scale_factor.csv

   ###############################################################################################################################
   cd ../..

##############################################################################################################################
# COPY TIME GPU to CPU
############################################################################################################################## 
   if [ ! -e copy_times_gpu_to_cpu_"$workload"_collected_varying_scale_factor.csv ]; then
       echo "scale_factor	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  copy_times_gpu_to_cpu_"$workload"_collected_varying_scale_factor.csv
  fi  
   cd $experiment/varying_data_placement_strategy 
   for i in `find . -maxdepth 1 -mindepth 1 -type d | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		DATA_DRIVEN_COPY_TIME_GPU_CPU=$(head -n 2 "$i/best_effort_gpu_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		DATA_DRIVEN_CHOPPING_COPY_TIME_GPU_CPU=$(head -n 2 "$i/query_chopping/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
	else
		BEST_EFFORT_GPU_COPY_TIME_GPU_CPU=$(head -n 2 "$i/best_effort_gpu_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		CRITICAL_PATH_COPY_TIME_GPU_CPU=$(head -n 2 "$i/critical_path_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		GREEDY_CHAINER_COPY_TIME_GPU_CPU=$(head -n 2 "$i/greedy_chainer_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		GREEDY_COPY_TIME_GPU_CPU=$(head -n 2 "$i/greedy_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		CPU_ONLY_COPY_TIME_GPU_CPU=$(head -n 2 "$i/greedy_heuristic_cpu_only/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		CHOPPING_COPY_TIME_GPU_CPU=$(head -n 2 "$i/query_chopping/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		
#		read line
	fi
   done

	CURRENT_LINE="$SCALE_FACTOR	$BEST_EFFORT_GPU_COPY_TIME_GPU_CPU	$CRITICAL_PATH_COPY_TIME_GPU_CPU	$GREEDY_CHAINER_COPY_TIME_GPU_CPU	$GREEDY_COPY_TIME_GPU_CPU	$CPU_ONLY_COPY_TIME_GPU_CPU	$CHOPPING_COPY_TIME_GPU_CPU	$DATA_DRIVEN_COPY_TIME_GPU_CPU	$DATA_DRIVEN_CHOPPING_COPY_TIME_GPU_CPU"
	echo "$CURRENT_LINE" >> ../../copy_times_gpu_to_cpu_"$workload"_collected_varying_scale_factor.csv

   ###############################################################################################################################
   cd ../..


##############################################################################################################################
# WASTED TIME DUE TO ABORTED GPU OPERATORS
##############################################################################################################################


   if [ ! -e wasted_time_by_aborts_"$workload"_collected_varying_scale_factor.csv ]; then
       echo "scale_factor	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  wasted_time_by_aborts_"$workload"_collected_varying_scale_factor.csv
  fi  
   cd $experiment/varying_data_placement_strategy 
   for i in `find . -maxdepth 1 -mindepth 1 -type d | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		DATA_DRIVEN_WASTED_TIME_BY_ABORTS=$(cat "$i/best_effort_gpu_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		DATA_DRIVEN_CHOPPING_WASTED_TIME_BY_ABORTS=$(cat "$i/query_chopping/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
	else
		BEST_EFFORT_GPU_WASTED_TIME_BY_ABORTS=$(cat "$i/best_effort_gpu_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		CRITICAL_PATH_WASTED_TIME_BY_ABORTS=$(cat "$i/critical_path_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		GREEDY_CHAINER_WASTED_TIME_BY_ABORTS=$(cat "$i/greedy_chainer_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		GREEDY_WASTED_TIME_BY_ABORTS=$(cat "$i/greedy_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		CPU_ONLY_WASTED_TIME_BY_ABORTS=$(cat "$i/greedy_heuristic_cpu_only/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		CHOPPING_WASTED_TIME_BY_ABORTS=$(cat "$i/query_chopping/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		
#		read line
	fi
   done

	CURRENT_LINE="$SCALE_FACTOR	$BEST_EFFORT_GPU_WASTED_TIME_BY_ABORTS	$CRITICAL_PATH_WASTED_TIME_BY_ABORTS	$GREEDY_CHAINER_WASTED_TIME_BY_ABORTS	$GREEDY_WASTED_TIME_BY_ABORTS	$CPU_ONLY_WASTED_TIME_BY_ABORTS	$CHOPPING_WASTED_TIME_BY_ABORTS	$DATA_DRIVEN_WASTED_TIME_BY_ABORTS	$DATA_DRIVEN_CHOPPING_WASTED_TIME_BY_ABORTS"
	echo "$CURRENT_LINE" >> ../../wasted_time_by_aborts_"$workload"_collected_varying_scale_factor.csv

   ###############################################################################################################################
   cd ../..

##############################################################################################################################
# Per Query Peformance Plot
##############################################################################################################################



   cd $experiment/varying_data_placement_strategy 
   for i in `find . -name experiment_result.csv  | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		#echo "Data Driven"
		#DATA_DRIVEN_COMPILE_TIME=$(tail -n 1 $i | awk '{print $2}')
		#DATA_DRIVEN_QUERY_CHOPPING=$(tail -n 1 $i | awk '{print $7}')		
		#CURRENT_LINE="$CURRENT_LINE	$DATA_DRIVEN_COMPILE_TIME	$DATA_DRIVEN_QUERY_CHOPPING"
		#echo "$CURRENT_LINE" >> ../../query_times_"$workload"_collected_varying_scale_factor.csv
                #echo "CSV: '$CURRENT_LINE'"
		#CURRENT_LINE=""
		tail -n +2 $i | awk '{print $2"\t"$7}' > /tmp/TMP_DATA_DRIVEN_QUERY_TIMINGS
		 
	else
		echo "Operator Driven"
		#CURRENT_LINE="$SCALE_FACTOR	"$(tail -n 1 $i)
		tail -n +2 $i > /tmp/TMP_OPERATOR_DRIVEN_QUERY_TIMINGS 
	fi
	echo $i	
   done
#   read line
   echo "#	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" > query_timings.csv
   paste /tmp/TMP_OPERATOR_DRIVEN_QUERY_TIMINGS  /tmp/TMP_DATA_DRIVEN_QUERY_TIMINGS >> query_timings.csv
   rm -f /tmp/TMP_OPERATOR_DRIVEN_QUERY_TIMINGS  /tmp/TMP_DATA_DRIVEN_QUERY_TIMINGS

   if [ "$workload" == "ssb_all" ]; then
       XTICS="set xtics   (\"Q1.1\" 0.00000, \"Q1.2\" 1.00000, \"Q1.3\" 2.00000, \"Q2.1\" 3.00000, \"Q2.2\" 4.00000,  \"Q2.3\" 5.00000, \"Q3.1\" 6.00000, \"Q3.2\" 7.00000, \"Q3.3\" 8.00000, \"Q3.4\" 9.00000,  \"Q4.1\" 10.00000,  \"Q4.2\" 11.00000,  \"Q4.3\" 12.00000)"
   else
	XTICS="set xtics border in scale 0,0 nomirror rotate by -45  autojustify
set xtics   (\"Q1\" 0.00000,\"Q2\" 1.00000,\"Q3\" 2.00000,\"Q4\" 3.00000,\"Q5\" 4.00000,\"Q6\" 5.00000,\"Q7\" 6.00000,\"Q8\" 7.00000,\"Q9\" 8.00000,\"Q10\" 9.00000,\"Q11\" 10.00000,\"Q12\" 11.00000,\"Q13\" 12.00000,\"Q14\" 13.00000,\"Q15\" 14.00000,\"Q16\" 15.00000,\"Q17\" 16.00000,\"Q18\" 17.00000,\"Q19\" 18.00000,\"Q20\" 19.00000,\"Q21\" 20.00000,\"Q22\" 21.00000)"
   fi

echo "			
set title \"Query Performance $workload Scale Factor: $SCALE_FACTOR, Parallel Users: 1\"
set auto x
set auto y
set key top right Left reverse samplen 1
set key below
set key box
#set key vertical maxrows 3
set key width 2.1


#set xlabel 'Queries of Star Schema Benchmark'
set ylabel 'Execution Time (s)'
#set yrange [0:300000]
#set xrange [0:14]
set style data histogram
#set style histogram cluster gap 1
#set style histogram errorbars gap 1 lw 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"
#set style fill pattern 0 border

$XTICS

set style histogram cluster gap 1
plot 'query_timings.csv' using (\$6/1000) title \"greedy_heuristic_cpu_only\", \
'query_timings.csv' using (\$5/1000) title \"greedy_heuristic\", \
'query_timings.csv' using (\$4/1000) title \"greedy_chainer_heuristic\", \
'query_timings.csv' using (\$3/1000) title \"critical_path_heuristic\", \
'query_timings.csv' using (\$2/1000) title \"best_effort_gpu_heuristic\", \
'query_timings.csv' using (\$7/1000) title \"query_chopping\", \
'query_timings.csv' using (\$8/1000) title \"data driven\", \
'query_timings.csv' using (\$9/1000) title \"data-driven query_chopping\"

set output \"query_performance_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > query_performance_diagram.gnuplot
    gnuplot query_performance_diagram.gnuplot
	

   ###############################################################################################################################
   cd ../..



#        #echo $i | awk 'BEGIN{FS="/"} {print $4}' | awk 'BEGIN{FS="-"} {print $14}'
#	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $4}' | awk 'BEGIN{FS="-"} {print $14}'); 
#        echo "Data Driven: '$IS_DATA_DRIVEN_RUN'"
#	if $IS_DATA_DRIVEN_RUN; then
#		echo "Data Driven"
#		DATA_DRIVEN_COMPILE_TIME=$(tail -n 1 $i | awk '{print $2}')
#		DATA_DRIVEN_QUERY_CHOPPING=$(tail -n 1 $i | awk '{print $7}')		
#		CURRENT_LINE="$CURRENT_LINE	$DATA_DRIVEN_COMPILE_TIME $DATA_DRIVEN_QUERY_CHOPPING"
#		echo "$CURRENT_LINE" >> workload_time_collected_sf"$SCALE_FACTOR".csv
#                echo "CSV: '$CURRENT_LINE'"
#		CURRENT_LINE=""
#	else
#		echo "Operator Driven"
#		CURRENT_LINE="$SCALE_FACTOR	"$(tail -n 1 $i)
#	fi
#   done
done 

#exit 0
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
 for experiment in $MULTI_USER_WORKLOAD_DIRECTORIES; do
   echo "$experiment"
   echo $experiment | awk 'BEGIN{FS="-";} {print $2}'  
   NUM_USER=$(echo $experiment | awk 'BEGIN{FS="-";} {print $2}' | awk 'BEGIN{FS="_";} {print $2}')
   if [ ! -e workload_time_"$workload"_collected_varying_parallel_users.csv ]; then
       echo "parallel_users	dummy	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  workload_time_"$workload"_collected_varying_parallel_users.csv 
   fi
   CURRENT_LINE=""

   cd $experiment/varying_data_placement_strategy 
   for i in `find . -name workload_execution_time.csv | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		echo "Data Driven"
		DATA_DRIVEN_COMPILE_TIME=$(tail -n 1 $i | awk '{print $2}')
		DATA_DRIVEN_QUERY_CHOPPING=$(tail -n 1 $i | awk '{print $7}')		
		CURRENT_LINE="$CURRENT_LINE	$DATA_DRIVEN_COMPILE_TIME	$DATA_DRIVEN_QUERY_CHOPPING"
		echo "$CURRENT_LINE" >> ../../workload_time_"$workload"_collected_varying_parallel_users.csv
                #echo "CSV: '$CURRENT_LINE'"
		CURRENT_LINE=""
	else
		echo "Operator Driven"
		CURRENT_LINE="$NUM_USER	"$(tail -n 1 $i)
	fi
   done
   cd ../..

##############################################################################################################################
#number of executed and aborted GPU operators
##############################################################################################################################
   if [ ! -e aborted_gpu_operators_"$workload"_collected_varying_parallel_users.csv ]; then
       echo "parallel_users	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  aborted_gpu_operators_"$workload"_collected_varying_parallel_users.csv
  fi

   if [ ! -e executed_gpu_operators_"$workload"_collected_varying_parallel_users.csv ]; then
       echo "parallel_users	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  executed_gpu_operators_"$workload"_collected_varying_parallel_users.csv
  fi

   cd $experiment/varying_data_placement_strategy 
   for i in `find . -maxdepth 1 -mindepth 1 -type d | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		DATA_DRIVEN=$(head -n 1 "$i/best_effort_gpu_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		DATA_DRIVEN_CHOPPING=$(head -n 1 "$i/query_chopping/number_of_aborted_and_executed_gpu_operators.csv")
	else

		BEST_EFFORT_GPU=$(head -n 1 "$i/best_effort_gpu_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		CRITICAL_PATH=$(head -n 1 "$i/critical_path_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		GREEDY_CHAINER=$(head -n 1 "$i/greedy_chainer_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		GREEDY=$(head -n 1 "$i/greedy_heuristic/number_of_aborted_and_executed_gpu_operators.csv")
		CPU_ONLY=$(head -n 1 "$i/greedy_heuristic_cpu_only/number_of_aborted_and_executed_gpu_operators.csv")
		CHOPPING=$(head -n 1 "$i/query_chopping/number_of_aborted_and_executed_gpu_operators.csv")
		echo "Operator Driven"
#		read line
	fi
   done
	###############################################################################################################################
	BEST_EFFORT_GPU_ABORTED_OPERATORS=$(echo "$BEST_EFFORT_GPU" | awk '{print $2}')
	CRITICAL_PATH_ABORTED_OPERATORS=$(echo "$CRITICAL_PATH" | awk '{print $2}')
	GREEDY_CHAINER_ABORTED_OPERATORS=$(echo "$GREEDY_CHAINER" | awk '{print $2}')
	GREEDY_ABORTED_OPERATORS=$(echo "$GREEDY" | awk '{print $2}')
	CPU_ONLY_ABORTED_OPERATORS=$(echo "$CPU_ONLY" | awk '{print $2}')
	CHOPPING_ABORTED_OPERATORS=$(echo "$CHOPPING" | awk '{print $2}')
	DATA_DRIVEN_ABORTED_OPERATORS=$(echo "$DATA_DRIVEN" | awk '{print $2}')
	DATA_DRIVEN_CHOPPING_ABORTED_OPERATORS=$(echo "$DATA_DRIVEN_CHOPPING" | awk '{print $2}')

	CURRENT_LINE="$NUM_USER	$BEST_EFFORT_GPU_ABORTED_OPERATORS	$CRITICAL_PATH_ABORTED_OPERATORS	$GREEDY_CHAINER_ABORTED_OPERATORS	$GREEDY_ABORTED_OPERATORS	$CPU_ONLY_ABORTED_OPERATORS	$CHOPPING_ABORTED_OPERATORS	$DATA_DRIVEN_ABORTED_OPERATORS	$DATA_DRIVEN_CHOPPING_ABORTED_OPERATORS"
	echo "$CURRENT_LINE" >> ../../aborted_gpu_operators_"$workload"_collected_varying_parallel_users.csv
	###############################################################################################################################
	BEST_EFFORT_GPU_EXECUTED_OPERATORS=$(echo "$BEST_EFFORT_GPU" | awk '{print $3}')
	CRITICAL_PATH_EXECUTED_OPERATORS=$(echo "$CRITICAL_PATH" | awk '{print $3}')
	GREEDY_CHAINER_EXECUTED_OPERATORS=$(echo "$GREEDY_CHAINER" | awk '{print $3}')
	GREEDY_EXECUTED_OPERATORS=$(echo "$GREEDY" | awk '{print $3}')
	CPU_ONLY_EXECUTED_OPERATORS=$(echo "$CPU_ONLY" | awk '{print $3}')
	CHOPPING_EXECUTED_OPERATORS=$(echo "$CHOPPING" | awk '{print $3}')
	DATA_DRIVEN_EXECUTED_OPERATORS=$(echo "$DATA_DRIVEN" | awk '{print $3}')
	DATA_DRIVEN_CHOPPING_EXECUTED_OPERATORS=$(echo "$DATA_DRIVEN_CHOPPING" | awk '{print $3}')

	CURRENT_LINE="$NUM_USER	$BEST_EFFORT_GPU_EXECUTED_OPERATORS	$CRITICAL_PATH_EXECUTED_OPERATORS	$GREEDY_CHAINER_EXECUTED_OPERATORS	$GREEDY_EXECUTED_OPERATORS	$CPU_ONLY_EXECUTED_OPERATORS	$CHOPPING_EXECUTED_OPERATORS	$DATA_DRIVEN_EXECUTED_OPERATORS	$DATA_DRIVEN_CHOPPING_EXECUTED_OPERATORS"
	echo "$CURRENT_LINE" >> ../../executed_gpu_operators_"$workload"_collected_varying_parallel_users.csv
   ###############################################################################################################################
   cd ../..


##############################################################################################################################
# COPY TIME CPU TO GPU
############################################################################################################################## 

   if [ ! -e copy_times_cpu_to_gpu_"$workload"_collected_varying_parallel_users.csv ]; then
       echo "parallel_users	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  copy_times_cpu_to_gpu_"$workload"_collected_varying_parallel_users.csv
  fi  
   cd $experiment/varying_data_placement_strategy 
   for i in `find . -maxdepth 1 -mindepth 1 -type d | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		DATA_DRIVEN_COPY_TIME_CPU_GPU=$(head -n 1 "$i/best_effort_gpu_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		DATA_DRIVEN_CHOPPING_COPY_TIME_CPU_GPU=$(head -n 1 "$i/query_chopping/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
	else
		BEST_EFFORT_GPU_COPY_TIME_CPU_GPU=$(head -n 1 "$i/best_effort_gpu_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		CRITICAL_PATH_COPY_TIME_CPU_GPU=$(head -n 1 "$i/critical_path_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		GREEDY_CHAINER_COPY_TIME_CPU_GPU=$(head -n 1 "$i/greedy_chainer_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		GREEDY_COPY_TIME_CPU_GPU=$(head -n 1 "$i/greedy_heuristic/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		CPU_ONLY_COPY_TIME_CPU_GPU=$(head -n 1 "$i/greedy_heuristic_cpu_only/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		CHOPPING_COPY_TIME_CPU_GPU=$(head -n 1 "$i/query_chopping/data_transfer_times_over_pcie_bus.csv" | awk '{print $2}')
		
#		read line
	fi
   done

	CURRENT_LINE="$NUM_USER	$BEST_EFFORT_GPU_COPY_TIME_CPU_GPU	$CRITICAL_PATH_COPY_TIME_CPU_GPU	$GREEDY_CHAINER_COPY_TIME_CPU_GPU	$GREEDY_COPY_TIME_CPU_GPU	$CPU_ONLY_COPY_TIME_CPU_GPU	$CHOPPING_COPY_TIME_CPU_GPU	$DATA_DRIVEN_COPY_TIME_CPU_GPU	$DATA_DRIVEN_CHOPPING_COPY_TIME_CPU_GPU"
	echo "$CURRENT_LINE" >> ../../copy_times_cpu_to_gpu_"$workload"_collected_varying_parallel_users.csv

   ###############################################################################################################################
   cd ../..

##############################################################################################################################
# COPY TIME GPU to CPU
############################################################################################################################## 
   if [ ! -e copy_times_gpu_to_cpu_"$workload"_collected_varying_parallel_users.csv ]; then
       echo "parallel_users	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  copy_times_gpu_to_cpu_"$workload"_collected_varying_parallel_users.csv
  fi  
   cd $experiment/varying_data_placement_strategy 
   for i in `find . -maxdepth 1 -mindepth 1 -type d | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		DATA_DRIVEN_COPY_TIME_GPU_CPU=$(head -n 2 "$i/best_effort_gpu_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		DATA_DRIVEN_CHOPPING_COPY_TIME_GPU_CPU=$(head -n 2 "$i/query_chopping/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
	else
		BEST_EFFORT_GPU_COPY_TIME_GPU_CPU=$(head -n 2 "$i/best_effort_gpu_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		CRITICAL_PATH_COPY_TIME_GPU_CPU=$(head -n 2 "$i/critical_path_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		GREEDY_CHAINER_COPY_TIME_GPU_CPU=$(head -n 2 "$i/greedy_chainer_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		GREEDY_COPY_TIME_GPU_CPU=$(head -n 2 "$i/greedy_heuristic/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		CPU_ONLY_COPY_TIME_GPU_CPU=$(head -n 2 "$i/greedy_heuristic_cpu_only/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		CHOPPING_COPY_TIME_GPU_CPU=$(head -n 2 "$i/query_chopping/data_transfer_times_over_pcie_bus.csv" | tail -n 1 | awk '{print $2}')
		
#		read line
	fi
   done

	CURRENT_LINE="$NUM_USER	$BEST_EFFORT_GPU_COPY_TIME_GPU_CPU	$CRITICAL_PATH_COPY_TIME_GPU_CPU	$GREEDY_CHAINER_COPY_TIME_GPU_CPU	$GREEDY_COPY_TIME_GPU_CPU	$CPU_ONLY_COPY_TIME_GPU_CPU	$CHOPPING_COPY_TIME_GPU_CPU	$DATA_DRIVEN_COPY_TIME_GPU_CPU	$DATA_DRIVEN_CHOPPING_COPY_TIME_GPU_CPU"
	echo "$CURRENT_LINE" >> ../../copy_times_gpu_to_cpu_"$workload"_collected_varying_parallel_users.csv

   ###############################################################################################################################
   cd ../..


##############################################################################################################################
# WASTED TIME DUE TO ABORTED GPU OPERATORS
##############################################################################################################################


   if [ ! -e wasted_time_by_aborts_"$workload"_collected_varying_parallel_users.csv ]; then
       echo "parallel_users	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" >  wasted_time_by_aborts_"$workload"_collected_varying_parallel_users.csv
  fi  
   cd $experiment/varying_data_placement_strategy 
   for i in `find . -maxdepth 1 -mindepth 1 -type d | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		DATA_DRIVEN_WASTED_TIME_BY_ABORTS=$(cat "$i/best_effort_gpu_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		DATA_DRIVEN_CHOPPING_WASTED_TIME_BY_ABORTS=$(cat "$i/query_chopping/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
	else
		BEST_EFFORT_GPU_WASTED_TIME_BY_ABORTS=$(cat "$i/best_effort_gpu_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		CRITICAL_PATH_WASTED_TIME_BY_ABORTS=$(cat "$i/critical_path_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		GREEDY_CHAINER_WASTED_TIME_BY_ABORTS=$(cat "$i/greedy_chainer_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		GREEDY_WASTED_TIME_BY_ABORTS=$(cat "$i/greedy_heuristic/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		CPU_ONLY_WASTED_TIME_BY_ABORTS=$(cat "$i/greedy_heuristic_cpu_only/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		CHOPPING_WASTED_TIME_BY_ABORTS=$(cat "$i/query_chopping/WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS.csv"  | awk '{print $2}')
		
#		read line
	fi
   done

	CURRENT_LINE="$NUM_USER	$BEST_EFFORT_GPU_WASTED_TIME_BY_ABORTS	$CRITICAL_PATH_WASTED_TIME_BY_ABORTS	$GREEDY_CHAINER_WASTED_TIME_BY_ABORTS	$GREEDY_WASTED_TIME_BY_ABORTS	$CPU_ONLY_WASTED_TIME_BY_ABORTS	$CHOPPING_WASTED_TIME_BY_ABORTS	$DATA_DRIVEN_WASTED_TIME_BY_ABORTS	$DATA_DRIVEN_CHOPPING_WASTED_TIME_BY_ABORTS"
	echo "$CURRENT_LINE" >> ../../wasted_time_by_aborts_"$workload"_collected_varying_parallel_users.csv

   ###############################################################################################################################
   cd ../..

##############################################################################################################################
# Per Query Peformance Plot
##############################################################################################################################



   cd $experiment/varying_data_placement_strategy 
   for i in `find . -name experiment_result.csv  | sort -V`; do 
	IS_DATA_DRIVEN_RUN=$(echo $i | awk 'BEGIN{FS="/"} {print $2}' | awk 'BEGIN{FS="-"} {print $14}'); 
	if $IS_DATA_DRIVEN_RUN; then
		#echo "Data Driven"
		#DATA_DRIVEN_COMPILE_TIME=$(tail -n 1 $i | awk '{print $2}')
		#DATA_DRIVEN_QUERY_CHOPPING=$(tail -n 1 $i | awk '{print $7}')		
		#CURRENT_LINE="$CURRENT_LINE	$DATA_DRIVEN_COMPILE_TIME	$DATA_DRIVEN_QUERY_CHOPPING"
		#echo "$CURRENT_LINE" >> ../../query_times_"$workload"_collected_varying_scale_factor.csv
                #echo "CSV: '$CURRENT_LINE'"
		#CURRENT_LINE=""
		tail -n +2 $i | awk '{print $2"\t"$7}' > /tmp/TMP_DATA_DRIVEN_QUERY_TIMINGS
		 
	else
		echo "Operator Driven"
		#CURRENT_LINE="$SCALE_FACTOR	"$(tail -n 1 $i)
		tail -n +2 $i > /tmp/TMP_OPERATOR_DRIVEN_QUERY_TIMINGS 
	fi
	echo $i	
   done
#   read line
   echo "#	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" > query_timings.csv
   paste /tmp/TMP_OPERATOR_DRIVEN_QUERY_TIMINGS  /tmp/TMP_DATA_DRIVEN_QUERY_TIMINGS >> query_timings.csv
   rm -f /tmp/TMP_OPERATOR_DRIVEN_QUERY_TIMINGS  /tmp/TMP_DATA_DRIVEN_QUERY_TIMINGS

   if [ "$workload" == "ssb_all" ]; then
       XTICS="set xtics   (\"Q1.1\" 0.00000, \"Q1.2\" 1.00000, \"Q1.3\" 2.00000, \"Q2.1\" 3.00000, \"Q2.2\" 4.00000,  \"Q2.3\" 5.00000, \"Q3.1\" 6.00000, \"Q3.2\" 7.00000, \"Q3.3\" 8.00000, \"Q3.4\" 9.00000,  \"Q4.1\" 10.00000,  \"Q4.2\" 11.00000,  \"Q4.3\" 12.00000)"
   else
	XTICS="set xtics border in scale 0,0 nomirror rotate by -45  autojustify
set xtics   (\"Q1\" 0.00000,\"Q2\" 1.00000,\"Q3\" 2.00000,\"Q4\" 3.00000,\"Q5\" 4.00000,\"Q6\" 5.00000,\"Q7\" 6.00000,\"Q8\" 7.00000,\"Q9\" 8.00000,\"Q10\" 9.00000,\"Q11\" 10.00000,\"Q12\" 11.00000,\"Q13\" 12.00000,\"Q14\" 13.00000,\"Q15\" 14.00000,\"Q16\" 15.00000,\"Q17\" 16.00000,\"Q18\" 17.00000,\"Q19\" 18.00000,\"Q20\" 19.00000,\"Q21\" 20.00000,\"Q22\" 21.00000)"
   fi

echo "			
set title \"Query Performance $workload, Parallel Users: $NUM_USER\"
set auto x
set auto y
set key top right Left reverse samplen 1
set key below
set key box
#set key vertical maxrows 3
set key width 2.1


#set xlabel 'Queries of Star Schema Benchmark'
set ylabel 'Execution Time (s)'
#set yrange [0:300000]
#set xrange [0:14]
set style data histogram
#set style histogram cluster gap 1
#set style histogram errorbars gap 1 lw 1
set style fill solid border -1
set boxwidth 0.9
set datafile separator \"\t\"
#set style fill pattern 0 border

$XTICS

set style histogram cluster gap 1
plot 'query_timings.csv' using (\$6/1000) title \"greedy_heuristic_cpu_only\", \
'query_timings.csv' using (\$5/1000) title \"greedy_heuristic\", \
'query_timings.csv' using (\$4/1000) title \"greedy_chainer_heuristic\", \
'query_timings.csv' using (\$3/1000) title \"critical_path_heuristic\", \
'query_timings.csv' using (\$2/1000) title \"best_effort_gpu_heuristic\", \
'query_timings.csv' using (\$7/1000) title \"query_chopping\", \
'query_timings.csv' using (\$8/1000) title \"data driven\", \
'query_timings.csv' using (\$9/1000) title \"data-driven query_chopping\"

set output \"query_performance_diagram.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > query_performance_diagram.gnuplot
    gnuplot query_performance_diagram.gnuplot
	

   ###############################################################################################################################
   cd ../..

 done 


#echo "			
#set title \"Workload $workload Execution Time $HOSTNAME\"
#set auto x
#set auto y

#set key top right Left reverse samplen 1
#set key box
##set key vertical maxrows 3
##set key width 2.1

#set ylabel 'Execution Time (s)'
#set xlabel 'Scale Factor'
#set yrange [0:]
##set style data histogram
##set style histogram cluster gap 1
#set style fill solid border -1
##set boxwidth 0.9
#set datafile separator \"\t\"

##set xtics(\"\" 0.00000)

#set style data points	

#plot 'workload_time_"$workload"_collected_varying_scale_factor.csv' using 1:(\$7/1000) title \"greedy_heuristic_cpu_only\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using 1:(\$6/1000) title \"greedy_heuristic\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using 1:(\$5/1000) title \"greedy_chainer_heuristic\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using 1:(\$4/1000) title \"critical_path_heuristic\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using 1:(\$3/1000) title \"best_effort_gpu_heuristic\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using 1:(\$7/1000) title \"query_chopping\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using 1:(\$8/1000) title \"data-driven\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using 1:(\$9/1000) title \"data-driven query_chopping\"
#set output \"workload_"$workload"_performance_diagram.pdf\"
#set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
#replot" > workload_performance_diagram_"$workload"_varying_scale_factor.gnuplot
#gnuplot workload_performance_diagram_"$workload"_varying_scale_factor.gnuplot

for experiment_type in scale_factor parallel_users; do

if [ "$experiment_type" == "scale_factor" ]; then
	XLABEL="Scale Factor"
else 
	XLABEL="Number of Parallel Users"
fi

echo "			
set title \"Workload $workload Execution Time $HOSTNAME\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key below
#set key vertical maxrows 3
set key width 2.1

set ylabel 'Execution Time (s)'
set xlabel '$XLABEL'
set yrange [0:]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
#set boxwidth 0.9
set datafile separator \"\t\"

#set xtics(\"\" 0.00000)

#set style data points	

plot 'workload_time_"$workload"_collected_varying_$experiment_type.csv' using (\$7/1000):xtic(1) title \"greedy_heuristic_cpu_only\", \
'workload_time_"$workload"_collected_varying_$experiment_type.csv' using (\$6/1000):xtic(1) title \"greedy_heuristic\", \
'workload_time_"$workload"_collected_varying_$experiment_type.csv' using (\$5/1000):xtic(1) title \"greedy_chainer_heuristic\", \
'workload_time_"$workload"_collected_varying_$experiment_type.csv' using (\$4/1000):xtic(1) title \"critical_path_heuristic\", \
'workload_time_"$workload"_collected_varying_$experiment_type.csv' using (\$3/1000):xtic(1) title \"best_effort_gpu_heuristic\", \
'workload_time_"$workload"_collected_varying_$experiment_type.csv' using (\$8/1000):xtic(1) title \"query_chopping\", \
'workload_time_"$workload"_collected_varying_$experiment_type.csv' using (\$9/1000):xtic(1) title \"data-driven\", \
'workload_time_"$workload"_collected_varying_$experiment_type.csv' using (\$10/1000):xtic(1) title \"data-driven query_chopping\"
set output \"workload_"$workload"_performance_diagram_varying_$experiment_type.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > workload_performance_diagram_"$workload"_varying_$experiment_type.gnuplot
gnuplot workload_performance_diagram_"$workload"_varying_$experiment_type.gnuplot


echo "			
set title \"Workload $workload Execution Time $HOSTNAME\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key below
#set key vertical maxrows 3
set key width 2.1

set ylabel '#Aborted GPU Operators'
set xlabel '$XLABEL'
set yrange [0:]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
#set boxwidth 0.9
set datafile separator \"\t\"

#set xtics(\"\" 0.00000)

#set style data points	

plot 'aborted_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$6):xtic(1) title \"greedy_heuristic_cpu_only\", \
'aborted_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$5):xtic(1) title \"greedy_heuristic\", \
'aborted_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$4):xtic(1) title \"greedy_chainer_heuristic\", \
'aborted_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$3):xtic(1) title \"critical_path_heuristic\", \
'aborted_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$2):xtic(1) title \"best_effort_gpu_heuristic\", \
'aborted_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$7):xtic(1) title \"query_chopping\", \
'aborted_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$8):xtic(1) title \"data-driven\", \
'aborted_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$9):xtic(1) title \"data-driven query_chopping\"
set output \"aborted_gpu_operators_"$workload"_performance_diagram_varying_$experiment_type.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > aborted_gpu_operators_diagram_"$workload"_varying_$experiment_type.gnuplot
gnuplot aborted_gpu_operators_diagram_"$workload"_varying_$experiment_type.gnuplot



echo "			
set title \"Workload $workload Execution Time $HOSTNAME\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key below
#set key vertical maxrows 3
set key width 2.1

set ylabel '#Aborted GPU Operators'
set xlabel '$XLABEL'
set yrange [0:]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
#set boxwidth 0.9
set datafile separator \"\t\"

#set xtics(\"\" 0.00000)

#set style data points	

plot 'executed_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$6):xtic(1) title \"greedy_heuristic_cpu_only\", \
'executed_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$5):xtic(1) title \"greedy_heuristic\", \
'executed_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$4):xtic(1) title \"greedy_chainer_heuristic\", \
'executed_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$3):xtic(1) title \"critical_path_heuristic\", \
'executed_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$2):xtic(1) title \"best_effort_gpu_heuristic\", \
'executed_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$7):xtic(1) title \"query_chopping\", \
'executed_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$8):xtic(1) title \"data-driven\", \
'executed_gpu_operators_"$workload"_collected_varying_$experiment_type.csv' using (\$9):xtic(1) title \"data-driven query_chopping\"
set output \"executed_gpu_operators_"$workload"_performance_diagram_varying_$experiment_type.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > executed_gpu_operators_diagram_"$workload"_varying_$experiment_type.gnuplot
gnuplot executed_gpu_operators_diagram_"$workload"_varying_$experiment_type.gnuplot




echo "			
set title \"Copy Time CPU to GPU $workload $HOSTNAME\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key below
#set key vertical maxrows 3
set key width 2.1

set ylabel 'Copy Time in ns'
set xlabel '$XLABEL'
set yrange [0:]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
#set boxwidth 0.9
set datafile separator \"\t\"

#set xtics(\"\" 0.00000)

#set style data points	

plot 'copy_times_cpu_to_gpu_"$workload"_collected_varying_$experiment_type.csv' using (\$6):xtic(1) title \"greedy_heuristic_cpu_only\", \
'copy_times_cpu_to_gpu_"$workload"_collected_varying_$experiment_type.csv' using (\$5):xtic(1) title \"greedy_heuristic\", \
'copy_times_cpu_to_gpu_"$workload"_collected_varying_$experiment_type.csv' using (\$4):xtic(1) title \"greedy_chainer_heuristic\", \
'copy_times_cpu_to_gpu_"$workload"_collected_varying_$experiment_type.csv' using (\$3):xtic(1) title \"critical_path_heuristic\", \
'copy_times_cpu_to_gpu_"$workload"_collected_varying_$experiment_type.csv' using (\$2):xtic(1) title \"best_effort_gpu_heuristic\", \
'copy_times_cpu_to_gpu_"$workload"_collected_varying_$experiment_type.csv' using (\$7):xtic(1) title \"query_chopping\", \
'copy_times_cpu_to_gpu_"$workload"_collected_varying_$experiment_type.csv' using (\$8):xtic(1) title \"data-driven\", \
'copy_times_cpu_to_gpu_"$workload"_collected_varying_$experiment_type.csv' using (\$9):xtic(1) title \"data-driven query_chopping\"
set output \"copy_times_cpu_to_gpu_"$workload"_performance_diagram_varying_$experiment_type.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > copy_times_cpu_to_gpu_diagram_"$workload"_varying_$experiment_type.gnuplot
gnuplot copy_times_cpu_to_gpu_diagram_"$workload"_varying_$experiment_type.gnuplot

echo "			
set title \"Copy Time GPU to CPU $workload $HOSTNAME\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key below
#set key vertical maxrows 3
set key width 2.1

set ylabel 'Copy Time in ns'
set xlabel '$XLABEL'
set yrange [0:]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
#set boxwidth 0.9
set datafile separator \"\t\"

#set xtics(\"\" 0.00000)

#set style data points	

plot 'copy_times_gpu_to_cpu_"$workload"_collected_varying_$experiment_type.csv' using (\$6):xtic(1) title \"greedy_heuristic_cpu_only\", \
'copy_times_gpu_to_cpu_"$workload"_collected_varying_$experiment_type.csv' using (\$5):xtic(1) title \"greedy_heuristic\", \
'copy_times_gpu_to_cpu_"$workload"_collected_varying_$experiment_type.csv' using (\$4):xtic(1) title \"greedy_chainer_heuristic\", \
'copy_times_gpu_to_cpu_"$workload"_collected_varying_$experiment_type.csv' using (\$3):xtic(1) title \"critical_path_heuristic\", \
'copy_times_gpu_to_cpu_"$workload"_collected_varying_$experiment_type.csv' using (\$2):xtic(1) title \"best_effort_gpu_heuristic\", \
'copy_times_gpu_to_cpu_"$workload"_collected_varying_$experiment_type.csv' using (\$7):xtic(1) title \"query_chopping\", \
'copy_times_gpu_to_cpu_"$workload"_collected_varying_$experiment_type.csv' using (\$8):xtic(1) title \"data-driven\", \
'copy_times_gpu_to_cpu_"$workload"_collected_varying_$experiment_type.csv' using (\$9):xtic(1) title \"data-driven query_chopping\"
set output \"copy_times_gpu_to_cpu_"$workload"_performance_diagram_varying_$experiment_type.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > copy_times_gpu_to_cpu_diagram_"$workload"_varying_$experiment_type.gnuplot
gnuplot copy_times_gpu_to_cpu_diagram_"$workload"_varying_$experiment_type.gnuplot

echo "			
set title \"Wasted Time by Aborts $workload $HOSTNAME\"
set auto x
set auto y

set key top right Left reverse samplen 1
set key box
set key below
#set key vertical maxrows 3
set key width 2.1

set ylabel 'Wasted Time in s'
set xlabel '$XLABEL'
set yrange [0:]
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
#set boxwidth 0.9
set datafile separator \"\t\"

#set xtics(\"\" 0.00000)

#set style data points	

plot 'wasted_time_by_aborts_"$workload"_collected_varying_$experiment_type.csv' using (\$6/1000000000):xtic(1) title \"greedy_heuristic_cpu_only\", \
'wasted_time_by_aborts_"$workload"_collected_varying_$experiment_type.csv' using (\$5/1000000000):xtic(1) title \"greedy_heuristic\", \
'wasted_time_by_aborts_"$workload"_collected_varying_$experiment_type.csv' using (\$4/1000000000):xtic(1) title \"greedy_chainer_heuristic\", \
'wasted_time_by_aborts_"$workload"_collected_varying_$experiment_type.csv' using (\$3/1000000000):xtic(1) title \"critical_path_heuristic\", \
'wasted_time_by_aborts_"$workload"_collected_varying_$experiment_type.csv' using (\$2/1000000000):xtic(1) title \"best_effort_gpu_heuristic\", \
'wasted_time_by_aborts_"$workload"_collected_varying_$experiment_type.csv' using (\$7/1000000000):xtic(1) title \"query_chopping\", \
'wasted_time_by_aborts_"$workload"_collected_varying_$experiment_type.csv' using (\$8/1000000000):xtic(1) title \"data-driven\", \
'wasted_time_by_aborts_"$workload"_collected_varying_$experiment_type.csv' using (\$9/1000000000):xtic(1) title \"data-driven query_chopping\"
set output \"wasted_time_by_aborts_"$workload"_performance_diagram_varying_$experiment_type.pdf\"
set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
replot" > wasted_time_by_aborts_diagram_"$workload"_varying_$experiment_type.gnuplot
gnuplot wasted_time_by_aborts_diagram_"$workload"_varying_$experiment_type.gnuplot

done

#echo "			
#set title \"Workload $workload Execution Time $HOSTNAME\"
#set auto x
#set auto y

#set key top right Left reverse samplen 1
#set key box
#set key below
##set key vertical maxrows 3
#set key width 2.1

#set ylabel 'Execution Time (s)'
#set xlabel 'Scale Factor'
#set yrange [0:]
#set style data histogram
#set style histogram cluster gap 1
#set style fill solid border -1
##set boxwidth 0.9
#set datafile separator \"\t\"

##set xtics(\"\" 0.00000)

##set style data points	

#plot 'workload_time_"$workload"_collected_varying_scale_factor.csv' using (\$7/1000):xtic(1) title \"greedy_heuristic_cpu_only\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using (\$6/1000):xtic(1) title \"greedy_heuristic\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using (\$5/1000):xtic(1) title \"greedy_chainer_heuristic\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using (\$4/1000):xtic(1) title \"critical_path_heuristic\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using (\$3/1000):xtic(1) title \"best_effort_gpu_heuristic\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using (\$8/1000):xtic(1) title \"query_chopping\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using (\$9/1000):xtic(1) title \"data-driven\", \
#'workload_time_"$workload"_collected_varying_scale_factor.csv' using (\$10/1000):xtic(1) title \"data-driven query_chopping\"
#set output \"workload_"$workload"_performance_diagram_varying_scale_factor.pdf\"
#set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
#replot" > workload_performance_diagram_"$workload"_varying_scale_factor.gnuplot
#gnuplot workload_performance_diagram_"$workload"_varying_scale_factor.gnuplot


#echo "			
#set title \"Workload $workload Execution Time $HOSTNAME\"
#set auto x
#set auto y

#set key top right Left reverse samplen 1
#set key box
#set key below
##set key vertical maxrows 3
#set key width 2.1

#set ylabel 'Execution Time (s)'
#set xlabel 'Number of Parallel Users'
#set yrange [0:]
#set style data histogram
#set style histogram cluster gap 1
#set style fill solid border -1
##set boxwidth 0.9
#set datafile separator \"\t\"

##set xtics(\"\" 0.00000)

##set style data points	

#plot 'workload_time_"$workload"_collected_varying_parallel_users.csv' using (\$7/1000):xtic(1) title \"greedy_heuristic_cpu_only\", \
#'workload_time_"$workload"_collected_varying_parallel_users.csv' using (\$6/1000):xtic(1) title \"greedy_heuristic\", \
#'workload_time_"$workload"_collected_varying_parallel_users.csv' using (\$5/1000):xtic(1) title \"greedy_chainer_heuristic\", \
#'workload_time_"$workload"_collected_varying_parallel_users.csv' using (\$4/1000):xtic(1) title \"critical_path_heuristic\", \
#'workload_time_"$workload"_collected_varying_parallel_users.csv' using (\$3/1000):xtic(1) title \"best_effort_gpu_heuristic\", \
#'workload_time_"$workload"_collected_varying_parallel_users.csv' using (\$8/1000):xtic(1) title \"query_chopping\", \
#'workload_time_"$workload"_collected_varying_parallel_users.csv' using (\$9/1000):xtic(1) title \"data-driven\", \
#'workload_time_"$workload"_collected_varying_parallel_users.csv' using (\$10/1000):xtic(1) title \"data-driven query_chopping\"
#set output \"workload_"$workload"_performance_diagram_varying_number_users.pdf\"
#set terminal pdfcairo font \"Arial-Bold,16\" size 6.1, 3
#replot" > workload_performance_diagram_"$workload"_varying_parallel_users.gnuplot
#gnuplot workload_performance_diagram_"$workload"_varying_parallel_users.gnuplot

 #echo $WORKLOAD_DIRECTORIES | grep single_user

 #echo "$SINGLE_USER_WORKLOAD_DIRECTORIES"

# for experiment in $WORKLOAD_DIRECTORIES;do
    
   

# done

# echo "scale_factor	dummy	best_effort_gpu_heuristic	critical_path_heuristic	greedy_chainer_heuristic	greedy_heuristic	greedy_heuristic_cpu_only	query_chopping	data_driven	data_driven_query_chopping" > workload_time_collected.csv

#CURRENT_LINE=""
#for i in `find . -name workload_execution_time.csv | grep $workload`; do 
#

done


exit 

exit 0

