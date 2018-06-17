
#for run_type in "cpu" "any"; do
for run_type in "any"; do

	rm -f experimental_result_data_query_chopping/measurements_$run_type.csv
        touch experimental_result_data_query_chopping/measurements_$run_type.csv

	echo -e "ID\tscale factor\tquery name\tnumber of users\tnumber of queries\tnumber of CPUs\tnumber of GPUs\tLOAD_BALANCING_STRATEGY\tOPTIMIZER\tUSE_MEMCOST_MODEL\tNUMBER_OF_EXECUTED_GPU_OPERATORS\tNUMBER_OF_ABORTED_GPU_OPERATORS\tWORKLOAD_EXECUTION_TIME-(ms)\tTOTAL_CPU_TIME-(ms)\tWASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS-(ns)\tMINIMAL QUERY EXECUTION TIME-(ms)\tMAXIMAL QUERY EXECUTION TIME-(ms)\tAVERAGE QUERY EXECUTION TIME-(ms)\tVARIANCE IN QUERY EXECUTION TIME" >> experimental_result_data_query_chopping/measurements_$run_type.csv
	cd experimental_result_data_query_chopping
	COUNTER=0
	for i in *_$run_type*.log; do

                echo "Processing File: $i"
		SCALE_FACTOR=`echo -n "$i" | awk 'BEGIN{FS="_";}{print $2}'`
		QUERY_NAME=`echo -n "$i" | awk 'BEGIN{FS="_";}{print $4}'` 
		NUMBER_OF_USERS=`echo -n "$i" | awk 'BEGIN{FS="_";}{print $6}'`
		NUMBER_OF_QUERIES=`echo -n "$i" | awk 'BEGIN{FS="_";}{print $9}'`
		NUMBER_OF_CPUS=`echo -n "$i" | awk 'BEGIN{FS="_";}{print $16}'`
		NUMBER_OF_GPUS=`echo -n "$i" | awk 'BEGIN{FS="_";}{print $19}' | sed -e 's/\.log//g'`	
		#sf_10_query_42_users_10_num_queries_100_run_8_runtype_any_num_cpus_8_num_gpus_7.log	
		 
		NUMBER_OF_EXECUTED_GPU_OPERATORS=`cat "$i" | grep "NUMBER_OF_EXECUTED_GPU_OPERATORS" | awk '{print $2}'` 
		if [ -z $NUMBER_OF_EXECUTED_GPU_OPERATORS ];then
		    NUMBER_OF_EXECUTED_GPU_OPERATORS=0
		fi
		NUMBER_OF_ABORTED_GPU_OPERATORS=`cat "$i" | grep "NUMBER_OF_ABORTED_GPU_OPERATORS" | awk '{print $2}'`
		if [ -z $NUMBER_OF_ABORTED_GPU_OPERATORS ];then
		    NUMBER_OF_ABORTED_GPU_OPERATORS=0
		fi
		
		WORKLOAD_EXECUTION_TIME=`cat "$i" | grep "WORKLOAD EXECUTION TIME" | awk '{print $4}' | sed -e 's/ms//g' | sed -e 's/,.*//g'` 
		TOTAL_CPU_TIME=`cat "$i" | grep "TOTAL CPU TIME" | awk '{print $4}' | sed -e 's/ms//g' | sed -e 's/,.*//g'` 
                LOAD_BALANCING_STRATEGY=`cat "$i" | grep "LOAD_BALANCING_STRATEGY" | awk '{print $2}'`
                USE_MEMORY_COST_MODELS=`cat "$i" | grep "USE_MEMORY_COST_MODELS" | awk '{print $2}'`
                QUERY_OPTIMIZER_MODE=`cat "$i" | grep "QUERY_OPTIMIZER_MODE" | awk '{print $2}'`
		WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS=`cat "$i" | grep "TOTAL_LOST_TIME_IN_NS_DUE_TO_ABORTED_GPU_OPERATORS" | awk '{print $2}' | sed -e 's/ms//g' | sed -e 's/,.*//g'`

		if [ -z $WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS ];then
		    WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS=0
		fi

		QUERY_RUNTIMES=`basename "$i" ".log"`_cogascript.timings
		
		#compute minimal, maximal, and average query execution time in one run as well as the variance in execution time
		#cat "$i" | grep "Execution Time" | awk '{print $3}' | sed -e 's/\./,/g' -e 's/,.*//g' > tmp
		cat "$QUERY_RUNTIMES" | awk '{print $2}' | sed -e 's/\./,/g' -e 's/,.*//g' -e 's/ms//g' > tmp
		QUERY_EXECUTION_STATISTICS=`cat tmp | awk -f ../compute_average.awk`
		rm tmp
		
		echo -e "$COUNTER\t$SCALE_FACTOR\t$QUERY_NAME\t$NUMBER_OF_USERS\t$NUMBER_OF_QUERIES\t$NUMBER_OF_CPUS\t$NUMBER_OF_GPUS\t$LOAD_BALANCING_STRATEGY\t$QUERY_OPTIMIZER_MODE\t$USE_MEMORY_COST_MODELS\t$NUMBER_OF_EXECUTED_GPU_OPERATORS\t$NUMBER_OF_ABORTED_GPU_OPERATORS\t$WORKLOAD_EXECUTION_TIME\t$TOTAL_CPU_TIME\t$WASTED_TIME_DUE_TO_ABORTED_GPU_OPERATORS\t$QUERY_EXECUTION_STATISTICS" >> measurements_$run_type.csv
		let COUNTER++
	done
	cd ..

done

exit 0
