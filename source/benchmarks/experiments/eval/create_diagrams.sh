#!/bin/bash

if [ $# -lt 1 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 [AGGREGATION|SELECTION|SORT]"
	exit -1
fi

#EVALUATION_MACHINE=$HOSTNAME
EVALUATION_MACHINE=sebastian-ESPRIMO-P700
#EVALUATION_MACHINE=sebastian-XPS-8700

#DATAFILE_PER_USECASE="averaged_cpu_only.data averaged_gpu_only.data averaged_response_time.data averaged_simple_round_robin.data averaged_throughput2.data averaged_throughput.data averaged_waiting_time_aware_response_time.data"

DATAFILE_PER_USECASE="cpu_only.data gpu_only.data response_time.data simple_round_robin.data throughput2.data throughput.data waiting_time_aware_response_time.data"

OPERATION_NAME=$1

if [[ "$OPERATION_NAME" != "SORT" && "$OPERATION_NAME" != "SELECTION" && "$OPERATION_NAME" != "AGGREGATION" && "$OPERATION_NAME" != "JOIN" ]]; then
	echo "First parameter has to be a valid Operation: [AGGREGATION|SELECTION|SORT]"
	echo "Your Input: $OPERATION_NAME"
	echo "Aborting..."
	exit -1
fi

./eval.sh "Results/$OPERATION_NAME/$EVALUATION_MACHINE-varying_dataset_size_benchmark_results.data" $OPERATION_NAME-varying_dataset_size
./eval.sh "Results/$OPERATION_NAME/$EVALUATION_MACHINE-varying_number_of_operations_benchmark_results.data" $OPERATION_NAME-varying_number_of_operations
./eval.sh "Results/$OPERATION_NAME/$EVALUATION_MACHINE-varying_number_of_datasets_benchmark_results.data" $OPERATION_NAME-varying_number_of_datasets
./eval.sh "Results/$OPERATION_NAME/$EVALUATION_MACHINE-varying_operator_queue_length_benchmark_results.data" $OPERATION_NAME-varying_operator_queue_length



#analyse data of use case over all experiments
mkdir -p $OPERATION_NAME-summary
#cp $OPERATION_NAME-*/
for data_file in $DATAFILE_PER_USECASE; do
	rm -f $OPERATION_NAME-summary/$data_file
	touch $OPERATION_NAME-summary/$data_file
	for experiment in "$OPERATION_NAME-varying_*"; do

		echo $experiment/$data_file

		cat $experiment/$data_file >> $OPERATION_NAME-summary/$data_file

	done

done

#SPEEDUP Diagram
mkdir $OPERATION_NAME-summary/speedups
cp $OPERATION_NAME-summary/*.data $OPERATION_NAME-summary/speedups/
cat script_templates/speedups_summary.plt | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > $OPERATION_NAME-summary/speedups/speedups_summary.plt
cp script_templates/merge_result_files_for_speedup_computation.sh $OPERATION_NAME-summary/speedups/
bash -c "cd $OPERATION_NAME-summary/speedups/; bash merge_result_files_for_speedup_computation.sh; gnuplot speedups_summary.plt"



SCRIPTS_FOR_USECASE_SUMMARY="device_utilization.plt average_estimation_errors_summary_per_usecase.plt relative_training_lengths.plt"

cd script_templates
for script in $SCRIPTS_FOR_USECASE_SUMMARY; do
	echo "$script"
	cat "$script" | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > "../$OPERATION_NAME-summary/$script"
done
cd ..

#exit 0

#DIR=$(pwd)
#for i in $OPERATION_NAME-varying_dataset_size $OPERATION_NAME-varying_number_of_operations $OPERATION_NAME-varying_number_of_datasets $OPERATION_NAME-varying_operator_queue_length; do
#    cd script_templates
#    for j in *.plt; do
#	cat $j | sed 's/%EXPERIMENT_NAME%/'$i'/g' > ../$i/$j 
#    done
#
#    cd ..
#done

DIR=$(pwd)
for i in $OPERATION_NAME-varying_dataset_size $OPERATION_NAME-varying_number_of_operations $OPERATION_NAME-varying_number_of_datasets $OPERATION_NAME-varying_operator_queue_length; do
    cd script_templates
 #   for j in *.plt; do
#	 
  #  done

    #cat average_relative_estimation_errors.plt | sed 's/%EXPERIMENT_NAME%/'$i'/g' > ../$i/average_relative_estimation_errors.plt
    #cat device_utilization.plt | sed 's/%EXPERIMENT_NAME%/'$i'/g' > ../$i/device_utilization.plt

    
if [ "$i" = "$OPERATION_NAME-varying_dataset_size" ]; then
    cat execution_times_varying_dataset_size.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/execution_times_varying_dataset_size.plt
    cat device_utilization_varying_dataset_size.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/device_utilization_varying_dataset_size.plt
    cat average_relative_estimation_errors_varying_dataset_size.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/average_relative_estimation_errors_varying_dataset_size.plt
elif [ "$i" = "$OPERATION_NAME-varying_number_of_operations" ]; then
    cat execution_times_varying_number_of_operations.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/execution_times_varying_number_of_operations.plt
    cat device_utilization_varying_number_of_operations.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/device_utilization_varying_number_of_operations.plt
    cat average_relative_estimation_errors_varying_number_of_operations.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/average_relative_estimation_errors_varying_number_of_operations.plt	
elif [ "$i" = "$OPERATION_NAME-varying_number_of_datasets" ]; then
    cat execution_times_varying_number_of_datasets.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/execution_times_varying_number_of_datasets.plt
    cat device_utilization_varying_number_of_datasets.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/device_utilization_varying_number_of_datasets.plt
    cat average_relative_estimation_errors_varying_number_of_datasets.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/average_relative_estimation_errors_varying_number_of_datasets.plt
	
elif [ "$i" = "$OPERATION_NAME-varying_operator_queue_length" ]; then
    cat execution_times_varying_operator_queue_length.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/execution_times_varying_operator_queue_length.plt
    cat device_utilization_varying_operator_queue_length.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/device_utilization_varying_operator_queue_length.plt
    cat average_relative_estimation_errors_varying_operator_queue_length.plt  | sed 's/%OPERATION_NAME%/'"$OPERATION_NAME"'/g' > ../$i/average_relative_estimation_errors_varying_operator_queue_length.plt
	
#elif [ "$1" = "varying_training_length" ]; then
fi


    cd ..
done



for i in $OPERATION_NAME-varying_dataset_size $OPERATION_NAME-varying_number_of_operations $OPERATION_NAME-varying_number_of_datasets $OPERATION_NAME-varying_operator_queue_length $OPERATION_NAME-summary; do
	cd $i
	echo "Generating Diagrams for Experiment: $i"
	for j in *.plt; do
		echo "gnuplot $j"
		gnuplot "$j"
	done
	#gnuplot *.plt
	cd ..
done

exit 0
