#!/bin/bash
set -x

if [ $# -lt 3 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 <default_optimization_criterion> <reuse_performance_models> <track_memory_usage>"
	exit -1
fi

if [ $# -gt 3 ]; then
	echo 'To many parameters!'
        echo "Usage: $0 <default_optimization_criterion> <reuse_performance_models> <track_memory_usage>"
	exit -1
fi


#sets the optimization criterion used
#possible values are:
#ResponseTime: 0
#WaitingTimeAwareResponseTime: 1 (default)
#Throughput: 2
#Simple_Round_Robin: 3
#ProbabilityBasedOutsourcing: 4
#Throughput2: 5
#default_optimization_criterion=1
#reuse_performance_models=1
#track_memory_usage=1


rm -f hype.conf
touch hype.conf

echo "default_optimization_criterion=$1" >> hype.conf
echo "reuse_performance_models=$2" >> hype.conf
echo "track_memory_usage=$3" >> hype.conf

#we choose a short training phase and 
#a short recomputation period
echo "recomputation_period=20" >> hype.conf
echo "length_of_trainingphase=10" >> hype.conf


exit 0
