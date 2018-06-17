#!/bin/bash

if [ $# -lt 2 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 <NUMBER_OF_CPUS> <NUMBER_OF_GPUS>"
	exit -1
fi

if [ $# -gt 2 ]; then
	echo 'To many parameters!'
	echo "Usage: $0 <NUMBER_OF_CPUS> <NUMBER_OF_GPUS>"
	exit -1
fi

rm -f hardware_specification.conf
touch hardware_specification.conf
echo "number_of_cpus=$1" >> hardware_specification.conf
echo "number_of_dedicated_gpus=$2" >> hardware_specification.conf


exit 0
