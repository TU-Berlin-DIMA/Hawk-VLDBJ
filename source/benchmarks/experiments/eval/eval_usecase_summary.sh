#!/bin/bash

#if any program instance fails, the entire script failes and returns with an error code
set -e

DATAFILE_CONTAINING_RESULTS=""

if [ $# -lt 2 ]; then
	echo "Missing parameter!"
	echo "Usage: $0 <inputfile> <output directory>"
	exit -1
fi

FUNCTION_RETURN_VALUE=0

function getMinimalValue {

	FILENAME=$1
	COLUMN_NUMBER=$2
	#cat $FILENAME | awk 'BEGIN{FS="\t";}{print $'$COLUMN_NUMBER'}'
	FUNCTION_RETURN_VALUE=$(cat $FILENAME | awk 'BEGIN{FS="\t";}{print $'$COLUMN_NUMBER'}' | awk 'BEGIN{FS="\t"} NR == 1 {min=$1;} {if($1<min){min=$1}} END{print min}')
	#echo $FUNCTION_RETURN_VALUE

#cat $FILENAME | awk 'BEGIN{FS="\t";}{print $'$COLUMN_NUMBER'}' | awk 'BEGIN{FS="\t"} NR == 1 {min=$1;} {if($1<min){min=$1}} END{print min}'
#awk 'BEGIN{c=0;sum=0;}\
#/^[^#]/{a[c++]=$1;sum+=$1;}\
#END{ave=sum/c;\
#if((c%2)==1){median=a[int(c/2)];}\
#else{median=(a[c/2]+a[c/2-1])/2;}\
#print sum,"\t",c,"\t",ave,"\t",median,"\t",a[0],"\t",a[c-1]}'

}  

function getMaximalValue {

	FILENAME=$1
	COLUMN_NUMBER=$2
	#cat $FILENAME | awk 'BEGIN{FS="\t";}{print $'$COLUMN_NUMBER'}'
	FUNCTION_RETURN_VALUE=$(cat $FILENAME | awk 'BEGIN{FS="\t";}{print $'$COLUMN_NUMBER'}' | awk 'BEGIN{FS="\t"} NR == 1 {max=$1;} {if($1<max){max=$1}} END{print max}')

}  

function getAverageValue {

	FILENAME=$1
	COLUMN_NUMBER=$2
	#cat $FILENAME | awk 'BEGIN{FS="\t";}{print $'$COLUMN_NUMBER'}'
	FUNCTION_RETURN_VALUE=$(cat $FILENAME | awk 'BEGIN{FS="\t";}{print $'$COLUMN_NUMBER'}' | awk 'BEGIN{FS="\t"; sum=0; counter=0} {sum+=$1} END{if(counter>0){print sum/counter}else{print ERROR}}')

} 


DATAFILE_CONTAINING_RESULTS="$1"
OUTPUT_DIRECTORY="$2"

#echo $(getMinimalValue $DATAFILE_CONTAINING_RESULTS 16)
getMinimalValue $DATAFILE_CONTAINING_RESULTS 16
MIN_VALUE=$FUNCTION_RETURN_VALUE
getMaximalValue $DATAFILE_CONTAINING_RESULTS 16
MAX_VALUE=$FUNCTION_RETURN_VALUE
getAverageValue $DATAFILE_CONTAINING_RESULTS 16
AVERAGE_VALUE=$FUNCTION_RETURN_VALUE

exit 0;
