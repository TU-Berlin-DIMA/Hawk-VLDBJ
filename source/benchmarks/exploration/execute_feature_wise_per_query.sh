#!/bin/bash

bash ../falcon_variant_experiments/detect_devices.sh > /dev/null 2>&1
RET=$?
if [ $RET -ne 0 ]; then
  echo "Error detecting OpenCL devices!"
  echo "Abort execution!"
  exit -1
fi

if [ ! -f PATH_TO_COGADB_EXECUTABLE ]
then
  echo "File 'PATH_TO_COGADB_EXECUTABLE' not found!"
  echo "Please enter the path to the cogadb executable:"
  read LINE
  echo "$LINE" > PATH_TO_COGADB_EXECUTABLE
fi
EXECUTABLE=$(cat PATH_TO_COGADB_EXECUTABLE)

if [ ! -f PATH_TO_DATABASE_SSB ]
then
    echo "File 'PATH_TO_DATABASE_SSB' not found!"
    echo "Please enter the path to the cogadb database:"
    read LINE
    echo "$LINE" > PATH_TO_DATABASE_SSB
fi

if [ ! -f PATH_TO_DATABASE_TPCH ]
then
    echo "File 'PATH_TO_DATABASE_TPCH' not found!"
    echo "Please enter the path to the cogadb database:"
    read LINE
    echo "$LINE" > PATH_TO_DATABASE_TPCH
fi

DATABASE_PATH_SSB=$(cat PATH_TO_DATABASE_SSB)
DATABASE_PATH_TPCH=$(cat PATH_TO_DATABASE_TPCH)

echo "Benchmarking: ${EXECUTABLE}"
echo "SSB dataset:  ${DATABASE_PATH_SSB}"
echo "TPCH dataset:  ${DATABASE_PATH_TPCH}"

rm -rf results
mkdir -v -p results

rm -f startup.coga
rm -f debug.log
rm -f error.log
rm -f exploration.csv
rm -f pdflatex.log
rm -f report.tex

rm -f *.c
rm -f *.pch
rm -f *.so
rm -f *.aux
rm -f *.log
rm -f *_feature_wise_best_variant.coga

for query in ssb_queries/*.coga; do
  for device in devices/*.coga; do
    device_name=$(echo $device | sed -e 's/devices\///g' -e 's/\.coga//g')
    query_name=$(echo $query | sed -e 's/ssb_queries\///g' -e 's/\.coga//g')
    echo "Execute experiments on device $device_name"
    echo "set path_to_database=${DATABASE_PATH_SSB}"         >  ./startup.coga
    cat ./default.coga                        >> ./startup.coga
    cat ${device}                             >> ./startup.coga
    echo "set code_gen.variant_exploration_mode=feature_wise_exploration" >> ./startup.coga
    cat ${query}                              >> ./startup.coga

    echo "quit"                               >> ./startup.coga

    # Execute cogadbd from $1
    # Rename first occurrence of [TrainingHeader] to [TrainingHead]
    # Remove all lines that contain [TrainingHeader]
    # Remove all occurrences of [Training] and [TrainingHead]
    DATE=$(date +%F-%H%M%S)
    ExpID="$DATE-$HOSTNAME-$device_name-$query_name"
    echo $ExpID >> error.log

    "$EXECUTABLE" > debug.log
    RET=$?
    #if we used STRG+C, terminate experiment
    if [ $RET -eq 130 ]; then
      echo "Experiment '$ExpID' aborted by user! I will not send an error report via email!"
      exit -1
    fi

    #if some kind of error occured, repeat
    if [ $RET -ne 0 ]; then
      echo "Fatal Error occured while executing experiments for query: $ExpID"
      bash ../send_error_email.sh "Fatal Error occured while executing experiments for query: '$ExpID' on device '$device_name'"
      exit -1
    fi

    mv -f ./startup.coga ./results/startup-"$ExpID".coga
    mv -f ./debug.log    ./results/debug-"$ExpID".log

    rm -f ./*.aux
    rm -f ./*.c
    rm -f ./*.log
  done
done

for query in tpch_queries_feature_wise_per_query/*.coga; do
  for device in devices/*.coga; do
    device_name=$(echo $device | sed -e 's/devices\///g' -e 's/\.coga//g')
    query_name=$(echo $query | sed -e 's/tpch_queries_feature_wise_per_query\///g' -e 's/\.coga//g')
    echo "Execute experiments on device $device_name"
    echo "set path_to_database=${DATABASE_PATH_TPCH}"         >  ./startup.coga
    cat ./default.coga                        >> ./startup.coga
    cat ${device}                             >> ./startup.coga
    echo "set code_gen.variant_exploration_mode=feature_wise_exploration" >> ./startup.coga
    cat ${query}                              >> ./startup.coga

    echo "quit"                               >> ./startup.coga

    # Execute cogadbd from $1
    # Rename first occurrence of [TrainingHeader] to [TrainingHead]
    # Remove all lines that contain [TrainingHeader]
    # Remove all occurrences of [Training] and [TrainingHead]
    DATE=$(date +%F-%H%M%S)
    ExpID="$DATE-$HOSTNAME-$device_name-$query_name"
    echo $ExpID >> error.log

    "$EXECUTABLE" > debug.log
    RET=$?
    #if we used STRG+C, terminate experiment
    if [ $RET -eq 130 ]; then
     echo "Experiment '$ExpID' aborted by user! I will not send an error report via email!"
      exit -1
    fi
    #if some kind of error occured, repeat
    if [ $RET -ne 0 ]; then
      echo "Fatal Error occured while executing experiments for query: $ExpID"
      bash ../send_error_email.sh "Fatal Error occured while executing experiments for query: '$ExpID' on device '$device_name'"
      exit -1
    fi

    mv -f ./startup.coga ./results/startup-"$ExpID".coga
    mv -f ./debug.log    ./results/debug-"$ExpID".log

    rm -f ./*.aux
    rm -f ./*.c
    rm -f ./*.log

  done
done

mv -f ./error.log ./results/error.log

DATE=$(date +%F-%H%M%S)

mv results exploration-"$DATE"-"$HOSTNAME"-per-query
zip -r exploration-"$DATE"-"$HOSTNAME"-per-query.zip exploration-"$DATE"-"$HOSTNAME"-per-query

bash ../send_success_mail.sh "FeatureWise Exploration Experiments successfully completed"

rm -rf ./output

rm -f startup.coga
rm -f debug.log
rm -f error.log
rm -f exploration.csv
rm -f pdflatex.log
rm -f report.tex

rm -f *.c
rm -f *.pch
rm -f *.so
rm -f *.aux
rm -f *.log
