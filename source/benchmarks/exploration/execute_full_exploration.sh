#!/bin/bash

bash ../falcon_variant_experiments/detect_devices.sh

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

if [ ! -f PATH_TO_DATABASE ]
then
    echo "File 'PATH_TO_DATABASE' not found!"
    echo "Please enter the path to the SSB cogadb database:"
    read LINE
    echo "$LINE" > PATH_TO_DATABASE
fi
DATABASE_PATH=$(cat PATH_TO_DATABASE)

echo "Benchmarking: ${EXECUTABLE}"
echo "SSB dataset:  ${DATABASE_PATH}"

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

for query in full_exploration_queries/*.coga; do
  for device in devices/*.coga; do
    query_name=$(echo $query | sed -e 's/full_exploration_queries\///g' -e 's/\.coga//g')
    device_name=$(echo $device | sed -e 's/devices\///g' -e 's/\.coga//g')
    echo "Execute experiments for query $query_name on device $device_name"

    echo "set path_to_database=${DATABASE_PATH}"         >  ./startup.coga
    cat ./default.coga                      >> ./startup.coga
    cat ${device}                           >> ./startup.coga

    echo "set code_gen.variant_exploration_mode=full_exploration" >> ./startup.coga
    cat ${query}                            >> ./startup.coga

    echo "quit"                             >> ./startup.coga

    cat ./report.tpl.tex.part1                           >  ./report.tex
    # Execute cogadbd from $1
    # Rename first occurrence of [TrainingHeader] to [TrainingHead]
    # Remove all lines that contain [TrainingHeader]
    # Remove all occurrences of [Training] and [TrainingHead]
    DATE=$(date +%F-%H%M%S)
    ExpID="$DATE-$HOSTNAME-${device_name}-${query_name}"
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
    csvdata=$(cat debug.log | grep -E "\[Training(Header)?\]" | sed -r -e '0,/^\[TrainingHeader\]/s//\[TrainingHead\]/' -e '/^\[TrainingHeader\]/d' -e 's/^\[Training(Head){0,1}\]//')

    #
    # Output to report
    #
    echo "$csvdata" | sed -e 's/_/-/g' >> ./report.tex

    #
    # Output to exploration.csv
    #
    # Prepend column on executed query
    csvdata="Query;$csvdata"
    csvdata=$(echo "$csvdata" | sed -e "2,\$s/^/${query_name};/")
    # Output
    echo "$csvdata" > exploration.csv

    #
    #
    #
    cat ./report.tpl.tex.part2                           >> ./report.tex
    echo "${query_name} on ${device_name} ($HOSTNAME)" | sed 's/_/-/g' >> ./report.tex
    cat ./report.tpl.tex.part3                           >> ./report.tex

    # pdflatex returns 1 on error
    pdflatex -interaction=nonstopmode -halt-on-error ./report.tex >  ./pdflatex.log
    pdflatex -interaction=nonstopmode -halt-on-error ./report.tex >> ./pdflatex.log

    mv -f ./startup.coga ./results/startup-"$ExpID".coga
    mv -f ./report.tex   ./results/report-"$ExpID".tex
    mv -f ./report.pdf   ./results/report-"$ExpID".pdf
    mv -f ./debug.log    ./results/debug-"$ExpID".log
    mv -f ./exploration.csv ./results/exploration-"$ExpID".csv
    mv -f ./pdflatex.log ./results/pdflatex-"$ExpID".log
    #SRCFILE=$(ls . | grep *.c | head -n 1 | tr -d '\n')
    #mv -f $SRCFILE       ./results/source-$ExpID.c

    rm -f ./*.aux
    rm -f ./*.c
    rm -f ./*.log

  done
done

DATE=$(date +%F-%H%M%S)

mkdir -p results/variant_summary
cd results/variant_summary
../../analysis/run.sh "$PWD"/..
cd ../..
mv results full-exploration-"$DATE"-"$HOSTNAME"
zip -r full-exploration-"$DATE"-"$HOSTNAME".zip full-exploration-"$DATE"-"$HOSTNAME"

bash ../send_success_mail.sh "Full Exploration Experiments successfully completed"

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

