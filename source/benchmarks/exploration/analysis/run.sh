#!/bin/sh
SCRIPTLOCATION="`dirname \"$0\"`"
# assume csv log files produced by ../execute.sh to be located in $1
rm *.csv.tex
bash ${SCRIPTLOCATION}/feature_wise_log_to_csv.sh $2
python3 ${SCRIPTLOCATION}/prepare.py $1 $2

for f in *.report.tex; do pdflatex "${f}"; done
rm data.csv
rm *.aux
rm *.log
