
set xlabel 'Buffer Size in GB'
set ylabel 'Execution time in seconds'

#set logscale x
set logscale y



plot 'false-false-measurements.csv' using ($1/(1024*1024*1024)):4 title 'No Pin, No DDP' with linespoints, \
'false-true-measurements.csv' using ($1/(1024*1024*1024)):4 title 'Pin, No DDP' with linespoints, \
'true-false-measurements.csv' using ($1/(1024*1024*1024)):4 title 'No Pin, DDP' with linespoints, \
'true-true-measurements.csv' using ($1/(1024*1024*1024)):4 title 'Pin, DDP' with linespoints \

#reset

#plot 'false-false-measurements.csv' using 1:5 title 'No Pin, No DDP' with linespoints, \
#'false-true-measurements.csv' using 1:5 title 'Pin, No DDP' with linespoints, \
#'true-false-measurements.csv' using 1:5 title 'No Pin, DDP' with linespoints, \
#'true-true-measurements.csv' using 1:5 title 'Pin, DDP' with linespoints \

