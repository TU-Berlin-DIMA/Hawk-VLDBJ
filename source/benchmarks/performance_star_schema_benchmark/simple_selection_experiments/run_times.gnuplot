
load "template.gnuplot"

set xlabel 'Buffer Size in GB'
set ylabel 'Execution time in seconds'

#set logscale x
set logscale y

set output "run_times.pdf"

plot 'false-false-measurements.csv' using ($1/(1024*1024*1024)):4 title 'No Pin, No DDP' w lp lw 6 pt 6 ps 1.5 lc rgb 'black', \
'false-true-measurements.csv' using ($1/(1024*1024*1024)):4 title 'Pin, No DDP' w lp lw 6 ps 1.5 lc rgb 'gray70', \
'true-false-measurements.csv' using ($1/(1024*1024*1024)):4 title 'No Pin, DDP' w lp lw 6 pt 8 ps 1.5 lc rgb 'gray50'
#'true-true-measurements.csv' using ($1/(1024*1024*1024)):4 title 'Pin, DDP' w lp lw 6 ps 1.5



#reset

#plot 'false-false-measurements.csv' using 1:5 title 'No Pin, No DDP' with linespoints, \
#'false-true-measurements.csv' using 1:5 title 'Pin, No DDP' with linespoints, \
#'true-false-measurements.csv' using 1:5 title 'No Pin, DDP' with linespoints, \
#'true-true-measurements.csv' using 1:5 title 'Pin, DDP' with linespoints \

