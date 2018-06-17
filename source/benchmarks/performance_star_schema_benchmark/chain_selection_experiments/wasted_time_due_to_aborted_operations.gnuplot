
load "template.gnuplot"

set terminal pdfcairo font "Helvetica,30" size 8.1, 8


set xlabel 'Number of Parallel Queries'
set ylabel 'Wasted Time in Seconds'

#set logscale x
set logscale y

set key outside center bottom

set output "wasted_time_due_to_aborted_operations.pdf"

#plot 'false-false-false-measurements.csv' using ($1):5 title 'No DDP, No QC, No Pull' w lp lw 6 pt 6 ps 1.5 lc rgb 'black', \
#'false-true-false-measurements.csv' using ($1):5 title 'No DDP, QC, No Pull' w lp lw 6 ps 1.5 lc rgb 'gray70', \
#'true-false-false-measurements.csv' using ($1):5 title 'DDP, No QC, No Pull' w lp lw 6 pt 8 ps 1.5 lc rgb 'gray50', \
#'false-false-true-measurements.csv' using ($1):5 title 'No DDP, No QC, Pull' w lp lw 6 ps 1.5 lc rgb 'black', \
#'false-true-true-measurements.csv' using ($1):5 title 'No DDP, QC, Pull' w lp lw 6 ps 1.5 lc rgb 'gray70', \
#'true-true-true-measurements.csv' using ($1):5 title 'DDP, QC, Pull' w lp lw 6 ps 1.5 lc rgb 'gray70', \
#'true-false-true-measurements.csv' using ($1):5 title 'DDP, No QC, Pull' w lp lw 6 pt 8 ps 1.5 lc rgb 'gray50'

plot 'false-false-false-measurements.csv' using ($1):15 title 'No DDP, No QC, No Pull' w lp lw 6 pt 6 ps 1.5 lc rgb 'black', \
'false-true-false-measurements.csv' using ($1):15 title 'No DDP, QC, No Pull' w lp lw 6 ps 1.5 lc rgb 'gray70', \
'true-false-false-measurements.csv' using ($1):15 title 'DDP, No QC, No Pull' w lp lw 6 pt 8 ps 1.5 lc rgb 'gray50', \
'false-true-true-measurements.csv' using ($1):15 title 'No DDP, QC, Pull' w lp lw 6 ps 1.5 lc rgb 'gray70', \
'true-true-true-measurements.csv' using ($1):15 title 'DDP, QC, Pull' w lp lw 6 ps 1.5 lc rgb 'gray70'




#'true-true-false-measurements.csv' using ($1):5 title 'Pin, DDP, No QC' w lp lw 6 ps 1.5



#reset

#plot 'false-false-measurements.csv' using 1:5 title 'No Pin, No DDP' with linespoints, \
#'false-true-measurements.csv' using 1:5 title 'Pin, No DDP' with linespoints, \
#'true-false-measurements.csv' using 1:5 title 'No Pin, DDP' with linespoints, \
#'true-true-measurements.csv' using 1:5 title 'Pin, DDP' with linespoints \

