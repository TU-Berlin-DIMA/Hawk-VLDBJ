
load "template.gnuplot"

set xlabel 'Number of Parallel Queries'
set ylabel 'Copy time CPU to GPU (s)' offset 0,-1
#set yrange [:26.5]

#set logscale x
#set logscale y
#set key at 4, 22
#6.5

set output "copy_times.pdf"

#set ytics ("0.1" 0.1, "1" 1, "26" 26)

plot 'false-false-false-measurements.csv' using ($1):8 title 'No Pin, No DDP, no QC' w lp lw 6 pt 6 ps 1.5 lc rgb 'black', \
'false-true-false-measurements.csv' using ($1):8 title 'Pin, No DDP, no QC' w lp lw 6 ps 1.5 lc rgb 'gray70', \
'true-false-false-measurements.csv' using ($1):8 title 'No Pin, DDP, no QC' w lp lw 6 ps 1.5 lc rgb 'gray50', \
'false-false-true-measurements.csv' using ($1):8 title 'No Pin, No DDP, QC' w lp lw 6 pt 6 ps 1.5 lc rgb 'black', \
'false-true-true-measurements.csv' using ($1):8 title 'Pin, No DDP, QC' w lp lw 6 ps 1.5 lc rgb 'gray70', \
'true-false-true-measurements.csv' using ($1):8 title 'No Pin, DDP, QC' w lp lw 6 ps 1.5 lc rgb 'gray50', \
'true-true-true-measurements.csv' using ($1):8 title 'Pin, DDP, QC' w lp lw 6 pt 8 ps 1.5 lc rgb 'gray50'

#

#'true-true-measurements.csv' using ($1/(1024*1024*1024)):7 title 'Pin, DDP' w lp lw 6 pt 6 ps 1.5 lc rgb 'gray30'


#reset

#plot 'false-false-measurements.csv' using 1:5 title 'No Pin, No DDP' with linespoints, \
#'false-true-measurements.csv' using 1:5 title 'Pin, No DDP' with linespoints, \
#'true-false-measurements.csv' using 1:5 title 'No Pin, DDP' with linespoints, \
#'true-true-measurements.csv' using 1:5 title 'Pin, DDP' with linespoints \

