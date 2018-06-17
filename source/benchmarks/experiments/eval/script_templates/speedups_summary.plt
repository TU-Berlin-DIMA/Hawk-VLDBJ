# set terminal pngcairo  transparent enhanced font "arial,10" fontscale 1.0 size 500, 350 
# set output 'boxplot.1.png'
set border 2 front linetype -1 linewidth 1.000
#set boxwidth 0.5 absolute
set boxwidth 0.9 absolute
set style fill   solid 0.25 border lt -1
unset key
set pointsize 0.5
set style data boxplot
#set xtics border in scale 0,0 nomirror norotate  offset character 0, 0, 0 autojustify
#set xtics  norangelimit
set xrange [0.2:10]
set xtics   ("SRT" 1.00000, "WTAR" 2.00000, "RR" 3.00000, "TBO" 4.00000, "PBO" 5.00000)
#set xtics   ("RT\n (CPU)" 1.00000, "RT\n (GPU)" 2.00000, "WTAR\n (CPU)" 3.00000, "WTAR\n (GPU)" 4.00000, "RR\n (CPU)" 5.00000, "RR\n (GPU)" 6.00000, "Th2\n (CPU)" 7.00000, "Th2\n (GPU)" 8.00000, "Th1\n (CPU)" 9.00000, "Th1\n (GPU)" 10.00000)
set ytics border in scale 1,0.5 nomirror norotate  offset character 0, 0, 0 autojustify
#set yrange [ 0.00000 : 100.000 ] noreverse nowriteback
#set yrange [ 0.01:  ]
#set xrange [ 0.2 : 4.3]
set xrange [ 0.2 : 5.3]

#set xlabel 'Optimization Heuristics' offset 0,-0.5
set xlabel 'Optimization Heuristics'
#set ylabel 'Speedup over all %OPERATION_NAME% Experiments'  offset 2
#set ylabel 'Speedup'  offset 2
set ylabel 'Speedup w.r.t. fastest PD'  offset 2,-0.5

#set logscale y

     # Input file contains tab-separated fields
     set datafile separator "\t"

#functions
min(x,y) = (x < y) ? x : y
max(x,y) = (x > y) ? x : y



#plot 'speedups.data' using (1):(min($6,$7)/($8)), \ #response_time
#'speedups.data' using (2):(min($6,$7)/($10)), \ #waiting_time_aware_response_time
#'speedups.data' using (3):(min($6,$7)/($9)), \ #simple_round_robin
#'speedups.data' using (4):(min($6,$7)/($11)), \ #threshold_based_outsourcing
#'speedups.data' using (5):(min($6,$7)/($12)) #threshold2

plot 'speedups.data' using (1):(min($6,$7)/($8)), \
'speedups.data' using (2):(min($6,$7)/($10)), \
'speedups.data' using (3):(min($6,$7)/($9)), \
'speedups.data' using (4):(min($6,$7)/($11)) , \
'speedups.data' using (5):(min($6,$7)/($12))



#plot 'cpu_only.data' using (1):2, '' using (2):(5*$3)

#set output "AGGREGATION_average_estimation_error_for_all_experiments.pdf" 
#set terminal pdfcairo mono font "Helvetica,30" size 6.1, 4
#replot

#set output "AGGREGATION_average_estimation_error_for_all_experiments.pdf" 
#set terminal pdfcairo mono font "Helvetica,30" size 12.2, 5
#replot

#set output "%OPERATION_NAME%_speedups_for_all_experiments.pdf" 
#set terminal pdfcairo mono font "Helvetica,15" size 5, 3
#replot

set output "%OPERATION_NAME%_speedups_for_all_experiments.pdf" 
#set terminal pdfcairo mono font "Helvetica,15" size 5, 2
set terminal pdfcairo mono font "Helvetica,30" size 6.1, 4
replot

#set output "AGGREGATION_average_estimation_error_for_all_experiments.pdf" 
#set terminal pdfcairo font "Helvetica,11" size 3, 3
#replot

