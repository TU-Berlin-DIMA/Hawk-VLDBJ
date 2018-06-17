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
set xrange [0.2:8]
#set xtics   ("RT" 1.00000, "WTAR" 2.00000, "RR" 3.00000, "Th2" 4.00000, "Th1" 5.00000)
set xtics   ("SRT\n (CPU)" 1.00000, "SRT\n (GPU)" 2.00000, "WTAR\n (CPU)" 3.00000, "WTAR\n (GPU)" 4.00000, "RR\n (CPU)" 5.00000, "RR\n (GPU)" 6.00000, "TBO\n (CPU)" 7.00000, "TBO\n (GPU)" 8.00000)
set ytics border in scale 1,0.5 nomirror norotate  offset character 0, 0, 0 autojustify
#set yrange [ 0.00000 : 100.000 ] noreverse nowriteback
#set yrange [ 0.1: 100.000]  noreverse nowriteback
set yrange [ 0.1: 10.000]  noreverse nowriteback

#set title '%OPERATION_NAME%: Average Estimation Error for all Experiments'

set xlabel 'Optimization Heuristics' offset 0,-0.5
set ylabel 'Average Estimation Error in %'  offset 2

set logscale y

     # Input file contains tab-separated fields
     set datafile separator "\t"

plot 'response_time.data' using (1):22, \
'response_time.data' using (2):23, \
'waiting_time_aware_response_time.data' using (3):22, \
'waiting_time_aware_response_time.data' using (4):23, \
'simple_round_robin.data' using (5):22, \
'simple_round_robin.data' using (6):23, \
'throughput.data' using (7):22, \
'throughput.data' using (8):23


#plot 'cpu_only.data' using (1):2, '' using (2):(5*$3)

#set output "%OPERATION_NAME%_average_estimation_error_for_all_experiments.pdf" 
#set terminal pdfcairo mono font "Helvetica,30" size 6.1, 5
#replot

set output "%OPERATION_NAME%_average_estimation_error_for_all_experiments.pdf" 
set terminal pdfcairo mono font "Helvetica,30" size 12.2, 5
replot

#set output "%OPERATION_NAME%_average_estimation_error_for_all_experiments.pdf" 
#set terminal pdfcairo mono font "Helvetica,15" size 5, 4
#replot

#set output "%OPERATION_NAME%_average_estimation_error_for_all_experiments.pdf" 
#set terminal pdfcairo font "Helvetica,11" size 3, 3
#replot

