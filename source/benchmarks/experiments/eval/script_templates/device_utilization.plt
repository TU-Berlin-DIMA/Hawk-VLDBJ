# set terminal pngcairo  transparent enhanced font "arial,10" fontscale 1.0 size 500, 350 
# set output 'boxplot.1.png'
set border 2 front linetype -1 linewidth 1.000
set boxwidth 0.9 absolute
set style fill   solid 0.25 border lt -1
unset key
set pointsize 0.5
set style data boxplot
#set xtics border in scale 0,0 nomirror norotate  offset character 0, 0, 0 autojustify
set xtics  norangelimit
#set xtics   ("RT" 1.00000, "WTAR" 2.00000, "RR" 3.00000, "Th2" 4.00000, "Th1" 5.00000)
#set xtics   ("RT\n (CPU)" 1.00000, "RT\n (GPU)" 2.00000, "WTAR\n (CPU)" 3.00000, "WTAR\n (GPU)" 4.00000, "RR\n (CPU)" 5.00000, "RR\n (GPU)" 6.00000, "TBO\n (CPU)" 7.00000, "TBO\n (GPU)" 8.00000)
set xtics   ("SRT" 1.00000, "WTAR" 2.00000, "RR" 3.00000, "TBO" 4.00000)
set ytics border in scale 1,0.5 nomirror norotate  offset character 0, 0, 0 autojustify
#set yrange [ 0.00000 : 100.000 ] noreverse nowriteback
set yrange [ 0 : 100 ]
set xrange [0.2:4.3]

#set title '%OPERATION_NAME%: Device Utilization'

#set xlabel 'Optimization Heuristics' offset 0,-0.5
set xlabel 'Optimization Heuristics'
set ylabel 'Device Utilization in %' offset 2


set style arrow 1 nohead filled size screen 0.025,30,45 ls 1 lw 6 linecolor rgb "gray60"
set arrow from graph(0,0),0.5 to graph(0,1.2),0.5 as 1

#set arrow from graph(0,0.3),0.2 to graph(0,0.4),0.2 as 1
set arrow from graph(0,0.73),0.98 to graph(0,0.83),0.98 as 1

set label "optimal" at graph(0,0.85),0.98

#set arrow from 0.35,graph(0,0) to 0.35,graph(1,1) nohead
#set arrow from graph(0,0),0.5 to graph(0,1),0.5 nohead

     # Input file contains tab-separated fields
     set datafile separator "\t"

plot 'response_time.data' using (1):($20*100), \
'waiting_time_aware_response_time.data' using (2):($20*100), \
'simple_round_robin.data' using (3):($20*100), \
'throughput.data' using (4):($20*100)

#set output "%OPERATION_NAME%_device_utilization_for_all_experiments.pdf" 
#set terminal pdfcairo font "Helvetica,9" size 5, 4
#replot

#set output "%OPERATION_NAME%_device_utilization_for_all_experiments.pdf" 
#set terminal pdfcairo mono font "Helvetica,30" size 12.2, 5
#replot

set output "%OPERATION_NAME%_device_utilization_for_all_experiments.pdf" 
set terminal pdfcairo mono font "Helvetica,30" size 6.1, 4
replot

#plot 'cpu_only.data' using (1):2, '' using (2):(5*$3)

