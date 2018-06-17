set xtics nomirror
set ytics nomirror
#set xlabel 'Data size in number of Elements (int)'
set ylabel 'Average Estimation Errors' offset 0,-2
#set key top left Left reverse samplen 1

set title '%EXPERIMENT_NAME%: Relative Estimation Errors'

#set logscale y

set key top right Left reverse samplen 1
set key below
set key box



set datafile separator "\t"


set xlabel '%EXPERIMENT_NAME%'

#set xlabel '%XLABEL%'

plot 'averaged_response_time.data' using 1:28 title "CPU: Response Time" w lp lw 4, \
'averaged_waiting_time_aware_response_time.data' using 1:28 title "CPU: Waiting Time Aware Response Time" w lp lw 4, \
'averaged_throughput.data' using 1:28 title "CPU: Throughput" w lp lw 4, \
'averaged_throughput2.data' using 1:28 title "CPU: Throughput2" w lp lw 4, \
'averaged_response_time.data' using 1:29 title "GPU: Response Time" w lp lw 4, \
'averaged_waiting_time_aware_response_time.data' using 1:29 title "GPU: Waiting Time Aware Response Time" w lp lw 4, \
'averaged_throughput.data' using 1:29 title "GPU: Throughput" w lp lw 4, \
'averaged_throughput2.data' using 1:29 title "GPU: Throughput2" w lp lw 4     

set output "%EXPERIMENT_NAME%_relative_errors_MAX_DATASET_SIZE_IN_MB.pdf"
set terminal pdfcairo font "Helvetica,9" size 5, 4
replot
