set xtics nomirror
set ytics nomirror
#set xlabel 'Data size in number of Elements (int)'
set ylabel 'CPU Device Utilization in %' offset 0,-2
#set key top left Left reverse samplen 1

#set logscale y

set key top right Left reverse samplen 1
set key below
set key box

set title '%OPERATION_NAME%: Device Utilization varying_dataset_size'

set datafile separator "\t"



set xlabel 'varying_dataset_size'

plot 'averaged_response_time.data' using 1:26 title "Hybrid: Response Time" w lp lw 4, \
'averaged_waiting_time_aware_response_time.data' using 1:26 title "Hybrid: Waiting Time Aware Response Time" w lp lw 4, \
'averaged_simple_round_robin.data' using 1:26 title "Hybrid: Simple Round Robin" w lp lw 4, \
'averaged_throughput.data' using 1:26 title "Hybrid: Throughput" w lp lw 4, \
'averaged_throughput2.data' using 1:26 title "Hybrid: Throughput2" w lp lw 4      

set output "%OPERATION_NAME%_device_utilization_varying_dataset_size.pdf"
set terminal pdfcairo font "Helvetica,9" size 5, 4
replot
