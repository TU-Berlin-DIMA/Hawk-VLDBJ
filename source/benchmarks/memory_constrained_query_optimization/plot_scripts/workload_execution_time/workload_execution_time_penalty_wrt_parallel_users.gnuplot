set xtics nomirror
set ytics nomirror
set xlabel 'Number of Parallel Users'
set ylabel 'Workload Execution Time in ms' offset 0,-2
#set key top left Left reverse samplen 1

set title 'Workload Execution Time and time penalty due to aborted GPU operators depending on Number of Users'

#set logscale y

set key top right Left reverse samplen 1
set key below
set key box



#set datafile separator ","
set datafile separator "|"


#set xlabel '%XLABEL%'

#plot 'SF1.csv' using 2:6 title "SF1 (WITH GPU)" w lp lw 4, \
#'SF1.csv' using 2:($8/(1000000)) title "SF1 (Time Penalty)" w lp lw 4, \
#'SF5.csv' using 2:6 title "SF5 (WITH GPU)" w lp lw 4, \
#'SF5.csv' using 2:($8/(1000000)) title "SF5 (Time Penalty)" w lp lw 4, \
#'SF10.csv' using 2:6 title "SF10 (WITH GPU)" w lp lw 4, \
#'SF10.csv' using 2:($8/(1000000)) title "SF10 (Time Penalty)" w lp lw 4, \
#'SF15.csv' using 2:6 title "SF15 (WITH GPU)" w lp lw 4, \
#'SF15.csv' using 2:($8/(1000000)) title "SF15 (Time Penalty)" w lp lw 4, \
#'SF20.csv' using 2:6 title "SF20 (WITH GPU)" w lp lw 4, \
#'SF20.csv' using 2:($8/(1000000)) title "SF20 (Time Penalty)" w lp lw 4, \
#'SF30.csv' using 2:6 title "SF30 (WITH GPU)" w lp lw 4, \
#'SF30.csv' using 2:($8/(1000000)) title "SF30 (Time Penalty)" w lp lw 4, \
#'SF40.csv' using 2:6 title "SF40 (WITH GPU)" w lp lw 4, \
#'SF40.csv' using 2:($8/(1000000)) title "SF40 (Time Penalty)" w lp lw 4, \
#'SF50.csv' using 2:6 title "SF50 (WITH GPU)" w lp lw 4, \
#'SF50.csv' using 2:($8/(1000000)) title "SF50 (Time Penalty)" w lp lw 4, \
#'SF60.csv' using 2:6 title "SF60 (WITH GPU)" w lp lw 4, \
#'SF60.csv' using 2:($8/(1000000)) title "SF60 (Time Penalty)" w lp lw 4, \
#'SF70.csv' using 2:6 title "SF70 (WITH GPU)" w lp lw 4, \
#'SF70.csv' using 2:($8/(1000000)) title "SF70 (Time Penalty)" w lp lw 4, \
#'SF80.csv' using 2:6 title "SF80 (WITH GPU)" w lp lw 4, \
#'SF80.csv' using 2:($8/(1000000)) title "SF80 (Time Penalty)" w lp lw 4, \
#'SF90.csv' using 2:6 title "SF90 (WITH GPU)" w lp lw 4, \
#'SF90.csv' using 2:($8/(1000000)) title "SF90 (Time Penalty)" w lp lw 4, \
#'SF100.csv' using 2:6 title "SF100 (WITH GPU)" w lp lw 4, \
#'SF100.csv' using 2:($8/(1000000)) title "SF100 (Time Penalty)" w lp lw 4


#plot 'SF1.csv' using 2:6 title "SF1 (WITH GPU)" w lp lw 4, \
#'SF1.csv' using 2:($8/(1000000)) title "SF1 (Time Penalty)" w lp lw 4, \
#'SF5.csv' using 2:6 title "SF5 (WITH GPU)" w lp lw 4, \
#'SF5.csv' using 2:($8/(1000000)) title "SF5 (Time Penalty)" w lp lw 4, \
#'SF10.csv' using 2:6 title "SF10 (WITH GPU)" w lp lw 4, \
#'SF10.csv' using 2:($8/(1000000)) title "SF10 (Time Penalty)" w lp lw 4, \
#'SF15.csv' using 2:6 title "SF15 (WITH GPU)" w lp lw 4, \
#'SF15.csv' using 2:($8/(1000000)) title "SF15 (Time Penalty)" w lp lw 4

plot 'SF1.csv' using 2:6 title "SF1 (WITH GPU)" w lp lw 4, \
'SF1.csv' using 2:($8/(1000000)) title "SF1 (Time Penalty)" w lp lw 4, \
'SF5.csv' using 2:6 title "SF5 (WITH GPU)" w lp lw 4, \
'SF5.csv' using 2:($8/(1000000)) title "SF5 (Time Penalty)" w lp lw 4, \
'SF10.csv' using 2:6 title "SF10 (WITH GPU)" w lp lw 4, \
'SF10.csv' using 2:($8/(1000000)) title "SF10 (Time Penalty)" w lp lw 4, \
'SF15.csv' using 2:6 title "SF15 (WITH GPU)" w lp lw 4, \
'SF15.csv' using 2:($8/(1000000)) title "SF15 (Time Penalty)" w lp lw 4, \
'SF20.csv' using 2:6 title "SF20 (WITH GPU)" w lp lw 4, \
'SF20.csv' using 2:($8/(1000000)) title "SF20 (Time Penalty)" w lp lw 4, \
'SF30.csv' using 2:6 title "SF30 (WITH GPU)" w lp lw 4, \
'SF30.csv' using 2:($8/(1000000)) title "SF30 (Time Penalty)" w lp lw 4, \
'SF40.csv' using 2:6 title "SF40 (WITH GPU)" w lp lw 4, \
'SF40.csv' using 2:($8/(1000000)) title "SF40 (Time Penalty)" w lp lw 4, \
'SF50.csv' using 2:6 title "SF50 (WITH GPU)" w lp lw 4, \
'SF50.csv' using 2:($8/(1000000)) title "SF50 (Time Penalty)" w lp lw 4, \
'SF60.csv' using 2:6 title "SF60 (WITH GPU)" w lp lw 4, \
'SF60.csv' using 2:($8/(1000000)) title "SF60 (Time Penalty)" w lp lw 4, \
'SF70.csv' using 2:6 title "SF70 (WITH GPU)" w lp lw 4, \
'SF70.csv' using 2:($8/(1000000)) title "SF70 (Time Penalty)" w lp lw 4, \
'SF80.csv' using 2:6 title "SF80 (WITH GPU)" w lp lw 4, \
'SF80.csv' using 2:($8/(1000000)) title "SF80 (Time Penalty)" w lp lw 4, \
'SF90.csv' using 2:6 title "SF90 (WITH GPU)" w lp lw 4, \
'SF90.csv' using 2:($8/(1000000)) title "SF90 (Time Penalty)" w lp lw 4, \
'SF100.csv' using 2:6 title "SF100 (WITH GPU)" w lp lw 4, \
'SF100.csv' using 2:($8/(1000000)) title "SF100 (Time Penalty)" w lp lw 4
   
   
set output "workload_execution_time_penalty_wrt_parallel_users.pdf"
set terminal pdfcairo font "Helvetica,9" size 5, 4
replot





