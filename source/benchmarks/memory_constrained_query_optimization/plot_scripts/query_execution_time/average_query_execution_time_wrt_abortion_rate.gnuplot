set xtics nomirror
set ytics nomirror
set xlabel 'Abortion Rate'
set ylabel 'Average Query Execution Time in ms' offset 0,-2
#set key top left Left reverse samplen 1

set title 'Average Query Execution Time Depending on Abortion Rate'

#set logscale y

set key top right Left reverse samplen 1
set key below
set key box



#set datafile separator ","
set datafile separator "|"


#set xlabel '%XLABEL%'

#plot 'SF1.csv' using 5:6 title "SF1" w lp lw 4, \
#'SF5.csv' using 5:6 title "SF5" w lp lw 4, \
#'SF10.csv' using 5:6 title "SF10" w lp lw 4, \
#'SF15.csv' using 5:6 title "SF15" w lp lw 4

plot 'SF1.csv' using 5:6 title "SF1" w lp lw 4, \
'SF5.csv' using 5:6 title "SF5" w lp lw 4, \
'SF10.csv' using 5:6 title "SF10" w lp lw 4, \
'SF15.csv' using 5:6 title "SF15" w lp lw 4, \
'SF20.csv' using 5:6 title "SF20" w lp lw 4, \
'SF30.csv' using 5:6 title "SF30" w lp lw 4, \
'SF40.csv' using 5:6 title "SF40" w lp lw 4, \
'SF50.csv' using 5:6 title "SF50" w lp lw 4, \
'SF60.csv' using 5:6 title "SF60" w lp lw 4, \
'SF70.csv' using 5:6 title "SF70" w lp lw 4, \
'SF80.csv' using 5:6 title "SF80" w lp lw 4, \
'SF90.csv' using 5:6 title "SF90" w lp lw 4, \
'SF100.csv' using 5:6 title "SF100" w lp lw 4

   
set output "average_query_execution_time_wrt_abortion_rate.pdf"
set terminal pdfcairo font "Helvetica,9" size 5, 4
replot





