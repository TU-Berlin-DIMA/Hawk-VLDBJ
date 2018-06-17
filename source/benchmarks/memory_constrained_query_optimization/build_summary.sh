

for query in ssb11 ssb23 ssb31 ssb42 ssball; do
rm -f $query"_summary.csv"
for i in *traditional_optimizer*$query*$HOSTNAME; do 
   echo $i; 
   echo $i >> $query"_summary.csv"; 
   echo "" >> $query"_summary.csv"; 
   echo "CPU ONLY measurements" >> $query"_summary.csv"; 
   cat $i/measurements_cpu.csv | sed -e 's/\./,/g' >> $query"_summary.csv"; 
   echo "" >> $query"_summary.csv"; 
   echo "HYBRID CPU AND GPU measurements" >> $query"_summary.csv"; 
   cat $i/measurements_any.csv | sed -e 's/\./,/g' >> $query"_summary.csv"; 

done

for i in *query_chopping*$query*$HOSTNAME; do 
   echo $i; 
   echo "" >> $query"_summary.csv"; 
   echo $i >> $query"_summary.csv"; 
   echo "" >> $query"_summary.csv"; 
   echo "CPU ONLY measurements" >> $query"_summary.csv"; 
   cat $i/measurements_cpu.csv | sed -e 's/\./,/g' >> $query"_summary.csv"; 
   echo "" >> $query"_summary.csv"; 
   echo $i >> $query"_summary.csv"; 
   echo "" >> $query"_summary.csv"; 
   echo "HYBRID CPU AND GPU measurements" >> $query"_summary.csv"; 
   cat $i/measurements_any.csv | sed -e 's/\./,/g' >> $query"_summary.csv"; 

done

done


