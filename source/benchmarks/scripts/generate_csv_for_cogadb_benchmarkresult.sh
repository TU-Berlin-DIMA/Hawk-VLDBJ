#!/bin/bash
for i in ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43; do
cat cogadb_ssb_measurement.log | grep $i | awk '{print $2}' | sed -e 's/(//g' -e 's/)//g' -e 's/ms//g' -e 's/\./,/g' > $i
done

echo -e "Q1.1\tQ1.2\tQ1.3\tQ2.1\tQ2.2\tQ2.3\tQ3.1\tQ3.2\tQ3.3\tQ3.4\tQ4.1\tQ4.2\tQ4.3" > cogadb_ssb_measurement.csv
paste ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43 >> cogadb_ssb_measurement.csv
exit 0
