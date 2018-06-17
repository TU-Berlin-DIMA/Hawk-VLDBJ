#!/bin/bash

cat cpu_only.data | awk 'BEGIN{FS="\t"} {print $1"\t"$2"\t"$3"\t"$5"\t"$10}' > tmp

for i in cpu_only.data gpu_only.data response_time.data simple_round_robin.data waiting_time_aware_response_time.data throughput.data throughput2.data; do
	echo "Add execution times for heuristic $i"
	cat "$i" | awk 'BEGIN{FS="\t"} {print $16}' > tmp2
	paste tmp tmp2 > tmp3
	mv tmp3 tmp
done

mv tmp speedups.data
rm tmp2

exit 0
