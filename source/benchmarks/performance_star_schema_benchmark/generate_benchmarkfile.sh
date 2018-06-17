#!/bin/bash
rm -f benchmark.coga
for u in {1..100}; do
for i in ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43; do
echo $i >> benchmark.coga
done
done

cp benchmark.coga benchmark_query_chopping.coga
for u in {1..100}; do
for i in ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43; do
echo $i >> benchmark_query_chopping.coga
done
done

for i in ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43; do
rm -f benchmark_query_"$i"_only.coga
for u in {1..100}; do
echo $i >> benchmark_query_"$i"_only.coga
done
done


exit 0
