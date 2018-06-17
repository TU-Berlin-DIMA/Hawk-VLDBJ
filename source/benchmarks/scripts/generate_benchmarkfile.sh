#!/bin/bash
for u in {1..100}; do
for i in ssb11 ssb12 ssb13 ssb21 ssb22 ssb23 ssb31 ssb32 ssb33 ssb34 ssb41 ssb42 ssb43; do
echo $i >> benchmark.coga
done
done
exit 0
