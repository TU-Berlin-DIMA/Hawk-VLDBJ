#!/bin/sh

for logf in $1/*.log; do 
    csvf=$(echo ${logf} | sed -e "s/\.log/.csv/")
    query_name=$(echo ${logf} | sed -e "s/\.log//" -e 's@.*igpu-@@' -e 's@.*dgpu-@@' -e 's@.*phi-@@' -e 's@.*cpu-@@')

    data=$(cat ${logf} | grep "\[Training\]" | sed -r -e "s/\[Training\]//")

    data=$(echo "${data}" | sed -e "1,\$s/^/${query_name};/")
    echo "Query;Host;DeviceType;Device;ExplorationMode;VariantTag;Variant;Min;Max;Median;Mean;Stdev;Var" > "${csvf}"
    echo "${data}" >> "${csvf}"
done
