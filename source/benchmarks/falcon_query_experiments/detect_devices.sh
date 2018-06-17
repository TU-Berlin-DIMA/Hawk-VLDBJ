
clinfo > device_info.tmp
RET=$?
if [ $RET -ne 0 ]; then
  echo "Error executing clinfo, check if this is installed and working correctly!"
  echo "Abort execution!"
  exit -1
fi

device_types=(`cat device_info.tmp | grep "Device Type" | sed -e 's/ //g' -e 's/\t//g' -e 's/DeviceType//g' -e 's/://g' -e 's/CL_DEVICE_TYPE_//g'`)
device_unified_memories=(`cat device_info.tmp | grep "Unified memory for Host and Device" | sed -e 's/Unified memory for Host and Device//g' -e 's/1/Yes/g' -e 's/0/No/g' | sed -e 's/ //g' -e 's/\t//g' -e 's/DeviceType//g' -e 's/://g' -e 's/CL_DEVICE_TYPE_//g'`)

#device_names=(`cat device_info.tmp | grep "Device Name" | sed -e 's/                                     /=/g' | awk 'BEGIN{FS="=";}{ print $2}' | sed -e 's/ /-/g' -e 's/@//g'`)
#device_types=(`cat device_info.tmp | grep "Device Type" | sed -e 's/                                     /=/g' | awk 'BEGIN{FS="=";}{ print $2}' | sed -e 's/ /-/g' -e 's/@//g'`)
#device_unified_memories=(`cat device_info.tmp | grep "Unified memory for Host and Device" | sed -e 's/              /=/g' | awk 'BEGIN{FS="=";}{ print $2}' | sed -e 's/ /-/g' -e 's/@//g'`)

DIR=devices
mkdir -p "$DIR"

for ((i=0; i<${#device_types[@]}; i++)); do
 # echo "${device_types[$i]} ${device_unified_memories[$i]}"

 if [ "${device_types[$i]}" == "GPU" ]; then

  if [ "${device_unified_memories[$i]}" == "Yes" ]; then
     #integrated GPU using the same memory as CPU
     echo "set code_gen.cl_device_type=igpu
set code_gen.num_threads=1
set enable_parallel_pipelines=false" > "$DIR"/igpu.coga

  elif [ "${device_unified_memories[$i]}" == "No" ]; then
     #GPU with dedicated device memory
     echo "set code_gen.cl_device_type=dgpu
set code_gen.num_threads=1
set enable_parallel_pipelines=false
set code_gen.enable_caching=true" > "$DIR"/dgpu.coga

  else
    echo "Fatal Error: Invalid Value!"
    exit -1
  fi

 elif [ "${device_types[$i]}" == "CPU" ]; then
     CORES=$(nproc)
     echo "set code_gen.cl_device_type=cpu
set code_gen.num_threads=$CORES
set code_gen.cl_command_queue_strategy=subdevices
set enable_parallel_pipelines=true" > "$DIR"/cpu.coga

 elif [ "${device_types[$i]}" == "Accelerator" ]; then
     echo "set code_gen.cl_device_type=phi
set code_gen.num_threads=1
set enable_parallel_pipelines=false
set code_gen.enable_caching=true
" > "$DIR"/phi.coga
 fi
done

rm device_info.tmp

#clinfo | grep "Device Name" | sed -e 's/                                     /=/g' | awk 'BEGIN{FS="=";}{ print $2}'

#  Device Type                                     GPU
#  Unified memory for Host and Device              No

