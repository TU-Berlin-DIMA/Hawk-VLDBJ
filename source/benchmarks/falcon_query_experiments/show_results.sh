
echo "==============================================================================="
echo "Execution Times (in seconds):"
for i in results/*_execution_time.csv; do echo $i; cat $i; done
echo "==============================================================================="
echo "Compilation Times (Host, in seconds):"
for i in results/*_compile_time_host.csv; do echo $i; cat $i; done
echo "==============================================================================="
echo "Compilation Times (Kernel, in seconds):"
for i in results/*_compile_time_kernel.csv; do echo $i; cat $i; done

