#!/bin/bash

nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,power.draw,temperature.gpu --format=csv,nounits --id=1 -lms 10 -f collected_gpu_statistics.csv

exit 0
