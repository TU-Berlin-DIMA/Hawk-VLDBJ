
NUM_RUNS=10

mkdir -p experimental_results_all_queries
mkdir -p config

bash detect_devices.sh
RET=$?
if [ $RET -ne 0 ]; then
  echo "Error detecting OpenCL devices!"
  echo "Abort execution!"
  exit -1
fi


for q in queries/*.coga; do
  query_name=$(echo $q | sed -e 's/queries\///g' -e 's/\.coga//g')
  echo "$query_name" > config/QUERY_NAME
  echo "Execute experiments for query: $query_name using $NUM_RUNS runs"
  rm -f queries.coga
  touch queries.coga
  for (( i=0; i<$NUM_RUNS; i++ )); do
    cat $q >> queries.coga
  done
  bash generate_variants.sh
  RET=$?
  if [ $RET -ne 0 ]; then
    echo "Fatal Error occured while executing experiments for query: $query_name"
    exit -1
  fi
  mv experimental_results experimental_results_all_queries/"$query_name"
done

DATE=$(date | sed -e 's/ /_/g' | sed -e 's/\:/_/g')
mv experimental_results_all_queries falcon_paper_experimental_results-$DATE-$HOSTNAME
#mkdir -p falcon_paper_experimental_results-$DATE-$HOSTNAME
zip -r falcon_paper_experimental_results-$DATE-$HOSTNAME.zip falcon_paper_experimental_results-$DATE-$HOSTNAME





