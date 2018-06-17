
NUM_RUNS=6

rm -rf experimental_results_all_queries
mkdir -p experimental_results_all_queries
mkdir -p config

rm -rf PATH_TO_DATABASE
cp -f PATH_TO_DATABASE_SSB PATH_TO_DATABASE

for q in ssb_queries/*.coga; do
  test -f "$q" || continue
  query_name=$(echo $q | sed -e 's/ssb_queries\///g' -e 's/\.coga//g')
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

rm -rf PATH_TO_DATABASE
cp -f PATH_TO_DATABASE_TPCH PATH_TO_DATABASE

for q in tpch_queries/*.coga; do
  test -f "$q" || continue
  query_name=$(echo $q | sed -e 's/tpch_queries\///g' -e 's/\.coga//g')
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
