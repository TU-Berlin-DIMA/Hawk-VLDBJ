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

if [ ! -f PATH_TO_COGADB_EXECUTABLE ]
then
    echo "File 'PATH_TO_COGADB_EXECUTABLE' not found!"
    echo "Please enter the path to the cogadb executable:"
    read LINE
    echo "$LINE" > PATH_TO_COGADB_EXECUTABLE
fi

if [ ! -f PATH_TO_DATABASE_SSB ]
then
    echo "File 'PATH_TO_DATABASE_SSB' not found!"
    echo "Please enter the path to the cogadb database:"
    read LINE
    echo "$LINE" > PATH_TO_DATABASE_SSB
fi

if [ ! -f PATH_TO_DATABASE_TPCH ]
then
    echo "File 'PATH_TO_DATABASE_TPCH' not found!"
    echo "Please enter the path to the cogadb database:"
    read LINE
    echo "$LINE" > PATH_TO_DATABASE_TPCH
fi

# SSB
rm -rf PATH_TO_DATABASE
cp -f PATH_TO_DATABASE_SSB PATH_TO_DATABASE

for q in ssb_queries/*.coga; do
  test -f "$q" || continue
  query_name=$(echo $q | sed -e 's/ssb_queries\///g' -e 's/\.coga//g')
  echo "$query_name" > config/QUERY_NAME
  echo -e "\n---------------------------------------------------------------"
  echo -e "Execute experiments for query: $query_name using $NUM_RUNS runs"

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

# TPCH
rm -rf PATH_TO_DATABASE
cp -f PATH_TO_DATABASE_TPCH PATH_TO_DATABASE

for q in tpch_queries/*.coga; do
  test -f "$q" || continue
  query_name=$(echo $q | sed -e 's/tpch_queries\///g' -e 's/\.coga//g')
  echo "$query_name" > config/QUERY_NAME
  echo -e "\n---------------------------------------------------------------"
  echo -e "Execute experiments for query: $query_name using $NUM_RUNS runs"

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
zip -qr falcon_paper_experimental_results-$DATE-$HOSTNAME.zip falcon_paper_experimental_results-$DATE-$HOSTNAME
bash ../send_success_mail.sh "Experiments successfully completed!"
