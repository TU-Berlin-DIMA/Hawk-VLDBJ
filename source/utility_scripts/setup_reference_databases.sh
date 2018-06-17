#!/bin/bash

set -e

echo "Welcome to the CoGaDB reference database setup assistant!"
echo "Please type the absolute path where the reference databases should be located:"
read DATABASE_PATH

if [ ! -d "$DATABASE_PATH" ]; then
  echo "Error: Directory '$DATABASE_PATH' does not exist!"
  echo "Aborting script..."
  exit -1
fi

if [ ! -w "$DATABASE_PATH" ]; then 
  echo "Error: Directory '$DATABASE_PATH' is not writable!"
  echo "Aborting script..."  
  exit -1
fi

if [[ "$DATABASE_PATH" == *[' ']* ]]; then
  echo "Error: Directory '$DATABASE_PATH' contains at least one whitespace, which is forbidden!"
  echo "Aborting script..."    
  exit -1
fi

DATABASE_PATH=$(echo "$DATABASE_PATH" | sed s'/\/*$//')
echo "$DATABASE_PATH"

cd "$DATABASE_PATH"
wget https://tubcloud.tu-berlin.de/index.php/s/lQYqGBRjBz9TNQC/download
mv download cogadb_reference_databases_v1.tar.gz
tar xvfz cogadb_reference_databases_v1.tar.gz 
mv cogadb_reference_databases cogadb_reference_databases_v1
mkdir -p "$HOME/.cogadb"
echo "" > "$HOME/.cogadb/test_config.coga"
echo "set path_to_ssb_sf1_database=$DATABASE_PATH/cogadb_reference_databases_v1/ssb_sf1" >> "$HOME/.cogadb/test_config.coga"
echo "set path_to_tpch_sf1_database="$DATABASE_PATH"/cogadb_reference_databases_v1/tpch_sf1" >> "$HOME/.cogadb/test_config.coga"

echo "============================================================================================"
echo "Successfully Setup Reference Databases!"
echo ""
echo "PATH_TO_REFERENCE DATABASES: '$DATABASE_PATH'"
echo "PATH_TO_GLOBAL_CONFIG_FILE: '$HOME/.cogadb/test_config.coga'"
echo ""
echo "Content of Global Config File: "
cat "$HOME/.cogadb/test_config.coga" 
echo ""
echo "To execute tests, go to your CoGaDB build directory and type 'make test'."

exit 0
