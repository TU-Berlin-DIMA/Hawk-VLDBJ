#!/bin/bash

#COGADB_RELEASE_NAME="cogadb"

if [ $# -lt 1 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 <COGADB_RELEASE_NAME>"
	exit -1
fi

if [ $# -gt 1 ]; then
	echo 'To many parameters!'
	echo "Usage: $0 <COGADB_RELEASE_NAME>" 
	exit -1
fi

COGADB_RELEASE_NAME=$1

cd ..
mkdir -p releases
hg archive releases/"$COGADB_RELEASE_NAME"
#cd cogadb
#hg archive ../releases/"$COGADB_RELEASE_NAME"/cogadb
#cd ../hype-library/
#hg archive ../releases/"$COGADB_RELEASE_NAME"/hype-library
cd releases/"$COGADB_RELEASE_NAME"
rm -rf GPUDBMS/ Latex/
cd ..
tar cvfz "$COGADB_RELEASE_NAME".tar.gz "$COGADB_RELEASE_NAME"


echo "Created Source Tarball: '$PWD/releases/$COGADB_RELEASE_NAME.tar.gz'"

exit 0
