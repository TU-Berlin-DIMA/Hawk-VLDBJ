#!/bin/bash
###############################################################
#   Bash script acting as front-end to delete a database in
#   the cogadb database farm.
#
#   v.0.2 16.12.15 Treinat
################################################################
cogadir="$HOME/.cogadb"
dbfarmdir="$cogadir/database_farm"
databaseprefix="database_"

if [ $# -ne 1 ]; then
    echo "usage cogadb_createdb [nameOfNewDatabase]"
    exit 1
fi

databaseDir="$dbfarmdir/$databaseprefix$1"

if [ ! -d $databaseDir ]; then
    echo "A cogadb database with the name $1 does not exists in the database farm $dbfarmdir"
    exit 1
fi

##TODO now we just to delete the directory and also change the config port mapping file. The database-data itself does net get deleted so far
rm -R $databaseDir

if [ $? -ne 0 ]; then
    echo "error: could not delete database with the name $1"
    exit 1
else
    echo "successfully deleted database $1"
    exit 0
fi
