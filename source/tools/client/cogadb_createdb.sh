#!/bin/bash
###############################################################
#   Bash script acting as front-end to create a database in
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

newDatabaseDir="$dbfarmdir/$databaseprefix$1"

if [ -d $newDatabaseDir ]; then
    # directory already exists
    echo "a database with the name $1 already exists in $dbfarmdir"
    exit 1
fi

echo "Please enter the path to the database you want to use: "
read pathToDatabase

if [ -z $pathToDatabase ]
then
    echo "The path to the database must not be empty"
    exit 2;
fi

correctNumber=0

while [ $correctNumber -ne 1 ]
do
    echo "Please enter the port number the global database should be listen to:"
    read portNumber

    if [ $portNumber -le 0 ]
    then
        echo "wrong portnumber"
        exit 1
    fi

    allPorts=($(grep -h -r listen $dbfarmdir | cut -d" " -f2))
    correctNumber=1
    for i in ${allPorts[@]}
    do
        if [ $i -eq $portNumber ]; then
            correctNumber=0
            echo "the port $portNumber is already used from another database."
        fi
    done
done

mkdir $newDatabaseDir
if [ $? -ne 0 ]; then
    echo "error creating new database $newDatabaseDir"
    exit 1
fi

databaseStartUpFile="$newDatabaseDir/startup.coga"
echo "set path_to_database=$pathToDatabase" > $databaseStartUpFile
echo "loaddatabase " >> $databaseStartUpFile
echo "listen $portNumber" >> $databaseStartUpFile

echo "successfully created new cogadb database $1"
exit 0
