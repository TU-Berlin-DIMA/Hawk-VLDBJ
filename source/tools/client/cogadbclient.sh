#!/bin/bash
###############################################################
#   Bash script acting as a client to the CoGaDB
#   Daemon "cogadbd"
#
#   v.0.5 22.12.15 Treinat
################################################################

cogadbd="/home/florian/projects/gpudbms/debug_build/cogadb/bin/cogadbd"
cogadir=$HOME"/.cogadb"
localStartupFile=$(pwd)"/startup.coga"
databaseprefix="database_"
dbfarmdir="$cogadir/database_farm"
startupFile=startup.coga
cogaStartUpFile=""
chosenDatabase=""

function createCogaDbConfigDir {

    mkdir $cogadir
    if [ "$?" -eq 0 ]
    then
        echo "successfully created new directory $cogadir"

        globalStartUpFile=$cogadir"/startup.coga"
        touch $globalStartUpFile

        if [ "$?" -ne 0 ]
        then
            echo "could not create a global startup file"
            exit 1
        fi

        echo "Creating global startup file for CoGaDB. Please enter the path to the database you want to use: "
        read globalDatabase

        if [ -z $globalDatabase ]
        then
            echo "something went wrong"
            exit 2;
        fi

        echo "Please enter the port number the global database should be listen to:"
        read portNumber

        if [ $portNumber -le 0 ]
        then
            echo "wrong portnumber"
            exit 1
        fi

        echo "set path_to_database=$globalDatabase" > $globalStartUpFile
        echo "loaddatabase " >> $globalStartUpFile
        echo "listen $portNumber" >> $globalStartUpFile
        echo "successfully created new global startup-file for CoGaDB. Starting CoGaDB..."
        $cogadbd $globalStartUpFile
    else
        echo "could not create new directory $cogaddir"
        exit 1
    fi
}

function checkLocalOrGlobal {
    if [ -f $startupFile ]
    then
        echo "We have a local start up file and start cogadb normally"
        $cogadbd $startupFile
    else
        echo "Could not find any local startup file --> look in configdir" $cogadir

        if [ -d $cogadir ]
        then
            startupFile=$cogadir/startup.coga
            if [ -f $startupFile ]
            then
                echo "Found global config file"
                $cogadbd $startupFile
            else
                echo "Could not found neither a local nor a global config file for cogadb"
                exit 0
            fi
        else
            echo ".cogadir does not exists. Creating it"
            createCogaDbConfigDir
        fi
    fi
}

function connectTo {
    cogadbListenPort=$(grep listen $1 | cut -d" " -f2)
    if [ -z $cogadbListenPort ]
    then
        echo "could not find any listen port for cogadb in startup file $1"
        exit 2
    fi

    echo "Cogadb listening on port $cogadbListenPort. Connecting to it..."
    nc 127.0.0.1 $cogadbListenPort
    exit 0
}

function startCogadbAndConnectTo {
    echo "generic function to start cogadbd with polling for pid file"

    databaseDir="$dbfarmdir/$databaseprefix$1"
    databaseStartUpFile="$databaseDir/startup.coga"
    pidfile="$databaseDir/pid"

    #start cogadb
    nohup $cogadbd $databaseStartUpFile &>/dev/null &

    echo "searching for pidfile $pidfile (wait for cogadb to finish loading initial database files)"
    # wait for creation of pidfile
    #TODO for now we wait forever if the file will not be created
    # we have to change this into a finite waiting
    while [ ! -f $pidfile ]; do sleep 1; done

    connectTo $databaseStartUpFile
}

function checkForRunningInstance {
    databaseStartUpFile=$cogadir"/startup_$1.coga"
    pidfile="$cogadir/$1.pid"

    if [ -f $pidfile ]
    then
        pid=$(cat $pidfile)
        echo "verify if pid $pid is cogadbd"
        ps ax | grep -q "$pid.*cogadb[d]"

        if [ $? -eq 0 ]
        then
            #running pid is cogadbd now try to connect
            connectTo $databaseStartUpFile
        else
            #the specified pid is not cogadb
            echo "$pid is no running cogadb instance"
            rm $pidfile

            startCogadbAndConnectTo $1
        fi
    else
        startCogadbAndConnectTo $1
    fi

    return 0
}

function killCogadb {

    databaseDir="$dbfarmdir/$databaseprefix$1"
    databaseStartUpFile="$databaseDir/startup.coga"
    pidfile="$databaseDir/pid"

    if [ -d $databaseDir ] && [ -f $databaseStartUpFile ]; then

        if [ -f $pidfile ]; then
            pid=$(cat $pidfile)
            echo "verify if pid $pid is cogadbd"
            ps ax | grep -q "$pid.*cogadb[d]"

            if [ $? -eq 0 ]
            then
                kill -15 $pid
                return $?
            else
                #the specified pid is not cogadb
                echo "$pid is no running cogadb instance"
                rm $pidfile
                return 1
            fi
        else
            return 1
        fi
    else
        return 1
    fi
}

function checkIfDatabaseIsRunning {

    databaseDir="$dbfarmdir/$databaseprefix$1"
    databaseStartUpFile="$databaseDir/startup.coga"
    pidfile="$databaseDir/pid"

    if [ -d $databaseDir ] && [ -f $databaseStartUpFile ]; then

        if [ -f $pidfile ]; then
            pid=$(cat $pidfile)
            echo "verify if pid $pid is cogadbd"
            ps ax | grep -q "$pid.*cogadb[d]"

            if [ $? -eq 0 ]
            then
                return 0
            else
                #the specified pid is not cogadb
                echo "$pid is no running cogadb instance"
                rm $pidfile
                return 1
            fi
        else
            return 1
        fi
    else
        return 1
    fi

}

function checkIfDatabaseExists {

    databaseDir="$dbfarmdir/$databaseprefix$1"
    databaseStartUpFile="$databaseDir/startup.coga"

    if [ -d $databaseDir ] && [ -f $databaseStartUpFile ]; then
        return 0
    else
        return 1
    fi

}

function startDatabase {
    echo "startDatabase called"
    if [ -z $1 ]; then
        echo "no database was specified"
        exit 5;
    fi

    checkIfDatabaseExists $1
    if [ $? -ne 0 ]; then
        echo "there does not exist a database named $1"
        exit 6
    fi

    checkIfDatabaseIsRunning $1

    if [ $? -eq 0 ]; then
        echo "database $1 is already running"
        exit 0
    else
        databaseDir="$dbfarmdir/$databaseprefix$1"
        databaseStartUpFile="$databaseDir/startup.coga"

        if [ -d $databaseDir ] && [ -f $databaseStartUpFile ]; then
            echo "starting cogadb with database $1"
            echo $cogadbd $databaseStartUpFile
            startCogadbAndConnectTo $1
            exit 0
        else
            echo "error while trying to start cogadb with database $1"
            exit 5
        fi
    fi
}

function listAllDatabases {

    databases=($(find ~/.cogadb/database_farm/* -prune -nowarn -type d -name "database_*" -exec basename {} \; 2> /dev/null))

    echo -e "List of databases:\n"

    if [ "${#databases[@]}" -eq 0 ]; then
        echo "no existing database found"
    else
        number=1
        for i in ${databases[@]}
        do
            cuttedName=$(echo $i | cut -d _ -f2)

            checkIfDatabaseIsRunning $cuttedName
            if [ $? -eq 0 ]; then
                echo "database is running"
                running="(running)"
            else
                echo "database is not running"
                running=""
            fi

            printf "%d. Database: %s %s\n\n\t\tconfig:\n" $number $cuttedName $running

            while read l; do
                echo -e "\t\t\t$l"
            done < "$dbfarmdir/$i/startup.coga"
            echo -e "\n\n"
            number=$((number+1))
        done
    fi
    exit 0
}

function stopDatabase {
    echo "stopDatabase called"
    if [ -z $1 ]; then
        echo "no database was specified"
        exit 5;
    fi

    checkIfDatabaseExists $1
    if [ $? -ne 0 ]; then
        echo "there does not exist a database named $1"
        exit 6
    fi

    checkIfDatabaseIsRunning $1

    if [ $? -eq 0 ]; then
        echo "stopping cogadb"
        killCogadb $1

        if [ $? -eq 0 ]; then
            echo "successfully stopped database $1"
            databaseDir="$dbfarmdir/$databaseprefix$1"
            databaseStartUpFile="$databaseDir/startup.coga"
            pidfile="$databaseDir/pid"
            # delete the pidfile
            rm $pidfile
            exit 0
        else
            echo "could not stop database $1"
            exit 11
        fi

    else
        echo "database $1 is not running"
        exit 0
    fi
}

while getopts :aD: opt
do
    case "$opt" in
        D) echo "Found the -D with value $OPTARG"
           chosenDatabase=$OPTARG;;
        *) echo "error. unknown option found: $opt"
           exit 1;;
    esac
done

shift $[ $OPTIND - 1]
while [ -n "$1" ]
do
    case "$1" in
        start) echo "we want to start something"
               shift
               startDatabase $1;;
        list)  listAllDatabases;;
        stop)  echo "we want to stop a database"
               shift
               stopDatabase $1;;
        *) echo "unknown argument found"
           exit 1;;
    esac
shift
done

if [ -z $chosenDatabase ]
then
    echo "no Database specified. check for local startup file or go global"
    checkLocalOrGlobal
else
    echo "chosen database is $chosenDatabase"

    checkIfDatabaseExists $chosenDatabase
    if [ $? -ne 0 ]; then
        echo "there does not exist a database named $chosenDatabase"
        exit 6
    fi

    databaseDir="$dbfarmdir/$databaseprefix$chosenDatabase"
    databaseStartUpFile="$databaseDir/startup.coga"

    checkIfDatabaseIsRunning $chosenDatabase

    if [ $? -eq 0 ]; then
        connectTo $databaseStartUpFile
    else
        if [ -d $databaseDir ] && [ -f $databaseStartUpFile ]; then
            echo "starting cogadb with database $chosenDatabase"
            startCogadbAndConnectTo $chosenDatabase
        else
            echo "error while trying to start cogadb with database $chosenDatabase"
            exit 5
        fi
    fi

    # databaseStartUpFile=$cogadir"/startup_$chosenDatabase.coga"
    # echo "trying to open $databaseStartUpFile"
    # if [ -f $databaseStartUpFile ]
    # then
    #     echo "Starting CoGaDB with database \"$chosenDatabase\""
    #     checkForRunningInstance $chosenDatabase
    #     exit 0
    # else
    #     echo "Could not found a database named \"$chosenDatabase\""
    #     exit 2
    # fi
fi
