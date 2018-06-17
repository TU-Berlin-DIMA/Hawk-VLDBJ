#!/bin/bash

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

COGADB_RELEASE_NAME="$1"

bash create_release.sh "$COGADB_RELEASE_NAME"

rm -rf cogadb_installer.run
touch cogadb_installer.run
echo '
dir=`dirname $0`;
if [ x$dir = "x." ]
then
    dir=`pwd`
fi
base=`basename $0`;
command -v lsb_release >/dev/null 2>&1 || { echo >&2 "Installer requires tool lsb_release but it is not installed. This installer supports Ubuntu only, but you appear to run on another distribution. Aborting."; exit 1; }

OS=$(lsb_release -si)
VER=$(lsb_release -sr)

if [[ "$OS" != Ubuntu ]]; then
    echo "Warning: found Unsupported OS: $OS I can try to install anyway, but it is likely I fail. Should I continue to install? [yes,no]:"
    yes_no_input=""
    while [[ $yes_no_input != yes && $yes_no_input != no ]]; do
	read yes_no_input in
   done
   if [[ $yes_no_input == no ]]; then
       echo "Exit installer..."
       exit -1;
   fi
fi

if [[ "$VER" == 14.04 || "$VER" == 14.10 ]]; then
   echo "Detected supported OS: $OS $VER"
else
   echo "Detected unsupported Ubuntu Version: $OS $VER"
   echo "Continue anyway? [yes,no]:"
   yes_no_input=""
    while [[ $yes_no_input != yes && $yes_no_input != no ]]; do
	read yes_no_input in
   done
   if [[ $yes_no_input == no ]]; then
       echo "Exit installer..."
       exit -1;
   fi

fi

printf "Enter the base directory for installation\n"
printf "> "
read path in
printf "CoGaDB will be installed in "$path"/$COGADB_RELEASE_NAME\n"
mkdir -p "$path"

yes_no_input=""
while [[ $yes_no_input != yes && $yes_no_input != no ]]; do
	printf "Automatically install dependencies of CoGaDB in your system (requires root) [yes,no]:"
	read yes_no_input in
done
if [[ $yes_no_input == yes ]]; then
' >> cogadb_installer.run

cat install_cogadb_dependencies.sh >> cogadb_installer.run

echo 'fi
(cd $path; uudecode "$dir/$base"; tar xzfv '"$COGADB_RELEASE_NAME"'.tar.gz)
mkdir -p "$path/'"$COGADB_RELEASE_NAME"'/release_build"
printf "Enter the base directory for where the databases should be kept:\n"
printf "> "
read path_to_database in
printf "Enter the name of the database:\n"
printf "> "
read database_name in
mkdir -p "$path/release_build/startup.coga"
echo "set path_to_database=$path_to_database/$database_name
loaddatabase
listen 8000" > "$path/'"$COGADB_RELEASE_NAME"'/release_build/startup.coga"
echo "You can customize CoGaDBs behavior by editing the following file: $path/'"$COGADB_RELEASE_NAME"'/release_build/startup.coga"
(cd "$path/'"$COGADB_RELEASE_NAME"'/release_build"; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_VERBOSE_MAKEFILE=ON ..; make -j4)
echo "cd $path/'"$COGADB_RELEASE_NAME"'/release_build; ./cogadb/bin/cogadbd" > "$path/'"$COGADB_RELEASE_NAME"'/release_build/start_cogadb.sh"
echo "CoGaDB is now ready for use! You can start CoGaDB by executing the following script: $path/'"$COGADB_RELEASE_NAME"'/release_build/start_cogadb.sh"
echo "You can access CoGaDB by using the following command: netcat localhost 8000"
exit 0;' >> cogadb_installer.run

uuencode ../releases/"$COGADB_RELEASE_NAME".tar.gz "$COGADB_RELEASE_NAME".tar.gz  >> cogadb_installer.run

echo "Successfully create installer!"

exit 0
