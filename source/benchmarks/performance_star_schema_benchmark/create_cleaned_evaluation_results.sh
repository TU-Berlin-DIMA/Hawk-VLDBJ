
if [ $# -lt 1 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 <name of experiment directory to clean>"
	exit -1
fi

RESULT_DIRECTORY="$1"

CLEANED_RESULT_DIRECTORY="$RESULT_DIRECTORY"_cleaned

for file in `find $RESULT_DIRECTORY -name "*.csv"` `find $RESULT_DIRECTORY -name "*.gnuplot"` `find $RESULT_DIRECTORY -name "*.pdf"`; do

	file_in_new_dir=$(echo "$file" | sed -e 's/'$RESULT_DIRECTORY'/'$CLEANED_RESULT_DIRECTORY'/g')
	mkdir -p $PWD/$(dirname $file_in_new_dir)
	cp "$file" "$file_in_new_dir"

done

zip -r "$CLEANED_RESULT_DIRECTORY".zip "$CLEANED_RESULT_DIRECTORY"


