#!/bin/bash

program_exist() {
	command -v $@ >/dev/null 2>&1
	return $?
}

DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COGADB_SOURCE_DIR="$DIRECTORY"/..

if program_exist clang-format; then
	CLANG_FORMAT_NAME=clang-format
elif program_exist clang-format-3.6; then
	CLANG_FORMAT_NAME=clang-format-3.6
else
	echo "No clang-format found!"
	exit 1
fi

echo "Formatting the directory '$COGADB_SOURCE_DIR'"

RUNS=10

if [[ $# -eq 1 ]]; then
	RUNS=${1}
fi

echo "We will run the formatter $RUNS time(s)"

i=0
while [ $i -lt $RUNS ]; do
	find "$COGADB_SOURCE_DIR/lib" \( -name '*.h' -or -name '*.hpp' \) -print0 | xargs -0 $CLANG_FORMAT_NAME -style="{BasedOnStyle: Google,  Language: Cpp,  NamespaceIndentation: All}" -i
	find "$COGADB_SOURCE_DIR/lib" \( -name '*.c' -or -name '*.cpp' -or -name '*.cu' \) -print0 | xargs -0 $CLANG_FORMAT_NAME -style="{BasedOnStyle: Google,  Language: Cpp,  NamespaceIndentation: None}" -i

	find "$COGADB_SOURCE_DIR/test/integration_tests" \( -name '*.h' -or -name '*.hpp' \) -print0 | xargs -0 $CLANG_FORMAT_NAME -style="{BasedOnStyle: Google,  Language: Cpp,  NamespaceIndentation: All}" -i
	find "$COGADB_SOURCE_DIR/test/integration_tests" \( -name '*.c' -or -name '*.cpp' -or -name '*.cu' \) -print0 | xargs -0 $CLANG_FORMAT_NAME -style="{BasedOnStyle: Google,  Language: Cpp,  NamespaceIndentation: None}" -i

	find "$COGADB_SOURCE_DIR/test/experiments" \( -name '*.h' -or -name '*.hpp' \) -print0 | xargs -0 $CLANG_FORMAT_NAME -style="{BasedOnStyle: Google,  Language: Cpp,  NamespaceIndentation: All}" -i
	find "$COGADB_SOURCE_DIR/test/experiments" \( -name '*.c' -or -name '*.cpp' -or -name '*.cu' \) -print0 | xargs -0 $CLANG_FORMAT_NAME -style="{BasedOnStyle: Google,  Language: Cpp,  NamespaceIndentation: None}" -i
	
	let i=i+1
done

exit 0
