#!/bin/bash

###########################################
# FUNCTIONS
###########################################

function print_preamble {

echo "\documentclass{article}
\usepackage{pgfplots, pgfplotstable}
\usepackage{filecontents}
\usetikzlibrary{arrows, decorations.markings}
\usetikzlibrary{positioning}
\usetikzlibrary{patterns}
\pgfplotscreateplotcyclelist{barplot cycle list}{
	{black,fill=black},
    {black,fill=gray!30!white}, %,postaction={pattern=sixpointed stars,pattern color=black}},%
    {black,fill=white},
	{black,fill=white,postaction={pattern=north east lines,pattern color=black}},
    {black,fill=white,postaction={pattern=north west lines,pattern color=black}},
    {black,fill=white,postaction={pattern=crosshatch,pattern color=black}},
    {black,fill=white,postaction={pattern=crosshatch dots,pattern color=black}},
}"

}

function print_file_content {
file_content=$(basename "$1")
echo "\begin{filecontents*}{$file_content}"
echo "id" > tmp_id
echo "0" >> tmp_id
paste tmp_id "$1"
echo "\end{filecontents*}"
rm tmp_id
}

function print_upper_plot {
TITLE=$(echo "$1" | sed -e 's/\_/-/g')
QUERYNAME=$(echo "$2" | sed -e 's/\_/-/g')
echo "
\pgfplotsset{compat=1.7}
\begin{document}

\title{$TITLE (query: $QUERYNAME, machine: $HOSTNAME)}
\maketitle

\begin{tikzpicture}[font=\scriptsize]
  \begin{axis}[ %grid=major, 
    ybar=0pt,
    ymin=0,
    xticklabel=\empty,
    xtick={\empty},
    ylabel={Time in ms},
    area legend,
    legend style={font=\scriptsize},
    bar width=15pt,
    legend style={at={(1.0,-1.0)},
    anchor=south,legend columns=1},
    scaled y ticks={real:0.001},
    ytick scale label code/.code={},
    nodes near coords={\pgfmathprintnumber[precision=2,zerofill,fixed]{\pgfplotspointmeta}},
    cycle list name=barplot cycle list
]"

}

function print_lower_plot {

echo "  \end{axis}
\end{tikzpicture}
\end{document}
"

}

function print_plot_file_command {

echo "    \addplot+[
        error bars/.cd,
        y dir=both,
        y explicit
    ]
    table[
            x=id,
            y=average,
            y error=standard_deviation
    ]
    {$1};"

}

if [ $# -lt 1 ]; then
	echo 'Missing parameter!'
	echo "Usage: $0 [CSV Files to plot]"
	exit -1
fi

###########################################
# CONFIG
###########################################

PLOT_GENERATOR_OUTPUT_FILE_NAME=$(cat config/PLOT_GENERATOR_OUTPUT_FILE_NAME)
PLOT_GENERATOR_OUTPUT_FILE_TITLE=$(cat config/PLOT_GENERATOR_OUTPUT_FILE_TITLE)
QUERY_NAME=$(cat config/QUERY_NAME)

TARGET_FILE="$PLOT_GENERATOR_OUTPUT_FILE_NAME.tex"
TARGET_PDF_FILE=$(echo "$TARGET_FILE" | sed -e 's/\.tex/.pdf/g')

###########################################
# CODE GENERATION SCRIPT
###########################################

#cleanup in case we have left over files (e.g., after aborts)
rm -f FILE_CONTENTS
rm -f PLOT_COMMANDS
rm -f LEGEND_COMMAND 
touch FILE_CONTENTS
touch PLOT_COMMANDS
touch LEGEND_COMMAND 
echo "\legend{" > LEGEND_COMMAND 

#generate file specific code
for var in "$@"
do
    echo "$var"
    base_var=$(basename "$var")
    print_file_content "$var" >> FILE_CONTENTS
    echo "File content: '"
    cat FILE_CONTENTS
    echo "'"
    print_plot_file_command "$base_var" >> PLOT_COMMANDS
    legend_name=$(basename "$var" | sed -e 's/\.csv//g'  | sed -e 's/\_/-/g')
    echo "$legend_name, " >> LEGEND_COMMAND 
done

echo "}" >> LEGEND_COMMAND 

#generated latex code
print_preamble > "$TARGET_FILE"
cat FILE_CONTENTS >> "$TARGET_FILE"
print_upper_plot $PLOT_GENERATOR_OUTPUT_FILE_TITLE $QUERY_NAME >> "$TARGET_FILE"
cat PLOT_COMMANDS >> "$TARGET_FILE"
cat LEGEND_COMMAND >> "$TARGET_FILE"
print_lower_plot  >> "$TARGET_FILE"

#cleanup
rm FILE_CONTENTS
rm LEGEND_COMMAND
rm PLOT_COMMANDS

#build generated latex code
mkdir -p plots/"$PLOT_GENERATOR_OUTPUT_FILE_NAME"
mv "$TARGET_FILE" plots/"$PLOT_GENERATOR_OUTPUT_FILE_NAME"/
cd plots/"$PLOT_GENERATOR_OUTPUT_FILE_NAME"
pdflatex "$TARGET_FILE" 
#&& evince "$TARGET_PDF_FILE"

exit 0

