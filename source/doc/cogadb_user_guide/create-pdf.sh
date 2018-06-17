#!/bin/bash
# Creates a PDF-Version of this How To-Guide
templateName="default.latex.cogadb"
logoName="cogadb-logo.png"

template="$(dirname $0)/templates/$templateName"
logo="$(dirname $0)/img/$logoName"

pandoc -s -S --toc --variable cogadblogo=$logo  --template $template -o CoGaDB-User-Guide.pdf cogadb_user_guide.md
