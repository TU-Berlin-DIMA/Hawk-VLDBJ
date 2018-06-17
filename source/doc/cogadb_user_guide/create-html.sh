#!/bin/bash
# Creates a HTML-Version of this How To-Guide

pandoc -s --toc -c css/foghorn.css -o CoGaDB-User-Guide.html cogadb_user_guide.md
