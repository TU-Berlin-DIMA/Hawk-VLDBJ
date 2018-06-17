
# LibXslt package only required for xsltproc tool
find_package(LibXslt)
if (NOT LIBXSLT_XSLTPROC_EXECUTABLE)
	message(WARNING "xsltproc not found. Will not generate cogadb-doc target!")
	return ()
endif ()

# Generate XHTML version of the SQL grammar
add_custom_command(OUTPUT sql_grammar.xhtml COMMAND
		   ${LIBXSLT_XSLTPROC_EXECUTABLE} --output "${CMAKE_CURRENT_BINARY_DIR}/sql_grammar.xhtml"
		   "${BISON_DATA_DIR}/xslt/xml2xhtml.xsl" "${CMAKE_CURRENT_BINARY_DIR}/sql_grammar.xml"
		   DEPENDS SQLParser)
			
find_package(Doxygen)
if (NOT ${DOXYGEN_FOUND})
	message(WARNING "Doxygen not found. Will not generate cogadb-doc target!")
	return ()
endif ()

if (${DOXYGEN_DOT_FOUND})
	set(DOXYGEN_HAVE_DOT YES)
else ()
	set(DOXYGEN_HAVE_DOT NO)
endif ()

execute_process(COMMAND ${DOXYGEN_EXECUTABLE} --version
		OUTPUT_VARIABLE DOXYGEN_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Doxygen version: ${DOXYGEN_VERSION}")

# Do not use the layout file with older Doxygens since its syntax
# changed and older Doxygens crash with the current one.
# FIXME: I don't know up to which version exactly is broken
if (${DOXYGEN_VERSION} VERSION_GREATER 1.7.4)
	set(DOXYGEN_LAYOUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/layout.xml")
endif ()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
	       "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
	       ESCAPE_QUOTES)

add_custom_target(cogadb-doc ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
		  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
			  "${CMAKE_CURRENT_BINARY_DIR}/sql_grammar.xhtml")
# FIXME: cleaning directories this way is broken in older CMake versions
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES
			 "${CMAKE_CURRENT_BINARY_DIR}/documentation")

# NOTE: hack to build doc target when installing
#install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\"
#			      --build \"${PROJECT_BINARY_DIR}\" --target cogadb-doc)")
#install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/documentation/html/"
#	DESTINATION share/doc/cogadb)


#generating the pdf reference manual
find_package(LATEX)
if (NOT (LATEX_COMPILER AND PDFLATEX_COMPILER AND
	 MAKEINDEX_COMPILER AND BIBTEX_COMPILER))
	message(WARNING "Latex not found. Will not generate cogadb-doc-print target!")
	return ()
endif ()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.print.in"
	       "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.print"
	       ESCAPE_QUOTES)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/documentation/latex")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/img/cogadb_logo_new.pdf"
	       "${CMAKE_CURRENT_BINARY_DIR}/documentation/latex/cogadb_logo_new.pdf"
	       COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/img/cogadb_architecture.pdf"
	       "${CMAKE_CURRENT_BINARY_DIR}/documentation/latex/cogadb_architecture.pdf"
	       COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/additional_documentation/literature.bib"
	       "${CMAKE_CURRENT_BINARY_DIR}/documentation/latex/literature.bib"
	       COPYONLY)

# FIXME: see additional_documentation/latex/Makefile: multiple pdflatex runs
# may be necessary
add_custom_target(cogadb-doc-print
		  COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.print"
		  COMMAND ${PDFLATEX_COMPILER} refman
		  COMMAND ${MAKEINDEX_COMPILER} refman
		  COMMAND ${BIBTEX_COMPILER} refman
		  COMMAND ${PDFLATEX_COMPILER} refman
		  COMMAND ${PDFLATEX_COMPILER} refman
		  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/documentation/latex"
		  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.print")
