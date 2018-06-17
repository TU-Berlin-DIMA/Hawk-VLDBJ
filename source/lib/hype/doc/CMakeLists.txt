
if (CMAKE_BUILD_TYPE MATCHES "Rel")
	set(DOXYFILE Doxyfile.release)
else ()
	set(DOXYFILE Doxyfile)
endif ()

find_package(Doxygen)
if (${DOXYGEN_DOT_FOUND})
	set(DOXYGEN_HAVE_DOT YES)
else ()
	set(DOXYGEN_HAVE_DOT NO)
endif ()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${DOXYFILE}.in"
	       "${CMAKE_CURRENT_BINARY_DIR}/${DOXYFILE}"
	       ESCAPE_QUOTES)

if (NOT ${DOXYGEN_FOUND})
	message(WARNING "Doxygen not found. Will not generate hype-doc target!")
	return ()
endif ()

add_custom_target(hype-doc ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/${DOXYFILE}"
		  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${DOXYFILE}")
# FIXME: cleaning directories this way is broken in older CMake versions
set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES
			 "${CMAKE_CURRENT_BINARY_DIR}/documentation")

# NOTE: hack to build doc target when installing
#install(CODE "execute_process(COMMAND \"${CMAKE_COMMAND}\"
#			      --build \"${PROJECT_BINARY_DIR}\" --target hype-doc)")
#install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/documentation/html/"
#	DESTINATION share/doc/hype)

# Print (PDF) version building:

find_package(LATEX)
if (NOT (LATEX_COMPILER AND PDFLATEX_COMPILER AND
	 MAKEINDEX_COMPILER AND BIBTEX_COMPILER))
	message(WARNING "Latex not found. Will not generate hype-doc-print target!")
	return ()
endif ()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.print.in"
	       "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.print"
	       ESCAPE_QUOTES)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/documentation/latex")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/img/hype_logo_300width.png"
	       "${CMAKE_CURRENT_BINARY_DIR}/documentation/latex/hype_logo_300width.png"
	       COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/additional_documentation/literature.bib"
	       "${CMAKE_CURRENT_BINARY_DIR}/documentation/latex/literature.bib"
	       COPYONLY)

# FIXME: see additional_documentation/latex/Makefile: multiple pdflatex runs
# may be necessary
add_custom_target(hype-doc-print
		  COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.print"
		  COMMAND ${PDFLATEX_COMPILER} refman
		  COMMAND ${MAKEINDEX_COMPILER} refman
		  COMMAND ${BIBTEX_COMPILER} refman
		  COMMAND ${PDFLATEX_COMPILER} refman
		  COMMAND ${PDFLATEX_COMPILER} refman
		  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/documentation/latex"
		  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.print")
