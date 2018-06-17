###
### Find GPU Ocelot, using OcelotConfig if available
### Defines:
###   Ocelot_LIBRARY
###   Ocelot_VERSION
###

find_program(OcelotConfig_EXECUTABLE OcelotConfig)
if (OcelotConfig_EXECUTABLE)
	execute_process(COMMAND ${OcelotConfig_EXECUTABLE} --version
			OUTPUT_VARIABLE Ocelot_VERSION
			OUTPUT_STRIP_TRAILING_WHITESPACE)

	execute_process(COMMAND ${OcelotConfig_EXECUTABLE} --libdir
			OUTPUT_VARIABLE OcelotConfig_LIBDIR
			OUTPUT_STRIP_TRAILING_WHITESPACE)
endif ()

find_library(Ocelot_LIBRARY NAMES ocelot libocelot
	     HINTS ${OcelotConfig_LIBDIR})
set(Ocelot_LIBRARIES ${Ocelot_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ocelot REQUIRED_VARS Ocelot_LIBRARY
				  VERSION_VAR Ocelot_VERSION)

mark_as_advanced(Ocelot_LIBRARY Ocelot_VERSION)
