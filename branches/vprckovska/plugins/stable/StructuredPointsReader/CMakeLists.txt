OPTION(PLUGINS_BUILD_STRUCTUREDPOINTSREADER "Build Structured Points Reader Plugin" ON)

if(PLUGINS_BUILD_STRUCTUREDPOINTSREADER)

set( BMIA_STRUCTUREDPOINTSREADER_PLUGIN )

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(Qt4 REQUIRED)
include(${QT_USE_FILE})

set(SRCS
	StructuredPointsReaderPlugin.cxx
	StructuredPointsReaderPlugin.h
)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${BMIA_LIBS_SOURCE_DIR})

qt4_wrap_cpp( MOC_SRCS StructuredPointsReaderPlugin.h )

add_library( StructuredPointsReaderPlugin SHARED ${SRCS} ${MOC_SRCS} )
target_link_libraries( StructuredPointsReaderPlugin bmia_libs )

endif(PLUGINS_BUILD_STRUCTUREDPOINTSREADER)