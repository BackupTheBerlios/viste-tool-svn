CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
PROJECT( ImageTypeConverterPlugin )

SET( BMIA_IMAGETYPECONVERTER_PLUGIN )

FIND_PACKAGE( VTK REQUIRED )
INCLUDE( ${VTK_USE_FILE} )

FIND_PACKAGE( Qt4 REQUIRED )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	)

SET( SRCS
	ImageTypeConverterPlugin.cxx
	ImageTypeConverterPlugin.h
	)

QT4_WRAP_CPP( MOC_SRCS
	ImageTypeConverterPlugin.h
	)

ADD_LIBRARY( ImageTypeConverterPlugin SHARED
	${SRCS} ${MOC_SRCS}
	)
    
TARGET_LINK_LIBRARIES( ImageTypeConverterPlugin
	bmia_plugin
	)
