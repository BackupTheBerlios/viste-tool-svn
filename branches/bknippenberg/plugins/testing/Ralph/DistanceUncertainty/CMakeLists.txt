CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
PROJECT( DistanceUncertaintyPlugin )

SET( BMIA_DISTANCEUNCERTAINTY_PLUGIN )

FIND_PACKAGE( VTK REQUIRED )
INCLUDE( ${VTK_USE_FILE} )

FIND_PACKAGE( Qt4 REQUIRED )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}
	)

SET( SRCS
	DistanceUncertaintyPlugin.cxx
	DistanceUncertaintyPlugin.h
	)

QT4_WRAP_CPP( MOC_SRCS
	DistanceUncertaintyPlugin.h
	)

ADD_LIBRARY( DistanceUncertaintyPlugin SHARED
	${SRCS} ${MOC_SRCS} 
	)
    
TARGET_LINK_LIBRARIES( DistanceUncertaintyPlugin
	bmia_plugin
	)
