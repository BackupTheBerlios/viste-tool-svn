CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
PROJECT( BootstrapFibersToDistancesPlugin )

SET( BMIA_BOOTSTRAPFIBERSTODISTANCES_PLUGIN )

FIND_PACKAGE( VTK REQUIRED )
INCLUDE( ${VTK_USE_FILE} )

FIND_PACKAGE( Qt4 REQUIRED )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	)

SET( SRCS
	BootstrapFibersToDistancesPlugin.cxx
	BootstrapFibersToDistancesPlugin.h
	vtkDistanceMeasure.cxx
	vtkDistanceMeasure.h
	vtkDistanceMeasureClosestPointDistance.cxx
	vtkDistanceMeasureClosestPointDistance.h
	vtkDistanceMeasureEndPointDistance.cxx
	vtkDistanceMeasureEndPointDistance.h
	vtkDistanceMeasureMeanOfClosestPointDistances.cxx
	vtkDistanceMeasureMeanOfClosestPointDistances.h
	vtkBootstrapStreamlineToDistanceTableFilter.cxx
	vtkBootstrapStreamlineToDistanceTableFilter.h
	vtkDistanceTable.cxx
	vtkDistanceTable.h
#	vtkDistanceTableCollection.cxx
#	vtkDistanceTableCollection.h
	)

QT4_WRAP_UI( UIC_SRCS
	BootstrapFibersToDistancesPlugin.ui
	)

QT4_WRAP_CPP( MOC_SRCS
	BootstrapFibersToDistancesPlugin.h
	)

ADD_LIBRARY( BootstrapFibersToDistancesPlugin SHARED
	${SRCS} ${MOC_SRCS} ${UIC_SRCS}
	)
    
TARGET_LINK_LIBRARIES( BootstrapFibersToDistancesPlugin
	bmia_plugin vtkCommon
	)
