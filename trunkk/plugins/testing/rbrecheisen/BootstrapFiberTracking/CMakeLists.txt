CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
PROJECT( BootstrapFiberTrackingPlugin )

SET( BMIA_BOOTSTRAPFIBERTRACKING_PLUGIN )

FIND_PACKAGE( VTK REQUIRED )
INCLUDE( ${VTK_USE_FILE} )

FIND_PACKAGE( Qt4 REQUIRED )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../../../stable/DTIReader
	${CMAKE_CURRENT_SOURCE_DIR}/../../../stable/DTIMeasures
	${CMAKE_CURRENT_SOURCE_DIR}/../../../stable/FiberTracking
#	${CMAKE_CURRENT_SOURCE_DIR}/../../../stable/FiberVisualization
	${CMAKE_CURRENT_SOURCE_DIR}/../../../../libs/TensorMath
	)

SET( SRCS
	BootstrapFiberTrackingPlugin.cxx
	BootstrapFiberTrackingPlugin.h
	vtkBootstrapFiberTrackingFilter.cxx
	vtkBootstrapFiberTrackingFilter.h
#	vtkBootstrapStreamlineToDistanceTableFilter.cxx
#	vtkBootstrapStreamlineToDistanceTableFilter.h
#	vtkBootstrapTensorFieldToStreamlineDistribution.cxx
#	vtkBootstrapTensorFieldToStreamlineDistribution.h
	)

QT4_WRAP_UI( UIC_SRCS
	BootstrapFiberTrackingPlugin.ui
	)

QT4_WRAP_CPP( MOC_SRCS
	BootstrapFiberTrackingPlugin.h
	)

ADD_LIBRARY( BootstrapFiberTrackingPlugin SHARED
	${SRCS} ${MOC_SRCS} ${UIC_SRCS}
	)
    
TARGET_LINK_LIBRARIES( BootstrapFiberTrackingPlugin
#	FiberVisualizationPlugin
	FiberTrackingPlugin
	DTIMeasuresPlugin
	DTIReaderPlugin
	bmia_plugin
	bmia_libs
	)
