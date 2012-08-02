OPTION(PLUGINS_BUILD_FIBERSTOVOXELS "Build Fibers to Voxels Plugin" ON)
IF(PLUGINS_BUILD_FIBERSTOVOXELS)

CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
PROJECT( FibersToVoxelsPlugin )

SET( BMIA_FIBERSTOVOXELS_PLUGIN )

FIND_PACKAGE( VTK REQUIRED )
INCLUDE( ${VTK_USE_FILE} )

FIND_PACKAGE( Qt4 REQUIRED )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../../../stable/DTIReader
	)

LINK_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}/../../../../stable/DTIReader/Debug
	)

SET( SRCS
	FibersToVoxelsPlugin.cxx
	FibersToVoxelsPlugin.h
	vtkMathExtensions.cxx
	vtkMathExtensions.h
	vtkStreamlineToVoxelDensity.cxx
	vtkStreamlineToVoxelDensity.h
#    vtkFiberSortFilter.cxx
#    vtkFiberSortFilter.h
	)

QT4_WRAP_CPP( MOC_SRCS
	FibersToVoxelsPlugin.h
	)

ADD_LIBRARY( FibersToVoxelsPlugin SHARED
	${SRCS} ${MOC_SRCS}
	)
    
TARGET_LINK_LIBRARIES( FibersToVoxelsPlugin
	DTIReaderPlugin
	bmia_plugin
    bmia_libs
	)

ENDIF(PLUGINS_BUILD_FIBERSTOVOXELS)
