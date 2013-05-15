OPTION(PLUGINS_BUILD_DISTANCEUNCERTAINTY "Build Distance Uncertainty Plugin (requires CUDA SDK)" ON)

if(PLUGINS_BUILD_DISTANCEUNCERTAINTY)

CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )
PROJECT( DistanceUncertaintyPlugin )

SET( BMIA_DISTANCEUNCERTAINTY_PLUGIN )

FIND_PACKAGE( VTK REQUIRED )
INCLUDE( ${VTK_USE_FILE} )

FIND_PACKAGE( Qt4 REQUIRED )
SET( QT_USE_QTOPENGL TRUE )
SET( QT_USE_QTXML TRUE )
INCLUDE( ${QT_USE_FILE} )

INCLUDE_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/../../../stable/RayCast
	${CMAKE_CURRENT_SOURCE_DIR}/../../../stable/examples/PolyDataVis
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/include"
	/usr/local/cuda/include
	/usr/X11/include
	/usr/include
	)

LINK_DIRECTORIES(
	${CMAKE_CURRENT_BINARY_DIR}/../../../../stable/RayCast/Debug
	${CMAKE_CURRENT_BINARY_DIR}/../../../../stable/examples/PolyDataVis/Debug
	"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v4.0/lib/Win32"
	/usr/local/cuda/lib64
	/usr/local/cuda/lib
	/usr/X11/lib
	/usr/lib
	)

SET( SRCS
    DistanceUncertaintyPlugin.cxx
    DistanceUncertaintyPlugin.h
    DistanceConfiguration.cxx
    DistanceConfiguration.h
    )

SET( PBA3D_SRCS
    PBA3D/binvoxreader.cpp
    PBA3D/binvoxwriter.cpp
    PBA3D/dtengine.cpp
    PBA3D/dtengine.h
    PBA3D/gpudefs.h
    PBA3D/myrandom.cpp
    PBA3D/myrandom.h
    PBA3D/plyptreader.cpp
    PBA3D/readers.h
    PBA3D/rndreader.cpp
    PBA3D/utils.h
    PBA3D/writerbase.cpp
    PBA3D/pba/pba3DKernel.h
    PBA3D/pba/pba3DHost.cu
    PBA3D/pba/pba3D.h
    )

SET( QT_SRCS
    QDistanceWidget.cxx
    QDistanceWidget.h
    QMinimalDistanceCanvasWidget.cxx
    QMinimalDistanceCanvasWidget.h
    QMinimalDistanceWidget.cxx
    QMinimalDistanceWidget.h
    QTumorMapCanvasWidget.cxx
    QTumorMapCanvasWidget.h
    QTumorMapWidget.cxx
    QTumorMapWidget.h
    )

SET( VTK_SRCS
    vtkInteractorStyleCellPicker.cxx
    vtkInteractorStyleCellPicker.h
    vtkImageDataToDistanceTransform.cxx
    vtkImageDataToDistanceTransform.h
    vtkDistanceArrowWidget.cxx
    vtkDistanceArrowWidget.h
    vtkPointMarkerWidget.cxx
    vtkPointMarkerWidget.h
    vtkParameterizePolyData.cxx
    vtkParameterizePolyData.h
    )

SET( CUDA_SRCS
	PBA3D/pba/pba3DHost.cu
	)

IF(UNIX)
SET(CUDA_OUTPUT_LIB ${CMAKE_CURRENT_BINARY_DIR}/DistanceUncertaintyCUDA.a)
ELSE(UNIX)
SET(CUDA_OUTPUT_LIB ${CMAKE_CURRENT_BINARY_DIR}/DistanceUncertaintyCUDA.lib)
ENDIF(UNIX)

IF( CMAKE_SIZEOF_VOID_P EQUAL 4 )
SET( MFLAG )
ELSE( CMAKE_SIZEOF_VOID_P EQUAL 4 )
SET( MFLAG -m64 )
ENDIF( CMAKE_SIZEOF_VOID_P EQUAL 4 )

IF(UNIX)
    EXECUTE_PROCESS(
    	COMMAND /usr/local/cuda/bin/nvcc -o ${CUDA_OUTPUT_LIB} 
    		-g -Xcompiler -fPIC ${MFLAG} -lib PBA3D/pba/pba3DHost.cu
    	COMMAND ranlib ${CUDA_OUTPUT_LIB}
    	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    	)
ELSE( UNIX )
    EXECUTE_PROCESS(
    	COMMAND nvcc -o ${CUDA_OUTPUT_LIB} 
    		--machine 32 -lib -g PBA3D/pba/pba3DHost.cu
    	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    	)
ENDIF( UNIX )

QT4_WRAP_CPP( MOC_SRCS
	DistanceUncertaintyPlugin.h
	QDistanceWidget.h
	QMinimalDistanceCanvasWidget.h
	QMinimalDistanceWidget.h
	QTumorMapCanvasWidget.h
	QTumorMapWidget.h
	)

ADD_LIBRARY( DistanceUncertaintyPlugin SHARED
        ${SRCS} ${PBA3D_SRCS} ${QT_SRCS} ${VTK_SRCS} ${MOC_SRCS}
	)
    
TARGET_LINK_LIBRARIES( DistanceUncertaintyPlugin
	cublas cudart cufft curand cusparse # cuda tlshook
	${QT_LIBRARIES}
	${CUDA_OUTPUT_LIB}
	PolyDataVisualizationPlugin
	RayCastPlugin
	bmia_plugin 
	bmia_gui
	)

endif(PLUGINS_BUILD_DISTANCEUNCERTAINTY)
